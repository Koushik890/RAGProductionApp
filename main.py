import asyncio
import logging
import datetime
import os
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import requests as http_requests

from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc
from data_loader import get_embed_dim, load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage

load_dotenv()
storage = None


def get_storage() -> QdrantStorage:
    global storage

    if storage is None:
        storage = QdrantStorage(dim=get_embed_dim())

    return storage


def close_storage() -> None:
    global storage

    if storage is None:
        return

    storage.close()
    storage = None

UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "uploads"))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))

INNGEST_API_BASE = os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1").rstrip("/")
INNGEST_SIGNING_KEY = os.getenv("INNGEST_SIGNING_KEY", "")

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=os.getenv("INNGEST_PRODUCTION", "false").lower() == "true",
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    # Limit ingestion to 5 concurrent runs to avoid overloading Qdrant/OpenRouter
    concurrency=[inngest.Concurrency(limit=5)],
    # Throttle to max 10 ingestions per minute
    throttle=inngest.Throttle(limit=10, period=datetime.timedelta(minutes=1)),
    retries=3,
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(vecs))]
        payloads = [{"source_id": source_id, "text": chunks[i]} for i in range(len(chunks))]
        get_storage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(ids))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested= await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
    # Limit queries to 5 concurrent runs (Inngest free plan cap)
    concurrency=[inngest.Concurrency(limit=5)],
    # Rate limit: max 30 queries per minute
    rate_limit=inngest.RateLimit(limit=30, period=datetime.timedelta(minutes=1)),
    retries=2,
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        found = get_storage().search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = ctx.event.data.get("top_k", 5)

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
        model=ANSWER_MODEL,
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            # Reasoning models spend part of this budget thinking before they
            # emit any answer text, so it has to be well above the answer length.
            "max_tokens": ANSWER_MAX_TOKENS,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    message = res["choices"][0]["message"]
    answer = (message.get("content") or "").strip()
    if not answer:
        # Reasoning models sometimes leave `content` null and put the text in
        # `reasoning` instead. Fall back rather than crashing on None.
        answer = (message.get("reasoning") or "").strip()
    if not answer:
        raise RuntimeError("The model returned an empty answer")

    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

app = FastAPI()


# --------------- Inngest run status helpers --------------------------------

# Terminal states reported by the Inngest REST API.
RUN_STATES_OK = {"Completed", "Succeeded", "Success", "Finished"}
RUN_STATES_BAD = {"Failed", "Cancelled"}


def _safe_filename(name: str) -> str:
    """Turn a client-supplied filename into a safe basename inside UPLOADS_DIR."""
    base = Path(name or "").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if not cleaned:
        cleaned = "upload.pdf"
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned}.pdf"
    return cleaned[-128:]


async def fetch_inngest_run(event_id: str) -> dict | None:
    """Return the latest run for *event_id*, or None if none exists yet.

    Raises HTTPException(429) with Retry-After when Inngest rate limits us, so
    callers back off instead of hammering the API.
    """
    headers = {}
    if INNGEST_SIGNING_KEY:
        headers["Authorization"] = f"Bearer {INNGEST_SIGNING_KEY}"

    url = f"{INNGEST_API_BASE}/events/{event_id}/runs"

    try:
        resp = await asyncio.to_thread(http_requests.get, url, headers=headers, timeout=10)
    except http_requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Inngest API: {exc}") from exc

    if resp.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail="Inngest API rate limit reached; retry shortly.",
            headers={"Retry-After": resp.headers.get("Retry-After", "5")},
        )

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Inngest API returned {resp.status_code}: {resp.text[:500]}",
        )

    runs = resp.json().get("data") or []
    return runs[0] if runs else None


def _describe_run(run: dict) -> dict:
    """Normalise an Inngest run record into the shape the UI polls for."""
    status = run.get("status")

    if status in RUN_STATES_OK:
        return {"state": "completed", "status": status, "output": run.get("output") or {}}

    if status in RUN_STATES_BAD:
        # Inngest puts the failing step's error in `output`. Surface it instead
        # of swallowing it, otherwise every failure looks identical.
        return {"state": "failed", "status": status, "error": run.get("output")}

    return {"state": "running", "status": status}


# --------------- API endpoints (routed through Inngest) --------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/deps")
async def health_deps():
    """Check the two dependencies every Inngest step needs.

    Both `rag/ingest_pdf` and `rag/query_pdf_ai` fail identically when the
    embedding provider or Qdrant is misconfigured, and the run error alone does
    not say which. This names the broken one directly.
    """

    def _check_embeddings() -> None:
        embed_texts(["ping"])

    def _check_qdrant() -> None:
        # Cosine distance rejects a zero vector, so probe with a unit vector.
        probe = [1.0] + [0.0] * (get_embed_dim() - 1)
        get_storage().search(probe, 1)

    results: dict[str, dict] = {}
    for name, check in (("embeddings", _check_embeddings), ("qdrant", _check_qdrant)):
        try:
            await asyncio.to_thread(check)
        except Exception as exc:
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        else:
            results[name] = {"ok": True}

    return results


@app.get("/status/{event_id}")
async def run_status(event_id: str):
    """Poll one Inngest run. Cheap and fast, so it is safe to call every few seconds."""
    run = await fetch_inngest_run(event_id)

    if run is None:
        return {"state": "pending"}

    return _describe_run(run)


@app.post("/upload", status_code=202)
async def upload_pdf(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF is larger than the {MAX_UPLOAD_BYTES} byte limit",
        )

    filename = _safe_filename(file.filename)
    file_path = UPLOADS_DIR / filename
    file_path.write_bytes(contents)

    try:
        event_ids = await inngest_client.send(
            inngest.Event(
                name="rag/ingest_pdf",
                data={
                    "pdf_path": str(file_path.resolve()),
                    "source_id": filename,
                },
            )
        )
    except Exception as exc:  # SendEventsError plus transport failures
        raise HTTPException(status_code=502, detail=f"Could not queue ingestion: {exc}") from exc

    # Return straight away. The client polls /status/{event_id}. Holding the
    # connection open for the whole ingestion is what piled up long-lived
    # requests on the instance and tripped upstream rate limiting.
    return {"status": "ingestion_started", "event_id": event_ids[0], "source_id": filename}


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/query", status_code=202)
async def query_pdf(req: QueryRequest):
    try:
        event_ids = await inngest_client.send(
            inngest.Event(
                name="rag/query_pdf_ai",
                data={
                    "question": req.question,
                    "top_k": req.top_k,
                },
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not queue query: {exc}") from exc

    return {"status": "query_started", "event_id": event_ids[0]}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    """Answer with JSON so clients can show a real reason.

    Without this, Starlette returns a bare `Internal Server Error` body and the
    UI has nothing useful to display.
    """
    logging.getLogger("uvicorn").exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


# --------------- Register Inngest functions --------------------------------

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
