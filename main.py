import asyncio
import logging
import datetime
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import requests as http_requests

from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc
from data_loader import EMBED_DIM, load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage

load_dotenv()
storage = None


def get_storage() -> QdrantStorage:
    global storage

    if storage is None:
        storage = QdrantStorage(dim=EMBED_DIM)

    return storage


def close_storage() -> None:
    global storage

    if storage is None:
        return

    storage.close()
    storage = None

UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "uploads"))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

INNGEST_API_BASE = os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")
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
    # Limit ingestion to 5 concurrent runs to avoid overloading Qdrant/Mistral
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
    # Limit queries to 10 concurrent runs
    concurrency=[inngest.Concurrency(limit=10)],
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
        auth_key=os.getenv("MISTRAL_API_KEY"),
        base_url="https://api.mistral.ai/v1",
        model="mistral-large-latest",
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

app = FastAPI()


# --------------- Inngest polling helper ------------------------------------


async def poll_inngest_run(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    """Poll the Inngest API until the run triggered by *event_id* finishes."""
    headers = {}
    if INNGEST_SIGNING_KEY:
        headers["Authorization"] = f"Bearer {INNGEST_SIGNING_KEY}"

    url = f"{INNGEST_API_BASE}/events/{event_id}/runs"
    start = time.time()

    while True:
        resp = await asyncio.to_thread(http_requests.get, url, headers=headers, timeout=10)
        resp.raise_for_status()
        runs = resp.json().get("data", [])

        if runs:
            run = runs[0]
            status = run.get("status")
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise HTTPException(status_code=502, detail=f"Inngest function run {status}")

        if time.time() - start > timeout_s:
            raise HTTPException(status_code=504, detail="Timed out waiting for Inngest run")

        await asyncio.sleep(poll_interval_s)


# --------------- API endpoints (routed through Inngest) --------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    file_path = UPLOADS_DIR / file.filename
    contents = await file.read()
    file_path.write_bytes(contents)

    event_ids = await inngest_client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(file_path.resolve()),
                "source_id": file.filename,
            },
        )
    )

    return {"status": "ingestion_triggered", "source_id": file.filename, "event_id": event_ids[0]}


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/query")
async def query_pdf(req: QueryRequest):
    event_ids = await inngest_client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": req.question,
                "top_k": req.top_k,
            },
        )
    )

    output = await poll_inngest_run(event_ids[0])
    return output


# --------------- Register Inngest functions --------------------------------

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])

