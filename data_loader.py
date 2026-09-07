import os
import threading

from openai import OpenAI, RateLimitError
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nemotron-3-embed-1b:free")

# The embedding endpoint has a finite context window, so long documents are sent
# in batches. Larger batches mean fewer requests, which matters because free
# OpenRouter models are capped at 20 requests/minute and 50/day per account.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


class EmbeddingQuotaExceeded(RuntimeError):
    """The embedding quota is spent for the current window.

    Distinct from a burst rate limit: waiting seconds will not help, so callers
    should stop retrying rather than spend the remaining budget on retries.
    """


def _is_daily_quota(exc: Exception) -> bool:
    text = str(exc).lower()
    return "per-day" in text or "per day" in text or "daily" in text


_client: OpenAI | None = None
_client_lock = threading.Lock()

_embed_dim: int | None = None
_embed_dim_lock = threading.Lock()


def get_client() -> OpenAI:
    global _client

    if _client is None:
        with _client_lock:
            if _client is None:
                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    raise RuntimeError("OPENROUTER_API_KEY is not set")

                # Optional attribution headers; OpenRouter uses them for its
                # public leaderboards and ignores them when absent.
                default_headers = {}
                site_url = os.getenv("OPENROUTER_SITE_URL")
                app_name = os.getenv("OPENROUTER_APP_NAME")
                if site_url:
                    default_headers["HTTP-Referer"] = site_url
                if app_name:
                    default_headers["X-Title"] = app_name

                _client = OpenAI(
                    base_url=OPENROUTER_BASE_URL,
                    api_key=api_key,
                    timeout=60.0,
                    max_retries=2,
                    default_headers=default_headers or None,
                )

    return _client


def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    client = get_client()
    vectors: list[list[float]] = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        try:
            response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        except RateLimitError as exc:
            if _is_daily_quota(exc):
                raise EmbeddingQuotaExceeded(
                    f"{EMBED_MODEL} has no requests left in the current quota window: {exc}"
                ) from exc
            raise
        # The API may return items out of order, so sort by the index it reports.
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)

    return vectors


def get_embed_dim() -> int:
    """Vector size produced by EMBED_MODEL.

    Set `EMBED_DIM` to skip the probe. Otherwise the size is measured once by
    embedding a short string, so swapping embedding models does not require
    hand-editing a dimension that then silently mismatches the Qdrant
    collection.
    """
    global _embed_dim

    if _embed_dim is None:
        with _embed_dim_lock:
            if _embed_dim is None:
                override = os.getenv("EMBED_DIM")
                if override:
                    _embed_dim = int(override)
                else:
                    _embed_dim = len(embed_texts(["dimension probe"])[0])

    return _embed_dim
