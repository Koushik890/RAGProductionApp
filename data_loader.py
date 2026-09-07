import os
import threading

from openai import OpenAI, RateLimitError
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

# Any OpenAI-compatible provider works. Defaults point at NVIDIA NIM, which
# serves the Nemotron models first-party and rate limits per minute rather than
# capping requests per day.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nemotron-3-embed-1b")

# The embedding endpoint has a finite context window, so long documents are sent
# in batches. Larger batches mean fewer requests, which keeps us clear of
# per-minute rate limits.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

# NVIDIA embedding models are asymmetric: a document and a question about that
# document must be embedded differently for retrieval to work well. Providers
# that do not accept the parameter would reject the request, so it is only sent
# to NVIDIA endpoints unless forced with EMBED_INPUT_TYPE=1 or 0.
def _input_type_supported() -> bool:
    forced = os.getenv("EMBED_INPUT_TYPE")
    if forced is not None:
        return forced.strip().lower() in ("1", "true", "yes")
    return "api.nvidia.com" in LLM_BASE_URL


# NIM rejects input longer than the model context unless told how to truncate.
EMBED_TRUNCATE = os.getenv("EMBED_TRUNCATE", "END")

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def get_api_key() -> str:
    """Read the provider key, accepting the older provider-specific names."""
    for name in ("LLM_API_KEY", "NVIDIA_API_KEY", "NIM_API_KEY", "OPENROUTER_API_KEY"):
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError("No API key set. Provide LLM_API_KEY (or NVIDIA_API_KEY).")


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
                api_key = get_api_key()

                # Optional attribution headers; OpenRouter uses them for its
                # public leaderboards, other providers ignore them.
                default_headers = {}
                site_url = os.getenv("OPENROUTER_SITE_URL")
                app_name = os.getenv("OPENROUTER_APP_NAME")
                if site_url:
                    default_headers["HTTP-Referer"] = site_url
                if app_name:
                    default_headers["X-Title"] = app_name

                _client = OpenAI(
                    base_url=LLM_BASE_URL,
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


def embed_texts(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """Embed *texts*. Use input_type="query" for a question, "passage" for a document."""
    if not texts:
        return []

    client = get_client()
    extra_body = {}
    if _input_type_supported():
        extra_body = {"input_type": input_type, "truncate": EMBED_TRUNCATE}

    vectors: list[list[float]] = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        try:
            response = client.embeddings.create(
                model=EMBED_MODEL, input=batch, extra_body=extra_body
            )
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
                    _embed_dim = len(embed_texts(["dimension probe"], input_type="query")[0])

    return _embed_dim
