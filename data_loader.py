import os

from mistralai import Mistral
from mistralai.utils import BackoffStrategy, RetryConfig
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "mistral-embed"
EMBED_DIM = int(os.getenv("MISTRAL_EMBED_DIM", "1024"))

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def get_mistral_client() -> Mistral:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set")

    return Mistral(
        api_key=api_key,
        timeout_ms=30_000,
        retry_config=RetryConfig(
            strategy="backoff",
            backoff=BackoffStrategy(
                initial_interval=500,
                max_interval=10_000,
                exponent=2.0,
                max_elapsed_time=60_000,
            ),
            retry_connection_errors=True,
        ),
    )

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

    response = get_mistral_client().embeddings.create(
        model=EMBED_MODEL,
        inputs=texts,
    )
    return [item.embedding for item in response.data]
