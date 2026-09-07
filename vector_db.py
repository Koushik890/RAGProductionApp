import logging
import os
import urllib.parse
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger("uvicorn")


def _recreate_allowed() -> bool:
    """Whether a remote collection may be dropped when its vector size is wrong.

    A collection built for a different embedding model cannot be searched or
    written to, so the app is dead until it is rebuilt. Set
    `QDRANT_ALLOW_RECREATE=false` to fail loudly instead of dropping it.
    """
    return os.getenv("QDRANT_ALLOW_RECREATE", "true").lower() != "false"


def describe_target() -> str:
    """Where Qdrant requests are being sent, safe to expose.

    /health/deps is public, so the cluster id is masked. Scheme and port are
    kept because they are what actually distinguishes the common
    misconfigurations: Qdrant Cloud speaks REST on 6333 and gRPC on 6334, and
    pointing a REST client at the gRPC port resets the connection, as does
    plain http against a TLS endpoint.
    """
    url = os.getenv("QDRANT_URL")
    if not url:
        return "embedded (QDRANT_URL is not set)"

    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or "?"
    labels = host.split(".")
    if len(labels) > 2:
        labels[0] = "***"
    masked = ".".join(labels)

    port = parsed.port
    if port is None:
        port = f"default:{443 if parsed.scheme == 'https' else 80}"

    return f"{parsed.scheme}://{masked}:{port}"


def create_client(url=None, path=None):
    """Build a Qdrant client. Returns (client, is_remote).

    Public so a connectivity check can run without knowing the embedding
    dimension, which requires a live call to the embedding provider.
    """
    resolved_url = url or os.getenv("QDRANT_URL")
    if resolved_url:
        api_key = os.getenv("QDRANT_API_KEY")
        return QdrantClient(url=resolved_url, api_key=api_key, timeout=30), True

    resolved_path = path or os.getenv("QDRANT_PATH")
    if resolved_path is None:
        resolved_path = Path(__file__).resolve().parent / "qdrant_local_storage"

    return QdrantClient(path=str(resolved_path), timeout=30), False


class QdrantStorage:
    def __init__(self, url=None, path=None, collection=None, dim=1024):
        self.dim = dim
        self.collection = collection or os.getenv("QDRANT_COLLECTION", "docs")
        self.client, self.is_remote = create_client(url=url, path=path)
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection):
            self._create_collection()
            return

        collection_info = self.client.get_collection(self.collection)
        configured_vectors = collection_info.config.params.vectors
        current_dim = getattr(configured_vectors, "size", None)

        if current_dim == self.dim:
            return

        if current_dim is None:
            # Named-vector collections expose a mapping rather than a single
            # size. Treating "unknown" as "mismatched" would delete a
            # collection this app simply does not understand.
            raise RuntimeError(
                f"Collection '{self.collection}' does not expose a single vector size, so it "
                f"cannot be compared against the embedding model's {self.dim} dimensions. It "
                f"may use named vectors. Inspect it manually, or point QDRANT_COLLECTION at a "
                f"different name."
            )

        if self.is_remote and not _recreate_allowed():
            raise RuntimeError(
                f"Collection '{self.collection}' is configured for vectors of size {current_dim}, "
                f"but the app is configured for {self.dim}. Recreate the remote collection, or set "
                f"QDRANT_ALLOW_RECREATE=true to let the app rebuild it (this deletes its contents)."
            )

        logger.warning(
            "Collection '%s' has vector size %s but the embedding model produces %s. "
            "Dropping and recreating it; all previously ingested documents must be re-uploaded.",
            self.collection,
            current_dim,
            self.dim,
        )
        self.client.delete_collection(self.collection)
        self._create_collection()

    def _create_collection(self):
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )

    def close(self):
        self.client.close()

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )
        contexts = []
        sources = set()

        for r in response.points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source_id", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
