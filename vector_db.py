import os
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantStorage:
    def __init__(self, url=None, path=None, collection=None, dim=1024):
        self.dim = dim
        self.collection = collection or os.getenv("QDRANT_COLLECTION", "docs")
        self.client, self.is_remote = self._create_client(url=url, path=path)
        self._ensure_collection()

    @staticmethod
    def _create_client(url=None, path=None):
        resolved_url = url or os.getenv("QDRANT_URL")
        if resolved_url:
            api_key = os.getenv("QDRANT_API_KEY")
            return QdrantClient(url=resolved_url, api_key=api_key, timeout=30), True

        resolved_path = path or os.getenv("QDRANT_PATH")
        if resolved_path is None:
            resolved_path = Path(__file__).resolve().parent / "qdrant_local_storage"

        return QdrantClient(path=str(resolved_path), timeout=30), False

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection):
            self._create_collection()
            return

        collection_info = self.client.get_collection(self.collection)
        configured_vectors = collection_info.config.params.vectors
        current_dim = getattr(configured_vectors, "size", None)

        if current_dim == self.dim:
            return

        if self.is_remote:
            raise RuntimeError(
                f"Collection '{self.collection}' is configured for vectors of size {current_dim}, "
                f"but the app is configured for {self.dim}. Recreate the remote collection or update its dimension settings."
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