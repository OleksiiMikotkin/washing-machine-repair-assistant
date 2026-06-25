import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PointStruct,
    SearchParams,
    VectorParams,
)

from vectordb.base import VectorDB

BATCH_SIZE = 500


class QdrantDB(VectorDB):
    def __init__(
        self,
        collection_name: str = "washing_machine",
        host: str = "localhost",
        port: int = 6334,
        url: str | None = None,
        api_key: str | None = None,
        m: int = 16,
        ef_construction: int = 100,
        ef_search: int = 100,
    ):
        if url:
            self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=True)
        else:
            self.client = QdrantClient(host=host, port=port, api_key=api_key, prefer_grpc=True)

        self.collection_name = collection_name
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def index(self, vectors: np.ndarray, ids: list[str], payloads: list[dict]) -> None:
        dim = vectors.shape[1]

        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=self.m, ef_construct=self.ef_construction),
        )

        total = len(ids)
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            points = [
                PointStruct(id=ids[i], vector=vectors[i].tolist(), payload=payloads[i])
                for i in range(start, end)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"  uploaded {end:,} / {total:,}", end="\r")
        print()

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[tuple[dict, float]]:
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec.tolist(),
            limit=top_k,
            query_filter=self._build_filter(filters),
            search_params=SearchParams(hnsw_ef=self.ef_search),
            with_payload=True,
        )
        return [(hit.payload, hit.score) for hit in hits.points]

    @staticmethod
    def _build_filter(filters: dict | None) -> Filter | None:
        if not filters:
            return None
        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filters.items()
            if value is not None
        ]
        return Filter(must=conditions) if conditions else None

    def disk_size_mb(self) -> float:
        info = self.client.get_collection(self.collection_name)
        n = info.points_count or 0
        vectors_cfg = info.config.params.vectors
        dim = vectors_cfg.size if hasattr(vectors_cfg, "size") else next(iter(vectors_cfg.values())).size
        return round((n * dim * 4) / (1024 * 1024) * 1.25, 1)

    def cleanup(self) -> None:
        self.client.delete_collection(self.collection_name)
