from abc import ABC, abstractmethod

import numpy as np


class VectorDB(ABC):

    @abstractmethod
    def index(self, vectors: np.ndarray, ids: list[str], payloads: list[dict]) -> None:
        """
        vectors  : shape (N, dim), float32, L2-normalised for cosine
        ids      : UUID strings parallel to vectors
        payloads : full metadata dicts stored alongside each vector
        """

    @abstractmethod
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[tuple[dict, float]]:
        """
        query_vec : shape (dim,)
        filters   : {field: value} metadata filter, implementation-specific conversion
        Returns   : [(payload_dict, score), ...] of length top_k
        """

    @abstractmethod
    def disk_size_mb(self) -> float:
        """Estimated index size on disk in MB (0 for in-memory)."""

    def cleanup(self) -> None:
        """Close connections / delete temporary data. No-op by default."""
