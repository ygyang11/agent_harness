"""Abstract base for vector stores."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class VectorDocument(BaseModel):
    """A document stored in a vector store."""
    id: str
    content: str
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorSearchResult(BaseModel):
    """A search result from a vector store."""
    document: VectorDocument
    score: float  # similarity score (higher = more similar)


class BaseVectorStore(ABC):
    """Abstract base class for vector stores.

    Vector stores provide semantic similarity search over documents.
    They are used by LongTermMemory for retrieval-augmented generation.
    """

    @abstractmethod
    async def upsert(self, documents: list[VectorDocument]) -> None:
        """Insert or update documents.

        Args:
            documents: Documents with pre-computed embeddings.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar documents.

        Args:
            query_embedding: The query vector.
            top_k: Maximum number of results.
            filter_metadata: Optional metadata filter.

        Returns:
            List of results sorted by similarity (descending).
        """
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return the total number of stored documents."""
        ...

    async def list_all(self) -> list[VectorDocument]:
        """Return all stored documents.

        Optional operation — not all vector stores support full scan.
        Default returns empty list. Subclasses that support iteration
        should override.
        """
        return []

    @abstractmethod
    async def clear(self) -> None:
        """Remove all documents."""
        ...
