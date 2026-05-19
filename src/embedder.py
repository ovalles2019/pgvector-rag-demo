"""Embedding generation with sentence-transformers."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from src.config import Settings


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self._model = SentenceTransformer(settings.embedding_model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]
