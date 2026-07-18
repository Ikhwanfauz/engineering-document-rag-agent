"""Create local semantic embeddings for document chunks and queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from sentence_transformers import SentenceTransformer


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration for the local sentence-transformer model."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize_embeddings: bool = True
    device: str | None = None

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")


DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()


class EmbeddingManager:
    """Load one embedding model and reuse it for documents and queries."""

    def __init__(
        self,
        config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
        model: Any | None = None,
    ) -> None:
        self.config = config
        self._model = model

    @property
    def model(self) -> Any:
        """Load the model only when embeddings are first requested."""
        if self._model is None:
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
            )
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Return the number of values produced for one embedding."""
        dimension = self.model.get_sentence_embedding_dimension()

        if dimension is None:
            raise RuntimeError("Embedding model did not report its output dimension")

        return int(dimension)

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Embed a batch of non-empty document texts."""
        text_list = list(texts)

        if not text_list:
            return []

        if any(not text.strip() for text in text_list):
            raise ValueError("Embedding input cannot contain empty text")

        embeddings = self.model.encode(
            text_list,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()

        return [list(vector) for vector in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Embed one user question using the same document model."""
        if not query.strip():
            raise ValueError("Query cannot be empty")

        return self.embed_texts([query])[0]
