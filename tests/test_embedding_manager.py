"""Tests for local embedding management without downloading a real model."""

import pytest

from src.embedding_manager import EmbeddingConfig, EmbeddingManager


class FakeEmbeddingModel:
    """Small predictable model used only for unit testing."""

    def __init__(self) -> None:
        self.last_texts: list[str] = []
        self.last_options: dict[str, object] = {}

    def encode(self, texts: list[str], **options: object) -> list[list[float]]:
        self.last_texts = texts
        self.last_options = options

        return [
            [float(len(text)), float(index), 1.0] for index, text in enumerate(texts)
        ]

    def get_sentence_embedding_dimension(self) -> int:
        return 3


def test_embed_texts_uses_configured_batch_options() -> None:
    fake_model = FakeEmbeddingModel()
    config = EmbeddingConfig(
        batch_size=8,
        normalize_embeddings=True,
    )
    manager = EmbeddingManager(config=config, model=fake_model)

    embeddings = manager.embed_texts(["Disconnect power.", "Inspect the clamp."])

    assert embeddings == [
        [17.0, 0.0, 1.0],
        [18.0, 1.0, 1.0],
    ]
    assert fake_model.last_options["batch_size"] == 8
    assert fake_model.last_options["normalize_embeddings"] is True
    assert fake_model.last_options["show_progress_bar"] is False
    assert fake_model.last_options["convert_to_numpy"] is True


def test_embed_query_returns_one_vector() -> None:
    manager = EmbeddingManager(model=FakeEmbeddingModel())

    embedding = manager.embed_query("How should the joint be supported?")

    assert embedding == [34.0, 0.0, 1.0]


def test_embedding_dimension_comes_from_model() -> None:
    manager = EmbeddingManager(model=FakeEmbeddingModel())

    assert manager.embedding_dimension == 3


def test_empty_embedding_batch_returns_without_loading_model() -> None:
    manager = EmbeddingManager()

    assert manager.embed_texts([]) == []
    assert manager._model is None


@pytest.mark.parametrize(
    "invalid_text",
    ["", "   ", "\n"],
)
def test_empty_text_is_rejected(invalid_text: str) -> None:
    manager = EmbeddingManager(model=FakeEmbeddingModel())

    with pytest.raises(ValueError, match="cannot contain empty text"):
        manager.embed_texts(["Valid text", invalid_text])


def test_empty_query_is_rejected() -> None:
    manager = EmbeddingManager(model=FakeEmbeddingModel())

    with pytest.raises(ValueError, match="Query cannot be empty"):
        manager.embed_query(" ")
