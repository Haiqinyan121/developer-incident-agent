
import pytest

from app.config import Settings
from app.exceptions import AppError
from app.models import RetrievedChunk
from app.rag.service import RAGService, safe_filename
from app.rag.splitter import DocumentChunk
from app.rag.vector_store import VectorStoreService
from tests.fakes import FakeEmbeddings, FakeVectorStore


def test_embedding_client_sends_prechunked_text_without_token_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class CapturingEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("app.rag.vector_store.OpenAIEmbeddings", CapturingEmbeddings)
    settings = Settings(
        openai_api_key="shared-key",
        openai_base_url="https://shared.example/v1",
        embedding_api_key="embedding-key",
        embedding_base_url="https://embedding.example/v1/",
        embedding_model="test-embedding",
    )

    VectorStoreService(settings)._get_embeddings()

    assert captured["api_key"] == "embedding-key"
    assert captured["base_url"] == "https://embedding.example/v1"
    assert captured["model"] == "test-embedding"
    assert captured["check_embedding_ctx_length"] is False


def test_safe_filename_blocks_both_path_separator_styles() -> None:
    assert safe_filename("../../secret.md") == "secret.md"
    assert safe_filename("..\\..\\secret.md") == "secret.md"


async def test_search_assigns_sources_and_deduplicates(settings: Settings) -> None:
    duplicate = RetrievedChunk(
        content="same",
        filename="a.md",
        page=1,
        document_id="id",
        distance=0.1,
    )
    unique = RetrievedChunk(
        content="other",
        filename="a.md",
        page=1,
        document_id="id",
        distance=0.2,
    )
    store = FakeVectorStore([duplicate, duplicate.model_copy(), unique])
    results = await RAGService(settings, store).search(" query ", 3)
    assert [item.source_id for item in results] == ["S1", "S2"]
    assert results[0].filename == "a.md"
    assert results[0].page == 1


async def test_search_truncates_excerpt(settings: Settings) -> None:
    item = RetrievedChunk(
        content="x" * 700,
        filename="a.md",
        page=1,
        document_id="id",
    )
    results = await RAGService(settings, FakeVectorStore([item])).search("x", 1)
    assert len(results[0].content) == 500


@pytest.mark.parametrize(("query", "top_k"), [(" ", 4), ("ok", 11), ("ok", 0)])
async def test_search_rejects_invalid_arguments(
    settings: Settings,
    query: str,
    top_k: int,
) -> None:
    with pytest.raises(AppError, match="INVALID_TOOL_ARGUMENTS"):
        await RAGService(settings, FakeVectorStore()).search(query, top_k)


async def test_local_chroma_add_search_and_delete(settings: Settings) -> None:
    store = VectorStoreService(settings, embeddings=FakeEmbeddings())
    assert await store.document_chunks("doc") == []
    chunk = DocumentChunk(
        id="doc:1:0",
        content="Redis 惰性删除",
        metadata={
            "document_id": "doc",
            "filename": "redis.md",
            "page": 1,
            "chunk_index": 0,
            "source_type": "markdown",
        },
    )
    await store.add_chunks([chunk])
    assert await store.document_chunks("doc")
    results = await store.search("Redis", 1)
    assert results[0].filename == "redis.md"
    assert results[0].page == 1
    await store.delete_document("doc")
    assert await store.document_chunks("doc") == []
