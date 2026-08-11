import asyncio
import hashlib
from pathlib import Path

import httpx
import pymupdf

from app.config import Settings
from app.dependencies import get_vector_store
from app.exceptions import AppError
from app.main import create_app
from tests.fakes import FakeVectorStore


def make_client(settings: Settings, store: FakeVectorStore) -> httpx.AsyncClient:
    app = create_app(settings)
    app.dependency_overrides[get_vector_store] = lambda: store
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


class FailingReadVectorStore(FakeVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.delete_calls = 0

    async def document_chunks(self, _document_id: str) -> list[dict]:
        raise AppError("VECTOR_STORE_ERROR", "读取向量存储失败", 500)

    async def delete_document(self, _document_id: str) -> None:
        self.delete_calls += 1


class DelayedVectorStore(FakeVectorStore):
    async def document_chunks(self, document_id: str) -> list[dict]:
        await asyncio.sleep(0.01)
        return await super().document_chunks(document_id)

    async def add_chunks(self, chunks: list) -> None:
        await asyncio.sleep(0.01)
        await super().add_chunks(chunks)


async def test_unsupported_extension_is_rejected(settings: Settings) -> None:
    async with make_client(settings, FakeVectorStore()) as client:
        response = await client.post(
            "/documents",
            files={"file": ("notes.txt", b"text", "text/plain")},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_FILE_TYPE"


async def test_too_large_file_returns_413(settings: Settings) -> None:
    limited = settings.model_copy(update={"max_upload_size_mb": 1})
    async with make_client(limited, FakeVectorStore()) as client:
        response = await client.post(
            "/documents",
            files={"file": ("big.md", b"x" * (1024 * 1024 + 1), "text/markdown")},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


async def test_upload_is_safe_and_duplicate_is_not_added_twice(settings: Settings) -> None:
    store = FakeVectorStore()
    async with make_client(settings, store) as client:
        first = await client.post(
            "/documents",
            files={"file": ("../../redis.md", "# Redis\n\n惰性删除。".encode(), "text/markdown")},
        )
        second = await client.post(
            "/documents",
            files={"file": ("../../redis.md", "# Redis\n\n惰性删除。".encode(), "text/markdown")},
        )
    assert first.status_code == 200
    assert first.json()["filename"] == "redis.md"
    assert first.json()["chunk_count"] == 1
    assert len(first.json()["document_id"]) == 64
    assert second.json()["duplicate"] is True
    assert store.add_calls == 1
    assert {chunk.metadata["page"] for chunks in store.chunks.values() for chunk in chunks} == {1}
    saved = list(settings.upload_dir.glob("*.md"))
    assert len(saved) == 1
    assert saved[0].parent == Path(settings.upload_dir)


async def test_concurrent_duplicate_upload_is_indexed_once(settings: Settings) -> None:
    store = DelayedVectorStore()
    content = "# Redis\n\n连接池超时。".encode()
    async with make_client(settings, store) as client:
        first, second = await asyncio.gather(
            client.post(
                "/documents",
                files={"file": ("runbook.md", content, "text/markdown")},
            ),
            client.post(
                "/documents",
                files={"file": ("runbook.md", content, "text/markdown")},
            ),
        )
    assert first.status_code == second.status_code == 200
    assert sorted([first.json()["duplicate"], second.json()["duplicate"]]) == [False, True]
    assert store.add_calls == 1
    assert len(store.chunks) == 1
    assert len(list(settings.upload_dir.glob("*.md"))) == 1


async def test_empty_upload_is_rejected_and_cleaned(settings: Settings) -> None:
    async with make_client(settings, FakeVectorStore()) as client:
        response = await client.post(
            "/documents",
            files={"file": ("empty.md", b" \n", "text/markdown")},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "EMPTY_DOCUMENT"
    assert not list(settings.upload_dir.glob("*.md"))


async def test_empty_upload_is_checked_before_embedding_configuration(
    settings: Settings,
) -> None:
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/documents",
            files={"file": ("empty.md", b" \n", "text/markdown")},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "EMPTY_DOCUMENT"


async def test_duplicate_check_failure_does_not_delete_existing_data(settings: Settings) -> None:
    content = b"# existing"
    document_id = hashlib.sha256(content).hexdigest()
    settings.upload_dir.mkdir(parents=True)
    existing_path = settings.upload_dir / f"{document_id}.md"
    existing_path.write_bytes(b"preserve me")
    store = FailingReadVectorStore()
    async with make_client(settings, store) as client:
        response = await client.post(
            "/documents",
            files={"file": ("existing.md", content, "text/markdown")},
        )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "VECTOR_STORE_ERROR"
    assert existing_path.read_bytes() == b"preserve me"
    assert store.delete_calls == 0
    assert not list(settings.upload_dir.glob(".upload-*"))


async def test_pdf_response_preserves_total_page_count(
    settings: Settings,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "pages.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "page one")
    document.new_page()
    third_page = document.new_page()
    third_page.insert_text((72, 72), "page three")
    document.save(pdf_path)
    document.close()
    store = FakeVectorStore()
    async with make_client(settings, store) as client:
        response = await client.post(
            "/documents",
            files={"file": ("pages.pdf", pdf_path.read_bytes(), "application/pdf")},
        )
    assert response.status_code == 200
    assert response.json()["page_count"] == 3
    metadata = [chunk.metadata for chunks in store.chunks.values() for chunk in chunks]
    assert [item["page"] for item in metadata] == [1, 3]
    assert {item["page_count"] for item in metadata} == {3}
