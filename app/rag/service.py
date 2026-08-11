import asyncio
import hashlib
import logging
from pathlib import Path
from uuid import uuid4

import pymupdf
from fastapi import UploadFile

from app.config import Settings
from app.exceptions import AppError
from app.models import DocumentUploadResponse, RetrievedChunk
from app.rag.loader import load_markdown, load_pdf
from app.rag.splitter import split_pages
from app.rag.vector_store import VectorStoreService

logger = logging.getLogger(__name__)
_INGEST_LOCK = asyncio.Lock()


def safe_filename(filename: str | None) -> str:
    candidate = (filename or "document").replace("\\", "/")
    return Path(candidate).name or "document"


class RAGService:
    """Coordinate safe uploads, parsing, chunking, persistence, and retrieval."""

    def __init__(self, settings: Settings, vector_store: VectorStoreService) -> None:
        self.settings = settings
        self.vector_store = vector_store

    async def ingest_document(self, upload: UploadFile) -> DocumentUploadResponse:
        filename = safe_filename(upload.filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in {".pdf", ".md"}:
            raise AppError("INVALID_FILE_TYPE", "仅支持 PDF 和 Markdown 文件", 400)

        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = self.settings.upload_dir / f".upload-{uuid4().hex}"
        digest = hashlib.sha256()
        total = 0
        maximum = self.settings.max_upload_size_mb * 1024 * 1024
        try:
            with temporary_path.open("wb") as output:
                while data := await upload.read(1024 * 1024):
                    total += len(data)
                    if total > maximum:
                        raise AppError("FILE_TOO_LARGE", "上传文件超过大小限制", 413)
                    digest.update(data)
                    output.write(data)
        except AppError:
            temporary_path.unlink(missing_ok=True)
            raise
        except OSError as exc:
            temporary_path.unlink(missing_ok=True)
            raise AppError("INTERNAL_ERROR", "上传文件保存失败", 500) from exc
        finally:
            await upload.close()

        document_id = digest.hexdigest()
        stored_path = self.settings.upload_dir / f"{document_id}{suffix}"
        source_type = "pdf" if suffix == ".pdf" else "markdown"
        try:
            async with _INGEST_LOCK:
                return await self._persist_document(
                    temporary_path=temporary_path,
                    stored_path=stored_path,
                    document_id=document_id,
                    filename=filename,
                    suffix=suffix,
                    source_type=source_type,
                )
        finally:
            temporary_path.unlink(missing_ok=True)

    async def _persist_document(
        self,
        *,
        temporary_path: Path,
        stored_path: Path,
        document_id: str,
        filename: str,
        suffix: str,
        source_type: str,
    ) -> DocumentUploadResponse:
        """Deduplicate and index one upload inside the process-wide critical section."""
        file_moved = False
        vector_write_started = False
        try:
            existing = await self.vector_store.document_chunks(document_id)
            if existing:
                return DocumentUploadResponse(
                    document_id=document_id,
                    filename=filename,
                    source_type=source_type,
                    page_count=max(
                        int(item.get("page_count", item.get("page", 1))) for item in existing
                    ),
                    chunk_count=len(existing),
                    duplicate=True,
                )

            await asyncio.to_thread(temporary_path.replace, stored_path)
            file_moved = True
            pages = load_pdf(stored_path) if suffix == ".pdf" else load_markdown(stored_path)
            if suffix == ".pdf":
                with pymupdf.open(stored_path) as pdf:
                    page_count = pdf.page_count
            else:
                page_count = 1
            chunks = split_pages(
                pages,
                document_id=document_id,
                filename=filename,
                source_type=source_type,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                page_count=page_count,
            )
            if not chunks:
                raise AppError("EMPTY_DOCUMENT", "文档中没有有效文本", 400)
            vector_write_started = True
            await self.vector_store.add_chunks(chunks)
        except Exception:
            if vector_write_started:
                await self.vector_store.delete_document(document_id)
            if file_moved:
                await asyncio.to_thread(stored_path.unlink, missing_ok=True)
            raise

        logger.info(
            "Document ingested id=%s pages=%d chunks=%d",
            document_id[:12],
            page_count,
            len(chunks),
        )
        return DocumentUploadResponse(
            document_id=document_id,
            filename=filename,
            source_type=source_type,
            page_count=page_count,
            chunk_count=len(chunks),
            duplicate=False,
        )

    async def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        cleaned = query.strip()
        if not cleaned:
            raise AppError("INVALID_TOOL_ARGUMENTS", "检索词不能为空", 422)
        if not 1 <= top_k <= self.settings.max_top_k:
            raise AppError("INVALID_TOOL_ARGUMENTS", "top_k 超出允许范围", 422)

        raw = await self.vector_store.search(cleaned, top_k)
        results: list[RetrievedChunk] = []
        seen: set[tuple[str, str, int]] = set()
        for item in raw:
            key = (item.content, item.filename, item.page)
            if key in seen:
                continue
            seen.add(key)
            item.source_id = f"S{len(results) + 1}"
            item.content = item.content[:500]
            results.append(item)
        return results
