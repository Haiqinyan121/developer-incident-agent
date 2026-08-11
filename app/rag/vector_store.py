import asyncio
import logging
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import Settings
from app.exceptions import AppError
from app.models import RetrievedChunk
from app.rag.splitter import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Lazy Chroma adapter used for persistent vector ingestion and search."""

    def __init__(
        self,
        settings: Settings,
        *,
        embeddings: Any | None = None,
        store: Any | None = None,
    ) -> None:
        self.settings = settings
        self._embeddings = embeddings
        self._store = store
        self._raw_client: Any | None = None
        self._raw_collection: Any | None = None

    def _get_embeddings(self) -> Any:
        if self._embeddings is not None:
            return self._embeddings
        if not self.settings.embedding_configured:
            raise AppError(
                "EMBEDDING_NOT_CONFIGURED",
                "尚未配置 Embedding 服务",
                503,
            )
        self._embeddings = OpenAIEmbeddings(
            api_key=self.settings.resolved_embedding_api_key,
            base_url=self.settings.resolved_embedding_base_url,
            model=self.settings.embedding_model,
            request_timeout=self.settings.model_timeout_seconds,
            check_embedding_ctx_length=False,
        )
        return self._embeddings

    def _get_store(self) -> Any:
        if self._store is None:
            try:
                self.settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
                self._store = Chroma(
                    collection_name=self.settings.chroma_collection_name,
                    persist_directory=str(self.settings.chroma_persist_dir),
                    embedding_function=self._get_embeddings(),
                )
            except AppError:
                raise
            except Exception as exc:
                raise AppError("VECTOR_STORE_ERROR", "向量存储初始化失败", 500) from exc
        return self._store

    def _get_raw_collection(self) -> Any:
        """Return a local collection handle that does not require an embedding client."""
        if self._raw_collection is None:
            try:
                self.settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
                self._raw_client = chromadb.PersistentClient(
                    path=str(self.settings.chroma_persist_dir)
                )
                self._raw_collection = self._raw_client.get_or_create_collection(
                    name=self.settings.chroma_collection_name
                )
            except Exception as exc:
                raise AppError("VECTOR_STORE_ERROR", "向量存储初始化失败", 500) from exc
        return self._raw_collection

    async def document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        try:
            collection = self._store if self._store is not None else self._get_raw_collection()
            result = await asyncio.to_thread(
                collection.get,
                where={"document_id": document_id},
                include=["metadatas"],
            )
            return list(result.get("metadatas") or [])
        except AppError:
            raise
        except Exception as exc:
            raise AppError("VECTOR_STORE_ERROR", "读取向量存储失败", 500) from exc

    async def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        try:
            await asyncio.to_thread(
                self._get_store().add_texts,
                texts=[chunk.content for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                ids=[chunk.id for chunk in chunks],
            )
        except AppError:
            raise
        except Exception as exc:
            raise AppError("VECTOR_STORE_ERROR", "文档向量写入失败", 502) from exc

    async def delete_document(self, document_id: str) -> None:
        try:
            collection = self._store if self._store is not None else self._get_raw_collection()
            await asyncio.to_thread(
                collection.delete,
                where={"document_id": document_id},
            )
        except Exception:
            logger.warning(
                "Failed to clean vector data document_id=%s",
                document_id[:12],
                exc_info=True,
            )

    async def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        try:
            matches = await asyncio.to_thread(
                self._get_store().similarity_search_with_score,
                query,
                k=top_k,
            )
        except AppError:
            raise
        except Exception as exc:
            raise AppError("VECTOR_STORE_ERROR", "向量检索失败", 502) from exc

        results: list[RetrievedChunk] = []
        for document, distance in matches:
            metadata = document.metadata
            results.append(
                RetrievedChunk(
                    content=document.page_content,
                    filename=str(metadata.get("filename", "")),
                    page=int(metadata.get("page", 1)),
                    document_id=str(metadata.get("document_id", "")),
                    distance=float(distance),
                )
            )
        return results
