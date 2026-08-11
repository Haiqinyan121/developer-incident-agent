import asyncio
from typing import Any

import httpx
from langchain_core.messages import AIMessage
from openai import APITimeoutError

from app.models import BlogArticleResult, ChatResponse, DocumentUploadResponse, RetrievedChunk


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


class FakeVectorStore:
    def __init__(self, search_results: list[RetrievedChunk] | None = None) -> None:
        self.chunks: dict[str, list[Any]] = {}
        self.search_results = search_results
        self.add_calls = 0

    async def document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        return [chunk.metadata for chunk in self.chunks.get(document_id, [])]

    async def add_chunks(self, chunks: list[Any]) -> None:
        self.add_calls += 1
        if chunks:
            self.chunks[chunks[0].metadata["document_id"]] = list(chunks)

    async def delete_document(self, document_id: str) -> None:
        self.chunks.pop(document_id, None)

    async def search(self, _query: str, top_k: int) -> list[RetrievedChunk]:
        if self.search_results is not None:
            return [item.model_copy(deep=True) for item in self.search_results[:top_k]]
        results: list[RetrievedChunk] = []
        for chunks in self.chunks.values():
            for chunk in chunks:
                results.append(
                    RetrievedChunk(
                        content=chunk.content,
                        filename=str(chunk.metadata["filename"]),
                        page=int(chunk.metadata["page"]),
                        document_id=str(chunk.metadata["document_id"]),
                        distance=0.1,
                    )
                )
        return results[:top_k]


class FakeRAGService:
    def __init__(self, results: list[RetrievedChunk] | None = None) -> None:
        self.results = results if results is not None else [
            RetrievedChunk(
                source_id="S1",
                content="连接池超时时应先区分建连超时、池耗尽和下游响应变慢。",
                filename="redis_connection_timeout.md",
                page=1,
                document_id="doc-1",
                distance=0.1,
            )
        ]
        self.calls = 0
        self.last_query = ""
        self.last_top_k = 0

    async def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.calls += 1
        self.last_query = query
        self.last_top_k = top_k
        return [item.model_copy(deep=True) for item in self.results[:top_k]]

    async def ingest_document(self, _upload: Any) -> DocumentUploadResponse:
        return DocumentUploadResponse(
            document_id="fake",
            filename="fake.md",
            source_type="markdown",
            page_count=1,
            chunk_count=1,
            duplicate=False,
        )


class FakeBlogClient:
    def __init__(self, results: list[BlogArticleResult] | None = None) -> None:
        self.results = results if results is not None else [
            BlogArticleResult(
                article_id=5,
                title="Go 服务连接池超时排查记录",
                excerpt="本文记录连接池耗尽与 goroutine 堆积的排查过程。",
                likes=3,
                author="alice",
                tags=["Go", "故障排查"],
            )
        ]
        self.calls = 0
        self.last_keyword = ""

    async def search_articles(self, keyword: str) -> list[BlogArticleResult]:
        self.calls += 1
        self.last_keyword = keyword
        return [item.model_copy(deep=True) for item in self.results]


class FakeChatModel:
    def __init__(self, mode: str = "documents") -> None:
        self.mode = mode
        self.calls = 0
        self.bound = False
        self.bound_tools: list[Any] = []
        self.bound_invocations = 0
        self.unbound_invocations = 0

    def bind_tools(self, tools: list[Any]) -> "FakeBoundChatModel":
        self.bound = True
        self.bound_tools = tools
        return FakeBoundChatModel(self)

    async def ainvoke(self, _messages: list[Any]) -> AIMessage:
        self.unbound_invocations += 1
        return await self._respond()

    async def _respond(self) -> AIMessage:
        self.calls += 1
        if self.mode == "timeout":
            raise TimeoutError
        if self.mode == "httpx_timeout":
            raise httpx.ReadTimeout("fake model timeout")
        if self.mode == "openai_timeout":
            raise APITimeoutError(request=httpx.Request("POST", "http://model.test"))
        if self.mode == "failure":
            raise RuntimeError("fake model failure")
        if self.calls > 1:
            if self.mode == "second_tool_attempt":
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_articles",
                            "args": {"keyword": "Go"},
                            "id": "call-2",
                            "type": "tool_call",
                        }
                    ],
                )
            if self.mode == "fabricated_citation":
                return AIMessage(content="这是不存在的来源。[S99]")
            if self.mode == "empty_results":
                return AIMessage(content="在已上传文档中未找到足够依据。")
            if self.mode == "empty_articles":
                return AIMessage(content="博客中没有找到匹配文章。")
            if self.mode == "articles":
                return AIMessage(content="博客中找到一篇 Go 服务排障文章。[B1]")
            return AIMessage(
                content="症状：连接池超时。可能原因：连接池耗尽。"
                "排查步骤：检查活跃连接数和下游延迟。[S1]"
            )
        if self.mode == "direct":
            return AIMessage(content="你好，有什么技术问题需要帮助？")
        if self.mode == "multiple":
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "Redis"},
                        "id": "call-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "search_articles",
                        "args": {"keyword": "Go"},
                        "id": "call-2",
                        "type": "tool_call",
                    },
                ],
            )
        if self.mode == "invalid_name":
            name, args = "unknown_tool", {}
        elif self.mode == "invalid_args":
            name, args = "search_documents", {"query": " "}
        elif self.mode == "extra_args":
            name, args = "search_documents", {"query": "Redis", "unknown": True}
        elif self.mode in {"articles", "empty_articles"}:
            name, args = "search_articles", {"keyword": "连接池超时"}
        else:
            name, args = "search_documents", {"query": "Redis 连接池超时"}
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": "call-1", "type": "tool_call"}],
        )


class FakeBoundChatModel:
    def __init__(self, model: FakeChatModel) -> None:
        self.model = model

    async def ainvoke(self, _messages: list[Any]) -> AIMessage:
        self.model.bound_invocations += 1
        return await self.model._respond()


class FakeAgentService:
    def __init__(self, response: ChatResponse) -> None:
        self.response = response

    async def chat(self, _question: str, _top_k: int) -> ChatResponse:
        await asyncio.sleep(0)
        return self.response
