from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from app.integrations.blog_client import BlogClient
from app.models import SearchArticlesInput, SearchDocumentsInput
from app.rag.service import RAGService

SEARCH_DOCUMENTS_DESCRIPTION = (
    "检索用户上传的 PDF 或 Markdown 故障手册、运行手册和技术文档。当工单包含错误日志、"
    "超时、资源异常或需要依据内部资料定位故障时使用。工具返回真实文档片段、文件名和页码。"
)
SEARCH_ARTICLES_DESCRIPTION = (
    "按标题关键词搜索用户自己的 Go 技术博客文章。只有当用户明确要求查询博客文章，"
    "或在博客中寻找相关排障实践时使用。它不是语义检索，不能代替故障手册检索。"
)


def tool_schemas() -> list[dict[str, Any]]:
    """Return OpenAI-compatible native function tool schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": SEARCH_DOCUMENTS_DESCRIPTION,
                "parameters": SearchDocumentsInput.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_articles",
                "description": SEARCH_ARTICLES_DESCRIPTION,
                "parameters": SearchArticlesInput.model_json_schema(),
            },
        },
    ]


def create_tools(rag_service: RAGService, blog_client: BlogClient) -> list[BaseTool]:
    """Create dependency-bound LangChain tools used by the agent."""

    async def document_tool(query: str, top_k: int = 4) -> list[dict[str, Any]]:
        arguments = SearchDocumentsInput(query=query, top_k=top_k)
        return await search_documents(rag_service, arguments)

    async def article_tool(keyword: str) -> list[dict[str, Any]]:
        arguments = SearchArticlesInput(keyword=keyword)
        return await search_articles(blog_client, arguments)

    return [
        StructuredTool.from_function(
            coroutine=document_tool,
            name="search_documents",
            description=SEARCH_DOCUMENTS_DESCRIPTION,
            args_schema=SearchDocumentsInput,
        ),
        StructuredTool.from_function(
            coroutine=article_tool,
            name="search_articles",
            description=SEARCH_ARTICLES_DESCRIPTION,
            args_schema=SearchArticlesInput,
        ),
    ]


async def search_documents(
    rag_service: RAGService,
    arguments: SearchDocumentsInput,
) -> list[dict[str, Any]]:
    matches = await rag_service.search(arguments.query, arguments.top_k)
    return [
        {
            "source_id": item.source_id,
            "content": item.content,
            "filename": item.filename,
            "page": item.page,
            "document_id": item.document_id,
        }
        for item in matches
    ]


async def search_articles(
    blog_client: BlogClient,
    arguments: SearchArticlesInput,
) -> list[dict[str, Any]]:
    matches = await blog_client.search_articles(arguments.keyword)
    return [
        {
            "source_id": f"B{index}",
            "article_id": item.article_id,
            "title": item.title,
            "excerpt": item.excerpt,
            "likes": item.likes,
            "author": item.author,
            "tags": item.tags,
        }
        for index, item in enumerate(matches, start=1)
    ]
