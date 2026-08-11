import json
from typing import Any

import httpx

from app.config import Settings
from app.exceptions import AppError
from app.models import BlogArticleResult


class BlogClient:
    """Asynchronous client for the existing Go blog title-search endpoint."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client

    async def search_articles(self, keyword: str) -> list[BlogArticleResult]:
        cleaned = keyword.strip()
        if not cleaned:
            raise AppError("INVALID_TOOL_ARGUMENTS", "博客搜索关键词不能为空", 422)

        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            base_url=self.settings.blog_api_base_url.rstrip("/"),
            timeout=self.settings.blog_api_timeout_seconds,
        )
        try:
            response = await client.get("/api/article/search", params={"keyword": cleaned})
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            raise AppError("BLOG_SERVICE_TIMEOUT", "博客服务请求超时", 504) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise AppError("BLOG_SERVICE_UNAVAILABLE", "博客服务不可用", 503) from exc
        except httpx.HTTPStatusError as exc:
            raise AppError("BLOG_SERVICE_UNAVAILABLE", "博客服务返回异常状态", 502) from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise AppError("INVALID_BLOG_RESPONSE", "博客服务响应不是有效 JSON", 502) from exc
        finally:
            if owns_client:
                await client.aclose()

        if not isinstance(payload, dict) or payload.get("code") != 0:
            raise AppError("INVALID_BLOG_RESPONSE", "博客服务返回业务错误", 502)
        data = payload.get("data")
        if not isinstance(data, list):
            raise AppError("INVALID_BLOG_RESPONSE", "博客服务响应结构无效", 502)

        results: list[BlogArticleResult] = []
        for item in data[: self.settings.blog_result_limit]:
            if not isinstance(item, dict):
                raise AppError("INVALID_BLOG_RESPONSE", "博客文章结构无效", 502)
            try:
                results.append(self._parse_article(item))
            except (TypeError, ValueError, KeyError) as exc:
                raise AppError("INVALID_BLOG_RESPONSE", "博客文章结构无效", 502) from exc
        return results

    @staticmethod
    def _parse_article(item: dict[str, Any]) -> BlogArticleResult:
        author_data = item.get("author")
        author = author_data.get("username") if isinstance(author_data, dict) else None
        tags_data = item.get("tags")
        tags = (
            [str(tag["name"]) for tag in tags_data if isinstance(tag, dict) and "name" in tag]
            if isinstance(tags_data, list)
            else []
        )
        return BlogArticleResult(
            article_id=int(item["id"]),
            title=str(item["title"]),
            excerpt=str(item.get("content", ""))[:500],
            likes=int(item.get("likes", 0)),
            author=str(author) if author is not None else None,
            tags=tags,
        )
