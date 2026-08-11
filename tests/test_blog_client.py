import json

import httpx
import pytest

from app.config import Settings
from app.exceptions import AppError
from app.integrations.blog_client import BlogClient


def response_transport(payload: object, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/article/search"
        assert request.url.params["keyword"] == "Go"
        assert "authorization" not in request.headers
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


async def call_client(settings: Settings, payload: object, status: int = 200):
    async with httpx.AsyncClient(
        transport=response_transport(payload, status),
        base_url="http://blog",
    ) as client:
        return await BlogClient(settings, client).search_articles(" Go ")


async def test_blog_normal_response_is_parsed(settings: Settings) -> None:
    payload = {
        "code": 0,
        "message": "success",
        "data": [
            {
                "id": 5,
                "title": "Go 学习笔记",
                "content": "内容",
                "likes": 3,
                "author": {"username": "alice"},
                "tags": [{"name": "Go"}],
            }
        ],
    }
    result = await call_client(settings, payload)
    assert result[0].article_id == 5
    assert result[0].author == "alice"
    assert result[0].tags == ["Go"]


async def test_empty_blog_data_is_normal(settings: Settings) -> None:
    assert await call_client(settings, {"code": 0, "data": []}) == []


async def test_missing_optional_blog_fields_is_allowed(settings: Settings) -> None:
    result = await call_client(
        settings,
        {"code": 0, "data": [{"id": 1, "title": "Go", "content": "body"}]},
    )
    assert result[0].author is None
    assert result[0].tags == []


async def test_blog_content_is_limited_to_500(settings: Settings) -> None:
    result = await call_client(
        settings,
        {"code": 0, "data": [{"id": 1, "title": "Go", "content": "x" * 700}]},
    )
    assert len(result[0].excerpt) == 500


@pytest.mark.parametrize(
    "payload",
    [
        {"code": 1, "data": []},
        {"code": 0, "data": {}},
        {"code": 0, "data": ["bad"]},
    ],
)
async def test_invalid_blog_structure_is_controlled(
    settings: Settings,
    payload: object,
) -> None:
    with pytest.raises(AppError, match="INVALID_BLOG_RESPONSE"):
        await call_client(settings, payload)


async def test_blog_http_500_is_unavailable(settings: Settings) -> None:
    with pytest.raises(AppError, match="BLOG_SERVICE_UNAVAILABLE"):
        await call_client(settings, {"error": True}, status=500)


@pytest.mark.parametrize(
    ("exception", "code"),
    [
        (httpx.ReadTimeout("late"), "BLOG_SERVICE_TIMEOUT"),
        (httpx.ConnectError("down"), "BLOG_SERVICE_UNAVAILABLE"),
    ],
)
async def test_blog_network_errors_are_mapped(
    settings: Settings,
    exception: Exception,
    code: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise exception

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://blog",
    ) as client:
        with pytest.raises(AppError, match=code):
            await BlogClient(settings, client).search_articles("Go")


async def test_invalid_json_is_controlled(settings: Settings) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not json")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://blog",
    ) as client:
        with pytest.raises(AppError, match="INVALID_BLOG_RESPONSE"):
            await BlogClient(settings, client).search_articles("Go")


async def test_blog_result_limit_is_applied(settings: Settings) -> None:
    payload = {
        "code": 0,
        "data": [{"id": index, "title": "Go", "content": json.dumps(index)} for index in range(8)],
    }
    result = await call_client(settings, payload)
    assert len(result) == settings.blog_result_limit


async def test_empty_blog_keyword_is_rejected(settings: Settings) -> None:
    with pytest.raises(AppError, match="INVALID_TOOL_ARGUMENTS"):
        await BlogClient(settings).search_articles(" ")
