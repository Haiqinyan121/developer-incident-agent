import httpx
import pytest

from app.agent.service import AgentService
from app.config import Settings
from app.dependencies import get_agent_service
from app.main import create_app
from tests.fakes import FakeBlogClient, FakeChatModel, FakeRAGService


def chat_client(settings: Settings, mode: str) -> httpx.AsyncClient:
    app = create_app(settings)
    agent = AgentService(FakeChatModel(mode), FakeRAGService(), FakeBlogClient())
    app.dependency_overrides[get_agent_service] = lambda: agent
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"question": " ", "top_k": 4},
        {"question": "ok", "top_k": 0},
        {"question": "ok", "top_k": 11},
        {"question": "x" * 2001, "top_k": 4},
    ],
)
async def test_chat_validation_returns_422(settings: Settings, payload: dict) -> None:
    async with chat_client(settings, "direct") as client:
        response = await client.post("/chat", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("mode", "expected_mode", "source_prefix"),
    [
        ("documents", "tool", "S"),
        ("articles", "tool", "B"),
        ("direct", "direct", None),
    ],
)
async def test_chat_modes_work_without_real_api_key(
    settings: Settings,
    mode: str,
    expected_mode: str,
    source_prefix: str | None,
) -> None:
    async with chat_client(settings, mode) as client:
        response = await client.post("/chat", json={"question": "测试问题", "top_k": 4})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == expected_mode
    if source_prefix:
        assert body["sources"][0]["source_id"].startswith(source_prefix)
    else:
        assert body["sources"] == []
        assert body["tool_call"] is None


async def test_unconfigured_real_chat_dependency_returns_error(settings: Settings) -> None:
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/chat", json={"question": "你好", "top_k": 4})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_NOT_CONFIGURED"


async def test_chat_model_timeout_uses_common_error(settings: Settings) -> None:
    async with chat_client(settings, "timeout") as client:
        response = await client.post("/chat", json={"question": "问题", "top_k": 4})
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "MODEL_TIMEOUT"

