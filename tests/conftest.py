from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.main import create_app


@pytest.fixture(autouse=True)
def isolate_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore developer settings and fail if a test uses production HTTP transport."""

    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)

    async def deny_network(*_args: object, **_kwargs: object) -> None:
        pytest.fail("测试禁止访问真实 HTTP 网络")

    monkeypatch.setattr(
        httpx.AsyncHTTPTransport,
        "handle_async_request",
        deny_network,
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="",
        chat_model="",
        embedding_model="",
        upload_dir=tmp_path / "uploads",
        chroma_persist_dir=tmp_path / "chroma",
    )


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client
