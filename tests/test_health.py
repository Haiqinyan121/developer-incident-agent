import httpx

from app import __version__


async def test_health_returns_ok_without_api_key(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "Developer Incident Agent",
        "version": "0.1.0",
        "chat_model_configured": False,
        "embedding_model_configured": False,
        "storage_ready": True,
    }


async def test_openapi_contains_only_core_routes(client: httpx.AsyncClient) -> None:
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert {"/health", "/documents", "/chat"} <= set(paths)
    assert response.json()["info"]["version"] == __version__
