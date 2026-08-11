from typing import Any

from app.config import Settings
from app.dependencies import get_chat_model


def test_chat_model_forwards_responses_api_and_reasoning_options() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        chat_model="test-model",
        use_responses_api=True,
        model_reasoning_effort="xhigh",
        disable_response_storage=True,
    )

    model = get_chat_model(settings)

    assert model.use_responses_api is True
    assert model.reasoning_effort == "xhigh"
    assert model.store is False


def test_chat_model_uses_dedicated_provider_configuration(monkeypatch: Any) -> None:
    captured: dict[str, object] = {}

    class CapturingChatModel:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("app.dependencies.ChatOpenAI", CapturingChatModel)
    settings = Settings(
        _env_file=None,
        openai_api_key="shared-key",
        openai_base_url="https://shared.example/v1",
        chat_api_key="chat-key",
        chat_base_url="https://chat.example/v1/",
        chat_model="chat-model",
    )

    get_chat_model(settings)

    assert captured["api_key"] == "chat-key"
    assert captured["base_url"] == "https://chat.example/v1"
    assert captured["model"] == "chat-model"


def test_shared_provider_configuration_remains_a_fallback() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="shared-key",
        openai_base_url="https://shared.example/v1/",
        chat_model="chat-model",
        embedding_model="embedding-model",
    )

    assert settings.resolved_chat_api_key == "shared-key"
    assert settings.resolved_embedding_api_key == "shared-key"
    assert settings.resolved_chat_base_url == "https://shared.example/v1"
    assert settings.resolved_embedding_base_url == "https://shared.example/v1"
