from typing import Annotated, Any

from fastapi import Depends, Request
from langchain_openai import ChatOpenAI

from app.agent.service import AgentService
from app.config import Settings, get_settings
from app.exceptions import AppError
from app.integrations.blog_client import BlogClient
from app.rag.service import RAGService
from app.rag.vector_store import VectorStoreService

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_vector_store(request: Request, settings: SettingsDep) -> VectorStoreService:
    service = getattr(request.app.state, "vector_store", None)
    if service is None:
        service = VectorStoreService(settings)
        request.app.state.vector_store = service
    return service


def get_rag_service(
    request: Request,
    settings: SettingsDep,
    vector_store: Annotated[VectorStoreService, Depends(get_vector_store)],
) -> RAGService:
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        service = RAGService(settings, vector_store)
        request.app.state.rag_service = service
    return service


def get_blog_client(request: Request, settings: SettingsDep) -> BlogClient:
    service = getattr(request.app.state, "blog_client", None)
    if service is None:
        service = BlogClient(settings)
        request.app.state.blog_client = service
    return service


def get_chat_model(settings: SettingsDep) -> ChatOpenAI:
    if not settings.chat_configured:
        raise AppError("MODEL_NOT_CONFIGURED", "尚未配置大模型服务", 503)
    model_options: dict[str, Any] = {}
    if settings.model_reasoning_effort:
        model_options["reasoning_effort"] = settings.model_reasoning_effort
    if settings.disable_response_storage:
        model_options["store"] = False
    return ChatOpenAI(
        api_key=settings.resolved_chat_api_key,
        base_url=settings.resolved_chat_base_url,
        model=settings.chat_model,
        timeout=settings.model_timeout_seconds,
        max_retries=0,
        use_responses_api=settings.use_responses_api,
        **model_options,
    )


def get_agent_service(
    model: Annotated[ChatOpenAI, Depends(get_chat_model)],
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
    blog_client: Annotated[BlogClient, Depends(get_blog_client)],
) -> AgentService:
    return AgentService(model, rag_service, blog_client)
