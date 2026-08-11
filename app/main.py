import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, File, UploadFile

from app import __version__
from app.agent.service import AgentService
from app.config import Settings, get_settings
from app.dependencies import SettingsDep, get_agent_service, get_rag_service
from app.exceptions import register_exception_handlers
from app.models import ChatRequest, ChatResponse, DocumentUploadResponse, HealthResponse
from app.rag.service import RAGService

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health(settings: SettingsDep) -> HealthResponse:
    storage_ready = True
    try:
        settings.ensure_data_dirs()
    except OSError:
        storage_ready = False
    return HealthResponse(
        app=settings.app_name,
        version=__version__,
        chat_model_configured=settings.chat_configured,
        embedding_model_configured=settings.embedding_configured,
        storage_ready=storage_ready,
    )


@router.post("/documents", response_model=DocumentUploadResponse, tags=["documents"])
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF 或 Markdown 故障手册或技术文档")],
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> DocumentUploadResponse:
    return await rag_service.ingest_document(file)


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    request: ChatRequest,
    agent: Annotated[AgentService, Depends(get_agent_service)],
) -> ChatResponse:
    return await agent.chat(request.question, request.top_k)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an application whose external services remain lazy and replaceable."""
    application_settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, application_settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        application_settings.ensure_data_dirs()
        yield

    app = FastAPI(
        title=application_settings.app_name,
        version=__version__,
        description="基于 LangChain Tool Calling 与 RAG 的轻量研发故障工单诊断 Agent",
        lifespan=lifespan,
    )
    app.dependency_overrides[get_settings] = lambda: application_settings
    app.include_router(router)
    register_exception_handlers(app)
    return app


app = create_app()
