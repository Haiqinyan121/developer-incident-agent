from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app: str
    version: str
    chat_model_configured: bool
    embedding_model_configured: bool
    storage_ready: bool


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    source_type: Literal["pdf", "markdown"]
    page_count: int
    chunk_count: int
    duplicate: bool


class RetrievedChunk(BaseModel):
    source_id: str = ""
    content: str
    filename: str
    page: int
    document_id: str
    distance: float | None = None


class BlogArticleResult(BaseModel):
    article_id: int
    title: str
    excerpt: str
    likes: int = 0
    author: str | None = None
    tags: list[str] = Field(default_factory=list)


class SearchDocumentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query 不能为空")
        return value


class SearchArticlesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(min_length=1, max_length=1000)

    @field_validator("keyword")
    @classmethod
    def strip_keyword(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("keyword 不能为空") 
        return value


class DocumentSource(BaseModel):
    source_id: str
    source_type: Literal["document"]
    excerpt: str
    filename: str
    page: int
    document_id: str


class BlogSource(BaseModel):
    source_id: str
    source_type: Literal["blog"]
    excerpt: str
    article_id: int
    title: str
    likes: int = 0
    author: str | None = None
    tags: list[str] = Field(default_factory=list)


Source = DocumentSource | BlogSource


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any]
    success: bool


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question 不能为空")
        return value


class ChatResponse(BaseModel):
    answer: str
    mode: Literal["direct", "tool"]
    sources: list[Source] = Field(default_factory=list)
    tool_call: ToolCallRecord | None = None
