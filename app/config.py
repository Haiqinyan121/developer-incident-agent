from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = "Developer Incident Agent"
    log_level: str = "INFO"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    chat_api_key: str = ""
    chat_base_url: str = ""
    chat_model: str = ""
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = ""
    model_timeout_seconds: float = Field(default=30, gt=0)
    use_responses_api: bool = False
    model_reasoning_effort: str = ""
    disable_response_storage: bool = False

    chroma_persist_dir: Path = Path("./data/chroma")
    chroma_collection_name: str = "developer_documents"
    upload_dir: Path = Path("./data/uploads")
    max_upload_size_mb: int = Field(default=10, gt=0)

    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    max_top_k: int = Field(default=10, ge=1, le=10)

    blog_api_base_url: str = "http://localhost:8080"
    blog_api_timeout_seconds: float = Field(default=5, gt=0)
    blog_result_limit: int = Field(default=5, ge=1, le=50)

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
        return self

    @property
    def chat_configured(self) -> bool:
        return bool(self.resolved_chat_api_key and self.chat_model.strip())

    @property
    def embedding_configured(self) -> bool:
        return bool(self.resolved_embedding_api_key and self.embedding_model.strip())

    @property
    def resolved_chat_api_key(self) -> str:
        return self.chat_api_key.strip() or self.openai_api_key.strip()

    @property
    def resolved_chat_base_url(self) -> str:
        return (self.chat_base_url.strip() or self.openai_base_url.strip()).rstrip("/")

    @property
    def resolved_embedding_api_key(self) -> str:
        return self.embedding_api_key.strip() or self.openai_api_key.strip()

    @property
    def resolved_embedding_base_url(self) -> str:
        return (self.embedding_base_url.strip() or self.openai_base_url.strip()).rstrip("/")

    def ensure_data_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""
    return Settings()
