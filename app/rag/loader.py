from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.exceptions import AppError


@dataclass(frozen=True)
class DocumentPage:
    page: int
    text: str


def _clean_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def load_pdf(path: Path) -> list[DocumentPage]:
    """Extract non-empty PDF pages while preserving one-based page numbers."""
    try:
        with pymupdf.open(path) as document:
            pages = [
                DocumentPage(page=index + 1, text=cleaned)
                for index, page in enumerate(document)
                if (cleaned := _clean_text(page.get_text()))
            ]
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise AppError("DOCUMENT_PARSE_FAILED", "PDF 文档解析失败", 400) from exc
    if not pages:
        raise AppError("EMPTY_DOCUMENT", "文档中没有可提取的文本", 400)
    return pages


def load_markdown(path: Path) -> list[DocumentPage]:
    """Load one UTF-8 Markdown page, accepting an optional BOM."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise AppError("DOCUMENT_PARSE_FAILED", "Markdown 文档解析失败", 400) from exc
    cleaned = _clean_text(text)
    if not cleaned:
        raise AppError("EMPTY_DOCUMENT", "文档中没有有效文本", 400)
    return [DocumentPage(page=1, text=cleaned)]
