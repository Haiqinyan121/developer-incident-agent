from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.loader import DocumentPage


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    content: str
    metadata: dict[str, str | int]


def split_pages(
    pages: list[DocumentPage],
    *,
    document_id: str,
    filename: str,
    source_type: str,
    chunk_size: int,
    chunk_overlap: int,
    page_count: int | None = None,
) -> list[DocumentChunk]:
    """Split each page independently and attach stable source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    chunks: list[DocumentChunk] = []
    for page in pages:
        for index, content in enumerate(splitter.split_text(page.text)):
            if not content.strip():
                continue
            metadata: dict[str, str | int] = {
                "document_id": document_id,
                "filename": filename,
                "page": page.page,
                "chunk_index": index,
                "source_type": source_type,
            }
            if page_count is not None:
                metadata["page_count"] = page_count
            chunks.append(
                DocumentChunk(
                    id=f"{document_id}:{page.page}:{index}",
                    content=content,
                    metadata=metadata,
                )
            )
    return chunks
