from pathlib import Path

import pymupdf
import pytest

from app.exceptions import AppError
from app.rag.loader import load_markdown, load_pdf


def test_markdown_loads_bom_and_keeps_code_block(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("\ufeff# 标题\n\n```python\nprint('ok')\n```\n", encoding="utf-8")
    pages = load_markdown(path)
    assert pages[0].page == 1
    assert "```python" in pages[0].text


def test_pdf_loads_non_empty_pages_with_one_based_numbers(tmp_path: Path) -> None:
    path = tmp_path / "notes.pdf"
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text((72, 72), "first page")
    document.new_page()
    third = document.new_page()
    third.insert_text((72, 72), "third page")
    document.save(path)
    document.close()
    pages = load_pdf(path)
    assert [page.page for page in pages] == [1, 3]
    assert pages[0].text == "first page"


@pytest.mark.parametrize("suffix", [".md", ".pdf"])
def test_empty_document_is_rejected(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"empty{suffix}"
    if suffix == ".md":
        path.write_text(" \n", encoding="utf-8")
        loader = load_markdown
    else:
        document = pymupdf.open()
        document.new_page()
        document.save(path)
        document.close()
        loader = load_pdf
    with pytest.raises(AppError, match="EMPTY_DOCUMENT"):
        loader(path)


def test_invalid_pdf_is_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"not a pdf")
    with pytest.raises(AppError, match="DOCUMENT_PARSE_FAILED"):
        load_pdf(path)
