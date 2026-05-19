"""Load text content from supported document formats."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_document(file_path: Path) -> tuple[str, str]:
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8"), suffix.lstrip(".")
    if suffix == ".pdf":
        return _load_pdf(file_path), "pdf"
    raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(file_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    content = "\n\n".join(page.strip() for page in pages if page.strip())
    if not content:
        raise ValueError(f"No extractable text found in PDF: {file_path}")
    return content
