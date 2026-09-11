#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""Loading source documents into generation-ready chunks.

A `Chunk` is one prompt-worth of source content: for a .txt/.md file this is
the entire file (front matter included, unmodified); for a .pdf file this is
a group of up to `pages_per_chunk` rendered page images. Generation and judge
validation both operate on a `Chunk`, so the rest of the pipeline doesn't need
to know or care which source type produced it.
"""

import base64
import hashlib
import io
from dataclasses import dataclass
from os import path
from pathlib import Path
from typing import Any

import frontmatter
import pypdfium2 as pdfium
from pydantic_ai import BinaryContent

SUPPORTED_TEXT_SUFFIXES = {".txt", ".md"}


@dataclass
class Chunk:
    index: int
    content: str | list[dict[str, Any]]  # dropped directly into a chat message's "content" field
    pages: tuple[int, int] | None = None  # 1-based inclusive (first, last) page range; None for text sources
    # For PDFs, this is a tuple of (first_page, last_page) 1-based and inclusive.
    # For text files, this is None.


def load_chunks(
    file_path: Path, *, 
    pages_per_chunk: int = 5, 
    image_scale: float = 2.0,
    hash_len: int = 12
) -> tuple[list[Chunk], dict[str, Any], str]:
    """Load a source document into one or more generation-ready chunks.

    Returns (chunks, metadata, source_id). Raises NotImplementedError for any
    suffix other than .txt/.md/.pdf.
    """
    with open(file_path, "rb") as f:
        source_id = hashlib.file_digest(f, "sha256").hexdigest()[:hash_len]

    if file_path.suffix in SUPPORTED_TEXT_SUFFIXES:
        post = frontmatter.load(file_path)
        return [Chunk(index=0, content=post.content)], post.metadata, source_id
    
    elif file_path.suffix == ".pdf":
        chunks = _load_pdf_chunks(file_path, pages_per_chunk, image_scale)
        metadata = {
            "source_path": str(file_path),
        }
        return chunks, metadata, source_id
    
    else:
        raise NotImplementedError(f"Document type {file_path.suffix} not supported yet.")


def _page_chunk_ranges(total_pages: int, pages_per_chunk: int) -> list[tuple[int, int]]:
    """Group page indices [0, total_pages) into (start, end) ranges (end
    exclusive) of at most `pages_per_chunk` pages each."""
    return [
        (start, min(start + pages_per_chunk, total_pages))
        for start in range(0, total_pages, pages_per_chunk)
    ]


def render_pdf_pages_png(file_path: Path, *, image_scale: float = 2.0) -> list[bytes]:
    """Render every page of a PDF to PNG bytes via pypdfium2 (no OCR, no text
    extraction). One bytes object per page, in page order."""
    doc = pdfium.PdfDocument(str(file_path))
    try:
        pages_png: list[bytes] = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            bitmap = page.render(scale=image_scale)
            pil_image = bitmap.to_pil()
            bitmap.close()
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            pages_png.append(buffer.getvalue())
    finally:
        doc.close()
    return pages_png


def _load_pdf_chunks(file_path: Path, pages_per_chunk: int, image_scale: float) -> list[Chunk]:
    """Render each PDF page directly to an image via pypdfium2 (no OCR, no
    text extraction, no docling document-understanding pipeline involved) and
    group pages into chunks of at most `pages_per_chunk` images each."""
    pages_png = render_pdf_pages_png(file_path, image_scale=image_scale)
    total_pages = len(pages_png)

    chunks: list[Chunk] = []
    for chunk_index, (start, end) in enumerate(_page_chunk_ranges(total_pages, pages_per_chunk)):
        content: list[str | BinaryContent] = [
            f"Pages {start + 1}-{end} of a {total_pages}-page document."
        ]
        for png_bytes in pages_png[start:end]:
            content.append(BinaryContent(data=png_bytes, media_type="image/png"))
        chunks.append(Chunk(index=chunk_index, content=content, pages=(start + 1, end)))
    return chunks
