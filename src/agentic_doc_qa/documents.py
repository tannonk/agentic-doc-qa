#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""Loading source documents into generation-ready chunks.

A `Chunk` is one prompt-worth of source content: for a .txt/.md file this is
the entire file (front matter included, unmodified); for a .pdf file with
`vision=True` (default) this is a group of up to `pages_per_chunk` rendered
page images, or with `vision=False` a single Markdown-text chunk converted
via anydoc for use with text-only models. Generation and judge validation
both operate on a `Chunk`, so the rest of the pipeline doesn't need to know
or care which source type produced it.
"""

import base64
import yaml
import hashlib
import io
import random
from dataclasses import dataclass
from os import path
from pathlib import Path
from typing import Any

from loguru import logger
import anydoc
import frontmatter
import pypdfium2 as pdfium
from pydantic_ai import BinaryContent


random.seed(42)  # for reproducible chunk sampling when max_chunks is set

@dataclass
class Chunk:
    index: int
    content: str | list[dict[str, Any]]  # dropped directly into a chat message's "content" field
    pages: tuple[int, int] | None = None  # 1-based inclusive (first, last) page range; None for text sources
    # For PDFs, this is a tuple of (first_page, last_page) 1-based and inclusive.
    # For text files, this is None.


def load_chunks(
    file_path: Path, *,
    vision: bool = True,
    pages_per_chunk: int = 5,
    image_scale: float = 2.0,
    hash_len: int = 12,
    max_chunks: int | None = None
) -> tuple[list[Chunk], dict[str, Any], str]:
    """Load a source document into one or more generation-ready chunks.

    `vision=False` matters only for .pdf sources: instead of rendering pages to
    images, the PDF is converted to Markdown text (via anydoc) for use with a
    text-only model. Returns (chunks, metadata, source_id). Raises
    NotImplementedError for any suffix other than .txt/.md/.pdf.
    """
    with open(file_path, "rb") as f:
        source_id = hashlib.file_digest(f, "sha256").hexdigest()[:hash_len]

    metadata = {
        "source_path": str(file_path),
    }

    if file_path.suffix in ".md":
        chunks, metadata = _load_markdown_chunks(file_path, metadata)

    elif file_path.suffix == ".pdf":
        if vision:
            chunks = _load_pdf_chunks(file_path, pages_per_chunk, image_scale)
        else:
            chunks = _load_other_chunks_as_text(file_path)

    else:
        chunks = _load_other_chunks_as_text(file_path)  # fallback to text conversion for unknown suffixes
    
    if max_chunks is not None and len(chunks) > max_chunks:
        original_chunk_count = len(chunks)
        # sample evenly spaced chunks from the full list
        sample_indices = random.sample(range(len(chunks)), max_chunks)
        chunks = [chunks[i] for i in sorted(sample_indices)]
        logger.debug(f"Sampled {len(chunks)} chunks from {file_path} (originally {original_chunk_count} chunks).")

    return chunks, metadata, source_id


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


def split_markdown_into_paragraph_chunks(markdown_text: str, chunk_size: int = 4092) -> list[Chunk]:
    """Split a Markdown text into chunks of approximately `chunk_size` characters each, preserving paragraph boundaries.
    This is faster and more lightweight than using Docling's more advanced chunking strategies."""
    paragraphs = markdown_text.split("\n\n")
    chunks: list[Chunk] = []
    current_chunk: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current_length + paragraph_length + 2 > chunk_size and current_chunk:
            # Save the current chunk and start a new one
            chunks.append(Chunk(index=len(chunks), content="\n\n".join(current_chunk)))
            current_chunk = []
            current_length = 0

        current_chunk.append(paragraph)
        current_length += paragraph_length + 2  # +2 for the two newlines

    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(Chunk(index=len(chunks), content="\n\n".join(current_chunk)))

    return chunks


def _load_other_chunks_as_text(file_path: Path, chunk_size: int | None = 4092) -> list[Chunk]:
    """Converts arbitrary file types to Markdown via docling's basic pipeline
    (no vision model, no OCR) and returns a list of chunks containing the
    Markdown text. This is a fallback for text-only models that can't handle
    image inputs, however, the conversion quality may result in lower-quality Q&A pairs.
    """
    logger.warning(f"Loading {file_path} as text via docling; this may result in lower-quality Q&A pairs.")
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as e:
        raise RuntimeError(
            "docling is required for PDF-to-text conversion. Install it via `pip install docling`."
        ) from e

    try:
        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        
        chunks = []
        if chunk_size:
            text = result.document.export_to_markdown()
            chunks = split_markdown_into_paragraph_chunks(text, chunk_size)
            logger.debug(f"Converted {file_path} to {len(chunks)} markdown chunk(s) with chunk size {chunk_size}.")
        else:
            chunks = [Chunk(index=0, content=result.document)]
            logger.debug(f"Converted {file_path} to a single markdown chunk (no chunking).")

    except Exception as e:
        logger.error(f"Failed to convert {file_path} to Markdown: {e}")
        raise RuntimeError(f"Failed to convert {file_path} to Markdown.") from e

    return chunks


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


def _load_markdown_chunks(file_path: Path, metadata: dict[str, str]) -> tuple[list[Chunk], dict[str, str]]:
    """Load a Markdown file with optional front matter and split it into chunks based 
    on page breaks (e.g. <!-- page 9 -->) if present, or as a single chunk otherwise."""
    
    import re
    page_break_pattern = re.compile(r"<!--\s*page\s+(\d+)\s*-->")
    
    # include front matter in the chunk content so that the generation agent can use it to produce metadata-aware questions
    post = frontmatter.load(file_path)

    # Update metadata with front matter if present
    if post.metadata:
        metadata.update(post.metadata)
        
    # split the content into chunks based on page breaks if present
    chunks = []
    if page_break_pattern.search(post.content):
        pages = page_break_pattern.findall(post.content)
        for i, page in enumerate(pages):
            start_index = post.content.find(f"<!-- page {page} -->")
            end_index = post.content.find(f"<!-- page {int(page) + 1} -->", start_index)
            if end_index == -1:
                end_index = len(post.content)
            chunk_content = post.content[start_index+len(f"<!-- page {page} -->"):end_index].strip()
            if chunk_content:  # only add non-empty chunks
                chunks.append(Chunk(index=i, content=chunk_content, pages=(int(page), len(pages))))
    else:
        chunks = [Chunk(index=0, content=post.content, pages=(1, 1))]  # single chunk with the entire content

    return chunks, metadata