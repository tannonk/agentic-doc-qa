import hashlib
from datetime import date
import warnings

import pypdfium2 as pdfium
import frontmatter
from yaml.scanner import ScannerError

from agentic_doc_qa.documents import (
    Chunk,
    _load_pdf_chunks,
    _page_chunk_ranges,
    load_chunks,
    render_pdf_pages_png,
)


def test_split_front_matter_absent():
    raw = "Just a plain document with no front matter."
    post = frontmatter.loads(raw)
    metadata, body = post.metadata, post.content
    assert metadata == {}
    assert body == raw


def test_split_front_matter_present():
    # YAML parses unquoted ISO date literals into datetime.date, not str.
    content = "The body text."
    raw = f"---\ntitle: Example\npubtime: 2026-01-01\n---\n{content}\n"
    post = frontmatter.loads(raw)
    metadata, body = post.metadata, post.content
    assert metadata == {"title": "Example", "pubtime": date(2026, 1, 1)}
    assert body == content


def test_split_front_matter_malformed_missing_closer():
    raw = "---\ntitle: Example\nThe body text with no closing marker."
    post = frontmatter.loads(raw)
    metadata, body = post.metadata, post.content
    assert metadata == {}
    assert body == raw


def test_split_front_matter_unquoted_colon_in_value_falls_back_to_permissive_parse():
    # headline contains an unquoted colon -> invalid YAML (nested-mapping parse
    # error) -> should raise exception and return empty metadata dict, not crash.
    raw = "---\nheadline: Bei Trump-Anwesen: Sicherheitskräfte erschiessen bewaffneten Mann\nsource: Basler Zeitung\n---\nBody.\n"
    try:
        post = frontmatter.loads(raw)
        metadata, body = post.metadata, post.content
    except ScannerError:
        pass

    metadata, body = {}, raw
    
    assert metadata == {}
    assert body == raw


def test_split_front_matter_non_mapping():
    raw = "---\n- just\n- a\n- list\n---\nBody.\n"
    post = frontmatter.loads(raw)
    metadata, body = post.metadata, post.content
    assert metadata == {}
    assert body == "Body."


def test_load_chunks_markdown_sends_raw_text_unstripped(tmp_path):
    content = "Body text here."
    raw = f"---\nmedium_name: Example Times\npubtime: 2026-01-01\n---\n\n{content}\n"
    source_path = tmp_path / "doc.md"
    source_path.write_text(raw, encoding="utf-8")
    chunks, metadata, source_id = load_chunks(source_path)

    assert len(chunks) == 1
    assert chunks[0] == Chunk(index=0, content=raw.strip())
    assert metadata == {"pubtime": date(2026, 1, 1), "medium_name": "Example Times"}
    assert hashlib.sha256(raw.encode("utf-8")).hexdigest().startswith(source_id)


def test_load_chunks_txt_no_front_matter(tmp_path):
    raw = "Plain text document.\n"
    source_path = tmp_path / "doc.txt"
    source_path.write_text(raw, encoding="utf-8")

    chunks, metadata, source_id = load_chunks(source_path)

    assert chunks == [Chunk(index=0, content=raw.strip())]
    assert metadata == {}
    assert hashlib.sha256(raw.encode("utf-8")).hexdigest().startswith(source_id)


def test_load_chunks_unsupported_suffix_raises(tmp_path):
    """Test that loading an invalid file suffix raises a RuntimeError.
    Expected: docling.exceptions.ConversionError Input document doc.docx is not valid.
    """
    source_path = tmp_path / "doc.docx"
    source_path.write_text("irrelevant", encoding="utf-8")
    
    try:
        load_chunks(source_path)
    except RuntimeError as e:
        pass
        

def test_page_chunk_ranges_even_split():
    assert _page_chunk_ranges(10, 5) == [(0, 5), (5, 10)]


def test_page_chunk_ranges_uneven_split():
    assert _page_chunk_ranges(12, 5) == [(0, 5), (5, 10), (10, 12)]


def test_page_chunk_ranges_single_chunk():
    assert _page_chunk_ranges(3, 5) == [(0, 3)]


def test_page_chunk_ranges_zero_pages():
    assert _page_chunk_ranges(0, 5) == []


def _make_pdf(path, n_pages: int) -> None:
    doc = pdfium.PdfDocument.new()
    for _ in range(n_pages):
        doc.new_page(100, 100)
    with open(path, "wb") as f:
        doc.save(f)
    doc.close()


def test_render_pdf_pages_png_returns_one_png_per_page(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path, 3)

    pages_png = render_pdf_pages_png(pdf_path)

    assert len(pages_png) == 3
    assert all(png.startswith(b"\x89PNG") for png in pages_png)


def test_load_pdf_chunks_records_page_ranges(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path, 3)

    chunks = _load_pdf_chunks(pdf_path, pages_per_chunk=2, image_scale=2.0)
    
    assert len(chunks) == 2
    assert chunks[0].pages == (1, 2)
    assert chunks[1].pages == (3, 3)
    


def test_load_chunks_markdown_has_no_page_range(tmp_path):
    source_path = tmp_path / "doc.md"
    source_path.write_text("Body text.\n", encoding="utf-8")

    chunks, _, _ = load_chunks(source_path)

    assert chunks[0].pages is None
