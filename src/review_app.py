#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""Streamlit app for manual review of generated QA pairs.

Thin UI layer over review_io: all filesystem/decision logic lives there so it
stays unit-testable without a running Streamlit session.

Launch:

    uv run --group review streamlit run src/doc_qa/review_app.py -- \\
        --input-dir data/local/swissdox/outputs \\
        --validated-dir data/local/swissdox/validated

(Streamlit forwards everything after `--` to the script.)
"""

from argparse import ArgumentParser
from pathlib import Path

import frontmatter
import streamlit as st
from pydantic import ValidationError

from agentic_doc_qa.documents import render_pdf_pages_png
from agentic_doc_qa.review_io import (
    approve,
    decision_for,
    list_all_source_records,
    load_record,
    reject,
    resolve_source,
)
from agentic_doc_qa.schemas import QARecord, ReviewDecision


def build_arg_parser() -> ArgumentParser:
    ap = ArgumentParser(description="Manually review generated QA pairs against their source document.")
    ap.add_argument("--input-dir", type=Path, required=True, help="Directory of pending {source_id} / NNN.json files.")
    ap.add_argument("--validated-dir", type=Path, default=None, help="Destination for approved and rejected pairs, annotated with the decision. If not specified, defaults to <input-dir>/validated.")
    ap.add_argument("--source-root", type=Path, default=Path("."), help="Base directory for resolving metadata.source_path (default: .).")
    ap.add_argument("--image-scale", type=float, default=2.0, help="PDF render scale (default: 2.0).")
    return ap


@st.cache_data(show_spinner="Rendering PDF...")
def _render_pdf(path_str: str, scale: float) -> list[bytes]:
    return render_pdf_pages_png(Path(path_str), image_scale=scale)


def _advance_to_next_pending(start: int) -> None:
    """Move st.session_state.pos to the next pending item, searching forward
    from `start` and wrapping. If no pending item exists anywhere, pos is
    left unchanged (used both for skip and for post-decision auto-advance)."""
    paths = st.session_state.paths
    input_dir = st.session_state.input_dir
    validated_dir = st.session_state.validated_dir
    n = len(paths)
    if n == 0:
        return
    for offset in range(1, n + 1):
        i = (start + offset) % n
        if decision_for(paths[i], input_dir, validated_dir) is None:
            st.session_state.pos = i
            return
    # nothing pending anywhere — stay put


def _first_pending_index(paths: list[Path], input_dir: Path, validated_dir: Path) -> int:
    """Index of the first item in `paths` with no recorded decision, in order,
    inclusive of index 0. Falls back to 0 if every item is already decided
    (e.g. all done — main() still needs a valid position to render)."""
    for i, path in enumerate(paths):
        if decision_for(path, input_dir, validated_dir) is None:
            return i
    return 0


def _skip_callback() -> None:
    st.session_state.editing = False
    _advance_to_next_pending(st.session_state.pos)


def _prev_callback() -> None:
    st.session_state.pos = max(0, st.session_state.pos - 1)
    st.session_state.editing = False


def _next_callback() -> None:
    n = len(st.session_state.paths)
    st.session_state.pos = min(n - 1, st.session_state.pos + 1)
    st.session_state.editing = False


def _edit_callback() -> None:
    st.session_state.editing = True


def _cancel_edit_callback() -> None:
    st.session_state.editing = False


def _approve_callback(input_dir: Path, validated_dir: Path) -> None:
    path = st.session_state.paths[st.session_state.pos]
    try:
        record = load_record(path)
        approve(record, path, input_dir, validated_dir)
    except (FileNotFoundError, ValidationError):
        pass  # nothing to approve; still advance past it below
    st.session_state.editing = False
    _advance_to_next_pending(st.session_state.pos)


def _reject_callback(input_dir: Path, validated_dir: Path) -> None:
    path = st.session_state.paths[st.session_state.pos]
    reject(path, input_dir, validated_dir)
    st.session_state.editing = False
    _advance_to_next_pending(st.session_state.pos)


def _save_and_approve_callback(input_dir: Path, validated_dir: Path) -> None:
    path = st.session_state.paths[st.session_state.pos]
    try:
        record = load_record(path)
    except (FileNotFoundError, ValidationError):
        st.session_state.editing = False
        _advance_to_next_pending(st.session_state.pos)
        return
    question = st.session_state.get(f"q-{path.name}", record.question)
    answer = st.session_state.get(f"a-{path.name}", record.answer)
    amended = record.model_copy(update={"question": question, "answer": answer})
    approve(amended, path, input_dir, validated_dir, edited=True)
    st.session_state.editing = False
    _advance_to_next_pending(st.session_state.pos)


def _page_prev_callback(page_key: str) -> None:
    st.session_state[page_key] = max(1, st.session_state[page_key] - 1)


def _page_next_callback(page_key: str, max_page: int) -> None:
    st.session_state[page_key] = min(max_page, st.session_state[page_key] + 1)


def _render_source(record, source_root: Path, path_name: str, image_scale: float) -> None:
    
    source_path = resolve_source(record, source_root)
    if not source_path.exists():
        st.error(f"Source file not found: {record.metadata.get('source_path')!r}")
        return

    if source_path.suffix == ".md":
        post = frontmatter.load(source_path)
        body = post.content
        with st.container(height=700):
            st.markdown(body)
    elif source_path.suffix == ".txt":
        with st.container(height=700):
            st.text(source_path.read_text(encoding="utf-8"))
    elif source_path.suffix == ".pdf":
        pages_png = _render_pdf(str(source_path), image_scale)
        if record.pages:
            st.info(f"Generated from pages {record.pages[0]}–{record.pages[1]} of {len(pages_png)}")
        page_key = f"page-{path_name}"
        if page_key not in st.session_state:
            st.session_state[page_key] = record.pages[0] if record.pages else 1
        if len(pages_png) > 1:
            col_prev, col_indicator, col_next = st.columns([1, 2, 1])
            col_prev.button(
                "◀ Prev page", key=f"prevpage-{path_name}",
                disabled=st.session_state[page_key] <= 1,
                on_click=_page_prev_callback, args=(page_key,),
            )
            col_indicator.markdown(f"Page {st.session_state[page_key]} / {len(pages_png)}")
            col_next.button(
                "Next page ▶", key=f"nextpage-{path_name}",
                disabled=st.session_state[page_key] >= len(pages_png),
                on_click=_page_next_callback, args=(page_key, len(pages_png)),
            )
        page = st.session_state[page_key]
        st.image(pages_png[page - 1], width="stretch")
    else:
        st.error(f"Unsupported source file type: {source_path.suffix}")


def _render_qa_panel(record, path, input_dir: Path, validated_dir: Path, decision: QARecord | None) -> None:
    if decision is not None and decision.annotation is not None:
        st.caption(f"Already {decision.annotation.decision.value} — {decision.annotation.timestamp}")

    if st.session_state.editing:
        with st.form(key=f"edit-{path.name}"):
            st.text_area("Question", value=record.question, key=f"q-{path.name}", height=120)
            st.text_area("Answer", value=record.answer, key=f"a-{path.name}", height=160)
            st.form_submit_button(
                "Save & approve", on_click=_save_and_approve_callback, args=(input_dir, validated_dir),
            )
        st.button("Cancel", key=f"cancel-{path.name}", on_click=_cancel_edit_callback)
        return

    st.markdown("##### Question")
    st.markdown(record.question)
    st.markdown("##### Answer")
    st.markdown(record.answer)

    badges = f"`{record.question_type.value}` · `{record.question_level.value}`"
    chunk_index = record.metadata.get("chunk_index")
    if chunk_index is not None:
        badges += f" · chunk {chunk_index}"
    st.caption(badges)

    with st.expander("Metadata", expanded=True):
        st.json(record.metadata)

    col_approve, col_edit, col_reject, col_skip = st.columns(4)
    col_approve.button(
        "✅ Approve", key=f"approve-{path.name}", on_click=_approve_callback, args=(input_dir, validated_dir),
    )
    col_edit.button("✏️ Edit", key=f"edit-{path.name}", on_click=_edit_callback)
    col_reject.button(
        "❌ Reject", key=f"reject-{path.name}", on_click=_reject_callback, args=(input_dir, validated_dir),
    )
    col_skip.button("⏭ Skip", key=f"skip-{path.name}", on_click=_skip_callback)


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.validated_dir is None:
        args.validated_dir = args.input_dir / "validated"

    st.set_page_config(page_title="QA Review", layout="wide")

    if "paths" not in st.session_state:
        st.session_state.input_dir = args.input_dir
        st.session_state.validated_dir = args.validated_dir
        st.session_state.paths = list_all_source_records(args.input_dir, args.validated_dir)
        st.session_state.total = len(st.session_state.paths)
        st.session_state.pos = _first_pending_index(st.session_state.paths, args.input_dir, args.validated_dir)
        st.session_state.editing = False

    paths = st.session_state.paths
    n = len(paths)

    if n == 0:
        st.info("No QA files found in the input directory.")
        return

    decisions = [decision_for(p, args.input_dir, args.validated_dir) for p in paths]
    pending_count = sum(1 for d in decisions if d is None)

    if pending_count == 0:
        decided = [d.annotation for d in decisions if d is not None and d.annotation is not None]
        approved = sum(1 for a in decided if a.decision == ReviewDecision.APPROVED)
        edited = sum(1 for a in decided if a.decision == ReviewDecision.EDITED)
        rejected = sum(1 for a in decided if a.decision == ReviewDecision.REJECTED)
        st.success(f"All done! Approved: {approved}, Edited: {edited}, Rejected: {rejected}.")

    pos = st.session_state.pos
    path = paths[pos]

    nav_l, _, nav_r = st.columns([1, 3, 1])
    nav_l.button("◀ Previous", key="nav-prev", disabled=(pos == 0), on_click=_prev_callback)
    nav_r.button("Next ▶", key="nav-next", disabled=(pos == n - 1), on_click=_next_callback)

    if not path.exists():
        st.error(f"File no longer exists: {path.name}")
        st.button("Skip", key="skip-missing", on_click=_skip_callback)
        return

    try:
        record = load_record(path)
    except ValidationError as e:
        st.error(f"Invalid QA record {path.name}: {e}")
        st.button("Skip", key="skip-invalid", on_click=_skip_callback)
        return

    done = st.session_state.total - pending_count
    st.caption(f"Reviewing {done}/{st.session_state.total} — item {pos + 1}/{n} — {path.name}")

    left, right = st.columns([2, 3])
    with left:
        _render_qa_panel(record, path, args.input_dir, args.validated_dir, decisions[pos])
    with right:
        _render_source(record, args.source_root, path.name, args.image_scale)


if __name__ == "__main__":
    main()
