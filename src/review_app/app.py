#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""Streamlit app for post-hoc, manual review of generated QA pairs.

Thin UI layer over review_io: all filesystem/decision logic lives there so it
stays unit-testable without a running Streamlit session.

Launch:

    python -m streamlit run src/review_app/app.py -- \
        --input-dir data/examples/outputs \
        --validated-dir data/examples/outputs/validated

(Streamlit forwards everything after `--` to the script.)
"""

import base64
import struct
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
    record_key,
    reject,
    resolve_source,
)
from agentic_doc_qa.schemas import QARecord, ReviewDecision


SOURCE_HEIGHT = 800  # px height of the source panel (PDF page, markdown, text)


def build_arg_parser() -> ArgumentParser:
    ap = ArgumentParser(description="Manually review generated QA pairs against their source document.")
    ap.add_argument("--input-dir", type=Path, required=True, help="Directory of pending {source_id} / NNN.json files.")
    ap.add_argument("--validated-dir", type=Path, default=None, help="Destination for approved and rejected pairs, annotated with the decision. If not specified, defaults to <input-dir>/validated.")
    ap.add_argument("--source-root", type=Path, default=Path("."), help="Base directory for resolving metadata.source_path (default: .).")
    ap.add_argument("--image-scale", type=float, default=2.0, help="PDF render scale (default: 2.0).")
    ap.add_argument("--instructions", type=Path, default=Path(__file__).parent / "instructions.md", help="Markdown file with reviewer guidelines, shown on request (default: instructions.md next to this app).")
    return ap


@st.dialog("Review guidelines", width="large")
def _show_instructions(path: Path) -> None:
    st.markdown(path.read_text(encoding="utf-8"))


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


def _jump_callback() -> None:
    st.session_state.pos = st.session_state["nav-jump"] - 1
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
    key = record_key(path, input_dir)
    question = st.session_state.get(f"q-{key}", record.question)
    answer = st.session_state.get(f"a-{key}", record.answer)
    amended = record.model_copy(update={"question": question, "answer": answer})
    approve(amended, path, input_dir, validated_dir, edited=True)
    st.session_state.editing = False
    _advance_to_next_pending(st.session_state.pos)


def _page_prev_callback(page_key: str) -> None:
    st.session_state[page_key] = max(1, st.session_state[page_key] - 1)


def _page_next_callback(page_key: str, max_page: int) -> None:
    st.session_state[page_key] = min(max_page, st.session_state[page_key] + 1)


def _render_source(record, source_root: Path, key: str, image_scale: float) -> None:
    source_path = resolve_source(record, source_root)
    if not source_path.exists():
        st.error(f"Source file not found: {record.metadata.get('source_path')!r}")
        return

    if source_path.suffix == ".md":
        post = frontmatter.load(source_path)
        body = post.content
        with st.container(height=SOURCE_HEIGHT):
            st.markdown(body)
    elif source_path.suffix == ".txt":
        with st.container(height=SOURCE_HEIGHT):
            st.text(source_path.read_text(encoding="utf-8"))
    elif source_path.suffix == ".pdf":
        pages_png = _render_pdf(str(source_path), image_scale)
        # if record.pages:
            # st.caption(f"Generated from pages {record.pages[0]}–{record.pages[1]} of {len(pages_png)}")
        page_key = f"page-{key}"
        if page_key not in st.session_state:
            st.session_state[page_key] = record.pages[0] if record.pages else 1
        col_prev, col_indicator, col_next, col_zoom = st.columns([1, 1, 1, 3], vertical_alignment="center")
        if len(pages_png) > 1:
            col_prev.button(
                "◀ Prev page", key=f"prevpage-{key}",
                disabled=st.session_state[page_key] <= 1,
                on_click=_page_prev_callback, args=(page_key,),
            )
            col_indicator.markdown(f"Page {st.session_state[page_key]} / {len(pages_png)}")
            col_next.button(
                "Next page ▶", key=f"nextpage-{key}",
                disabled=st.session_state[page_key] >= len(pages_png),
                on_click=_page_next_callback, args=(page_key, len(pages_png)),
            )
        page = st.session_state[page_key]
        zoom = col_zoom.slider("Zoom", 100, 300, 100, step=25, format="Zoom %d%%", label_visibility="collapsed", key="zoom")
        try:
            png = pages_png[page - 1]
        except IndexError:
            st.error(f"Page {page} not found in PDF (only {len(pages_png)} pages) for {source_path.name}.")
            return
        # st.image caps width at the column width, and st.markdown/containers can
        # clip or resize the <img>, so zoom lives in an iframe with its own scroll area.
        # At 100% the whole page fits the frame: width is the smaller of the frame
        # width and (frame height x page aspect ratio). Zoom scales that fit size.
        px_w, px_h = struct.unpack(">II", png[16:24])  # PNG IHDR: width, height
        b64 = base64.b64encode(png).decode()
        st.iframe(
            '<body style="margin:0"><div style="height:100vh;overflow:auto">'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="display:block;width:min({zoom}%,{zoom * px_w / px_h:.2f}vh)">'
            '</div></body>',
            height=SOURCE_HEIGHT,
        )
    else:
        st.error(f"Unsupported source file type: {source_path.suffix}")


def _doc_type(record) -> str:
    """Name of the folder holding the source file (e.g. 'SOP', 'Lecture_Slides')."""
    return Path(str(record.metadata.get("source_path", ""))).parent.name or "unknown"


def _render_qa_panel(record, key: str, input_dir: Path, validated_dir: Path, decision: QARecord | None) -> None:
    if decision is not None and decision.annotation is not None:
        show = st.error if decision.annotation.decision == ReviewDecision.REJECTED else st.success
        show(f"Already {decision.annotation.decision.value} — {decision.annotation.timestamp}")

    if st.session_state.editing:
        with st.form(key=f"edit-{key}"):
            st.text_area("Question", value=record.question, key=f"q-{key}", height=120)
            st.text_area("Answer", value=record.answer, key=f"a-{key}", height=160)
            st.form_submit_button(
                "Save & approve", on_click=_save_and_approve_callback, args=(input_dir, validated_dir),
            )
        st.button("Cancel", key=f"cancel-{key}", on_click=_cancel_edit_callback)
        return

    badges = f"`{_doc_type(record)}` · `{record.question_type}` · `{record.question_level}`"
    # chunk_index = record.metadata.get("chunk_index")
    # if chunk_index is not None:
    #     badges += f" · chunk {chunk_index}"
    st.caption(badges)

    with st.container(border=True):
        st.markdown("**Question**")
        st.markdown(f"{record.question}")
        st.markdown("**Answer**")
        st.markdown(record.answer)

    col_approve, col_edit, col_reject, col_skip = st.columns(4)
    col_approve.button(
        "✅  Approve", key=f"approve-{key}", on_click=_approve_callback, args=(input_dir, validated_dir),
        type="primary", shortcut="A",
    )
    col_edit.button("✏️  Edit", key=f"edit-{key}", on_click=_edit_callback, shortcut="E")
    col_reject.button(
        "❌  Reject", key=f"reject-{key}", on_click=_reject_callback, args=(input_dir, validated_dir),
        shortcut="X",
    )
    col_skip.button("⏭  Skip", key=f"skip-{key}", on_click=_skip_callback, shortcut="S")

    with st.expander("Metadata", expanded=False):
        st.json(record.metadata)


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

    done = st.session_state.total - pending_count
    nav_l, nav_mid, nav_r, nav_help = st.columns([1, 4, 1, 1.3], vertical_alignment="center")
    nav_l.button("◀ Previous", key="nav-prev", disabled=(pos == 0), on_click=_prev_callback, shortcut="P")
    if n > 1:  # st.slider needs min < max
        st.session_state["nav-jump"] = pos + 1  # keep the slider in step with Prev/Next/Skip
        nav_mid.slider(
            "Jump to item", 1, n, format=f"Item %d / {n}", label_visibility="collapsed",
            key="nav-jump", on_change=_jump_callback,
        )
    nav_mid.caption(f"Reviewed {done}/{st.session_state.total} — {path.name}")
    nav_r.button("Next ▶", key="nav-next", disabled=(pos == n - 1), on_click=_next_callback, shortcut="N")
    if args.instructions.exists() and nav_help.button("📖 Guidelines", key="nav-help", shortcut="H"):
        _show_instructions(args.instructions)

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

    key = record_key(path, args.input_dir)
    left, right = st.columns([2, 3], gap="large", vertical_alignment="top")
    with left:
        _render_qa_panel(record, key, args.input_dir, args.validated_dir, decisions[pos])
    with right:
        _render_source(record, args.source_root, key, args.image_scale)


if __name__ == "__main__":
    main()
