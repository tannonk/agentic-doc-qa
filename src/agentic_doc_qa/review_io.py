#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""IO layer for the manual QA-pair review app.

Contract:
- The input directory is IMMUTABLE: originals are never moved, edited, or deleted.
- Every decided record is written to `validated_dir`, at the same path relative
  to `input_dir` as the original (e.g. `input_dir/dc7bbd0e82de/001.json` ->
  `validated_dir/dc7bbd0e82de/001.json`), with a `ReviewAnnotation` stamped onto
  it recording the decision (approved / rejected / edited) and when.
- A record's decision state is just "does a file exist at that path in
  validated_dir" — there is no separate log to keep in sync. Reversing a
  decision (approve after reject, or vice versa) simply overwrites that same
  file with a new annotation.
- Records are written with the exact serialization convention of
  generate.write_qa_set (indent=2, ensure_ascii=False, sort_keys=True), and
  QARecord allows extra fields, so anything the original generated file
  carried (e.g. `judgement`) survives the round-trip untouched.
- Building the final QA test suite is: glob validated_dir, load each record,
  keep the ones where `annotation.decision` is APPROVED or EDITED.

No streamlit imports here: everything in this module is plain filesystem +
pydantic logic.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from agentic_doc_qa.schemas import QARecord, ReviewAnnotation, ReviewDecision


def record_key(path: Path, input_dir: Path) -> str:
    """Canonical identity for a QA record: its path relative to input_dir,
    POSIX-separated. Used both as the destination path under validated_dir
    and as the value stamped into annotation.source_key."""
    return path.relative_to(input_dir).as_posix()


def list_all_source_records(input_dir: Path, validated_dir: Path) -> list[Path]:
    """All generated QA JSON files in input_dir, excluding anything under
    validated_dir (relevant when validated_dir is nested inside input_dir,
    e.g. the default `<input_dir>/validated`), sorted by path."""
    validated_resolved = validated_dir.resolve()
    return [
        p for p in sorted(input_dir.rglob("*.json"))
        if validated_resolved not in p.resolve().parents
    ]


def load_record(path: Path) -> QARecord:
    return QARecord.model_validate_json(path.read_text(encoding="utf-8"))


def decision_for(path: Path, input_dir: Path, validated_dir: Path) -> QARecord | None:
    """The validated record for `path`, if a review decision has been made,
    else None (still pending)."""
    dest = validated_dir / record_key(path, input_dir)
    if not dest.exists():
        return None
    return load_record(dest)


def list_pending(input_dir: Path, validated_dir: Path) -> list[Path]:
    """All QA JSON files in input_dir with no recorded decision, sorted by path."""
    return [
        p for p in list_all_source_records(input_dir, validated_dir)
        if decision_for(p, input_dir, validated_dir) is None
    ]


def resolve_source(record: QARecord, source_root: Path) -> Path:
    """metadata['source_path'] is stored repo-root-relative; resolve it against
    --source-root so the app works from any working directory."""
    return source_root / str(record.metadata.get("source_path", ""))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump_record(record: QARecord) -> str:
    """Serialize exactly like generate.write_qa_set (keys sorted, non-ASCII raw)."""
    data = record.model_dump(mode="json")
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)


def _write_decision(
    record: QARecord,
    source_path: Path,
    input_dir: Path,
    validated_dir: Path,
    decision: ReviewDecision,
) -> Path:
    key = record_key(source_path, input_dir)
    dest = validated_dir / key
    dest.parent.mkdir(parents=True, exist_ok=True)

    original_question = original_answer = None
    if decision == ReviewDecision.EDITED:
        original = load_record(source_path)
        original_question, original_answer = original.question, original.answer

    annotated = record.model_copy(update={
        "annotation": ReviewAnnotation(
            decision=decision, source_key=key, timestamp=_now_iso(),
            original_question=original_question, original_answer=original_answer,
        ),
    })
    logger.debug(f"Writing {decision.value} record to {dest}")
    dest.write_text(_dump_record(annotated), encoding="utf-8")
    return dest


def approve(
    record: QARecord,
    source_path: Path,
    input_dir: Path,
    validated_dir: Path,
    *,
    edited: bool = False,
) -> Path:
    """Write `record` (amended in place by the caller if edited) to
    validated_dir with an "approved" or "edited" annotation. The original file
    is never touched, so its pre-edit text stays recoverable there (and is
    also copied into the annotation when edited, for traceability)."""
    decision = ReviewDecision.EDITED if edited else ReviewDecision.APPROVED
    return _write_decision(record, source_path, input_dir, validated_dir, decision)


def reject(source_path: Path, input_dir: Path, validated_dir: Path) -> Path:
    """Write the (unedited) original record to validated_dir with a
    "rejected" annotation."""
    record = load_record(source_path)
    return _write_decision(record, source_path, input_dir, validated_dir, ReviewDecision.REJECTED)
