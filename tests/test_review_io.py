import json
from pathlib import Path

from agentic_doc_qa.review_io import (
    approve,
    list_all_source_records,
    list_pending,
    load_record,
    reject,
    record_key,
)
from agentic_doc_qa.schemas import QARecord, ReviewDecision


def _write_source(input_dir: Path, rel_path: str, **overrides) -> Path:
    data = {
        "question": "Q?",
        "answer": "A.",
        "question_type": "factual",
        "question_level": "easy",
        "metadata": {},
        **overrides,
    }
    path = input_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_record_key_is_posix_relative_path(tmp_path):
    """record_key must be the source-relative path in POSIX form, since it's reused as both the validated_dir destination and the stamped source_key."""
    input_dir = tmp_path / "generated"
    path = input_dir / "abc123" / "001.json"
    assert record_key(path, input_dir) == "abc123/001.json"


def test_list_all_source_records_excludes_validated_dir(tmp_path):
    """Records already written under validated_dir (nested inside input_dir) must not be treated as source records again."""
    input_dir = tmp_path / "generated"
    validated_dir = input_dir / "validated"
    _write_source(input_dir, "abc/001.json")
    _write_source(validated_dir, "abc/001.json")

    records = list_all_source_records(input_dir, validated_dir)

    assert records == [input_dir / "abc" / "001.json"]


def test_list_pending_excludes_decided_records(tmp_path):
    """A record with a matching file already in validated_dir is decided and must be excluded from the pending list."""
    input_dir = tmp_path / "generated"
    validated_dir = tmp_path / "validated"
    source = _write_source(input_dir, "abc/001.json")
    _write_source(input_dir, "abc/002.json")

    approve(load_record(source), source, input_dir, validated_dir)

    pending = list_pending(input_dir, validated_dir)

    assert pending == [input_dir / "abc" / "002.json"]


def test_approve_writes_approved_annotation(tmp_path):
    """Approving a record should stamp it APPROVED and leave the original source file untouched."""
    input_dir = tmp_path / "generated"
    validated_dir = tmp_path / "validated"
    source = _write_source(input_dir, "abc/001.json")

    dest = approve(load_record(source), source, input_dir, validated_dir)

    written = load_record(dest)
    assert written.annotation.decision == ReviewDecision.APPROVED
    assert written.annotation.source_key == "abc/001.json"
    assert written.annotation.original_question is None
    assert json.loads(source.read_text())["question"] == "Q?"  # original untouched


def test_approve_edited_records_original_text_in_annotation(tmp_path):
    """Approving an edited record must keep the pre-edit question/answer in the annotation so the edit stays auditable."""
    input_dir = tmp_path / "generated"
    validated_dir = tmp_path / "validated"
    source = _write_source(input_dir, "abc/001.json")

    original = load_record(source)
    edited = original.model_copy(update={"question": "Edited Q?", "answer": "Edited A."})
    dest = approve(edited, source, input_dir, validated_dir, edited=True)

    written = load_record(dest)
    assert written.annotation.decision == ReviewDecision.EDITED
    assert written.question == "Edited Q?"
    assert written.annotation.original_question == "Q?"
    assert written.annotation.original_answer == "A."


def test_reject_writes_rejected_annotation_with_original_content(tmp_path):
    """Rejecting a record loads the original from disk and stamps it REJECTED, without needing the caller to pass it in."""
    input_dir = tmp_path / "generated"
    validated_dir = tmp_path / "validated"
    source = _write_source(input_dir, "abc/001.json")

    dest = reject(source, input_dir, validated_dir)

    written = load_record(dest)
    assert written.annotation.decision == ReviewDecision.REJECTED
    assert written.question == "Q?"


def test_reversing_decision_overwrites_same_file(tmp_path):
    """Re-deciding a record (reject after approve) must overwrite the same validated_dir file, not create a second one."""
    input_dir = tmp_path / "generated"
    validated_dir = tmp_path / "validated"
    source = _write_source(input_dir, "abc/001.json")

    dest_approved = approve(load_record(source), source, input_dir, validated_dir)
    dest_rejected = reject(source, input_dir, validated_dir)

    assert dest_approved == dest_rejected
    assert load_record(dest_rejected).annotation.decision == ReviewDecision.REJECTED
