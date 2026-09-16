import pytest
from pydantic import ValidationError

from agentic_doc_qa.schemas import (
    QAJudgement,
    QAPair,
    QARecord,
    QAVerdict,
    QAVerdictDecision,
    ReviewAnnotation,
    ReviewDecision,
)


def test_qarecord_allows_extra_fields_roundtrip():
    """QARecord must keep unknown fields (e.g. 'judgement') on load, since review_io round-trips generated files verbatim."""
    raw = {
        "question": "Q?",
        "answer": "A.",
        "question_type": "factual",
        "question_level": "easy",
        "judgement": {"decision": "accept", "score": 8, "rationale": "solid"},
    }
    record = QARecord.model_validate(raw)
    assert record.model_dump()["judgement"] == raw["judgement"]


def test_qarecord_pages_property_absent():
    """pages should be None when metadata has no 'pages' key (e.g. non-PDF sources or older generated files)."""
    record = QARecord(question="Q?", answer="A.", question_type="factual", question_level="easy")
    assert record.pages is None


def test_qarecord_pages_property_present():
    """pages should convert the stored [first, last] list into a (first, last) int tuple."""
    record = QARecord(
        question="Q?", answer="A.", question_type="factual", question_level="easy",
        metadata={"pages": [3, 5]},
    )
    assert record.pages == (3, 5)


def test_qarecord_annotation_defaults_to_none():
    """A freshly generated (not yet reviewed) record must have no annotation."""
    record = QARecord(question="Q?", answer="A.", question_type="factual", question_level="easy")
    assert record.annotation is None


def test_qaverdict_score_out_of_range_rejected():
    """Score is a 1-10 rating; values outside that range must fail validation so bad judge output is caught early."""
    with pytest.raises(ValidationError):
        QAVerdict(candidate_index=0, decision=QAVerdictDecision.ACCEPT, score=11, rationale="too high")


def test_qajudgement_holds_multiple_verdicts():
    """QAJudgement should preserve the list of per-candidate verdicts from the judge agent's output."""
    judgement = QAJudgement(verdicts=[
        QAVerdict(candidate_index=0, decision=QAVerdictDecision.ACCEPT, score=7, rationale="ok"),
        QAVerdict(candidate_index=1, decision=QAVerdictDecision.REJECT_DUPLICATE, score=2, rationale="dup"),
    ])
    assert [v.decision for v in judgement.verdicts] == [
        QAVerdictDecision.ACCEPT, QAVerdictDecision.REJECT_DUPLICATE,
    ]
