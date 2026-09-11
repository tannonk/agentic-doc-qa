#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""Pydantic models and prompt text shared across the doc_qa package."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QAType(str, Enum):
    FACTOID = "factoid"
    DEFINITION = "definition"
    YESNO = "yes-no"
    SUMMARY = "summary"

class QALevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    IMPOSSIBLE = "impossible"

class QAPair(BaseModel):
    question: str
    answer: str
    question_type: QAType
    question_level: QALevel

class QAPairs(BaseModel):
    pairs: list[QAPair] = Field(default_factory=list)

class ReviewDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"

class ReviewAnnotation(BaseModel):
    """Stamped onto a QARecord when a reviewer decides it, so the decision
    travels with the record itself instead of living in a separate log.

    original_question/original_answer are only set for EDITED records (the
    pre-edit text); for APPROVED/REJECTED, `question`/`answer` on the record
    itself already are the original text, so duplicating it would be redundant.
    """
    decision: ReviewDecision
    source_key: str = Field(description="Path of the original generated file, relative to the review input dir, e.g. 'dc7bbd0e82de/001.json'.")
    timestamp: str = Field(description="ISO8601 UTC timestamp of the decision.")
    original_question: str | None = None
    original_answer: str | None = None

class QARecord(QAPair):
    """A QA pair as persisted on disk by generate.write_qa_set.

    All source-document and provenance information lives in the open `metadata`
    dict. `metadata["chunk_index"]` (int) and `metadata["pages"]`
    ([first, last], 1-based inclusive, PDFs only) are absent in files generated
    before provenance was recorded — consumers must treat them as optional.

    `annotation` is absent until a reviewer approves, rejects, or edits the
    record. Fields this model doesn't know about (e.g. `judgement`, written by
    the generation pipeline) are kept as-is (extra="allow") so they survive a
    load/write round-trip through the review app rather than being silently
    dropped, preserving full traceability of a record's history.

    Building the final QA test suite: keep records where
    `annotation.decision in (ReviewDecision.APPROVED, ReviewDecision.EDITED)`.
    """
    model_config = ConfigDict(extra="allow")

    metadata: dict[str, Any] = Field(default_factory=dict)
    annotation: ReviewAnnotation | None = None

    @property
    def pages(self) -> tuple[int, int] | None:
        """1-based inclusive (first, last) source page range, if recorded."""
        pages = self.metadata.get("pages")
        return (int(pages[0]), int(pages[1])) if pages else None

class QAVerdictDecision(str, Enum):
    ACCEPT = "accept"
    REJECT_DUPLICATE = "reject_duplicate"
    REJECT_TOO_EASY = "reject_too_easy"
    REJECT_OTHER = "reject_other"

class QAVerdict(BaseModel):
    candidate_index: int
    decision: QAVerdictDecision
    score: int = Field(ge=1, le=10, description="Value/challenge score; used to rank accepted candidates against each other.")
    rationale: str

class QAJudgement(BaseModel):
    verdicts: list[QAVerdict]

class JudgedQAPair(BaseModel):
    """A QAPair paired with the judge verdict that accepted it -- lets
    callers audit post-hoc why a given pair was kept. Never used as an LLM
    output_type: it's assembled from a QAPair and QAVerdict after judging,
    not produced directly by a model."""
    pair: QAPair
    verdict: QAVerdict