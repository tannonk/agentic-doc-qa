import json

from agentic_doc_qa.documents import Chunk
from agentic_doc_qa.pipeline import _build_judge_user_content, _summarize_rejections, save_qa_pairs, select_accepted, select_ranked
from agentic_doc_qa.schemas import JudgedQAPair, QAJudgement, QAPair, QAVerdict, QAVerdictDecision


def _pair(question: str) -> QAPair:
    return QAPair(question=question, answer="A.", question_type="factual", question_level="easy")


def _judgement(*decisions_scores: tuple[QAVerdictDecision, int]) -> QAJudgement:
    return QAJudgement(verdicts=[
        QAVerdict(candidate_index=i, decision=d, score=s, rationale="r")
        for i, (d, s) in enumerate(decisions_scores)
    ])


def test_select_accepted_keeps_only_accepted_candidates():
    """select_accepted must drop rejected candidates and candidates with no verdict at all."""
    candidates = [_pair("Q1"), _pair("Q2"), _pair("Q3")]
    judgement = _judgement(
        (QAVerdictDecision.ACCEPT, 5),
        (QAVerdictDecision.REJECT_TOO_EASY, 2),
        (QAVerdictDecision.ACCEPT, 8),
    )

    accepted = select_accepted(candidates, judgement)

    assert [jp.pair.question for jp in accepted] == ["Q1", "Q3"]


def test_select_accepted_missing_verdict_is_dropped():
    """A candidate with no matching verdict index (judge omitted it) must not be treated as accepted."""
    candidates = [_pair("Q1"), _pair("Q2")]
    judgement = _judgement((QAVerdictDecision.ACCEPT, 5))  # only covers index 0

    accepted = select_accepted(candidates, judgement)

    assert [jp.pair.question for jp in accepted] == ["Q1"]


def test_select_ranked_orders_by_score_descending():
    """select_ranked must sort accepted candidates by judge score, highest first, and exclude rejects."""
    candidates = [_pair("low"), _pair("high"), _pair("rejected")]
    judgement = _judgement(
        (QAVerdictDecision.ACCEPT, 3),
        (QAVerdictDecision.ACCEPT, 9),
        (QAVerdictDecision.REJECT_DUPLICATE, 1),
    )

    ranked = select_ranked(candidates, judgement)

    assert [jp.pair.question for jp in ranked] == ["high", "low"]


def test_summarize_rejections_includes_only_non_accepted_with_rationale():
    """The rejection summary fed back into regeneration must list only rejected candidates, with their reason."""
    candidates = [_pair("kept"), _pair("dropped")]
    judgement = QAJudgement(verdicts=[
        QAVerdict(candidate_index=0, decision=QAVerdictDecision.ACCEPT, score=5, rationale="fine"),
        QAVerdict(candidate_index=1, decision=QAVerdictDecision.REJECT_TOO_EASY, score=2, rationale="too trivial"),
    ])

    summary = _summarize_rejections(candidates, judgement)

    assert "kept" not in summary
    assert '"dropped" was rejected (reject_too_easy): too trivial' in summary


def test_build_judge_user_content_includes_parent_context_for_follow_ups():
    """When judging follow-up candidates, the judge must be shown the parent pair and told to reject non-deepening candidates."""
    chunk = Chunk(index=0, content="Some passage.")
    candidates = [_pair("follow-up Q")]
    parent = _pair("root Q")

    content = _build_judge_user_content(chunk, candidates, parent_pair=parent)

    assert "root Q" in content
    assert "follow-up" in content.lower()


def test_build_judge_user_content_omits_parent_block_by_default():
    """Without a parent_pair, the judge content should read exactly as it did before follow-ups existed."""
    chunk = Chunk(index=0, content="Some passage.")
    candidates = [_pair("Q1")]

    content = _build_judge_user_content(chunk, candidates)

    assert "follow-up" not in content.lower()


def test_save_qa_pairs_writes_numbered_files_with_provenance(tmp_path):
    """Each accepted pair should be written as a numbered JSON file carrying the source metadata plus its chunk's provenance (index and, for PDFs, page range), plus its own assigned id."""
    chunk = Chunk(index=2, content="text", pages=(5, 6))
    verdict = QAVerdict(candidate_index=0, decision=QAVerdictDecision.ACCEPT, score=7, rationale="good")
    judged = JudgedQAPair(pair=_pair("Q1"), verdict=verdict)

    written = save_qa_pairs([(judged, chunk, None)], source_id="doc123", metadata={"title": "Example"}, output_dir_base=tmp_path)

    assert [path for path, _ in written] == [tmp_path / "doc123" / "001.json"]
    path, record_id = written[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metadata"]["id"] == record_id
    assert "parent_id" not in data["metadata"]
    assert {k: v for k, v in data["metadata"].items() if k != "id"} == {"title": "Example", "chunk_index": 2, "pages": [5, 6]}
    assert data["judgement"]["score"] == 7


def test_save_qa_pairs_appends_to_existing_numbering(tmp_path):
    """Calling save_qa_pairs a second time for the same source_id (e.g. another round in a chat session) must continue numbering, not overwrite prior files."""
    chunk = Chunk(index=0, content="text")
    verdict = QAVerdict(candidate_index=0, decision=QAVerdictDecision.ACCEPT, score=5, rationale="ok")
    judged = JudgedQAPair(pair=_pair("Q1"), verdict=verdict)

    save_qa_pairs([(judged, chunk, None)], source_id="doc123", metadata={}, output_dir_base=tmp_path)
    second = save_qa_pairs([(judged, chunk, None)], source_id="doc123", metadata={}, output_dir_base=tmp_path)

    assert [path for path, _ in second] == [tmp_path / "doc123" / "002.json"]


def test_save_qa_pairs_records_parent_id_for_follow_ups(tmp_path):
    """A follow-up pair's saved metadata must link back to its parent's assigned id, so a thread can be reconstructed."""
    chunk = Chunk(index=0, content="text")
    verdict = QAVerdict(candidate_index=0, decision=QAVerdictDecision.ACCEPT, score=5, rationale="ok")
    root = JudgedQAPair(pair=_pair("root question"), verdict=verdict)
    follow_up = JudgedQAPair(pair=_pair("follow-up question"), verdict=verdict)

    [(_, root_id)] = save_qa_pairs([(root, chunk, None)], source_id="doc123", metadata={}, output_dir_base=tmp_path)
    [(follow_up_path, follow_up_id)] = save_qa_pairs(
        [(follow_up, chunk, root_id)], source_id="doc123", metadata={}, output_dir_base=tmp_path
    )

    data = json.loads(follow_up_path.read_text(encoding="utf-8"))
    assert data["metadata"]["parent_id"] == root_id
    assert data["metadata"]["id"] == follow_up_id
    assert follow_up_id != root_id
