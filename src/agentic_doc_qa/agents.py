#!/usr/bin/env python3
#-*- coding: utf-8 -*-


from pathlib import Path
from dataclasses import dataclass, field, replace

from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic_ai.models import Model

from agentic_doc_qa.documents import Chunk
from agentic_doc_qa.domains import DomainConfig
from agentic_doc_qa.schemas import JudgedQAPair, QAJudgement, QAPairs


@dataclass
class GenerationDeps:
    chunk: Chunk
    n_candidates: int
    feedback: str | None = None
    reviewer_feedback: str | None = None
    existing_questions: list[str] = field(default_factory=list)


def build_judge_agent(model: Model, domain_cfg: DomainConfig) -> Agent:
    return Agent(
        model,
        name="qa_judge_agent",
        instructions=domain_cfg.judge_instructions,
        model_settings=domain_cfg.judge_settings,
        output_type=QAJudgement,
    )


def build_generation_agent(model: Model, domain_cfg: DomainConfig, n_candidates: int = 4) -> Agent:
    generation_agent = Agent(
        model,
        name="qa_generation_agent",
        deps_type=GenerationDeps,
        instructions=domain_cfg.generation_instructions.format(n_candidates=n_candidates),
        model_settings=domain_cfg.generation_settings,
        output_type=QAPairs,
    )

    @generation_agent.instructions
    def add_vision_addendum(ctx: RunContext[GenerationDeps]) -> str:
        if ctx.deps.chunk.pages is not None:  # PDF-sourced chunk
            return domain_cfg.generation_vision_addendum
        return ""

    @generation_agent.instructions
    def add_feedback(ctx: RunContext[GenerationDeps]) -> str:
        if ctx.deps.feedback is None:
            return ""
        return (
            f"A reviewer/judge rejected your previous attempt for this reason:\n"
            f"{ctx.deps.feedback}\n"
            f"Produce an improved set that addresses this feedback."
        )

    @generation_agent.instructions
    def add_reviewer_feedback(ctx: RunContext[GenerationDeps]) -> str:
        if ctx.deps.reviewer_feedback is None:
            return ""
        return (
            f"A human reviewer gave the following feedback on an earlier batch for this "
            f"chunk:\n{ctx.deps.reviewer_feedback}\n"
            f"Produce a new batch that addresses this feedback."
        )

    @generation_agent.instructions
    def add_existing_questions(ctx: RunContext[GenerationDeps]) -> str:
        if not ctx.deps.existing_questions:
            return ""
        listing = "\n".join(f"- {q}" for q in ctx.deps.existing_questions)
        return (
            f"These questions were already accepted for this chunk in an earlier attempt. "
            f"Do NOT repeat or closely paraphrase them; cover different content instead:\n{listing}"
        )

    return generation_agent


CHAT_COORDINATOR_INSTRUCTIONS = """You help a human reviewer work through a source document's chunks one at a
time, proposing AI-generated QA pairs and saving only the ones the reviewer wants to keep.

Workflow:
- Chunks are numbered 0..N-1 (use `document_info` if you need the count). Work through them in order.
- Before showing the reviewer anything for a chunk, call `propose` for that chunk_index. Never claim a
  candidate is ready to save without calling `propose` first -- every candidate you show has already been
  judge-vetted, and `propose` is the only way that happens.
- Show the proposed pairs to the reviewer as a numbered list (question, answer, type, difficulty level), then
  wait for their response. Do not decide anything on their behalf.
- If the reviewer says which pairs to keep (by number), call `save` with exactly those pair indexes for the
  current chunk_index.
- If the reviewer gives quality feedback instead (too easy, wrong, wants a different focus, wants it harder,
  etc.), call `propose` again for the SAME chunk_index, passing their feedback verbatim as `reviewer_feedback`.
  Never edit, rewrite, or fabricate a corrected pair yourself -- only a fresh `propose` call is judge-vetted.
- If the reviewer says to move on / the chunk is done, call `propose` for chunk_index + 1.
- Once every chunk has been covered, tell the reviewer the document is fully reviewed.
"""


def build_doc_qa_agent(
    model, source: Path, domain_cfg: DomainConfig, approved_dir: Path, default_n_candidates: int = 4,
) -> Agent:
    """Tools close over document state (chunks/metadata/domain) loaded once
    when the chat/web session starts, rather than using deps_type -- to_cli_sync()
    and to_web() drive their own run loop, so per-session state lives in the
    closure, not in a deps object threaded through run() calls you don't control."""
    from datetime import datetime, timezone

    from agentic_doc_qa.documents import load_chunks
    from agentic_doc_qa.pipeline import propose_qa_pairs, save_qa_pairs
    from agentic_doc_qa import review_io

    chunks, metadata, source_id = load_chunks(source)

    log_path = approved_dir / "review_log.jsonl"

    doc_qa_agent = Agent(model, name="doc_qa_agent", instructions=CHAT_COORDINATOR_INSTRUCTIONS)
    last_proposals: dict[int, list[JudgedQAPair]] = {}  # chunk_index -> judge-accepted JudgedQAPair list, most recent proposal only
    chunk_seen_questions: dict[int, set[str]] = {}  # chunk_index -> every question text shown so far, across all propose() rounds
    saved_counts: dict[int, int] = {}  # chunk_index -> number of pairs saved so far

    @doc_qa_agent.tool_plain
    def document_info() -> str:
        progress = ", ".join(f"chunk {i}: {n} saved" for i, n in sorted(saved_counts.items())) or "none saved yet"
        return f"{len(chunks)} chunk(s), domain={domain_cfg.domain!r}, source={source.name}. Progress: {progress}."

    @doc_qa_agent.tool_plain
    async def propose(
        chunk_index: int, n_candidates: int = default_n_candidates, reviewer_feedback: str | None = None
    ) -> list[dict] | str:
        """Generate and judge-gate candidates for one chunk. Returns only
        judge-accepted pairs -- this is the ONLY way candidates enter the
        conversation, so the invariant holds by construction. Pass
        reviewer_feedback when the reviewer asked for a different/better
        batch of the same chunk."""
        if not (0 <= chunk_index < len(chunks)):
            return f"There is no chunk {chunk_index}. This document has {len(chunks)} chunk(s) (indexes 0-{len(chunks) - 1})."

        avoid_questions = sorted(chunk_seen_questions.get(chunk_index, set()))
        accepted = await propose_qa_pairs(
            model, domain_cfg, chunks[chunk_index], n_candidates,
            reviewer_feedback=reviewer_feedback,
            avoid_questions=avoid_questions,
        )
        last_proposals[chunk_index] = accepted
        chunk_seen_questions.setdefault(chunk_index, set()).update(jp.pair.question for jp in accepted)
        return [jp.model_dump() for jp in accepted]

    @doc_qa_agent.tool_plain
    def save(chunk_index: int, pair_indexes: list[int]) -> str:
        """Write the reviewer's chosen subset of the last proposal for this
        chunk. Only call this in direct response to the reviewer naming which
        pairs to keep -- that message is the actual approval gate."""
        pairs = [last_proposals[chunk_index][i] for i in pair_indexes]
        written = save_qa_pairs([(p, chunks[chunk_index]) for p in pairs], source_id, metadata, approved_dir)
        for path in written:
            review_io.append_log(log_path, {
                "file": path.name,
                "decision": "approved",
                "edited": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        saved_counts[chunk_index] = saved_counts.get(chunk_index, 0) + len(written)
        return f"Saved {len(written)} pair(s): {', '.join(p.name for p in written)}."

    return doc_qa_agent
