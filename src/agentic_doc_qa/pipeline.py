#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any

from loguru import logger
from dataclasses import replace

from agentic_doc_qa.agents import GenerationDeps, build_generation_agent, build_judge_agent
from agentic_doc_qa.documents import Chunk
from agentic_doc_qa.domains import DomainConfig
from agentic_doc_qa.schemas import JudgedQAPair, QAJudgement, QAPair, QAVerdictDecision


def _build_judge_user_content(
    chunk: Chunk, candidates: list[QAPair], avoid_questions: list[str] | None = None
) -> str | list:
    listing = "\n".join(
        f"[{i}] Q: {c.question} A: {c.answer} (type={c.question_type}, level={c.question_level})"
        for i, c in enumerate(candidates)
    )
    listing_block = f"Candidate QA pairs to judge:\n{listing}"
    if avoid_questions:
        avoid_block = "\n".join(f"- {q}" for q in avoid_questions)
        listing_block += (
            f"\n\nAlready-accepted questions from earlier rounds (reject anything that "
            f"duplicates or closely paraphrases these):\n{avoid_block}"
        )
    if isinstance(chunk.content, str):
        return f"{chunk.content}\n\n---\n{listing_block}"
    return [*chunk.content, listing_block]


def select_accepted(candidates: list[QAPair], judgement: QAJudgement) -> list[JudgedQAPair]:
    """Return a list of JudgedQAPair for all candidates that were accepted by the judge."""
    verdict_by_index = {v.candidate_index: v for v in judgement.verdicts}
    accepted = []
    for i, candidate in enumerate(candidates):
        verdict = verdict_by_index.get(i)
        if verdict is not None and verdict.decision == QAVerdictDecision.ACCEPT:
            accepted.append(JudgedQAPair(pair=candidate, verdict=verdict))
    return accepted


def select_ranked(candidates: list[QAPair], judgement: QAJudgement) -> list[JudgedQAPair]:
    """Similar to select_accepted, but returns a list of JudgedQAPair ordered by the judge's score, highest first."""
    verdict_by_index = {v.candidate_index: v for v in judgement.verdicts}
    scored = []
    for i, candidate in enumerate(candidates):
        verdict = verdict_by_index.get(i)
        if verdict is not None and verdict.decision == QAVerdictDecision.ACCEPT:
            scored.append((verdict.score, JudgedQAPair(pair=candidate, verdict=verdict)))
    scored.sort(key=lambda scored_pair: scored_pair[0], reverse=True)
    return [judged for _, judged in scored]


def _summarize_rejections(candidates: list[QAPair], judgement: QAJudgement) -> str:
    """Return a string summarizing the rejected candidates and their rejection reasons."""
    verdict_by_index = {v.candidate_index: v for v in judgement.verdicts}
    lines = []
    for i, candidate in enumerate(candidates):
        verdict = verdict_by_index.get(i)
        if verdict is not None and verdict.decision != QAVerdictDecision.ACCEPT:
            lines.append(f'- "{candidate.question}" was rejected ({verdict.decision.value}): {verdict.rationale}')
    return "\n".join(lines)


def get_agent_instructions(agent) -> str:
    """Return the full instructions for an agent, including any dynamic
    instructions added via @agent.instructions-decorated functions.
    """
    instructions = ""
    for instr_item in agent._instructions:
        if isinstance(instr_item.instruction, str):
            instructions += instr_item.instruction + "\n"
    return instructions


async def propose_qa_pairs(
    model, domain_cfg: DomainConfig, chunk: Chunk, n_candidates: int,
    reviewer_feedback: str | None = None,
    avoid_questions: list[str] | None = None,
    total_chunks: int | None = None,
) -> list[JudgedQAPair]:
    """Generate, judge, and — if anything was rejected — regenerate once with
    the judge's feedback, then judge again. Every accepted pair from both
    attempts is kept in a pool; if the pool holds more than n_candidates
    pairs, a final judge pass over the whole pool scores and ranks them, and
    only the top n_candidates are returned. This is the single function every
    surface (batch CLI, chat, web) must call; nothing else may hand a QAPair
    to a caller.

    reviewer_feedback: free-text steering from a human reviewer (distinct
    from the judge's own rejection rationale below), passed through to every
    attempt.
    avoid_questions: question texts already shown/accepted for this chunk in
    an earlier propose_qa_pairs() call (e.g. an earlier round in the same
    chat session) — both the generation agent and the judge are told not to
    repeat/accept duplicates of these.
    """
    generation_agent = build_generation_agent(model, domain_cfg, n_candidates=n_candidates)
    logger.debug(f"Initialized generation agent with instructions:\n\n{get_agent_instructions(generation_agent)}")

    judge_agent = build_judge_agent(model, domain_cfg)
    logger.debug(f"Initialized judge agent with instructions:\n\n{get_agent_instructions(judge_agent)}")

    avoid_questions = list(avoid_questions or [])
    deps = GenerationDeps(
        chunk=chunk,
        n_candidates=n_candidates,
        reviewer_feedback=reviewer_feedback,
        existing_questions=avoid_questions,
    )
    pool: list[JudgedQAPair] = []
    current_usage = None
    for attempt in range(2):
        logger.info(f"Generating QA pairs for chunk {chunk.index} (of {total_chunks}), attempt {attempt + 1}")

        result = await generation_agent.run(chunk.content, deps=deps, usage=current_usage)
        candidates = result.output.pairs

        current_usage = result.usage # update usage for next attempt
        logger.info(f"Current usage after generation iteration {attempt + 1}: {current_usage}")

        judgement_result = await judge_agent.run(
            _build_judge_user_content(chunk, candidates, avoid_questions), usage=current_usage
        )
        current_usage = judgement_result.usage # update usage for next attempt
        logger.info(f"Current usage after judging iteration {attempt + 1}: {current_usage}")
        accepted = select_accepted(candidates, judgement_result.output)
        logger.info(f"Accepted {len(accepted)} out of {len(candidates)} pairs for chunk {chunk.index}")

        pool.extend(accepted)
        if len(accepted) >= len(candidates) or attempt == 1:
            break

        deps = replace(
            deps,
            feedback=_summarize_rejections(candidates, judgement_result.output),
            existing_questions=[*avoid_questions, *(jp.pair.question for jp in pool)],
        )
        logger.info(f"Feedback for next attempt: {deps.feedback}")

    if len(pool) <= n_candidates:
        return pool

    logger.info(f"Pool has {len(pool)} candidates for chunk {chunk.index}; running final validation to pick top {n_candidates}")
    pool_pairs = [jp.pair for jp in pool]
    final_judgement = await judge_agent.run(
        _build_judge_user_content(chunk, pool_pairs, avoid_questions), usage=current_usage
    )

    logger.info(f"Current usage after final validation: {final_judgement.usage}")

    return select_ranked(pool_pairs, final_judgement.output)[:n_candidates]


def save_qa_pairs(
    accepted: list[tuple[JudgedQAPair, Chunk]],
    source_id: str,
    metadata: dict[str, Any],
    output_dir_base: Path,
) -> list[Path]:
    """Write each accepted pair to its own numbered JSON file under
    output_dir_base/source_id/. Numbering continues from whatever's already
    in that directory, so calling this more than once for the same source_id
    (e.g. one save per chunk in an interactive chat session) appends rather
    than overwriting earlier saves. Returns the list of files written, in
    the same order as `accepted`."""
    written: list[Path] = []

    output_dir = output_dir_base / f"{source_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    start_index = len(list(output_dir.glob("*.json")))

    for offset, (judged, chunk) in enumerate(accepted):
        qa_pair = judged.pair
        output_filename = output_dir / f"{start_index + offset + 1:03d}.json"
        if output_filename.exists():
            logger.warning(f"Output file {output_filename} already exists. Overwriting.")

        # per-pair copy: chunk provenance differs between pairs of the same document
        pair_metadata = {**metadata, "chunk_index": chunk.index}
        if chunk.pages is not None:
            pair_metadata["pages"] = list(chunk.pages)
        with open(output_filename, "w", encoding="utf-8") as f:
            # combine the QA pair with the metadata for this source document
            combined_data = {
                "question": qa_pair.question,
                "answer": qa_pair.answer,
                "question_type": qa_pair.question_type,
                "question_level": qa_pair.question_level,
                "metadata": pair_metadata,
                "judgement": {
                    "decision": judged.verdict.decision,
                    "score": judged.verdict.score,
                    "rationale": judged.verdict.rationale,
                },
            }
            f.write(json.dumps(combined_data, indent=2, ensure_ascii=False, sort_keys=True))
        written.append(output_filename)

    logger.info(f"Saved {len(written)} accepted QA pairs to {output_dir}")

    return written
