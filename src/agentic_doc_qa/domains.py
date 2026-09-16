#!/usr/bin/env python
#-*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import random

import yaml
from loguru import logger

BASE_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "domains" / "_base.yaml"


@dataclass
class DomainConfig:
    domain: str
    question_types: dict[str, str]      # name -> description; single source of truth for the taxonomy
    difficulty_levels: dict[str, str]   # name -> description
    generation_instructions: str        # includes {n_candidates} placeholder, unfilled
    generation_vision_addendum: str      # empty string if none
    generation_settings: dict[str, Any]
    generation_include_metadata_block: bool
    judge_instructions: str
    judge_settings: dict[str, Any]


def _render_definitions_block(definitions: dict[str, str]) -> str:
    """Renders a block of definitions (question types or difficulty levels) as a Markdown list for inclusion in the prompt text."""
    return "\n".join(f'- "{name}": {description}' for name, description in definitions.items())


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file from the given path, returning an empty dict if the file does not exist."""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_domain_config(configs_path: Path | None) -> DomainConfig:
    """Load the domain configuration from the given YAML file path, merging it with the base config."""

    # load the base config
    base = _load_yaml(BASE_CONFIG_PATH)
    logger.debug(f"Loaded base domain config from {BASE_CONFIG_PATH}: {base}")

    # load the domain-specific config
    override = {}
    if configs_path:
        override = _load_yaml(configs_path)  # {} if the file doesn't exist -- unknown domains fall back to base only, same as today's dict.get(domain, _build_generic_prompt)
        logger.debug(f"Loaded domain-specific config from {configs_path}: {override}")
    else:
        logger.warning("No domain-specific config provided. Falling back to base config settings.")

    domain_base = base.get("domain", "base")
    domain_override = override.get("domain", domain_base)

    question_types = {**base.get("question_types", {}), **override.get("question_types", {})}
    difficulty_levels = {**base.get("difficulty_levels", {}), **override.get("difficulty_levels", {})}

    gen_base = base.get("generation", {})
    gen_override = override.get("generation", {})

    instructions = gen_base.get("instructions", "")
    instructions = instructions.replace("{question_types_block}", _render_definitions_block(question_types))
    instructions = instructions.replace("{difficulty_levels_block}", _render_definitions_block(difficulty_levels))
    addendum = gen_override.get("instructions_addendum")
    if addendum:
        example_questions = gen_override.get("example_questions")
        if example_questions:
            sample_size = gen_override.get("example_questions_sample_size", len(example_questions))
            sampled = random.sample(example_questions, min(len(example_questions), sample_size))
            addendum = addendum.format(example_questions="\n".join(f"    - {q}" for q in sampled))
        instructions = f"{instructions}\n\n{addendum}"

    judge_base = base.get("judge", {})
    judge_override = override.get("judge", {})
    judge_instructions = judge_base.get("instructions", "")
    judge_addendum = judge_override.get("instructions_addendum")
    if judge_addendum:
        judge_instructions = f"{judge_instructions}\n\n{judge_addendum}"

    return DomainConfig(
        domain=domain_override,
        question_types=question_types,
        difficulty_levels=difficulty_levels,
        generation_instructions=instructions,
        generation_vision_addendum=gen_base.get("vision_addendum", ""),
        generation_settings={**gen_base.get("model_settings", {}), **gen_override.get("model_settings", {})},
        generation_include_metadata_block=gen_override.get("include_metadata_block", False),
        judge_instructions=judge_instructions,
        judge_settings={**judge_base.get("model_settings", {}), **judge_override.get("model_settings", {})},
    )