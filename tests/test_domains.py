from pathlib import Path

from agentic_doc_qa.domains import _load_yaml, _render_definitions_block, load_domain_config


def test_render_definitions_block_formats_as_markdown_list():
    """Definitions (question types or difficulty levels) should render as one quoted-name/description bullet per line."""
    block = _render_definitions_block({"easy": "simple recall", "hard": "multi-step reasoning"})
    assert block == '- "easy": simple recall\n- "hard": multi-step reasoning'


def test_load_yaml_missing_file_returns_empty_dict(tmp_path):
    """A domain config path that doesn't exist should behave as 'no overrides', not raise."""
    assert _load_yaml(tmp_path / "missing.yaml") == {}


def test_load_yaml_empty_file_returns_empty_dict(tmp_path):
    """An empty YAML file parses to None; this must be normalized to an empty dict."""
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert _load_yaml(path) == {}


def test_load_domain_config_with_no_override_uses_base_only():
    """With configs_path=None, the config should fall back to the base config's domain and taxonomy untouched."""
    cfg = load_domain_config(None)
    assert cfg.domain == "base"
    assert "factual" in cfg.question_types
    assert "easy" in cfg.difficulty_levels
    assert "{question_types_block}" not in cfg.generation_instructions
    assert "{difficulty_levels_block}" not in cfg.generation_instructions


def test_load_domain_config_override_merges_and_extends_taxonomy(tmp_path):
    """A domain override should rename the domain, add new question types/levels on top of the base set, and append its instructions addendum."""
    override_path = tmp_path / "custom.yaml"
    override_path.write_text(
        "domain: custom\n"
        "question_types:\n"
        "  custom_type: a domain-specific type\n"
        "generation:\n"
        "  instructions_addendum: Extra domain-specific guidance.\n",
        encoding="utf-8",
    )

    cfg = load_domain_config(override_path)

    assert cfg.domain == "custom"
    assert "factual" in cfg.question_types  # base types kept
    assert cfg.question_types["custom_type"] == "a domain-specific type"
    assert cfg.generation_instructions.endswith("Extra domain-specific guidance.")


def test_load_domain_config_example_questions_sampled_into_addendum(tmp_path):
    """example_questions in the override should be sampled and substituted into the addendum's {example_questions} placeholder."""
    override_path = tmp_path / "custom.yaml"
    override_path.write_text(
        "domain: custom\n"
        "generation:\n"
        "  instructions_addendum: |\n"
        "    Examples:\n"
        "    {example_questions}\n"
        "  example_questions:\n"
        "    - What was the reported failure rate?\n"
        "  example_questions_sample_size: 1\n",
        encoding="utf-8",
    )

    cfg = load_domain_config(override_path)

    assert "What was the reported failure rate?" in cfg.generation_instructions
