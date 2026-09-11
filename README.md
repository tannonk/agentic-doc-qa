# AGENT-DOC-QA

AGENT-DOC-QA is a Python package for building high-quality question-answering pairs from input documents.



**Note:** This tool is a redesign of the original [auto-doc-qa](https://gitlab.uzh.ch/LiRI/projects/auto-doc-qa), which used single-turn LLM prompts to generate Q&A pairs. AGENT-DOC-QA leverages `pydantic-ai`.

## Setup

To set up the AGENT-DOC-QA package, follow these steps:

```bash
conda create -n agentic-doc-qa python=3.13 -y && conda activate agentic-doc-qa
uv pip install -e .
```

Example usage:

```python
agentic-doc-qa generate "data/local/microbio-rag/raw_data/Packungsbeilagen Bakteriologie/BinaxNOW Legionella.pdf" \
    --output-dir "data/local/agentic-doc-qa/test" \
    --n-candidates 4 \
    --model-name "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
```


# TODO

- fix bug in review logic relating to file paths and approved subdir. Previous version used guaranteed unique file names, but now we have multiple files with the same name in different subdirs. Need to ensure that the review logic correctly identifies and handles these cases.