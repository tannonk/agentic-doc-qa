# AGENTIC-DOC-QA

AGENTIC-DOC-QA is a Python package for building high-quality question-answering pairs from input documents.

**Note:** This tool is a redesign of the original [auto-doc-qa](https://gitlab.uzh.ch/LiRI/projects/auto-doc-qa), which used single-turn LLM prompts to generate Q&A pairs. AGENTIC-DOC-QA leverages `pydantic-ai`. See below for the motivation.

## Setup

To set up the AGENTIC-DOC-QA package, follow these steps:

```bash
conda create -n agentic-doc-qa python=3.13 -y && conda activate agentic-doc-qa
uv pip install -e . 
# alternatively, if you want to install for development, you can use:
uv pip install -e . --group dev
```

### Example usage:

#### Generate and Review Q&A Pairs

**Step 1:** Generate Q&A pairs from a PDF document using a specified model.

```bash
agentic-doc-qa generate "data/examples/raw/1706.03762v7.pdf" \
    --output-dir "data/examples/outputs/" \
    --n-candidates 4 \
    --model-name "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"
```

To run the generation step on multiple documents, you can use a shell loop:

```bash
for doc in data/examples/raw/*.pdf; do
    agentic-doc-qa generate "$doc" \
        --output-dir "data/examples/outputs/" \e
        --n-candidates 4 \
        --model-name "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"
done
```

**Step 2:** Review the generated Q&A pairs using a Streamlit app.

```bash
agentic-doc-qa review \
    --input-dir data/examples/outputs/ \
    --output-dir-base data/examples/outputs/ # validated outputs will be written to <output-dir-base>/validated/
```

#### Interactive Chat Mode

This is an experimental feature that allows you to interact with the Q&A generation and review process in a chat-like manner. You can use this mode to ask questions, request clarifications, or provide feedback on the generated Q&A pairs.

To review and refine Q&A pairs interactively, use the `chat` subcommand:

```bash
agentic-doc-qa chat "data/examples/raw/1706.03762v7.pdf" \
    --output-dir "data/examples/outputs/" \
    --n-candidates 2 \
    --model-name "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8" 
```

### Domain-specific Support

Q&A generation is aided by domain-specific configuration files, which can be specified using the `--domain-config` argument. 
Each configuration defines domain-specific rules, prompts, question types, and other parameters to guide the generation and review of Q&A pairs. By default, the system uses a base domain configuration, but allows for domain-specific overrides, meaning that you don't have redefine default behavior for every domain. See the `configs/domains/` directory for examples of domain configurations.


### Input Document Formats

Currently, we support the following input document formats:
- **PDF**: PDFs are processed as page images. The generation model must be able to handle image inputs.
- **Markdown**: Markdown files with optional YAML front matter. Where front matter is present, it will be included in the content sent to the generation model so that the model can generate metadata-aware questions. The front matter is also returned as part of the metadata for each chunk.
- **Other**: All other file types are converted to Markdown using the `docling` library, which uses a basic pipeline (no vision model, no OCR). This is a fallback and not fully validated, so the quality of the generated Q&A pairs may be lower.


### Model Support

For the generation and review steps, you can use any OpenAI-compatible model. 

For PDF input documents, we strongly recommend using a model that supports both text and image inputs, such as the `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` model.

If using a model that does not support image inputs, you can still process PDF documents by setting `--no-vision`. For non-markdown input documents, we will convert them to Markdown using `docling`'s basic pipeline (no vision model, no OCR) for processing. Note, this uses a default chunk size of 4092 characters, assuming a rather small context window.

### Why Pydantic-AI for this project?

The original auto-doc-qa project used single-turn LLM prompts with structured output to generate Q&A pairs using a deterministic pipeline (e.g. generate candidates -> judgea candidates -> save approved).

By leveraging `pydantic-ai` agents, we still run a deterministic pipeline but each task is handled by a specialized agent. This allos the system to ensure that each step is correctly executed before moving on to the next one. For example, the generation agent can be designed to ensure that the generated Q&A pairs are valid and meet certain criteria before passing them to the review agent, hopefully minimizing the number of rejected candidates and improving the overall quality of the generated Q&A pairs.
Since `pydantic-ai` agents abstract away the LLM prompt and response handling, we can focus on defining the structured data models for the Q&A pairs and the logic for generating and reviewing them, rather than worrying about the specifics of how to prompt the LLM or parse its responses.

We also adopt `pydantic-ai` in order to be able to interact with the agent during the generation and review process. This is an experimental approach, as a `chat`-like interaction could alternatively be implemented with a well-defined deterministic pipeline, but using `pydantic-ai` agents allows us to leverage the existing agent framework and its features, such as logging, error handling, and state management.

Potential other benefits of using pydantic-ai agents include:
- **Subagents**: could be used to handle specific tasks within the generation or review process, such as validating looking up references or checking for duplicates, which could further improve the quality of the generated Q&A pairs.
- **Web search**: could be used to validate the generated Q&A pairs against external sources, ensuring that the information is accurate and up-to-date.
