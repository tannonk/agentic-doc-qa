#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
CLI entrypoint for agentic-doc-qa.

Example usage:

    # Generate Q&A pairs from a PDF document using a specified model and domain config.
    python -m agentic_doc_qa.cli generate \
        "data/local/microbio-rag/raw_data/Packungsbeilagen Bakteriologie/Anaerotest.pdf" \
        --output-dir-base "data/local/agentic-doc-qa/test" \
        --domain-config "configs/domains/microbio_preanalytical.yaml" \
        --model-name "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"

    # For markdown documents with frontmatter:
    python -m agentic_doc_qa.cli generate \
        "data/local/swissdox/markdown_samples/54036612.md" \
        --output-dir-base "data/local/agentic-doc-qa/test-swissdox" \
        --domain-config "configs/domains/swissdox.yaml" \
        --model-name "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"

    python -m agentic_doc_qa.cli chat \
        "data/local/swissdox/markdown_samples/56446175.md" \
        --output-dir-base "data/local/agentic-doc-qa/test-swissdox" \
        --domain-config "configs/domains/swissdox.yaml" \
        --model-name "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"

    # For batch generation of multiple PDFs in a directory, e.g. for the "Packungsbeilagen Bakteriologie" dataset:
    for pdf in "data/local/microbio-rag/raw_data/Packungsbeilagen Bakteriologie"/*.pdf; do
        python -m agentic_doc_qa.cli generate ${pdf} \
            --output-dir-base "data/local/microbio-rag/eval_data/agentic-doc-qa/Packungsbeilagen Bakteriologie" \
            --domain-config "configs/domains/microbio_preanalytical.yaml" \
            --pages-per-chunk 1 \
            --n-candidates 2 \
            --model-name "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"
    done

"""

import argparse
from pathlib import Path
import sys

from loguru import logger
from tqdm import tqdm


def _add_generation_args(p: argparse.ArgumentParser) -> None:
    """Add generation-related arguments to the given ArgumentParser."""
    p.add_argument("source", type=Path, help="Path to the source document to generate QA pairs from.")
    p.add_argument("--model-name", required=True,
            help="Name of the model to use, as recognized by the OpenAI-compatible endpoint at --base-url.")
    p.add_argument("--base-url", default="http://localhost:8000/v1",
        help="Base URL of the OpenAI-compatible model endpoint (default: %(default)s).")
    p.add_argument("--api-key", default="none",
        help="API key for --base-url, if required (default: %(default)s).")
    p.add_argument("--no-vision", action="store_true",
        help="Model at --model-name is text-only. For a .pdf source, converts it to "
            "Markdown text via anydoc instead of rendering page images.")
    p.add_argument("--domain-config", type=Path, default=None,
        help="Optional domain config to use for this document. If not provided, "
            "the base domain config will be used."
    )
    p.add_argument("-n", "--n-candidates", type=int, default=4,
        help="Number of candidate QA pairs to generate per chunk. Longer documents "
            "will end up with more total Q&A pairs (default: %(default)s).")
    p.add_argument("--pages-per-chunk", type=int, default=4,
        help="For .pdf sources, number of pages to render per chunk (default: %(default)s).")

def _add_logging_args(p: argparse.ArgumentParser) -> None:
    """Add logging-related arguments to the given ArgumentParser."""
    p.add_argument("--verbose", action="store_true", help="Enable debug-level logging.")
    p.add_argument("--send-to-logfire", action="store_true", help="Send traces to Logfire.")


def _add_output_dir_arg(p: argparse.ArgumentParser) -> None:
    """Add the --output-dir argument to the given ArgumentParser, 
    but store it as args.validated_dir for relevance to the review/chat/web commands."""
    # user-facing flag stays --output-dir for consistency with `generate`;
    # stored as args.validated_dir for relevance to the review/chat/web commands
    p.add_argument("-o", "--output-dir-base", type=Path, required=True,
        help="Base directory to write outputs to. "
        "For generation, a subdirectory will be created based on "
        "the source document's stem, e.g. <output-dir-base>/<source-stem>/001.json, 002.json, etc."
        "For review/chat/web, this output directory will <output-dir-base>/validated/<source-stem>/001.json, 002.json, etc.")

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="agentic-doc-qa")
    sub = ap.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Batch-generate QA pairs, unattended.")
    _add_output_dir_arg(gen)
    _add_generation_args(gen)
    _add_logging_args(gen)
    
    review = sub.add_parser("review", help="Launch the batch Streamlit reviewer.")
    review.add_argument("--input-dir", type=Path, required=True,
        help="Directory containing the output of a previous 'generate' run.")
    _add_output_dir_arg(review)
    _add_logging_args(review)

    chat = sub.add_parser("chat", help="Interactive terminal review during generation.")
    _add_output_dir_arg(chat)
    _add_generation_args(chat)
    _add_logging_args(chat)

    return ap


async def _generate_async(args) -> None:
    """Mirrors old generate.py's run(): load_domain_config, load_chunks, loop
    chunks through pipeline.propose_qa_pairs(), write to --output-dir via
    pipeline.save_qa_pairs().
    """
    
    from agentic_doc_qa.models import build_model
    from agentic_doc_qa.documents import load_chunks
    from agentic_doc_qa.domains import load_domain_config
    from agentic_doc_qa.pipeline import propose_qa_pairs, save_qa_pairs

    # load the domain config if provided, else None (pipeline.propose_qa_pairs() will handle None by falling back to base)
    domain_cfg = load_domain_config(args.domain_config)
    
    if not args.source.exists():
        raise FileNotFoundError(f"Source document not found: {args.source}")

    # load the document into chunks, plus metadata and a source_id
    chunks, metadata, source_id = load_chunks(args.source, vision=not args.no_vision, pages_per_chunk=args.pages_per_chunk)

    # initalize the model
    model = build_model(args.model_name, base_url=args.base_url, api_key=args.api_key)

    qa_pairs = []
    for chunk in tqdm(chunks):
        chunk_qa_pairs = await propose_qa_pairs(
            model=model,
            domain_cfg=domain_cfg,
            chunk=chunk,
            n_candidates=args.n_candidates,
            total_chunks=len(chunks),
        )
        qa_pairs.extend((chunk_qa_pairs, chunk) for chunk_qa_pairs in chunk_qa_pairs)

    save_qa_pairs(accepted=qa_pairs, source_id=source_id, metadata=metadata, output_dir_base=args.output_dir_base)

    return


def cmd_generate(args) -> None:
    """wraps _generate_async() in asyncio.run() so we can call it from the CLI
    """
    import logfire

    logfire.configure(send_to_logfire=args.send_to_logfire)
    logfire.instrument_pydantic_ai()

    import asyncio
    asyncio.run(_generate_async(args))


def cmd_review(args) -> None:
    """Shell out to streamlit -- review_app.py can't be called as a plain
    function without the Streamlit runtime."""
    import subprocess
    
    app_path = Path(__file__).parent.parent / "review_app.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_path), "--",
        "--input-dir", str(args.input_dir), "--validated-dir", str(args.output_dir_base / "validated"),
    ])


def cmd_chat(args) -> None:
    """Build doc_qa_agent (see agents.py) bound to this document's chunks/
    metadata/approved-dir, fire off the first chunk's proposal immediately
    (to_cli_sync() only reacts to what the reviewer types, so nothing shows
    up until we trigger this ourselves), then hand off to agent.to_cli_sync()
    for the rest of the session."""
    import logfire
    from agentic_doc_qa.agents import build_doc_qa_agent
    from agentic_doc_qa.domains import load_domain_config
    from agentic_doc_qa.models import build_model

    if not args.source.exists():
        raise FileNotFoundError(f"Source document not found: {args.source}")

    logfire.configure(send_to_logfire=args.send_to_logfire)
    logfire.instrument_pydantic_ai()

    domain_cfg = load_domain_config(args.domain_config)
    model = build_model(args.model_name, base_url=args.base_url, api_key=args.api_key)
    agent = build_doc_qa_agent(model, args.source, domain_cfg, args.output_dir_base / "validated", default_n_candidates=args.n_candidates, vision=not args.no_vision)

    # trigger the first chunk's proposal immediately
    bootstrap = agent.run_sync(
        f"Begin the review session: propose QA pairs for chunk_index=0 "
        f"using n_candidates={args.n_candidates}, then present them to me."
    )
    print(bootstrap.output)
    agent.to_cli_sync(prog_name="agentic-doc-qa chat", message_history=bootstrap.all_messages())


def cmd_web(args) -> None:
    """Placeholder for a future web-based review interface. Currently, this is not implemented."""
    raise NotImplementedError("Web-based review interface is not yet implemented.")


def main() -> None:
    args = build_arg_parser().parse_args()

    # set up stderr logging (loguru uses DEBUG level by default)
    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    {"generate": cmd_generate, "review": cmd_review, "chat": cmd_chat, "web": cmd_web}[args.command](args)


if __name__ == "__main__":
    main()