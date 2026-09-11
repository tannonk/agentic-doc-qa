#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
CLI entrypoint for agentic-doc-qa.

Example usage:

    python -m agentic_doc_qa.cli generate \
        --source "data/local/microbio-rag/raw_data/Packungsbeilagen Bakteriologie/Anaerotest.pdf" \
        --output-dir "data/local/agentic-doc-qa/test" \
        --domain "microbio_preanalytical" \
        --n-candidates 4 \
        --model-name "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
"""

import argparse
import logging
from pathlib import Path
from venv import logger
from venv import logger
from tqdm import tqdm


def _add_generation_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("source", type=Path, help="Path to the source document to generate QA pairs from.")
    p.add_argument("--model-name", required=True,
            help="Name of the model to use, as recognized by the OpenAI-compatible endpoint at --base-url.")
    p.add_argument("--base-url", default="http://localhost:8000/v1",
        help="Base URL of the OpenAI-compatible model endpoint (default: %(default)s).")
    p.add_argument("--api-key", default="none",
        help="API key for --base-url, if required (default: %(default)s).")
    p.add_argument("--domain-config", type=Path, default=None,
        help="Optional domain config to use for this document. If not provided, "
            "the base domain config will be used."
    )
    p.add_argument("--n-candidates", type=int, default=4,
        help="Number of candidate QA pairs to generate per chunk. Longer documents "
            "will end up with more total Q&A pairs (default: %(default)s).")


def _add_logging_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--verbose", action="store_true", help="Enable debug-level logging.")
    p.add_argument("--send-to-logfire", action="store_true", help="Send traces to Logfire.")


def _add_validated_dir_arg(p: argparse.ArgumentParser) -> None:
    # user-facing flag stays --output-dir for consistency with `generate`;
    # stored as args.validated_dir for relevance to the review/chat/web commands
    p.add_argument("--output-dir", dest="validated_dir", type=Path, required=True,
        metavar="OUTPUT_DIR", help="Directory to write human-validated QA pairs to.")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="agentic-doc-qa")
    sub = ap.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Batch-generate QA pairs, unattended.")
    gen.add_argument("--output-dir", type=Path, required=True,
        help="Directory to write generated QA pairs to.")
    _add_generation_args(gen)
    _add_logging_args(gen)
    # pages-per-chunk / image-scale as in the old generate.py's arg parser

    review = sub.add_parser("review", help="Launch the batch Streamlit reviewer.")
    review.add_argument("--input-dir", type=Path, required=True,
        help="Directory containing the output of a previous 'generate' run.")
    _add_validated_dir_arg(review)
    _add_logging_args(review)

    chat = sub.add_parser("chat", help="Interactive terminal review during generation.")
    _add_validated_dir_arg(chat)
    _add_generation_args(chat)
    _add_logging_args(chat)

    web = sub.add_parser("web", help="Interactive browser review during generation.")
    _add_validated_dir_arg(web)
    _add_generation_args(web)
    web.add_argument("--host", default="127.0.0.1", help="Host to bind the web server to (default: %(default)s).")
    web.add_argument("--port", type=int, default=7932, help="Port to bind the web server to (default: %(default)s).")
    web.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    _add_logging_args(web)

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
    chunks, metadata, source_id = load_chunks(args.source)

    # initalize the model
    model = build_model(args.model_name, base_url=args.base_url, api_key=args.api_key)

    qa_pairs = []
    for chunk in tqdm(chunks):
        chunk_qa_pairs = await propose_qa_pairs(
            model=model,
            domain_cfg=domain_cfg,
            chunk=chunk,
            n_candidates=args.n_candidates,
        )
        qa_pairs.extend((chunk_qa_pairs, chunk) for chunk_qa_pairs in chunk_qa_pairs)

    save_qa_pairs(accepted=qa_pairs, source_id=source_id, metadata=metadata, output_dir_base=args.output_dir)

    return


def cmd_generate(args) -> None:
    """wraps _generate_async() in asyncio.run() so we can call it from the CLI
    """
    import asyncio
    asyncio.run(_generate_async(args))


def cmd_review(args) -> None:
    """Shell out to streamlit -- review_app.py can't be called as a plain
    function without the Streamlit runtime."""
    import subprocess
    import sys
    app_path = Path(__file__).parent / "review_app.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_path), "--",
        "--input-dir", str(args.input_dir), "--validated-dir", str(args.validated_dir),
    ])


def cmd_chat(args) -> None:
    """Build doc_qa_agent (see agents.py) bound to this document's chunks/
    metadata/approved-dir, fire off the first chunk's proposal immediately
    (to_cli_sync() only reacts to what the reviewer types, so nothing shows
    up until we trigger this ourselves), then hand off to agent.to_cli_sync()
    for the rest of the session."""
    from agentic_doc_qa.agents import build_doc_qa_agent
    from agentic_doc_qa.domains import load_domain_config
    from agentic_doc_qa.models import build_model

    if not args.source.exists():
        raise FileNotFoundError(f"Source document not found: {args.source}")

    domain_cfg = load_domain_config(args.domain_config)
    model = build_model(args.model_name, base_url=args.base_url, api_key=args.api_key)
    agent = build_doc_qa_agent(model, args.source, domain_cfg, args.validated_dir, default_n_candidates=args.n_candidates)

    bootstrap = agent.run_sync(
        f"Begin the review session: propose QA pairs for chunk_index=0 "
        f"using n_candidates={args.n_candidates}, then present them to me."
    )
    print(bootstrap.output)
    agent.to_cli_sync(prog_name="agentic-doc-qa chat", message_history=bootstrap.all_messages())


def cmd_web(args) -> None:
    """Same setup as cmd_chat, then uvicorn.run(agent.to_web(), host=..., port=...)."""
    ...


def main() -> None:
    import logfire

    args = build_arg_parser().parse_args()

    logger.setLevel(logging.DEBUG) if args.verbose else logger.setLevel(logging.INFO)

    logfire.configure(send_to_logfire=args.send_to_logfire)
    logfire.instrument_pydantic_ai()

    {"generate": cmd_generate, "review": cmd_review, "chat": cmd_chat, "web": cmd_web}[args.command](args)

