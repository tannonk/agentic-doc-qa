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


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="agentic-doc-qa")
    sub = ap.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Batch-generate QA pairs, unattended.")
    gen.add_argument("source", type=Path)
    gen.add_argument("--output-dir", type=Path, required=True)
    gen.add_argument("--domain-config", type=Path, default=None)
    gen.add_argument("--n-candidates", type=int, default=4, help="Number of candidate QA pairs to generate per chunk. Longer documents will end up with more total Q&A pairs.")
    gen.add_argument("--model-name", required=True)
    gen.add_argument("--base-url", default="http://localhost:8000/v1")
    gen.add_argument("--api-key", default="none")
    gen.add_argument("--verbose", action="store_true")
    gen.add_argument("--send-to-logfire", action="store_true")
    # pages-per-chunk / image-scale as in the old generate.py's arg parser

    review = sub.add_parser("review", help="Launch the batch Streamlit reviewer.")
    review.add_argument("--input-dir", type=Path, required=True)
    review.add_argument("--approved-dir", type=Path, required=True)
    review.add_argument("--verbose", action="store_true")

    chat = sub.add_parser("chat", help="Interactive terminal review during generation.")
    chat.add_argument("source", type=Path)
    chat.add_argument("--approved-dir", type=Path, required=True)
    chat.add_argument("--domain-config", type=Path, default=None)
    chat.add_argument("--n-candidates", type=int, default=4, help="Default number of candidate QA pairs to propose per chunk. The reviewer can ask for a different count in chat.")
    chat.add_argument("--model-name", required=True)
    chat.add_argument("--base-url", default="http://localhost:8000/v1")
    chat.add_argument("--api-key", default="none")
    chat.add_argument("--verbose", action="store_true")
    chat.add_argument("--send-to-logfire", action="store_true")

    web = sub.add_parser("web", help="Interactive browser review during generation.")
    # same args as chat, plus:
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=7932)
    web.add_argument("--reload", action="store_true")
    web.add_argument("--verbose", action="store_true")
    web.add_argument("--send-to-logfire", action="store_true")

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
    """Mirrors old generate.py's run(): load_domain_config, load_chunks, loop
    chunks through pipeline.propose_qa_pairs(), write to --output-dir via
    pipeline.save_qa_pairs()."""
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
        "--input-dir", str(args.input_dir), "--approved-dir", str(args.approved_dir),
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
    agent = build_doc_qa_agent(model, args.source, domain_cfg, args.approved_dir, default_n_candidates=args.n_candidates)

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

