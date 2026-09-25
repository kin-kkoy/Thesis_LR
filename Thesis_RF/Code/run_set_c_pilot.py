"""Explicit entry point for the one approved Phase 8 diagnostic pilot."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from orchestrator import load_config, run_set_c_feasibility_pilot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the approved aggregate-only model-free Set C feasibility pilot."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default_experiment.yaml",
    )
    parser.add_argument(
        "--execute-approved-pilot",
        action="store_true",
        help="Required explicit authorization gate for the exact approved matrix.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.execute_approved_pilot:
        parser.error("--execute-approved-pilot is required; the harness is disabled by default")
    config_path = args.config.resolve()
    result = run_set_c_feasibility_pilot(
        load_config(str(config_path)),
        explicitly_authorized=True,
        config_path=config_path,
    )
    print(f"Phase 8 aggregate status: {result['status']}")
    print(f"Phase 8 aggregate path: {result['path']}")
    print(f"Phase 8 aggregate SHA-256: {result['payload_sha256']}")
    print(f"Phase 8 v2 acceptance status: {result['acceptance_status']}")
    return 0 if result["acceptance_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
