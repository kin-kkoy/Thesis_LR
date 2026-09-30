"""Explicit entry point for separately authorized Gate 7 package generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from modules.data_loader import EnvironmentManager
from modules.set_c_generation import (
    preflight_generation,
    run_authorized_generation,
    GENERATION_AUTHORIZATION,
)
from modules.set_c_population_adequacy import activate_approved_population_config
from orchestrator import load_config


APPROVED_CONFIG = (Path(__file__).resolve().parent / "config" / "default_experiment.yaml").resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the immutable authoritative Set C package after Gate 7 approval."
    )
    parser.add_argument("--config", type=Path, default=APPROVED_CONFIG)
    parser.add_argument(
        "--execute-authorized-generation",
        action="store_true",
        help="Required one-run Gate 7 approval switch; disabled by default.",
    )
    parser.add_argument("--approved-source-revision")
    parser.add_argument("--authorization-reference")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.execute_authorized_generation:
        parser.error("--execute-authorized-generation is required; production generation is disabled")
    if not args.approved_source_revision or not args.authorization_reference:
        parser.error("Gate 7 approval requires --approved-source-revision and --authorization-reference")
    if args.authorization_reference != GENERATION_AUTHORIZATION:
        raise PermissionError("--authorization-reference differs from the Gate 7 contract")
    config_path = args.config.resolve()
    if config_path != APPROVED_CONFIG:
        raise PermissionError("Only the checked-in default experiment configuration is allowed")
    config = activate_approved_population_config(load_config(str(config_path)))
    preflight_generation(
        config_path=config_path,
        config=config,
        approved_revision=args.approved_source_revision,
    )
    manager = EnvironmentManager(config["environment"])
    manager.load_rasters()
    manager.build_masks()
    manager.normalize_layers()
    result = run_authorized_generation(
        config=config,
        config_path=config_path,
        environment=manager.get_environment(),
        authorization_reference=args.authorization_reference,
        approved_revision=args.approved_source_revision,
        generation_command=sys.argv,
        production_authorized=True,
    )
    print(f"Set C package ID: {result['package_id']}")
    print(f"Set C package path: {result['path']}")
    print(f"Set C family manifest SHA-256: {result['family_manifest_sha256']}")
    print(f"Set C run order SHA-256: {result['run_order_sha256']}")
    print(f"Set C source revision: {result['source_revision']}")
    print(f"Set C published rows: {result['rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
