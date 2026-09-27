"""Explicit entry point for the disabled Phase 9 site inventory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml

from modules.data_loader import EnvironmentManager
from modules.set_c_site_inventory import (
    activate_approved_inventory_config,
    approved_destination,
    build_source_context,
    run_site_inventory,
)


def load_config(config_path: str) -> dict:
    """Load the approved YAML without importing the simulation orchestrator."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    config["environment"]["raster_dir"] = str(
        (Path(__file__).resolve().parent / config["environment"]["raster_dir"]).resolve()
    )
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the aggregate-only Phase 9 site-inventory preflight."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default_experiment.yaml",
    )
    parser.add_argument(
        "--execute-approved-inventory",
        action="store_true",
        help="Required explicit gate after separate production-execution approval.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.execute_approved_inventory:
        parser.error(
            "--execute-approved-inventory is required; production execution is not "
            "authorized by the checked-in source"
        )
    config_path = args.config.resolve()
    config = activate_approved_inventory_config(load_config(str(config_path)))
    source_context = build_source_context(config, config_path)

    def load_environment() -> dict:
        manager = EnvironmentManager(config["environment"])
        manager.load_rasters()
        manager.build_masks()
        manager.normalize_layers()
        return manager.get_environment()

    result = run_site_inventory(
        config,
        environment_loader=load_environment,
        destination=approved_destination(),
        source_context=source_context,
    )
    print(f"Phase 9 inventory status: {result['status']}")
    print(f"Phase 9 inventory path: {result['path']}")
    print(f"Phase 9 inventory SHA-256: {result['payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
