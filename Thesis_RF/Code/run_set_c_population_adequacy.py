"""Explicit entry point for a future approved Gate 6 aggregate diagnostic."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata
import platform
from pathlib import Path
import subprocess
from typing import Sequence

from modules.data_loader import EnvironmentManager
from modules.set_c_population_adequacy import (
    PRODUCTION_GENERATION_COMMAND,
    PRODUCTION_SOURCE_RELATIVE_PATHS,
    activate_approved_population_config,
    approved_destination,
    preflight_population_launch,
    run_population_adequacy,
)
from modules.set_c_site_inventory import EXPECTED_RASTER_SHA256
from modules.transition_observer import mask_sha256
from orchestrator import load_config


APPROVED_CONFIG_PATH = (
    Path(__file__).resolve().parent / "config" / "default_experiment.yaml"
).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the separately approved aggregate-only Set C adequacy expansion."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default_experiment.yaml",
    )
    parser.add_argument(
        "--execute-approved-edge-diagnostic",
        action="store_true",
        help="Required future one-run authorization gate; absent by default.",
    )
    return parser


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_context(
    config: dict, config_path: Path, environment: dict
) -> dict[str, object]:
    code_root = Path(__file__).resolve().parent
    repository_root = code_root.parents[1]
    revision = subprocess.run(
        ["git", "-c", f"safe.directory={repository_root.as_posix()}", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository_root.as_posix()}",
            "status",
            "--porcelain",
            "--",
            ".",
            ":(exclude)Thesis-try.qgz",
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_hashes = {
        relative: _file_sha256(code_root / relative)
        for relative in PRODUCTION_SOURCE_RELATIVE_PATHS
    }
    raster_dir = code_root / config["environment"]["raster_dir"]
    raster_hashes = {
        filename: _file_sha256(raster_dir / filename)
        for filename in EXPECTED_RASTER_SHA256
    }
    if raster_hashes != EXPECTED_RASTER_SHA256:
        raise RuntimeError("Approved raster provenance changed")
    input_hashes = dict(raster_hashes)
    input_hashes.update(
        {
            "simulation_valid_mask": mask_sha256(environment["burnable_mask"] & ~environment["nodata_mask"]),
            "mapped_building_mask": mask_sha256(environment["building_presence"] > 0),
        }
    )
    packages = {}
    for package in ("numpy", "scipy", "rasterio", "PyYAML"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "unavailable"
    return {
        "source_revision": revision,
        "source_dirty": bool(dirty),
        "source_hashes": source_hashes,
        "input_hashes": input_hashes,
        "generation_command": PRODUCTION_GENERATION_COMMAND,
        "environment_record": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.execute_approved_edge_diagnostic:
        parser.error(
            "--execute-approved-edge-diagnostic is required; production execution is unapproved by default"
        )
    config_path = args.config.resolve()
    if config_path != APPROVED_CONFIG_PATH:
        raise PermissionError("Only the checked-in default experiment configuration is allowed")
    config = activate_approved_population_config(load_config(str(config_path)))
    preflight_population_launch(config, approved_destination())
    manager = EnvironmentManager(config["environment"])
    manager.load_rasters()
    manager.build_masks()
    manager.normalize_layers()
    environment = manager.get_environment()
    result = run_population_adequacy(
        config,
        destination=approved_destination(),
        source_context=_source_context(config, config_path, environment),
        environment=environment,
    )
    print(f"Gate 6 diagnostic status: {result['status']}")
    print(f"Gate 6 diagnostic path: {result['path']}")
    print(f"Gate 6 diagnostic payload SHA-256: {result['payload_sha256']}")
    print(f"Gate 6 adequacy status: {result['adequacy_status']}")
    return 0 if result["adequacy_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
