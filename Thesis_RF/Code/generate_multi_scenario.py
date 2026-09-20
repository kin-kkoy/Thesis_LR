"""Prepare provenance-bearing Set C simulated transitions for wind scenarios.

This wrapper does not run the CA. It varies explicit scenario metadata and delegates
construction from configured simulator-produced adjacent state-at-t/state-at-t+1
rasters. These are not observed temporal fire rasters. Running this wrapper creates
new immutable artifacts and requires separate approval.

Research grounding:
  - Thesis Section 1.4: "evaluates fire spread scenarios under varied wind conditions"
  - Thesis Section 4.6: "sensitivity analysis varying key parameters like wind weighting"
  - Gao et al. 2008: wind velocity and direction directly affect fire spread patterns

Usage:
    python generate_multi_scenario.py
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

import pandas as pd
import yaml

from dataset_generator import SyntheticDatasetGenerator
from modules.feature_pipeline import (
    CANONICAL_FEATURE_NAMES as FEATURE_NAMES,
    COMBINED_MANIFEST_VERSION,
    OBSERVATION_MANIFEST_VERSION,
    POSITIVE_LABEL,
    OBSERVATION_METADATA_NAMES as METADATA_NAMES,
    SET_C_SCHEMA_VERSION,
    TARGET_NAME,
    TARGET_VERSION,
)
from modules.wind_convention import WindContract


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_scenarios(base_config: dict) -> list[dict]:
    """Build only explicitly recorded, wind-matched simulated state pairs."""
    dataset_cfg = base_config.get("dataset_generation", {})
    provenance = dataset_cfg.get("provenance", {})
    for field in ("experiment_id",):
        if provenance.get(field) in (None, ""):
            raise ValueError(
                f"dataset_generation.provenance.{field} must be configured before scenarios are built"
            )
    if re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", str(provenance["experiment_id"])
    ) is None:
        raise ValueError("experiment_id must be a safe, path-free identifier")
    records = dataset_cfg.get("scenario_records")
    if not isinstance(records, list) or not records:
        raise ValueError(
            "dataset_generation.scenario_records must contain explicit wind-matched "
            "simulator-produced state-at-t/state-at-t+1 records"
        )

    required = {
        "event_id",
        "scenario_id",
        "run_id",
        "seed",
        "timestep_t",
        "grid_id",
        "speed_kmh",
        "direction_deg",
        "state_t_file",
        "state_t1_file",
    }

    scenarios = []
    identities: set[tuple[str, str, str, int]] = set()
    state_pair_bindings: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"scenario_records[{index}] must be a mapping")
        missing = sorted(
            field for field in required if record.get(field) in (None, "")
        )
        if missing:
            raise ValueError(f"scenario_records[{index}] is incomplete: {missing}")
        transition_source_type, transition_source_provenance = (
            SyntheticDatasetGenerator.validate_transition_source(
                record.get("transition_source_type"),
                record.get("transition_source_provenance"),
            )
        )
        if int(transition_source_provenance["seed"]) != int(record["seed"]):
            raise ValueError(
                f"scenario_records[{index}] simulated source seed is inconsistent"
            )
        scenario_wind_config = copy.deepcopy(base_config["wind"])
        scenario_wind_config["speed_kmh"] = float(record["speed_kmh"])
        scenario_wind_config["direction_deg"] = float(record["direction_deg"])
        scenario_wind = WindContract.from_config(scenario_wind_config).to_config()
        if transition_source_provenance["wind"] != scenario_wind:
            raise ValueError(
                f"scenario_records[{index}] wind does not match its simulated source"
            )
        identity = (
            str(record["event_id"]),
            str(record["scenario_id"]),
            str(record["run_id"]),
            int(record["timestep_t"]),
        )
        identifier_values = (*identity[:3], str(record["grid_id"]))
        if any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None
            for value in identifier_values
        ):
            raise ValueError(
                f"scenario_records[{index}] event, scenario, run, and grid identifiers "
                "must be safe, path-free values"
            )
        if identity in identities:
            raise ValueError(f"Duplicate scenario outcome identity: {identity}")
        identities.add(identity)
        state_pair_binding = (str(record["state_t_file"]), str(record["state_t1_file"]))
        if state_pair_binding in state_pair_bindings:
            raise ValueError(
                "A state-at-t/state-at-t+1 pair cannot be rebound to another wind scenario"
            )
        state_pair_bindings.add(state_pair_binding)

        cfg = copy.deepcopy(base_config)
        cfg["wind"] = scenario_wind
        cfg["simulation"]["seed"] = int(record["seed"])
        cfg["dataset_generation"]["target"].update(
            state_t_file=str(record["state_t_file"]),
            state_t1_file=str(record["state_t1_file"]),
        )
        scenario_provenance = cfg["dataset_generation"].setdefault("provenance", {})
        scenario_provenance.update(
            {
                "set_id": "SetC",
                "event_id": identity[0],
                "scenario_id": identity[1],
                "run_id": identity[2],
                "seed": int(record["seed"]),
                "timestep_t": identity[3],
                "grid_id": str(record["grid_id"]),
                "transition_source_type": transition_source_type,
                "transition_source_provenance": transition_source_provenance,
            }
        )
        experiment_id = str(scenario_provenance["experiment_id"])
        cfg["dataset_generation"]["output_csv"] = (
            f"dataFiles/{experiment_id}_{identity[0]}_{identity[1]}_{identity[2]}_t{identity[3]}.csv"
        )
        scenarios.append(cfg)

    return scenarios


def run_all_scenarios(scenarios: list[dict], output_csv: str) -> str:
    """Run each scenario and combine results into one dataset."""
    all_frames: list[pd.DataFrame] = []
    scenario_manifests: list[dict] = []
    total = len(scenarios)

    for i, cfg in enumerate(scenarios):
        speed = cfg["wind"]["speed_kmh"]
        direction = cfg["wind"]["direction_deg"]
        seed = cfg["simulation"]["seed"]
        print(f"\n{'='*60}")
        print(f"Scenario {i+1}/{total}: wind={speed} km/h, dir={direction}°, seed={seed}")
        print(f"{'='*60}")

        generator = SyntheticDatasetGenerator(cfg)
        scenario_path = generator.generate()
        df = pd.read_csv(scenario_path)
        all_frames.append(df)
        manifest_path = Path(scenario_path).with_suffix(
            Path(scenario_path).suffix + ".manifest.json"
        )
        if not manifest_path.exists():
            raise FileNotFoundError(f"Scenario manifest is missing: {manifest_path}")
        content = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_provenance = cfg["dataset_generation"]["provenance"]
        target_counts = {
            str(key): int(value)
            for key, value in df[TARGET_NAME].value_counts().sort_index().items()
        }
        if content.get("dataset_sha256") != _file_sha256(Path(scenario_path)):
            raise ValueError(f"Scenario {i} dataset hash disagrees with its manifest")
        if content.get("row_count") != len(df) or content.get("target_counts") != target_counts:
            raise ValueError(f"Scenario {i} row/class counts disagree with its manifest")
        for field in ("experiment_id", "event_id", "scenario_id", "run_id", "timestep_t"):
            if str(content.get(field)) != str(expected_provenance[field]):
                raise ValueError(f"Scenario {i} manifest disagrees on {field}")
        if int(content.get("seed")) != int(seed):
            raise ValueError(f"Scenario {i} manifest disagrees on seed")
        expected_wind = WindContract.from_config(cfg["wind"]).to_manifest()
        if content.get("wind") != expected_wind:
            raise ValueError(f"Scenario {i} manifest disagrees on wind binding")
        if (
            content.get("transition_source_type")
            != expected_provenance["transition_source_type"]
            or content.get("transition_source_provenance")
            != expected_provenance["transition_source_provenance"]
        ):
            raise ValueError(f"Scenario {i} manifest disagrees on transition source")
        source_rasters = content.get("source_rasters", {})
        target_cfg = cfg["dataset_generation"]["target"]
        raster_dir = Path(cfg["environment"]["raster_dir"])
        if not raster_dir.is_absolute():
            raster_dir = Path(__file__).resolve().parent / raster_dir
        for state_name, config_name in (("state_t", "state_t_file"), ("state_t1", "state_t1_file")):
            recorded = source_rasters.get(state_name, {}).get("path")
            expected_state = Path(target_cfg[config_name])
            if not expected_state.is_absolute():
                expected_state = raster_dir / expected_state
            if not recorded or Path(recorded).resolve() != expected_state.resolve():
                raise ValueError(
                    f"Scenario {i} manifest is not bound to configured {config_name}"
                )
        scenario_manifests.append(
            {
                "path": str(manifest_path),
                "sha256": _file_sha256(manifest_path),
                "content": content,
            }
        )
        print(f"  -> {len(df)} rows, {int(df['Ignited'].sum())} positives")

    if not all_frames:
        print("\nERROR: No scenarios produced data!")
        return ""

    expected_columns = list(METADATA_NAMES) + list(FEATURE_NAMES) + [
        TARGET_NAME,
        "feature_schema_version",
        "target_version",
    ]
    for index, frame in enumerate(all_frames):
        if list(frame.columns) != expected_columns:
            raise ValueError(
                f"Scenario frame {index} does not match the canonical Set C dataset order"
            )
    combined = pd.concat(all_frames, ignore_index=True)

    # Save combined dataset
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / output_csv
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable combined dataset: {out_path}")
    combined_manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    if combined_manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable combined manifest: {combined_manifest_path}"
        )
    base_manifest = scenario_manifests[0]["content"]
    for index, entry in enumerate(scenario_manifests):
        content = entry["content"]
        if content.get("manifest_schema_version") != OBSERVATION_MANIFEST_VERSION:
            raise ValueError(f"Scenario manifest {index} has an unsupported schema version")
        if content.get("artifact_kind") != "set_c_observation_dataset":
            raise ValueError(f"Scenario manifest {index} has the wrong artifact kind")
    for field in (
        "experiment_id",
        "source_revision",
        "source_dirty_state",
        "environment_record",
        "spatial_block",
        "sampling_policy",
        "roi",
        "transition_source_type",
    ):
        values = [entry["content"].get(field) for entry in scenario_manifests]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Scenario manifests disagree on {field}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    combined_manifest = {
        "manifest_schema_version": COMBINED_MANIFEST_VERSION,
        "artifact_kind": "set_c_combined_dataset",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "set_id": "SetC",
        "feature_schema_version": SET_C_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "target_name": TARGET_NAME,
        "target_version": TARGET_VERSION,
        "positive_label": POSITIVE_LABEL,
        "metadata_names": list(METADATA_NAMES),
        "dataset_path": str(out_path),
        "dataset_sha256": _file_sha256(out_path),
        "experiment_id": base_manifest["experiment_id"],
        "source_revision": base_manifest["source_revision"],
        "source_dirty_state": base_manifest["source_dirty_state"],
        "generation_command": base_manifest["generation_command"],
        "environment_record": base_manifest["environment_record"],
        "spatial_block": base_manifest["spatial_block"],
        "source_rasters": [
            entry["content"]["source_rasters"] for entry in scenario_manifests
        ],
        "row_count": int(len(combined)),
        "target_counts": {
            str(key): int(value)
            for key, value in combined[TARGET_NAME].value_counts().sort_index().items()
        },
        "scenario_manifests": scenario_manifests,
        "split_status": "unassigned; split_role is stored in a separate immutable split manifest",
        "duplicate_policy": "sha256_of_canonical_float32_feature_row",
        "sampling_policy": base_manifest["sampling_policy"],
        "roi": base_manifest["roi"],
        "transition_source_type": base_manifest["transition_source_type"],
        "transition_source_provenance": [
            entry["content"]["transition_source_provenance"]
            for entry in scenario_manifests
        ],
        "wind_contract": {
            key: value
            for key, value in WindContract.from_config(scenarios[0]["wind"])
            .to_manifest()
            .items()
            if key not in {"speed_kmh", "direction_deg"}
        },
    }
    combined_manifest_path.write_text(
        json.dumps(combined_manifest, indent=2), encoding="utf-8"
    )

    n_pos = int(combined["Ignited"].sum())
    n_total = len(combined)
    print(f"\n{'='*60}")
    print(f"COMBINED DATASET")
    print(f"{'='*60}")
    print(f"  Total rows: {n_total:,}")
    print(f"  Positives:  {n_pos:,} ({n_pos/n_total*100:.2f}%)")
    print(f"  Saved to:   {out_path}")
    print(f"  Features:   {list(combined.columns)}")

    # Quick wind distribution check
    print(f"\n  Wind speed distribution:")
    for speed, count in combined["wind_speed"].value_counts().sort_index().items():
        print(f"    {speed} km/h: {count:,} rows")
    print(f"\n  Wind direction (sin) unique values: {combined['wind_sin'].nunique()}")

    return str(out_path)


def main():
    config_path = Path(__file__).resolve().parent / "config" / "default_experiment.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    scenarios = build_scenarios(base_config)
    print(f"Built {len(scenarios)} explicitly bound simulated transition scenarios")

    experiment_id = base_config.get("dataset_generation", {}).get("provenance", {}).get(
        "experiment_id"
    )
    if experiment_id in (None, ""):
        raise ValueError("dataset_generation.provenance.experiment_id is required")
    output_csv = f"dataFiles/{experiment_id}_combined.csv"
    run_all_scenarios(scenarios, output_csv)


if __name__ == "__main__":
    main()
