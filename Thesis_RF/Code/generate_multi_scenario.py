"""Prepare provenance-bearing Set C simulated transitions for wind scenarios.

This retired binary-raster wrapper retains scenario-contract validation but
fails closed before publication. Active Set C uses the bounded five-state
collector.

Research grounding:
  - Thesis Section 1.4: "evaluates fire spread scenarios under varied wind conditions"
  - Thesis Section 4.6: "sensitivity analysis varying key parameters like wind weighting"
  - Gao et al. 2008: wind velocity and direction directly affect fire spread patterns

Usage:
    python generate_multi_scenario.py
"""

from __future__ import annotations

import copy
from pathlib import Path
import re

import yaml

from dataset_generator import SyntheticDatasetGenerator
from modules.wind_convention import WindContract


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
    """Reject the legacy binary-raster path for authoritative active Set C."""
    raise PermissionError(
        "Legacy multi-scenario binary state publication is non-authoritative for "
        "active Set C. Use the bounded five-state SetCCollector path; production "
        "publication remains separately unauthorized."
    )


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
