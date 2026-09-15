"""Generate training data from multiple fire scenarios with varied wind conditions.

Wraps Kent's SyntheticDatasetGenerator to run the CA simulation under
different wind speeds and directions, producing a richer dataset.

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

import numpy as np
import pandas as pd
import yaml

from dataset_generator import SyntheticDatasetGenerator
from modules.feature_pipeline import FeatureAssembler


def build_scenarios(base_config: dict) -> list[dict]:
    """Create config variants for each wind speed x direction x run combination.

    Wind speeds: 5, 10, 15, 20, 25 km/h
    Wind directions: 0, 45, 90, 135, 180, 225, 270, 315 degrees (8 compass points)
    Runs per scenario: 5 (different seeds = different ignition points)
    Total: 5 x 8 x 5 = 200 scenarios

    Key changes from v1 (40 scenarios):
    - n_ignition_points: 5 -> 10 (more simultaneous fire starts per run)
    - timesteps_to_record: 60 -> 100 (longer burn, more cells visited by fire)
    - runs_per_scenario: 1 -> 5 (different random ignition positions per wind config)
    These changes address the low positive count problem (102 positives from 40 scenarios).
    """
    wind_speeds = [5.0, 10.0, 15.0, 20.0, 25.0]
    wind_directions = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    runs_per_scenario = 5

    scenarios = []
    scenario_id = 0

    for speed in wind_speeds:
        for direction in wind_directions:
            for run in range(runs_per_scenario):
                cfg = copy.deepcopy(base_config)
                cfg["wind"]["speed_kmh"] = speed
                cfg["wind"]["direction_deg"] = direction

                # Different seed per run so ignition points land in different locations
                cfg["simulation"]["seed"] = 42 + scenario_id

                cfg["dataset_generation"]["n_ignition_points"] = 10
                cfg["dataset_generation"]["timesteps_to_record"] = 100

                # Each scenario writes to a temp file (we combine later)
                cfg["dataset_generation"]["output_csv"] = f"dataFiles/temp_scenario_{scenario_id}.csv"

                scenarios.append(cfg)
                scenario_id += 1

    return scenarios


def run_all_scenarios(scenarios: list[dict], output_csv: str) -> str:
    """Run each scenario and combine results into one dataset."""
    all_frames: list[pd.DataFrame] = []
    total = len(scenarios)

    for i, cfg in enumerate(scenarios):
        speed = cfg["wind"]["speed_kmh"]
        direction = cfg["wind"]["direction_deg"]
        seed = cfg["simulation"]["seed"]
        print(f"\n{'='*60}")
        print(f"Scenario {i+1}/{total}: wind={speed} km/h, dir={direction}°, seed={seed}")
        print(f"{'='*60}")

        try:
            generator = SyntheticDatasetGenerator(cfg)
            temp_path = generator.generate()

            df = pd.read_csv(temp_path)
            all_frames.append(df)
            print(f"  -> {len(df)} rows, {int(df['Ignited'].sum())} positives")

            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            print(f"  -> FAILED: {e}")
            continue

    if not all_frames:
        print("\nERROR: No scenarios produced data!")
        return ""

    combined = pd.concat(all_frames, ignore_index=True)

    # Save combined dataset
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

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
    print(f"Built {len(scenarios)} scenarios (5 wind speeds x 8 directions x 5 runs)")

    output_csv = "dataFiles/multi_scenario_dataset.csv"
    run_all_scenarios(scenarios, output_csv)


if __name__ == "__main__":
    main()
