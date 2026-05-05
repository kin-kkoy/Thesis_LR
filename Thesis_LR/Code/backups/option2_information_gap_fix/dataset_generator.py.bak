"""Generate synthetic ML training rows by running CA dynamics on a cropped raster ROI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import convolve

from modules.automata_engine import (
    FireAutomata,
    STATE_BLAZING,
    STATE_IGNITED,
    STATE_NOT_YET_BURNING,
)
from modules.data_loader import EnvironmentManager
from modules.feature_pipeline import FeatureAssembler


class SyntheticDatasetGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.base_dir = Path(__file__).resolve().parent
        self.dataset_cfg = config["dataset_generation"]
        self.roi = self.dataset_cfg["roi"]
        self.n_ignition_points = int(self.dataset_cfg["n_ignition_points"])
        self.timesteps_to_record = int(self.dataset_cfg["timesteps_to_record"])
        self.output_csv = self.dataset_cfg["output_csv"]
        self.flammability_weights = config.get("flammability_weights", {})

        environment_cfg = dict(config["environment"])
        raster_dir = Path(environment_cfg["raster_dir"])
        if not raster_dir.is_absolute():
            environment_cfg["raster_dir"] = str((self.base_dir / raster_dir).resolve())

        self.environment_manager = EnvironmentManager(environment_cfg)
        self.feature_assembler: FeatureAssembler | None = None
        self.rng = np.random.default_rng(int(config["simulation"].get("seed", 42)))

    def _crop_environment(self, env: dict, roi: dict) -> dict:
        row_start = int(roi["row_start"])
        row_end = int(roi["row_end"])
        col_start = int(roi["col_start"])
        col_end = int(roi["col_end"])

        row_slice = slice(row_start, row_end)
        col_slice = slice(col_start, col_end)

        cropped = {
            "slope_risk": env["slope_risk"][row_slice, col_slice],
            "proximity_risk": env["proximity_risk"][row_slice, col_slice],
            "building_presence": env["building_presence"][row_slice, col_slice],
            "material_risk": env["material_risk"][row_slice, col_slice],
            "burnable_mask": env["burnable_mask"][row_slice, col_slice],
            "nodata_mask": env["nodata_mask"][row_slice, col_slice],
            "transform": env["transform"],
            "crs": env["crs"],
        }
        cropped["grid_shape"] = cropped["slope_risk"].shape
        return cropped

    def _choose_ignition_points(self, environment: dict) -> list[tuple[int, int]]:
        eligible_mask = environment["burnable_mask"] & (environment["building_presence"] > 0)
        eligible_indices = np.flatnonzero(eligible_mask.ravel())

        if eligible_indices.size < self.n_ignition_points:
            raise ValueError(
                "Not enough burnable building cells in ROI for ignition points: "
                f"requested {self.n_ignition_points}, available {eligible_indices.size}"
            )

        chosen = self.rng.choice(eligible_indices, size=self.n_ignition_points, replace=False)
        rows, cols = np.unravel_index(chosen, environment["grid_shape"])
        return [(int(r), int(c)) for r, c in zip(rows, cols)]

    def generate(self) -> str:
        self.environment_manager.load_rasters()
        self.environment_manager.build_masks()
        self.environment_manager.normalize_layers()

        full_environment = self.environment_manager.get_environment()
        cropped_environment = self._crop_environment(full_environment, self.roi)

        self.feature_assembler = FeatureAssembler(
            cropped_environment,
            self.config["wind"],
            self.flammability_weights,
        )
        automata = FireAutomata(cropped_environment, self.config)
        automata.set_ignition(self._choose_ignition_points(cropped_environment))

        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8)
        recorded_blocks: list[np.ndarray] = []

        for _ in range(self.timesteps_to_record):
            grid_before = automata.get_grid()

            blazing_neighbor_count = convolve(
                (grid_before == STATE_BLAZING).astype(np.int8),
                kernel,
                mode="constant",
                cval=0,
            )
            grid_features = self.feature_assembler.assemble_grid_features(blazing_neighbor_count)

            # CRITICAL FIX: Only record cells that have at least 1 blazing neighbor. 
            # Cells with 0 neighbors have mathematically 0% chance of ignition, so recording them 
            # introduces 20+ million 'impossible negative' rows that skew disk usage and training.
            candidate_mask = ((grid_before == STATE_NOT_YET_BURNING) & (blazing_neighbor_count > 0)).ravel()
            
            automata.step()
            grid_after = automata.get_grid()

            ignited_now = (
                (grid_before == STATE_NOT_YET_BURNING) & (grid_after == STATE_IGNITED)
            ).ravel()

            if np.any(candidate_mask):
                features_candidates = grid_features[candidate_mask]
                labels_candidates = ignited_now[candidate_mask].astype(np.float32)
                block = np.column_stack([features_candidates, labels_candidates]).astype(np.float32)
                recorded_blocks.append(block)

        if recorded_blocks:
            dataset = np.vstack(recorded_blocks).astype(np.float32)
        else:
            dataset = np.empty((0, len(FeatureAssembler.FEATURE_NAMES) + 1), dtype=np.float32)

        columns = FeatureAssembler.FEATURE_NAMES + ["Ignited"]
        dataframe = pd.DataFrame(dataset, columns=columns)

        output_path = Path(self.output_csv)
        if not output_path.is_absolute():
            output_path = self.base_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataframe.to_csv(output_path, index=False)

        row_count = len(dataframe)
        ignited_fraction = float(dataframe["Ignited"].mean()) if row_count > 0 else 0.0
        print(f"Saved synthetic dataset: {output_path}")
        print(f"Rows recorded: {row_count}")
        print(f"Ignited=1 percentage: {ignited_fraction * 100.0:.2f}%")

        return str(output_path)


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "config" / "default_experiment.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    generator = SyntheticDatasetGenerator(config)
    generator.generate()