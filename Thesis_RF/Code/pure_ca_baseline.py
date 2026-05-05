"""Standalone deterministic CA baseline without ML model dependency."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol
import yaml

try:
    from modules.data_loader import EnvironmentManager
    from modules.feature_pipeline import FeatureAssembler
except ModuleNotFoundError:
    from Code.modules.data_loader import EnvironmentManager
    from Code.modules.feature_pipeline import FeatureAssembler


STATE_NON_BURNABLE = np.int8(1)
STATE_NOT_YET_BURNED = np.int8(2)
STATE_IGNITED = np.int8(3)
STATE_BLAZING = np.int8(4)
STATE_EXTINGUISHED = np.int8(5)


class RuleBasedAutomata:
    """Deterministic 5-state CA that uses composite flammability as p_ignite baseline."""

    def __init__(self, environment: dict, config: dict, features: FeatureAssembler):
        self.environment = environment
        self.config = config

        self.grid_shape = environment["grid_shape"]
        self.burnable_mask = np.asarray(environment["burnable_mask"], dtype=bool)
        self.building_presence = np.asarray(environment["building_presence"], dtype=np.float32)
        self.transform = environment["transform"]
        self.crs = environment["crs"]

        transition_cfg = config.get("placeholder_transition", {})
        self.t_3_to_4 = int(transition_cfg.get("T_3_to_4", 1))
        self.t_4_to_5 = int(transition_cfg.get("T_4_to_5", 1))
        self.wind_multiplier = 1.0 + float(transition_cfg.get("wind_weight", 0.0))

        # Deterministic analogue of Bernoulli ignition draw: ignite when effective
        # probability clears this fixed threshold.
        default_threshold = float(transition_cfg.get("base_ignition_prob", 0.12))
        self.ignition_threshold = float(transition_cfg.get("deterministic_ignition_threshold", default_threshold))

        self.p_ignite_base = np.clip(
            np.asarray(features.composite_flammability, dtype=np.float32),
            0.0,
            1.0,
        )
        self.p_ignite_base[~self.burnable_mask] = 0.0

        self.grid = np.full(self.grid_shape, STATE_NOT_YET_BURNED, dtype=np.int8)
        self.grid[~self.burnable_mask] = STATE_NON_BURNABLE

        self.ignition_timers = np.zeros(self.grid_shape, dtype=np.int16)
        self.blazing_timers = np.zeros(self.grid_shape, dtype=np.int16)
        self.timestep = 0

    def set_ignition(self, points: list[tuple[int, int]]) -> None:
        rows, cols = self.grid_shape
        for row, col in points:
            if row < 0 or col < 0 or row >= rows or col >= cols:
                print(f"Warning: Ignition point ({row}, {col}) is out of bounds.")
                continue
            if self.grid[row, col] == STATE_NON_BURNABLE:
                print(f"Warning: Ignition point ({row}, {col}) is non-burnable.")
                continue
            if self.grid[row, col] == STATE_NOT_YET_BURNED:
                self.grid[row, col] = STATE_BLAZING

    def _count_blazing_neighbors(self, blazing_mask: np.ndarray) -> np.ndarray:
        padded = np.pad(blazing_mask.astype(np.int8), pad_width=1, mode="constant", constant_values=0)
        return (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ).astype(np.int8)

    def step(self) -> None:
        current_grid = self.grid
        next_grid = current_grid.copy()

        ignited_now = current_grid == STATE_IGNITED
        blazing_now = current_grid == STATE_BLAZING

        self.ignition_timers[ignited_now] += np.int16(1)
        self.blazing_timers[blazing_now] += np.int16(1)

        ignited_to_blazing = ignited_now & (self.ignition_timers >= self.t_3_to_4)
        next_grid[ignited_to_blazing] = STATE_BLAZING
        self.ignition_timers[ignited_to_blazing] = np.int16(0)
        self.blazing_timers[ignited_to_blazing] = np.int16(0)

        blazing_to_extinguished = blazing_now & (self.blazing_timers >= self.t_4_to_5)
        next_grid[blazing_to_extinguished] = STATE_EXTINGUISHED
        self.blazing_timers[blazing_to_extinguished] = np.int16(0)

        blazing_neighbor_count = self._count_blazing_neighbors(blazing_now)
        susceptible = (current_grid == STATE_NOT_YET_BURNED) & (blazing_neighbor_count > 0)

        if np.any(susceptible):
            neighbor_factor = blazing_neighbor_count.astype(np.float32) / np.float32(8.0)
            p_effective = np.clip(self.p_ignite_base * neighbor_factor * self.wind_multiplier, 0.0, 1.0)
            ignite_mask = susceptible & (p_effective >= self.ignition_threshold)
            next_grid[ignite_mask] = STATE_IGNITED
            self.ignition_timers[ignite_mask] = np.int16(0)
            self.blazing_timers[ignite_mask] = np.int16(0)

        self.grid = next_grid
        self.timestep += 1

    def is_active(self) -> bool:
        return bool(np.any((self.grid == STATE_IGNITED) | (self.grid == STATE_BLAZING)))

    def get_state_counts(self) -> dict[str, int]:
        return {
            "non_burnable": int(np.count_nonzero(self.grid == STATE_NON_BURNABLE)),
            "not_yet_burned": int(np.count_nonzero(self.grid == STATE_NOT_YET_BURNED)),
            "ignited": int(np.count_nonzero(self.grid == STATE_IGNITED)),
            "blazing": int(np.count_nonzero(self.grid == STATE_BLAZING)),
            "extinguished": int(np.count_nonzero(self.grid == STATE_EXTINGUISHED)),
        }

    def get_grid(self) -> np.ndarray:
        return self.grid.copy()


def load_config(config_path: str) -> dict:
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)

    code_dir = Path(__file__).resolve().parent
    raster_dir = Path(config["environment"]["raster_dir"])
    if not raster_dir.is_absolute():
        raster_dir = code_dir / raster_dir
    config["environment"]["raster_dir"] = str(raster_dir.resolve())
    return config


def _convert_geographic_ignition_points(
    automata: RuleBasedAutomata,
    ignition_points: list,
) -> list[tuple[int, int]]:
    rows, cols = automata.grid_shape
    converted: list[tuple[int, int]] = []

    for point in ignition_points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(
                "Each ignition point must be [x, y] in the raster CRS (EPSG:32651). "
                f"Received: {point}"
            )

        x = float(point[0])
        y = float(point[1])

        row_idx, col_idx = rowcol(automata.transform, x, y, op=np.floor)
        row = int(row_idx)
        col = int(col_idx)

        col_affine, row_affine = ~automata.transform * (x, y)
        row_affine_idx = int(np.floor(row_affine))
        col_affine_idx = int(np.floor(col_affine))
        if row != row_affine_idx or col != col_affine_idx:
            raise ValueError(
                "Ignition coordinate conversion mismatch between rowcol() and inverse affine. "
                f"Input (x={x:.4f}, y={y:.4f}) -> rowcol=({row}, {col}), "
                f"inverse_affine=({row_affine_idx}, {col_affine_idx})."
            )

        if row < 0 or row >= rows or col < 0 or col >= cols:
            raise ValueError(
                "Ignition coordinate is outside the matrix extent. "
                f"Input (x={x:.4f}, y={y:.4f}) maps to (row={row}, col={col}), "
                f"valid row range is [0, {rows - 1}] and col range is [0, {cols - 1}]."
            )

        converted.append((row, col))

    return converted


def _pick_deterministic_ignition_points(
    automata: RuleBasedAutomata,
    n_points: int = 3,
) -> list[tuple[int, int]]:
    candidate_mask = automata.burnable_mask & (automata.building_presence > 0)
    candidate_cells = np.argwhere(candidate_mask)
    if candidate_cells.size == 0:
        return []

    scores = automata.p_ignite_base[candidate_mask]
    rows = candidate_cells[:, 0]
    cols = candidate_cells[:, 1]
    order = np.lexsort((cols, rows, -scores))

    count = min(n_points, candidate_cells.shape[0])
    selected = candidate_cells[order[:count]]
    return [(int(row), int(col)) for row, col in selected]


def run_simulation(config: dict) -> None:
    sim_cfg = config.get("simulation", {})

    env_manager = EnvironmentManager(config["environment"])
    env_manager.load_rasters()
    env_manager.build_masks()
    env_manager.normalize_layers()
    env_manager.summary()

    environment = env_manager.get_environment()
    features = FeatureAssembler(
        environment,
        config.get("wind", {}),
        config.get("flammability_weights", {}),
    )

    automata = RuleBasedAutomata(environment, config, features)

    ignition_points = sim_cfg.get("ignition_points", [])
    if ignition_points:
        ignition_tuples = _convert_geographic_ignition_points(automata, ignition_points)
    else:
        ignition_tuples = _pick_deterministic_ignition_points(automata, n_points=3)

    if not ignition_tuples:
        raise ValueError("No valid ignition points available from config or deterministic candidate selection.")

    automata.set_ignition(ignition_tuples)
    print(f"Ignition points: {ignition_tuples}")

    max_timesteps = int(sim_cfg.get("max_timesteps", 0))
    early_stop = bool(sim_cfg.get("early_stop", True))

    for step in range(1, max_timesteps + 1):
        automata.step()

        if step % 10 == 0:
            print(f"Step {step}: {automata.get_state_counts()}")

        if early_stop and (not automata.is_active()):
            print(f"Simulation stopped early at step {step}: no active fire cells remain.")
            break

    output_cfg = config.get("output", {})
    output_dir = Path(output_cfg.get("output_dir", "output"))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    final_grid = automata.get_grid().astype(np.int8)
    output_path = output_dir / "final_state_baseline.tif"
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=final_grid.shape[0],
        width=final_grid.shape[1],
        count=1,
        dtype="int8",
        crs=automata.crs,
        transform=automata.transform,
    ) as dst:
        dst.write(final_grid, 1)

    print(f"Final baseline state written to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone deterministic CA baseline (no ML).")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config" / "default_experiment.yaml"),
        help="Path to experiment YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_simulation(config)


if __name__ == "__main__":
    main()
