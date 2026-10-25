"""Mean +/- std across seeds for the multi-seed wind-fix runs.

For each configuration, scores every available final_state.tif (seed 42 from
run_overnight_validation.sh, seeds 43-46 from run_multiseed.sh) whole-map and
buildings-only, and prints per-seed values plus mean, sample std, min and max.
Missing seeds are reported and left out. Reads outputs only; writes nothing.

Usage (from Thesis_1/Code with the venv active):
    python spatial_validation/aggregate_seeds.py
"""

from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np
import rasterio

from score_both_areas import FIRE_STATES, RASTERS, scores

CODE = Path(__file__).resolve().parent.parent
CONFIGS = {
	"B_nasa_t040": CODE / "output/windfix",
	"B_nasa_t000": CODE / "output/windfix",
	"C_nasa_t000": CODE / "sandbox_kent/output/windfix",
	"C_nasa_t030": CODE / "sandbox_kent/output/windfix",
	"A_nasa_t000": CODE / "sandbox_kent/output/windfix",
}
SEEDS = (42, 43, 44, 45, 46)
METRICS = ("f1", "recall", "precision", "jaccard")


def run_dir(base: str, seed: int) -> str:
	return base if seed == 42 else f"{base}_s{seed}"


def main() -> None:
	truth = rasterio.open(RASTERS / "stack_ground_truth.tif").read(1) > 0
	building = rasterio.open(RASTERS / "stack_buildings.tif").read(1) > 0

	for base, root in CONFIGS.items():
		per_area: dict[str, list[dict]] = {"whole map": [], "buildings only": []}
		used = []
		for seed in SEEDS:
			path = root / run_dir(base, seed) / "final_state.tif"
			if not path.exists():
				print(f"  {base}: seed {seed} missing ({path.relative_to(CODE)})")
				continue
			pred = np.isin(rasterio.open(path).read(1), FIRE_STATES)
			per_area["whole map"].append(scores(pred, truth))
			per_area["buildings only"].append(scores(pred & building, truth & building))
			used.append(seed)

		print(f"\n{'=' * 72}\n  {base}  (seeds {', '.join(map(str, used))}; n={len(used)})\n{'=' * 72}")
		for area, runs in per_area.items():
			if not runs:
				continue
			print(f"  {area}")
			for metric in METRICS:
				vals = [r[metric] for r in runs]
				std = statistics.stdev(vals) if len(vals) > 1 else 0.0
				per_seed = "  ".join(f"{v:.3f}" for v in vals)
				print(f"    {metric:<10} mean {statistics.mean(vals):.4f} ± {std:.4f}  "
					f"(min {min(vals):.4f}, max {max(vals):.4f})   per seed: {per_seed}")
			recall_pass = sum(r["recall"] >= 0.80 for r in runs)
			print(f"    recall >= 0.80 in {recall_pass}/{len(runs)} seeds")


if __name__ == "__main__":
	main()
