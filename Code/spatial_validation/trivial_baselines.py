"""Trivial baselines (Gap 2.2): the floor the ML system must beat.

Two baselines that use no simulation and no learning:
  1. Circle: predict "fire" for every cell within radius R of the ignition
     point, for several R values.
  2. Random: predict each cell as fire independently with probability equal
     to the ground truth's positive rate (sanity-check floor).

If a plain circle scores close to the full CA+ML system, the system isn't
adding much -- reviewers want to see this floor reported either way.
Note the best circle radius is picked by looking at the ground truth, so it
is an optimistic floor, not a fair competitor.

Every baseline is scored whole map and buildings only, with the same
arithmetic as score_both_areas.py.

Usage (from Thesis_1/Code with the venv active):
    python spatial_validation/trivial_baselines.py \
        --ignition 606748.78 1141695.77 \
        --radii 50 75 100 125 150 200 300
"""

from __future__ import annotations

import argparse

import numpy as np
import rasterio

from score_both_areas import RASTERS, scores


def report(name: str, pred: np.ndarray, truth: np.ndarray, building: np.ndarray) -> None:
	whole = scores(pred, truth)
	bldg = scores(pred & building, truth & building)
	print(f"{name:<22}{whole['predicted']:>9,}"
		f"{whole['precision']:>8.4f}{whole['recall']:>8.4f}{whole['f1']:>8.4f}{whole['jaccard']:>8.4f}"
		f"   {bldg['precision']:>8.4f}{bldg['recall']:>8.4f}{bldg['f1']:>8.4f}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Trivial baseline metrics against a ground truth")
	parser.add_argument("--gt", default=str(RASTERS / "stack_ground_truth.tif"))
	parser.add_argument("--buildings", default=str(RASTERS / "stack_buildings.tif"))
	parser.add_argument("--ignition", nargs=2, type=float, required=True,
		metavar=("X", "Y"), help="Ignition point in EPSG:32651 metres")
	parser.add_argument("--radii", nargs="+", type=float, default=[50, 75, 100, 125, 150, 200, 300],
		help="Circle radii in metres")
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	with rasterio.open(args.gt) as src:
		truth = src.read(1) > 0
		row0, col0 = src.index(*args.ignition)
		pixel = src.transform.a  # 3.0 m
	building = rasterio.open(args.buildings).read(1) > 0
	if truth.shape != building.shape:
		raise SystemExit(f"Shape mismatch: {truth.shape} vs {building.shape}")

	h, w = truth.shape
	if not (0 <= row0 < h and 0 <= col0 < w):
		raise SystemExit(f"Ignition point maps to pixel ({row0},{col0}) outside the {h}x{w} grid.")

	rows, cols = np.ogrid[:h, :w]
	dist2 = (rows - row0) ** 2 + (cols - col0) ** 2  # squared pixel distances

	print(f"Ignition pixel: (row {row0}, col {col0}); ground truth fire cells: {int(truth.sum()):,} "
		f"({int((truth & building).sum()):,} on buildings)\n")
	print(f"{'':<31}{'--------- whole map ---------':>32}   {'---- buildings only ----':>24}")
	print(f"{'Baseline':<22}{'Pred':>9}{'Prec':>8}{'Recall':>8}{'F1':>8}{'Jacc':>8}   {'Prec':>8}{'Recall':>8}{'F1':>8}")

	for r in args.radii:
		r_px = r / pixel
		report(f"circle R={r:.0f} m", dist2 <= r_px * r_px, truth, building)

	rng = np.random.default_rng(args.seed)
	p_rate = truth.mean()
	report(f"random p={p_rate:.4%}", rng.random(truth.shape) < p_rate, truth, building)

	print("\nCompare against the operational system (Stage B + t=0.40, NASA wind, seeds 42-46): "
		"whole-map F1 0.634 +/- 0.010, Recall 0.913, Precision 0.486; buildings-only F1 0.679.")


if __name__ == "__main__":
	main()
