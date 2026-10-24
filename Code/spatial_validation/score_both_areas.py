"""Score one simulation output two ways: whole map, and building cells only.

Whole map is what validate_simulation.py reports. Building-only keeps just the
cells that are mapped buildings in stack_buildings.tif (value > 0) and scores
those, for both the prediction and the ground truth. Roughly 42% of the traced
Sta. Maria burn scar (1,393 of 3,312 cells) lies outside the building mask, so
the two numbers answer different questions:
  - whole map: how well does the simulation reproduce the traced burn scar?
  - buildings only: how well does it predict which mapped buildings burned?

This is an approximation of Kent's RF evaluation domain (his also drops
slope-invalid cells); agree on the exact mask before comparing LR vs RF.

Usage (from Thesis_1/Code with the venv active):
    python spatial_validation/score_both_areas.py --final output/windfix/B_nasa_t040/final_state.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

FIRE_STATES = (3, 4, 5)  # same as validate_simulation.py
CODE = Path(__file__).resolve().parent.parent
RASTERS = CODE / "processedData/raster/raster"


def scores(pred: np.ndarray, truth: np.ndarray) -> dict:
	tp = int(np.sum(pred & truth))
	fp = int(np.sum(pred & ~truth))
	fn = int(np.sum(~pred & truth))
	precision = tp / (tp + fp) if tp + fp else 0.0
	recall = tp / (tp + fn) if tp + fn else 0.0
	f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
	jaccard = tp / (tp + fp + fn) if tp + fp + fn else 0.0
	return dict(precision=precision, recall=recall, f1=f1, jaccard=jaccard,
		tp=tp, fp=fp, fn=fn, truth=int(truth.sum()), predicted=int(pred.sum()))


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--final", required=True, help="Simulation final_state.tif")
	parser.add_argument("--gt", default=str(RASTERS / "stack_ground_truth.tif"))
	parser.add_argument("--buildings", default=str(RASTERS / "stack_buildings.tif"))
	parser.add_argument("--label", default="")
	args = parser.parse_args()

	pred = np.isin(rasterio.open(args.final).read(1), FIRE_STATES)
	truth = rasterio.open(args.gt).read(1) > 0
	building = rasterio.open(args.buildings).read(1) > 0
	if not (pred.shape == truth.shape == building.shape):
		raise SystemExit(f"Shape mismatch: {pred.shape} {truth.shape} {building.shape}")

	rows = {
		"whole map": scores(pred, truth),
		"buildings only": scores(pred & building, truth & building),
	}

	print(f"\n  {args.label or args.final}")
	print(f"  {'area':<16}{'Precision':>10}{'Recall':>9}{'F1':>8}{'Jaccard':>9}{'truth':>8}{'pred':>8}")
	for area, s in rows.items():
		print(f"  {area:<16}{s['precision']:>10.4f}{s['recall']:>9.4f}{s['f1']:>8.4f}{s['jaccard']:>9.4f}{s['truth']:>8,}{s['predicted']:>8,}")


if __name__ == "__main__":
	main()
