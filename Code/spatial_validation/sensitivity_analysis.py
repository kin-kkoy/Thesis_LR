"""Ground-truth sensitivity analysis (Gap 1.3).

The ground truth was hand-traced from satellite imagery, so its exact
perimeter is uncertain. This script quantifies how much that uncertainty
moves the reported metrics: it dilates and erodes the ground-truth mask by
1..N cells (1 cell = 3 m) and re-validates the same simulation output(s)
against each perturbed variant. The min-max range across variants is the
honest uncertainty band to report in the paper.

Pass several --final rasters (e.g. the five seeds) to get the mean per
variant. Scored whole map and buildings only, same arithmetic as
score_both_areas.py.

Usage (from Thesis_1/Code with the venv active):
    python spatial_validation/sensitivity_analysis.py \
        --final output/windfix/B_nasa_t040/final_state.tif output/windfix/B_nasa_t040_s4*/final_state.tif \
        --max-cells 3
"""

from __future__ import annotations

import argparse

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, binary_erosion

from score_both_areas import FIRE_STATES, RASTERS, scores


def main() -> None:
	parser = argparse.ArgumentParser(description="Dilate/erode ground truth and re-validate")
	parser.add_argument("--final", nargs="+", required=True, help="Simulation final_state raster(s) (CA states 1-5)")
	parser.add_argument("--gt", default=str(RASTERS / "stack_ground_truth.tif"))
	parser.add_argument("--buildings", default=str(RASTERS / "stack_buildings.tif"))
	parser.add_argument("--max-cells", type=int, default=3, help="Max dilation/erosion in cells (default 3)")
	args = parser.parse_args()

	truth = rasterio.open(args.gt).read(1) > 0
	building = rasterio.open(args.buildings).read(1) > 0
	preds = [np.isin(rasterio.open(f).read(1), FIRE_STATES) for f in args.final]
	for p in preds:
		if not (p.shape == truth.shape == building.shape):
			raise SystemExit(f"Shape mismatch: {p.shape} {truth.shape} {building.shape}")

	variants = [("original", truth)]
	for n in range(1, args.max_cells + 1):
		variants.append((f"erode {n} ({n*3} m)", binary_erosion(truth, iterations=n)))
		variants.append((f"dilate {n} ({n*3} m)", binary_dilation(truth, iterations=n)))

	print(f"Simulation outputs: {len(preds)} (values are means across them)")
	print(f"{'':<16}{'------------ whole map ------------':>44}   {'------- buildings only -------':>34}")
	print(f"{'Variant':<16}{'GT cells':>10}{'Prec':>8}{'Recall':>8}{'F1':>8}{'Jacc':>8}   "
		f"{'GT cells':>10}{'Prec':>8}{'Recall':>8}{'F1':>8}")
	results = []
	for name, gt in variants:
		whole = [scores(p, gt) for p in preds]
		bldg = [scores(p & building, gt & building) for p in preds]
		mean = lambda rows, k: float(np.mean([r[k] for r in rows]))
		results.append((name, mean(whole, "f1"), mean(whole, "recall"), mean(bldg, "f1"), mean(bldg, "recall")))
		print(f"{name:<16}{int(gt.sum()):>10,}{mean(whole, 'precision'):>8.4f}{mean(whole, 'recall'):>8.4f}"
			f"{mean(whole, 'f1'):>8.4f}{mean(whole, 'jaccard'):>8.4f}   {int((gt & building).sum()):>10,}"
			f"{mean(bldg, 'precision'):>8.4f}{mean(bldg, 'recall'):>8.4f}{mean(bldg, 'f1'):>8.4f}")

	print("\nUncertainty band across all variants:")
	for label, i in (("whole-map F1", 1), ("whole-map Recall", 2), ("buildings F1", 3), ("buildings Recall", 4)):
		vals = [r[i] for r in results]
		print(f"  {label:<18} {min(vals):.3f} - {max(vals):.3f}")


if __name__ == "__main__":
	main()
