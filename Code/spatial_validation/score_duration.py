"""Score the duration-study snapshots (stopping protocol frozen with the RF side, 2026-11-03).

For each fire x seed x horizon: whole map, buildings (whole map), and buildings inside the
RF side's crop window (to compare with the RF simulator's buildings-in-crop domain).
Reached = states 3/4/5 (cumulative footprint). Active = states 3/4 at the snapshot.

Usage: .venv/bin/python spatial_validation/score_duration.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_both_areas import FIRE_STATES, scores  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
RASTERS = CODE / "processedData/raster/raster"
OUT = CODE / "output/windfix/duration"
SEEDS = ["", "_s43", "_s44", "_s45", "_s46"]
SEED_NAMES = [42, 43, 44, 45, 46]

FIRES = {
	# horizon (updates) -> clock label (min per update)
	"airbase": dict(gt="stack_ground_truth_airbase.tif", crop=(820, 1989),
		horizons={43: "2 min/update", 86: "1 min/update (primary)", 172: "0.5 min/update"}),
	"stamaria": dict(gt="stack_ground_truth.tif", crop=(256, 2176),
		horizons={106: "2 min/update", 212: "1 min/update (primary)", 424: "0.5 min/update"}),
}
CROP = 1024


def main() -> None:
	buildings = rasterio.open(RASTERS / "stack_buildings.tif").read(1) > 0
	rows = []
	lines = []
	for fire, cfg in FIRES.items():
		truth = rasterio.open(RASTERS / cfg["gt"]).read(1) == 1
		r0, c0 = cfg["crop"]
		crop = np.zeros_like(truth)
		crop[r0:r0 + CROP, c0:c0 + CROP] = True
		crop_b = crop & buildings
		lines.append(f"\n{fire}: truth {int(truth.sum())} cells, {int((truth & buildings).sum())} on buildings; "
			f"RF-side crop origin {cfg['crop']}: {int(crop_b.sum())} building cells, "
			f"{int((truth & crop_b).sum())} positives")
		for seed, seed_name in zip(SEEDS, SEED_NAMES):
			run = OUT / f"{fire}_B_t040{seed}"
			term_path = run / "termination.json"
			if not term_path.exists():
				lines.append(f"  seed {seed_name}: MISSING")
				continue
			term = json.loads(term_path.read_text())
			snaps = {s["requested_step"]: s for s in term["snapshots"]}
			for horizon, clock in cfg["horizons"].items():
				grid = rasterio.open(run / f"snapshot_step_{horizon:04d}.tif").read(1)
				pred = np.isin(grid, FIRE_STATES)
				whole = scores(pred, truth)
				bld = scores(pred & buildings, truth & buildings)
				crp = scores(pred & crop_b, truth & crop_b)
				s = snaps[horizon]
				rows.append(dict(fire=fire, seed=seed_name, horizon=horizon, clock=clock,
					actual_step=s["actual_step"], status=s["status"], active=s["active_cells"],
					reached=s["reached_cells"], whole=whole, buildings=bld, crop_buildings=crp))

	for fire, cfg in FIRES.items():
		for horizon, clock in cfg["horizons"].items():
			sel = [r for r in rows if r["fire"] == fire and r["horizon"] == horizon]
			if not sel:
				continue
			lines.append(f"\n{fire}, {horizon} updates ({clock}), n={len(sel)}")
			lines.append("  seed  stop  status                 reached  active | whole F1   R      P    "
				"| bld F1    R      P    | crop-bld F1  R      P")
			for r in sel:
				w, b, c = r["whole"], r["buildings"], r["crop_buildings"]
				lines.append(f"  {r['seed']}  {r['actual_step']:4d}  {r['status']:21s} {r['reached']:8,d} {r['active']:6,d} | "
					f"{w['f1']:.4f} {w['recall']:.3f} {w['precision']:.3f} | "
					f"{b['f1']:.4f} {b['recall']:.3f} {b['precision']:.3f} | "
					f"{c['f1']:.4f}  {c['recall']:.3f} {c['precision']:.3f}")
			for area in ("whole", "buildings", "crop_buildings"):
				for metric in ("f1", "recall", "precision"):
					v = np.array([r[area][metric] for r in sel])
					sd = v.std(ddof=1) if len(v) > 1 else float("nan")
					lines.append(f"  {area:15s} {metric:9s} {v.mean():.4f} +/- {sd:.4f} ({v.min():.4f}-{v.max():.4f})")
			v = np.array([r["reached"] for r in sel])
			lines.append(f"  reached cells   {v.mean():,.0f} +/- {v.std(ddof=1) if len(v) > 1 else 0:,.0f}")

	text = "\n".join(lines)
	print(text)
	(OUT / "duration_scores.txt").write_text(text + "\n")
	(OUT / "duration_scores.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
	main()
