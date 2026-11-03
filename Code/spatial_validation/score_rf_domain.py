"""Rescore saved LR outputs on the RF-specification domain (no simulation).

Domain = RF crop (1024 x 1024) & four-layer validity (data_loader.build_masks) & buildings == 10,
the domain both sides confirmed by digest on 2026-11-03 (Airbase 215,553 / 178; Sta. Maria 135,992 / 1,910).
Covers the 100-step baselines and the duration-study snapshots. Historical scores stay as they are;
this is a separately labelled column for matched LR-vs-RF comparison.

Usage: .venv/bin/python spatial_validation/score_rf_domain.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "spatial_validation"))
from modules.data_loader import EnvironmentManager  # noqa: E402
from score_both_areas import FIRE_STATES, scores  # noqa: E402

RASTERS = CODE / "processedData/raster/raster"
OUT = CODE / "output/windfix"
SEEDS = [("", 42), ("_s43", 43), ("_s44", 44), ("_s45", 45), ("_s46", 46)]
FIRES = {
	"airbase": dict(gt="stack_ground_truth_airbase.tif", crop=(820, 1989),
		baseline="airbase_B_t040{s}/final_state.tif",
		snapshots={43: "2 min/update", 86: "1 min/update (primary)", 172: "0.5 min/update"}),
	"stamaria": dict(gt="stack_ground_truth.tif", crop=(256, 2176),
		baseline="B_nasa_t040{s}/final_state.tif",
		snapshots={106: "2 min/update", 212: "1 min/update (primary)", 424: "0.5 min/update"}),
}


def main():
	env = EnvironmentManager(dict(raster_dir=str(RASTERS), slope_file="stack_slope_final.tif",
		proximity_file="stack_proximity.tif", buildings_file="stack_buildings.tif",
		materials_file="stack_materials.tif", nodata_value=-9999))
	env.load_rasters()
	env.build_masks()
	valid_b10 = env.burnable_mask & (env.buildings_raw == 10)
	del env

	lines, rows = [], []
	for fire, cfg in FIRES.items():
		truth = rasterio.open(RASTERS / cfg["gt"]).read(1) == 1
		r0, c0 = cfg["crop"]
		dom = np.zeros_like(truth)
		dom[r0:r0 + 1024, c0:c0 + 1024] = True
		dom &= valid_b10
		ref = truth & dom
		lines.append(f"\n{fire}: RF-spec domain {int(dom.sum()):,} cells, {int(ref.sum()):,} positives")
		runs = [("100 steps (paper baseline)", lambda s: OUT / cfg["baseline"].format(s=s))]
		runs += [(f"{h} updates ({clock})", lambda s, h=h: OUT / f"duration/{fire}_B_t040{s}/snapshot_step_{h:04d}.tif")
			for h, clock in cfg["snapshots"].items()]
		for label, path_of in runs:
			sel = []
			for suffix, seed in SEEDS:
				pred = np.isin(rasterio.open(path_of(suffix)).read(1), FIRE_STATES)
				r = scores(pred & dom, ref)
				r.update(fire=fire, setting=label, seed=seed, predicted_outside_domain_buildings=int((pred & (env_b := valid_b10) & ~dom).sum()))
				sel.append(r)
				rows.append(r)
			f1 = np.array([r["f1"] for r in sel])
			rec = np.array([r["recall"] for r in sel])
			prec = np.array([r["precision"] for r in sel])
			lines.append(f"  {label:32s} F1 {f1.mean():.4f} +/- {f1.std(ddof=1):.4f} ({f1.min():.4f}-{f1.max():.4f})  "
				f"R {rec.mean():.4f} +/- {rec.std(ddof=1):.4f}  P {prec.mean():.4f} +/- {prec.std(ddof=1):.4f}  "
				f"per-seed F1 {' / '.join(f'{v:.4f}' for v in f1)}")
	text = "\n".join(lines)
	print(text)
	(OUT / "duration/rf_domain_scores.txt").write_text(text + "\n")
	(OUT / "duration/rf_domain_scores.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
	main()
