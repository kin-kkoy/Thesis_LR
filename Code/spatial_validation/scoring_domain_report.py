"""Write the LR side's scoring-domain report (files, masks, fingerprints, cell counts).

Answers the RF side's request (2026-11-03): which reference and which cells the LR scores use
for Airbase and Sta. Maria, so RF and LR scoring domains can be matched. No simulation.

Usage: .venv/bin/python spatial_validation/scoring_domain_report.py OUT_DIR
Writes OUT_DIR/LR_scoring_domain_report.json and .md
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

CODE = Path(__file__).resolve().parents[1]
RASTERS = CODE / "processedData/raster/raster"
CROP = 1024
FIRES = {
	"airbase": dict(gt="stack_ground_truth_airbase.tif", crop_origin=[820, 1989], ignition_rc=[1332, 2501],
		ignition_utm=[606063.2784, 1140377.2699], crop_note="RF side's Airbase crop (from its LR Airbase report)"),
	"stamaria": dict(gt="stack_ground_truth.tif", crop_origin=[256, 2176], ignition_rc=[892, 2730],  # as used in every LR Sta. Maria run (logs)
		ignition_utm=[606748.78, 1141695.77], crop_note="RF side's F01 crop (from its phase16 config)"),
}
SCRIPTS = ["spatial_validation/score_both_areas.py", "spatial_validation/score_duration.py",
	"spatial_validation/scoring_domain_report.py"]


def sha_file(path: Path) -> str:
	h = hashlib.sha256()
	with open(path, "rb") as fh:
		for chunk in iter(lambda: fh.read(1 << 20), b""):
			h.update(chunk)
	return h.hexdigest()


def sha_mask(mask: np.ndarray) -> str:
	"""SHA-256 of the row-major boolean mask packed to bits (np.packbits, C order)."""
	return hashlib.sha256(np.packbits(np.ascontiguousarray(mask, dtype=bool).ravel()).tobytes()).hexdigest()


def raster_info(path: Path) -> dict:
	with rasterio.open(path) as r:
		arr = r.read(1)
		vals, counts = np.unique(arr, return_counts=True)
		return dict(file=str(path.relative_to(CODE)), sha256=sha_file(path), crs=str(r.crs),
			width=r.width, height=r.height, dtype=str(arr.dtype), nodata=r.nodata,
			transform=list(r.transform)[:6], value_counts={str(v): int(c) for v, c in zip(vals.tolist(), counts.tolist())})


def git(*args) -> str:
	return subprocess.run(["git", *args], cwd=CODE, capture_output=True, text=True).stdout.strip()


def main() -> None:
	out_dir = Path(sys.argv[1]).expanduser()
	out_dir.mkdir(parents=True, exist_ok=True)
	bpath = RASTERS / "stack_buildings.tif"
	buildings = rasterio.open(bpath).read(1) > 0
	shape = buildings.shape
	report = dict(
		generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
		side="LR side",
		purpose="Exact reference and scoring cells used by the LR side for Airbase and Sta. Maria",
		rules=dict(
			prediction="cell counted as burned if final/snapshot state in {3,4,5} (ignited, blazing, extinguished)",
			reference="ground-truth raster value > 0 (rasters are 0/1)",
			building_mask="stack_buildings.tif value > 0; no other exclusion (no slope/nodata/common-valid filter)",
			whole_map_domain="every cell of the 5489 x 6896 grid; no exclusion",
			buildings_domain="building_mask over the whole grid; prediction and reference both ANDed with it",
			crop_buildings_domain="building_mask inside the RF side's 1024 x 1024 crop; reported only to match the RF domain",
			mask_fingerprint="SHA-256 of np.packbits(mask.ravel()) over the full 5489 x 6896 grid, row-major (C order)",
			metrics="TP/FP/FN over the domain; precision, recall, F1, Jaccard (score_both_areas.scores)",
		),
		rasters=dict(buildings=raster_info(bpath)),
		masks=dict(buildings_full_grid=dict(cells=int(buildings.sum()), sha256=sha_mask(buildings))),
		fires={},
		code=dict(repo_head=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
			scripts={s: dict(sha256=sha_file(CODE / s), git_status=git("status", "--short", s) or "committed, unchanged")
				for s in SCRIPTS}),
	)

	for fire, cfg in FIRES.items():
		gpath = RASTERS / cfg["gt"]
		truth = rasterio.open(gpath).read(1) > 0
		assert truth.shape == shape
		r0, c0 = cfg["crop_origin"]
		crop = np.zeros(shape, bool)
		crop[r0:r0 + CROP, c0:c0 + CROP] = True
		domains = {
			"whole_map": np.ones(shape, bool),
			"buildings": buildings,
			"crop_buildings": buildings & crop,
		}
		ref_b = truth & buildings
		rows, cols = np.nonzero(ref_b)
		with rasterio.open(gpath) as g:
			crop_transform = list(g.transform * g.transform.translation(c0, r0))[:6]
		report["fires"][fire] = dict(
			reference=raster_info(gpath),
			ignition=dict(row_col=cfg["ignition_rc"], utm_epsg32651=cfg["ignition_utm"]),
			crop=dict(origin_row_col=cfg["crop_origin"], size=[CROP, CROP], affine_transform=crop_transform,
				note=cfg["crop_note"],
				reference_cells_inside=int((truth & crop).sum()), reference_cells_total=int(truth.sum())),
			reference_cells=dict(total=int(truth.sum()), on_buildings=int(ref_b.sum()),
				off_buildings_excluded=int((truth & ~buildings).sum())),
			domains={name: dict(cells=int(m.sum()), positives=int((truth & m).sum()), mask_sha256=sha_mask(m),
				positive_mask_sha256=sha_mask(truth & m)) for name, m in domains.items()},
			building_positive_cells_row_col=[[int(r), int(c)] for r, c in zip(rows, cols)],
		)

	(out_dir / "LR_scoring_domain_report.json").write_text(json.dumps(report, indent=1))

	b = report["rasters"]["buildings"]
	md = [
		"# LR side: scoring reference and scoring cells (Airbase, Sta. Maria)",
		"",
		f"Generated {report['generated_utc']} by `{SCRIPTS[2]}`. Full data, including every positive building cell (row, col), is in `LR_scoring_domain_report.json`. No simulation was run.",
		"",
		"## Rules",
	]
	md += [f"- **{k.replace('_', ' ')}:** {v}" for k, v in report["rules"].items()]
	md += [
		"",
		"## Shared grid and building raster",
		f"- `{b['file']}`, sha256 `{b['sha256']}`",
		f"- {b['crs']}, {b['width']} x {b['height']} (width x height), dtype {b['dtype']}, nodata {b['nodata']}, transform {b['transform']}",
		f"- Values: {b['value_counts']}",
		f"- Building mask (value > 0), full grid: **{report['masks']['buildings_full_grid']['cells']:,} cells**, mask sha256 `{report['masks']['buildings_full_grid']['sha256']}`",
		"",
	]
	for fire, f in report["fires"].items():
		ref = f["reference"]
		md += [
			f"## {'Airbase' if fire == 'airbase' else 'Sta. Maria'}",
			f"- Reference: `{ref['file']}`, sha256 `{ref['sha256']}`, dtype {ref['dtype']}, values {ref['value_counts']}",
			f"- Reference cells: {f['reference_cells']['total']:,} total; {f['reference_cells']['on_buildings']:,} on buildings; {f['reference_cells']['off_buildings_excluded']:,} excluded from building scoring (off buildings)",
			f"- Ignition: row/col {f['ignition']['row_col']}, UTM {f['ignition']['utm_epsg32651']}",
			f"- Crop (for the crop-buildings domain): origin row/col {f['crop']['origin_row_col']}, 1024 x 1024, transform {f['crop']['affine_transform']}; {f['crop']['reference_cells_inside']:,} of {f['crop']['reference_cells_total']:,} reference cells inside. {f['crop']['note']}.",
			"",
			"| Domain | Cells | Positives | Mask sha256 | Positive-mask sha256 |",
			"|---|---:|---:|---|---|",
		]
		md += [f"| {name} | {d['cells']:,} | {d['positives']:,} | `{d['mask_sha256'][:16]}…` | `{d['positive_mask_sha256'][:16]}…` |"
			for name, d in f["domains"].items()]
		md.append("")
	md += [
		"## Code",
		f"- Repo HEAD `{report['code']['repo_head']}` (branch `{report['code']['branch']}`)",
	]
	md += [f"- `{s}`: sha256 `{v['sha256'][:16]}…` ({v['git_status']})" for s, v in report["code"]["scripts"].items()]
	md += [
		"",
		"## For matching with the RF side",
		"- Building positives can be compared cell by cell using `building_positive_cells_row_col` (global grid row/col) in the JSON.",
		"- Full-length mask hashes are in the JSON. To compare, compute the same hash on the RF domain mask expanded to the full 5489 x 6896 grid: SHA-256 of `np.packbits(mask.ravel())`.",
		"- Known difference: the LR side has no common-valid/slope filter, so its crop-building domain is slightly larger than the RF side's.",
		"- Ignition check: LR Sta. Maria runs ignite at row 892, col 2730 (UTM 606748.78, 1141695.77 falls in that cell). The RF side's F01 local ignition [638, 554] with origin [256, 2176] is global row 894, col 2730, 2 cells (6 m) south. Please confirm which cell the RF side uses. Airbase matches (row 1332, col 2501).",
	]
	(out_dir / "LR_scoring_domain_report.md").write_text("\n".join(md) + "\n")
	print("\n".join(md))


if __name__ == "__main__":
	main()
