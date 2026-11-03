"""Addendum to the LR scoring-domain report: answers the RF side's three follow-up questions (2026-11-03).

Builds an RF-specification scoring domain from the LR side's own files (crop & all-four-layer validity
& buildings == 10), with RF-format mask digests, environment-raster identities, and the ignition history.
No simulation, no rescoring.

Usage: .venv/bin/python spatial_validation/scoring_domain_addendum.py OUT_DIR
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from modules.data_loader import EnvironmentManager  # noqa: E402

RASTERS = CODE / "processedData/raster/raster"
FILES = dict(slope="stack_slope_final.tif", proximity="stack_proximity.tif",
	buildings="stack_buildings.tif", materials="stack_materials.tif")
RF_EXPECTED_INPUT_SHA = dict(  # from RF config phase16_candidate_r2_f01_nasa_hr11_exec_r1.yaml (branch codex/rf-phase16)
	slope="dd61a0da016cd9c0d21d957ffedebd7ce871ef94f9084d5d4feb90e7e32b8482",
	proximity="1b12936cd8bc78d1e638fd31ddf1241bd753cb040cc9db556c623b8193de14b5",
	buildings="da53c3e63b661471b3366fc68249793c6a5a84e0d63bcc96492f60baaf4b83c7",
	materials="21f5c98c7936b4be6e368e7c8c0472a3c045b9c85b1abcb0a0e3fc8c1be0336a")
FIRES = dict(
	airbase=dict(gt="stack_ground_truth_airbase.tif", crop=(820, 1989)),
	stamaria=dict(gt="stack_ground_truth.tif", crop=(256, 2176)),
)
CROP = 1024


def sha_file(path):
	h = hashlib.sha256()
	with open(path, "rb") as fh:
		for chunk in iter(lambda: fh.read(1 << 20), b""):
			h.update(chunk)
	return h.hexdigest()


def rf_mask_sha256(mask):
	"""Same as RF modules/transition_observer.py mask_sha256: sha256(str(shape) | b'|bool|' | uint8 bytes)."""
	a = np.ascontiguousarray(np.asarray(mask, dtype=bool))
	h = hashlib.sha256()
	h.update(str(a.shape).encode("ascii"))
	h.update(b"|bool|")
	h.update(a.view(np.uint8).tobytes())
	return h.hexdigest()


def main():
	out_dir = Path(sys.argv[1]).expanduser()
	out_dir.mkdir(parents=True, exist_ok=True)
	env = EnvironmentManager(dict(raster_dir=str(RASTERS), slope_file=FILES["slope"], proximity_file=FILES["proximity"],
		buildings_file=FILES["buildings"], materials_file=FILES["materials"], nodata_value=-9999))
	env.load_rasters()
	env.build_masks()  # identical rule to the RF data_loader: nodata (-9999) in any layer, or all four == 0
	valid = env.burnable_mask
	b10 = env.buildings_raw == 10

	layers = {}
	for name, fname in FILES.items():
		path = RASTERS / fname
		raw = getattr(env, name + "_raw")
		usable = raw[raw != -9999]
		with rasterio.open(path) as src:
			meta = dict(dtype=src.dtypes[0], nodata=src.nodata, crs=str(src.crs), shape=[src.height, src.width],
				transform=list(src.transform)[:6])
		sha = sha_file(path)
		layers[name] = dict(file=str(path.relative_to(CODE)), sha256=sha, **meta,
			nodata_cells=int((raw == -9999).sum()), nonfinite_cells=int((~np.isfinite(raw)).sum()),
			usable_min=float(usable.min()), usable_max=float(usable.max()),
			matches_rf_expected_sha256=(sha == RF_EXPECTED_INPUT_SHA[name]))
		if name in ("materials", "buildings"):
			vals, counts = np.unique(raw, return_counts=True)
			layers[name]["value_counts"] = {str(v): int(c) for v, c in zip(vals.tolist(), counts.tolist())}

	report = dict(
		generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
		side="LR side",
		purpose="Answers to the RF side's three follow-up questions on the LR scoring-domain report",
		q1_runs_and_domains=dict(
			simulator="LR simulator: Thesis_LR repo master (commit 0fde68f), Code/orchestrator.py + Code/modules/, "
				"Stage B LR models/fire_lr_windweighted_windfix.joblib, proba_threshold 0.40",
			runs={
				"Sta. Maria 100-step baseline": "config/windfix_B_nasa_t040*.yaml, seeds 42-46",
				"Airbase 100-step baseline": "config/windfix_airbase_B_t040*.yaml, seeds 42-46",
				"Duration study (both fires)": "config/duration_{airbase,stamaria}_B_t040*.yaml, seeds 42-46, snapshots",
			},
			domains_actually_used={
				"100-step baselines": "score_both_areas.py: whole map (all cells) and full-grid buildings (value > 0). No crop, no validity filter.",
				"duration study": "score_duration.py: whole map, full-grid buildings, and buildings in the RF crop without the validity "
					"filter (identical to full-grid buildings in every case). No validity filter.",
				"RF-specification domain below": "NOT used for any reported LR score yet.",
			},
			note="The LR simulator only ignites valid cells (burnable_mask = the same four-layer validity), so no LR prediction "
				"falls on an invalid cell. Its full-grid buildings scores differ from RF-domain scores only through reference "
				"positives on invalid building cells (0 for Airbase, 9 for Sta. Maria) and predicted building cells outside the "
				"crop (none in the duration study). Precision, recall and F1 do not depend on TN, so the extra non-positive cells do not change them.",
		),
		q2_rf_spec_domain=dict(
			rule="crop & valid & (buildings == 10) & reference-valid; valid = not (-9999 in slope/proximity/buildings/materials) "
				"and not (all four == 0), computed by the LR side's modules/data_loader.py build_masks (same code as the RF data_loader)",
			digest_format="RF format (transition_observer.mask_sha256): sha256(str(shape) + '|bool|' + uint8 bytes), "
				"on the 1024 x 1024 crop-local mask",
			environment_layers=layers,
			full_grid=dict(valid_cells=int(valid.sum()), building_cells=int(b10.sum()),
				building_cells_invalid=int((b10 & ~valid).sum()),
				buildings_equal_materials_present=bool(np.array_equal(b10, env.materials_raw > 0))),
			fires={},
			needs_evidence=[
				"RF materials raster sha256 21f5c98c... was not observed by the LR side. Every LR-side copy (LR repo, LR sandbox, "
				"response-ref from the RF side, Google Drive) is e2432e59... with classes 0-3 only. Which file is 21f5c98c..., its "
				"value counts, and whether it changes validity or the RF features: NEEDS EVIDENCE from the RF side.",
				"RF environment-valid crop tif (simulation_environment_valid.tif) digests were not compared: NEEDS EVIDENCE "
				"(send the RF environment_sha256 / receiving_sha256 for each crop to compare with the LR digests in this report).",
			],
		),
		q3_ignition=dict(
			airbase=dict(lr_row_col=[1332, 2501], rf_row_col=[1332, 2501], utm=[606063.2784, 1140377.2699], status="match"),
			stamaria=dict(
				lr_row_col=[892, 2730], rf_row_col_reported=[894, 2730], utm=[606748.78, 1141695.77],
				lr_source="UTM 606748.78, 1141695.77 entered in LR config on 2026-05-03 (commit 65e45c5) with the comment "
					"'Sitio Santa Maria, Dec 12 2023 (per Kent)', i.e. supplied by the RF side. Converted with rasterio index "
					"(floor): row = (1144374.7699 - 1141695.77) / 3 = 892.99997 -> 892; col = 2730.0005 -> 2730. The point lies "
					"0.0001 m from the row 892/893 boundary.",
				rf_source="RF crop inventory family p8f01 (894, 2730) and F01 local [638, 554] with origin [256, 2176]. "
					"How 894 was derived from the UTM point: NEEDS EVIDENCE from the RF side.",
				historical="All LR Sta. Maria runs (100-step baselines and duration study) used row 892, col 2730. Not changed.",
				proposal="For any new shared Sta. Maria experiment, agree one cell in advance; until then each side keeps its "
					"historical cell and the 6 m difference is disclosed. No LR config change is proposed here.",
			),
		),
	)

	for fire, cfg in FIRES.items():
		truth = rasterio.open(RASTERS / cfg["gt"]).read(1) == 1
		r0, c0 = cfg["crop"]
		sl = (slice(r0, r0 + CROP), slice(c0, c0 + CROP))
		v, b, t = valid[sl], b10[sl], truth[sl]
		dom = v & b
		dropped = np.argwhere(t & b & ~v) + [r0, c0]
		report["q2_rf_spec_domain"]["fires"][fire] = dict(
			crop_origin=[r0, c0], crop_shape=[CROP, CROP],
			reference_file_sha256=sha_file(RASTERS / cfg["gt"]),
			reference_cells_in_crop=int(t.sum()), reference_cells_total=int(truth.sum()),
			crop_valid_cells=int(v.sum()), crop_building_cells=int(b.sum()), crop_building_cells_invalid=int((b & ~v).sum()),
			domain_cells=int(dom.sum()), domain_positives=int((t & dom).sum()),
			rf_format_digests=dict(environment_valid=rf_mask_sha256(v), building=rf_mask_sha256(b),
				domain=rf_mask_sha256(dom), domain_positives=rf_mask_sha256(t & dom)),
			building_positives_excluded_by_validity_row_col=dropped.tolist(),
		)

	(out_dir / "LR_scoring_domain_addendum.json").write_text(json.dumps(report, indent=1))

	q1, q2, q3 = report["q1_runs_and_domains"], report["q2_rf_spec_domain"], report["q3_ignition"]
	md = ["# LR side: answers to the RF side's scoring-domain questions", "",
		f"Generated {report['generated_utc']} by `spatial_validation/scoring_domain_addendum.py`. Full data in `LR_scoring_domain_addendum.json`. No simulation or rescoring was run.", "",
		"## 1. Which LR simulator, configs and runs; which domain the scores used", f"- Simulator: {q1['simulator']}"]
	md += [f"- {k}: `{v}`" for k, v in q1["runs"].items()]
	md += [f"- **Domain, {k}:** {v}" for k, v in q1["domains_actually_used"].items()]
	md += [f"- {q1['note']}", "", "## 2. RF-specification domain, built from LR files (separately labelled)",
		f"- Rule: {q2['rule']}", f"- Digest format: {q2['digest_format']}", "",
		"| Layer | sha256 | Matches RF config | dtype | nodata | -9999 cells | usable range |", "|---|---|---|---|---|---:|---|"]
	md += [f"| {n} | `{l['sha256'][:16]}…` | {'yes' if l['matches_rf_expected_sha256'] else '**NO**'} | {l['dtype']} | {l['nodata']} | {l['nodata_cells']:,} | {l['usable_min']:g}–{l['usable_max']:g} |"
		for n, l in q2["environment_layers"].items()]
	md += ["", f"- Materials values (LR file): {q2['environment_layers']['materials']['value_counts']}",
		f"- Full grid: {q2['full_grid']['valid_cells']:,} valid cells; {q2['full_grid']['building_cells']:,} building cells, {q2['full_grid']['building_cells_invalid']:,} of them invalid; buildings == materials > 0: {q2['full_grid']['buildings_equal_materials_present']}", "",
		"| Fire | Crop origin | Valid cells | Building cells | Invalid building cells | **Domain cells** | **Domain positives** | Domain digest (RF format) |", "|---|---|---:|---:|---:|---:|---:|---|"]
	for fire, f in q2["fires"].items():
		md.append(f"| {fire} | {f['crop_origin']} | {f['crop_valid_cells']:,} | {f['crop_building_cells']:,} | {f['crop_building_cells_invalid']:,} | **{f['domain_cells']:,}** | **{f['domain_positives']:,}** | `{f['rf_format_digests']['domain'][:16]}…` |")
	md += ["", "Full RF-format digests (environment_valid, building, domain, domain_positives) and the excluded positive cells are in the JSON.", "", "**NEEDS EVIDENCE:**"]
	md += [f"- {x}" for x in q2["needs_evidence"]]
	s = q3["stamaria"]
	md += ["", "## 3. Ignition", "- Airbase: LR and RF both row 1332, col 2501. Match.",
		f"- Sta. Maria, LR: row 892, col 2730. {s['lr_source']}", f"- Sta. Maria, RF: row 894, col 2730. {s['rf_source']}",
		f"- Historical: {s['historical']}", f"- Proposal: {s['proposal']}", ""]
	(out_dir / "LR_scoring_domain_addendum.md").write_text("\n".join(md) + "\n")
	print("\n".join(md))


if __name__ == "__main__":
	main()
