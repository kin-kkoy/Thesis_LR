# LR / RF Same-Kitchen Results — Seeds 42–46

Date: **2 November 2026 (Asia/Manila)**. **PASS — all eight requested new simulations completed and were scored.** The existing RF and LR seed-42 outcomes were retained; neither was rerun. This table now contains five seeds per model.

## Per-seed results — VERIFIED

Same evaluation domain for every run: **135,992 common-valid mapped-building cells**, with **1,910 reference positives**. RF is the frozen sigmoid-calibrated release; LR is the supplied uncalibrated pipeline reconstructed under scikit-learn 1.8.0.

| Seed | RF F1 | RF Recall | RF Precision | LR F1 | LR Recall | LR Precision | RF burned cells | LR burned cells |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.852327 | 0.939791 | 0.779757 | 0.787569 | 0.928796 | 0.683622 | 2,302 | 2,595 |
| 43 | 0.663509 | 0.880105 | 0.532468 | 0.665079 | 0.950262 | 0.511556 | 3,157 | 3,548 |
| 44 | 0.780627 | 0.932461 | 0.671315 | 0.763644 | 0.930366 | 0.647595 | 2,653 | 2,744 |
| 45 | 0.692833 | 0.956545 | 0.543103 | 0.652527 | 0.926178 | 0.503702 | 3,364 | 3,512 |
| 46 | 0.691607 | 0.934031 | 0.549092 | 0.647027 | 0.948691 | 0.490924 | 3,249 | 3,691 |

## Averages across the five seeds

Values are **mean ± sample standard deviation** (denominator `n−1`, `n=5`), computed from full-precision per-run values. These describe seed variability for this one development case; they are not confidence intervals or results from five independent incidents.

| Model | F1 | Recall | Precision | Burned cells |
|---|---:|---:|---:|---:|
| RF | 0.736180 ± 0.078442 | 0.928586 ± 0.028733 | 0.615147 ± 0.107982 | 2945.0 ± 450.5 |
| LR | 0.703169 ± 0.066985 | 0.936859 ± 0.011628 | 0.567480 ± 0.090779 | 3218.0 ± 507.9 |

## Confusion counts and stopping

Confusion matrices use `[[TN, FP], [FN, TP]]`. All ten runs terminated **inactive**, with zero active cells. Burned cells are state-5 raster cells, not counts of distinct buildings; full-crop and scoring-domain burned counts agree for every run. Updates are abstract simulation steps.

| Seed | RF confusion matrix | RF inactive at update | LR confusion matrix | LR inactive at update |
|---:|---|---:|---|---:|
| 42 | `[[133575, 507], [115, 1795]]` | 135 | `[[133261, 821], [136, 1774]]` | 151 |
| 43 | `[[132606, 1476], [229, 1681]]` | 129 | `[[132349, 1733], [95, 1815]]` | 153 |
| 44 | `[[133210, 872], [129, 1781]]` | 163 | `[[133115, 967], [133, 1777]]` | 173 |
| 45 | `[[132545, 1537], [83, 1827]]` | 167 | `[[132339, 1743], [141, 1769]]` | 149 |
| 46 | `[[132617, 1465], [126, 1784]]` | 173 | `[[132203, 1879], [98, 1812]]` | 151 |

## Controls and checks — PASS

For both models: crop 1024×1024 at global `[row,col]=[256,2176]`; local ignition `[638,554]`; wind 17.7 km/h from **48.825208° grid north**; wind weight 0.20; timers `T_3_to_4=1`, `T_4_to_5=1`; maximum 300 updates with inactivity early stopping; `stochastic_probability`; all three classification thresholds `null`. Environmental raster hashes, geometry, masks, feature order, resource policy, model artifacts and the RF calibration manifest were preserved. LR has no calibration manifest.

The new Phase16 guard accepts integer seeds 42–46 only with the matching RF/LR scenario ID. The older 315° Phase15 profile remains seed-42-only. Eight new disabled-on-disk configs were created. Relative to each model's seed-42 effective configuration, only `simulation.seed` and authorization/run/scenario metadata changed. The output root remains `output/phase15`; unique run IDs select separate output folders.

Focused orchestrator tests: **386 passed**. Every new simulation exited 0, with no recorded resource-policy violations. Execution used the existing Windows hard-link publication outside the sandbox, preserving no-overwrite behavior. Source hashes confirm that CA, feature, loader and scoring code match the retained seed-42 runs; the orchestrator allowance changed. Git/source records therefore differ and are retained in each run's binding.

Each new TIFF was scored through unchanged `final_results.py` loading and evaluation functions, with its exact termination-linked output hash registered only within the scoring process. Reference/mask/geometry checks were preserved. Record links, execution bindings, output hashes, domain/reference counts, raster-state totals and metric arithmetic were checked. Seed-42 scores were reused from the identified prior scoring report after checking preservation hashes.

## Evidence and final TIFFs

Repository-relative paths:

- New evidence: `Thesis_RF/Code/output/lr_same_kitchen/20261102-seeds42-46-r1/`: `preparation.json`, `scores.json`, `pytest.log`, `multiseed.py`, and one stdout/stderr/exit record per new run. `scores.json` retains full-precision metrics, input/source/output identities and record hashes.
- Seed-42 evidence: `Thesis_RF/Code/output/lr_same_kitchen/20261102-r1/scores.json`; earlier partner report `docs/LR_SameKitchen_Result.md` is preserved.
- New configs: `Thesis_RF/Code/config/phase16_candidate_r2_f01_nasa_hr11[_lr]_s43_exec_r1.yaml`, with corresponding files for seeds 44–46. Omit `[_lr]` for RF; use `_lr` for LR.
- RF model SHA-256: `78828dd4b2fcc66e9cdfd8ecb4a3ccd18c268478f3b5e6a4374c3286e3903e54`.
- LR model SHA-256: `41e6f55d3e07fa27a57b42d7d0be0828c72cf87ff6cc92c3834b0c0dfd8d217b`.

Each run folder contains `start.json`, `termination.json`, `final_state.tif`, and `simulation_environment_valid.tif`:

| Seed | RF run folder under `Thesis_RF/Code/output/phase15/` | LR run folder under the same root |
|---:|---|---|
| 42 | `p16-f01-nasa-hr11-s42-r1/` | `p16-f01-nasa-hr11-lr-s42-r1/` |
| 43 | `p16-f01-nasa-hr11-s43-r1/` | `p16-f01-nasa-hr11-lr-s43-r1/` |
| 44 | `p16-f01-nasa-hr11-s44-r1/` | `p16-f01-nasa-hr11-lr-s44-r1/` |
| 45 | `p16-f01-nasa-hr11-s45-r1/` | `p16-f01-nasa-hr11-lr-s45-r1/` |
| 46 | `p16-f01-nasa-hr11-s46-r1/` | `p16-f01-nasa-hr11-lr-s46-r1/` |

## Limits — WARNING

This is a matched-control **F01 calibration/development comparison of calibrated RF and uncalibrated LR releases**. Five seeds quantify stochastic variability within this case; they do not establish held-out accuracy, independent incident replication, general model-family superiority or physical/arrival-time validity. Equal seeds do not pair identical random draws with identical cells after spread fronts diverge. LR's original train-only provenance remains partner-reported; this extension did not refit or recalibrate either model or access the synthetic final-test role.

The reference is an approximate satellite-derived footprint. **1,402 of 3,312 reference-positive crop cells (42.33%) are excluded** from the scoring domain. The NASA true-north interpretation remains **INFERRED / owner-adopted**, converted at the ignition pixel centre and applied as a constant crop bearing. Preserve the owner-identified Santa Maria location, nearby NASA query-point qualification, HR=11/BFP11:31 adoption, reported screenshot year2024 versus owner year2023, and unresolved DSWD13:25 discrepancy.

Confidence is high in the recorded runs, matched controls and metric arithmetic. No additional approval is needed for these completed eight runs; no further seeds were run. Project memory records this extension separately from the earlier single-seed result.
