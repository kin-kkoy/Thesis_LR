# LR Cutoff 0.40 — Seeds 42–46

Date: **2 November 2026 (Asia/Manila)**. **PASS — one no-cutoff LR regression and five new LR cutoff runs completed. RF was not rerun.** All five cutoff outcomes are reported; no other cutoff was tried.

## Requested rule and implementation

The fixed operational rule is **p < 0.40 → 0; p ≥ 0.40 → p**, followed by the existing `rng.random() < p` ignition roll. The model is unchanged and no fitting or recalibration occurred. The threshold applies to float32 per-update ignition probabilities before sampling, not to the final map. A cell at p=0.40 retains its probability; p=0.75 remains stochastic.

The attachment's suggested `inference_threshold` edits were written for older source. The current engine already implements the required rule in the tested explicit gate mode. These LR runs therefore reuse `inference_mode: stochastic_probability_gate` and `proba_threshold: 0.40`, with the three legacy classification/threshold fields remaining null. A distinct LR-only execution schema binds the exact model, rule, seed and scenario. The CA engine is byte-identical to the one used for the retained RF 0.40 runs; only the orchestrator LR allowance and tests were extended.

The supplied file is retained as `instructions.md` in the evidence directory. The partner's statements that the LR cutoff was fixed in May and improved F1 by +0.046 in their own CA are **partner-reported**, not independently verified or assumed to transfer to these runs. This is an exploratory comparison after prior F01 no-cutoff results were observed.

## No-cutoff regression — PASS before the cutoff runs

The old LR seed-42 simulation settings were rerun once in a fresh output folder; only authorization/run metadata changed. The original outputs were preserved. The regression reproduced **all original metrics**, matrix **[[133261, 821], [136, 1774]]**, and the **exact original final-state TIFF SHA-256**. It was scored and checked before the driver started any cutoff run.

Regression final-state SHA-256: `1706cc63db125e5cfd68aa4547e5ff5fea88feed8b5fc49d8dd75a7d278a266e`. Regression folder: `Thesis_RF/Code/output/phase15/p16-f01-nasa-hr11-lr-s42-regression-r1/`. This control is excluded from the five-seed averages.

## LR cutoff results — VERIFIED

Same domain for every run: **135,992 common-valid mapped-building cells**, with **1,910 reference positives**. All five runs terminated inactive with zero active cells. Burned counts represent raster cells, not distinct buildings; domain and full-crop burned counts agree. Confusion matrices use `[[TN, FP], [FN, TP]]`.

| Seed | F1 | Recall | Precision | Burned cells | Inactive at update | Confusion matrix |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.789321 | 0.928796 | 0.686267 | 2,585 | 149 | `[[133271, 811], [136, 1774]]` |
| 43 | 0.675978 | 0.950262 | 0.524566 | 3,460 | 201 | `[[132437, 1645], [95, 1815]]` |
| 44 | 0.782302 | 0.930366 | 0.674896 | 2,633 | 161 | `[[133226, 856], [133, 1777]]` |
| 45 | 0.666918 | 0.926178 | 0.521060 | 3,395 | 145 | `[[132456, 1626], [141, 1769]]` |
| 46 | 0.662039 | 0.948691 | 0.508418 | 3,564 | 151 | `[[132330, 1752], [98, 1812]]` |

## Five-seed averages and retained comparisons

**Mean ± sample SD** (denominator n−1, n=5), computed from full-precision values. These describe simulation-seed variability within one development incident, not confidence intervals or five independent incidents.

| Model / ignition policy | F1 | Recall | Precision | Burned cells |
|---|---:|---:|---:|---:|
| LR, no cutoff — retained | 0.703169 ± 0.066985 | 0.936859 ± 0.011628 | 0.567480 ± 0.090779 | 3218.0 ± 507.9 |
| LR, cutoff 0.40 — new | 0.715312 ± 0.064599 | 0.936859 ± 0.011628 | 0.583041 ± 0.089334 | 3127.4 ± 477.4 |
| RF, no cutoff — retained | 0.736180 ± 0.078442 | 0.928586 ± 0.028733 | 0.615147 ± 0.107982 | 2945.0 ± 450.5 |
| RF, cutoff 0.40 — retained | 0.748423 ± 0.077230 | 0.928586 ± 0.028733 | 0.632112 ± 0.108076 | 2863.4 ± 431.2 |

LR cutoff minus no-cutoff mean F1: **+0.012142** (+1.214 percentage points), descriptive only.

### Same-seed RF/LR F1 comparison

The RF values below are retained measurements, not newly rerun RF simulations.

| Seed | RF no cutoff | RF cutoff 0.40 | LR no cutoff | LR cutoff 0.40 |
|---:|---:|---:|---:|---:|
| 42 | 0.852327 | 0.860086 | 0.787569 | 0.789321 |
| 43 | 0.663509 | 0.677549 | 0.665079 | 0.675978 |
| 44 | 0.780627 | 0.796868 | 0.763644 | 0.782302 |
| 45 | 0.692833 | 0.708140 | 0.652527 | 0.666918 |
| 46 | 0.691607 | 0.699471 | 0.647027 | 0.662039 |

## Matched controls and verification — PASS

All cutoff RF/LR runs share the same numerical gate and CA engine, crop 1024×1024 at global `[row,col]=[256,2176]`, local ignition `[638,554]`, seeds 42–46, wind 17.7 km/h from 48.825208° grid north, wind weight 0.20, timers 1/1, cap 300 updates with early stopping, raster identities, geometry, masks, canonical 11-feature order and resource policy. The RF release is sigmoid-calibrated; the LR release is uncalibrated and unweighted. Matching the cutoff does not remove that pipeline difference.

The reconstructed LR path/hash and lack of calibration manifest are unchanged. Each LR cutoff effective configuration recovers the corresponding original LR configuration when only gate/schema/authorization/run/scenario metadata are removed. The no-cutoff regression preserves the original seed-42 scenario. Six unique output IDs prevent replacement of prior artifacts.

**560 focused tests passed** across automata, execution-readiness, teacher, observer and wind contracts. The exact attached vector `[0.10, 0.39, 0.40, 0.75]` becomes `[0, 0, 0.40, 0.75]` with the cutoff and remains unchanged without it. Existing below/equal/above-boundary, stochastic-draw and full-grid RNG checks remain covered. Wrong model, manifest, route, cutoff, mode, scenario and seed are rejected.

All six new LR executions exited 0 with no recorded resource-policy violations. The original Windows hard-link no-overwrite publisher was used outside the sandbox. Scoring used unchanged `final_results.py` loading/evaluation functions with each exact, termination-linked TIFF hash registered only in the scoring process. Source/configuration/execution/record/output identities, geometry/domain/reference counts, raster-state totals and metric arithmetic were checked. The regression was checked before five cutoff runs; no failed-run replacement or result selection occurred.

## Evidence and TIFF locations

Repository-relative evidence: `Thesis_RF/Code/output/lr_threshold_sensitivity/20261102-gate040-s42-46-r1/`, containing instructions, preparation/bindings, regression verification, per-run scores/stdout/stderr/exit commands, full-precision `scores.json`, tests and final verification. LR no-cutoff baseline: `Thesis_RF/Code/output/lr_same_kitchen/20261102-seeds42-46-r1/scores.json`. Retained RF cutoff: `Thesis_RF/Code/output/rf_threshold_sensitivity/20261102-gate040-s42-46-r1/scores.json`.

Five new cutoff configs: `Thesis_RF/Code/config/phase16_candidate_r2_f01_nasa_hr11_lr_t040_s{42..46}_exec_r1.yaml`. Regression config: `.../phase16_candidate_r2_f01_nasa_hr11_lr_s42_regression_exec_r1.yaml`. The new LR profile is `phase16_f01_lr_probability_gate_040.v1`; accepted no-cutoff and RF cutoff profiles remain intact.

Each folder below contains `start.json`, `termination.json`, `final_state.tif`, and `simulation_environment_valid.tif` under `Thesis_RF/Code/output/phase15/`:

| Seed | LR cutoff run folder |
|---:|---|
| 42 | `p16-f01-nasa-hr11-lr-t040-s42-r1/` |
| 43 | `p16-f01-nasa-hr11-lr-t040-s43-r1/` |
| 44 | `p16-f01-nasa-hr11-lr-t040-s44-r1/` |
| 45 | `p16-f01-nasa-hr11-lr-t040-s45-r1/` |
| 46 | `p16-f01-nasa-hr11-lr-t040-s46-r1/` |

LR model SHA-256: `41e6f55d3e07fa27a57b42d7d0be0828c72cf87ff6cc92c3834b0c0dfd8d217b`. Scores SHA-256: `1d5ae924b9a5836a7b55609726dc71360f7e9441b21f242509c592142c0b12c7`.

## Interpretation and limits — WARNING

This completes a matched-control comparison of the supplied RF and LR releases with the same 0.40 probability gate in the shared engine. It remains **F01 calibration/development footprint agreement**, not general model-family superiority, independent held-out incident accuracy, probability-calibration proof or physical/arrival-time validation. The approximate satellite-derived reference includes 3,312 positive crop cells; **1,402 (42.33%) are excluded** from the mapped-building scoring domain. The RF is calibrated and the LR is not. Cutoff 0.40 was fixed for this request; no LR or RF threshold sweep was performed. Identical seeds do not establish a formally paired common-random-number statistical design once spread fronts differ.

Preserve the inferred owner-adopted NASA true-49° interpretation, constant ignition-centre conversion to grid-from 48.825208°, nearby NASA point/owner-identified Santa Maria qualifications, HR=11/BFP11:31 adoption, reported screenshot year 2024 versus owner year 2023, and unresolved DSWD13:25 discrepancy. Original train-only LR history and May cutoff history remain partner-reported. No model fitting/recalibration, RF rerun, synthetic final-test access, old-output overwrite, cleanup or Git mutation occurred. Confidence is high in bounded execution, controls and arithmetic; broader scientific validity remains qualified.
