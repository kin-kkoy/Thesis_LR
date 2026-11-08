# RF Probability Gate 0.40 — Seeds 42–46

Date: **2 November 2026 (Asia/Manila)**. **PASS — five new RF simulations completed and scored.** This is an explicitly requested exploratory inference-policy sensitivity experiment. The frozen RF model was not retrained or recalibrated, and the cutoff was not searched or optimized in this experiment.

## Exact rule

For each susceptible cell, obtain the model's positive-class probability `p`, stored as float32. Apply:

```text
p_effective = 0                 if p < 0.40
p_effective = p                 if p >= 0.40
ignite if rng.random() < p_effective
```

The threshold is inclusive for retaining probabilities: **p=0.40 is retained**. A cell with p=0.75 still goes through the ordinary random draw; it is not forced to ignite. The full-grid RNG draw remains in its original place. `ml_model.proba_threshold=0.40` is the operational gate; the older classification/reporting threshold fields remain null.

This matches the user-supplied LR gating formula, also observed read-only at `Thesis_LR/Code/sandbox_kent/modules/automata_engine.py:296–303`. The statement that 0.40 is the partner's operational Stage B cutoff fixed in May is **partner-reported**, not independently established training/history evidence. The 0.30 sensitivity variant was not used.

## Results — VERIFIED

All runs use the same **135,992 common-valid mapped-building cells**, including **1,910 reference-positive cells**. Burned counts are raster cells, not distinct buildings. All five new runs terminated inactive, with no active cells; their domain and full-crop burned counts agree.

| Seed | RF F1, no gate (retained) | RF F1, gate 0.40 | Gate recall | Gate precision | Gate Jaccard | Gate burned cells | Inactive after updates |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.852327 | 0.860086 | 0.939791 | 0.792845 | 0.754519 | 2,264 | 135 |
| 43 | 0.663509 | 0.677549 | 0.880105 | 0.550786 | 0.512344 | 3,052 | 195 |
| 44 | 0.780627 | 0.796868 | 0.932461 | 0.695703 | 0.662328 | 2,560 | 161 |
| 45 | 0.692833 | 0.708140 | 0.956545 | 0.562154 | 0.548155 | 3,250 | 157 |
| 46 | 0.691607 | 0.699471 | 0.934031 | 0.559072 | 0.537835 | 3,191 | 173 |

### Five-seed averages

**Mean ± sample SD**, denominator `n−1`, `n=5`, calculated from full-precision run metrics. This is seed variability for one incident; it is not a confidence interval or five independent incident evaluations.

| RF inference policy | F1 | Recall | Precision | Burned cells |
|---|---:|---:|---:|---:|
| Original stochastic, no gate | 0.736180 ± 0.078442 | 0.928586 ± 0.028733 | 0.615147 ± 0.107982 | 2945.0 ± 450.5 |
| Stochastic with 0.40 gate | 0.748423 ± 0.077230 | 0.928586 ± 0.028733 | 0.632112 ± 0.108076 | 2863.4 ± 431.2 |

Gate minus no-gate mean F1: **+0.012242** (+1.224 percentage points), descriptive only. No statistical significance claim is made.

### Gate-run confusion matrices

Orientation: `[[TN, FP], [FN, TP]]`.

| Seed | Matrix |
|---:|---|
| 42 | `[[133613, 469], [115, 1795]]` |
| 43 | `[[132711, 1371], [229, 1681]]` |
| 44 | `[[133303, 779], [129, 1781]]` |
| 45 | `[[132659, 1423], [83, 1827]]` |
| 46 | `[[132675, 1407], [126, 1784]]` |

## Controls, implementation and checks — PASS

Only the inference gate and experiment identifiers changed relative to each seed-matched RF baseline. Fixed controls: crop 1024×1024 at global origin `[row,col]=[256,2176]`; local ignition `[638,554]`; wind 17.7 km/h from 48.825208° relative to grid north; wind weight 0.20; timers 1/1; maximum 300 updates; inactivity early stop; same environmental raster identities, masks, geometry, canonical 11-feature schema, resource policy and frozen sigmoid-calibrated RF release.

The new `phase16_f01_rf_probability_gate_040.v1` schema uses the explicit `stochastic_probability_gate` mode and binds the exact 0.40 cutoff, float32 boundary, unchanged sampling, model/manifest and per-seed scenario. Accepted no-gate configurations remain intact. CA source changed only to support this opt-in policy and reject its use in the model-free teacher; the orchestrator changed to validate and bind the separate route. Other bound feature/loader/scoring source hashes and package versions match the retained baselines. This is a new exploratory branch of the D-010 stochastic method, not a silent change to the accepted method.

**524 targeted tests passed** across automata, execution readiness, model-free teacher, transition-observer and wind contracts. Synthetic cases test the float32 values immediately below, equal to, and above 0.40 and p=0.75, stochastic acceptance/rejection, unchanged full-grid RNG call/count, unchanged no-gate seeded behavior and rejection of wrong cutoff/model/manifest/schema/mode/seed/scenario. The float32 comparison matches the partner's supplied NumPy formula under the installed NumPy 2.4 environment; see [NumPy scalar-promotion rules](https://numpy.org/doc/2.4/reference/arrays.promotion.html).

Each new run exited 0 with no recorded resource-policy violation. The existing no-overwrite Windows hard-link publisher was preserved and executed outside the sandbox. Unchanged `final_results.py` loader/evaluator functions scored each new exact, termination-linked TIFF identity registered only within the scoring process. Record/configuration/execution bindings, output hashes, geometry/mask/domain/reference counts, raster-state totals and metric arithmetic were checked. Old no-gate outputs and scores were reused and preserved; they were not rerun.

## Seed 42 and fair reporting

D-121 records seed 42 for the earlier 315° engineering case. D-122 froze seed 42 for the 49° case before its D-139 execution and D-156 score. This supports prespecification relative to the 49° F1 result; the inspected records do not establish why 42 was originally selected or rule out all earlier seed exploration. In the subsequent five-seed no-gate comparison,42 happened to have the highest F1 for RF and LR. Reporting every seed and mean±SD avoids presenting that single outcome as typical.

These five seeds vary the **simulation RNG**, not the trained forest: the same frozen estimator is used throughout. The gate preserves RNG implementation and draw placement, but altered spread fronts can change later features and draw consumption. Equal seed numbers do not by themselves establish a formally paired common-random-number experiment.

**Comparison boundary:** the LR runs previously executed in this shared RF engine had no gate. This report adds five RF 0.40 runs only. The partner's separate LR Stage B result is not a newly measured LR 0.40 result in this same engine, so this report does not claim a completed matched-engine RF-versus-LR comparison with both gates enabled.

## Evidence and final TIFFs

Paths are repository-relative. New evidence: `Thesis_RF/Code/output/rf_threshold_sensitivity/20261102-gate040-s42-46-r1/`, including `experiment.py`, `preparation.json`, `scores.json`, `pytest.log`, logs/exit commands and final verification manifests. Full-precision metrics and record/input/source hashes are in `scores.json`. Retained five-seed baseline: `Thesis_RF/Code/output/lr_same_kitchen/20261102-seeds42-46-r1/scores.json`.

New configs are `Thesis_RF/Code/config/phase16_candidate_r2_f01_nasa_hr11_rf_gate040_s{42..46}_exec_r1.yaml`, each disabled on disk and requiring explicit bound execution. Each new run folder under `Thesis_RF/Code/output/phase15/` contains `start.json`, `termination.json`, `final_state.tif`, and `simulation_environment_valid.tif`:

| Seed | Run folder |
|---:|---|
| 42 | `p16-f01-nasa-hr11-rf-gate040-s42-r1/` |
| 43 | `p16-f01-nasa-hr11-rf-gate040-s43-r1/` |
| 44 | `p16-f01-nasa-hr11-rf-gate040-s44-r1/` |
| 45 | `p16-f01-nasa-hr11-rf-gate040-s45-r1/` |
| 46 | `p16-f01-nasa-hr11-rf-gate040-s46-r1/` |

Scores JSON SHA-256: `153760a5cb8bdb0d5e62eb12a2f2afe73e91f294691b2178281001768aea20f4`. Frozen RF model SHA-256: `78828dd4b2fcc66e9cdfd8ecb4a3ccd18c268478f3b5e6a4374c3286e3903e54`.

## Limits — WARNING

F01 remains calibration/development evidence using an approximate satellite-derived footprint. Of 3,312 reference-positive crop cells, **1402 (42.33%) are excluded** from the scoring domain; 1,910 are included. This is not held-out accuracy, a general model-family advantage or physical/arrival-time validation. The cutoff is partner-supplied and fixed for this sensitivity experiment, not demonstrated optimal for RF.

Preserve inferred/owner-adopted NASA true 49° conversion to constant grid-from 48.825208° at the ignition pixel centre, nearby query-point and owner-identified Santa Maria qualifications, HR=11/BFP 11:31 adoption, reported screenshot year 2024 versus owner year 2023 and unresolved DSWD 13:25 discrepancy. No model fitting, recalibration, synthetic final-test access, threshold sweep, old-output replacement or Git mutation occurred. Confidence is high in the bounded execution/controls/metric arithmetic; broader scientific validity remains qualified.
