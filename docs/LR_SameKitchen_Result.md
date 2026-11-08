# LR_SameKitchen_Result

Run date: **2 November 2026 (Asia/Manila)**. Status: **PASS — one seed-42 LR simulation completed and scored.**

Kristian's supplied LR was rebuilt under scikit-learn 1.8.0 and run in Kent's Phase 16 F01 setup using the same controls as the retained NASA 49° RF run. LR reached inactivity after **151 simulation updates**, with **2,595 burned cells**. Its building-domain F1 was **0.787569**.

## Results — VERIFIED

Both predictions were scored with the unchanged `final_results.py` loader and evaluation functions against the same approximate reference mask. The evaluation domain contains **135,992 common-valid mapped-building cells**, including **1,910 reference-positive cells**.

| Measure | LR, uncalibrated — new run | RF, sigmoid-calibrated — retained 49° run |
|---|---:|---:|
| True negatives (TN) | 133,261 | 133,575 |
| False positives (FP) | 821 | 507 |
| False negatives (FN) | 136 | 115 |
| True positives (TP) | 1,774 | 1,795 |
| Precision | 0.683622 | 0.779757 |
| Recall | 0.928796 | 0.939791 |
| F1 | **0.787569** | **0.852327** |
| Jaccard / reference IoU | 0.649579 | 0.742656 |
| Accuracy | 0.992963 | 0.995426 |
| Predicted burned cells | **2,595** | **2,302** |
| Update count at inactivity | **151** | **135** |
| Active cells at termination | 0 | 0 |

Confusion matrices use `[[TN, FP], [FN, TP]]`:

```text
LR: [[133261, 821], [136, 1774]]
RF: [[133575, 507], [115, 1795]]
```

LR produced 293 more burned cells, 314 more false positives, and 21 more false negatives. LR minus RF F1 was **−0.064757** (−6.476 percentage points). Burned-cell counts agree between the primary scoring domain and the full crop in both runs; these are raster cells, not distinct buildings.

## Matched simulation settings — VERIFIED

| Control | Setting for both runs |
|---|---|
| Crop | 1024 × 1024; global `[row, col]` origin `[256, 2176]` |
| Ignition | Local `[row, col] = [638, 554]` |
| Seed | 42 |
| Wind | 17.7 km/h; meteorological-from **48.825208° relative to grid north** |
| Wind source interpretation | NASA 49° relative to true north, converted at the ignition pixel centre |
| Wind weight | 0.20 |
| Transition timers | `T_3_to_4 = 1`, `T_4_to_5 = 1` |
| Maximum updates / stopping | 300; early stop when inactive |
| Model inference | `stochastic_probability` |
| Thresholds | `threshold`, `inference_threshold`, `classification_threshold` all `null` |
| Feature contract | Canonical 11-feature Set C order; positive class 1 |
| Inputs / masks | Same environmental raster hashes, crop geometry, input policy and environmental-validity mask |

Comparison with the retained RF run's effective configuration found exactly seven changed fields: model path/hash, two calibration-manifest fields changed to `null`, and authorization/run/scenario identifiers. The output root remains `output/phase15`; the distinct LR run directory prevents RF output replacement. All other effective configuration values match.

CA, feature and loader source hashes and recorded package versions match the RF run. The orchestrator has a narrow allowance for this exact rebuilt LR path/hash, a fixed LR scenario ID, and no calibration manifest. The existing RF release checks remain in place. The recorded Git revisions differ (RF `66d08e0e7eb1951aa00ed26582d2dce6ee7f88f0`; LR `734b42a07e3d1f3880a58b74da7d680d3d7ec6c0` with working-tree edits recorded).

## LR reconstruction and validation — PASS

The supplied `rebuild_lr.py` and parameter JSON were copied into a fresh RF evidence directory and executed there. The partner's original folder was preserved. Reconstruction restored the exported scaler statistics, coefficients, intercept and classes; **no fitting or recalibration was performed**. The original scikit-learn 1.9 joblib was not loaded. Cross-version pickle loading is unsupported by [scikit-learn's persistence guidance](https://scikit-learn.org/stable/model_persistence.html).

The rebuilt pipeline exposes the canonical feature names, `classes_ = [0, 1]`, and `predict_proba`. Validation on the identified role reproduced every supplied four-decimal metric:

| Validation measure | Recomputed value | Partner's rounded expectation |
|---|---:|---:|
| Average precision | 0.8330769072 | 0.8331 |
| ROC-AUC | 0.7439479076 | 0.7439 |
| Brier score | 0.1921094102 | 0.1921 |
| Log loss | 0.5643243213 | 0.5643 |

Validation contained 136,341 rows, including 87,258 positives. Pipeline probabilities matched the directly evaluated exported scaling/logistic formula with maximum observed error **0.0** on those rows. Float32 inference, as used by the CA feature pipeline, also returned finite probabilities within `[0, 1]`.

**Partner-reported / NEEDS VERIFICATION:** original fitting used only the 227,576-row train role, unweighted L2 logistic regression (`C=1`, `lbfgs`, `max_iter=2000`, `random_state=42`), without calibration. The export records 227,576 scaler samples and reconstruction is numerically verified, but the original training history and exclusion of other roles were not independently audited. The validation probability metrics above are separate from final-footprint metrics.

## Reproducibility and evidence

Paths below are relative to the repository root.

- LR configuration: `Thesis_RF/Code/config/phase16_candidate_r2_f01_nasa_hr11_lr_exec_r1.yaml`
- LR artifacts: `Thesis_RF/Code/output/phase15/p16-f01-nasa-hr11-lr-s42-r1/` (`start.json`, `termination.json`, `final_state.tif`, `simulation_environment_valid.tif`).
- RF comparison artifacts: `Thesis_RF/Code/output/phase15/p16-f01-nasa-hr11-s42-r1/`. RF simulation was retained; it was not rerun.
- Rebuild, preparation, execution log and scoring evidence: `Thesis_RF/Code/output/lr_same_kitchen/20261102-r1/`, particularly `rebuild_verification.json`, `run_preparation.json`, `simulation_exit.json`, `scores.json`, and retained preparation/scoring scripts.

| Identity | SHA-256 |
|---|---|
| Rebuilt LR model | `41e6f55d3e07fa27a57b42d7d0be0828c72cf87ff6cc92c3834b0c0dfd8d217b` |
| Supplied parameter JSON | `8e45dff6c6ea96153406f9a0bff460a7ea690513eed8f0685fab9f0422f112a6` |
| Validation CSV | `6bdd0f1855360c5bb4ff6af50cce774bb46e3a2a5583d4717742d4ee336e2079` |
| LR execution binding | `2be8c2fb424f036c065a14898a79a6dbd1469ad0086aebea2273203d31f6c20c` |
| LR final-state TIFF | `1706cc63db125e5cfd68aa4547e5ff5fea88feed8b5fc49d8dd75a7d278a266e` |

The execution exited 0 with no recorded resource-policy violations. Focused orchestrator verification passed **350 tests**. Windows hard-link publication required execution outside the sandbox; the publication policy was preserved. Scoring registered the exact new LR output fingerprint in the scoring process, then reused the unchanged `final_results.py` input checks and metric functions. Confusion counts, metric arithmetic, record links, output hashes, inactivity and raster-state counts were checked. A separate read-only RF Validator review passed the source/configuration, record integrity and metric-arithmetic checks; it did not independently rescore raster bytes.

## Interpretation and limits — WARNING

This is a **single-seed F01 calibration/development comparison of the two supplied model releases**, with the same simulation controls. RF is calibrated and LR is uncalibrated. It does not establish a general advantage of either model family, held-out incident accuracy, or physical/arrival-time validity. Identical seeds do not guarantee that identical random draws are assigned to the same cells once the simulated fronts diverge. There is no multi-seed mean or standard deviation; seeds 43–46 were not run.

The reference is an approximate satellite-derived footprint. Of 3,312 reference-positive crop cells, 1,910 are included and **1,402 (42.33%) are excluded** from the mapped-building scoring domain. Scores therefore describe this restricted domain. The NASA true-north interpretation remains **INFERRED and owner-adopted**, using a constant ignition-centre-derived bearing across the crop. F01 is owner-identified Santa Maria; HR=11/BFP 11:31 is owner/partner-adopted. The reported screenshot year 2024 versus owner year 2023, nearby NASA query location, and DSWD 13:25 discrepancy remain unresolved.

Confidence is high in the recorded execution, matched controls and reported arithmetic. Broader LR training provenance and historical/physical validity remain qualified. No additional run approval is needed for this completed request; further seeds or methodology changes would be separate work.
