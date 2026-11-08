# LR Cutoff 0.40 on Kristian's CA — Airbase, Seeds 42–46

**PASS — all five requested LR simulations completed successfully.** The existing 300-update cap and inactivity stop were retained. No new duration, target-size, firefighter or suppression stopping rule was implemented; no checkpoint was selected using the Airbase reference.

Completion timestamp (UTC): `2026-11-02T16:01:59.341732+00:00`. Campaign folder `20261102-gate040-s42-46-r1` is an identifier; exact execution timestamps and commands are retained in the records.

## What was run

The same frozen, rebuilt **uncalibrated/unweighted LR Pipeline (StandardScaler + LogisticRegression)** previously used in the Santa Maria comparison was run inside Kristian's existing CA. There was no model fitting, recalibration or model-file replacement. Model SHA-256: `41e6f55d3e07fa27a57b42d7d0be0828c72cf87ff6cc92c3834b0c0dfd8d217b`.

Cutoff rule: **p < 0.40 becomes 0; p ≥ 0.40 stays p; then the usual random draw ignites if rng.random() < p.** This is an inference-time probability gate, not deterministic ignition at 0.40. Existing float32 gate and full-grid RNG sampling were reused without modifying the engine. The three legacy classification/threshold fields remain null.

Airbase reference: `Thesis_RF/Code/dataFiles/stack_ground_truth_airbase.tif`, SHA-256 `1627ac903354f029572fa9abc9c61088909b38a5357c09228361f9021d2f52db`. It is used only for scoring, not for model inference or CA transitions.

Fixed Airbase controls: native 1024×1024 crop, global origin [row820,col1989], ignition local [512,512] / global [1332,2501], centre (606063.2784,1140377.2699); EPSG:32651, 3 m cells, transform `[3,0,604525.7784,0,-3,1141914.7699]`. Wind remains 10.6 km/h from owner-prescribed true 61°, represented as grid-from60.826537°. Same NASA HR13/LST/north-reference qualifications, 11-feature order, timers1/1, windweight0.20, flammabilityweights, four environmental rasters and resource policy as retained Airbase RF.

Seeds42–46 change only the simulation RNG; the fitted LR stays fixed. Preparation checks restore the model/gate and run/profile/authorization fields and exactly recover each same-seed RF Airbase baseline effective config. Stopping remains **max300 updates + inactivity**, with no duration_basis or physical-minute interpretation.

## Scores — VERIFIED

The primary domain is **215,553 common-valid mapped-building cells**, including **178 reference positives**. Of the full reference's **281** positive cells, **103 (36.65%)** are excluded solely because they fall outside mapped buildings. All281 are inside the crop. This is the same domain as the retained Airbase RF results, rather than the partner's whole-map scoring domain.

| Seed | F1 | Recall | Precision | Jaccard | Accuracy | Reached cells | Updates | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.254022 | 0.842697 | 0.149551 | 0.145490 | 0.995913 | 1,003 | 103 | inactive / complete |
| 43 | 0.274476 | 0.882022 | 0.162526 | 0.159068 | 0.996149 | 966 | 123 | inactive / complete |
| 44 | 0.246954 | 0.853933 | 0.144349 | 0.140871 | 0.995699 | 1,053 | 135 | inactive / complete |
| 45 | 0.297479 | 0.994382 | 0.174901 | 0.174729 | 0.996122 | 1,012 | 125 | inactive / complete |
| 46 | 0.269587 | 0.898876 | 0.158573 | 0.155794 | 0.995978 | 1,009 | 99 | inactive / complete |

### Five-seed average

Mean ± **sample SD** (n−1 denominator, n=5), calculated from full precision. This is seed variability within one incident, not five independent fires or a confidence interval.

| Metric | Mean ± sample SD |
|---|---:|
| F1 | 0.268503 ± 0.019688 |
| Recall | 0.894382 ± 0.060167 |
| Precision | 0.157980 ± 0.011876 |
| Jaccard | 0.155190 ± 0.013194 |
| Accuracy | 0.995972 ± 0.000181 |
| Reached cells | 1008.6 ± 30.9 |

### Confusion matrices

Orientation: `[[TN, FP], [FN, TP]]`. Counts and derived metrics were independently recomputed directly from native raster positions, without the scoring helper, and matched all five saved scores.

| Seed | Matrix | Active cells at termination |
|---:|---|---:|
| 42 | `[[214522, 853], [28, 150]]` | 0 |
| 43 | `[[214566, 809], [21, 157]]` | 0 |
| 44 | `[[214474, 901], [26, 152]]` | 0 |
| 45 | `[[214540, 835], [1, 177]]` | 0 |
| 46 | `[[214526, 849], [18, 160]]` | 0 |

Reached cells include states3/4/5 (ever ignited through termination). An inactive result has only state5 in this footprint. A capped active result would include3/4/5 and be marked censored rather than called fire-out. These are raster cells, not distinct buildings. Full-crop counts are recorded separately in scores.json.

## Retained RF context — no RF rerun

These are the two observed pipelines in the same Airbase CA/domain. **RF is sigmoid-calibrated/no cutoff; LR is uncalibrated/cutoff0.40.** Their policies differ; this comparison does not isolate model family or establish a general superiority claim. The earlier H90 RF sensitivity uses a different stopping horizon and is not mixed into this table.

| Pipeline | F1 mean ± SD | Recall mean ± SD | Precision mean ± SD | Reached cells mean ± SD |
|---|---:|---:|---:|---:|
| RF calibrated, no cutoff — retained | 0.243773 ± 0.029503 | 0.860674 ± 0.043408 | 0.142412 ± 0.019780 | 1097.8 ± 207.7 |
| LR uncalibrated, cutoff0.40 — new | 0.268503 ± 0.019688 | 0.894382 ± 0.060167 | 0.157980 ± 0.011876 | 1008.6 ± 30.9 |

## Interpretation and limits — WARNING

The supplied screenshot describes the partner's separate CA, scoring populations and trajectory. Those numbers and its step20/step100 observation are **partner-reported**, not independently reproduced by this run. Our table reports this LR inside Kristian's CA. We did not evaluate a circle comparison, tune a stop point against the reference, or extract an earlier best-scoring checkpoint.

Low precision/F1 with a much larger prediction than the reference indicates overprediction in this evaluation domain. A final-footprint score alone does not establish that spread direction or arrival timing is correct. Existing CA behavior has no firefighter/suppression mechanism; lack of suppression may be relevant, but this experiment does not establish it as the sole cause. Updates have no verified minutes mapping, and inactivity is not proof of historical fire-out.

No new stopping rule was added. A future rule should be justified independently of this fire's final reference; stopping at step20 because it matches that reference would use the answer to choose the prediction. The previous owner-assumed one-minute H90 RF experiment remains a separate exploratory sensitivity, not time calibration for these LR runs.

Reference construction/date/registration remain unverified, 103/281 positives remain excluded, and the Airbase crop geometrically overlaps23/26 frozen family windows. Actual row-role exposure/prior-use independence is unresolved; no held-out claim or protected synthetic final-test access. Constant hourly point wind and INFERRED geographic-north conversion retain prior qualifications. High accuracy is dominated by reference-negative cells. Santa Maria and Airbase use different domains and are not pooled as one accuracy estimate.

## TIFF outputs

Native final-state TIFFs preserve codes1=nonburnable,2=not-yet-burning,3=ignited,4=blazing,5=extinguished. For cumulative area compare states3/4/5; for inactive outcomes this equals state5. No resampling or forced extinguishing was applied.

| Seed | Final-state TIFF |
|---:|---|
| 42 | [Open TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/phase15/p16-airbase-lr-t040-s42-r1/final_state.tif) |
| 43 | [Open TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/phase15/p16-airbase-lr-t040-s43-r1/final_state.tif) |
| 44 | [Open TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/phase15/p16-airbase-lr-t040-s44-r1/final_state.tif) |
| 45 | [Open TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/phase15/p16-airbase-lr-t040-s45-r1/final_state.tif) |
| 46 | [Open TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/phase15/p16-airbase-lr-t040-s46-r1/final_state.tif) |

## Reproducibility and checks

**679 tests passed in59.02s**, covering new exact Airbase LR profile/gate/model/manifest/domain contracts and existing RF/F01/H90/automata/teacher/observer/wind behavior. All five real child processes exited0 with valid outputs and no recorded policy violations. Native hash/geometry/mask/category, record/config/binding/model/input/output identities and state totals were checked. No failed real-run retries, seed selection, fitting or new stopping-rule search.

New config paths: `Thesis_RF/Code/config/phase16_airbase_lr_t040_s{42..46}_exec_r1.yaml`. Profile `phase16_airbase_lr_probability_gate_040.v1`; disabled on disk and execution requires an exact approved binding. Existing RF/F01/H90 profiles remain frozen. The generic metric equations and CA engine were unchanged; runner and scorer received only the narrow LR Airbase allowance.

Evidence directory: [Open folder](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/airbase_lr/20261102-gate040-s42-46-r1). Contains saved prior source/hash inventory, partner screenshot, preparation and exact bindings, testlog, five stdout/stderr/exit/score records, scores.json, direct raster count checks and driver/reportbuilder/final verification. Scores SHA-256 `e8ebdd09ac4984b96ae25842df1972d8c52f78d4621130b21801293fba04cf7e`; preparation SHA-256 `4eb3acffc379b79f79e7388e3415691118312d2c95f52d610f4b970a7a4a4871`.

Branch `codex/rf-airbase-simulation`, HEAD `734b42a07e3d1f3880a58b74da7d680d3d7ec6c0`; actual working source/runtime hashes and execution timestamps are captured by each start record. Existing uncommitted work preserved; no commit/push, partner edits or manuscript changes. Old RF/LR results, models and references remain preserved. Memory records this bounded experiment as D-169.
