# Airbase RF — 90-Update Duration Sensitivity, Seeds 42–46

Run date: **2 November 2026 (Asia/Manila)**. **PASS — five runs completed successfully at the approved 90-update limit.** All five were still active and are recorded as **censored**, not as simulated fire-out.

## Question and time assumption

The owner reports Sitio Sewage (Sanggi Interior), Pusok/Airbase, on 24 January 2025 starting **13:28** and declared fire-out **14:58**, a **90-minute** interval. The owner explicitly approved **ASSUMPTION: 1 minute per update; stop at 90 updates** for this exploratory check. These reported times were not independently verified.

The previous five Airbase runs became inactive at updates **103, 161, 129, 125 and 189**. Therefore their 300-update cap did not bind. This experiment imposes an earlier horizon; it does not demonstrate that the old cap represented 300 real minutes or that duration alone caused the overprediction.

The owner also reports Santa Maria starting 11:30 and being **controlled** at 16:00–16:30 (270–300 minutes). Controlled and fire-out are different operational endpoints. This new report does not replace the historical F01 adopted 11:31 input basis, nor establish a calibrated clock transferable between incidents.

A CA update needs a justified relationship to physical time before minutes can be treated as validated model time. For example, Tan and Vicente explicitly chose one minute per update and separately specified cell burning timers from cited experiments; that design is evidence that both choices must be stated, not that its timing transfers to this RF. [Tan & Vicente (2019), §§3.4–3.7](https://isprs-archives.copernicus.org/articles/XLII-4-W19/415/2019/isprs-archives-XLII-4-W19-415-2019.pdf).

## Fixed controls and scoring

Only the run/profile/scenario/authorization identities, explicit duration metadata and maximum update count changed relative to each same-seed baseline. Preparation checks restore these fields and prove every other effective setting identical. The automata engine and model were not changed or fitted. RF no-cutoff, five seeds 42–46, timers 1/1, wind weight 0.20 and existing flammability weights remain fixed.

The frozen sigmoid RF SHA-256 is `78828dd4b2fcc66e9cdfd8ecb4a3ccd18c268478f3b5e6a4374c3286e3903e54`; calibration manifest SHA-256 `3141b8554662d703691c06a327f9a6c96c79f20840ce3738c358d4bdb6632afb`. Wind remains owner-prescribed 10.6 km/h from true 61°, represented as grid-from **60.826537°**, under the existing qualified NASA HR13/LST basis.

Crop: 1024×1024, global origin [820,1989], local ignition [512,512] (global [1332,2501]); EPSG:32651, 3 m cells, transform `[3,0,604525.7784,0,-3,1141914.7699]`. No resampling. Airbase reference SHA-256 `1627ac903354f029572fa9abc9c61088909b38a5357c09228361f9021d2f52db`.

Primary scoring domain stays **215,553 common-valid mapped-building cells** with **178 reference positives**. The full reference contains 281 positive cells; 103 (36.65%) remain excluded solely for lying outside mapped buildings. The reference was accessed for scoring only, not by inference or transitions.

At a truncated horizon, the cumulative reached footprint includes **states 3, 4 and 5** (ignited, blazing, extinguished). Counting only state 5 would omit cells still burning. The binary TIFF exports use 1 for any of these states and 0 otherwise; 0 is valid data, not nodata. Original inactive footprints contain state 5 only. Burned counts below mean cumulative reached cells, not distinct buildings.

## Per-seed results — VERIFIED

| Seed | H90 F1 | H90 Recall | H90 Precision | H90 reached cells | Active at H90 | Original F1 | Original reached cells | Original inactive update |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.270718 | 0.825843 | 0.161894 | 908 | 19 | 0.266430 | 948 | 103 |
| 43 | 0.307992 | 0.887640 | 0.186321 | 848 | 20 | 0.250794 | 1,082 | 161 |
| 44 | 0.301310 | 0.775281 | 0.186992 | 738 | 18 | 0.242060 | 987 | 129 |
| 45 | 0.313253 | 0.803371 | 0.194558 | 735 | 25 | 0.265323 | 1,013 | 125 |
| 46 | 0.276836 | 0.825843 | 0.166290 | 884 | 12 | 0.194258 | 1,459 | 189 |

## Five-seed averages

Mean ± **sample SD**, n−1 denominator, n=5. These quantify random-seed variability within one incident; they are not confidence intervals or five independent fire incidents.

| Metric | Original, run to inactivity | Stop at update 90 |
|---|---:|---:|
| F1 | 0.243773 ± 0.029503 | 0.294022 ± 0.019082 |
| Recall | 0.860674 ± 0.043408 | 0.823596 ± 0.041398 |
| Precision | 0.142412 ± 0.019780 | 0.179211 ± 0.014260 |
| Jaccard | 0.139058 ± 0.018832 | 0.172465 ± 0.013076 |
| Accuracy | 0.995503 ± 0.000927 | 0.996718 ± 0.000343 |
| Reached cells | 1097.8 ± 207.7 | 822.6 ± 81.5 |

### Confusion matrices and direct spatial check

Orientation: `[[TN, FP], [FN, TP]]`. Every matrix was independently recomputed from native aligned raster cell positions. Every H90 cumulative mask is an exact subset of its same-seed original inactive mask; all removed cells are reached after the earlier horizon. Full-crop and primary-domain reached counts agree.

| Seed | H90 confusion matrix | Cells removed vs original | TP cells removed |
|---:|---|---:|---:|
| 42 | `[[214614, 761], [31, 147]]` | 40 | 3 |
| 43 | `[[214685, 690], [20, 158]]` | 234 | 0 |
| 44 | `[[214775, 600], [40, 138]]` | 249 | 3 |
| 45 | `[[214783, 592], [35, 143]]` | 278 | 15 |
| 46 | `[[214638, 737], [31, 147]]` | 575 | 12 |

## Interpretation — WARNING

The earlier horizon reduces the mean predicted footprint by **25.07%**. The table shows the accompanying precision, recall and F1 changes; stopping early can remove both false positives and true positives. Overprediction remains substantial relative to 178 included reference cells. High accuracy is dominated by the large reference-negative population.

This supports the limited conclusion that the chosen update horizon affects this model's footprint. It does **not** prove that one update equals one minute, that the historical fire would follow this trajectory, or that the model reproduces suppression/fire-out. The 90-update runs retain active fire: a numerical stop is not a modeled extinguishing mechanism. Timers 1/1 remain state-residence parameters; they were not set to the incident duration.

Treat this as an owner-prespecified duration sensitivity, not a newly calibrated or held-out result. For a defensible physical clock, seek independently timed perimeters, arrival times or spread observations and validate a chosen time mapping on separate evidence. Do not choose the checkpoint with best agreement to the same Airbase final mask and then report it as independent validation.

The earlier limitations still apply: Airbase crop overlaps 23/26 frozen family windows geometrically; actual row-role exposure and prior-use independence remain unverified. Reference construction/date/registration remain unverified, 103/281 positives are excluded, and constant hourly point wind is not an observed evolving local wind field. No synthetic final-test roles were opened. Santa Maria and Airbase scores use different domains and must not be pooled as common-domain accuracy.

## TIFFs for visual inspection

Use the **binary cumulative footprint** to compare area. The final-state TIFF retains all five state codes and shows the remaining active cells. These files represent update 90 under the assumed clock, not simulated fire-out.

| Seed | Binary cumulative footprint | Final-state TIFF |
|---:|---|---|
| 42 | [Open binary TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/airbase_duration/20261102-h90-s42-46-r1/seed42_burned_through_90_updates.tif) | [Open state TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/phase15/p16-airbase-rf-nocutoff-h90-s42-r1/final_state.tif) |
| 43 | [Open binary TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/airbase_duration/20261102-h90-s42-46-r1/seed43_burned_through_90_updates.tif) | [Open state TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/phase15/p16-airbase-rf-nocutoff-h90-s43-r1/final_state.tif) |
| 44 | [Open binary TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/airbase_duration/20261102-h90-s42-46-r1/seed44_burned_through_90_updates.tif) | [Open state TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/phase15/p16-airbase-rf-nocutoff-h90-s44-r1/final_state.tif) |
| 45 | [Open binary TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/airbase_duration/20261102-h90-s42-46-r1/seed45_burned_through_90_updates.tif) | [Open state TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/phase15/p16-airbase-rf-nocutoff-h90-s45-r1/final_state.tif) |
| 46 | [Open binary TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/airbase_duration/20261102-h90-s42-46-r1/seed46_burned_through_90_updates.tif) | [Open state TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/phase15/p16-airbase-rf-nocutoff-h90-s46-r1/final_state.tif) |

## Evidence and validation

**631 targeted tests passed** in 53.34 s after the final source/test edit. Seven suites cover Airbase profile/scoring, simulation readiness, automata, teacher, observer and wind contracts. Legacy F01 and original Airbase profiles retain their 300-update contracts; the new profile alone accepts the exact H90 assumption. All five real child processes exited 0 with valid outputs and no recorded violations, `timestep_limit`, `status=censored`, 90 updates. No failed real-run retries or seed selection.

Evidence: [duration evidence folder](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/airbase_duration/20261102-h90-s42-46-r1); `preparation.json`, `pytest-r2.log`, five bound execution/stdout/stderr/exit/score records, `scores.json`, exports and reproducible driver/report builder. Aggregate scores SHA-256 `16aebca0ff02e944a5c971a3b4bed2f68d30e3ce162a6c09136923f93557c627`. Prior aggregate scores SHA-256 `6b7f395324f44e0d998ab843e75e2136b581c6e12b66d2ca9eb9a6dfe74f166a`.

Configs: `Thesis_RF/Code/config/phase16_airbase_rf_duration90_s{42..46}_exec_r1.yaml`; schema `phase16_airbase_rf_duration90.v1`, disabled on disk and activated only through the approved binding. Each start record retains actual source, model, input, configuration and execution hashes. Branch `codex/rf-airbase-simulation`, HEAD `734b42a07e3d1f3880a58b74da7d680d3d7ec6c0`, existing uncommitted work preserved; no commit/push.

Previous Airbase results and report remain unchanged. Partner files, models, datasets, the engine and manuscript were not modified by this duration check. Project memory records this bounded sensitivity as D-168.
