# Airbase / Sitio Sewage RF — No Cutoff, Seeds 42–46

Run date: **2 November 2026 (Asia/Manila)**. **PASS — all five requested RF simulations completed and went inactive.**

Incident: **Sitio Sewage (Sanggi Interior), near Airbase, Lapu-Lapu City, 24 January 2025 at 1:28 PM**, as identified by the owner. The frozen sigmoid-calibrated RF was used without retraining, recalibration or a cutoff. All five outcomes are reported.

**Result:** the model captured most eligible reference-positive cells but substantially overpredicted the burned footprint. This is an exploratory separate-incident evaluation; geographic independence and prior dataset exposure are not established.

## Per-seed metrics — VERIFIED

The primary domain contains **215,553 common-valid mapped-building cells**, including **178 reference positives**. The owner-supplied Airbase reference contains **281 positive cells**; **103 (36.65%) are outside mapped buildings and excluded** from this domain. All 281 lie within the approved crop. Full-crop and primary-domain predicted burned counts agree.

| Seed | F1 | Recall | Precision | Jaccard | Accuracy | Burned cells | Inactive at update |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.266430 | 0.842697 | 0.158228 | 0.153689 | 0.996168 | 948 | 103 |
| 43 | 0.250794 | 0.887640 | 0.146026 | 0.143376 | 0.995621 | 1,082 | 161 |
| 44 | 0.242060 | 0.792135 | 0.142857 | 0.137695 | 0.995904 | 987 | 129 |
| 45 | 0.265323 | 0.887640 | 0.155972 | 0.152953 | 0.995941 | 1,013 | 125 |
| 46 | 0.194258 | 0.893258 | 0.108979 | 0.107578 | 0.993881 | 1,459 | 189 |

### Five-seed averages

**Mean ± sample SD**, denominator n−1, n=5, computed from full-precision per-run metrics. These describe stochastic variability within this one incident, not five independent incident evaluations or confidence intervals.

| Metric | Mean ± sample SD |
|---|---:|
| F1 | 0.243773 ± 0.029503 |
| Recall | 0.860674 ± 0.043408 |
| Precision | 0.142412 ± 0.019780 |
| Jaccard | 0.139058 ± 0.018832 |
| Accuracy | 0.995503 ± 0.000927 |
| Burned cells | 1097.8 ± 207.7 |

Accuracy is dominated by the large reference-negative population; the low precision and F1 reveal the overprediction. Burned cells are raster cells, not counts of distinct buildings. Updates are abstract simulation steps, not elapsed fire minutes. All five terminations were inactive with zero active cells; none was censored at the 300-update cap.

### Confusion matrices

Orientation: `[[TN, FP], [FN, TP]]`.

| Seed | Confusion matrix |
|---:|---|
| 42 | `[[214577, 798], [28, 150]]` |
| 43 | `[[214451, 924], [20, 158]]` |
| 44 | `[[214529, 846], [37, 141]]` |
| 45 | `[[214520, 855], [20, 158]]` |
| 46 | `[[214075, 1300], [19, 159]]` |

## Reference, ignition and crop

- Active Airbase scoring reference: `Thesis_RF/Code/dataFiles/stack_ground_truth_airbase.tif`, SHA-256 `1627ac903354f029572fa9abc9c61088909b38a5357c09228361f9021d2f52db`.
- Native parent grid: 5489 rows × 6896 columns, EPSG:32651, 3 m cells; the reference and all four environmental rasters have exactly matching shape, CRS and transform. Reference values are 0/1 with all cells natively valid.
- Supplied ignition `(606063.28, 1140377.27)` resolves to global **[row 1332,col 2501]**. Exact pixel centre is **(606063.2784,1140377.2699)**; its reference value is 1, and it is an eligible mapped-building cell.
- Owner-approved centered crop: **1024×1024**, global origin **[row 820,col 1989]**; ignition local **[512,512]**. Crop transform is `[3,0,604525.7784,0,-3,1141914.7699]`. The minimum ignition-to-boundary margin 511 cells exceeds the 300-step propagation cap.
- All 281 reference positives lie within global rows 1322–1347, cols 2494–2517. Of these, 178 enter the primary building domain and 103 are excluded solely because they are outside mapped buildings; none is excluded by environmental, reference or prediction validity.

The previous Santa Maria reference, configurations, scorer and outputs remain preserved. Airbase uses a separate frozen incident profile and `airbase_results.py`; changing the legacy F01 scorer's default path alone would violate its fixed geometry/reference checks.

The Airbase mask is an **owner-supplied mapped reference**. Its acquisition/construction method, timestamp accuracy and independent registration were not verified. Raw metric records inherit a generic reference-description string from the existing evaluator; that string is not evidence that this new mask was independently established as satellite-derived.

## NASA wind check and adopted input

The attached NASA POWER CSV is retained as `nasa_power_airbase.csv`. Its header identifies location **10.3143°N,123.9681°E**, the date 2025-01-24, wind speed at 10 m in m/s, wind direction in degrees, and time standard LST. The selected row is:

```text
YEAR,MO,DY,HR,WS10M,WD10M
2025,1,24,13,2.95,60.8
```

`2.95×3.6 = 10.62 km/h`, consistent with the owner's prescribed rounded **10.6 km/h**. Source direction 60.8° is consistent with the prescribed rounded **61° from ENE**. The experiment uses the owner's prescribed values, not a newly selected alternative hour or wind variant.

NASA documents LST as **Local Solar Time**, based on longitude bands; it may differ from government civil time. HR=13 is adopted here because it matches the owner-prescribed wind for the reported 13:28 incident, not as an independently verified instantaneous 13:28 civil-time observation. POWER hourly values are averages. [NASA POWER Hourly API/time standards](https://power.larc.nasa.gov/docs/services/api/temporal/hourly/).

Following the previous incident's convention, geographic true north is retained as an **INFERRED interpretation** of the prescribed bearing. At the ignition pixel centre (lat 10.314786435731733,lon 123.96867370380014), the projected geographic-north tangent gives meridian convergence approximately **+0.1734632694°**. Thus `(61−gamma) mod 360 = 60.8265367306°`, frozen to **60.826537° meteorological-from relative to grid north**, with no extra 180° reversal. This is one constant anchor-derived bearing across the crop; six decimals specify reproducibility, not physical wind precision.

The calculation used the installed Rasterio/PROJ transform. Symmetric latitude steps 0.001,0.0001,0.00001,0.000001° all yield the same six-decimal grid bearing. A standalone pyproj package was unavailable; no dependency was installed. These calculations share PROJ and are not independent geodetic implementations. [Rasterio windowed read guidance](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html).

CSV SHA-256: `746afb4b11d10e61d18717838f178b267d5747962e8f1e27c7791d4a958cc67f`.

## Model and fixed controls

Frozen sigmoid RF SHA-256: `78828dd4b2fcc66e9cdfd8ecb4a3ccd18c268478f3b5e6a4374c3286e3903e54`; calibration manifest SHA-256: `3141b8554662d703691c06a327f9a6c96c79f20840ce3738c358d4bdb6632afb`.

Seeds 42–46 vary the simulation RNG; the trained forest stays the same. Inference mode is `stochastic_probability`, all threshold fields null, no gate. Timers 1/1, wind weight 0.20, existing flammability weights, maximum 300 updates, inactivity early stop, no checkpoint/resume, canonical 11-feature schema and the existing resource/output policies are retained.

The new schema `phase16_airbase_rf_nocutoff.v1` freezes Airbase crop/ignition/wind/basis and admits only the frozen RF+manifest. Configs are disabled on disk and require explicit authorization plus an exact execution binding. The Airbase reference is read only during scoring; it is not an input to model inference or CA transitions.

## Evidence, checks and TIFF files

Branch: **`codex/rf-airbase-simulation`**, created at HEAD `734b42a07e3d1f3880a58b74da7d680d3d7ec6c0`. Pre-existing uncommitted work was preserved and carried in the working tree. No commit or push was created. Per-run bindings record actual runtime source hashes and dirty state.

**601 targeted tests passed**, including new Airbase profile/scorer checks and existing F01/RF/LR/gate/teacher/observer/wind regression contracts. The scorer verifies native aligned crop reads, hashes, geometry/masks/categories, record/config/binding/output identities, model/manifest observed identities and final-state counts. The existing final-footprint evaluator is reused. No raster was resampled. All five child processes exited 0 with no recorded resource-policy violations; the Windows hard-link no-overwrite publisher ran outside the sandbox.

Evidence directory: `Thesis_RF/Code/output/airbase_rf/20261102-nocutoff-s42-46-r1/`: input preflight, NASA copy, preparation/bindings and overlap inventory, test log, five stdout/stderr/exit/score records, aggregate `scores.json`, reproducible driver/report builder and final verification. New configs: `Thesis_RF/Code/config/phase16_airbase_rf_nocutoff_s{42..46}_exec_r1.yaml`.

Each run folder contains `start.json`, `termination.json`, `final_state.tif`, and `simulation_environment_valid.tif`. Final TIFFs for visual checking:

| Seed | Final-state TIFF |
|---:|---|
| 42 | [Open TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/phase15/p16-airbase-rf-nocutoff-s42-r1/final_state.tif) |
| 43 | [Open TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/phase15/p16-airbase-rf-nocutoff-s43-r1/final_state.tif) |
| 44 | [Open TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/phase15/p16-airbase-rf-nocutoff-s44-r1/final_state.tif) |
| 45 | [Open TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/phase15/p16-airbase-rf-nocutoff-s45-r1/final_state.tif) |
| 46 | [Open TIFF](C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/phase15/p16-airbase-rf-nocutoff-s46-r1/final_state.tif) |

Aggregate scores SHA-256: `6b7f395324f44e0d998ab843e75e2136b581c6e12b66d2ca9eb9a6dfe74f166a`.

## Interpretation and unresolved evidence — WARNING

This is a different owner-identified fire incident and a fixed five-seed no-cutoff experiment. It is **not established as held-out or spatially independent**: the approved Airbase crop geometrically intersects 23 of 26 frozen dataset-family windows, including a 460×837-cell overlap with F01, and contains the recorded F02 family centre. This is a source-window overlap finding; actual row-role overlap and full prior-use history were not inspected, so it does not by itself establish training-role leakage. No protected synthetic final-test rows were accessed.

Do not pool these scores with Santa Maria as though they used one common reference/domain. Reference construction and independent historical/geographic validity remain qualified, 103 of 281 reference positives are excluded, and NASA reanalysis/one-point hourly wind is not a measured evolving fire-site wind field. Timers and update counts have no verified mapping to physical elapsed fire time. Low precision and F1 are reported as observed; no threshold search, model fitting, rerun selection or reference/mask adjustment was used to improve them. Further experiments require a separately stated scope.
