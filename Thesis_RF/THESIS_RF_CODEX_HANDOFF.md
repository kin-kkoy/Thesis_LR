# Thesis_RF - Codex Session Handoff

## 1. Thesis and Project Overview

The overall thesis is **Predictive Fire Spread Modeling Using Cellular Automata with Machine Learning**. It combines a 2D cellular automaton (CA), aligned GIS raster layers, and machine-learning estimates of per-cell ignition probability to simulate urban fire spread at 3 x 3 m resolution.

This handoff covers only the `Thesis_RF` Random Forest implementation. The Logistic Regression implementation is out of scope unless shared architecture is needed to explain an interface or comparison. Previous AI summaries are context, not proof; repository artifacts and executable code take precedence.

**Confidence convention:** `VERIFIED` means directly supported by a current file or artifact; `INFERRED` means reasonably concluded; `ASSUMPTION` means temporarily adopted for analysis; `NEEDS VERIFICATION` means evidence is insufficient.

## 2. Thesis_RF Purpose

`Thesis_RF` is the Random Forest branch of the thesis implementation. It provides:

- Raster loading and alignment checks for slope, proximity, buildings, and materials.
- Feature engineering for static environmental and dynamic neighborhood/wind inputs.
- Synthetic training-data generation from raster stacks and ground-truth fire perimeter data.
- Random Forest training, Optuna tuning, probability inference, and feature-importance reporting.
- A vectorized five-state CA that consumes `predict_proba()` output.
- Raster-level comparison of the simulated burned perimeter with historical ground truth.
- A pure CA baseline for comparison with ML-assisted simulation.

The current technical objective is validation and verification, not automatic acceptance of prior implementation choices.

## 3. Repository / Component Overview

All RF implementation code is under `Thesis_RF/Code/`.

| Component | Role | Relationship |
|---|---|---|
| `Code/main.py` | Main entry point | Loads YAML, applies defaults, and runs the RF-assisted simulation through `orchestrator.py`. |
| `Code/orchestrator.py` | Pipeline wiring | Loads configuration, resolves paths, loads the model, converts EPSG:32651 ignition coordinates to raster indices, runs timesteps, and writes a final GeoTIFF. |
| `Code/config/default_experiment.yaml` | Current default experiment | Defines raster names, seed, wind, transition values, ML model path, threshold, inference chunk size, output path, and dataset-generation settings. |
| `Code/modules/data_loader.py` | Raster preparation | Reads four aligned rasters, checks shape/CRS/transform, builds nodata/ocean masks, normalizes layers, and exposes the environment. |
| `Code/modules/feature_pipeline.py` | Feature assembly | Defines the current 11-feature schema, material-class validation, material risk, composite flammability, wind components, and grid feature arrays. |
| `Code/modules/automata_engine.py` | CA simulation | Maintains `np.int8` state grids and timers, computes Moore-neighborhood counts and wind convolution, performs chunked model inference, and advances states. |
| `Code/dataset_generator.py` | Single-stack dataset generation | Aligns rasters and ground truth, crops the configured ROI, retains all positives, samples negatives, and writes a CSV. |
| `Code/generate_multi_scenario.py` | Multi-scenario data generation | Runs 5 wind speeds x 8 directions x 5 runs, then combines temporary CSVs. |
| `Code/train_rf_optuna.py` | Current tuning entry point | Tunes an 11-feature RF on `multi_scenario_dataset.csv` with 5-fold stratified CV and F1 objective. |
| `Code/modules/model_trainer.py` | General trainer | Loads CSV data, filters susceptible rows, performs a stratified 80/20 split, trains RF, evaluates thresholded metrics, saves `.joblib`/`.pkl`, and reports feature importance. |
| `Code/validation_engine.py` | Spatial validation | Compares final-state value 5 with ground-truth value 1 after CRS, shape, transform, and nodata checks. Reports confusion matrix, precision, recall, F1, AUC-ROC, and Jaccard. |
| `Code/pure_ca_baseline.py` | Non-ML comparison | Runs the CA without the RF model. Exact current baseline protocol requires verification. |
| `Code/ThesisPaper/Methodology_Paper.md` | RF methodology notes | Records feature-order and wind-convolution checks plus historical RF behavior observations. |
| `Code/ThesisPaper/Predictive Fire Spread Modelling Using Cellular Automata with Machine Learning.md` | Thesis document | Describes the proposed CA/ML methodology and validation goals; it is not by itself proof that an experiment ran. |
| `Code/basicCA.py`, `simpleCA.py`, `demChecker.py`, `slopeCalculator.py` | Supporting/older utilities | Relevant historical or exploratory utilities; their role in the current RF pipeline is not fully established. |

The repository guidance states that generated CSVs, rasters, models, and outputs are intentionally ignored by Git. Therefore, a missing tracked artifact does not prove that it was never generated, but it does mean provenance must be recovered locally or from logs.

The local artifact inventory captured on 2026-09-10 includes these RF-side datasets: `dataset-B.csv` (21,079 rows), `multi_scenario_dataset.csv` (24,922 rows), `revised_dataset.csv` (10,000 rows, legacy schema), `revised_dataset_5tier.csv` (50,000 rows), `synthetic_20m_dataset.csv` (20,923,273 rows), `synthetic_fire_dataset.csv` (50,000 rows), and `test.csv` (2,000 rows, legacy schema). Their experiment assignments are not all established. Local model artifacts include `fire_rf_model.joblib`, `optuna_rf_search_summary.json`, and `optuna_study_rf.joblib`; exact model-to-set mapping still requires metadata inspection. Local outputs include `final_sitio_santa_maria_11_params.tif`, `final_state.tif`, `final_state_baseline.tif`, `final_state_sitio_million_flower.tif`, checkpoints, and an emergency checkpoint. These files are present locally but are not sufficient by filename alone to prove which historical experiment produced them.

## 4. Random Forest Experimental Design

### Set A - 10 Parameters

**Verified schema from `docs/findings_SetA.md` and `docs/Documentation.md`:**

`slope_risk`, `proximity_risk`, `building_presence`, `material_risk`, `material_class`, `wind_speed`, `wind_sin`, `wind_cos`, `neighbor_burning_count`, `composite_flammability`.

The omitted feature relative to Set C is `wind_weighted_score`. The target is `Ignited`, a binary label, as stated in the training and dataset-generation code.

- **Purpose:** baseline RF before adding dynamic wind weighting.
- **Dataset file and exact generation run:** **NEEDS VERIFICATION**. `Code/dataFiles` currently contains several candidate CSVs, but no file is explicitly named Set A.
- **Training algorithm and hyperparameters:** **NEEDS VERIFICATION** for the historical Set A run. The generic trainer uses RF defaults of 200 trees, max depth 15, seed 42, and `balanced_subsample`; the Optuna script is for the 11-feature dataset and must not be attributed to Set A without evidence.
- **Validation configuration:** a CA threshold sweep was documented. Exact model artifact, ignition configuration, timestep configuration, and command log are **NEEDS VERIFICATION**.
- **Documented threshold results:**

| Threshold | Precision | Recall | F1 | AUC-ROC | Jaccard |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.840 | 0.463 | 0.597 | 0.732 | 0.426 |
| 0.40 | 0.831 | 0.463 | 0.595 | 0.732 | 0.423 |
| 0.30 | 0.604 | 0.435 | 0.506 | 0.718 | 0.338 |
| 0.10 | 0.510 | 0.738 | 0.603 | 0.869 | 0.431 |
| 0.05 | 0.447 | 0.767 | 0.564 | 0.883 | 0.393 |

These values are `VERIFIED` as entries in `docs/findings_SetA.md`, but independent reproducibility is `NEEDS VERIFICATION`.

- **Current interpretation:** best documented F1 is 0.603 at threshold 0.10; best documented recall is 0.767 at 0.05, below the 0.80 target. This is a documented result, not a newly rerun experiment.
- **Known issue:** threshold changes the CA's emergent spread and create a precision/recall tradeoff. The notes also report severe class imbalance, but the exact data provenance and denominator require verification.

### Set B - 10 Parameters + Weighted Class

**Verified CSV schema:** `Code/dataFiles/dataset-B.csv` has 10 input columns plus `Ignited`:

`slope_risk`, `proximity_risk`, `building_presence`, `material_risk`, `wind_speed`, `wind_sin`, `wind_cos`, `neighbor_burning_count`, `composite_flammability`, `wind_weighted_score`.

Compared with Set A, the documented Set B schema removes `material_class` and adds `wind_weighted_score`. It is therefore not simply a class-weight-only copy of Set A; the feature schema also differs.

- **Purpose:** evaluate weighted-class training together with the wind-weighted feature while omitting `material_class`.
- **Dataset:** `dataset-B.csv` is verified to exist locally and has 21,079 data rows. The exact generation command and whether every row belongs to the historical Set B run are **NEEDS VERIFICATION**.
- **Class weighting:** the findings describe weighted-class behavior, but the exact historical weight mapping is **NEEDS VERIFICATION**. Current `model_trainer.py` supports `balanced` and `balanced_subsample`; current `train_rf_optuna.py` searches explicit positive-class weights of 3, 5, or 7. Do not conflate these APIs.
- **Training hyperparameters:** exact Set B training command and RF parameters are **NEEDS VERIFICATION**.
- **Validation configuration:** documented CA threshold sweep; exact artifact, seed, ignition points, and command log are **NEEDS VERIFICATION**.
- **Documented threshold results:**

| Threshold | Precision | Recall | F1 | AUC-ROC | Jaccard |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 1.000 | 0.009 | 0.019 | 0.505 | 0.009 |
| 0.10 | 0.475 | 0.241 | 0.319 | 0.620 | 0.190 |
| 0.05 | 0.515 | 0.792 | 0.624 | 0.896 | 0.453 |
| 0.03 | 0.411 | 0.936 | 0.571 | 0.968 | 0.400 |

These values are `VERIFIED` as documentation entries in `docs/findings_SetB.md`, while independent reproducibility is `NEEDS VERIFICATION`.

- **Current interpretation:** threshold 0.03 meets the recall target but has lower F1 than threshold 0.05. The model is documented as conservative at threshold 0.30 and highly threshold-sensitive.
- **Known issue:** because Set B changes both feature schema and class weighting, an isolated causal claim about class weighting is not currently justified.

### Set C - 11 Parameters

**Verified schema from current source and dataset:**

`slope_risk`, `proximity_risk`, `building_presence`, `material_risk`, `material_class`, `wind_speed`, `wind_sin`, `wind_cos`, `neighbor_burning_count`, `composite_flammability`, `wind_weighted_score`.

The added feature relative to Set A is `wind_weighted_score`; relative to Set B, `material_class` is restored. The intended reason is to combine dynamic directional-neighbor influence with material-class information. The causal interpretation remains under review.

- **Purpose:** full 11-feature RF configuration and current CA integration target.
- **Dataset:** `Code/dataFiles/multi_scenario_dataset.csv`, 24,922 data rows, is verified locally. Its header matches the 11-feature schema. The generator defines 200 scenarios: 5 speeds, 8 directions, and 5 runs, with 10 ignition points and 100 recorded timesteps per scenario. Whether the existing CSV was produced by exactly the current generator version is **NEEDS VERIFICATION**.
- **Training configuration:** `train_rf_optuna.py` uses seed 42, 100 Optuna trials, 5-fold stratified CV, susceptible-row filtering (`neighbor_burning_count > 0`), F1 as the objective, and RF search ranges of 50-300 estimators, depth 10-30, split 2-20, leaf 1-4, and explicit class-weight choices `{1:3}`, `{1:5}`, `{1:7}` with negative class weight 1. The saved summary reports the selected parameters below.
- **Recorded tuning result:** 24,922 rows after filtering, positive rate 0.0721049675, best CV F1 0.1347604290, class-weight choice `cw_1_7`, 58 estimators, max depth 27, min split 20, min leaf 4. This is verified in `Code/models/optuna_rf_search_summary.json`; it is a CV F1 result, not the spatial CA result.
- **Validation configuration and spatial results:** `docs/findings_SetC.md` documents a CA threshold sweep. Exact command, model artifact-to-run mapping, and raw output are **NEEDS VERIFICATION**.
- **Documented threshold results:**

| Threshold | Precision | Recall | F1 | AUC-ROC | Jaccard |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.650 | 0.827 | 0.728 | 0.914 | 0.572 |
| 0.10 | 0.447 | 0.943 | 0.607 | 0.971 | 0.435 |
| 0.05 | 0.381 | 0.947 | 0.544 | 0.974 | 0.373 |
| 0.00 | 0.741 | 0.495 | 0.594 | 0.748 | 0.422 |

The current YAML sets `ml_model.proba_threshold: 0.30` and labels it an optimal Set C threshold, but the label is a project claim that should be verified against the actual model and output.

## 5. Current Validation Objective

The current objective is to establish whether the RF implementation and reported results can be trusted before thesis comparison or final CA interpretation. Specifically verify:

- All four rasters have compatible shape, CRS, transform, nodata semantics, and expected class values.
- Each dataset's feature order and target construction match its named experiment.
- There is no leakage from ground truth, repeated spatial cells, scenario runs, or post-event variables into validation.
- Susceptible-cell filtering and negative sampling are documented and applied consistently.
- Class weighting is present only where intended and is not confused with threshold calibration.
- Training and validation splits are appropriate for spatial and multi-scenario data.
- Metrics are calculated on the intended unit: per-cell model predictions versus final CA perimeter versus historical perimeter.
- The saved model, feature schema, threshold, YAML, and output raster belong to the same experiment.
- The three sets are comparable enough to support claims about added features or weighting.
- RF `predict_proba()` output can be integrated into the model-agnostic CA interface without feature mismatch.

Important metric distinction: `model_trainer.py` evaluates held-out CSV rows, while `validation_engine.py` evaluates raster perimeter masks. These outputs must not be reported as the same validation result.

## 6. Instructions From the Thesis Owner

The following requirements are explicit in the current project request and repository guidance:

- Focus on the Random Forest implementation; do not modify Logistic Regression unless explicitly authorized.
- Inspect existing work before rewriting it and preserve working code where possible.
- Prefer small, explainable changes and validate before implementing new functionality.
- Do not fabricate results, experiments, citations, tests, or claims of successful execution.
- Distinguish verified facts, inferences, assumptions, and `NEEDS VERIFICATION` items.
- Use datasets, outputs, source code, configurations, model metadata, history, and notes as evidence, in that order of trust.
- Protect reproducibility and explain significant technical decisions.
- Flag questionable methodology even if it weakens expected thesis results.
- Keep the RF and LR conclusions separate.
- The CA must remain model-agnostic: model loading must accept `.pkl` or `.joblib` and validate a callable `predict_proba()`; the CA engine must not hardcode a specific estimator such as `RandomForestClassifier`.
- Use deterministic seeds and avoid unseeded global randomness.
- Preserve vectorized NumPy/SciPy grid operations; do not introduce nested Python loops for grid iteration.
- Use the five CA states: 1 non-burnable, 2 not yet burning, 3 ignited, 4 blazing, 5 extinguished.
- Use Moore-neighborhood behavior and retain the thesis metric targets of recall and F1 at least 0.80 as evaluation goals, not as guaranteed outcomes.

## 7. Established Agent / Engineering Workflow

The established engineering sequence is:

**Analyze -> Find evidence -> Identify issue -> Propose change -> Implement the smallest justified fix -> Test -> Compare results -> Document findings.**

Responsibility boundaries:

- **Tech Lead:** owns scope, evidence hierarchy, architecture, phase sequencing, risk assessment, and delegation prompts. It should not invent missing experiment facts.
- **Coder / ML Engineer:** implements explicitly assigned RF files, preserves public interfaces, supports generic model loading, keeps CA operations vectorized, and supplies commands and outputs used to verify the change.
- **Researcher / Thesis reviewer:** checks feature definitions, equations, methodology claims, metrics, and thesis consistency; updates research notes with evidence labels.
- **Data Analyst / Checker:** verifies CSV schemas, row counts, class distributions, split integrity, leakage risks, raster alignment, and model/output provenance.

Tasks touching the same file or depending on another phase should be sequential. Independent documentation review and artifact inventory may run in parallel. The current repository has `AGENTS.md` guidance but no separate committed Tech Lead/Researcher/Coder role definition under `Thesis_RF`; role details above come from the established project workflow and must still be treated as operating guidance rather than experiment evidence.

## 8. Findings and Discoveries So Far

**Finding:** The current RF feature pipeline defines exactly 11 features in a fixed order.

**Evidence:** `Code/modules/feature_pipeline.py`, `Code/train_rf_optuna.py`, and `multi_scenario_dataset.csv` header.

**Impact:** Feature order is a hard model/CA contract; any Set A/B model needs explicit schema alignment.

**Status:** VERIFIED.

**Finding:** The CA uses vectorized state updates, Moore-neighbor convolution, directional wind convolution, `np.int8` state storage, and chunked susceptible-cell inference.

**Evidence:** `Code/modules/automata_engine.py`.

**Impact:** The current architecture addresses the previous full-grid inference memory risk, but behavior still depends strongly on calibration and timer settings.

**Status:** VERIFIED.

**Finding:** Model loading validates supported `.pkl`/`.joblib` extensions and callable `predict_proba()` through the orchestrator/engine boundary.

**Evidence:** `Code/orchestrator.py` and `Code/modules/automata_engine.py`.

**Impact:** RF is not required as a class import by the CA integration, although the training module itself intentionally imports RF.

**Status:** VERIFIED.

**Finding:** Current Set C Optuna CV performance is low compared with the thesis target.

**Evidence:** `optuna_rf_search_summary.json` reports best CV F1 0.1347604290.

**Impact:** The documented spatial Set C results cannot be used as a substitute for the low row-level CV result; the two evaluation levels need reconciliation.

**Status:** VERIFIED.

**Finding:** Historical Set A, B, and C threshold sweeps show strong threshold sensitivity and no documented set reaches both precision and recall of 0.80.

**Evidence:** `docs/findings_SetA.md`, `docs/findings_SetB.md`, `docs/findings_SetC.md`.

**Impact:** Threshold is part of the CA experiment, and claims of model superiority require fixed, predeclared calibration rules.

**Status:** VERIFIED as documented; raw-run reproducibility NEEDS VERIFICATION.

**Finding:** Set B changes feature schema as well as class-weighting context.

**Evidence:** Set B header omits `material_class` and includes `wind_weighted_score`; Set C contains both.

**Impact:** Set B versus Set A/C comparisons cannot isolate class weighting without matched-feature controls.

**Status:** VERIFIED.

**Finding:** The current source/config contains signs of version drift relative to the packed methodology notes: current default YAML uses 11 features and a 0.30 threshold, while the documented older experiments refer to different stage settings.

**Evidence:** current `default_experiment.yaml`, `repomix-output.xml`, Git history, and findings files.

**Impact:** Model/config/output provenance must be reconstructed before another run is called a replication.

**Status:** VERIFIED.

**Finding:** The dataset generator sets `neighbor_burning_count` to zero in generated training rows, while the current Optuna script filters to rows with `neighbor_burning_count > 0`.

**Evidence:** `Code/dataset_generator.py` writes a zero-filled neighbor column; `Code/train_rf_optuna.py` filters positive neighbor counts.

**Impact:** The current single-scenario generator output would be emptied or fall back to the full dataset during susceptible filtering; the multi-scenario CSV's generation semantics require inspection. This is a high-priority pipeline consistency issue.

**Status:** VERIFIED in source; effect on each existing CSV NEEDS VERIFICATION.

**Finding:** Spatial validation computes AUC-ROC from binary predicted perimeter masks rather than continuous probability surfaces.

**Evidence:** `Code/validation_engine.py` passes `y_pred_flat` to `roc_auc_score`.

**Impact:** The reported spatial AUC is a thresholded-mask discrimination statistic, not a probability-ranking AUC; interpretation must be corrected or the validator must receive probabilities.

**Status:** VERIFIED.

## 9. Technical Decisions Already Made

| Decision | Reason/evidence | Alternatives | State |
|---|---|---|---|
| Use a five-state CA with `np.int8` states. | Current engine and thesis architecture use states 1-5. | Other encodings were not documented. | Final for current architecture. |
| Use Moore-neighborhood convolution for burning-neighbor counts. | Current engine and thesis method. | Custom neighbor iteration would be slower and less consistent. | Final for current architecture. |
| Use `predict_proba()[:, 1]` for ML ignition probability. | Model contract and current engine. | Hard labels would discard calibration information. | Final interface decision. |
| Restrict inference to susceptible cells and process in chunks. | Prior memory failure and current engine implementation. | Full-grid inference caused an allocation failure. | Final implementation decision; benchmark still useful. |
| Compare Set A/B/C with threshold sweeps. | Needed to study CA calibration and experiment behavior. | A single default threshold is insufficient for observed sensitivity. | Method exists; protocol still under review. |
| Treat Set C as the current integration target. | Current default schema, model path, Optuna script, and YAML all point to the 11-feature pipeline. | Earlier 10-feature experiments remain comparison baselines. | Current decision, subject to provenance verification. |
| Do not attribute Set B differences solely to class weighting. | Set B also changes features by removing `material_class`. | Matched-feature ablation is needed. | Final caution; experiment design remains open. |

## 10. Experiment Comparison

| Category | Set A | Set B | Set C |
|---|---|---|---|
| Configuration | 10 parameters, standard baseline | 10 parameters, weighted-class experiment | 11 parameters |
| Features | 10-feature list in Set A section; no `wind_weighted_score` | 10-column header in `dataset-B.csv`; no `material_class`, includes wind score | Current 11-feature list; includes both `material_class` and wind score |
| Class weighting | NEEDS VERIFICATION | Weighted-class claim documented; exact mapping NEEDS VERIFICATION | Optuna choices `{1:3}`, `{1:5}`, `{1:7}`; selected `cw_1_7` in summary |
| Dataset | Exact file/run NEEDS VERIFICATION | `dataset-B.csv`, 21,079 rows, schema verified | `multi_scenario_dataset.csv`, 24,922 rows, schema verified |
| Model configuration | NEEDS VERIFICATION | NEEDS VERIFICATION | 100 trials, 5-fold stratified CV; selected 58 trees, depth 27, split 20, leaf 4, `cw_1_7` |
| Validation method | CA threshold sweep documented | CA threshold sweep documented | CA threshold sweep documented; raster validator exists |
| Key metrics | Best documented F1 0.603; best recall 0.767 | Best documented F1 0.624; best recall 0.936 | Best documented F1 0.728 and recall 0.827 at 0.30 |
| Strengths | Baseline isolates effect of wind score | Demonstrates weighted/feature variant and high recall at low threshold | Richest schema and current integration path |
| Weaknesses | Does not meet documented recall target | Confounds class weighting with feature changes; threshold-sensitive | Row-level CV F1 is only 0.13476; spatial result provenance needs verification |
| Current status | Historical baseline; artifact mapping unresolved | Historical/current branch baseline; exact run provenance unresolved | Current source/config target; artifacts and spatial replication unresolved |
| Needs verification | Dataset, weights, hyperparameters, raw outputs | Exact weights, model, run commands, raw outputs | CSV generation version, model-output pairing, leakage and split validity |

## 11. Open Problems / Unresolved Questions

1. Which exact CSV, model, YAML, seed, and command produced each Set A/B/C result?
2. Are the historical threshold metrics from the same validation implementation now in `validation_engine.py`?
3. Why does the current Optuna CV F1 (0.13476) differ so substantially from the documented spatial Set C metrics?
4. Does multi-scenario data contain repeated spatial cells or scenario-correlated rows across train and validation folds?
5. Is filtering `neighbor_burning_count > 0` valid when the generator writes zero for single-scenario rows?
6. Were class weights applied consistently, and was Set B compared against a matched-feature unweighted control?
7. Is `material_class` treated as a categorical variable appropriately by the RF, and are invalid classes excluded consistently?
8. Does `dataset_generator.py` preserve the ROI transform after cropping? It currently retains the original transform in the cropped metadata; the consequence for generated data provenance needs review.
9. Does the CA correctly apply the configured `proba_threshold`? The current engine uses stochastic comparison against `p_effective`; deterministic threshold-gating behavior described in older notes is not visible in the current `automata_engine.py` excerpt and must be reconciled.
10. Is spatial AUC intended to use probabilities rather than binary masks?
11. Are output GeoTIFFs tagged with nodata and experiment metadata, and can their model/config lineage be recovered?
12. Can the model's `feature_names_in_` safely align all three schemas, or should each experiment define an explicit schema manifest?
13. Are threshold sweeps selected on validation data and then reported on independent test data, or are they optimized on the same perimeter used for reporting?
14. Are the ignored local models and rasters backed up reproducibly? Current tracked source alone is insufficient to reconstruct every result.

## 12. Evidence and Source-of-Truth Rules

Use this hierarchy when facts conflict:

1. Actual datasets.
2. Actual experiment outputs and raw logs.
3. Source code.
4. Configuration files.
5. Saved model metadata and serialized artifacts.
6. Git history.
7. Thesis documentation.
8. Research notes.
9. Previous AI-generated summaries.

Previous notes can identify where to investigate but cannot override a dataset, output, or executable source. Never fill an unknown feature, hyperparameter, metric, dataset identity, or result by inference. Write `NEEDS VERIFICATION - not enough evidence available in the current session.`

## 13. Where We Are Right Now

- **Last major task completed:** documented Set A, Set B, and Set C threshold-sweep findings and integrated the current 11-feature RF/CA path.
- **Current task:** validation and verification of implementation correctness, preprocessing, data leakage, class imbalance, model provenance, metric semantics, and comparability.
- **Experiment currently most relevant:** Set C is the current source/config integration target, while Set A and Set B are historical comparison baselines.
- **Relevant files:** `Code/modules/feature_pipeline.py`, `Code/modules/model_trainer.py`, `Code/modules/automata_engine.py`, `Code/train_rf_optuna.py`, `Code/dataset_generator.py`, `Code/generate_multi_scenario.py`, `Code/config/default_experiment.yaml`, `Code/validation_engine.py`, the three root `docs/findings_Set*.md` files, and `Code/models/optuna_rf_search_summary.json`.
- **Known blockers:** local model/output artifacts exist but their experiment lineage and raw run logs are unresolved; Set A model/dataset identity is unresolved; current data-generation/filtering semantics may conflict; metric levels are mixed; current spatial AUC semantics are questionable.
- **Next inspection focus:** recover artifact provenance, audit dataset distributions and duplicate/spatial leakage, verify model feature names and class weights, then reproduce one Set C validation run from a clean manifest.

## 14. Recommended Next Actions

**P0 - Critical**

1. Inventory local ignored CSVs, models, rasters, outputs, timestamps, and serialized model metadata; produce a manifest linking each artifact to a set, schema, seed, and configuration.
2. Audit all three datasets for columns, row counts, class rates, neighbor-count distributions, duplicates, repeated scenarios, and spatial overlap; resolve the zero-neighbor/filter contradiction.
3. Reconstruct and run a clean Set C baseline with a saved config, explicit feature manifest, model metadata, and raw metric JSON; verify that the output model and CA use the same 11 columns.
4. Correct or document spatial AUC semantics before using it in thesis claims.

**P1 - Important**

5. Reproduce Set A and Set B with matched controls: same split protocol, seed, features where causal comparison is intended, and explicitly recorded class weights.
6. Test leakage-resistant validation, preferably grouped by scenario/fire event and spatial region rather than only random stratified rows.
7. Compare threshold selection on a calibration/validation partition against final metrics on an untouched test partition.

**P2 - Later**

8. Add automated schema/provenance tests and small synthetic raster tests.
9. Add model metadata sidecars containing feature names, dataset hash, class weights, seed, threshold, and training timestamp.
10. Update thesis methodology and findings only after the clean replications are complete.

## 15. Instructions for the New Codex Session

1. Read this handoff completely.
2. Inspect the referenced repository files and local artifacts before accepting any conclusion.
3. Verify every `NEEDS VERIFICATION` item that affects a result.
4. Do not modify code immediately; first confirm the three experiment schemas and artifact identities.
5. Compare this handoff against the current branch and report inconsistencies.
6. Keep RF and LR evidence separate.
7. Treat documented metrics as recorded claims until raw outputs and commands are recovered or reproduced.
8. Preserve the model-agnostic CA contract and generic `.pkl`/`.joblib` loading.
9. Propose the next smallest validation task only after the evidence inventory is complete.
10. Record new findings with evidence, impact, and one of the confidence labels used in this document.
