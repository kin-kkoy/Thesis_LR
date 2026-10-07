# Phase 13A — RF tuning and uncalibrated refit protocol

- **Protocol ID:** `phase13_set_c_rf_protocol.v1`
- **Revision:** Draft 1, 2026-10-06 (Asia/Manila).
- **Status:** PROPOSED — requires independent review and owner acceptance. D-055 authorizes planning only.
- **Purpose:** Resolve every numerical and execution choice raised in the owner-provided Phase 13A.1 validator report. This file is the complete proposal for re-review; D-055 alone is not the numerical specification.
- **Evidence distinction:** Package identities/counts below are recorded accepted evidence from D-053/D-054, not newly observed reads. Search and resource choices are prospective design judgments, not measured optimal settings or demonstrated production capacity.

## 1. Immutable input and population contract

Use only `C:\Users\gibgib\Thesis-Cellular-Automata\SetC_Development\set_c_v1`.

| Identity | Accepted SHA-256 |
|---|---|
| Development `attestation.json` | `4fcf286156892c9da0289e56712210e90cccdcd608641c7ee573a8db1dcea23f` |
| `train.csv` | `20987f233ce6ad8fd8fb030b0c80eec9505a63efedcd264c9c669f3c7ba593e2` |
| `validation.csv` | `64b4949564b5f064ae150860598056edec0b9fbdfff582b3946a0c790d84e4a3` |

The source CSV identity `7f0f1867445141f365687d127e449117f234348cec571990d70bc2b226ba40f6` and source manifest identity `b700f1b9f196893f23e7190daf1bc3c68b0f569f99a977ec9e919f0de73e9e3c` are attestation-bound provenance. The Phase 13 path must not open those canonical files to verify them. Its training-source revision is a future clean reviewed commit, distinct from the release's accepted export revision `8b750ae733ab0346d5cb9291780310c87c2cc752`.

| Role | Recorded components | Rows | Positive | Negative | Phase 13 use |
|---|---:|---:|---:|---:|---|
| train | 11 | 4541 | 338 | 4203 | All trial fitting |
| validation | 5 | 2064 | 153 | 1911 | Trial scoring and selection |
| train + validation | 16 | 6605 | 491 | 6114 | One final uncalibrated refit after selection |

The accepted roles remain unchanged. No random row split, regrouping, new allocation or reconstruction of component IDs is permitted. The development projection omits those IDs, so the design is a fixed group-disjoint holdout; it does not implement grouped CV or support component-level uncertainty calculations.

The selective loader may inspect the development directory inventory and pinned attestation, then read/hash only train and validation CSV contents. Calibration file contents remain unopened, including for hashing. No protected directory, complete mapping, final-test rows, canonical source, incumbent model, raster or ground truth is needed. No package modification or re-export is permitted.

## 2. Feature and target contract

Predictors retain this exact order: `slope_risk`, `proximity_risk`, `building_presence`, `material_risk`, `material_class`, `wind_speed`, `wind_sin`, `wind_cos`, `neighbor_burning_count`, `composite_flammability`, `wind_weighted_score`.

Use named feature columns represented as `np.float32`; binary labels as `np.int8`. Reject reordered/missing/extra predictors, non-finite or out-of-domain values, and missing classes. The positive label is integer `1`; locate its probability column through `classes_` and the existing positive-class helper.

Preserve the accepted `all_eligible` population and encodings. No row dropping, imputation, scaling, resampling, feature selection, one-hot expansion, SMOTE, manually specified sample weights or numeric class-cost ratios are proposed. These exclusions concern preprocessing; ordinary RF bootstrap sampling remains enabled below.

## 3. Objective, search and baseline candidates

Each trial fits a fresh RandomForestClassifier on train only. Maximize pooled validation **average precision (AP)** using continuous probabilities for positive label 1, unweighted across the retained validation observations. AP is not trapezoidal PR-AUC. Validation labels/probabilities must retain identical row ordering; compute and retain full finite precision for selection, rather than rounded display values.

All searched parameters use the following ordered categorical choices; no unspecified continuous/integer range or adaptive range expansion is allowed:

| Parameter | Exact ordered choices |
|---|---|
| `n_estimators` | `[200, 400, 600]` |
| `max_depth` | `[null, 5, 10, 15, 20]` (`null` maps to estimator `None`) |
| `min_samples_split` | `[2, 5, 10, 20]` |
| `min_samples_leaf` | `[1, 2, 4, 8, 16]` |
| `max_features` | `["sqrt", 0.5, 1.0]` |
| `class_weight` token | `["none", "balanced", "balanced_subsample"]` |

Fixed estimator values are `criterion="gini"`, `bootstrap=True`, `random_state=42`, `n_jobs=2`, `oob_score=False`, `warm_start=False`, `max_samples=None`, `max_leaf_nodes=None`, `min_weight_fraction_leaf=0.0`, `min_impurity_decrease=0.0`, `ccp_alpha=0.0`, `monotonic_cst=None`, and `verbose=0`. Record the complete resolved estimator parameter dictionary under the pinned environment. The `none` token maps to `class_weight=None`; the other tokens retain their documented data-derived behavior.

Trials 0, 1 and 2 are enqueued baselines in weight order `none`, `balanced`, `balanced_subsample`. Each uses `n_estimators=200`, `max_depth=15`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features="sqrt"` and all fixed values above. They are ordinary eligible selection candidates and count within the 60-trial total. Their paired settings permit a descriptive comparison of weighting strategies on the same validation population, not an independent performance test.

**Design rationale:** Shallower and unconstrained-depth candidates cover alternatives omitted by the legacy depth 10–30 range. Leaf sizes through 16 cover progressively smoother forests; feature subsampling tests randomness/regularization alternatives without changing the 11-feature schema. The tree grid bounds computation, while 60 evaluations bound repeated use of validation. None of these numerical choices is asserted optimal. Selection must not be changed after looking at results because the score is disappointing.

## 4. Study, reproducibility, ties and stopping

- Start one fresh single-objective study, direction `maximize`, with no imported trials, automatic resumption, distributed workers or parallel trials.
- Use `TPESampler(seed=42, n_startup_trials=10, multivariate=False, group=False, constant_liar=False)` and `NopPruner`. Record the full resolved sampler settings/version; remaining settings use the pinned Optuna 4.8.0 defaults. No custom sampler callbacks, categorical distance or constraint functions are proposed. The three completed baseline trials count toward startup history; with no failures/pruning, the first ten trials are startup observations including these baselines.
- Require exactly **60 successfully completed trials**, including the three baselines. Run study `n_jobs=1`, forest `n_jobs=2`; never use `-1`. Duplicate sampled parameter combinations are allowed and consume budget; 60 trials does not imply 60 unique configurations.
- Keep feature/label row order fixed within each role; concatenate train before validation for refit. Use forest seed 42 for every fit. Do not search seeds. Record Python and dependency versions plus platform, processor count and worker settings; make no cross-environment bitwise-reproducibility claim.
- Choose the maximum full-precision validation AP. On an exact equality of stored finite scores, select the smallest trial number; use no epsilon tie interval or secondary metric.
- Set the study timeout to **5400 seconds**, measured from the start of study optimization including baselines. It limits the optimization loop and must not be described as forcibly terminating an active fit. A last running trial may finish after this soft boundary. If fewer than 60 complete, mark the experiment incomplete and do not select/promote a best-so-far model or perform the final refit.
- No pruning or skipping failed trials. Unexpected fitting/scoring exceptions, invalid probabilities/metrics, non-finite AP or evidence-write failures stop the experiment. Record available trial state and diagnostics; the complete package must not be published. Do not silently catch failures and continue with a reduced budget.

The primary objective is AP only. After selecting the train-only model, descriptive validation ROC-AUC, Brier score and log loss may be retained from its probabilities. They do not break ties, alter selection, choose a threshold or select calibration methods. Record the validation positive fraction as the AP baseline reference.

## 5. Resource envelope and failure disposition

These are proposed procedural limits for the human-run experiment, not empirically established requirements:

| Control | Exact proposed rule |
|---|---|
| Available physical memory | Start only at or above 4 GiB; stop if a sample is below 4 GiB |
| Free disk on C: | Start only at or above 2 GiB; stop if a sample is below 2 GiB |
| Monitoring cadence | Operator records both resource readings at least once every 60 seconds during execution |
| Overall elapsed time | Operator stops at 7200 seconds from training-command process launch, including loading, study, refit and publication |
| Parallelism | One trial; two forest workers; no second simultaneous study |

Record start/end timestamps, elapsed time, resource samples and observed minima, actual exit status and stop reason. Sampling does not establish continuous minima or peak process RSS. Optuna's 5400-second timeout is not the 7200-second overall stop. The eventual reviewed PowerShell run card must specify tested capture, how to interrupt the identified process, and how to verify it has exited; no new generic supervisor framework is presumed necessary.

Preflight must check the exact clean source revision, approved configuration/protocol hashes, input/path/reparse/attestation/train/validation identities, required versions/imports, resource floors, and absence of the approved experiment destination or interrupted staging. The prior Python location warning is retained and classified; import success alone does not resolve its origin. If checks fail, do not begin fitting. Do not mutate the environment to bypass a failed gate.

An interruption, resource breach, missing evidence, trial failure, incomplete study, refit failure or failed publication leaves the experiment incomplete. Retain logs and surviving partial artifacts; do not overwrite, automatically retry/resume, promote a partial model or silently clean up. A successor experiment or recovery needs an explicit scope and a fresh output identity. A success requires all 60 trials, one final refit, complete retained evidence and a valid complete package within the overall envelope.

## 6. Selection freeze and one uncalibrated refit

Retain the actual selected train-only forest from the successful study; reuse its fitted object for selection-model serialization rather than fitting it again. Record/freeze its trial number, AP, chosen hyperparameters and class-weight token before refitting.

Fit one new uncalibrated forest on train followed by validation (6605 recorded rows, 491 positives, 6114 negatives) with the selected parameters and fixed settings. Retain the selected weighting strategy, not manually frozen numeric weights: `balanced` is recomputed from the refit labels; `balanced_subsample` is recomputed per refit bootstrap sample. This is the documented strategy applied to the new fit population.

The selected model's validation AP describes the train-only selection model. The refitted base model has used validation labels, so its score on those same rows must not be presented as held-out performance. Phase 13 does not fit calibration, select calibration method or reporting threshold, or perform final testing/CA inference. The two joblib objects are uncalibrated development artifacts, not the final deployment model.

## 7. Candidate package and record contract

The proposed final destination is `C:\Users\gibgib\Thesis-Cellular-Automata\Thesis_RF\Code\output\phase13\set_c_rf_v1`. Do not create it during planning/source-only testing. It must be published as a new complete package without replacing an incumbent model or any existing experiment. Synthetic publication tests use OS-temporary paths only. Use the existing small publication/serialization patterns where suitable; avoid a separate framework.

Expected contents: `selection_rf.joblib`, `base_rf.joblib`, `trials.jsonl`, `selection_validation_metrics.json`, an exact configuration copy, retained run diagnostics/evidence references, and `manifest.json`. An additional existing capture log may remain in its approved evidence path with its hash referenced; its location is finalized in the execution run card before launch. All named package files are hashed/size-recorded by the final manifest, excluding that manifest's self-hash, which is recorded by the completion receipt/output.

The manifest schema proposal is `set_c_rf_phase13_run.v1`, artifact kind `set_c_rf_uncalibrated_candidate_run`. Required information:

| Field group | Required information |
|---|---|
| Identity/state | Schema, artifact kind, protocol ID/document SHA-256, run ID, approval reference, package version, completion status and output paths |
| Inputs | Exact development path, pinned attestation/train/validation SHA-256, source CSV/manifest and export revision recorded from accepted provenance; no canonical membership hash recomputation |
| Training source | Clean full training-code commit, dirty state false, exact command/working directory, configuration hash and full resolved configuration |
| Environment | Python, scikit-learn, Optuna, NumPy, pandas, SciPy, joblib and YAML versions, platform, worker settings and retained warnings |
| Schema/population | Ordered 11 names, feature/target schema versions, dtypes, positive label/classes, recorded/validated role row/class totals, fitting/selection/refit roles |
| Study | Direction/objective, sampler/pruner/resolved settings, seeds, exact search choices, baseline definitions/order, budget/timeout, all trial numbers/parameters/states/full AP values and timestamps, selected trial and exact tie policy |
| Models | Distinct selected/refit identities, full estimator settings, feature_names_in_/classes_, training population for each, joblib filenames/hash/size |
| Metrics | Selected train-only validation AP, descriptive ROC-AUC/Brier/log loss and AP prevalence reference; explicit selection-score limitation and no refit held-out score |
| Excluded operations | Calibration method null, threshold null, calibration contents not opened, final test not loaded/evaluated, CA not run |
| Execution/evidence | Start/end/elapsed, resource policy/samples/observed minima, actual exit status, stop reason, logs/tracebacks/capture references and hashes, final completion verification |

Preserve the generic estimator interface (`predict_proba`, ordered feature names and `classes_`) and existing generic `.joblib` `load_model()` architecture; no CA or LR edit is part of this proposal. Artifact deserialization for the later independent result review requires its own explicitly scoped read-only authority for these newly created artifacts.

## 8. Acceptance and remaining phases

1. Independent validator re-reviews this complete file and identifies its exact SHA-256; return PASS, corrections or unresolved assumptions without changing source or reading real data.
2. Owner accepts that identified protocol. Parent then issues the exact source-only 13B loading prompt. Approval of the protocol is not production execution approval.
3. 13B adds the selective train/validation loader and focused synthetic tests while preserving the accepted default loader/exporter/schema. 13C adapts the tuning CLI/config/candidate writer and creates the tested human run card. Separate these phases because 13C depends on 13B.
4. Independent 13D review accepts the settled source/synthetic tests and run card; obtain a clean reviewed commit. The human's separate execution approval binds that commit, input/config/protocol identity, destinations and limits.
5. The human runs the reviewed preflight and command in PowerShell. No agent stays active merely to wait for training. Preserve all evidence and report success or failure to the parent.
6. An independently authorized result review may load only these trusted new joblib artifacts and train/validation inputs, recompute selection-model validation metrics without refitting, and verify provenance/package hashes.
7. Owner acceptance closes Phase 13. Phase 14 planning then addresses calibration and later frozen-estimator testing under their separate scopes.

Repeated adaptive use of validation can overfit selection AP; its magnitude is unknown. Five validation components are limited evidence, and no confidence interval, row bootstrap, physical independence, teacher-independent temporal truth or broad real-world generalization claim follows. A complete 60-trial study demonstrates completion of this bounded search, not global optimality or measured resource sufficiency for other runs.

## Primary documentation

- [Scikit-learn 1.8 average precision](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.average_precision_score.html): objective definition and distinction from interpolated PR area.
- [Scikit-learn 1.8 RandomForestClassifier](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html): estimator parameters, weighting, parallelism and seed semantics.
- [Optuna 4.8 Study](https://optuna.readthedocs.io/en/v4.8.0/reference/generated/optuna.study.Study.html): queued trials, trial count, stopping, exceptions and parallelism.
- [Optuna 4.8 reproducibility FAQ](https://optuna.readthedocs.io/en/v4.8.0/faq.html): seed and sequential-study limits.

Numerical search/resource values are the thesis owner's proposed prospective design choices to review; these documentation sources do not establish them as optimal or empirically sufficient.
