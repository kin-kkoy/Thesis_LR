# Project State — Random Forest Thesis Track

## Record contract

- **Purpose:** Canonical compact navigation and current-state index for Codex and thesis collaborators. It reduces repeated repository scans but does not replace claim-specific verification.
- **Authority:** Repository-local, Git-reviewable memory. Source code, identified data, immutable outputs, configurations, model metadata, and reproducible command records remain the evidence for their respective claims.
- **Scope:** Random Forest work and shared repository coordination. `Thesis_LR/` remains partner-owned and read-only.
- **Last updated:** 2026-09-15T00:15:19+08:00.
- **Last reviewed Git revision:** `2e49a71d426be1e682798a58f8cc3ee4d246eea0` on branch `RF-11-Params`.
- **Worktree note:** The review was performed with pre-existing uncommitted and untracked user work. A remembered fact is not evidence that those working files are unchanged.
- **Stale-information rule:** Treat a statement as stale when its linked source, configuration, data/model identity, relevant dependency, or reviewed Git revision has changed. Inspect the affected diff and evidence before reuse.
- **Size policy:** Keep this file below 16 KiB and focused on navigation, current verified state, and active blockers. Move durable rationale to `DECISIONS.md`; link to detailed evidence rather than copying it.

<!-- BEGIN CODEX DIGEST -->
## Compact startup digest

- **Active objective:** Validate and verify the Random Forest-assisted Cellular Automata pipeline before relying on prior thesis comparisons or results.
- **Ownership:** Write only authorized RF implementation paths. `Thesis_LR/**`, datasets, rasters, model artifacts, outputs, and manuscripts are read-only unless the thesis owner explicitly approves a change.
- **Current integration target:** Set C is the approved active development candidate for the 11-feature RF/CA pipeline. Set A and Set B remain separate historical chains. No cross-set metric comparison is valid without equivalent target, feature, grouping, split, evaluation, and provenance contracts.
- **Approved methodology contract:** `docs/ai/DECISIONS.md` D-009 defines next-timestep ignition as the primary RF estimand; group-first evaluation; separate training, validation, calibration, and final-test roles; distinct class-weight and threshold selection; separate estimator and CA metrics; duplicate-group containment; and immutable per-set provenance manifests.
- **Approved burnable domain:** Eligible RF/CA target cells must be raster-valid building cells. Invalid/nodata and non-building cells are excluded, use separate documented masks, and cannot enter evaluation as true negatives. Existing implementation and artifact compliance remain to be verified.
- **Execution boundary:** Contract formalization does not authorize training, dataset regeneration, simulations, model deserialization/replacement, raster generation, manuscript changes, immutable-artifact mutation, or any write under `Thesis_LR/**`.
- **Validation focus:** artifact provenance; feature/schema alignment; repeated-cell, scenario, and spatial leakage; class weighting versus threshold calibration; model/CA metric separation; and spatial AUC semantics.
- **Highest-priority blockers:** exact CSV/model/config/output lineage for Sets A/B/C; the zero-neighbor generation versus positive-neighbor filtering mismatch; leakage-resistant split evidence; and probability-versus-binary-mask AUC interpretation.
- **Primary entry points:** `Thesis_RF/Code/config/default_experiment.yaml`, `Thesis_RF/Code/modules/feature_pipeline.py`, `Thesis_RF/Code/modules/automata_engine.py`, `Thesis_RF/Code/modules/model_trainer.py`, `Thesis_RF/Code/dataset_generator.py`, `Thesis_RF/Code/generate_multi_scenario.py`, `Thesis_RF/Code/train_rf_optuna.py`, and `Thesis_RF/Code/validation_engine.py`.
- **Detailed leads:** `Thesis_RF/THESIS_RF_CODEX_HANDOFF.md` and `docs/findings_SetA.md`, `docs/findings_SetB.md`, `docs/findings_SetC.md`. These are navigation/evidence leads, not automatic proof of reproducibility.
- **Testing rule:** Reuse a prior test only when its revision or worktree state, command, environment, scope, inputs, timestamp, and result are recorded and still applicable. Otherwise run the smallest relevant verification.
- **Agent tooling:** Context7 MCP is configured project-locally in `.codex/config.toml`, with optional startup grace disabled so Codex waits for its configured 30-second startup timeout while building the tool catalog. `CODEX_HOME` is set at Windows user scope to the existing `C:\Users\gibgib\.codex` directory. A fresh trusted Codex CLI session on 2026-09-13 reported `context7: connected (2 tools)` and listed `resolve-library-id` and `query-docs`; the already-running task cannot acquire newly configured tools retroactively. Other agents retain the approved CLI fallback, while `rf_engineer` remains MCP-only.
<!-- END CODEX DIGEST -->

## Current verified state

| Status | Current fact | Evidence | Recheck trigger |
|---|---|---|---|
| `VERIFIED` | The active project priority is validation and verification of the RF track; LR is outside the writable scope. | `AGENTS.md`; `Thesis_RF/AGENTS.md` | Owner changes scope or ownership. |
| `VERIFIED` | RF implementation entry points are under `Thesis_RF/Code/`; generated datasets, rasters, models, checkpoints, and outputs are not normal source-edit targets. | `Thesis_RF/AGENTS.md`; current repository tree | Repository layout or guidance changes. |
| `VERIFIED` | The owner approved Set C as the active development candidate and preserved Sets A/B as separate historical chains with no unsupported cross-set metric comparison. | `docs/ai/DECISIONS.md` D-009 | Owner supersedes the decision or equivalent provenance is established. |
| `VERIFIED` | The approved primary RF estimand is next-timestep ignition of an eligible cell; final burn masks are CA-level outcome evidence, while any final-burn classifier is a separate susceptibility/outcome task. | `docs/ai/DECISIONS.md` D-009 | Owner changes the scientific estimand. |
| `VERIFIED` | The approved evaluation contract is group-first separation with distinct training, hyperparameter-validation, calibration/threshold, and untouched final-test roles. Class weighting and thresholding are separate selections. | `docs/ai/DECISIONS.md` D-009 | Owner supersedes the methodology contract. |
| `VERIFIED` | The approved physical burnable domain excludes non-building cells. Raster-validity and burnability masks remain separate, and neither invalid nor non-building cells may enter evaluation as true negatives. | `docs/ai/DECISIONS.md` D-009 | Ground-truth semantics or owner policy changes. |
| `VERIFIED` | Each Set A/B/C chain requires an immutable independent provenance manifest; missing lineage remains `NEEDS VERIFICATION` and existing artifacts cannot be overwritten. | `docs/ai/DECISIONS.md` D-009 | Owner supersedes provenance or artifact policy. |
| `VERIFIED` | The current feature pipeline and Optuna entry point define an ordered 11-feature schema containing `material_class`, `neighbor_burning_count`, and `wind_weighted_score`. | `Thesis_RF/Code/modules/feature_pipeline.py`; `Thesis_RF/Code/train_rf_optuna.py` | Either file or a model schema changes. |
| `VERIFIED` | The CA uses five `np.int8` states, convolution-based neighborhoods, susceptible-cell filtering, chunked `predict_proba()` inference, and a callable estimator interface. | `Thesis_RF/Code/modules/automata_engine.py`; `Thesis_RF/Code/orchestrator.py` | CA engine, feature pipeline, or model-loading boundary changes. |
| `VERIFIED` | Spatial validation currently supplies a binary perimeter mask to `roc_auc_score`, not a continuous probability surface. | `Thesis_RF/Code/validation_engine.py` | Validator inputs or metric implementation changes. |
| `VERIFIED` | Spatial validation sets invalid/nodata pixels to false in both binary masks but flattens the complete arrays, so invalid pixels enter the metrics as true negatives. | `Thesis_RF/Code/validation_engine.py` | Nodata indexing or metric-array construction changes. |
| `VERIFIED` | `ModelTrainer.evaluate()` selects an F1 threshold on `self.y_test` and reports threshold-dependent metrics on the same test partition. | `Thesis_RF/Code/modules/model_trainer.py` | Threshold-selection or evaluation partitioning changes. |
| `VERIFIED` | Current RF training paths use stratified row-level splitting/CV and do not implement grouped, spatial, temporal, or scenario-separated partitions. Actual leakage in an identified dataset remains unverified. | `Thesis_RF/Code/modules/model_trainer.py`; `Thesis_RF/Code/train_rf_optuna.py` | Split logic or active dataset lineage changes. |
| `NEEDS VERIFICATION` | Existing Set A/B/C metrics can be reproduced from identified datasets, models, configurations, seeds, commands, and raw outputs. | Leads: `docs/findings_SetA.md`, `docs/findings_SetB.md`, `docs/findings_SetC.md`, `Thesis_RF/THESIS_RF_CODEX_HANDOFF.md` | Close only with an experiment manifest and primary artifacts. |

## Active issues and unknowns

1. `NEEDS VERIFICATION` — Recover an explicit provenance chain for each Set A/B/C dataset, trained model, feature schema, class weights, seed, YAML/configuration, command, output, and metric record.
2. `WARNING` — `dataset_generator.py` records `neighbor_burning_count` as zero while `train_rf_optuna.py` filters for positive values, with a fallback when the filter empties the data. Determine the actual semantics of every retained dataset before training conclusions are accepted.
3. `FAIL` — Current row-level splits do not satisfy the approved group-first event/scenario/spatial-block contract. Actual historical split leakage remains `NEEDS VERIFICATION` until identified split membership is inspected.
4. `FAIL` — Current reporting does not satisfy the approved separation of estimator-level probability/calibration metrics from CA raster/temporal metrics.
5. `FAIL` — Current spatial AUC uses a thresholded binary mask and does not satisfy the approved probability ROC-AUC contract.
6. `FAIL` — Spatial validation currently retains invalid/nodata pixels as true negatives and does not yet enforce the approved separate validity and building-cell burnability masks.
7. `FAIL` — `ModelTrainer.evaluate()` tunes its F1 threshold on the reporting test partition and does not satisfy the approved training/validation/calibration/final-test separation.
8. `NEEDS VERIFICATION` — Confirm that the model feature names/order, current YAML, probability behavior, output raster, and validation command all belong to the same experiment.

## Known-good navigation map

| Need | Start here | Read further only if needed |
|---|---|---|
| Active configuration | `Thesis_RF/Code/config/default_experiment.yaml` | `Thesis_RF/Code/main.py`, `Thesis_RF/Code/orchestrator.py` |
| Raster alignment and nodata | `Thesis_RF/Code/modules/data_loader.py` | Identified raster metadata and quality reports |
| Feature contract | `Thesis_RF/Code/modules/feature_pipeline.py` | Saved model `feature_names_in_` or an approved schema manifest |
| CA behavior | `Thesis_RF/Code/modules/automata_engine.py` | Configuration plus focused synthetic-array tests |
| Dataset construction | `Thesis_RF/Code/dataset_generator.py`, `Thesis_RF/Code/generate_multi_scenario.py` | Identified CSV metadata and generation logs |
| RF training | `Thesis_RF/Code/modules/model_trainer.py`, `Thesis_RF/Code/train_rf_optuna.py` | Model sidecar/serialized metadata and split records |
| Spatial metrics | `Thesis_RF/Code/validation_engine.py` | Identified output/ground-truth rasters and raw metric record |
| Historical experiment leads | `docs/findings_SetA.md`, `docs/findings_SetB.md`, `docs/findings_SetC.md` | Primary artifacts named by each claim |
| Durable project decisions | `docs/ai/DECISIONS.md` | Evidence linked from the relevant decision |

## Test evidence available for reuse

No current RF runtime or experiment test result has been established. A controlled source-only four-agent dry run on 2026-09-13 verified the code-level validation findings above but did not inspect active datasets or artifacts, train models, or execute the RF pipeline. Configuration syntax checks for Codex profiles are not evidence that the RF pipeline passes. Before reusing any older result, require the complete test-evidence record defined above.

## Update procedure

1. Inspect the Git delta since the last reviewed state and the evidence relevant to the task.
2. Update only facts affected by a material, verified change; preserve uncertainty labels.
3. Update the timestamp and reviewed revision/worktree description.
4. Append to `DECISIONS.md` only when a durable choice, approval, or supersession occurred.
5. Review the implementation diff and memory diff together.
6. Keep detailed logs, output tables, citations, and experiment artifacts in their evidence locations and link them here.
