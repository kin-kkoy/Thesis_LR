# Project State — Random Forest Thesis Track

## Record contract

- **Purpose:** Canonical compact navigation and current-state index for Codex and thesis collaborators. It reduces repeated repository scans but does not replace claim-specific verification.
- **Authority:** Repository-local, Git-reviewable memory. Source code, identified data, immutable outputs, configurations, model metadata, and reproducible command records remain the evidence for their respective claims.
- **Scope:** Random Forest work and shared repository coordination. `Thesis_LR/` remains partner-owned and read-only.
- **Last updated:** 2026-09-22T10:49:26+08:00.
- **Last reviewed Git revision:** `9e3a8d87e709c78a0c7937f546c9e6b2204a01bd` on branch `RF-11-Params`.
- **Worktree note:** The reviewed D-013 source/test/memory delta is based on the recorded revision and is intended for one scoped commit. A remembered fact is not evidence that later working files remain unchanged.
- **Stale-information rule:** Treat a statement as stale when its linked source, configuration, data/model identity, relevant dependency, or reviewed Git revision has changed. Inspect the affected diff and evidence before reuse.
- **Size policy:** Keep this file below 16 KiB and focused on navigation, current verified state, and active blockers. Move durable rationale to `DECISIONS.md`; link to detailed evidence rather than copying it.

<!-- BEGIN CODEX DIGEST -->
## Compact startup digest

- **Active objective:** Validate and verify the Random Forest-assisted Cellular Automata pipeline before relying on prior thesis comparisons or results.
- **Ownership:** Write only authorized RF implementation paths. `Thesis_LR/**`, datasets, rasters, model artifacts, outputs, and manuscripts are read-only unless the thesis owner explicitly approves a change.
- **Current integration target:** Set C is the approved active development candidate for the 11-feature RF/CA pipeline. Set A and Set B remain separate historical chains. No cross-set metric comparison is valid without equivalent target, feature, grouping, split, evaluation, and provenance contracts.
- **Approved methodology contract:** `docs/ai/DECISIONS.md` D-009 defines next-timestep ignition as the primary RF estimand; group-first evaluation; separate training, validation, calibration, and final-test roles; distinct class-weight and threshold selection; separate estimator and CA metrics; duplicate-group containment; and immutable per-set provenance manifests.
- **Approved burnable domain:** Eligible RF/CA target cells must be raster-valid building cells. Invalid/nodata and non-building cells are excluded, use separate documented masks, and cannot enter evaluation as true negatives. Existing implementation and artifact compliance remain to be verified.
- **Approved Phase 4 refinement:** `docs/ai/DECISIONS.md` D-010 fixes the Set C order and positive label, keeps observation metadata out of predictors, stores split assignments separately, and distinguishes frozen classification-reporting thresholds from threshold-free stochastic CA probabilities.
- **Approved next-stage contracts:** `docs/ai/DECISIONS.md` D-013 fixes meteorological from-bearing wind relative to raster grid north, makes common-valid mapped-building agreement the primary CA final-footprint domain, and approves a source-only in-memory transition observer with BLAZING-at-`t` feature semantics, newly-IGNITED-at-`t+1` labels, mutation isolation, and complete simulated provenance.
- **Current source-only worktree:** D-009–D-013 source contracts are implemented: centralized `simulated_wind.v1`, common-valid mapped-building evaluation, and a provenance-bound five-state `np.int8` transition observer. Its disabled-by-default orchestrator path suppresses output; pure `ca_transition_rows.v1` retains all eligible rows in memory; legacy binary publication fails closed.
- **Execution boundary:** Contract formalization does not authorize training, dataset regeneration, simulations, model deserialization/replacement, raster generation, manuscript changes, immutable-artifact mutation, or any write under `Thesis_LR/**`.
- **Validation focus:** artifact provenance; feature/schema alignment; repeated-cell, scenario, and spatial leakage; class weighting versus threshold calibration; model/CA metric separation; and spatial AUC semantics.
- **Highest-priority blockers:** establish trusted real-run provenance and approved in-memory collector policy; establish group/block feasibility evidence; and obtain stage-specific authorization before any simulation or artifact production.
- **Primary entry points:** `Thesis_RF/Code/config/default_experiment.yaml`, `Thesis_RF/Code/modules/feature_pipeline.py`, `Thesis_RF/Code/modules/automata_engine.py`, `Thesis_RF/Code/modules/model_trainer.py`, `Thesis_RF/Code/dataset_generator.py`, `Thesis_RF/Code/generate_multi_scenario.py`, `Thesis_RF/Code/train_rf_optuna.py`, and `Thesis_RF/Code/validation_engine.py`.
- **Detailed leads:** `Thesis_RF/THESIS_RF_CODEX_HANDOFF.md` and `docs/findings_SetA.md`, `docs/findings_SetB.md`, `docs/findings_SetC.md`. These are navigation/evidence leads, not automatic proof of reproducibility.
- **Testing rule:** Reuse a prior test only when its revision or worktree state, command, environment, scope, inputs, timestamp, and result are recorded and still applicable. Otherwise run the smallest relevant verification.
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
| `VERIFIED` | Set C uses `all_eligible`; approved class-weight candidates are `none`, `balanced`, and `balanced_subsample`. | `docs/ai/DECISIONS.md` D-011; current RF source/config worktree | Owner supersedes D-011 or the source contracts change. |
| `VERIFIED` | Active Set C accepts only provenance-bound `authorized_simulated` adjacent-state pairs. The satellite-derived manual burn mask is an approximate CA final-footprint reference, not observed temporal or calibration evidence; claims are limited to the case-study area and tested simulated scenarios. | `docs/ai/DECISIONS.md` D-012; current RF source/config worktree | Owner supersedes D-012 or the source contracts change. |
| `VERIFIED` | D-013 source contracts are implemented: from-bearing wind, common-valid building evaluation, and a mutation-isolated five-state observer with BLAZING-at-`t` predictors and eligible `2→3` labels. Observer mode suppresses output/checkpoints; pure rows retain all eligible cells; legacy binary publication fails closed. | D-013; cited RF source; 109 synthetic tests and RF-validator acceptance, 2026-09-22 | Source/test or production integration changes. |
| `VERIFIED` | The aligned approximate reference mask contains 3,312 burned cells. Of these, 1,393 are outside the mapped-building mask; 20 coincide with slope-invalid cells, including 9 mapped-building cells. A building-only final-footprint evaluation therefore measures building-fire agreement, not the complete manually delineated footprint. | Read-only raster preflight on `stack_ground_truth.tif`, `stack_buildings.tif`, and `stack_slope_final.tif`, 2026-09-17 | Any raster identity/content, mask value semantics, or evaluation-domain decision changes. |
| `VERIFIED` | The current feature pipeline and Optuna entry point define an ordered 11-feature schema containing `material_class`, `neighbor_burning_count`, and `wind_weighted_score`. | `Thesis_RF/Code/modules/feature_pipeline.py`; `Thesis_RF/Code/train_rf_optuna.py` | Either file or a model schema changes. |
| `VERIFIED` | The CA uses five `np.int8` states, convolution-based neighborhoods, susceptible-cell filtering, chunked `predict_proba()` inference, and a callable estimator interface. | `Thesis_RF/Code/modules/automata_engine.py`; `Thesis_RF/Code/orchestrator.py` | CA engine, feature pipeline, or model-loading boundary changes. |
| `VERIFIED` | The current Phase 4 worktree separates continuous estimator metrics from hard CA-mask metrics and excludes invalid/non-building cells before flattening. | `Thesis_RF/Code/validation_engine.py`; synthetic smoke check | Validator or domain-mask logic changes. |
| `VERIFIED` | The current Phase 4 worktree no longer selects a threshold on the reporting test. A hard reporting threshold requires a calibration-origin/objective record; stochastic CA uses the positive-class probability unchanged. | `Thesis_RF/Code/modules/model_trainer.py`; `Thesis_RF/Code/modules/automata_engine.py`; synthetic smoke check | Evaluation or CA inference logic changes. |
| `VERIFIED` | The current Phase 4 worktree consumes preassigned roles from a separate split manifest and rejects scenario, spatial-block, duplicate-group, cell, and multi-event overlap. Actual historical leakage remains unverified. | `Thesis_RF/Code/modules/model_trainer.py`; `Thesis_RF/Code/train_rf_optuna.py`; synthetic smoke check | Split logic or active dataset lineage changes. |
| `VERIFIED` | The current Phase 4 worktree rejects unbound wind permutations, extra or missing split assignments, single-class split roles, stale calibration-selection records, and partial publication of a model/study/summary version set. | `Thesis_RF/Code/generate_multi_scenario.py`; `Thesis_RF/Code/modules/model_trainer.py`; `Thesis_RF/Code/train_rf_optuna.py`; direct contract smoke check | Scenario construction, manifests, split logic, calibration selection, or artifact publication changes. |
| `NEEDS VERIFICATION` | Existing Set A/B/C metrics can be reproduced from identified datasets, models, configurations, seeds, commands, and raw outputs. | Leads: `docs/findings_SetA.md`, `docs/findings_SetB.md`, `docs/findings_SetC.md`, `Thesis_RF/THESIS_RF_CODEX_HANDOFF.md` | Close only with an experiment manifest and primary artifacts. |

## Active issues and unknowns

1. `NEEDS VERIFICATION` — Recover an explicit provenance chain for each Set A/B/C dataset, trained model, feature schema, class weights, seed, YAML/configuration, command, output, and metric record.
2. `NEEDS VERIFICATION` — Pure `ca_transition_rows.v1` retains all eligible rows but publishes nothing and contains only dynamic fields. A full 11-feature publisher needs a new decision; legacy binary publication fails closed.
3. `NEEDS VERIFICATION` — Establish group inventory and spatial-correlation/feasibility evidence before approving block scale or train/validation/calibration/test allocation.
4. `NEEDS VERIFICATION` — Select a calibration method using grouped non-test evidence; approve a cost/safety objective before deriving any hard reporting threshold.
5. `NEEDS EVIDENCE` — Historical transition artifacts, if any, have not been checked against `simulated_wind.v1`; source now rejects missing, noncanonical, or scenario-mismatched originating wind provenance.
6. `NEEDS VERIFICATION` — Before training, approve the RF search space, trial budget, tuning objective, calibration candidates/design, and actual group-first allocation. Nested grouped CV remains fail-closed until group inventory proves it necessary.
7. `WARNING` — Phase 4 source guardrails passed synthetic verification but have not been exercised against real datasets, models, rasters, or production simulations. Caller-attested provenance and external-callback behavior remain trusted boundaries; measured impact remains unverified.
8. `WARNING` — D-013 makes common-valid mapped-building agreement the primary CA final-footprint domain. Source now requires the exact simulation-valid mask and dynamically reports exclusions, but no real-output validation was run. The preflight found 1,393 of 3,312 reference-burned cells outside mapped buildings and 20 on slope-invalid cells (9 also mapped as buildings); broader-domain metrics and artifact production remain unauthorized.
9. `NEEDS VERIFICATION` — Confirm that the model feature names/order, current YAML, probability behavior, output raster, and validation command all belong to the same experiment.

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

No RF experiment result is established. On 2026-09-22, all 109 synthetic RF contract tests passed with caches disabled; the RF Validator independently passed 63 focused observer tests. Coverage includes timing/state/label semantics, provenance, noninterference, output suppression, legacy rejection, wind, evaluation, and training contracts. No production or artifact operation ran.

## Update procedure

1. Inspect the Git delta since the last reviewed state and the evidence relevant to the task.
2. Update only facts affected by a material, verified change; preserve uncertainty labels.
3. Update the timestamp and reviewed revision/worktree description.
4. Append to `DECISIONS.md` only when a durable choice, approval, or supersession occurred.
5. Review the implementation diff and memory diff together.
6. Keep detailed logs, output tables, citations, and experiment artifacts in their evidence locations and link them here.
