# Phase 14A Calibration and Final-Test Protocol Proposal

**Task:** 14A.1 — documentation-only proposal  
**Proposal ID:** `phase14a_calibration_final_test_protocol.v1`  
**Draft:** 4  
**Draft 1 identity:** same proposal ID; prior SHA-256 `a666be188c1198084b47c575b8b31ab89437fab2d52e6f33b8af67e100580398`  
**Draft 2 identity:** same proposal ID; prior SHA-256 `8834d2578451521723c52593e26acecad519ce2a9f5afb767d10f22c307d7c09`  
**Draft 3 identity:** same proposal ID; prior SHA-256 `fdc50bf1a7b72162def184e39d62844df8b07d837f706f004d96e1cc173357b4`  
**Status:** PROPOSED — owner methodology/reporting acceptance required; no source or execution authorization  
**Classification:** SHARED  
**Prepared:** 2026-10-09 (Asia/Manila)

This proposal binds later work to the accepted Phase 13 candidate while keeping calibration and final testing as separate approvals. It describes a candidate protocol, not an accepted result or an empirical finding. All future execution remains denied until the applicable owner approval is recorded.

### Draft 2 correction summary (preserved)

1. Separates one primary final evaluation from one separately authorized, bounded independent numerical replay on the identical locked model, rows, and configuration.
2. Separates immutable execution receipts from hash-linked review records; reviewer dispositions are excluded from execution receipts.
3. Names the future Phase 14 supervised launch, working directory, shared supervision core/adapter, and exact synthetic contract-test scope. The Phase 12 and Phase 13 launchers are not used unchanged.
4. Pins the Phase 14 calibration/final-test configurations and versioned output, report, receipt, and evidence paths.
5. Defines reliability-bin interval endpoints, treatment of `p=1`, and null values for both empty-bin statistics.
6. Pins the accepted D-053/D-054 adequacy review and exact pair identities while stating that release metadata is not a concentration certificate and no groups will be reconstructed.
7. Lists the exact published `set_c_final_test_rows.v1` columns in order, including authoritative metadata and trailing `leakage_component_id`, with no `split_role`.

### Draft 3 correction summary

1. Adds a two-revision freeze sequence: calibration runs on its pinned calibration revision; after calibrated-package review, the final-evaluation config is finalized and independently reviewed, then separately owner-approved and committed; an independent reviewer verifies the clean evaluation revision and config hash; then the owner approves the external pre-test lock before any final-row access. Config, commit, and lock identities are linked without self-hash or circular dependencies.
2. Defines receipt inventory boundaries and finalization order: receipts exclude themselves and later review records, are hashed externally by the supervisor completion record, and are never rewritten.

### Draft 4 correction summary

1. Separates child computation-stage `execution_receipt.json` from post-exit/bounded-shutdown `supervisor_completion.json`; absent child receipts remain missing, never synthesized.
2. Makes overall success contingent on verified child outputs and receipt plus normal supervisory completion, observed exit 0, completed capture, resolved process state, and no relevant finalization errors. Later review records bind both records; self/forward-hash exclusions and immutable failure retention remain explicit.

## 1. Immutable input identity

The sole accepted base-model candidate is the Phase 13 run `phase13-setc-rf-61e8f740-r2`, produced from source revision `61e8f74031d61c9be22a9b6408aec3ad74f4e892`.

| Input | Accepted identity |
|---|---|
| Phase 13 protocol | `phase13_set_c_rf_protocol.v1`; SHA-256 `69c39d1d97430ebb903ec2d856e6205a2c1b8ad77641798955d6ad2bdb671637` |
| Phase 13 configuration | SHA-256 `3800fe006463392f102a85ca3c491a6386b5340aeb8d2211eb31ae079a63bfc8` |
| Phase 13 run manifest | SHA-256 `b0d0d976a51d8559db73f9ee0db8365c0133c4414de5242cd328d0eb7ff96f3c` |
| `base_rf.joblib` | SHA-256 `7c3b431dcad6554d9ea80e0e6b0d13826245f8fcf3de9534ccfde78bd06b4ce1` |
| Accepted development release attestation | SHA-256 `4fcf286156892c9da0289e56712210e90cccdcd608641c7ee573a8db1dcea23f` |
| Accepted source CSV | SHA-256 `7f0f1867445141f365687d127e449117f234348cec571990d70bc2b226ba40f6` |
| Accepted source manifest | SHA-256 `b700f1b9f196893f23e7190daf1bc3c68b0f569f99a977ec9e919f0de73e9e3c` |
| Phase 13 release path | `SetC_Development/set_c_v1` |

These identities are copied from the accepted decision records; the artifacts are not reopened or rehashed for this proposal. Phase 13 outputs, both model files, receipts, logs, releases, and earlier evidence remain immutable. New artifacts must use new versioned destinations and exclusive creation; no existing path may be overwritten.

The recorded Phase 13 environment is Python 3.14.3, scikit-learn 1.8.0, Optuna 4.8.0, Joblib 1.5.3, and NumPy 2.4.4. This is the recorded environment identity, not a fresh environment check. A later run must record its actual interpreter and package versions and fail closed if they do not match the owner-approved environment lock.

## 2. Calibration-method policy

D-010 requires grouped non-test evidence to select a calibration method. The sanitized accepted development projection does not expose group identifiers, so the current source cannot produce that evidence without a new authorized data/design basis. No group IDs may be invented, reconstructed from protected membership, or inferred from row order. Ordinary row-level cross-validation is not grouped validation. Calibration resubstitution scores are not independent method-selection evidence, and final-test data or scores must never select a method.

**Minimal proposal:** prespecify one sigmoid calibration method before fitting. It is a prospective policy choice, not an empirically selected method, a claim of suitability, or proof of superiority. The owner must explicitly approve the methodology/legacy-contract adjustment to D-010 and the calibration-selection-record contract before implementation. The current Phase 13 helper is not that implementation: `train_rf_optuna.py::fit_calibrated_model` first fits a new forest on train plus validation and therefore violates the frozen-base requirement.

If the owner instead requires empirical comparison, that is a separate design gate. Before implementation, it needs an explicitly authorized grouped non-test method-selection source with valid group identifiers, a predeclared comparison design and selection rule, and adequate disjoint data. Method comparison may not score competing calibrators on the same rows used to fit them. Any additional role, data access, membership handling, or split change needs its own approval. No comparison is proposed using the present sanitized release alone.

Current upstream scikit-learn documentation describes `CalibratedClassifierCV` with `FrozenEstimator` for a previously fitted estimator and places responsibility for disjoint fitting/calibration data on the caller. It documents sigmoid, isotonic, and temperature methods in current stable material. The Context7 result is current upstream documentation, not a guarantee of API equivalence with the recorded scikit-learn 1.8.0 runtime. Any implementation must verify the exact 1.8.0 API in synthetic tests before use; it must not silently switch methods or upgrade the environment.

## 3. Calibration population and fitting boundary

Use every eligible row in the existing `calibration` role, as published under the accepted `all_eligible` policy. Do not resample, reweight by class, duplicate, downsample, or reallocate rows. Give each row equal weight (`sample_weight=None`, equivalent to unit row weights). This retains the role's eligible-cell prevalence for the calibration fit.

A future public **calibration-only selective loader** must validate the pinned development attestation and schema, then open only `calibration.csv` and minimum approved attestation/receipt metadata. It must not call the all-role loader and filter afterward. It returns only the ordered predictor matrix and integer target; it does not expose train or validation rows. It enforces the canonical 11-feature order, `float32` predictors, and `int8` target. Permitted scalar checks are: expected schema/version, exact accepted development-attestation hash, calibration-file hash, and the recorded calibration row/positive/negative counts (`2060` / `153` / `1907`) against that attestation. These checks do not inspect identifiers or prove component concentration. Also validate nonempty data, exact binary labels `{0, 1}`, both classes, finite/domain-valid features, and nonconstant finite base-estimator scores; failure stops before model publication.

**Pinned adequacy evidence and limit:** D-053 records the independent protected review of the exact Phase 12 pair and calibration-role aggregate (5 components, 2060 rows, 153 positives, 1907 negatives, maximum component share of positives `0.320261`); D-054 records owner acceptance of that exact pair. Bind those decisions to protected attestation SHA-256 `a92d798fbbd2e6cc2d031a5927c4b15530573f50774039c36b29f1452e522c26` and development attestation SHA-256 `4fcf286156892c9da0289e56712210e90cccdcd608641c7ee573a8db1dcea23f`. The accepted calibration CSV identity recorded in D-053 is SHA-256 `e424865be9143823f5a15fce27a3eb6df74abe802be84f142505f1209e3a77dc`. These are historical decision-record identities and aggregates, not fresh artifact reads. The development attestation's role counts and hashes do **not** certify group counts or concentration. No new machine-verifiable concentration certificate is claimed or fabricated. Owner must either accept D-053/D-054 as the already-reviewed adequacy basis for this protocol, or require a separately authorized evidence design; in the latter case, calibration remains blocked until that gate is resolved. Do not reconstruct group membership or reopen protected files to manufacture a certificate.

Load the trusted `base_rf.joblib` only after its source, run, manifest, configuration, and model hashes have been checked against the identities above. Apply the prespecified calibrator to that fitted estimator using only calibration rows. Do not call `RandomForestClassifier.fit`, clone and refit the forest, alter its parameters, run Optuna, or create a train-plus-validation refit. Capture a deterministic forest-structure fingerprint before calibration and compare it afterward (tree count and each tree's structure/value arrays plus estimator parameters); any difference fails publication. Preserve the accepted model file byte-for-byte.

The calibrated package must remain a generic `.joblib` estimator exposing callable `predict_proba()` and `classes_`. Positive class is integer `1`, located through `classes_`; no hard-coded probability column is allowed. Preserve the existing canonical feature order. The interface must remain compatible with `modules/feature_pipeline.py::predict_positive_probability` and the CA's model-agnostic estimator contract.

## 4. Reporting protocol

Report probability-only results. `classification_threshold` is always `null`; no hard-label confusion matrix, precision, recall, F-score, Jaccard, or thresholded score is part of this protocol. A future hard-classification objective, threshold, and cost/safety/policy rationale require a separate owner decision and a separate report section. The stochastic CA remains threshold-free and uses calibrated model probabilities unchanged as its Bernoulli probabilities. Do not multiply them by a second wind factor; wind remains represented through the approved input features.

For both calibration-fit diagnostics and the later final-test report, prespecify:

- average precision (reported as average precision, not trapezoidal PR area);
- ROC AUC;
- binary Brier score, with the runtime's definition explicitly recorded as mean squared probability error and positive label `1`;
- log loss with labels `[0, 1]`;
- row count, positive count, negative count, and prevalence `positive_count / row_count`;
- a reliability diagram using 10 fixed equal-width bins: `[0.0,0.1)`, `[0.1,0.2)`, …, `[0.8,0.9)`, `[0.9,1.0]`. Each bin reports fixed zero-based index, lower/upper bounds, endpoint rule, count (including zero-count bins), mean predicted probability, and observed positive fraction. `p=0` belongs to bin 0; an exact internal edge belongs to the bin starting at that edge; `p=1` belongs to bin 9. For an empty bin, report count `0`, `mean_predicted_probability: null`, `observed_positive_fraction: null`, and `status: "empty"`.

All calibration-fit diagnostics are labeled **in-sample calibration-fit diagnostics**. They are not independent performance evidence and cannot select a model or method after this policy is accepted. A lower Brier score or log loss alone does not establish better calibration; these metrics also depend on discrimination and prevalence. Do not state that calibration improves discrimination or resolves Phase 13 limitations.

Validate probability and label shapes, binary labels, finite probabilities in `[0,1]`, and positive-class mapping before computing metrics. If a required metric is mathematically undefined (for example ROC AUC without both classes), record `value: null`, `status: "undefined"`, and the reason; do not substitute zero or omit the field. If an input contract, accepted role count, or required calibration suitability gate fails, stop before publication and retain a failure receipt. The final-test evaluator must fail closed on a one-class or otherwise invalid final role; its failure report may identify the undefined metrics and reason without disclosing row details. No metric failure permits recalibration, threshold choice, model selection, or a second final-test attempt.

## 5. Artifact and pre-test freeze contract

Proposed configuration paths (new files, not created by this task):

- `Thesis_RF/Code/config/phase14_setc_calibration_v1.yaml`
- `Thesis_RF/Code/config/phase14_setc_final_test_v1.yaml`

Proposed immutable versioned output roots (created only by their separately approved jobs):

- Calibration: `Thesis_RF/Code/output/phase14/phase14-setc-rf-61e8f740-r1/calibration/<calibration_run_id>/`
- Independent calibration-package review: `Thesis_RF/Code/output/phase14/calibration-review/<calibration_run_id>/review_record.json`
- Primary test evaluation: `Thesis_RF/Code/output/phase14/final-test/<model_lock_sha12>/primary/<primary_run_id>/`
- Verification replay: `Thesis_RF/Code/output/phase14/final-test/<model_lock_sha12>/verification/<verification_run_id>/`
- Per-job supervisor evidence: `Thesis_RF/Code/output/phase14/evidence/<run_id>/`
- Pre-test lock: `Thesis_RF/Code/output/phase14/model-lock/<lock_id>.json` (the lock's content hash is computed externally; it is not embedded in its own filename or contents)

Each `evidence/<run_id>/` contains the exclusive-created `stdout.log`, `stderr.log`, `operator_resources.jsonl`, `process.json`, `supervisor_handoff.json`, and `supervisor_completion.json`. The child finalizes `execution_receipt.json` after its computation-stage outputs. That receipt contains computation evidence only; it makes no claim about supervisor-observed exit, completed capture, final process state, or overall supervisory completion. After child exit or bounded shutdown/capture attempts, the supervisor writes `supervisor_completion.json` with those observations. This completion record is the authoritative final record for supervisory observations; supporting logs/process/resource files do not substitute for it. The supervisor externally hashes the finalized child receipt if present and records its path and digest; if absent, it records the receipt as missing and never manufactures a successful receipt. The child-receipt inventory is a snapshot of only finalized earlier outputs: it excludes that receipt itself, `supervisor_completion.json`, and every later-created review record. It makes no forward hash claim for a file not yet finalized. No finalized receipt or artifact is rewritten to add a later digest or review. Partial jobs retain what exists and an as-of-finalization inventory in the child receipt when present, plus supervisory failure evidence in the completion record. Calibration's manifest/report/child receipt and model live under its calibration run root; primary report/child receipt live under its primary run root; verifier metrics/child receipt/review record live under its verification run root.

`<run_id>` is unique and created exclusively. The primary and verification outputs and evidence use distinct directories, approvals, run IDs, receipts, and owners. No output root is created during source work.

The final-test config path is a default-deny template until calibration has completed and its package has passed independent review. Sequence: independently review the finalized calibration package and receipt; then populate the final configuration with the accepted calibrated-model SHA-256 and approved final-row identity; independently review that finalized configuration; obtain separate owner approval for the configuration change and its commit; then an independent reviewer verifies the resulting clean full `evaluation_revision`, exact final-config SHA-256, and evaluator-source identity against the reviewed change. Record that verification before the owner lock decision. The configuration does not contain its own file hash, the commit that contains it, or a pre-test-lock hash. Those identities are recorded later in the external lock/authorization chain, avoiding circular references.

Use distinct revision identities: `calibration_revision` is the clean full source pin used for the calibration job and remains fixed in its calibration manifest/receipt; `evaluation_revision` is the later clean full source pin containing the independently reviewed, separately approved final-test configuration and evaluator source. The final lock is created only after verifying the clean `evaluation_revision` and final config hash. The lock binds these known identities and the finalized calibration-package review/receipt identities; its content hash is computed externally and recorded in a separate owner authorization record. The lock contains neither its own hash nor a hash/reference that depends on its own hash; the final configuration contains no lock hash. No final-test row may be opened, enumerated, hashed, loaded, or scored before an owner explicitly approves that immutable lock by its externally computed hash.

Proposed files and schemas:

| File | Proposed schema / contents |
|---|---|
| `base_rf_calibrated.joblib` | New immutable calibrated estimator; never replace `base_rf.joblib` |
| `calibration_manifest.json` | `phase14a_calibrated_model_manifest.v1`: proposal ID/hash; run and source pin; Phase 13 protocol/config/manifest/release/model identities; calibration method and owner decision reference; feature/target schema; input role and row hash/count summary; software/environment; calibrator/model artifact hashes; forest fingerprint before/after; output paths; command; start/end timestamps; status |
| `calibration_report.json` | `phase14a_calibration_report.v1`: probability metrics, prevalence, fixed reliability-bin summaries, threshold null, explicit in-sample label, failure/undefined statuses |
| `execution_receipt.json` | `phase14a_execution_receipt.v1`, child-finalized: unique run ID, approval reference, exact child argv, full source revision/clean-state evidence, input identities/hashes, hashes only of outputs finalized before receipt finalization, environment, computation start/end timestamps, computation status and child-side failure reason, and retained partial-output inventory. Its inventory excludes itself, `supervisor_completion.json`, and later review records. It contains no supervisor-observed resource samples, exit, capture, final process state, overall-completion claim, or independent-review disposition. No receipt is rewritten. |
| `supervisor_completion.json` | `phase14a_supervisor_completion.v1`, supervisor-finalized after observed child exit or bounded shutdown/capture attempts: run ID; child receipt status (`present` with path and externally computed SHA-256, or `missing`); observed exit status (null if unknown); capture completion/status; final observed process state; resource observations; shutdown/stop observations; supervisory/finalization errors; completion timestamp; and overall supervisory status. It records missing evidence as missing and cannot synthesize a successful child receipt. Its own hash and later review records are not embedded. |

The calibration-package review record uses `phase14a_calibration_package_review.v1` and is stored at the separate path above. It records the reviewer, reviewed model/manifest/report identities, and both child execution receipt and supervisor completion record identities, with externally computed hashes when present or an explicit missing status. It is written only after those evidence states are fixed. The review record is excluded from their output inventories; no digest or disposition is written back into either record.

Calibration output also includes `base_rf_calibrated.joblib`, `calibration_manifest.json` (`phase14a_calibrated_model_manifest.v1`), and `calibration_report.json` (`phase14a_calibration_report.v1`). The calibration execution receipt is `calibration/<calibration_run_id>/execution_receipt.json`; its supervisor evidence is under `evidence/<calibration_run_id>/`.

Primary final-evaluation output files are `final_test_report.json` (`phase14a_final_test_report.v1`) and child `execution_receipt.json` (`phase14a_execution_receipt.v1`) under the primary root. Verification-replay output files are `verification_metrics.json` (`phase14a_verification_metrics.v1`), child `execution_receipt.json`, and a separate `review_record.json` (`phase14a_independent_review.v1`) under the verification root. The review record binds both the primary and verification child receipts and both corresponding `supervisor_completion.json` records, recording each externally computed hash when present or an explicit missing status, plus the locked model, test-row, configuration, source, and environment identities. It is created only after those evidence states are fixed and is excluded from their inventories. Supervisor logs and handoff/process/resource records remain in the corresponding `evidence/<run_id>/` directory. No review record is written into or substituted for either execution or completion record.

After independent calibrated-package review and the separately approved final-config commit, verify a clean `evaluation_revision` and final-config SHA-256, then create a separate owner-approved **pre-test lock** before opening any final-test row. The lock binds the `calibration_revision` and `evaluation_revision`; final calibrated-model hash; Phase 13 source/run/model identities; calibration method and its approval; calibration manifest and report identities; present, valid child execution-receipt digest and `supervisor_completion.json` identity for the successfully completed calibration; feature and positive-class contract; reporting fields and bin edges; threshold `null`; evaluator source/config hashes; approved environment; final-row and metadata identities by prior accepted record (not newly read); and lock creation time. A missing or failed calibration receipt/supervision gate prevents package acceptance and lock creation. Compute the lock's SHA-256 externally and place it in the separate owner authorization record that approves the lock; do not put the lock's own hash or a hash-dependent approval reference in the lock. The lock is immutable. Final-test values or scores cannot select another model/method, change calibration or reporting, choose a threshold, revise the lock, or trigger a new model within this evaluation. A proposed model revision requires a new evaluation plan and owner approval before any test access; the current final-test outcome remains reported as observed and cannot serve as new model-selection data.

Independent package review and owner lock acceptance are mandatory gates. They are separate from calibration authorization and final-test authorization. Preserve all Phase 13 evidence and artifacts throughout.

## 6. Narrow final-evaluation contract

The later evaluator may read only:

1. the pinned `SetC_Protected/set_c_v1/final_test_rows.csv`;
2. the exact minimum approved attestation/receipt metadata needed to verify its schema, hash, row count, accepted source/run binding, and publication receipt; and
3. the frozen calibrated model and its pre-test lock.

It must not open, enumerate, hash, or reconstruct `canonical.csv`, `split_manifest.csv`, any complete mapping, or any complement. In particular, it must not call `set_c_split_release.py::_verify_publication_pair` or another broad pair verifier that reads or hashes those excluded inputs. Metadata checks must be scalar checks on the already-approved attestation/receipt; they must not follow references to excluded files. No read of calibration contents is needed during final evaluation.

The exact published `set_c_final_test_rows.v1` CSV column order, from `feature_pipeline.py::AUTHORITATIVE_CSV_COLUMNS` plus the publication append in `set_c_split_release.py`, is:

```text
set_id, experiment_id, event_id, scenario_family_id, scenario_id, run_id,
seed, timestep_t, timestep_t1, cell_row, cell_col, grid_id, stable_cell_id,
duplicate_group_id, spatial_block_id, ignition_set_sha256, run_identity_sha256,
provenance_sha256, feature_schema_version, target_version,
slope_risk, proximity_risk, building_presence, material_risk, material_class,
wind_speed, wind_sin, wind_cos, neighbor_burning_count,
composite_flammability, wind_weighted_score, Ignited,
leakage_component_id
```

It has **no `split_role`**. The existing `evaluate_authorized_final_test()` currently requires that field. Reconcile this in source: validate exact schema/version and the pinned final-row hash through the narrow attestation/receipt path, then treat the caller's approved exact `final_test_rows.csv` input as the final-test role. Do not rewrite or republish the accepted artifact. Use only the canonical 11 ordered predictors and `Ignited`; keep every metadata field and `leakage_component_id` out of prediction and all output records. The evaluator may use row count and class counts in memory but must not emit row keys or identifier values.

There is exactly one **primary final evaluation**, performed by the owner-authorized primary operator after the pre-test lock and fresh primary preflight. The child produces the immutable primary report and computation-stage execution receipt; the supervisor produces its separate completion record. A separately designated independent verifier may perform exactly one **bounded numerical verification replay** only after a distinct owner authorization and fresh verification preflight. That replay uses the identical pinned final rows, locked calibrated model, final-test configuration, source revision, and metric/bin policy. It reproduces aggregate metrics and bin summaries solely to verify the primary report; it cannot select a model/method, change calibration/reporting, select a threshold, or be treated as a second primary result. No exploratory test evaluation, automatic retry, or additional replay is allowed. Compare counts/statuses exactly and numeric aggregates under the prespecified verifier tolerance (`rel_tol=1e-10`, `abs_tol=1e-12`); tolerance is a proposed reporting choice requiring owner approval. A mismatch produces a `NEEDS EVIDENCE` review record and stops; no re-evaluation or model revision follows.

The verifier writes its own immutable child execution receipt and `supervisor_completion.json`, then a separate review record binding both records for each run and the lock. If either child receipt is missing, the review records that state and binds the corresponding completion record; no receipt is fabricated. Both operators have only the narrow read scope above plus source/config/lock/evidence needed for their assigned step. This is procedural separation; the approved folder layout and same Windows identity do not prove OS-enforced blindness. No Windows-account or ACL change is proposed.

Only redacted aggregate evidence may leave that scope: metric values/status, aggregate row/class counts and prevalence, fixed bin counts/means/fractions, artifact and source hashes, environment, timestamps, run/approval references, and PASS/WARNING/FAIL/NEEDS EVIDENCE findings. Never include row values, component IDs, cell coordinates, membership, per-row predictions, or data excerpts in reports, logs, screenshots, or chat.

## 7. Numerical execution and failure rules

These are proposed conservative operational caps, not measured resource requirements. Each job requires a separate owner authorization and a fresh preflight after a clean source pin.

| Job | Max processes/jobs | Wall-clock soft stop | Minimum free memory | Minimum free disk | Sampling |
|---|---:|---:|---:|---:|---:|
| Calibration/package creation | 1 supervised child; 1 calibration job; no CV/parallel jobs | 1,800 seconds end-to-end | 4 GiB available physical memory | 2 GiB free on output volume | 1 second |
| Primary final-test evaluation | 1 supervised child; exactly 1 primary pass; no retry | 1,800 seconds end-to-end | 4 GiB available physical memory | 2 GiB free on output volume | 1 second |
| Independent verification replay | 1 separately authorized supervised child; exactly 1 bounded replay of identical inputs; no retry | 1,800 seconds end-to-end | 4 GiB available physical memory | 2 GiB free on output volume | 1 second |

The caps apply to the complete supervised operation including input checks, model load, inference, metric/report creation, capture, flush, close, child-receipt finalization, and supervisor-completion finalization. These proposed values reuse the accepted earlier resource floors and one-second monitoring cadence, but are new Phase 14 policy choices requiring owner acceptance. They do not claim that performance or sufficiency has been measured for these jobs. A one-second sample is a point sample, not a continuous memory/disk bound or peak-RSS measurement. Shutdown remains procedural and best-effort; no hard OS termination or strict end-to-end shutdown guarantee is claimed.

Default deny. Calibration and final evaluation each require their own explicit owner approval reference; approval references are procedural audit evidence, not authentication. At each phase, require a fresh preflight for the exact source commit and a clean runtime-source checkout, input/output path containment, immutable input hashes/receipts, environment, absent unique output destination, and memory/disk floors. Keep the pre-existing dirty shared memory files out of that checkout; do not rewrite, stage, or commit unrelated user work as part of Phase 14. Use a new unguessable run ID and exclusive no-overwrite creation. Do not reuse consumed approvals or run IDs.

On any failed preflight, input/schema/hash check, suitability check, fingerprint comparison, output capture, resource floor, timeout, child-receipt verification, or supervisory-completion check: stop; do not retry automatically, clean up, promote partial output, manufacture a missing receipt, or overwrite. Retain partial files, any child receipt that exists, and supervisor failure evidence for manual disposition; a missing child receipt is recorded as missing in `supervisor_completion.json`. Do not start the next phase. If the stop deadline expires, apply only the reviewed bounded supervisor stop action and record process IDs and whether exit was observed; synchronous OS/process operations may exceed the nominal deadline. Do not claim guaranteed or hard OS termination. No recovery/cleanup authorization is implied.

Each child execution receipt records computation-stage command/argv, working directory, source revision/clean-state check, approval reference, unique run ID, computation start/end UTC and local timestamps with timezone, actual runtime versions, input identities/hashes, finalized earlier output hashes, computation status/failure reason, and partial-output inventory. It does not claim supervisor-observed resource samples, exit, completed capture, final process state, or overall success. The supervisor finalizes `supervisor_completion.json` only after child exit or bounded shutdown/capture attempts. It records the child receipt's externally computed SHA-256 and path if present, otherwise marks it missing; it records observed exit (null if unknown), capture status, final process state, resource observations, shutdown attempts, and supervisor/finalization errors. Neither record includes its own hash or forward references to later review records, and neither is rewritten after finalization. Later immutable review records bind both records. Approval references remain procedural evidence, not authentication.

Overall job success requires verified child outputs and a present, valid child receipt, plus normal supervisory completion, observed exit code 0, completed capture, resolved process state, and no relevant child or supervisor finalization errors. A computation-stage success without any of these supervisory conditions is not overall success. If the receipt is absent or any condition fails or is unknown, record failure/needs-evidence, retain partial outputs and both records when available, and do not manufacture a receipt or retry automatically.

**Supervised launch contract (Draft 4 proposal; not executable):** working directory for the operator shell, supervisor, and child is `C:\Users\gibgib\Thesis-Cellular-Automata\Thesis_RF\Code`. The Phase 12 Python supervisor is pinned to `run_set_c_split.py`; the Phase 13 PowerShell supervisor's completion-record validation and production caller are Phase-13-specific. Neither is a Phase 14 launcher and neither will be invoked unchanged.

Propose extracting the reusable process/capture/resource loop from `run_phase13_supervised.ps1` into `Thesis_RF/Code/modules/phase_supervision.psm1`, preserving the Phase 13 wrapper's existing contract. A new fixed-mode adapter `Thesis_RF/Code/run_phase14_supervised.ps1` will use that shared core, enforce Phase 14 default-deny/preflight/caps and validate Phase 14 execution receipts. It accepts only fixed job modes `calibration`, `final_primary`, and `final_verification`; it does not accept arbitrary child commands. This reuses one supervision core rather than adding an independent supervisor framework.

After separate owner authorization and clean reviewed source pin, the exact outer launch form is:

```powershell
Set-Location 'C:\Users\gibgib\Thesis-Cellular-Automata\Thesis_RF\Code'
powershell.exe -NoProfile -NonInteractive -File `
  'C:\Users\gibgib\Thesis-Cellular-Automata\Thesis_RF\Code\run_phase14_supervised.ps1' `
  -Job '<calibration|final_primary|final_verification>' `
  -ApprovalReference '<fresh-job-specific-owner-approval>' `
  -ExpectedRevision '<full-clean-source-sha>' `
  -RunId '<unique-job-specific-run-id>'
```

The adapter selects a fixed child command by job mode. Calibration child: `C:\Users\gibgib\Thesis-Cellular-Automata\.venv\Scripts\python.exe -B -m modules.phase14_calibration --config C:\Users\gibgib\Thesis-Cellular-Automata\Thesis_RF\Code\config\phase14_setc_calibration_v1.yaml --approval-reference <reference> --expected-revision <sha> --run-id <id>`. Primary and verification children call `evaluate_rf_final_test.py` with `--mode primary` or `--mode verification`, respectively, and the exact `phase14_setc_final_test_v1.yaml`; the verifier must have its own approval reference and run ID. The adapter derives the fixed destination and `output/phase14/evidence/<run_id>/` path from validated mode/run ID and refuses any existing target. These command forms, hashes, final config contents, and output paths require source review and owner approval before becoming a run card. No launcher, preflight, or job is run by this proposal task.

## 8. Minimal implementation scope and acceptance gates

**This task creates only this SHARED proposal document.** If accepted later, the smallest proposed source/test scope is:

| Proposed future path | Classification | Purpose |
|---|---|---|
| `Thesis_RF/Code/modules/phase_supervision.psm1` | RF_OWNED | Extract the existing Phase 13 process/capture/resource loop as one shared core with pluggable, schema-specific completion validation |
| `Thesis_RF/Code/run_phase13_supervised.ps1` | RF_OWNED | Adapt the existing Phase 13 wrapper to the shared core while preserving its accepted Phase 13 arguments, receipt schema, and behavior |
| `Thesis_RF/Code/run_phase14_supervised.ps1` | RF_OWNED | Add a default-deny fixed-mode Phase 14 adapter for calibration, primary evaluation, and verification replay |
| `Thesis_RF/Code/modules/set_c_split_release.py` | RF_OWNED | Add a public calibration-only selective loader using the existing verified release contract |
| `Thesis_RF/Code/modules/phase14_calibration.py` | RF_OWNED | Fit the approved calibrator around the frozen model; produce versioned manifest/report/receipt; verify forest fingerprint |
| `Thesis_RF/Code/evaluate_rf_final_test.py` | RF_OWNED | Accept the published final-row schema through narrow metadata validation and produce probability-only aggregates |
| `Thesis_RF/Code/config/phase14_setc_calibration_v1.yaml` | RF_OWNED | Pin approved method, resource caps, calibration paths, input identities and report schema |
| `Thesis_RF/Code/config/phase14_setc_final_test_v1.yaml` | RF_OWNED | Pin locked model, final-row identity, metric/bin policy, primary/verification mode, paths and resource caps |
| `Thesis_RF/Code/tests/test_phase13_runner_contract.py` | RF_OWNED | Regression-check unchanged Phase 13 behavior after extracting shared supervision logic |
| `Thesis_RF/Code/tests/test_phase14_supervisor_contract.py` | RF_OWNED | Synthetic harmless-child checks for default deny, fixed job modes, time/resource stops, separate computation and supervisory records, missing receipt and non-success handling, completed capture, observed exit/process state, no-overwrite, and procedural shutdown status |
| `Thesis_RF/Code/tests/test_phase14_calibration_contract.py` | RF_OWNED | Synthetic temporary-path checks for selective loading, frozen estimator, dtypes, hashes, no-overwrite, fingerprint, and failures |
| `Thesis_RF/Code/tests/test_phase14_final_evaluator_contract.py` | RF_OWNED | Synthetic temporary-path checks for exact published schema, narrow file access, one primary plus separately authorized replay, metrics/redaction, and default deny |
| `Thesis_RF/Code/output/phase14/**` | GENERATED_OR_IMMUTABLE | Future versioned runtime outputs only, after separately approved execution; never create during source work |
| `docs/Phase14A_Calibration_Final_Test_Protocol_Proposal.md` | SHARED | This hash-bound planning artifact |

This is the exact proposed future source/test allowlist; implementation may not expand it without a separately reviewed scope. Source implementation requires protocol owner acceptance first. Tests use only synthetic temporary files/estimators and harmless child processes. They must show the calibration loader opens no train/validation roles; the final evaluator never touches canonical/mapping paths; the supervisor refuses arbitrary commands, writes separate child and supervisory records, records a missing receipt without synthesizing success, recognizes overall success only when every stated gate passes, retains partial evidence, and preserves Phase 13 behavior. Then require independent source review and a clean reviewed source pin. A separate human calibration-run approval authorizes only calibration. After independent calibrated-package review, owner accepts the exact pre-test lock. A separate primary-evaluation approval authorizes exactly one primary pass. A distinct reviewer approval authorizes exactly one bounded numerical replay of the same lock and rows. Independent review/owner closure follow. No real execution is authorized by this proposal.

Do not invoke legacy `train_rf_optuna.py::fit_calibrated_model` (it refits a forest and allows empirical method choice), `load_verified_development_release`/`load_development_release` for calibration (they load all development roles), `ModelTrainer.__init__` (it loads train, validation, and calibration together), the legacy calibration-selection record path that requires grouped evidence/canonical and complete-split hashes, `run_set_c_supervised.py` (its `_RUNNER` is the Set C split runner), the Phase 12 Python supervisor (its child is `run_set_c_split.py`), either Phase 13 production entry point unchanged, or `_verify_publication_pair` for final evaluation (it accesses canonical and mapping). The final evaluator CLI remains default-deny until separately approved. Reuse verified schema, attestation, hash, positive-class and generic probability helpers only where the access boundary remains intact. Phase 14 uses one extracted shared supervision core with two thin phase-specific adapters; no parallel supervisor framework is proposed.

No CA simulation, raster evaluation, ground-truth comparison, deployment, manuscript work, or `Thesis_LR/**` change belongs to Phase 14 calibration/final-estimator testing.

## 9. Unresolved owner choices and acceptance gates

The owner must explicitly decide:

1. Whether to approve the single prespecified sigmoid policy as an adjustment to D-010 and the legacy method-selection record contract, or instead require a separately authorized grouped non-test method-comparison design.
2. Whether the D-053/D-054 accepted independent protected review and owner acceptance are sufficient as the historical adequacy basis, with the explicit absence of a new machine-verifiable concentration certificate, or whether a separately authorized evidence design is required before calibration.
3. Whether to accept the unweighted all-eligible population, scalar role checks, probability-only metrics, exact 10-bin endpoint/empty-bin rules, threshold `null`, and undefined-result handling.
4. Whether to accept the separate single primary pass and separately authorized bounded verification replay, the proposed numerical comparison tolerance, receipt/review schemas, paths, exact source/test allowlist, shared supervisor-core extraction, resource caps, procedural supervision, and failure rules.
5. Who will perform calibration, package review, primary evaluation, and independent verification under the narrow read boundary.
6. The exact environment lock and any run-specific config values needed for a hash-bound source/run card.

Until these choices are accepted, no implementation, preflight, calibration, serialization, final-test access, or evaluation is authorized. Acceptance of this proposal alone would still not authorize a real calibration or final-test run; each remains separately gated.

## 10. Sources and evidence limits

**Project decisions and records consulted:** `docs/ai/PROJECT_STATE.md`; `docs/ai/DECISIONS.md` D-009, D-010, D-011, D-038, D-053, D-054, D-071, D-072, D-073, D-074, and D-075. D-074 governs the configuration/lock sequence and receipt hash exclusions; D-075 governs child-versus-supervisor evidence ownership. The accepted input hashes and Phase 13 run/environment facts above are transcribed from those records and were not re-derived from actual artifacts.

**Source/document allowlist for Draft 4 review:** `AGENTS.md`; `Thesis_RF/AGENTS.md`; `docs/ai/PROJECT_STATE.md`; D-053, D-054 and D-073–D-075 entries in `docs/ai/DECISIONS.md`; this proposal; `Thesis_RF/Code/run_phase13_supervised.ps1`; `Thesis_RF/Code/run_set_c_supervised.py`; `Thesis_RF/Code/modules/feature_pipeline.py` (`AUTHORITATIVE_OBSERVATION_METADATA_NAMES`, `AUTHORITATIVE_CSV_COLUMNS`, `CANONICAL_FEATURE_NAMES`); and `Thesis_RF/Code/modules/set_c_split_release.py` (final-row publication). No broad search was used for this revision.

**Earlier proposal source contracts:** `Thesis_RF/Code/modules/model_trainer.py` (`load_verified_development_release`, `ModelTrainer.__init__`, `evaluate`); `Thesis_RF/Code/train_rf_optuna.py` (`fit_calibrated_model`, `load_calibration_selection_record`); `Thesis_RF/Code/modules/automata_engine.py` (threshold validation and model probability transition); `Thesis_RF/Code/validation_engine.py::evaluate_estimator_probabilities`; and `Thesis_RF/Code/evaluate_rf_final_test.py::evaluate_authorized_final_test`.

**Current primary documentation:** Context7 resolved scikit-learn to `/websites/scikit-learn_stable` and returned current stable documentation for frozen-estimator calibration/disjointness and probability metrics. See the [current calibration guide](https://scikit-learn.org/stable/modules/calibration.html), [current `CalibratedClassifierCV` API](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html), and [current model-evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html). D-073 also records the [versioned scikit-learn 1.8 calibration guide](https://scikit-learn.org/1.8/modules/calibration.html) and [versioned 1.8 `CalibratedClassifierCV` API](https://scikit-learn.org/1.8/modules/generated/sklearn.calibration.CalibratedClassifierCV.html) as the documentation check for the recorded runtime. Documentation establishes the documented interface, not host execution or environment identity; synthetic compatibility checks remain a later source gate. No dataset/release rows, protected files, calibration contents, or models were intentionally opened, and no preflight, fit, serialization, evaluation, test, or simulation was run for this proposal.

**Evidence limits:** This is a proposed protocol, not owner-approved methodology. Sigmoid suitability, calibration outcome, runtime resources, and final-test performance remain unmeasured. Phase 13's development scores do not establish calibration performance, held-out performance of the refit, spatial/event-generalization, or real-fire predictive usefulness.

**Search-scope incident:** One delegated broad recursive source search incidentally traversed generated Phase 8 aggregate-report JSON and surfaced some aggregate summary values in tool output. Those values were not used in this proposal. No dataset/release rows, protected files, calibration contents, or models were opened. No out-of-scope write was attempted.
