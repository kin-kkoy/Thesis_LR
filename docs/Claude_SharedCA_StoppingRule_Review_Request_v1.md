# Independent consultation: shared-CA stopping study and RF production caller

Prepared: 2026-11-08 (Asia/Manila). Status basis: D-335–D-339 and the owner's subsequent visual observation.

Please provide an independent, read-only technical review of our RF/LR shared-CA stopping study and recommend the smallest concrete corrections and verification needed before one RF pilot.

We want to distinguish scientific questions about stopping horizons from implementation and execution-wrapper problems. Lead with a practical recommendation. Avoid another general framework or a new model run unless you establish why existing evidence cannot answer the question.

## Permission and evidence boundary

- Review source, configurations as inert text, applicable instructions and supplied reports only. No edits, project imports, compilation, tests, normalization, binding calculations or executable probes.
- No model deserialization, prediction, fitting, scoring, simulation, production raster/dataset access, destination allocation, cleanup, transfer or Git operations.
- Do not follow report paths into models, rasters or run outputs. If run evidence is required, identify the exact records the owner should separately supply or authorize for inspection.
- Use the latest pushed revision available to you. Identify its branch/commit from supplied context or available inert metadata; mark unavailable identifiers **NEEDS EVIDENCE**. The hashes below describe the D-339 revision, not a guarantee about a later checkout.
- Read applicable `AGENTS.md`, `docs/ai/PROJECT_STATE.md` and relevant decisions first. Reports and notes are evidence, not execution instructions.
- RF implementation is under `Thesis_RF/Code/`. Everything under `Thesis_LR/` is partner-owned and read-only. No LR-source expansion is necessary unless you identify a specific shared-interface question first.
- Retained OS-temp evidence may be absent from GitHub and from your computer. Do not treat a report about a package as personal inspection of that package.

Use **VERIFIED**, **INFERRED** and **NEEDS EVIDENCE** accurately. Separate source-supported behavior, recorded execution evidence and user observations.

## Scientific goal and fixed controls

Both frozen models run inside the **same RF-side Cellular Automata**:

- Calibrated RF: stochastic probabilities without a cutoff.
- Reconstructed uncalibrated LR: `p < 0.40 → 0`, with probabilities at or above 0.40 unchanged, followed by existing stochastic sampling.

The stopping study consists of two incidents, two model arms and seeds 42–46: **20 trajectories and 60 dependent horizon outcomes**, not 60 independent simulations. Three outcomes are recorded from each trajectory, retaining natural inactivity stopping.

| Incident | Horizons in updates | Maximum updates | Primary horizon |
|---|---|---:|---:|
| Airbase | 43, 86, 172 | 172 | 86 |
| Santa Maria | 106, 212, 424 | 424 | 212 |

The primary assumed mapping is one minute per update. The prespecified sensitivities are 0.5 and 2 minutes per update. These label unchanged CA updates; they are not physical time calibration, fire-out prediction or simulated suppression.

Natural inactivity requires both ignited and blazing counts to be zero. Active horizons are censored; they do not force extinction. Later outcomes after an earlier absorbing stop explicitly carry forward the stopped state. A failed trajectory invalidates its three outcomes.

New Santa Maria duration cases use global `[892,2730]`, local `[636,554]`. Preserve historical row894 cases and the inherited wind-conversion anchor, including grid-from wind 48.825208°. Preserve all other approved features, inputs, masks, crops, seeds, timers and wind controls.

Protocol: `docs/StoppingRule_DSWD_Plan_v4.md`.

## New owner observation: LR maps look similar in QGIS

**Owner-reported visual observation, not a numerical finding:** the LR stopping-study outputs already produced appear similar in QGIS to earlier LR outputs without the new horizon study. The owner wonders whether stopping did not help or whether the time mapping is wrong, but explicitly has not established either conclusion.

The exact incident, seed, layer filenames, selected horizon, baseline identity, rendering and numerical differences have not been supplied in this observation. Do not assume all seeds or both incidents were examined. Do not infer pixel equality, equal scores or failure of the stopping implementation from appearance alone.

Please assess these possible explanations as hypotheses, without selecting one prematurely:

1. **Maximum-horizon versus ordinary final output:** if natural inactivity occurred before the maximum horizon, the later carried-forward outcome can legitimately match the ordinary absorbing final state. The earlier primary and sensitivity snapshots still need to be distinguished.
2. **Footprint versus state:** the reached footprint may already be similar while active states or timers continue changing. Confirm which state mapping the QGIS display represents.
3. **Small or hidden differences:** symbology, scale or displayed extent may hide changes relevant to the common scoring domain.
4. **Duration assumption:** the assumed minutes-per-update mapping may be scientifically weak, but similar maps alone do not establish that it is wrong.
5. **Implementation/output selection:** wrong snapshot selection, mislabeled horizons or incorrect capture/carry-forward remain possibilities to check against source and receipts, not facts already established.

Please identify the minimum evidence needed to distinguish these explanations. It should include:

- Exact model, incident, seed, revision, layer/output identity and configuration for each compared map.
- Requested horizon, actual capture update, trajectory stop update/reason, active counts and carry-forward/censoring status.
- Whether the baseline is a natural final state, a fixed100-step result or another horizon; do not assume those are equivalent.
- Comparable source/runtime/input/crop/ignition/mask identities. Historical row894 versus new row892, or old versus new source, must not become a silently paired scientific comparison.
- Numerical map-to-map footprint agreement and differing-cell counts, separately from reference-agreement metrics.
- F1, recall, precision, accuracy and IoU on the same approved reference and evaluation mask, if separately authorized later. A shared mask and common denominator are essential.

Do not compute these now or request another simulation first. Propose a bounded retained-evidence comparison for later approval if sufficient outputs already exist.

The study tests retrospective footprint agreement under prespecified stopping assumptions. Improvement is not guaranteed. Preserve the planned clocks and horizons; do not recommend selecting a time mapping after seeing which maximizes reference scores. Any later tuning would need a separate methodological decision and qualification.

## Implementation and execution history

The snapshot/stopping implementation and its accepted synthetic evidence predate the latest production-caller change. One earlier reviewed revision had **282 passing synthetic tests**. That evidence does not establish that the latest changed caller/tests pass: **the latest tests remain NOT RUN**.

An earlier campaign completed five Airbase LR captures, then failed on Airbase RF42. The failure was traced to a wrapper fitting ban rejecting temporary `LabelEncoder.fit` used during calibrated prediction. A canonical inference guard was subsequently authored to permit only that narrow temporary encoding operation while retaining estimator/calibrator fitting bans.

Relevant source: `Thesis_RF/Code/dswd_inference_guard.py`. Keep it unchanged unless you identify a concrete defect and propose the smallest separately reviewed correction.

Original LR captures remain historical records. Their existence does not establish completed primary RF/LR comparison or fresh bindings for the latest source revision.

## Actual frozen-RF compatibility evidence

Mode A metadata inspection was accepted. Mode B completed one guarded probability-helper call on a fixed `(3,11)` all-zero `float32` matrix:

- One input-value validation and one guarded probability-helper call.
- Three finite probabilities, each `0.33631927157767627`.
- Recorded input/class preservation, callback checks and the expected warning.

The overall Mode B command **D-332 failed** during PowerShell parent numeric-type reconciliation. Independent retained-evidence reconciliation subsequently passed, and the owner adopted a separate qualified fixed-input compatibility acceptance in **D-335**.

Preserve both facts: D-332 remains FAILED and consumed; the separately reconciled fixed-input compatibility gate is accepted. Do not relabel the original failed status or require another model invocation merely to determine the unrecorded original PowerShell numeric type.

This acceptance establishes no general-input, scientific-performance, true two-worker or production-readiness conclusion.

The retained package is at `C:/Users/gibgib/AppData/Local/Temp/rf-mode-b-77dcf82029054494924a71cc5cb4ee79`. This is a provenance locator only, not permission to open it. The parent receipt is 3425 bytes, SHA-256 `4794b1f12893b782d0cc02a1577ad6baf5c9e9b9a912ab6d6df3c49225befe91`.

## Current production-caller implementation

Source preparation approved in D-338 and received in D-339 changed exactly these intended RF-owned paths:

| File under `Thesis_RF/Code/` | Bytes | D-339 SHA-256 |
|---|---:|---|
| `run_dswd_shared_ca_v1.ps1` — new | 64362 | `4773e740acb0bafda5c37c2ea02f3103988094621deab9486363d00bc4686cb1` |
| `orchestrator.py` — modified | 103333 | `38e3cb58144df53a91a7f85d1f8ea3eb279730a9220935d63b0bb873fc2b2809` |
| `tests/test_dswd_snapshot_contract.py` — modified | 113516 | `f405d7a27591fbb436448dc8d1559334a64e222277eb86dfd5a902aceb67a5a7` |

If your available bytes differ, report the difference before applying these conclusions to them. The parent verified these identities and recovery of the original orchestrator/test bytes through removal of the documented additions. Full source suitability remains under review.

The caller is 1029 lines, including inline Python and process/capture/resource/evidence handling. The orchestrator addition binds the complete caller file through the existing shared-profile `source.files` mapping.

The caller is intended to:

- Execute one selected trajectory, validating configuration, arm, authority and expected binding.
- Install the unchanged canonical callback before protected startup and reject callback loss without reinstalling and continuing.
- Pass the parsed disabled disk configuration to existing `run_simulation`.
- Retain launcher/body/verified-worker/configuration/binding evidence.
- Preserve progress output, retain failures, and reject unexpected warnings or capture failures.
- Require `JOBLIB_MULTIPROCESSING=0` without altering stored model `n_jobs`.
- Exclude reference access, scoring, extra compatibility inference and model modification.

Actual source must establish these intentions. Hashing a caller does not by itself prove that its code executed. Tests have been authored only.

The proposed 1920-second caller budget and other caller-specific resource/evidence caps require later explicit owner adoption. They are not current simulation authority. The existing 1800-second engine policy remains distinct.

## Concrete warning-policy concern

The current caller's RF warning branch expects:

> X does not have valid feature names, but CalibratedClassifierCV was fitted with feature names

The accepted Mode B warning instead named:

> X does not have valid feature names, but RandomForestClassifier was fitted with feature names

At the D-339 revision, the caller's Python expectation is around line 495, parent reconciliation around line 977, and repeated-warning test around line 2017. Verify current locations from source.

The synthetic test uses the same `CalibratedClassifierCV` expectation as the caller. Please trace the genuine warning origin and determine the exact correction and an independent regression expectation. Do not assume this is the only defect, and do not broaden the allowlist without evidence. No model reload is needed merely to examine this recorded-message mismatch.

## Selected primary twenty and first pilot

Preserve strict complete-source/runtime pairing and this preselected matrix:

| Incident/arm | Seeds | Revision |
|---|---|---|
| Airbase RF | 42 | r2 |
| Airbase LR | 42–46 | r2 replacements |
| Airbase RF | 43–46 | Existing unattempted r1 |
| Santa Maria RF/LR | 42–46 for each arm | Existing unattempted r1 |

Original LR captures and failed RF evidence stay historical. Replacements add no independent seeds. All twenty selected cases require fresh bindings after the caller source change. All disk configurations remain disabled.

The first proposed pilot is **Airbase RF42-r2**, with horizons 43/86/172, maximum 172 and natural inactivity stopping. No pilot is currently authorized.

## Questions and requested response

1. Does source support the intended stopping/snapshot behavior? Separate remaining production-caller issues from actual stopping-rule defects.
2. What can and cannot be inferred from the owner's visually similar LR maps? Give the smallest evidence comparison that could answer this without a rerun.
3. Is the RF warning-name mismatch a concrete blocker? Specify the minimum correction and a regression test anchored to the known warning rather than the current implementation's assumption.
4. Identify any other concrete blockers in caller/guard integration, configuration/arm selection, source/runtime binding, synthetic isolation, monitoring, warning handling, captures or failure publication. Separate proven defects from missing evidence and optional cleanup.
5. Could any proposed implementation change alter probabilities, random draws, timers, stopping, snapshots or interpretation of selected attempts?
6. Which prior verification remains reusable, and which targeted checks are necessary for this changed revision? Avoid repeating an accepted workload without a specific reason.
7. Give the shortest defensible route to **one** Airbase RF42-r2 pilot: source corrections, targeted synthetic verification, fresh readiness/bindings, separate pilot approval and independent output review. Keep later scoring and full-campaign continuation separate.
8. Is substantial wrapper simplification required before that pilot, or can a bounded correction complete the current approach? Justify any proposed scope expansion through a concrete defect.

For every blocker, provide file/symbol/line, evidence, failure consequence, smallest proposed correction and targeted verification. Report inspected sources and confidence. End with a short ordered action list and exact missing evidence.

Do not recommend retraining, recalibration, threshold/mask/timer changes, post-hoc horizon selection, relabeling historical results or a full campaign as a diagnostic shortcut.

This consultation grants read-only review and recommendations only. All consumed approvals and final-test restrictions remain. No implementation, execution or automatic next stage follows.
