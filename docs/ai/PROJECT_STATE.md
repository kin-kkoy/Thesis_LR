# Project State — Random Forest Thesis Track

## Record contract

- **Purpose:** Compact, Git-reviewable navigation and current-state index. Source, identified inputs, immutable artifacts, configurations, commands, and raw records remain the primary evidence.
- **Scope:** Random Forest work and shared coordination. `Thesis_LR/**` is partner-owned and read-only.
- **Last updated:** 2026-10-03 (Asia/Manila).
- **Last reviewed repository revision:** D-042 diff against `1c5a9727ae6c74b9017efae042eb1999b036a8f0`; resulting commit is identified by Git HEAD. No clean production pin.
- **Worktree note:** D-043 authorizes committing six reviewed edits; untracked `docs/Phase12Block.md` remains unchanged. Git status governs cleanliness.
- **Current reviewed RF work:** D-042 diagnostics passed synthetic/independent review. R2 consumed; historical cause unknown; production/model work blocked.
- **Staleness rule:** Recheck a fact when its source revision, input/artifact hash, configuration, dependency, or approval changes.
- **Size policy:** Keep this file below 16 KiB. Durable methodology belongs in `DECISIONS.md`.

<!-- BEGIN CODEX DIGEST -->
## Compact startup digest

- **Active objective:** Commit the reviewed diagnostics and verify synthetically; historical allocation cause remains unknown.
- **Methodology:** D-009–D-015 define the Set C estimand, 11 predictors, model-free `all_eligible` labels, 128×128 containment, controlled population, duplicate policy, and adequacy gates. D-029 makes predictor collisions audit-only.
- **Accepted evidence:** Phase 8/9 immutable aggregate reports establish the controlled population and prospective feasibility; D-033/D-034 identify and accept Set C v1 after Tasks 11.1/11.2.
- **Current authorization:** D-042/D-043: six-file diagnostics repair, commit after validator PASS, synthetic checks only. R2 consumed; package/real parents untouched.
- **Hard boundary:** Do not run the aggregate expansion, publish rows, assign roles, train/tune/calibrate, access the final test, publish models/rasters, access `stack_ground_truth.tif`, change manuscripts, or write `Thesis_LR/**`.
- **Next gate:** Resolve untracked handoff disposition and clean pin, then separately approve exact supervised run card, retained redacted logs, and protected-validation scope. No solver-limit change.
- **Agent tooling:** `rf_engineer` is pinned to `gpt-6-luna`/`medium`; `rf_validator` to `gpt-6-sol`/`medium`; `tech_lead` remains `gpt-6-sol`/`high`; and `thesis_researcher` remains `gpt-6-sol`/`medium` (D-030). The `tech_lead` instruction conflict remains governed by the parent agent and repository boundaries.
<!-- END CODEX DIGEST -->

## Current verified state

| Status | Current fact | Evidence | Recheck trigger |
|---|---|---|---|
| `VERIFIED` | RF is the writable implementation track; LR remains read-only. | `AGENTS.md`; `Thesis_RF/AGENTS.md` | Ownership changes. |
| `VERIFIED` | Set C is the active 11-feature candidate; Sets A/B remain separate historical chains. | D-009–D-014 | Owner supersedes the methodology. |
| `VERIFIED` | The RF estimand is eligible mapped-building-cell ignition at `t+1` from state/features at `t`; dynamic neighbor features use BLAZING at `t`. | D-009, D-010, D-013, D-014 | Target/timing contract changes. |
| `VERIFIED` | Authoritative Set C labels come only from the versioned model-free stochastic teacher. Incumbent RF transitions are not independent truth. | D-014; Phase 7 source and validator reports | Teacher contract changes. |
| `VERIFIED` | Phase 7 provides model isolation, seeded D-014 ignition, five-state `np.int8` transitions, inactive/censored/failure records, canonical `all_eligible` rows, bounded exact-retry collection, and disabled publication. | Commit `10b1b7c`; Phase 7 acceptance; 160-test source verification | Phase 7 interfaces change. |
| `VERIFIED` | Phase 8 harness is disabled by default, permits only the exact model-free path, retains no row dataset, and can publish only one hash-bound aggregate report atomically without overwrite. | Commits `ffbda28`, `2fd0e68`, `3d8439c`; repeated Task 8.4 PASS | Harness/publisher changes. |
| `VERIFIED` | Full-grid building/material mapped identities agree after material-specific normalization; combined validity still excludes slope-invalid cells. | Revision `3d8439c`; real-input preflight; approved domain hashes | Loader or raster changes. |
| `VERIFIED` | Phase 8 v2 completed from clean revision `df65217`: 32 authoritative inactive runs, 74/74 transitions, 384 rows (26 positive), five both-label leakage components at every tested block size, exact v1 reconciliation, and no forbidden operation or extra output. | Immutable v1/v2 reports; final RF Validator PASS | Report/hash/source/input changes. |
| `VERIFIED` | The thesis owner formally accepted Phase 8, including the v2 extension, and authorized Phase 9 planning only under a separate bounded contract. | Owner acceptance, 2026-09-25 | Owner grants or revises Phase 9 authorization. |
| `VERIFIED` | D-015 fixes 128×128 origin-(0,0) conservative grouping, a controlled 26-family/17-condition population design, aggregate-only site discovery, role-level adequacy minima, and conflict-preserving duplicate containment. | Phase 9 Task 9.1; owner approval, 2026-09-27 | Owner supersedes D-015 or production evidence changes the contract. |
| `VERIFIED` | Phase 8's five components and 26 positives are inadequate for production row publication; aggregate-only independent-family expansion is required. | Phase 9 Task 9.1 RF Validator review; D-015 | Approved adequacy expansion passes all gates. |
| `VERIFIED` | Gate 4 v2 source assigns each candidate component only to its first present material class in order 1–5, records aggregate material composition and assigned class, preserves global block reservation, and targets a non-overwriting v2 report. | `modules/set_c_site_inventory.py`; `config/default_experiment.yaml`; focused tests; 255-test full synthetic suite | Source/config changes or v2 production evidence contradicts the synthetic contract. |
| `VERIFIED` | D-018 assigns mixed-material components to the first eligible class in deterministic order 1–5 and globally reserves their full leakage footprint. This is an inventory rule, not a prevalence or independence claim. | Owner approval; D-018 | Owner revises D-018. |
| `VERIFIED` | D-016 authorizes only the bounded aggregate population-adequacy diagnostic after its inventory/manifest prerequisites. Production row publication, split execution, training, and final-test access remain unauthorized. | Phase 9 Tasks 9.1–9.3; owner approval, 2026-09-27 | Owner broadens or revokes the authorization. |
| `VERIFIED` | D-017 approves the disabled Task 9.3 source architecture: versioned package/staging, canonical rows, joint leakage components, development-only loaders, and a separate fail-closed final evaluator. Numerical role targets and protected storage/ACL remain separately gated. | Owner approval; 61 focused tests; final read-only RF Validator PASS | Source or approval changes. |
| `VERIFIED` | D-029 implements the approved narrow duplicate-policy revision: published duplicate identities are provenance-based observation records; exact predictor collisions remain aggregate audit statistics. The v3 replay is disabled and block-only; checkpoint package identity/state-digest/report-verifier checks are synthetic-only. | D-029; focused 47/47 and complete RF synthetic 286/286 | Fresh independent validator review or separately authorized replay. |
| `VERIFIED` | Gate 6 v3 independently validates 26 qualifying singleton block components and feasible unexposed role minima; all 97 predictor-collision family links remain audit-only. | Immutable v3 payload/file `3891f54b...9ca96` / `a6cd757f...8bbc`; final RF Validator PASS | V3 artifact, D-015 policy, or source binding changes. |
| `VERIFIED` | Agent profiles pin `rf_engineer` to `gpt-6-luna`/`medium`, `rf_validator` to `gpt-6-sol`/`medium`, `tech_lead` to `gpt-6-sol`/`high`, and `thesis_researcher` to `gpt-6-sol`/`medium`. | `.codex/agents/*.toml`; D-030; successful TOML parse | Profile settings or available models change. |
| `VERIFIED` | The authorized Set C v1 package from clean revision `161b8d0` has 884 ordered complete, inactive, authoritative runs and 10,733 rows (797 positive, 9,936 negative). Its manifest, dataset, all run/batch hashes, and exact 3,902-file inventory passed independent validation; no split role was published. | Manifest file SHA-256 `b700f1b9...e9e3c`, payload `7e993176...e8cbd`; independent RF Validator PASS | Package bytes or validation contract change. |
| `VERIFIED` | Task 11.1 found no checked row/schema/domain/identity/count defect. All 1,508 batches and 884 runs reconcile; all runs are inactive, complete, authoritative, and transition-count matched. | `docs/Phase11_Tasks_11_1_11_2_Report.md`; independent RF Validator | Package, validator contract, or approved methodology changes. |
| `VERIFIED` | Task 11.2 reconstructed 26 singleton-family components with 9–58 positives and 188–496 negatives each. A role-blind solve satisfied every D-015 component/class/material/concentration minimum without publishing assignments. | Same report; actual CSV; Gate 6 v3 comparison | Package bytes or D-015/D-029 policy changes. |
| `VERIFIED` | D-035 approves exact 11/5/5/5 component counts, D-015 minima, material coverage, ≤50% positive concentration, seedless digest order, and soft balance targets. | D-035; Task 12.1 report | Owner revises policy or dataset identity. |
| `VERIFIED` | D-040 staging rejection and D-041 default-deny opt-in/audit-reference/clean-pin guards remain active. | Runner/exporter; focused tests | Source/approval change. |
| `VERIFIED` (recorded) | R2 at main HEAD `1c5a9727` exited 1/RuntimeError after 37.846 s. Parents recorded empty; no targets/stages observed. No fresh path inspection. | `docs/Phase12Block.md` §§5–8; D-042 | New authorized evidence. |
| `NEEDS EVIDENCE` | Historical message/traceback/objective missing; timeout hypothetical. Resource floors passed sampled checks only; peak RSS unknown. | Same report; source | New authorized evidence. |
| `VERIFIED` | Unstaged diagnostics retain safe solver/stage context and redact chains; primary/independent tests 76/1 skipped. | D-042; four RF edits | Source/test/environment change. |
| `VERIFIED` | Read-only reconstruction of the generation runner's loaded and activated configuration reproduces manifest `configuration_sha256` `f18953d5854286d29e9082caf8e5cd35f1719038627840a2c227ba42227dddfe`; the configuration-file hash also matches. | `run_set_c_generation.py`; `orchestrator.py::load_config`; `set_c_population_adequacy.py::activate_approved_population_config`; `set_c_generation.py::_canonical_hash`; manifest | Configuration, normalization source, or package changes. |
| `WARNING` | The CSV supports source/provenance-consistent `all_eligible` and state-at-`t` timing, but row publication alone cannot prove execution behavior without teacher replay. | Task 11.1 evidence limits | A bounded replay is approved and verified. |
| `WARNING` | The package binds the approved resource limits but does not contain independent observed peak RSS, memory/disk minima, or elapsed-time telemetry. | Package manifest and independent RF Validator review | Independent execution telemetry is provided and verified. |
| `WARNING` | `tech_lead` still has contradictory delegation/output instructions: parent-owned delegation and no human prompt copying conflict with “Prompt to Copy,” direct researcher invocation, and a request to update Markdown despite read-only/manuscript boundaries. | `.codex/agents/tech-lead.toml` lines 21, 42, 50–54 | Profile wording is explicitly repaired and reviewed. |
| `NEEDS VERIFICATION` | Historical Set A/B/C metric claims are reproducible from identified datasets, models, configurations, splits, commands, and immutable outputs. | `docs/findings_SetA.md`, `docs/findings_SetB.md`, `docs/findings_SetC.md` | Close only with primary provenance. |

## Active blockers and authorized follow-up

1. `PASS` — Phase 11 acceptance remains historical evidence; package hashes and contents were not rechecked during the current source-only diagnosis/repair.
2. `WARNING` — 128×128 blocks are conservative containment, not measured independence. Predictor collisions remain audit-only under D-029; 543 conflicting-label vectors are retained.
3. `NEEDS EVIDENCE` — Actual resource extrema and independent execution proof of teacher timing/complete eligible capture remain unverified.
4. `PASS` — D-035 unchanged. No real assignment derived here; failed-run internal progress unknown.
5. `PASS` — D-038 locates the ordinary `SetC_Protected/set_c_v1` and `SetC_Development/set_c_v1` folders inside the repository checkout and removes the `SetCDev` requirement. This is a procedural holdout, not OS-enforced blindness; the canonical CSV and deterministic policy permit recomputation of membership.
6. `PASS` — D-039 exclusions/loading boundary remain. Preserve R2's existing real parents; no folder change here.
7. `PASS` — D-040 accepts repaired source-only Phase 12F/12G after independent review; receipt-bearing staging is rejected publicly.
8. `NEEDS EVIDENCE` — Historical cause/production performance unknown. Stub reproduces pre-stage failure; varied fixture passed eight objectives before budget stop. No historical/production proof.
9. `BLOCKED` — R2 consumed; real data, roles, protected inspection, production, folders/ACL, models, ground truth, manuscripts and LR remain prohibited. Protected-only publication requires manual recovery.
10. `PASS` — Diagnostics validated; D-043 conditionally authorizes the six-file commit. No numerical/limit change. Future supervision must retain actual redacted output.

## Known-good navigation

| Need | Start here |
|---|---|
| Durable methodology | `docs/ai/DECISIONS.md` D-009–D-015 |
| Phase 9 handoff | `docs/HandoffPhase9.md` |
| Phase 11 handoff | `docs/HandoffPhase11.md` |
| Phase 11 validation report | `docs/Phase11_Tasks_11_1_11_2_Report.md` |
| Phase 12 Task 12.1 policy review | `docs/Phase12_Task12_1_Grouped_Allocation_Policy_Review.md` |
| Phase 12 failed run / diagnostics scope | Preserved `docs/Phase12Block.md`; D-042; runner/exporter and focused tests |
| Active configuration | `Thesis_RF/Code/config/default_experiment.yaml` |
| Raster loading/domains | `Thesis_RF/Code/modules/data_loader.py` |
| Model-free teacher | `Thesis_RF/Code/modules/model_free_teacher.py`; `modules/automata_engine.py` |
| Set C collector | `Thesis_RF/Code/modules/set_c_collector.py` |
| Pilot harness/reporting | `Thesis_RF/Code/modules/set_c_pilot.py`; `run_set_c_pilot.py`; `orchestrator.py` |
| Phase 9 site inventory | `Thesis_RF/Code/modules/set_c_site_inventory.py`; `run_set_c_site_inventory.py`; `tests/test_set_c_site_inventory_contract.py` |
| Phase 10 source architecture | `modules/set_c_publication.py`; `modules/set_c_leakage.py`; `evaluate_rf_final_test.py`; `tests/test_phase10_architecture_contract.py` |
| Accepted Phase 8 evidence | `Thesis_RF/Code/output/phase8/phase8_set_c_feasibility_pilot_v2.aggregate.json`; preserved v1 report |
| Later split/training code | `modules/model_trainer.py`; `train_rf_optuna.py` |

## Test and review evidence

- Phase 7–11 test commands, immutable hashes, and validation limits are recorded in D-015, D-029, D-033–D-034, and the Phase 11 report; the table above indexes the current claims.
- D-042: unstaged RF patch on `1c5a9727`; primary 76/1 skipped (9.00 s), independent 76/1 skipped (8.96 s). Command/environment/limits in D-042. Windows symlink privilege case skipped; simulated reparse passed.

## Update procedure

1. Inspect the Git delta and claim-specific evidence.
2. Update only affected facts and preserve uncertainty labels.
3. Record the reviewed revision, artifact identity, command, and authorization boundary.
4. Append to `DECISIONS.md` only after explicit approval of a durable methodology choice.
5. Never overwrite an immutable artifact; use a separately approved versioned successor.
