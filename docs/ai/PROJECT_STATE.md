# Project State — Random Forest Thesis Track

## Record contract

- **Purpose:** Compact, Git-reviewable navigation and current-state index. Source, identified inputs, immutable artifacts, configurations, commands, and raw records remain the primary evidence.
- **Scope:** Random Forest work and shared coordination. `Thesis_LR/**` is partner-owned and read-only.
- **Last updated:** 2026-09-27.
- **Last reviewed base revision:** `df65217b91c1624bbfcff82e95cca15a45a6df2c` on `RF-11-Params`.
- **Worktree note:** The accepted v2 pilot recorded `source_dirty: false`. Afterward, the preserved visualization project `Thesis-try.qgz` was restored as an untracked item. This authorized memory update preserves the user's earlier uncommitted edits to this file.
- **Current reviewed RF work:** The revision containing D-018 and this record combines the accepted Ponytail cleanup, disabled Task 9.2 inventory, disabled Phase 10 source architecture, and mixed-material rule. Its clean-source gate ignores only preserved root `Thesis-try.qgz` while rejecting every other change. The full RF synthetic suite passed 254 tests with bytecode/cache disabled. No production execution occurred.
- **Staleness rule:** Recheck a fact when its source revision, input/artifact hash, configuration, dependency, or approval changes.
- **Size policy:** Keep this file below 16 KiB. Durable methodology belongs in `DECISIONS.md`.

<!-- BEGIN CODEX DIGEST -->
## Compact startup digest

- **Active objective:** Establish one reviewed Phase 9/10 revision and pass the full synthetic suite; then seek separate authorization for one inventory execution.
- **Methodology:** D-009–D-015 define next-timestep eligible-cell ignition, canonical Set C predictors, `all_eligible`, model-free teacher labels, 128×128 conservative grouping, controlled population expansion, duplicate containment, and role-level adequacy gates.
- **Accepted source:** Phase 7 supplies the fail-closed model-free teacher and bounded 11-feature collector. Phase 8 v2 revision `df65217b91c1624bbfcff82e95cca15a45a6df2c` supplies the disabled aggregate-only pilot extension; 110 focused synthetic tests passed before execution.
- **Accepted diagnostic artifact:** `Thesis_RF/Code/output/phase8/phase8_set_c_feasibility_pilot_v2.aggregate.json`; canonical payload SHA-256 `9a4f48ea6b2b7f882985d63ff3cad7ef3c785736e223860f82637fdd9e45cbf3`; whole-file SHA-256 `b62a7bf542d0ab74a3ac71399e85048b3169d0a04b33c95625aec3ee39339d17`.
- **Diagnostic result:** All 32 runs were inactive, complete, authoritative, and count-matched. They produced 384 aggregate-observed eligible rows: 26 positive and 358 negative. Every diagnostic block size yielded five authoritative leakage components, all five independently containing both labels; four-role allocation is mathematically feasible, but no roles or final block dimensions were assigned.
- **Current authorization:** D-016 authorizes the bounded aggregate diagnostic under prerequisites; D-017 authorizes only disabled source architecture; D-018 closes the mixed-material policy gate. Inventory execution and every production/model/final-test operation remain separately gated.
- **Hard boundary:** Do not run the production inventory, approve sites automatically, run the teacher expansion, publish rows, assign roles, train/tune/calibrate, access the final test, publish models/rasters, access `stack_ground_truth.tif`, change manuscripts, or write `Thesis_LR/**`.
- **Next gate:** Resolve the class-1-to-5 mixed-material reservation policy, commit a clean reviewed revision, separately authorize and validate one inventory execution, and approve the exact family manifest before implementing or executing the bounded aggregate adequacy expansion.
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
| `VERIFIED` | Task 9.2 implements a disabled, fail-closed, aggregate-only site-inventory harness with exact provenance/input contracts, 8-connected component selection, global 128×128 footprint reservation, four primary plus four reserve candidates per material class, aggregate-only atomic non-overwrite publication, and no teacher/model execution. | `modules/set_c_site_inventory.py`; `run_set_c_site_inventory.py`; synthetic contracts; final read-only RF Validator PASS | Source/config changes or production evidence contradicts the synthetic contract. |
| `VERIFIED` | D-018 assigns mixed-material components to the first eligible class in deterministic order 1–5 and globally reserves their full leakage footprint. This is an inventory rule, not a prevalence or independence claim. | Owner approval; `_select_candidates`; mixed-material synthetic contract | Owner revises D-018. |
| `VERIFIED` | D-016 authorizes only the bounded aggregate population-adequacy diagnostic after its inventory/manifest prerequisites. Production row publication, split execution, training, and final-test access remain unauthorized. | Phase 9 Tasks 9.1–9.3; owner approval, 2026-09-27 | Owner broadens or revokes the authorization. |
| `VERIFIED` | D-017 approves the disabled Task 9.3 source architecture: versioned package/staging, canonical rows, joint leakage components, development-only loaders, and a separate fail-closed final evaluator. Numerical role targets and protected storage/ACL remain separately gated. | Owner approval; 61 focused tests; final read-only RF Validator PASS | Source or approval changes. |
| `NEEDS VERIFICATION` | Historical Set A/B/C metric claims are reproducible from identified datasets, models, configurations, splits, commands, and immutable outputs. | `docs/findings_SetA.md`, `docs/findings_SetB.md`, `docs/findings_SetC.md` | Close only with primary provenance. |

## Accepted Phase 8 aggregate evidence

- **V1:** `output/phase8/phase8_set_c_feasibility_pilot_v1.aggregate.json`; payload/file SHA-256 `23cd71eb...98afc6` / `e4c52c...d332`; clean revision `3d8439c`; preserved diagnostic predecessor.
- **V2:** `output/phase8/phase8_set_c_feasibility_pilot_v2.aggregate.json`; payload/file SHA-256 `9a4f48ea...45cbf3` / `b62a7bf...39d17`; clean revision `df65217`.
- **V2 result:** 32 authoritative inactive runs, 74/74 transitions, 384 rows, 26 positive, 358 negative, and five both-label leakage components at every tested block size. Detailed family/resource evidence remains in the immutable report and Phase 9 handoff.
- **Boundary:** Mathematical feasibility only; no role assignment, row publication, model operation, final CA raster, or real-fire accuracy claim.

## Active blockers and authorized follow-up

1. `PASS` — Phase 8's four-role feasibility gate is satisfied: five authoritative leakage components independently contain both labels at every diagnostic block size.
2. `WARNING` — V2 contains 20 conflicting-label duplicate groups. Phase 9 must keep each duplicate/leakage-connected group intact; random row splitting is prohibited.
3. `APPROVED POLICY` — Use 128×128-cell blocks at origin `(0,0)` as conservative containment; do not claim measured decorrelation or spatial independence.
4. `PASS` — The disabled aggregate site-inventory source and synthetic contracts passed read-only RF validation: four primary and four reserve candidates for each material class 1–5.
5. `NEEDS EVIDENCE` — Discover, validate, and separately approve exact new-family coordinates, ignition hashes, material contexts, component identities, and block footprints.
6. `PASS` — D-018 approves deterministic class-1-to-5 ownership and global footprint reservation for mixed-material components.
7. `AUTHORIZED WITH PREREQUISITES` — D-016 authorizes the bounded aggregate diagnostic source and synthetic tests, plus one later diagnostic execution only after a clean revision, separately authorized/validated inventory, mixed-material resolution, and exact-manifest approval.
8. `BLOCKED` — Source architecture exists, but row publication and split roles still require D-015/D-016 evidence, separate execution approval, exact package identity, and all later allocation gates.
9. `NEEDS EVIDENCE` — RF search space, tuning budget/objective, class weight, calibration, reporting threshold, protected final test, dataset/split identities, and final evaluation remain unset.
10. `BLOCKED` — Training, tuning, calibration, final-test access, model/raster publication, manuscripts, and `Thesis_LR/**` writes remain unauthorized.

## Known-good navigation

| Need | Start here |
|---|---|
| Durable methodology | `docs/ai/DECISIONS.md` D-009–D-015 |
| Phase 9 handoff | `docs/HandoffPhase9.md` |
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

- Phase 7 acceptance: 160 synthetic tests, RF Validator Tasks 7.2/7.4 PASS.
- Phase 8 harness and repairs: repeated Task 8.4 acceptance-gate PASS reports.
- Revision `3d8439c`: 215 complete RF synthetic tests passed with bytecode and pytest cache disabled; focused validator run passed 107 tests.
- V1 artifact: independent Task 8.5 read-only validation passed integrity, provenance, termination, resource, reconciliation, and protected-operation gates; readiness failed only because four-role feasibility is false.
- Revision `df65217`: 110 focused synthetic tests passed with bytecode and pytest cache disabled before the v2 run.
- V2 artifact: the final read-only RF Validator passed all ten acceptance gates, including exact v1 reconciliation, both-label new-family coverage, five qualifying components, protected-operation checks, and sole-output verification.
- Phase 8 owner acceptance: 2026-09-25; Phase 9 planning only, with no dataset, split, training, calibration, model, raster, manuscript, or LR-write authorization.
- Phase 9 Task 9.1: RF Validator decision `AGGREGATE-ONLY EXPANSION REQUIRED`; owner approved D-015 without revisions on 2026-09-27. Task 9.2 source-only site-inventory implementation is authorized; production execution is not.
- Pre-Task 9.2 Ponytail cleanup: four unused RF prototype/utility scripts and unreachable legacy binary-publication bodies were removed without changing active fail-closed guards; 220 complete synthetic tests and a final 68-test focused rerun passed. No production input, artifact, model, raster, manuscript, or LR operation occurred.
- Phase 9 Task 9.2: final read-only RF Validator `PASS` with one policy warning after current-source inspection and an 80-test focused run. The complete RF synthetic suite passed 244 tests; `git diff --check` passed; `output/phase9` remained absent. Owner approval of the mixed-material class-order policy, a clean revision, and separate production authorization remain required.
- Phase 9 Task 9.3: read-only architecture audit passed the teacher/collector foundation but failed production publication, leakage-graph, split-allocation, and protected-test readiness. D-016 consequently authorizes only the bounded aggregate diagnostic; the publisher architecture remains proposed with numerical final-allocation targets and protected storage/ACL separately gated.
- Phase 10 source-only Task 9.3 implementation: 61 focused synthetic temporary-path tests passed with bytecode/cache disabled. The final read-only RF Validator passed after feature-domain, per-run provenance, null-identity, dataset-hash, and destination-pattern repairs. No production artifact or operation was authorized.
- Consolidated Phase 9/10 revision: the complete RF synthetic suite passed 254 tests with bytecode and pytest caching disabled; tracked worktree/index were clean afterward except for preserved untracked `Thesis-try.qgz`.

## Update procedure

1. Inspect the Git delta and claim-specific evidence.
2. Update only affected facts and preserve uncertainty labels.
3. Record the reviewed revision, artifact identity, command, and authorization boundary.
4. Append to `DECISIONS.md` only after explicit approval of a durable methodology choice.
5. Never overwrite an immutable artifact; use a separately approved versioned successor.
