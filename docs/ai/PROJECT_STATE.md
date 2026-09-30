# Project State — Random Forest Thesis Track

## Record contract

- **Purpose:** Compact, Git-reviewable navigation and current-state index. Source, identified inputs, immutable artifacts, configurations, commands, and raw records remain the primary evidence.
- **Scope:** Random Forest work and shared coordination. `Thesis_LR/**` is partner-owned and read-only.
- **Last updated:** 2026-09-30.
- **Last reviewed base revision:** `f40e8c327327bc4ccd25a2c086aed22952b97b1a`.
- **Worktree note:** Gate 6 v2 recorded `source_dirty: false`; `Thesis-try.qgz` remains preserved.
- **Current reviewed RF work:** Gate 6 v3 passed independent validation. The first authorized Gate 7 generation invocation failed when the publisher rejected a required timestep-specific eligible-mask hash; no immutable package exists. The repair at `da25565` passed independent RF Validator review and 299/299 full synthetic tests.
- **Staleness rule:** Recheck a fact when its source revision, input/artifact hash, configuration, dependency, or approval changes.
- **Size policy:** Keep this file below 16 KiB. Durable methodology belongs in `DECISIONS.md`.

<!-- BEGIN CODEX DIGEST -->
## Compact startup digest

- **Active objective:** Seek fresh exact one-run Gate 7 authorization bound to the clean repaired revision. Do not rerun generation under the consumed approval.
- **Methodology:** D-009–D-015 define next-timestep eligible-cell ignition, canonical Set C predictors, `all_eligible`, model-free teacher labels, 128×128 conservative grouping, controlled population expansion, duplicate containment, and role-level adequacy gates.
- **Accepted source:** Phase 7 supplies the fail-closed model-free teacher and bounded 11-feature collector. Phase 8 v2 revision `df65217b91c1624bbfcff82e95cca15a45a6df2c` supplies the disabled aggregate-only pilot extension; 110 focused synthetic tests passed before execution.
- **Accepted diagnostic artifact:** `Thesis_RF/Code/output/phase8/phase8_set_c_feasibility_pilot_v2.aggregate.json`; canonical payload SHA-256 `9a4f48ea6b2b7f882985d63ff3cad7ef3c785736e223860f82637fdd9e45cbf3`; whole-file SHA-256 `b62a7bf542d0ab74a3ac71399e85048b3169d0a04b33c95625aec3ee39339d17`.
- **Diagnostic result:** All 32 runs were inactive, complete, authoritative, and count-matched. They produced 384 aggregate-observed eligible rows: 26 positive and 358 negative. Every diagnostic block size yielded five authoritative leakage components, all five independently containing both labels; four-role allocation is mathematically feasible, but no roles or final block dimensions were assigned.
- **Current authorization:** Gate 6 is complete. The one-run Gate 7 generation authorization was consumed by a failed invocation; a retry needs a fresh exact approval. Checkpoints/resume, splits, models, and final-test access remain unauthorized.
- **Hard boundary:** Do not run the aggregate expansion, publish rows, assign roles, train/tune/calibrate, access the final test, publish models/rasters, access `stack_ground_truth.tif`, change manuscripts, or write `Thesis_LR/**`.
- **Next gate:** Rebind the exact package, hashes, resource envelope, and one-time command to the clean repaired revision in a fresh owner authorization.
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
| `FAIL` | Gate 6 v2 completed 884/884 authoritative runs from clean revision `dfc3703` but failed D-015: feature-vector duplicate edges yield only three components (23, 2, 1 families), while spatial-block edges yield 26 singleton families. | Immutable v2 report; payload `b4ff93b2...0c197`; file `a998601d...93c8`; parent integrity validation | D-015 duplicate-edge policy changes or a new population is approved and evaluated. |
| `VERIFIED` | Gate 6 now prints flushed progress after committed runs 100–800 and 884. Synthetic temporary-path checkpoints resume at the first uncommitted run, reject drift/corruption, roll back failed-run identities, and conservatively charge failed/crashed-attempt time. Production resume is unreachable. | D-028; 26 focused and 281 complete synthetic tests; RF Validator PASS | Checkpoint source, policy, or production gate changes. |
| `VERIFIED` | D-029 implements the approved narrow duplicate-policy revision: published duplicate identities are provenance-based observation records; exact predictor collisions remain aggregate audit statistics. The v3 replay is disabled and block-only; checkpoint package identity/state-digest/report-verifier checks are synthetic-only. | D-029; focused 47/47 and complete RF synthetic 286/286 | Fresh independent validator review or separately authorized replay. |
| `VERIFIED` | The v3 replay runner requires an explicit flag, exact immutable source/destination/hashes, clean Git state, D-015 resource floors, a 300-second ceiling, atomic no-overwrite publication, and sole aggregate output. | Focused 31/31; complete RF 288/288; independent RF Validator PASS | Runner/replay source changes or execution is authorized. |
| `VERIFIED` | Gate 6 v3 independently validates 26 qualifying singleton block components and feasible unexposed role minima; all 97 predictor-collision family links remain audit-only. | Immutable v3 payload/file `3891f54b...9ca96` / `a6cd757f...8bbc`; final RF Validator PASS | V3 artifact, D-015 policy, or source binding changes. |
| `VERIFIED` | Agent profiles pin `rf_engineer` to `gpt-6-luna`/`medium`, `rf_validator` to `gpt-6-sol`/`medium`, `tech_lead` to `gpt-6-sol`/`high`, and `thesis_researcher` to `gpt-6-sol`/`medium`. | `.codex/agents/*.toml`; D-030; successful TOML parse | Profile settings or available models change. |
| `VERIFIED` | The disabled Gate 7 runner connects the approved 26-family/884-run plan to sequential model-free collection and bounded immutable publication. It binds per-run configuration hashes and approved run order, reserves publication disk, and enforces the shared runtime deadline through atomic promotion. | Revision `df63ef5`; focused 18/18 and complete RF 298/298; independent RF Validator PASS | Runner, publisher, approved artifacts, limits, or authorization changes. |
| `FAIL` | The first Gate 7 production invocation from revision `45733b1` raised `Batch domain-mask identity is inconsistent` during batch staging. The batch includes a timestep-specific eligible-mask hash that the publisher's two-key equality check omitted. The subsequent collector finalization error followed the staging exception. No immutable package exists. | Owner traceback; transition observer, collector, and publisher source; destination absence | Repaired source passes independent validation, clean revision, and fresh one-run authorization. |
| `VERIFIED` | The publisher now requires the two stable package mask hashes plus one valid timestep-specific eligible-mask hash. The failed attempt left no package or staging residue. | Repair `da25565`; focused 19/19; full RF 299/299; independent RF Validator PASS | A fresh authorized production execution or source change. |
| `WARNING` | `tech_lead` still has contradictory delegation/output instructions: parent-owned delegation and no human prompt copying conflict with “Prompt to Copy,” direct researcher invocation, and a request to update Markdown despite read-only/manuscript boundaries. | `.codex/agents/tech-lead.toml` lines 21, 42, 50–54 | Profile wording is explicitly repaired and reviewed. |
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
7. `PASS` — The immutable v2 report, exact 20 primaries and 20 reserves, and Gate 5 source-only harness are approved and independently validated; no production expansion has run.
8. `PASS` — Gate 6 v3 replay preserves the v2 execution evidence, treats its 97 predictor collisions as audit-only, yields 26 qualifying block-only components, and passes every D-015 adequacy gate without rerunning simulations.
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
- Phase 9 Task 9.2: final read-only RF Validator `PASS` with one policy warning after current-source inspection and an 80-test focused run. The complete RF synthetic suite passed 244 tests; `git diff --check` passed; `output/phase9` remained absent. D-018 later approved the mixed-material class-order policy.
- Phase 9 Task 9.3: read-only architecture audit passed the teacher/collector foundation but failed production publication, leakage-graph, split-allocation, and protected-test readiness. D-016 consequently authorizes only the bounded aggregate diagnostic; the publisher architecture remains proposed with numerical final-allocation targets and protected storage/ACL separately gated.
- Phase 10 source-only Task 9.3 implementation: 61 focused synthetic temporary-path tests passed with bytecode/cache disabled. The final read-only RF Validator passed after feature-domain, per-run provenance, null-identity, dataset-hash, and destination-pattern repairs. No production artifact or operation was authorized.
- Gate 4 v2: the repair passed the complete 255-test RF synthetic suite; one authorized execution from revision `6074064` produced the immutable v2 report (payload `25a0a27a...158cc`, file `26fb06f3...1de75`), and final read-only RF validation passed ownership, composition, provenance, disjointness, sole-output, and aggregate-only gates. The exact manifest and Gate 5 source-only implementation are approved by D-021.
- Gate 5 source-only implementation: the disabled 884-run aggregate harness is bound to the approved v2 manifest, enforces temporary-path and production-provenance boundaries, and retains only aggregate identity/counter state. Focused contract tests passed 17/17; the complete RF synthetic suite passed 272/272; final read-only RF Validator PASS. No Gate 5 aggregate, production run, row dataset, split, model, or final-test artifact exists.
- Gate 6 first attempt: clean revision `8b45b10`; preflight passed at 7,095,463,936 available memory bytes and approximately 95 GB free disk, then the command exited 1 before teacher execution because Git rejected a Windows-style `safe.directory`. Independent RF Validator confirmed the cause and absence of report/temp output. D-015 adequacy is `NEEDS EVIDENCE`; Gate 7 and Phase 11 remain blocked.
- Gate 6 repaired retry: clean revision `560ea9b`; focused 18/18 and full 273/273 synthetic tests plus RF Validator PASS preceded execution. The owner stopped the CPU-bound run at 45m11s. No report was published; an intact 69,632-byte diagnostic temp store indicates at least 171 sequential runs and first-six-family progress. Runtime is plausibly several hours under the approved six-hour cap, not demonstrably hung. D-015 adequacy remains `NEEDS EVIDENCE`.
- Gate 6 manual full-window result: immutable report payload/file SHA-256 `fe0322be...8cdb` / `6caf02cd...4bb`; 884/884 runs complete in 14,238.62 seconds; 10,733 rows (797 positive, 9,936 negative); only three qualifying leakage components (23, 2, and 1 families), so both the ≥23-component and unexposed role-feasibility gates fail. Independent RF Validator result: `FAIL`. Recorded clean revision `2fec62f` differs from the exact authorized `560ea9b`; Gate 7 and Phase 11 remain blocked.
- Gate 6 v2 source-only recovery: the existing aggregate identity store now yields duplicate-only, block-only, and joint component-size summaries plus per-family-pair cause counts without publishing identity values or rows. Execution remains disabled behind a new flag and versioned destination. Engineer focused/full tests passed 22/22 and 277/277; independent RF Validator and parent focused runs passed 22/22. V1 hash is unchanged and no v2 artifact exists. D-015 remains unchanged; production cause is `NEEDS EVIDENCE`.
- Gate 6 v2 execution: immutable report payload/file SHA-256 `b4ff93b2...0c197` / `a998601d...93c8`; 884/884 runs complete in 14,253.75 seconds from clean revision `dfc3703`; 10,733 rows (797 positive, 9,936 negative). Resource and protected-operation gates passed. Duplicate-only and joint components are 23, 2, and 1 families; block-only components are 26 singletons, proving feature-vector collisions are the sole reported cross-family edge cause. D-015 adequacy remains `FAIL`; no residual SQLite store exists.
- Gate 6 resumability source: progress prints only after committed runs 100, 200, …, 800, and 884. The synthetic-only SQLite journal binds configuration, source context, families, schema, and run order; it resumes from the first uncommitted run and retains no failed-run identities. Focused tests passed 26/26 and the complete RF suite passed 281/281. Production checkpoint/resume remains rejected pending a D-016 replacement covering trusted storage, crash cleanup, and complete resource telemetry.
- Gate 6 v3 replay result: payload/file SHA-256 `3891f54b...9ca96` / `a6cd757f...8bbc`; 26/26 block-only components qualify, 97 predictor-collision family links remain audit-only, all six D-015 gates pass, and independent RF Validator review passed. The replay took 0.034 seconds and created no checkpoint or row-level output.

## Update procedure

1. Inspect the Git delta and claim-specific evidence.
2. Update only affected facts and preserve uncertainty labels.
3. Record the reviewed revision, artifact identity, command, and authorization boundary.
4. Append to `DECISIONS.md` only after explicit approval of a durable methodology choice.
5. Never overwrite an immutable artifact; use a separately approved versioned successor.
