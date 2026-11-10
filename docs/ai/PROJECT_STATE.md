# Project State — Random Forest Thesis Track

## Record contract

- **Purpose:** Navigation only; source/inputs/artifacts/raw records are evidence.
- **Scope:** RF work/shared coordination. `Thesis_LR/**` is LR side owned/read-only.
- **Last updated:** 2026-11-11 (Asia/Manila), Frozen Route A RF/LR benchmark accepted D-397.
- **Snapshot:** `734b42a`, `codex/rf-airbase-simulation`; prior edits preserved; work uncommitted.
- **Current handoff:** D-397 accepts the frozen RF/LR benchmark. CA-only design review is next.
- **Historical index:** Tables retain October 12 evidence, not current execution status.
- **Staleness rule:** Recheck changed source, input/artifact hashes, configuration, dependencies or approvals.
- **Size policy:** Keep this file below 16 KiB. Durable methodology belongs in `DECISIONS.md`.

<!-- BEGIN CODEX DIGEST -->
## Compact startup digest

- **Status:** Phases14/15 closed; F01 closeout-ready D-157; Airbase D-167–169 complete, independence unverified.
- **Accepted:** Cropped Set C, train-only RF, frozen sigmoid/F01; D-120/D-121 owner acceptances.
- **Limits:** F01 calibration. F1 315°:0.785902, 49°:0.852327 on 135,992 cells; 42.33% reference-burn exclusion. No held-out/physical-validity claim.
- **Gate0.40:** D-165 seeds42–46 meanF1 RF0.748423±0.077230 / LR0.715312±0.064599; tests560 PASS.
- **TP sets:** D-166 RF cutoff/no-cutoff identical at all5 seeds;315°/49°42 share1793 of1795, two unique each. Direct recall counts match.
- **Airbase:** D-167 no-cutoff42–46 meanF1 0.243773±0.029503; domain215553/ref178of281; five inactive; tests601 PASS.
- **H90:** D-168 owner-assumed1min/update;42–46 meanF1 .294022±.019082, recall .823596, precision .179211; reached822.6; five censored/active90;631tests PASS. Clock uncalibrated, no suppression/fire-out claim.
- **Airbase LR0.40:** D-169 seeds42–46 meanF1 .268503±.019688, recall .894382, precision .157980; reached1008.6; inactive103/123/135/125/99. Existing cap300, no new stop/tuning;679tests PASS.
- **Adopted:** INFERRED NASA true-49 converts to constant grid-from 48.825208, ignition pixel centre/six decimals. F01 is owner-identified Santa Maria.
- **Timing:** D-171 owner confirms Airbase=Sitio Sangi Interior,Brgy.Pajo via BFP1328H row. DSWD86min/StaMaria212min retained; BFP1328≠DSWD1332; historical HR11 basis preserved.
- **Plan:** D-198/D-200 both frozen models in RF CA; RF/noGate, LR/.40, clocks1/.5/2. New StaMaria892; historical894/v3 preserved. ISPRS415=Tan;319=Patac.
- **Next:** Prepare a matched CA-only design for both incidents before implementation or execution.
- **Pending:** Manuscript/viewer review; A/B, matched CA and LR side handoff qualified. 16B prior-use/overlap/protocol and synthetic final test gated.
- **Boundaries:** D-366 and earlier execution approvals consumed. Preserve pilot outputs and evidence.
<!-- END CODEX DIGEST -->

## Current verified state

| Status | Current fact | Evidence | Recheck trigger |
|---|---|---|---|
| `VERIFIED` | RF is the writable implementation track; LR remains read-only. | `AGENTS.md`; `Thesis_RF/AGENTS.md` | Ownership changes. |
| `PASS` / `WARNING` | Airbase LR0.40 meanF1 .268503, recall .894382, precision .157980; domain215553/ref178of281. UncalibratedLR/gated vs calibratedRF/ungated; independence unverified. | D-169; current handoff | Inputs/model/protocol changes. |
| `PASS` / `ASSUMPTION` | H90 footprints25.07% smaller; active counts19/20/18/25/12; same-seed original subsets. Assumed clock, not fire-out. | D-168; duration report above | Time mapping/horizon changes. |
| `VERIFIED` | Set C is the active 11-feature candidate; Sets A/B remain separate historical chains. | D-009–D-014 | Owner supersedes the methodology. |
| `VERIFIED` | The RF estimand is eligible mapped-building-cell ignition at `t+1` from state/features at `t`; dynamic neighbor features use BLAZING at `t`. | D-009, D-010, D-013, D-014 | Target/timing contract changes. |
| `VERIFIED` | Authoritative Set C labels come only from the versioned model-free stochastic teacher. Incumbent RF transitions are not independent truth. | D-014; Phase 7 source and validator reports | Teacher contract changes. |
| `VERIFIED` | Phase 7: isolated D-014 teacher, five-state `np.int8` transitions, canonical `all_eligible` collector, outcome records, exact retries, disabled publication. | `10b1b7c`; acceptance; 160 tests | Interfaces change. |
| `VERIFIED` | Phase8 harness: disabled, exact model-free path, aggregate-only/no-overwrite atomic publication. | `ffbda28`, `2fd0e68`, `3d8439c`; Task8.4 PASS | Harness changes. |
| `VERIFIED` | Full-grid building/material mapped identities agree after material-specific normalization; combined validity still excludes slope-invalid cells. | Revision `3d8439c`; real-input preflight; approved domain hashes | Loader or raster changes. |
| `VERIFIED` | Phase8v2 df65217:32 inactive runs,74/74 transitions,384rows/26positive,5both-label components;v1 reconciled. | Immutable reports; Validator PASS | Evidence changes. |
| `VERIFIED` | The thesis owner formally accepted Phase 8, including the v2 extension, and authorized Phase 9 planning only under a separate bounded contract. | Owner acceptance, 2026-09-25 | Owner grants or revises Phase 9 authorization. |
| `VERIFIED` | D-015:128×128 origin(0,0) grouping;26 families/17 conditions; aggregate discovery, role minima, conflict-preserving duplicates. | Task9.1; owner2026-09-27 | Methodology changes. |
| `VERIFIED` | Phase 8's five components and 26 positives are inadequate for production row publication; aggregate-only independent-family expansion is required. | Phase 9 Task 9.1 RF Validator review; D-015 | Approved adequacy expansion passes all gates. |
| `VERIFIED` | Gate 4 v2: first-present material 1–5 assignment, aggregate composition, global block reservation, no-overwrite report. | `modules/set_c_site_inventory.py`; configuration; 255 synthetic tests | Source/config/evidence changes. |
| `VERIFIED` | D-018 assigns mixed-material components to the first eligible class in deterministic order 1–5 and globally reserves their full leakage footprint. This is an inventory rule, not a prevalence or independence claim. | Owner approval; D-018 | Owner revises D-018. |
| `VERIFIED` | Historical D-016 authorized bounded aggregate adequacy only after prerequisites; production/split/train/final-test separately gated. | Tasks9.1–9.3; owner2026-09-27 | Authority changes. |
| `VERIFIED` | D-017 disabled architecture: versioned staging/canonical rows/joint leakage components/development loaders/fail-closed final evaluator. Targets/storage separately gated. | Owner;61tests; Validator PASS | Source/approval changes. |
| `VERIFIED` | D-029: provenance-based duplicate identities; predictor collisions audit-only. V3 replay disabled/block-only; checkpoint identity/state/report checks synthetic-only. | D-029; 47 focused/286 full tests | Review or authorized replay. |
| `VERIFIED` | Gate6v3:26 qualifying singleton components/feasible unexposed minima;97collision links audit-only. | Payload/file3891f54b...9ca96/a6cd757f...8bbc; Validator PASS | Artifact/policy/source changes. |
| `VERIFIED` | Agent profile pins and parsed configuration recorded in D-030. | `.codex/agents/*.toml`; D-030 | Profile/model changes. |
| `VERIFIED` | Set C v1 `161b8d0`: 884 ordered complete/inactive authoritative runs; 10,733 rows (797 positive/9,936 negative). Manifest/dataset/run/batch hashes and 3,902-file inventory independently PASS; no split published. | Manifest `b700f1b9...e9e3c`, payload `7e993176...e8cbd`; Validator PASS | Package/contract changes. |
| `VERIFIED` | Task11.1:1508batches/884runs reconcile; no checked row/schema/domain/count defect; allcomplete/inactive/authoritative/transitionmatched. | Phase11 report; Validator | Package/contract changes. |
| `VERIFIED` | Task 11.2 reconstructed 26 singleton-family components with 9–58 positives and 188–496 negatives each. A role-blind solve satisfied every D-015 component/class/material/concentration minimum without publishing assignments. | Same report; actual CSV; Gate 6 v3 comparison | Package bytes or D-015/D-029 policy changes. |
| `VERIFIED` | D-035:11/5/5/5 components,D-015 minima,material coverage,≤50%positive concentration,seedless digest order/soft balance. | Task12.1 | Policy/dataset changes. |
| `VERIFIED` | D-040 staging rejection and D-041 default-deny opt-in/audit-reference/clean-pin guards remain active. | Runner/exporter; focused tests | Source/approval change. |
| `VERIFIED` (recorded) | R2 at main HEAD `1c5a9727` exited 1/RuntimeError after 37.846 s. Parents recorded empty; no targets/stages observed. No fresh path inspection. | `docs/Phase12Block.md` §§5–8; D-042 | New authorized evidence. |
| `NEEDS EVIDENCE` | Historical message/traceback/objective missing; timeout hypothetical. Resource floors passed sampled checks only; peak RSS unknown. | Same report; source | New authorized evidence. |
| `VERIFIED` | Diagnostics at `7f6730b` retain safe solver/stage context and redact chains; recorded tests 76/1 skipped. | D-042/D-043; Git commit | Source/test/environment change. |
| `PASS` / `VERIFIED` | A1 redaction/cap 60 synchronized; recorded suites each 143/1 skipped. Production sufficiency untested. | D-045/D-048 | Source/test/run changes. |
| `FAIL` / `VERIFIED` | R3 exited 1 after 40.210 s: objective 7, calibration row deviation, exact optimum not proven. Resources/evidence PASS; no targets/stages. | D-047; retained JSONL/final response | Separately authorized evidence review or source/run change. |
| `FAIL` / `VERIFIED` | R4 failed internal prepublication final-row consistency, exit 1/172.242 s. Capture/resources PASS; no targets/stages. | D-051; retained response | Owner-directed diagnosis. |
| `CLOSED` / `VERIFIED` | Phase 12 accepted after R5 paired publication and independent protected review. | D-053/D-054 | Artifact/decision change. |
| `VERIFIED` | Activated config SHA `f18953d5854286d29e9082caf8e5cd35f1719038627840a2c227ba42227dddfe` matches manifest; file hash matches. | Generation runner/config helpers; manifest | Config/source/package changes. |
| `WARNING` | The CSV supports source/provenance-consistent `all_eligible` and state-at-`t` timing, but row publication alone cannot prove execution behavior without teacher replay. | Task 11.1 evidence limits | A bounded replay is approved and verified. |
| `WARNING` | The package binds the approved resource limits but does not contain independent observed peak RSS, memory/disk minima, or elapsed-time telemetry. | Package manifest and independent RF Validator review | Independent execution telemetry is provided and verified. |
| `WARNING` | Lead profile delegation/output wording conflicts with parent and read-only boundaries; parent/user rules govern. | `.codex/agents/tech-lead.toml`; D-030 | Profile repair. |
| `NEEDS VERIFICATION` | Historical Set A/B/C metric claims are reproducible from identified datasets, models, configurations, splits, commands, and immutable outputs. | `docs/findings_SetA.md`, `docs/findings_SetB.md`, `docs/findings_SetC.md` | Close only with primary provenance. |

## Active blockers and authorized follow-up

1. `CLOSED` — Phase 12 R5 publication/protected review accepted; historical failure/evidence limits remain D-042–D-054.
2. `WARNING` — 128×128 blocks provide conservative containment, not measured independence; D-029 predictor collisions remain audit-only.
3. `NEEDS EVIDENCE` — Continuous resource extrema and independent teacher timing/complete eligible capture remain unverified.
4. `WARNING` — Same-account D-038 procedural holdout cannot enforce blindness or prevent canonical-CSV recomputation.
5. `PASS` — D-035/D-040 guards retained; R2/R3/R4 causes/limits remain historical. No mathematical optimum certificate.
6. `PASS` — 13B.1 selective train/validation loader; validator reports 70 passed/1 skip; calibration contents unopened.
9. `WARNING` — No development component IDs or reconstructed CV. D-077 accepts prespecified sigmoid, not empirical superiority.
10. `VERIFIED` — Owner host: Windows PowerShell 5.1.26100.9444/Desktop; pwsh absent from PATH. Recovery host gate PASS, D-063; exit cause unverified.
11. `CLOSED` — Phase 13 accepted, D-071; no production Phase 14 authority.
12. `CLOSED` — 14E/14F source-only accepted; 14F both 24/0 suites PASS, D-099.
13. `FAIL` — 14G 19/1 (32.32s); callback error/record/conversion evidence missing. Prep incomplete; no launch, D-118.

## Known-good navigation

| Need | Start here |
|---|---|
| Parent handoff | `docs/RF_Parent_Handoff_2026-10-10.md` |
| Durable methodology | `docs/ai/DECISIONS.md` D-009–D-015 |
| Phase 9 handoff | `docs/HandoffPhase9.md` |
| Phase 11 handoff | `docs/HandoffPhase11.md` |
| Phase 11 validation report | `docs/Phase11_Tasks_11_1_11_2_Report.md` |
| Phase 12 Task 12.1 policy review | `docs/Phase12_Task12_1_Grouped_Allocation_Policy_Review.md` |
| Phase 12 failure / supervisor | Unchanged `docs/Phase12Block.md`; D-042–D-045; `run_set_c_supervised.py`; focused tests |
| Current review / authority | D-184 new two-file scope needs approval; native parity open |
| Local evidence copy | `Thesis_RF/Code/output/phase16_evidence/p16a-closeout-20261102-r1/`; D-160 manifest |
| Phase 13 numerical proposal for re-review | `docs/Phase13A_RF_Tuning_Protocol_Proposal.md` Draft 1; D-055 |
| Phase 14 protocol / source gates | `docs/Phase14A_Calibration_Final_Test_Protocol_Proposal.md` Draft 4; D-077–D-118 |
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
- D-042/D-043: diagnostics `7f6730b`, recorded 76/1 skipped (9.05 s). D-044 frame mismatch superseded by D-045 repair/source PASS and fresh 131/1 skipped (21.51 s); no production/optimality proof.
- D-047: capture/evidence PASS; production FAIL. D-048: engineer 143/1 skipped (14.31 s), validator 143/1 skipped (14.21 s); synthetic only.
- D-052: both reviewers147/1 skip; CSV fidelity/tamper PASS, synthetic only; R4 subtype unknown.
- D-053/D-054: R5/protected review PASS; no mathematical certificate; owner accepted2026-10-06.
- D-057/13B.1: Validator70 PASS/1 symlink skip,11.36s; reparse PASS; project python,3files,`-B -p no:cacheprovider`,fresh temp. Parent source only; no production proof.
- D-058–D-061 / 13D: Validator 96 PASS/1 skip/1 deselected; D-060 resolved. Parent source/hash review only.
- D-080/116: 14D accepted (104/1 skip); 14E accepted (both 198/0); 14F accepted (both 24/0 exit0). 14G patch PASS; fresh suite released.

## Update procedure

1. Inspect the Git delta and claim-specific evidence.
2. Update only affected facts and preserve uncertainty labels.
3. Record the reviewed revision, artifact identity, command, and authorization boundary.
4. Append to `DECISIONS.md` only after explicit approval of a durable methodology choice.
5. Never overwrite an immutable artifact; use a separately approved versioned successor.
