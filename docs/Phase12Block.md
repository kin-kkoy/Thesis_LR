# Phase 12 Block: failed Set C allocation/export

Report date: 2026-10-03 (Asia/Manila). Failed invocation: 2026-10-02.

Intended audience: the parent orchestrator and a new diagnostic/fix session. This report preserves the failure evidence and boundaries; it is not authorization to retry, inspect membership, or change the allocation policy.

## 1. Outcome and immediate hold

The separately authorized one-run Phase 12 allocation/export invocation at commit `1c5a9727ae6c74b9017efae042eb1999b036a8f0` exited with code **1** after **37.846 seconds**. The captured error class was **`RuntimeError`**. No complete protected/development release pair, final target directory, attestation, or remaining staging directory was observed afterward. Both approved parent directories remained present and empty.

**The historical root cause is unresolved.** The raw stderr traceback and exact exception message were not retained. A source review identifies allocator failures as plausible explanations, especially failure to prove an exact optimum within an individual solve, but it does not establish which failure occurred.

The one-run approval was consumed when the process started. There is no remaining authorization to retry, clean up, change ACLs, train, or inspect protected membership. Preserve the existing directories and repository work until the orchestrator supplies a new structured plan and any necessary approval.

| Gate or claim | Finding | Evidence and limit |
| --- | --- | --- |
| Approved preflight | PASS / VERIFIED from recorded tool results | Exact hashes, clean pinned revision, paths, absent targets/stages, and resource floors passed before parent creation. |
| Parent creation order | PASS / VERIFIED from recorded tool results | Both empty parents were created after successful preflight. |
| Allocation/export execution | FAIL / VERIFIED | Process exited 1; no successful release was reported. |
| Resource supervision | PASS / VERIFIED within sampled evidence | No supervisory stop; sampled available memory and disk remained above approved floors. Continuous extrema and process peak RSS were not measured. |
| Paired publication | FAIL / VERIFIED at postfailure inspection | No final targets or attestations; paired-receipt status was `no-complete-pair-receipt`. |
| Exact exception and historical root cause | NEEDS EVIDENCE | Only the error class survived; solver status, objective index, exception message, and traceback are unavailable. |
| Individual three-second solve limit as cause | WARNING / INFERRED hypothesis | Consistent with source and timing, but not proven by the retained run evidence. |
| Production retry permission | FAIL / not authorized | The approval explicitly allowed one invocation and no retry. |

## 2. Evidence provenance and scope

The run telemetry below comes from identifiable preflight, supervisor, and postfailure tool responses in the originating chat. It is transcribed here for handoff; there is no retained raw stdout/stderr log attached to this report. Source conclusions come from the read-only validator review and targeted inspection of the current source at the pinned revision.

Use these distinctions throughout the handoff:

- **VERIFIED:** supported by recorded tool output or the inspected source.
- **INFERRED:** consistent with verified evidence, without a direct historical observation.
- **NEEDS VERIFICATION / NEEDS EVIDENCE:** the retained evidence cannot establish the claim.

This report contains no observation rows, component-to-role mapping, split seed, or final-test membership. No accepted CSV or manifest contents were opened for this report. Hashes are historical preflight results, not a fresh hash operation.

The compact project-state/startup records still describe an earlier source-only stage. They do not override the later explicit owner authorization, recorded invocation, or failure evidence. The orchestrator should reconcile that stale navigation context before planning further work.

## 3. Accepted package and source contract

The accepted Set C v1 package was identified by these approved whole-file hashes:

| File | SHA-256 |
| --- | --- |
| `Thesis_RF/Code/output/datasets/authoritative_set_c/v1/set_c.v1.csv` | `7f0f1867445141f365687d127e449117f234348cec571990d70bc2b226ba40f6` |
| `Thesis_RF/Code/output/datasets/authoritative_set_c/v1/manifest.json` | `b700f1b9f196893f23e7190daf1bc3c68b0f569f99a977ec9e919f0de73e9e3c` |

Previously accepted Phase 11 evidence describes 10,733 observations from 884 complete model-free teacher runs and 26 qualifying singleton-family leakage components. Those accepted aggregate properties are background evidence, not newly revalidated dataset properties in this report.

Relevant decision records are D-015, D-029, D-033, D-034, D-035, D-038, D-040, and D-041 in `docs/ai/DECISIONS.md`. Read their full wording before proposing a fix. In particular:

- D-035 requires exactly **11/5/5/5** components for train/validation/calibration/final test.
- Role positive minima are **200/100/100/100** and negative minima are **800/400/400/400**.
- Each qualifying component has at least five positives and five negatives; material classes 1–5 must occur in each role; no component may contribute more than 50% of a role's positives.
- Soft positive targets are **338/153/153/153**; soft row targets are **4541/2064/2064/2064**. Sequential optimization implements the approved lexicographic objectives and deterministic tie-breaking.
- Component order uses a dataset-bound digest rather than a selected random split seed. Equal predictor vectors remain audit collisions under D-029, rather than automatic cross-family leakage edges.
- D-038 fixes the main-repository source and release destinations. Protected-first publication, no overwrite, and an incomplete-publication hold are part of the source contract.
- D-041 makes owner approval reference text audit evidence, not machine-verifiable proof of approval. An operator opt-in and a clean pinned revision remain required.

The procedural holdout does not claim cryptographic unlinkability or enforced blindness for an account able to read the canonical package. No Windows identity check was required for the actual owner-authorized invocation.

## 4. Pinned checkout and earlier validation

Main checkout:

`C:\Users\gibgib\Thesis-Cellular-Automata`

Execution revision:

`1c5a9727ae6c74b9017efae042eb1999b036a8f0`

Earlier independently reviewed source revision:

`46701689dbf6fea6b2dfb2d9561614c605365e63`

Preauthorization comparison found the runner/exporter, focused tests, and ignore rules unchanged from the reviewed revision; the main project-state documentation differed. The recorded preflight found the main checkout clean. Read-only postfailure checks also found the same HEAD and clean status before this new report was created.

Recorded test evidence:

| Validation | Recorded result |
| --- | --- |
| Phase 12H.1 / independent H.2 runner and release contract tests | 20 passed |
| Four focused test files, parent review | 52 passed, 1 skipped |
| Independent preauthorization rerun at the main pinned revision | 52 passed, 1 skipped in 13.22 seconds |

The skipped test required Windows symlink-creation privilege. Simulated reparse-point coverage passed. These results demonstrate the covered synthetic source contract; they do not demonstrate successful optimization on the accepted production component inventory.

Historical preauthorization command, run from `Thesis_RF/Code`:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '..\..\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider --basetemp "$env:TEMP\setc_phase12_preauth_main_1c5a972" tests/test_set_c_split_runner_contract.py tests/test_set_c_split_release_contract.py tests/test_set_c_split_release.py tests/test_training_contract.py
```

No tests were rerun while creating this report. Prior results apply to their recorded environment, revision, command, and synthetic inputs.

## 5. Exact historical authorization

Audit reference:

`P12-SETC-V1-ONERUN-1C5A9727-R2`

The owner authorized one fresh invocation at the clean pinned revision, subject to successful preflight. The conditions were:

1. Verify the exact accepted CSV and manifest hashes, clean revision, approved paths, absent targets/staging, at least **4 GiB available memory**, and at least **2 GiB free disk on C:**.
2. Only after preflight passes, create the two required empty parent folders.
3. Invoke the reviewed command once under **30-minute supervision**.
4. Any invocation consumes approval. No retry, cleanup, ACL change, training, or final-test inspection is authorized.

A contemplated separate postpublication read-only protected validator was not activated by this failed invocation. Do not treat its proposed future scope as current membership-access permission.

## 6. Preflight and parent creation

An initial sandboxed preflight encountered `Get-Item` access denial on the accepted CSV path. It stopped without hashing or writing. The same authorized preflight was then performed with elevated tool access and passed. That access retry was not a production invocation.

Successful preflight timestamp: **2026-10-02T04:37:56.3477151Z**, or **12:37:56.3477151 Asia/Manila**.

| Reading | Recorded result |
| --- | --- |
| `preflightPass` | `true` |
| Git HEAD | `1c5a9727ae6c74b9017efae042eb1999b036a8f0` |
| `gitClean` | `true` |
| CSV SHA-256 | `7f0f1867445141f365687d127e449117f234348cec571990d70bc2b226ba40f6` — matched |
| Manifest SHA-256 | `b700f1b9f196893f23e7190daf1bc3c68b0f569f99a977ec9e919f0de73e9e3c` — matched |
| Release parents absent | `true` |
| Tracked release files | `0` |
| Both release trees ignored | `true` |
| Reparse paths detected | `[]` |
| Available physical memory | `6,003,359,744` bytes; approximately 5.591 GiB |
| Available C: disk space | `89,923,002,368` bytes; approximately 83.747 GiB |
| Memory / disk floor checks | Both passed |

Memory was read from `Win32_OperatingSystem.FreePhysicalMemory`, converted from KiB to bytes. Disk was read from `.NET DriveInfo('C').AvailableFreeSpace`. These are system-availability readings, not Python process RSS.

Parents were created at **2026-10-02T04:38:20.2150383Z**, or **12:38:20.2150383 Asia/Manila**, after preflight:

- `C:\Users\gibgib\Thesis-Cellular-Automata\SetC_Protected`
- `C:\Users\gibgib\Thesis-Cellular-Automata\SetC_Development`

Both were confirmed empty at creation. No ACL changes were made. Approved final targets were each parent's `set_c_v1` child.

## 7. Single invocation and supervision

The following is a **historical command record, not permission or an instruction to rerun it**:

```powershell
& 'C:\Users\gibgib\Thesis-Cellular-Automata\.venv\Scripts\python.exe' -B 'C:\Users\gibgib\Thesis-Cellular-Automata\Thesis_RF\Code\run_set_c_split.py' --execute-approved-run --owner-approval-reference P12-SETC-V1-ONERUN-1C5A9727-R2 --expected-revision 1c5a9727ae6c74b9017efae042eb1999b036a8f0
```

The supervisor actually launched the interpreter through `.NET ProcessStartInfo`, without a shell, with a hidden window and redirected stdout/stderr. Working directory was `C:\Users\gibgib\Thesis-Cellular-Automata\Thesis_RF\Code`.

The supervisor was configured to stop the process tree at 30 minutes, below the available-memory/disk floors, or on monitoring failure. Resource checks used approximately two-second waits plus monitoring overhead. It did not terminate the run.

| Run field | Recorded value |
| --- | --- |
| Invocation started | `true` |
| Start UTC | `2026-10-02T04:39:48.5724646Z` |
| End UTC | `2026-10-02T04:40:26.4185883Z` |
| Start / end Manila | `12:39:48.5724646` / `12:40:26.4185883` |
| Elapsed | `37.846` seconds |
| Exit code | `1` |
| Supervisory stop reason | `none` |
| Initial available memory | `6,062,411,776` bytes; approximately 5.646 GiB |
| Minimum sampled available memory | `5,832,896,512` bytes; approximately 5.432 GiB |
| Initial available C: disk | `89,922,879,488` bytes |
| Minimum sampled available C: disk | `89,922,748,416` bytes; approximately 83.747 GiB |
| Resource samples | `18` |
| stdout length | `0` characters |
| stderr length | `1,090` characters |
| Retained stderr error class | `RuntimeError` |

The recorded minima are sampled minima. They cannot rule out every transient between samples. There is no observed Python peak RSS, exact solver timing, solver status/message, failing objective number, or traceback in the retained evidence.

## 8. Postfailure artifact state

Receipt/path inspection timestamp: **2026-10-02T04:40:59.6767571Z**, or **12:40:59.6767571 Asia/Manila**.

| Metadata | Recorded value |
| --- | --- |
| Protected final target exists | `false` |
| Development final target exists | `false` |
| Protected staging names | `[]` |
| Development staging names | `[]` |
| Protected attestation exists | `false` |
| Development attestation exists | `false` |
| Protected publication state | Empty / unavailable |
| Development receipt state | Empty / unavailable |
| Protected attestation receipt hash matches | `false` because no pair exists |
| Paired-receipt status | `no-complete-pair-receipt` |

Subsequent read-only metadata inspection confirmed both parents existed and were empty. No membership content was inspected. There was no manual cleanup or retry.

**Do not infer that staging never existed.** The exporter removes its private staging directories on certain exception paths. Empty parents after failure are compatible with an allocation failure before staging, or with some failures after staging followed by the exporter's own exception cleanup. They do not establish whether internal assignments were ever computed.

## 9. Source trace and candidate RuntimeError paths

Primary source: `Thesis_RF/Code/modules/set_c_split_release.py`. Line references describe the inspected pinned source; recheck them if the revision changes.

Verified control flow:

1. `export_set_c_split` verifies source/provenance and derives the graph inventory internally.
2. It calls `_allocate_components` around line 491.
3. Private stages are created only afterward, around lines 507–508.
4. Protected publication occurs first, around line 599.
5. Development staging is privately validated and renamed without overwrite to complete publication.
6. Post-protected-publication failures are wrapped around lines 631–634, leaving a protected-only incomplete bundle on manual hold.
7. The outer exception handler around lines 636–640 removes private stages that still exist.

| Candidate path | Source location | Consistency with observations | Status |
| --- | --- | --- | --- |
| `D-035 allocation did not prove an exact optimum` | `_allocate_components.solve`, approximately line 1115 | Occurs before stage creation. A non-success result, nonzero status, or missing solution can produce this error. Repeated solves make tens of seconds plausible. | Leading hypothesis; exact historical result unknown. |
| `D-035 allocation has a fractional solution` | Approximately line 1118 | Also before stage creation. Requires a returned solution failing the integrality check. | Plausible source path; no run evidence of fractional values. |
| `D-035 allocation violates a hard constraint` | Approximately line 1122 | Also before stage creation, after rounding and constraint checks. | Plausible source path; no run evidence of a constraint violation. |
| `D-035 optimizer returned an invalid assignment` | Approximately line 1142 | Before stage creation, after sequential solves and final assignment checks. | Plausible source path; no retained optimizer output. |
| `Development publication failed; protected publication remains incomplete and requires human recovery` | Approximately lines 631–634 | Under inspected flow, a protected final directory should remain. This conflicts with empty parents unless an unobserved external event occurred. | Disfavored by artifact evidence; no evidence of external removal. |
| Other dependency-originated RuntimeError | Called validation, solver, or filesystem dependencies | An error class alone cannot exclude all exceptions raised inside dependencies. | Not exhaustively excluded. |

The solve uses `options={"time_limit": 3, "mip_rel_gap": 0}`. The sequential objectives comprise eight deviation objectives and 26 component tie-break objectives: up to **34 MILP calls**. Each successful objective is fixed before the next solve.

The exact-optimum guard requires `result.success`, `result.status == 0`, and a non-null solution. Its error text collapses several different failure reasons. A time limit is one possibility; infeasibility, another solver failure, or a missing solution are also possibilities. The 37.846-second wall time does not identify a solve index or prove a timeout. The code's individual solve limits are distinct from the external whole-run supervision.

Earlier Phase 12G found public loading/pair verification could accept an unpublished development stage after receipt creation. That defect was repaired and synthetic regression coverage added. No evidence establishes it as the cause of this run failure.

## 10. Smallest next diagnostic: synthetic only

The read-only validator recommended the following minimal mechanism check. **It has not been run as part of this report.** It is a proposed diagnostic for the orchestrator's plan, not a historical-root-cause finding.

1. Construct an in-memory synthetic inventory of 26 components with synthetic identities and sufficient positive/negative/material counts.
2. Stub the MILP call to return an unsuccessful, time-limited result with no accepted exact optimum.
3. Assert the precise `D-035 allocation did not prove an exact optimum` RuntimeError.
4. Assert staging creation and publication were never reached. Use synthetic OS-temporary paths only if the exporter wrapper is exercised.
5. Emit only failure-stage/result metadata, not assignments or row identities.

This establishes that the suspected mechanism produces the observed error class and no-stage behavior. It cannot establish that the historical run actually took that path. Additional synthetic cases can cover fractional solutions and constraint rejection if justified by the first check.

Existing synthetic success fixtures and publication interruption tests do not substitute for this failure-result coverage. A production replay is not the smallest permitted diagnostic and is not authorized.

## 11. Requirements for the orchestrator's structured fix plan

The next session should first read root and RF `AGENTS.md`, this report, relevant decisions, and the runner/exporter/test source. The plan should separate source diagnosis from any proposed production retry.

Recommended sequence:

1. Preserve the current parents, accepted package, and all pre-existing work. Confirm current Git state and metadata without opening protected membership or accepted rows.
2. Establish the smallest synthetic failure-result regression and inspect the current failure reporting. Record what it proves and what remains unknown.
3. Propose scoped redacted diagnostics that retain the exception message/traceback, failing stage, and solver status/objective index without printing rows, mapping, or final-test membership. Obtain the appropriate file-scope approval before modifying source or tests.
4. Review any proposed solver/resource policy change against D-035. Do not silently increase limits, accept approximate optima, weaken hard constraints, change deterministic tie-breaking, or treat prospective feasibility as proof every sequential solve terminates within three seconds.
5. Run bounded synthetic success, failure, interruption, and loader-rejection tests, with independent source validation.
6. If source or project-memory records change, prepare a new reviewed clean revision and report its full SHA. The old execution pin cannot describe a new commit.
7. Only then seek a new, explicit production authorization with exact command, preflight, resource limits, permitted folders, output retention, and failure hold. The consumed R2 approval cannot be reused.

Any diagnostic that reads the accepted rows, derives actual roles, or executes real allocation requires a separately authorized scope. No training, tuning, calibration, final evaluation, ground-truth access, ACL changes, or manual artifact cleanup belongs to the current report request.

## 12. Inspected evidence, commands, writes, and memory

Paths inspected in the originating investigation included:

- `AGENTS.md`, `Thesis_RF/AGENTS.md`, `docs/ai/PROJECT_STATE.md`, and relevant decision records.
- `Thesis_RF/Code/run_set_c_split.py`.
- `Thesis_RF/Code/modules/set_c_split_release.py`.
- Relevant helper source: `set_c_leakage.py`, `set_c_publication.py`, and `feature_pipeline.py`.
- Focused synthetic split/release/runner tests; prior training-contract validation evidence.
- Only release-path directory metadata during the postfailure audit.

Read-only command families used in the investigation were `Get-Content`, targeted `rg`, per-command-safe-directory `git rev-parse` / `git status`, and `Get-Item` / `Get-ChildItem` for path metadata. The postfailure validator did not run tests, change files, hash/read accepted inputs, rerun production, or clean folders.

For this report task, inspections included root instructions, project state, Git HEAD/status, checking for a pre-existing report or nested docs instructions, and targeted exporter source excerpts. The only intended persistent write is this new file, `docs/Phase12Block.md`. No source, decisions, project-state record, dataset, release folder, ACL, model, or ground-truth file is modified by this task.

Creating this report makes the previously clean main checkout contain a new documentation file. No commit is created; HEAD remains the recorded pin. A future production preflight must assess its own current clean revision rather than reuse the earlier clean-status claim.

Memory impact: unchanged — this report consolidates already reported findings without changing source, decisions, or validation conclusions. The stale project-state/run-status record is a project-memory update candidate for the orchestrator's separately scoped reconciliation.

**Next exact approval needed:** none to read this report and prepare a bounded source/synthetic diagnostic proposal. Source/test edits, a real diagnostic replay, cleanup, protected inspection, or a new production invocation each require authorization appropriate to their scope. The historical one-run approval is exhausted.
