# Phase 14 RF manual run card

**Status: PREPARATION ONLY — NO PRODUCTION RUN AUTHORIZED.** The calibration configuration was activated in commit `215a61942f6f8a7f1811dff7b4517c18e64c6674`, but this card is not a run approval. Do not launch calibration or final-test work until the host rebaseline and its source/configuration are independently reviewed and committed, followed by a separate hash-bound owner approval.

## Recorded environment

The human-operated Stage 14H.2 collection completed with native exit status 0 and no stderr output. Its retained stdout SHA-256 is `8259a172687c9467bc349978f11a276878a487af77145c2deb557a408f9fc383`; the empty stderr SHA-256 is `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. These values describe that collection only.

| Item | Observed value |
|---|---|
| Python | 3.14.3 |
| scikit-learn | 1.8.0 |
| Optuna | 4.8.0 |
| Joblib | 1.5.3 |
| NumPy | 2.4.4 |
| pandas | 3.0.2 |
| SciPy | 1.17.1 |
| PyYAML | 6.0.3 |
| Platform | Windows-11-10.0.26200-SP0 |
| Interpreter path | `C:\Users\gibgib\Thesis-Cellular-Automata\.venv\Scripts\python.exe` |
| Windows PowerShell | 5.1.26100.9444, Desktop; `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe` |
| Rasterio (auxiliary) | 1.5.0 |
| pytest (test runtime only; not part of the production environment envelope) | 9.0.0 |

The nine runtime version strings above hash to `9e3c76eaafc18a7e59db39285c86358c49064d3e4f22c42feeea1081c1fab726`, using the implementation's sorted-key compact UTF-8 JSON representation. This lock digest does not certify the current source revision or authorize execution. The earlier interpreter-location warning did not recur in this collection; this does not establish its historical cause or prove that it is fixed in every context.

A later read-only host check reported Windows PowerShell `5.1.26100.9711`, `Desktop`, at the same executable path. This supersedes the earlier build only as the currently observed host value; it does not refresh the nine-package version evidence or prove a complete current environment collection. The current source/configuration rebaseline and synthetic checks must be reviewed and committed before a new calibration authorization is prepared.

## Calibration readiness gates — all must be completed later

1. **Do not launch from the current working tree.** Preserve unrelated changes. The calibration activation is already committed at `215a61942f6f8a7f1811dff7b4517c18e64c6674`; prepare and independently review a separate host-rebaseline commit before launch. Resolve the final source identities only after that commit exists; do not add a commit hash or config hash that creates a circular self-reference. An independently reviewed, clean evaluation/source revision and a current exact source inventory are required at launch.
2. **Respect the separate execution gate.** Calibration configuration activation was committed at `215a61942f6f8a7f1811dff7b4517c18e64c6674` with SHA-256 `6d24fbe570029ef5b778bd00e08f719919e83424f6d991f981d78fba635ff78a`. Activation does not authorize execution. The current host rebaseline requires fresh environment collection, source/configuration review and any required commit before a new one-run authorization. The final-test configuration remains disabled, unresolved and untouched.
3. **Obtain separate, fresh calibration authorization.** It must bind the final reviewed source commit, exact calibration configuration SHA-256, protocol SHA-256 `c7791fe57672541bbcde375b7858fe6b73cbdfc35eb41975730b49937abe6efc`, environment envelope and digest, job `calibration`, a fresh approval reference, a unique run ID, and exact source-file hash inventory. The authorization transport JSON and its SHA-256 are supplied through `PHASE14_AUTHORIZATION_JSON` and `PHASE14_AUTHORIZATION_SHA256`. Do not invent any of these future values, reuse an approval/run ID, or treat an approval reference as authentication.
4. **Use the rebaselined host and location after approval.** The proposed host pin is ordinary, non-administrator Windows PowerShell 5.1.26100.9711/Desktop, at `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`, with the exact interpreter above and `Thesis_RF\Code` as the working directory. It is not accepted for production until the rebaseline has passed independent review and is committed. Do not change execution policy, use bypass flags, substitute interpreters, or alter security/ACL settings.
5. **Allow the wrapper to perform its fresh prelaunch checks.** The reviewed wrapper has no standalone `--preflight-only` mode. On invocation it verifies job/approval/revision binding; authorization digest and schema; host and working directory; exact runtime-source inventory, clean state, and hashes; config hash and enabled state; fixed output/evidence path absence; and fresh memory/disk/time gates. A check failure means stop and retain evidence—do not run the child manually or retry.
6. **Confirm outputs are absent without creating them.** The runner derives the calibration destination under `Thesis_RF\Code\output\phase14\phase14-setc-rf-61e8f740-r1\calibration\<approved-run-id>` and evidence under `Thesis_RF\Code\output\phase14\evidence\<approved-run-id>`. Never pre-create these paths. If either exists, stop; no overwrite or cleanup is permitted.
7. **Confirm the approved operation and limits.** The future calibration job is one sigmoid calibrator using all 2,060 eligible calibration rows (153 positive, 1,907 negative), equal row weights, `n_jobs=1`, and the immutable Phase 13 forest. No forest refit/tuning, resampling, cross-validation, method comparison, threshold selection, final-test access, replay, CA simulation, ground-truth comparison, or deployment is included.

## Future launch form (NOT AUTHORIZED FOR EXECUTION)

This is only the source-supported outer invocation shape. It is deliberately unusable until a separate owner approval supplies and independently reviews every placeholder and the disabled configuration has separately been authorized and committed. Do not paste or run it now.

```powershell
$Repo = 'C:\Users\gibgib\Thesis-Cellular-Automata'
$Code = Join-Path $Repo 'Thesis_RF\Code'
$Runner = Join-Path $Code 'run_phase14_supervised.ps1'
$PowerShell = 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
$ApprovalReference = '<fresh-owner-approved-calibration-reference>'
$ExpectedRevision = '<full-reviewed-clean-commit-sha>'
$RunId = '<fresh-unique-owner-approved-run-id>'
$PreviousErrorActionPreference = $ErrorActionPreference
$LocationPushed = $false
$OriginalLaunchError = $null
$InvocationErrorRecords = @()
$LocationRestoreError = $null
$SupervisorExitCode = $null

try {
    $ErrorActionPreference = 'Stop'
    if (-not (Test-Path -LiteralPath $PowerShell -PathType Leaf -ErrorAction Stop)) {
        throw 'Required Windows PowerShell executable is unavailable'
    }
    Push-Location -LiteralPath $Code -ErrorAction Stop
    $LocationPushed = $true
    $PriorErrorCount = $Error.Count
    $PriorNewestError = if ($PriorErrorCount -gt 0) { $Error[0] } else { $null }
    $global:LASTEXITCODE = $null
    $ErrorActionPreference = 'Continue'
    & $PowerShell -NoProfile -NonInteractive -File $Runner `
        -Job 'calibration' `
        -ApprovalReference $ApprovalReference `
        -ExpectedRevision $ExpectedRevision `
        -RunId $RunId
    $SupervisorExitCode = $global:LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    $InvocationErrorObserved = ($Error.Count -gt $PriorErrorCount) -or `
        ($null -ne $PriorNewestError -and $Error.Count -gt 0 -and `
         -not [object]::ReferenceEquals($Error[0], $PriorNewestError))
    if ($InvocationErrorObserved) {
        $NewErrorCount = if ($Error.Count -gt $PriorErrorCount) { $Error.Count - $PriorErrorCount } else { 1 }
        $InvocationErrorRecords = @($Error | Select-Object -First $NewErrorCount)
        $OriginalLaunchError = $InvocationErrorRecords[0]
        throw $OriginalLaunchError
    }
    if ($null -eq $SupervisorExitCode) { throw 'Supervisor native exit status was not observed' }
    if ($SupervisorExitCode -ne 0) { throw ('Supervisor failed with native exit status ' + $SupervisorExitCode) }
}
catch {
    if ($null -eq $OriginalLaunchError) { $OriginalLaunchError = $_ }
    throw
}
finally {
    try {
        if ($LocationPushed) { Pop-Location -ErrorAction Stop }
    }
    catch {
        $LocationRestoreError = $_
        if ($null -eq $OriginalLaunchError) { throw }
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
}
```

The block saves and restores the operator's prior `$ErrorActionPreference`, makes setup errors terminating, checks that the exact reviewed `powershell.exe` path is a file, and pushes the code directory only inside the protected `try`. During the native call only, it uses `Continue` so a PowerShell 5.1 error-stream record cannot interrupt exit-status capture; it immediately captures `$global:LASTEXITCODE`, then restores `Stop` and fails by rethrowing the first newly observed invocation error record. It clears `$global:LASTEXITCODE` immediately before launch, so a launch error cannot inherit a stale status. A launch/setup error, observed invocation error record, null exit status, nonzero exit, or location-restoration error fails the block. The original launch/invocation error is retained and rethrown. A location-restoration error is retained separately; it is thrown if there is no earlier failure, and does not mask an earlier launch failure. The prior error preference is restored on every path. Never infer exit 0. These behaviors follow the Windows PowerShell 5.1 documentation for [`$LASTEXITCODE`](https://learn.microsoft.com/en-us/PowerShell/module/microsoft.powershell.core/about/about_automatic_variables?view=powershell-5.1), [native-program error handling](https://learn.microsoft.com/en-us/PowerShell/module/microsoft.powershell.core/about/about_error_handling?view=powershell-5.1), and [`$ErrorActionPreference`](https://learn.microsoft.com/en-us/PowerShell/module/microsoft.powershell.core/about/about_preference_variables?view=powershell-5.1). Preserve the console transcript and all runner-created evidence using the separately reviewed run procedure. Do not add redirection, execution-policy overrides, retries, cleanup, or manual process termination.

## Supervision, records, and stop rules

- Approved caps are one child/job, 1,800 seconds supervised stop deadline, at least 4 GiB available physical memory, at least 2 GiB free on the output volume, and one-second resource sampling. They are not evidence that runtime/resource sufficiency has been measured.
- Shutdown is procedural and best-effort. Bounded capture draining and process-exit polling do not guarantee that synchronous OS operations finish by the deadline or that every process terminates. If the supervisor is unresponsive or process/capture/finalization state is unresolved, remain present, note elapsed time and observed process IDs, preserve evidence, and request a separate decision. Do not stop processes yourself absent separate authorization.
- The child receipt records computation-stage evidence only. The supervisor completion record separately records observed child exit, capture, resource/process state, and finalization. Later review binds both immutable records. No missing receipt may be fabricated, no partial package promoted, and no output/evidence overwritten or cleaned up.
- Overall PASS requires verified outputs and a valid present child receipt, normal supervisory completion, observed exit 0, completed capture, resolved process state, and no relevant finalization errors. Any failed/unknown gate is incomplete; retain all outputs and staging and stop. No automatic retry.

## Later final-test gates — not runnable from this card

After a completed calibration job, independent package review and separate owner acceptance are required before the calibrated candidate is treated as accepted. Final evaluation then requires a finalized and independently reviewed final configuration, a separately owner-approved configuration commit, independent verification of the clean evaluation revision and config hash, and an external pre-test lock approved by hash **before any final-row access**. The primary evaluation requires its own fresh approval. At most one verification replay may occur only under a distinct authorization and on the identical locked model, rows, configuration, and reporting rules. This card supplies no final-test command or identity. No calibration result, final-test access/evaluation, replay, threshold, CA simulation, or deployment is authorized here.

## Scope and evidence references

The protocol is `phase14a_calibration_final_test_protocol.v1`, Draft 4, SHA-256 `c7791fe57672541bbcde375b7858fe6b73cbdfc35eb41975730b49937abe6efc`. The accepted Phase 13 model remains the immutable uncalibrated input; calibration must not refit its RF trees. This card and the environment update are preparation artifacts only. A later independent review and separate explicit owner authorization are required before any operational step.
