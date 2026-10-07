# Phase 13 RF manual run card

This card is an operator procedure for the accepted Phase 13 Set C development run. It does not grant execution approval. Proceed only after approval binds the clean reviewed source commit, configuration and protocol identities, approval reference, and fresh run ID.

The Python command reads the pinned development train and validation roles, runs 60 train-only candidates, selects by full-precision validation average precision, and creates a separate uncalibrated train-plus-validation refit. It does not read calibration contents or run calibration, threshold selection, final-test evaluation, or CA simulation.

## Set the approved identities

Open a fresh PowerShell window. Replace only the three marked approval values. Keep the paths and hashes fixed.

```powershell
$Repo = 'C:\Users\gibgib\Thesis-Cellular-Automata'
$Code = Join-Path $Repo 'Thesis_RF\Code'
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
$Config = Join-Path $Code 'config\phase13_set_c_rf_v1.yaml'
$ConfigSha256 = '3800fe006463392f102a85ca3c491a6386b5340aeb8d2211eb31ae079a63bfc8'
$ProtocolSha256 = '69c39d1d97430ebb903ec2d856e6205a2c1b8ad77641798955d6ad2bdb671637'
$SourceRevision = '<clean reviewed 40-character source commit>'
$ApprovalReference = '<separate execution approval reference>'
$RunId = '<fresh approved run id>'
$PackageVersion = 'set_c_rf_v1'
$OutputRoot = Join-Path $Code 'output\phase13'
$EvidenceDir = Join-Path $OutputRoot "evidence\$RunId"
$Destination = Join-Path $OutputRoot $PackageVersion

if ($SourceRevision -notmatch '^[0-9a-f]{40}$') { throw 'SourceRevision must be the exact reviewed commit.' }
if ([string]::IsNullOrWhiteSpace($ApprovalReference) -or $ApprovalReference.StartsWith('<')) { throw 'Supply the separate execution approval reference.' }
if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$') { throw 'RunId is not a safe approved identifier.' }
if ((Get-FileHash -LiteralPath $Config -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ConfigSha256) { throw 'Configuration hash mismatch.' }
if ($ProtocolSha256 -ne '69c39d1d97430ebb903ec2d856e6205a2c1b8ad77641798955d6ad2bdb671637') { throw 'Protocol hash mismatch.' }
if (Test-Path -LiteralPath $Destination) { throw 'Destination exists; never overwrite a package.' }
if (Test-Path -LiteralPath $EvidenceDir) { throw 'Evidence directory exists; use the approved fresh RunId.' }
```

## Run preflight

Preflight checks the clean source revision, package versions, pinned config and protocol, resource floors, output/staging paths, and accepted development release attestation and selected-role data. It does not fit a model or create a package.

```powershell
Push-Location $Code
try {
    & $Python -B train_rf_optuna.py `
        --preflight-only `
        --config $Config `
        --source-revision $SourceRevision `
        --expected-config-sha256 $ConfigSha256 `
        --evidence-dir $EvidenceDir
    if ($LASTEXITCODE -ne 0) { throw "Preflight failed with exit code $LASTEXITCODE. Stop." }
}
finally { Pop-Location }
```

If preflight fails, preserve its output and stop. Do not change the environment or paths to bypass a failed gate.

## Execute through the tested supervisor

Only run after separate execution approval. The native script accepts only source revision, approval reference, and run ID. It hard-wires the approved Python executable, RF entry point, config, hashes, package name, resource limits, and destination. Its internal child-process function is dot-sourceable for synthetic tests; the production command line does not accept an executable or arbitrary arguments.

```powershell
Push-Location $Code
try {
    & (Join-Path $Code 'run_phase13_supervised.ps1') `
        $SourceRevision $ApprovalReference $RunId
    if ($LASTEXITCODE -ne 0) { throw "Phase 13 runner failed with exit code $LASTEXITCODE." }
}
catch {
    Write-Error $_
    throw
}
finally { Pop-Location }
```

The supervisor writes `process.json` promptly after launch, captures stdout and stderr separately, and samples physical memory and free C: space every 60 seconds. It stops the process tree at the 7200-second overall limit, below 4 GiB available physical memory, or below 2 GiB free on C:. The Optuna 5400-second timeout is soft and does not terminate an active fit. The runner records the child exit status only after observing it; otherwise `actual_exit_status` remains null. It records the process IDs, resource samples, logs and hashes, stop reason, and package verification in `operator_completion.json`. It refuses an existing package, evidence directory, or interrupted `.set_c_rf_v1.staging-*` directory.

Synthetic tests exercise the supervisor's configured overall-time stop and a deterministic resource-floor breach with dummy process trees, including zero-exit children with valid synthetic completion receipts; they verify the stop reason, incomplete runner result, retained exit and receipt-integrity evidence, and process termination. They do not exercise Ctrl+C, closing the PowerShell window, or a resource breach on the production machine. Do not assume those operator actions trigger the tested cleanup path. For any interrupted or failed run, use the persisted `process.json` IDs to verify that every process has exited; retain all evidence and staging, and do not retry, clean, overwrite, or promote a partial result. A later attempt requires a fresh approved run ID and scope.

## Completion check

The runner reports success only when the child exits with status 0, the process tree is gone, `stop_reason` is `completed`, the completion receipt identifies this run and status, and its manifest SHA-256 matches the package manifest. A resource-floor stop, overall-time stop, or runner error is incomplete even when the child exits 0 and the package receipt is valid. The operator receipt retains the observed child exit status and receipt-integrity result as evidence; those fields do not override the stop reason. The output package contains both uncalibrated forests, trial and selection evidence, config, diagnostics, manifest, and Python completion receipt. The operator completion receipt and captured streams remain in the sibling evidence directory. The train-plus-validation refit has seen validation labels and has no held-out performance claim.
