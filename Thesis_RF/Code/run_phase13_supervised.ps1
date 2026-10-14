# Phase 13 operator runner. Dot-sourcing exposes the internal supervision function
# for synthetic tests; the executable entry point never accepts a child command.

Import-Module (Join-Path $PSScriptRoot 'modules\phase_supervision.psm1') -ErrorAction Stop

function Write-Phase13JsonAtomic {
    param([string]$Path, [object]$Value, [string]$TemporaryDirectory)
    $parent = if ($TemporaryDirectory) { $TemporaryDirectory } else { Split-Path -Parent $Path }
    $temporary = Join-Path $parent ('.' + [IO.Path]::GetFileName($Path) + '.tmp-' + $PID + '-' + [Guid]::NewGuid().ToString('N'))
    try {
        $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
        Move-Item -LiteralPath $temporary -Destination $Path -Force -ErrorAction Stop
    }
    finally { Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue }
}

function Get-Phase13FileSha256 {
    param([Parameter(Mandatory)][string]$Path)
    return Get-PhaseFileSha256 -Path $Path
}

function Get-Phase13ProcessTree {
    param([int[]]$KnownIds)
    return ,@(Get-PhaseProcessTree -KnownIds $KnownIds)
}

function Stop-Phase13ProcessTree {
    param([int[]]$KnownIds, [Parameter(Mandatory)][DateTime]$Deadline)
    return Stop-PhaseProcessTree -KnownIds $KnownIds -Deadline $Deadline
}

function Get-Phase13MemoryAndDisk {
    return Get-PhaseMemoryAndDisk
}

function Get-Phase13ObservedExitStatus {
    param([System.Diagnostics.Process]$Process)
    return Get-PhaseObservedExitStatus -Process $Process
}

function Wait-Phase13CaptureTasks {
    param([System.Threading.Tasks.Task[]]$Tasks, [int]$TimeoutMilliseconds)
    return [System.Threading.Tasks.Task]::WaitAll($Tasks, $TimeoutMilliseconds)
}

function Test-Phase13CompletionReceipt {
    param([string]$Destination, [string]$RunId)
    $manifestPath = Join-Path $Destination 'manifest.json'
    $completionPath = Join-Path $Destination 'completion.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $completionPath -PathType Leaf)) { return $null }
    try {
        $manifestHash = Get-Phase13FileSha256 -Path $manifestPath
        $manifest = Get-Content -LiteralPath $manifestPath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
        $receipt = Get-Content -LiteralPath $completionPath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
        if ($receipt.schema_version -ne 'phase13_set_c_rf_completion.v1' -or
            $receipt.status -ne 'complete' -or $receipt.actual_exit_status -ne 0 -or
            $receipt.manifest_sha256 -ne $manifestHash -or $receipt.run_id -ne $RunId -or
            $manifest.schema_version -ne 'set_c_rf_phase13_run.v1' -or
            $manifest.artifact_kind -ne 'set_c_rf_uncalibrated_candidate_run' -or
            $manifest.run_id -ne $RunId -or $manifest.completion_status -ne 'complete') { return $null }
        return [ordered]@{ verified = $true; manifest_sha256 = $manifestHash }
    }
    catch { return $null }
}

function Invoke-Phase13Supervision {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Executable,
        [Parameter(Mandatory)][string[]]$ChildArguments,
        [Parameter(Mandatory)][string]$WorkingDirectory,
        [Parameter(Mandatory)][string]$Destination,
        [Parameter(Mandatory)][string]$EvidenceDirectory,
        [Parameter(Mandatory)][string]$RunId,
        [int]$OverallLimitSeconds = 7200,
        [int]$SampleIntervalSeconds = 60,
        [int64]$MemoryFloorBytes = 4294967296,
        [int64]$DiskFloorBytes = 2147483648,
        [scriptblock]$ResourceProvider = { Get-Phase13MemoryAndDisk },
        [scriptblock]$Clock = { [DateTime]::UtcNow }
    )
    $startUtc = & $Clock
    $stdoutPath = Join-Path $EvidenceDirectory 'stdout.log'
    $stderrPath = Join-Path $EvidenceDirectory 'stderr.log'
    $resourcePath = Join-Path $EvidenceDirectory 'operator_resources.jsonl'
    $pidPath = Join-Path $EvidenceDirectory 'process.json'
    $handoffPath = Join-Path $EvidenceDirectory 'supervisor_handoff.json'
    $receiptPath = Join-Path $EvidenceDirectory 'operator_completion.json'
    $receiptWriteError = $null

    if (Test-Path -LiteralPath $Destination) { throw "Refusing existing output destination: $Destination" }
    if (Test-Path -LiteralPath $EvidenceDirectory) { throw "Refusing existing evidence/staging path: $EvidenceDirectory" }
    $destinationParent = Split-Path -Parent $Destination
    if (-not (Test-Path -LiteralPath $destinationParent -PathType Container)) { throw "Output parent does not exist: $destinationParent" }
    $staging = @(Get-ChildItem -LiteralPath $destinationParent -Directory -Force -Filter '.set_c_rf_v1.staging-*' -ErrorAction Stop)
    if ($staging.Count -gt 0) { throw "Refusing interrupted staging directory: $($staging[0].FullName)" }
    New-Item -ItemType Directory -Path $EvidenceDirectory -ErrorAction Stop | Out-Null


    $writeJson = (Get-Command Write-Phase13JsonAtomic -CommandType Function -ErrorAction Stop).ScriptBlock
    $verifyCompletion = (Get-Command Test-Phase13CompletionReceipt -CommandType Function -ErrorAction Stop).ScriptBlock
    $beforeStart = {
        $handoffNonce = [Guid]::NewGuid().ToString('N')
        & $writeJson -Path $handoffPath -Value ([ordered]@{
            schema_version = 'phase13_set_c_rf_supervisor_handoff.v1'
            run_id = $RunId
            nonce = $handoffNonce
            created_at_utc = (& $Clock).ToString('o')
            executable = $Executable
        })
        return @{ PHASE13_SUPERVISOR_NONCE = $handoffNonce }
    }.GetNewClosure()
    $recordProcess = {
        param($record)
        $phase13Record = [ordered]@{ schema_version = 'phase13_set_c_rf_process.v1' }
        foreach ($key in $record.Keys) { $phase13Record[$key] = $record[$key] }
        & $writeJson -Path $pidPath -Value $phase13Record -TemporaryDirectory $destinationParent
    }.GetNewClosure()
    $inspectExit = {
        param($exitStatus, $treeState, $stopReason)
        $package = $null
        if ($treeState.process_exited -and $exitStatus -eq 0) {
            $package = & $verifyCompletion -Destination $Destination -RunId $RunId
        }
        if ($stopReason -eq 'completed') {
            if ($null -eq $exitStatus) { $stopReason = 'child exit status unobserved' }
            elseif ($exitStatus -ne 0) { $stopReason = 'child exited nonzero' }
            elseif (-not $package) { $stopReason = 'missing or invalid completion receipt/manifest hash' }
            elseif (-not $treeState.process_exited) { $stopReason = 'process tree remains active' }
        }
        return @{ evidence = $package; stop_reason = $stopReason }
    }.GetNewClosure()
    # Resolve the current caller seam; a module-local name lookup would bypass test injections.
    $captureWait = (Get-Command Wait-Phase13CaptureTasks -CommandType Function -ErrorAction Stop).ScriptBlock
    $observed = Invoke-PhaseSupervisionCore -Executable $Executable -ChildArguments $ChildArguments `
        -WorkingDirectory $WorkingDirectory -EvidenceDirectory $EvidenceDirectory -RunId $RunId `
        -ResourceLogPath $resourcePath -OverallLimitSeconds $OverallLimitSeconds `
        -SampleIntervalSeconds $SampleIntervalSeconds -MemoryFloorBytes $MemoryFloorBytes `
        -DiskFloorBytes $DiskFloorBytes -ResourceProvider $ResourceProvider -Clock $Clock `
        -CaptureWait $captureWait -BeforeStart $beforeStart -RecordProcess $recordProcess -InspectExit $inspectExit
    $process = $observed.process
    $knownIds = $observed.knownIds
    $exitStatus = $observed.exitStatus
    $stopReason = $observed.stopReason
    $samples = $observed.samples
    $treeState = $observed.treeState
    $processStartedUtc = $observed.processStartedUtc
    $processStarted = $observed.processStarted
    $stdoutCopyTask = $observed.stdoutCopyTask
    $stderrCopyTask = $observed.stderrCopyTask
    $captureComplete = $observed.captureComplete
    $captureStatus = $observed.captureStatus
    $captureError = $observed.captureError
    $captureDrainTimedOut = $observed.captureDrainTimedOut
    $captureDrainTimeoutMilliseconds = $observed.captureDrainTimeoutMilliseconds
    $stdoutHash = $observed.stdoutHash
    $stderrHash = $observed.stderrHash
    $resourceHash = $observed.resourceHash
    $runnerError = $observed.runnerError
    $cleanupError = $observed.cleanupError
    $finalizationError = $observed.finalizationError
    $package = $observed.completionEvidence
    $handoffHash = $null
    try { if (Test-Path $handoffPath) { $handoffHash = Get-Phase13FileSha256 -Path $handoffPath } }
    catch { if (-not $finalizationError) { $finalizationError = $_.Exception.ToString() } }
    if ($stopReason -eq 'completed' -and -not $captureComplete) {
        $stopReason = if ($captureDrainTimedOut) { 'stream capture drain timed out' } else { 'stream capture failed' }
    }
    $completionConstructionError = $null
    try {
        $completion = [ordered]@{
        schema_version = 'phase13_set_c_rf_operator_completion.v1'; run_id = $RunId
        started_at_utc = $startUtc.ToString('o'); process_started_at_utc = if ($processStartedUtc) { $processStartedUtc.ToString('o') } else { $null }
        ended_at_utc = (& $Clock).ToString('o'); process_id = if ($processStarted) { [int]$process.Id } else { $null }
        process_ids = $knownIds; process_id_record = if (Test-Path $pidPath) { $pidPath } else { $null }
        supervisor_handoff_path = if (Test-Path $handoffPath) { $handoffPath } else { $null }
        supervisor_handoff_sha256 = $handoffHash
        process_exited = [bool]$treeState.process_exited; remaining_process_ids = $treeState.remaining_process_ids
        actual_exit_status = $exitStatus; stop_reason = $stopReason; resource_samples = @($samples.ToArray())
        package_verified = [bool]$package; package_manifest_sha256 = if ($package) { $package.manifest_sha256 } else { $null }
        capture_complete = [bool]$captureComplete; capture_status = $captureStatus
        capture_drain_timeout_ms = $captureDrainTimeoutMilliseconds
        stdout_copy_task_status = if ($stdoutCopyTask) { $stdoutCopyTask.Status.ToString() } else { 'not_started' }
        stderr_copy_task_status = if ($stderrCopyTask) { $stderrCopyTask.Status.ToString() } else { 'not_started' }
        capture_error = if ($captureError) { $captureError } else { $null }
        stdout_path = $stdoutPath; stdout_sha256 = $stdoutHash; stderr_path = $stderrPath; stderr_sha256 = $stderrHash
        operator_resource_log_path = $resourcePath; operator_resource_log_sha256 = $resourceHash
        runner_error = if ($runnerError) { $runnerError } else { $null }
        cleanup_error = if ($cleanupError) { $cleanupError } else { $null }
        finalization_error = if ($finalizationError) { $finalizationError } else { $null }
        }
    }
    catch {
        $completionConstructionError = $_.Exception.ToString()
        if ($finalizationError) { $finalizationError += "`nCompletion record construction failed: $completionConstructionError" }
        else { $finalizationError = "Completion record construction failed: $completionConstructionError" }
        $completion = [ordered]@{
            schema_version = 'phase13_set_c_rf_operator_completion.v1'; run_id = $RunId
            actual_exit_status = $exitStatus; stop_reason = $stopReason
            capture_complete = $false; capture_status = $captureStatus
            capture_error = $captureError; runner_error = $runnerError
            cleanup_error = $cleanupError; finalization_error = $finalizationError
            completion_record_construction_error = $completionConstructionError
        }
    }
    try { Write-Phase13JsonAtomic -Path $receiptPath -Value $completion }
    catch { $receiptWriteError = $_.Exception.ToString() }
    if ($runnerError) {
        if ($receiptWriteError) { throw "$runnerError`nOperator completion evidence write also failed: $receiptWriteError; finalization_error=$finalizationError" }
        throw $runnerError
    }
    if ($receiptWriteError) {
        throw "Unable to persist operator completion evidence at ${receiptPath}: $receiptWriteError; primary_status=$stopReason; actual_exit_status=$exitStatus; capture_error=$captureError; finalization_error=$finalizationError"
    }
    if ($stopReason -ne 'completed' -or $null -eq $exitStatus -or $exitStatus -ne 0 -or
        -not $package -or -not $treeState.process_exited -or -not $captureComplete -or
        $cleanupError -or $finalizationError) {
        throw "Phase 13 run incomplete: exit=$exitStatus; reason=$stopReason; capture=$captureStatus; evidence=$receiptPath"
    }
    return $completion
}

function Invoke-Phase13Production {
    param([string]$SourceRevision, [string]$ApprovalReference, [string]$RunId)
    $repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
    $code = Join-Path $repo 'Thesis_RF\Code'
    $python = Join-Path $repo '.venv\Scripts\python.exe'
    $config = Join-Path $code 'config\phase13_set_c_rf_v1.yaml'
    $configHash = '3800fe006463392f102a85ca3c491a6386b5340aeb8d2211eb31ae079a63bfc8'
    $protocolHash = '69c39d1d97430ebb903ec2d856e6205a2c1b8ad77641798955d6ad2bdb671637'
    $packageVersion = 'set_c_rf_v1'
    if ($SourceRevision -notmatch '^[0-9a-f]{40}$' -or [string]::IsNullOrWhiteSpace($ApprovalReference) -or
        $RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$') { throw 'Source revision, approval reference, and safe run ID are required.' }
    if ((Get-Phase13FileSha256 -Path $config) -ne $configHash -or
        $protocolHash -ne '69c39d1d97430ebb903ec2d856e6205a2c1b8ad77641798955d6ad2bdb671637') { throw 'Pinned Phase 13 identity mismatch.' }
    $outputRoot = Join-Path $code 'output\phase13'
    $destination = Join-Path $outputRoot $packageVersion
    $evidence = Join-Path $outputRoot "evidence\$RunId"
    if (Test-Path $destination) { throw "Refusing existing output destination: $destination" }
    if (Test-Path $evidence) { throw "Refusing existing evidence/staging path: $evidence" }
    New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
    $args = @('-B', 'train_rf_optuna.py', '--execute', '--config', $config,
        '--source-revision', $SourceRevision, '--expected-config-sha256', $configHash,
        '--approval-reference', $ApprovalReference, '--run-id', $RunId,
        '--package-version', $packageVersion, '--evidence-dir', $evidence)
    return Invoke-Phase13Supervision -Executable $python -ChildArguments $args -WorkingDirectory $code `
        -Destination $destination -EvidenceDirectory $evidence -RunId $RunId `
        -OverallLimitSeconds 7200 -SampleIntervalSeconds 60 -MemoryFloorBytes 4294967296 -DiskFloorBytes 2147483648
}

if ($MyInvocation.InvocationName -ne '.') {
    if ($args.Count -ne 3) { throw 'Usage: run_phase13_supervised.ps1 <source-revision> <approval-reference> <run-id>' }
    Invoke-Phase13Production -SourceRevision $args[0] -ApprovalReference $args[1] -RunId $args[2]
}
