# Phase 13 operator runner. Dot-sourcing exposes the internal supervision function
# for synthetic tests; the executable entry point never accepts a child command.

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
    $sha = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    }
    finally { $stream.Dispose(); $sha.Dispose() }
}

function Get-Phase13ProcessTree {
    param([int[]]$KnownIds)
    $ids = [System.Collections.Generic.List[int]]::new()
    foreach ($id in $KnownIds) { if (-not $ids.Contains($id)) { $ids.Add($id) } }
    do {
        $newIds = @(Get-CimInstance Win32_Process | Where-Object {
            $ids.Contains([int]$_.ParentProcessId) -and -not $ids.Contains([int]$_.ProcessId)
        } | ForEach-Object { [int]$_.ProcessId })
        foreach ($id in $newIds) { if (-not $ids.Contains($id)) { $ids.Add($id) } }
    } while ($newIds.Count -gt 0)
    return ,@($ids.ToArray())
}

function Stop-Phase13ProcessTree {
    param([int[]]$KnownIds, [Parameter(Mandatory)][DateTime]$Deadline)
    $ids = @($KnownIds | Where-Object { $_ -gt 0 } | Select-Object -Unique)
    foreach ($id in ($ids | Sort-Object -Descending)) {
        if ([DateTime]::UtcNow -ge $Deadline) { break }
        try { Stop-Process -Id $id -Force -ErrorAction Stop } catch { }
    }
    $remaining = @($ids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue } | ForEach-Object { [int]$_.Id })
    do {
        if ($remaining.Count -eq 0) { break }
        $remainingMs = [Math]::Floor(($Deadline - [DateTime]::UtcNow).TotalMilliseconds)
        if ($remainingMs -le 0) { break }
        Start-Sleep -Milliseconds ([int][Math]::Min(100, $remainingMs))
        $remaining = @($ids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue } | ForEach-Object { [int]$_.Id })
    } while ([DateTime]::UtcNow -lt $Deadline)
    return [ordered]@{ process_ids = $ids; remaining_process_ids = $remaining; process_exited = ($remaining.Count -eq 0) }
}

function Get-Phase13MemoryAndDisk {
    $os = Get-CimInstance Win32_OperatingSystem
    return [ordered]@{
        available_physical_memory_bytes = [int64]$os.FreePhysicalMemory * 1KB
        free_c_bytes = [int64](Get-PSDrive -Name C).Free
    }
}

function Get-Phase13ObservedExitStatus {
    param([System.Diagnostics.Process]$Process)
    if (-not $Process) { return $null }
    try {
        $Process.Refresh()
        if (-not $Process.HasExited) { return $null }
        $code = $Process.ExitCode
        if ($null -eq $code) { return $null }
        return [int]$code
    }
    catch { return $null }
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
    $process = $null
    $knownIds = @()
    $exitStatus = $null
    $stopReason = 'runner initialization failed'
    $startUtc = & $Clock
    $stdoutPath = Join-Path $EvidenceDirectory 'stdout.log'
    $stderrPath = Join-Path $EvidenceDirectory 'stderr.log'
    $resourcePath = Join-Path $EvidenceDirectory 'operator_resources.jsonl'
    $pidPath = Join-Path $EvidenceDirectory 'process.json'
    $handoffPath = Join-Path $EvidenceDirectory 'supervisor_handoff.json'
    $receiptPath = Join-Path $EvidenceDirectory 'operator_completion.json'
    $samples = [System.Collections.Generic.List[object]]::new()
    $package = $null
    $treeState = [ordered]@{ process_ids = @(); remaining_process_ids = @(); process_exited = $true }
    $processStartedUtc = $null
    $processStarted = $false
    $stdoutStream = $null
    $stderrStream = $null
    $stdoutCopyTask = $null
    $stderrCopyTask = $null
    $captureComplete = $false
    $captureStatus = 'not_started'
    $captureError = $null
    $captureDrainTimedOut = $false
    $captureDrainTimeoutMilliseconds = 0
    $finalizationError = $null
    $receiptWriteError = $null
    $teardownDeadline = $null

    if (Test-Path -LiteralPath $Destination) { throw "Refusing existing output destination: $Destination" }
    if (Test-Path -LiteralPath $EvidenceDirectory) { throw "Refusing existing evidence/staging path: $EvidenceDirectory" }
    $destinationParent = Split-Path -Parent $Destination
    if (-not (Test-Path -LiteralPath $destinationParent -PathType Container)) { throw "Output parent does not exist: $destinationParent" }
    $staging = @(Get-ChildItem -LiteralPath $destinationParent -Directory -Force -Filter '.set_c_rf_v1.staging-*' -ErrorAction Stop)
    if ($staging.Count -gt 0) { throw "Refusing interrupted staging directory: $($staging[0].FullName)" }
    New-Item -ItemType Directory -Path $EvidenceDirectory -ErrorAction Stop | Out-Null

    try {
        $handoffNonce = [Guid]::NewGuid().ToString('N')
        Write-Phase13JsonAtomic -Path $handoffPath -Value ([ordered]@{
            schema_version = 'phase13_set_c_rf_supervisor_handoff.v1'
            run_id = $RunId
            nonce = $handoffNonce
            created_at_utc = (& $Clock).ToString('o')
            executable = $Executable
        })
        # CreateNoWindow keeps the native child hidden while preserving its process handle.
        $argumentLine = [string]::Join(' ', @($ChildArguments | ForEach-Object {
            '"' + ([string]$_).Replace('"', '\"') + '"'
        }))
        $startInfo = New-Object System.Diagnostics.ProcessStartInfo
        $startInfo.FileName = $Executable
        $startInfo.Arguments = $argumentLine
        $startInfo.WorkingDirectory = $WorkingDirectory
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        $startInfo.EnvironmentVariables['PHASE13_SUPERVISOR_NONCE'] = $handoffNonce
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $startInfo
        $stdoutStream = [IO.File]::Open($stdoutPath, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::Read)
        $stderrStream = [IO.File]::Open($stderrPath, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::Read)
        if (-not $process.Start()) { throw 'Child process could not be started.' }
        $processStarted = $true
        $stdoutCopyTask = $process.StandardOutput.BaseStream.CopyToAsync($stdoutStream)
        $stderrCopyTask = $process.StandardError.BaseStream.CopyToAsync($stderrStream)
        $knownIds = @([int]$process.Id)
        $processStartedUtc = & $Clock
        $pidRecord = [ordered]@{
            schema_version = 'phase13_set_c_rf_process.v1'; run_id = $RunId
            process_id = [int]$process.Id; process_ids = $knownIds
            process_started_at_utc = $processStartedUtc.ToString('o')
            executable = $Executable; working_directory = $WorkingDirectory
        }
        Write-Phase13JsonAtomic -Path $pidPath -Value $pidRecord -TemporaryDirectory $destinationParent
        $stopReason = 'completed'
        $lastSampleUtc = [DateTime]::MinValue
        $naturalTreeDone = $false

        while (-not $naturalTreeDone) {
            $now = & $Clock
            $elapsed = ($now - $processStartedUtc).TotalSeconds
            $knownIds = Get-Phase13ProcessTree -KnownIds $knownIds
            $observedExitStatus = Get-Phase13ObservedExitStatus -Process $process
            if ($null -ne $observedExitStatus) { $exitStatus = $observedExitStatus }
            $alive = @($knownIds | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
            $stopNow = $false
            if ($elapsed -ge $OverallLimitSeconds) {
                $stopReason = 'overall time limit reached'; $stopNow = $true
            }
            if ($stopNow -or ($now - $lastSampleUtc).TotalSeconds -ge $SampleIntervalSeconds) {
                $resource = & $ResourceProvider
                $sample = [ordered]@{
                    run_id = $RunId; timestamp_utc = $now.ToString('o'); elapsed_seconds = [Math]::Round($elapsed, 3)
                    available_physical_memory_bytes = [int64]$resource.available_physical_memory_bytes
                    free_c_bytes = [int64]$resource.free_c_bytes
                }
                $samples.Add($sample)
                $sample | ConvertTo-Json -Compress | Add-Content -LiteralPath $resourcePath -Encoding UTF8
                $lastSampleUtc = $now
                if ([int64]$sample.available_physical_memory_bytes -lt $MemoryFloorBytes -or
                    [int64]$sample.free_c_bytes -lt $DiskFloorBytes) {
                    $stopReason = 'resource floor breached'; $stopNow = $true
                }
            }
            if ($stopNow) {
                $pidRecord.process_ids = $knownIds
                $pidRecord.stop_reason = $stopReason
                $pidRecord.updated_at_utc = $now.ToString('o')
                Write-Phase13JsonAtomic -Path $pidPath -Value $pidRecord -TemporaryDirectory $destinationParent
                if (-not $teardownDeadline) { $teardownDeadline = [DateTime]::UtcNow.AddSeconds(10) }
                $treeState = Stop-Phase13ProcessTree -KnownIds $knownIds `
                    -Deadline $teardownDeadline
                $observedExitStatus = Get-Phase13ObservedExitStatus -Process $process
                if ($null -ne $observedExitStatus) { $exitStatus = $observedExitStatus }
                break
            }
            if ($alive.Count -eq 0) { $naturalTreeDone = $true; break }
            Start-Sleep -Milliseconds 100
        }
        if ($null -eq $exitStatus) {
            $observedExitStatus = Get-Phase13ObservedExitStatus -Process $process
            if ($null -ne $observedExitStatus) { $exitStatus = $observedExitStatus }
        }
        if ($treeState.process_ids.Count -eq 0) {
            $treeState = [ordered]@{ process_ids = $knownIds; remaining_process_ids = @(); process_exited = $true }
        }
        if ($treeState.process_exited -and $exitStatus -eq 0) {
            $package = Test-Phase13CompletionReceipt -Destination $Destination -RunId $RunId
        }
        if ($stopReason -eq 'completed') {
            if ($null -eq $exitStatus) { $stopReason = 'child exit status unobserved' }
            elseif ($exitStatus -ne 0) { $stopReason = 'child exited nonzero' }
            elseif (-not $package) { $stopReason = 'missing or invalid completion receipt/manifest hash' }
            elseif (-not $treeState.process_exited) { $stopReason = 'process tree remains active' }
        }
    }
    catch {
        $originalRunnerError = $_.Exception.ToString()
        $stopReason = "runner error: $($_.Exception.Message)"
        if ($process -and $processStarted) {
            try {
                $knownIds = Get-Phase13ProcessTree -KnownIds (@($knownIds) + @([int]$process.Id))
                if (-not $teardownDeadline) { $teardownDeadline = [DateTime]::UtcNow.AddSeconds(10) }
                $treeState = Stop-Phase13ProcessTree -KnownIds $knownIds `
                    -Deadline $teardownDeadline
                $observedExitStatus = Get-Phase13ObservedExitStatus -Process $process
                if ($null -ne $observedExitStatus) { $exitStatus = $observedExitStatus }
            }
            catch { $cleanupError = $_.Exception.ToString() }
        }
        $runnerError = $originalRunnerError
    }

    $copyTasks = @()
    if ($stdoutCopyTask) { $copyTasks += $stdoutCopyTask }
    if ($stderrCopyTask) { $copyTasks += $stderrCopyTask }
    if ($copyTasks.Count -eq 2) {
        if ($processStartedUtc) {
            $drainDeadline = $processStartedUtc.AddSeconds($OverallLimitSeconds)
            if ($teardownDeadline -and $teardownDeadline -lt $drainDeadline) {
                $drainDeadline = $teardownDeadline
            }
            $remainingMs = [Math]::Floor(($drainDeadline - (& $Clock)).TotalMilliseconds)
            $captureDrainTimeoutMilliseconds = [int][Math]::Max(0, [Math]::Min([int]::MaxValue, $remainingMs))
        }
        try {
            $waitCompleted = Wait-Phase13CaptureTasks -Tasks ([Threading.Tasks.Task[]]$copyTasks) `
                -TimeoutMilliseconds $captureDrainTimeoutMilliseconds
        }
        catch { $captureError = $_.Exception.ToString() }

        $stdoutTaskStatus = $stdoutCopyTask.Status.ToString()
        $stderrTaskStatus = $stderrCopyTask.Status.ToString()
        $taskFaults = @($copyTasks | Where-Object { $_.IsFaulted })
        $taskCancellations = @($copyTasks | Where-Object { $_.IsCanceled })
        $tasksComplete = @($copyTasks | Where-Object { -not $_.IsCompleted }).Count -eq 0
        if (-not $captureError -and $taskFaults.Count -gt 0) {
            $captureError = [string]::Join("`n", @($taskFaults | ForEach-Object { $_.Exception.Flatten().ToString() }))
        }
        if ($taskFaults.Count -gt 0) {
            $captureStatus = 'faulted'
        }
        elseif ($taskCancellations.Count -gt 0) {
            $captureStatus = 'canceled'
            if (-not $captureError) { $captureError = 'At least one stream-copy task was canceled.' }
        }
        elseif ($captureError) {
            $captureStatus = 'failed'
        }
        elseif (-not $waitCompleted -or -not $tasksComplete) {
            $captureStatus = 'drain_timed_out'
            $captureDrainTimedOut = $true
            if (-not $captureError) { $captureError = 'Stream-copy tasks did not complete within the remaining overall run budget.' }
        }
        else {
            $captureStatus = 'complete'
            $captureComplete = $true
        }
    }
    else { $captureError = 'Both stdout and stderr copy tasks were not started.' }

    if (-not $captureComplete -and $processStarted) {
        try {
            $knownIds = Get-Phase13ProcessTree -KnownIds (@($knownIds) + @([int]$process.Id))
            if (-not $teardownDeadline) { $teardownDeadline = [DateTime]::UtcNow.AddSeconds(10) }
            $treeState = Stop-Phase13ProcessTree -KnownIds $knownIds `
                -Deadline $teardownDeadline
        }
        catch {
            if (-not $cleanupError) { $cleanupError = $_.Exception.ToString() }
            $remainingIds = @($knownIds | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue } | ForEach-Object { [int]$_ })
            $treeState = [ordered]@{
                process_ids = $knownIds; remaining_process_ids = $remainingIds
                process_exited = ($remainingIds.Count -eq 0)
            }
        }
    }
    if ($stdoutStream) {
        try { $stdoutStream.Dispose() }
        catch {
            if (-not $finalizationError) { $finalizationError = $_.Exception.ToString() }
            $captureComplete = $false; $captureStatus = 'failed'
        }
    }
    if ($stderrStream) {
        try { $stderrStream.Dispose() }
        catch {
            if (-not $finalizationError) { $finalizationError = $_.Exception.ToString() }
            $captureComplete = $false; $captureStatus = 'failed'
        }
    }

    $stdoutHash = $null
    $stderrHash = $null
    if ($captureComplete) {
        try {
            $stdoutHash = Get-Phase13FileSha256 -Path $stdoutPath
            $stderrHash = Get-Phase13FileSha256 -Path $stderrPath
        }
        catch {
            $stdoutHash = $null; $stderrHash = $null
            if (-not $finalizationError) { $finalizationError = $_.Exception.ToString() }
        }
    }
    $resourceHash = $null
    try {
        if (Test-Path -LiteralPath $resourcePath -PathType Leaf) { $resourceHash = Get-Phase13FileSha256 -Path $resourcePath }
    }
    catch { if (-not $finalizationError) { $finalizationError = $_.Exception.ToString() } }
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
