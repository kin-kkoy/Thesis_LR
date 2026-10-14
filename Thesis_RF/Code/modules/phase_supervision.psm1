# Shared observed process/capture mechanics; callers retain publication and completion policy.

function Get-PhaseFileSha256 {
    param([Parameter(Mandatory)][string]$Path)
    $sha = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    }
    finally { $stream.Dispose(); $sha.Dispose() }
}

function Get-PhaseProcessTree {
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

function Stop-PhaseProcessTree {
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

function Get-PhaseMemoryAndDisk {
    $os = Get-CimInstance Win32_OperatingSystem
    return [ordered]@{
        available_physical_memory_bytes = [int64]$os.FreePhysicalMemory * 1KB
        free_c_bytes = [int64](Get-PSDrive -Name C).Free
    }
}

function Get-PhaseObservedExitStatus {
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

function Invoke-PhaseSupervisionCore {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Executable,
        [Parameter(Mandatory)][string[]]$ChildArguments,
        [Parameter(Mandatory)][string]$WorkingDirectory,
        [Parameter(Mandatory)][string]$EvidenceDirectory,
        [Parameter(Mandatory)][string]$RunId,
        [Parameter(Mandatory)][string]$ResourceLogPath,
        [Parameter(Mandatory)][int]$OverallLimitSeconds,
        [Parameter(Mandatory)][int]$SampleIntervalSeconds,
        [Parameter(Mandatory)][int64]$MemoryFloorBytes,
        [Parameter(Mandatory)][int64]$DiskFloorBytes,
        [Parameter(Mandatory)][scriptblock]$ResourceProvider,
        [Parameter(Mandatory)][scriptblock]$Clock,
        [Parameter(Mandatory)][scriptblock]$CaptureWait,
        [Parameter(Mandatory)][scriptblock]$BeforeStart,
        [Parameter(Mandatory)][scriptblock]$RecordProcess,
        [Parameter(Mandatory)][scriptblock]$InspectExit
    )
    $process = $null
    $knownIds = @()
    $exitStatus = $null
    $stopReason = 'runner initialization failed'
    $stdoutPath = Join-Path $EvidenceDirectory 'stdout.log'
    $stderrPath = Join-Path $EvidenceDirectory 'stderr.log'
    $resourcePath = $ResourceLogPath
    $samples = [System.Collections.Generic.List[object]]::new()
    $completionEvidence = $null
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
    $teardownDeadline = $null

    try {
        $childEnvironment = & $BeforeStart
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
        foreach ($key in $childEnvironment.Keys) { $startInfo.EnvironmentVariables[$key] = $childEnvironment[$key] }
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
            run_id = $RunId
            process_id = [int]$process.Id; process_ids = $knownIds
            process_started_at_utc = $processStartedUtc.ToString('o')
            executable = $Executable; working_directory = $WorkingDirectory
        }
        & $RecordProcess $pidRecord
        $stopReason = 'completed'
        $lastSampleUtc = [DateTime]::MinValue
        $naturalTreeDone = $false

        while (-not $naturalTreeDone) {
            $now = & $Clock
            $elapsed = ($now - $processStartedUtc).TotalSeconds
            $knownIds = Get-PhaseProcessTree -KnownIds $knownIds
            $observedExitStatus = Get-PhaseObservedExitStatus -Process $process
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
                & $RecordProcess $pidRecord
                if (-not $teardownDeadline) { $teardownDeadline = [DateTime]::UtcNow.AddSeconds(10) }
                $treeState = Stop-PhaseProcessTree -KnownIds $knownIds `
                    -Deadline $teardownDeadline
                $observedExitStatus = Get-PhaseObservedExitStatus -Process $process
                if ($null -ne $observedExitStatus) { $exitStatus = $observedExitStatus }
                break
            }
            if ($alive.Count -eq 0) { $naturalTreeDone = $true; break }
            Start-Sleep -Milliseconds 100
        }
        if ($null -eq $exitStatus) {
            $observedExitStatus = Get-PhaseObservedExitStatus -Process $process
            if ($null -ne $observedExitStatus) { $exitStatus = $observedExitStatus }
        }
        if ($treeState.process_ids.Count -eq 0) {
            $treeState = [ordered]@{ process_ids = $knownIds; remaining_process_ids = @(); process_exited = $true }
        }
        $decision = & $InspectExit $exitStatus $treeState $stopReason
        $completionEvidence = $decision.evidence
        $stopReason = $decision.stop_reason
    }
    catch {
        $originalRunnerError = $_.Exception.ToString()
        $stopReason = "runner error: $($_.Exception.Message)"
        if ($process -and $processStarted) {
            try {
                $knownIds = Get-PhaseProcessTree -KnownIds (@($knownIds) + @([int]$process.Id))
                if (-not $teardownDeadline) { $teardownDeadline = [DateTime]::UtcNow.AddSeconds(10) }
                $treeState = Stop-PhaseProcessTree -KnownIds $knownIds `
                    -Deadline $teardownDeadline
                $observedExitStatus = Get-PhaseObservedExitStatus -Process $process
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
            $waitCompleted = & $CaptureWait -Tasks ([Threading.Tasks.Task[]]$copyTasks) `
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
            $knownIds = Get-PhaseProcessTree -KnownIds (@($knownIds) + @([int]$process.Id))
            if (-not $teardownDeadline) { $teardownDeadline = [DateTime]::UtcNow.AddSeconds(10) }
            $treeState = Stop-PhaseProcessTree -KnownIds $knownIds `
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
            $stdoutHash = Get-PhaseFileSha256 -Path $stdoutPath
            $stderrHash = Get-PhaseFileSha256 -Path $stderrPath
        }
        catch {
            $stdoutHash = $null; $stderrHash = $null
            if (-not $finalizationError) { $finalizationError = $_.Exception.ToString() }
        }
    }
    $resourceHash = $null
    try {
        if (Test-Path -LiteralPath $resourcePath -PathType Leaf) { $resourceHash = Get-PhaseFileSha256 -Path $resourcePath }
    }
    catch { if (-not $finalizationError) { $finalizationError = $_.Exception.ToString() } }
    return [ordered]@{
        process = $process
        knownIds = $knownIds
        exitStatus = $exitStatus
        stopReason = $stopReason
        samples = $samples
        completionEvidence = $completionEvidence
        treeState = $treeState
        processStartedUtc = $processStartedUtc
        processStarted = $processStarted
        stdoutCopyTask = $stdoutCopyTask
        stderrCopyTask = $stderrCopyTask
        captureComplete = $captureComplete
        captureStatus = $captureStatus
        captureError = $captureError
        captureDrainTimedOut = $captureDrainTimedOut
        captureDrainTimeoutMilliseconds = $captureDrainTimeoutMilliseconds
        stdoutHash = $stdoutHash
        stderrHash = $stderrHash
        resourceHash = $resourceHash
        runnerError = $runnerError
        cleanupError = $cleanupError
        finalizationError = $finalizationError
    }
}

Export-ModuleMember -Function Invoke-PhaseSupervisionCore, Get-PhaseFileSha256, Get-PhaseProcessTree, Stop-PhaseProcessTree, Get-PhaseMemoryAndDisk, Get-PhaseObservedExitStatus

