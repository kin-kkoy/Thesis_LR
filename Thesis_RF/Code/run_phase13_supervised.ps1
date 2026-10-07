# Phase 13 operator runner. Dot-sourcing exposes the internal supervision function
# for synthetic tests; the executable entry point never accepts a child command.

function Write-Phase13JsonAtomic {
    param([string]$Path, [object]$Value)
    $temporary = "$Path.tmp-$PID"
    $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $Path -Force -ErrorAction Stop
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
    param([int[]]$KnownIds, [int]$RootId, [int]$VerifySeconds = 10)
    $ids = Get-Phase13ProcessTree -KnownIds $KnownIds
    if ($RootId -gt 0) {
        try { & taskkill.exe /PID $RootId /T /F 2>$null | Out-Null } catch { }
    }
    foreach ($id in ($ids | Sort-Object -Descending)) {
        try { Stop-Process -Id $id -Force -ErrorAction Stop } catch { }
    }
    $until = [DateTime]::UtcNow.AddSeconds($VerifySeconds)
    do {
        $remaining = @($ids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
        if ($remaining.Count -eq 0) { break }
        Start-Sleep -Milliseconds 100
    } while ([DateTime]::UtcNow -lt $until)
    return [ordered]@{ process_ids = $ids; remaining_process_ids = $remaining; process_exited = ($remaining.Count -eq 0) }
}

function Get-Phase13MemoryAndDisk {
    $os = Get-CimInstance Win32_OperatingSystem
    return [ordered]@{
        available_physical_memory_bytes = [int64]$os.FreePhysicalMemory * 1KB
        free_c_bytes = [int64](Get-PSDrive -Name C).Free
    }
}

function Test-Phase13CompletionReceipt {
    param([string]$Destination, [string]$RunId)
    $manifestPath = Join-Path $Destination 'manifest.json'
    $completionPath = Join-Path $Destination 'completion.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $completionPath -PathType Leaf)) { return $null }
    try {
        $manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
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
    $receiptPath = Join-Path $EvidenceDirectory 'operator_completion.json'
    $samples = [System.Collections.Generic.List[object]]::new()
    $package = $null
    $treeState = [ordered]@{ process_ids = @(); remaining_process_ids = @(); process_exited = $true }
    $processStartedUtc = $null

    if (Test-Path -LiteralPath $Destination) { throw "Refusing existing output destination: $Destination" }
    if (Test-Path -LiteralPath $EvidenceDirectory) { throw "Refusing existing evidence/staging path: $EvidenceDirectory" }
    $destinationParent = Split-Path -Parent $Destination
    if (-not (Test-Path -LiteralPath $destinationParent -PathType Container)) { throw "Output parent does not exist: $destinationParent" }
    $staging = @(Get-ChildItem -LiteralPath $destinationParent -Directory -Force -Filter '.set_c_rf_v1.staging-*' -ErrorAction Stop)
    if ($staging.Count -gt 0) { throw "Refusing interrupted staging directory: $($staging[0].FullName)" }
    New-Item -ItemType Directory -Path $EvidenceDirectory -ErrorAction Stop | Out-Null

    try {
        # Always use hidden process windows. This is part of the tested implementation.
        $argumentLine = [string]::Join(' ', @($ChildArguments | ForEach-Object {
            '"' + ([string]$_).Replace('"', '\"') + '"'
        }))
        $process = Start-Process -FilePath $Executable -ArgumentList $argumentLine `
            -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        $knownIds = @([int]$process.Id)
        $processStartedUtc = & $Clock
        $pidRecord = [ordered]@{
            schema_version = 'phase13_set_c_rf_process.v1'; run_id = $RunId
            process_id = [int]$process.Id; process_ids = $knownIds
            process_started_at_utc = $processStartedUtc.ToString('o')
            executable = $Executable; working_directory = $WorkingDirectory
        }
        Write-Phase13JsonAtomic -Path $pidPath -Value $pidRecord
        $stopReason = 'completed'
        $lastSampleUtc = [DateTime]::MinValue
        $naturalTreeDone = $false

        while (-not $naturalTreeDone) {
            $now = & $Clock
            $elapsed = ($now - $processStartedUtc).TotalSeconds
            $knownIds = Get-Phase13ProcessTree -KnownIds $knownIds
            $process.Refresh()
            if ($process.HasExited -and $null -eq $exitStatus) { $exitStatus = [int]$process.ExitCode }
            $alive = @($knownIds | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
            $stopNow = $false
            if ($elapsed -ge $OverallLimitSeconds) {
                $stopReason = 'overall time limit reached'; $stopNow = $true
            }
            if ($stopNow -or ($now - $lastSampleUtc).TotalSeconds -ge $SampleIntervalSeconds) {
                $resource = & $ResourceProvider
                $sample = [ordered]@{
                    timestamp_utc = $now.ToString('o'); elapsed_seconds = [Math]::Round($elapsed, 3)
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
                Write-Phase13JsonAtomic -Path $pidPath -Value $pidRecord
                $treeState = Stop-Phase13ProcessTree -KnownIds $knownIds -RootId ([int]$process.Id)
                $process.Refresh()
                if ($process.HasExited) { $exitStatus = [int]$process.ExitCode }
                break
            }
            if ($alive.Count -eq 0) { $naturalTreeDone = $true; break }
            Start-Sleep -Milliseconds 100
        }
        if ($null -eq $exitStatus) {
            $process.Refresh()
            if ($process.HasExited) { $exitStatus = [int]$process.ExitCode }
        }
        if ($treeState.process_ids.Count -eq 0) {
            $treeState = [ordered]@{ process_ids = $knownIds; remaining_process_ids = @(); process_exited = $true }
        }
        if ($treeState.process_exited -and $exitStatus -eq 0) {
            $package = Test-Phase13CompletionReceipt -Destination $Destination -RunId $RunId
        }
        if ($stopReason -eq 'completed') {
            if ($exitStatus -ne 0) { $stopReason = 'child exited nonzero' }
            elseif (-not $package) { $stopReason = 'missing or invalid completion receipt/manifest hash' }
            elseif (-not $treeState.process_exited) { $stopReason = 'process tree remains active' }
        }
    }
    catch {
        $originalRunnerError = $_.Exception.ToString()
        $stopReason = "runner error: $($_.Exception.Message)"
        if ($process) {
            try {
                $knownIds = Get-Phase13ProcessTree -KnownIds (@($knownIds) + @([int]$process.Id))
                $treeState = Stop-Phase13ProcessTree -KnownIds $knownIds -RootId ([int]$process.Id)
                $process.Refresh()
                if ($process.HasExited) { $exitStatus = [int]$process.ExitCode }
            }
            catch { $cleanupError = $_.Exception.ToString() }
        }
        $runnerError = $originalRunnerError
    }

    $stdoutHash = if (Test-Path -LiteralPath $stdoutPath -PathType Leaf) { (Get-FileHash -LiteralPath $stdoutPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
    $stderrHash = if (Test-Path -LiteralPath $stderrPath -PathType Leaf) { (Get-FileHash -LiteralPath $stderrPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
    $resourceHash = if (Test-Path -LiteralPath $resourcePath -PathType Leaf) { (Get-FileHash -LiteralPath $resourcePath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
    $completion = [ordered]@{
        schema_version = 'phase13_set_c_rf_operator_completion.v1'; run_id = $RunId
        started_at_utc = $startUtc.ToString('o'); process_started_at_utc = if ($processStartedUtc) { $processStartedUtc.ToString('o') } else { $null }
        ended_at_utc = (& $Clock).ToString('o'); process_id = if ($process) { [int]$process.Id } else { $null }
        process_ids = $knownIds; process_id_record = if (Test-Path $pidPath) { $pidPath } else { $null }
        process_exited = [bool]$treeState.process_exited; remaining_process_ids = $treeState.remaining_process_ids
        actual_exit_status = $exitStatus; stop_reason = $stopReason; resource_samples = @($samples.ToArray())
        package_verified = [bool]$package; package_manifest_sha256 = if ($package) { $package.manifest_sha256 } else { $null }
        stdout_path = $stdoutPath; stdout_sha256 = $stdoutHash; stderr_path = $stderrPath; stderr_sha256 = $stderrHash
        operator_resource_log_path = $resourcePath; operator_resource_log_sha256 = $resourceHash
        runner_error = if ($runnerError) { $runnerError } else { $null }
        cleanup_error = if ($cleanupError) { $cleanupError } else { $null }
    }
    try { Write-Phase13JsonAtomic -Path $receiptPath -Value $completion }
    catch { throw "Unable to persist operator completion evidence at ${receiptPath}: $($_.Exception.Message)" }
    if ($runnerError) { throw $runnerError }
    if ($stopReason -ne 'completed' -or $exitStatus -ne 0 -or -not $package -or -not $treeState.process_exited) {
        throw "Phase 13 run incomplete: exit=$exitStatus; reason=$stopReason; evidence=$receiptPath"
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
    if ((Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash.ToLowerInvariant() -ne $configHash -or
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
