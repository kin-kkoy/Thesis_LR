# Fixed Phase 14 jobs only; dot-sourcing defines interfaces without execution.
[CmdletBinding()]
param([string]$Job, [string]$ApprovalReference, [string]$ExpectedRevision, [string]$RunId)
Import-Module (Join-Path $PSScriptRoot 'modules\phase_supervision.psm1') -ErrorAction Stop
$script:Phase14ProtocolHash = 'c7791fe57672541bbcde375b7858fe6b73cbdfc35eb41975730b49937abe6efc'
$script:Phase14Sources = @(
    'Thesis_RF/Code/run_phase14_supervised.ps1','Thesis_RF/Code/evaluate_rf_final_test.py',
    'Thesis_RF/Code/modules/__init__.py','Thesis_RF/Code/modules/phase_supervision.psm1',
    'Thesis_RF/Code/modules/phase14_calibration.py','Thesis_RF/Code/modules/feature_pipeline.py',
    'Thesis_RF/Code/modules/wind_convention.py','Thesis_RF/Code/modules/ca_state.py',
    'Thesis_RF/Code/modules/data_loader.py','Thesis_RF/Code/modules/automata_engine.py',
    'Thesis_RF/Code/modules/set_c_split_release.py','Thesis_RF/Code/modules/set_c_leakage.py',
    'Thesis_RF/Code/modules/set_c_publication.py','Thesis_RF/Code/modules/set_c_collector.py',
    'Thesis_RF/Code/modules/transition_observer.py','Thesis_RF/Code/config/phase13_set_c_rf_v1.yaml',
    'Thesis_RF/Code/config/phase14_setc_calibration_v1.yaml','Thesis_RF/Code/config/phase14_setc_final_test_v1.yaml',
    'docs/Phase14A_Calibration_Final_Test_Protocol_Proposal.md'
)

function Get-Phase14TextHash {
    param([string]$Text)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash([Text.UTF8Encoding]::new($false,$true).GetBytes($Text)))).Replace('-','').ToLowerInvariant() }
    finally { $sha.Dispose() }
}
function ConvertFrom-Phase14Json {
    param([string]$Text)
    # ConvertFrom-Json loses duplicate keys; reject them before conversion.
    if (-not $Text -or $Text[0] -eq [char]0xfeff) { throw 'Invalid JSON transport' }
    $tokens = [regex]::Matches($Text, '"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"|[{}\[\]:,]|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|true|false|null')
    $stack = [Collections.Generic.Stack[object]]::new()
    $position = 0
    foreach ($token in $tokens) {
        if ($Text.Substring($position,$token.Index-$position).Trim().Length) { throw 'Invalid JSON token' }
        $position = $token.Index+$token.Length
        if ($token.Value -eq '{') { $stack.Push([Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)) }
        elseif ($token.Value -eq '[') { $stack.Push($null) }
        elseif ($token.Value -eq '}' -or $token.Value -eq ']') {
            if ($stack.Count -eq 0) { throw 'Invalid JSON structure' }; $null=$stack.Pop()
        }
        elseif ($token.Value.StartsWith('"') -and $Text.Substring($position).TrimStart().StartsWith(':')) {
            if ($stack.Count -eq 0 -or $null -eq $stack.Peek()) { throw 'Invalid JSON member' }
            $key = ConvertFrom-Json -InputObject ('{"key":'+$token.Value+'}') -ErrorAction Stop
            if (-not $stack.Peek().Add([string]$key.key)) { throw 'Duplicate JSON member' }
        }
    }
    if ($stack.Count -ne 0 -or $Text.Substring($position).Trim().Length) { throw 'Invalid JSON structure' }
    return ConvertFrom-Json -InputObject $Text -ErrorAction Stop
}
function Assert-Phase14Fields {
    param([object]$Value,[string[]]$Fields)
    if ($null -eq $Value -or $Value -isnot [pscustomobject] -or
        @(Compare-Object @($Fields | Sort-Object) @($Value.PSObject.Properties.Name | Sort-Object) -CaseSensitive).Count) { throw 'Invalid field inventory' }
}
function Assert-Phase14Identifier {
    param([object]$Value)
    if ($Value -isnot [string] -or $Value -cnotmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$') { throw 'Invalid procedural identifier' }
}
function Assert-Phase14Hash {
    param([object]$Value)
    if ($Value -isnot [string] -or $Value -cnotmatch '^[0-9a-f]{64}$') { throw 'Unresolved hash identity' }
}
function Get-Phase14CanonicalJson {
    param([object]$Value)
    function Order-Phase14Value {
        param($Item)
        if ($Item -is [array]) {
            $result=[object[]]::new($Item.Length)
            for ($index=0; $index -lt $Item.Length; $index++) { $result[$index]=Order-Phase14Value $Item[$index] }
            return ,$result
        }
        if ($null -eq $Item -or $Item -is [string] -or $Item -is [ValueType]) { return $Item }
        if ($Item -is [pscustomobject] -or $Item -is [Collections.IDictionary]) {
            $result=[ordered]@{}
            $keys=if ($Item -is [pscustomobject]) { $Item.PSObject.Properties.Name } else { $Item.Keys }
            foreach ($key in @($keys | Sort-Object -CaseSensitive)) { $result[$key]=Order-Phase14Value $Item.$key }
            return $result
        }
        return $Item
    }
    $text=ConvertTo-Json -InputObject (Order-Phase14Value $Value) -Depth 50 -Compress
    return [regex]::Replace($text,'[^\x00-\x7f]',{param($match) '\u'+([int][char]$match.Value).ToString('x4')})
}
function Assert-Phase14Path {
    param([string]$Path,[string]$Root)
    $absolute=[IO.Path]::GetFullPath($Path); $base=[IO.Path]::GetFullPath($Root).TrimEnd('\')
    if ($absolute -ne $base -and -not $absolute.StartsWith($base+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Path outside approved root' }
    $probe=$absolute
    while ($probe) {
        if (Test-Path -LiteralPath $probe) {
            if ((Get-Item -LiteralPath $probe -Force -ErrorAction Stop).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse path forbidden' }
        }
        $parent=Split-Path -Parent $probe
        if ($parent -eq $probe) { break }; $probe=$parent
    }
    return $absolute
}
function Write-Phase14JsonNew {
    param([string]$Path,[object]$Value)
    $bytes=[Text.UTF8Encoding]::new($false,$true).GetBytes((ConvertTo-Json -InputObject $Value -Depth 50 -Compress))
    $stream=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try { $stream.Write($bytes,0,$bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
}
function Read-Phase14Record {
    param([string]$Path)
    $null=Assert-Phase14Path $Path (Split-Path -Parent $Path)
    return ConvertFrom-Phase14Json ([IO.File]::ReadAllText($Path,[Text.UTF8Encoding]::new($false,$true)))
}
function Assert-Phase14Lock {
    param([object]$Lock,[object]$Authorization)
    Assert-Phase14Fields $Lock @('schema_version','lock_id','created_at_utc','configuration_sha256','evaluation_revision','environment','model_sha256','final_rows_sha256','protected_attestation_sha256','development_attestation_sha256','protocol','calibration_revision','evaluator_source_sha256','phase13','calibration','prediction_contract','reporting')
    Assert-Phase14Identifier $Lock.lock_id
    if ($Lock.schema_version -cne 'phase14a_pretest_lock.v1' -or $Lock.evaluation_revision -isnot [string] -or $Lock.evaluation_revision -cnotmatch '^[0-9a-f]{40}$' -or $Lock.evaluation_revision -cne $Authorization.expected_revision -or $Lock.lock_id -cne $Authorization.pretest_lock.lock_id -or $Lock.configuration_sha256 -cne $Authorization.configuration_sha256 -or (Get-Phase14CanonicalJson $Lock.environment) -cne (Get-Phase14CanonicalJson $Authorization.environment)) { throw 'Full lock binding mismatch' }
    if ($Lock.created_at_utc -isnot [string] -or $Lock.created_at_utc -cnotmatch '(Z|\+00:00)$' -or [DateTimeOffset]::Parse($Lock.created_at_utc).Offset -ne [TimeSpan]::Zero -or $Lock.calibration_revision -isnot [string] -or $Lock.calibration_revision -cnotmatch '^[0-9a-f]{40}$') { throw 'Invalid lock revision/timestamp' }
    foreach ($key in @('configuration_sha256','model_sha256','final_rows_sha256','protected_attestation_sha256','development_attestation_sha256','evaluator_source_sha256')) { Assert-Phase14Hash $Lock.$key }
    Assert-Phase14Fields $Lock.protocol @('id','sha256')
    if ($Lock.protocol.id -cne 'phase14a_calibration_final_test_protocol.v1' -or $Lock.protocol.sha256 -cne $script:Phase14ProtocolHash -or $Lock.evaluator_source_sha256 -cne $Authorization.source_files.'Thesis_RF/Code/evaluate_rf_final_test.py') { throw 'Lock protocol/source mismatch' }
    $phase13=[ordered]@{run_id='phase13-setc-rf-61e8f740-r2'; source_revision='61e8f74031d61c9be22a9b6408aec3ad74f4e892'; protocol_sha256='69c39d1d97430ebb903ec2d856e6205a2c1b8ad77641798955d6ad2bdb671637'; configuration_sha256='3800fe006463392f102a85ca3c491a6386b5340aeb8d2211eb31ae079a63bfc8'; manifest_sha256='b0d0d976a51d8559db73f9ee0db8365c0133c4414de5242cd328d0eb7ff96f3c'; base_model_sha256='7c3b431dcad6554d9ea80e0e6b0d13826245f8fcf3de9534ccfde78bd06b4ce1'; development_attestation_sha256='4fcf286156892c9da0289e56712210e90cccdcd608641c7ee573a8db1dcea23f'; source_csv_sha256='7f0f1867445141f365687d127e449117f234348cec571990d70bc2b226ba40f6'; source_manifest_sha256='b700f1b9f196893f23e7190daf1bc3c68b0f569f99a977ec9e919f0de73e9e3c'}
    if ((Get-Phase14CanonicalJson $Lock.phase13) -cne (Get-Phase14CanonicalJson $phase13)) { throw 'Accepted Phase 13 scalar binding mismatch' }
    Assert-Phase14Fields $Lock.calibration @('run_id','approval_reference','method','method_decision_reference','configuration_sha256','source_sha256','calibration_csv_sha256','manifest_sha256','report_sha256','receipt_sha256','supervisor_completion_sha256','package_review_sha256','forest_fingerprint_sha256')
    Assert-Phase14Identifier $Lock.calibration.run_id; Assert-Phase14Identifier $Lock.calibration.approval_reference
    foreach ($key in @($Lock.calibration.PSObject.Properties.Name | Where-Object { $_.EndsWith('sha256') })) { Assert-Phase14Hash $Lock.calibration.$key }
    if ($Lock.calibration.method -cne 'sigmoid' -or $Lock.calibration.method_decision_reference -cne 'D-077/D-079' -or $Lock.calibration.calibration_csv_sha256 -cne 'e424865be9143823f5a15fce27a3eb6df74abe802be84f142505f1209e3a77dc') { throw 'Calibration scalar policy mismatch' }
    $prediction=@{feature_schema_version='set_c.v1'; feature_names=@('slope_risk','proximity_risk','building_presence','material_risk','material_class','wind_speed','wind_sin','wind_cos','neighbor_burning_count','composite_flammability','wind_weighted_score'); feature_dtype='float32'; target_name='Ignited'; target_version='ignition_t_plus_1.v1'; target_dtype='int8'; required_labels=@(0,1); positive_label=1}
    $reporting=@{metrics=@('average_precision','roc_auc','brier_score','log_loss'); brier_definition='mean_squared_probability_error'; log_loss_labels=@(0,1); bin_edges=ConvertFrom-Phase14Json '[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]'; endpoint_rule='exact_internal_edge_starts_next_bin; p1_in_last_bin'; empty_bin_statistics=$null; classification_threshold=$null; verification_rel_tol=1e-10; verification_abs_tol=1e-12}
    if ((Get-Phase14CanonicalJson $Lock.prediction_contract) -cne (Get-Phase14CanonicalJson $prediction) -or (Get-Phase14CanonicalJson $Lock.reporting) -cne (Get-Phase14CanonicalJson $reporting)) { throw 'Lock prediction/reporting policy mismatch' }
}
function Assert-Phase14Authorization {
    param([string]$raw,[string]$hash,[string]$Job,[string]$ApprovalReference,[string]$ExpectedRevision,[string]$RunId,[string]$python)
    Assert-Phase14Hash $hash
    if ((Get-Phase14TextHash $raw) -cne $hash) { throw 'Exact transport digest mismatch' }
    $auth=ConvertFrom-Phase14Json $raw
    Assert-Phase14Fields $auth @('schema_version','job','approval_reference','run_id','expected_revision','created_at_utc','configuration_sha256','protocol_sha256','source_files','environment','environment_sha256','pretest_lock','primary_evidence')
    foreach ($field in @('schema_version','job','approval_reference','run_id','expected_revision','created_at_utc','configuration_sha256','protocol_sha256','environment_sha256')) { if ($auth.$field -isnot [string]) { throw 'Invalid authorization scalar type' } }
    if ($auth.schema_version -cne 'phase14a_job_authorization.v1' -or $auth.job -cne $Job -or $auth.run_id -cne $RunId -or $auth.approval_reference -cne $ApprovalReference -or $auth.expected_revision -cne $ExpectedRevision -or $auth.protocol_sha256 -cne $script:Phase14ProtocolHash) { throw 'Job authorization mismatch' }
    if ($auth.created_at_utc -cnotmatch '(Z|\+00:00)$' -or [DateTimeOffset]::Parse($auth.created_at_utc).Offset -ne [TimeSpan]::Zero) { throw 'UTC timestamp required' }
    Assert-Phase14Hash $auth.configuration_sha256
    Assert-Phase14Fields $auth.source_files $script:Phase14Sources
    foreach ($member in $script:Phase14Sources) { Assert-Phase14Hash $auth.source_files.$member }
    $configurationMember=if ($Job -eq 'calibration') { 'Thesis_RF/Code/config/phase14_setc_calibration_v1.yaml' } else { 'Thesis_RF/Code/config/phase14_setc_final_test_v1.yaml' }
    if ($auth.source_files.$configurationMember -cne $auth.configuration_sha256 -or $auth.source_files.'docs/Phase14A_Calibration_Final_Test_Protocol_Proposal.md' -cne $script:Phase14ProtocolHash) { throw 'Source/config/protocol cross-binding mismatch' }
    Assert-Phase14Fields $auth.environment @('status','versions','lock_sha256','interpreter','powershell','auxiliary_versions')
    Assert-Phase14Fields $auth.environment.versions @('python','scikit_learn','optuna','joblib','numpy','pandas','scipy','pyyaml','platform')
    Assert-Phase14Fields $auth.environment.interpreter @('path')
    Assert-Phase14Fields $auth.environment.powershell @('version','edition')
    Assert-Phase14Fields $auth.environment.auxiliary_versions @('rasterio')
    foreach ($value in @($auth.environment.status,$auth.environment.lock_sha256,$auth.environment.interpreter.path,$auth.environment.powershell.version,$auth.environment.powershell.edition,$auth.environment.auxiliary_versions.rasterio)) { if ($value -isnot [string] -or -not $value) { throw 'Invalid environment scalar type' } }
    foreach ($version in $auth.environment.versions.PSObject.Properties.Value) { if ($version -isnot [string] -or -not $version) { throw 'Incomplete versions' } }
    if ($auth.environment.status -cne 'complete_locked' -or $auth.environment.interpreter.path -ne $python -or $auth.environment.powershell.version -cne $PSVersionTable.PSVersion.ToString() -or $auth.environment.powershell.edition -cne $PSVersionTable.PSEdition -or $auth.environment.auxiliary_versions.rasterio -isnot [string] -or -not $auth.environment.auxiliary_versions.rasterio -or $auth.environment.lock_sha256 -cne (Get-Phase14TextHash (Get-Phase14CanonicalJson $auth.environment.versions)) -or $auth.environment_sha256 -cne (Get-Phase14TextHash (Get-Phase14CanonicalJson $auth.environment))) { throw 'Environment envelope mismatch' }
    Assert-Phase14Identifier $ApprovalReference; Assert-Phase14Identifier $RunId
    if ($ExpectedRevision -cnotmatch '^[0-9a-f]{40}$') { throw 'Invalid authorization revision' }
    if ($Job -eq 'calibration') {
        if ($null -ne $auth.pretest_lock -or $null -ne $auth.primary_evidence) { throw 'Calibration cannot bind final evidence' }
    } else {
        Assert-Phase14Fields $auth.pretest_lock @('lock_id','sha256','owner_approval_reference')
        Assert-Phase14Identifier $auth.pretest_lock.lock_id; Assert-Phase14Identifier $auth.pretest_lock.owner_approval_reference; Assert-Phase14Hash $auth.pretest_lock.sha256
        if ($Job -eq 'final_primary' -and $null -ne $auth.primary_evidence) { throw 'Primary cannot bind future evidence' }
        if ($Job -eq 'final_verification') {
            Assert-Phase14Fields $auth.primary_evidence @('run_id','report_sha256','receipt_sha256','supervisor_completion_sha256')
            Assert-Phase14Identifier $auth.primary_evidence.run_id
            if ($auth.primary_evidence.run_id -ceq $RunId) { throw 'Distinct primary run required' }
            foreach ($field in @('report_sha256','receipt_sha256','supervisor_completion_sha256')) { Assert-Phase14Hash $auth.primary_evidence.$field }
        }
    }
    return $auth
}
function Invoke-Phase14Git {
    param([string]$Repository,[string[]]$Arguments)
    $result=@(& git --literal-pathspecs -C $Repository @Arguments 2>$null)
    if ($LASTEXITCODE -ne 0) { throw 'Source observation failed' }
    return $result
}
function Get-Phase14ImportSources {
    param([string]$Repository)
    $excluded=@('tests','output','data','datafiles','processeddata','dataset','datasets','model','models','trainedmodels','trained_models','release','releases','artifacts','results','protected','setc_protected','setc_development','ground_truth','groundtruth','thesispaper','manuscript','manuscripts','.git','.venv','__pycache__')
    $pending=New-Object 'Collections.Generic.Stack[string]'
    $pending.Push((Assert-Phase14Path (Join-Path $Repository 'Thesis_RF\Code') $Repository))
    $sources=New-Object 'Collections.Generic.List[string]'
    $inspected=0
    while ($pending.Count) {
        foreach ($entry in @(Get-ChildItem -LiteralPath ($pending.Pop()) -Force -ErrorAction Stop)) {
            if ($entry.Name.ToLowerInvariant() -cin $excluded) { continue }
            $inspected++
            if ($inspected -gt 4096) { throw 'Runtime source inventory bound exceeded' }
            if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Runtime source reparse entry denied' }
            if ($entry.PSIsContainer) { $pending.Push($entry.FullName) }
            elseif ($entry.Extension.ToLowerInvariant() -cin @('.py','.ps1','.psm1')) {
                $sources.Add($entry.FullName.Substring($Repository.TrimEnd('\').Length+1).Replace('\','/'))
            }
        }
    }
    return @($sources.ToArray() | Sort-Object)
}
function Get-Phase14Arguments {
    param([string]$Job,[string]$ApprovalReference,[string]$ExpectedRevision,[string]$RunId,[string]$ConfigurationHash,[string]$code,[string]$config,[string]$evidence)
    $arguments=if ($Job -eq 'calibration') { @('-B','-m','modules.phase14_calibration') } else { @('-B','evaluate_rf_final_test.py','--mode',$Job.Substring(6)) }
    $arguments+=@('--config',$config,'--approval-reference',$ApprovalReference,'--expected-revision',$ExpectedRevision,'--run-id',$RunId,'--expected-config-sha256',$ConfigurationHash,'--evidence-dir',$evidence)
    return $arguments
}
function Get-Phase14UtcNow { return [DateTime]::UtcNow }
function Get-Phase14JobContext {
    param([string]$Job,[string]$ApprovalReference,[string]$ExpectedRevision,[string]$RunId)
    Assert-Phase14Identifier $ApprovalReference; Assert-Phase14Identifier $RunId
    if ($Job -cnotin @('calibration','final_primary','final_verification') -or $ExpectedRevision -cnotmatch '^[0-9a-f]{40}$') { throw 'Invalid fixed job/revision' }
    if ($PSVersionTable.PSVersion.ToString() -ne '5.1.26100.9444' -or $PSVersionTable.PSEdition -ne 'Desktop') { throw 'Required host mismatch' }
    $code=$PSScriptRoot; $repo=[IO.Path]::GetFullPath((Join-Path $code '..\..'))
    if ((Get-Location).Path -ne $code) { throw 'Fixed cwd mismatch' }
    $python=Assert-Phase14Path (Join-Path $repo '.venv\Scripts\python.exe') $repo
    $raw=[Environment]::GetEnvironmentVariable('PHASE14_AUTHORIZATION_JSON')
    $hash=[Environment]::GetEnvironmentVariable('PHASE14_AUTHORIZATION_SHA256')
    $auth=Assert-Phase14Authorization $raw $hash $Job $ApprovalReference $ExpectedRevision $RunId $python
    $head=Invoke-Phase14Git $repo @('rev-parse','HEAD')
    if ($head -cne $ExpectedRevision) { throw 'Observed revision mismatch' }
    $dirty=@(Invoke-Phase14Git $repo (@('status','--porcelain=v1','--untracked-files=all','--')+$script:Phase14Sources))
    if ($dirty.Count) { throw 'Runtime source dirty' }
    $inventory=@(Get-Phase14ImportSources $repo)
    $tracked=@()
    if ($inventory.Count) { $tracked=@(Invoke-Phase14Git $repo (@('ls-files','--cached','--')+$inventory)) }
    $additional=@($inventory | Where-Object { $_ -cnotin $script:Phase14Sources })
    if (@($additional | Where-Object { $_ -cnotin $tracked }).Count) { throw 'Unreviewed import source addition' }
    if ($additional.Count -and @(Invoke-Phase14Git $repo (@('status','--porcelain=v1','--untracked-files=all','--')+$additional)).Count) { throw 'Unreviewed import source addition' }
    foreach ($member in $script:Phase14Sources) {
        $path=Assert-Phase14Path (Join-Path $repo $member) $repo
        if ((Get-PhaseFileSha256 $path) -cne $auth.source_files.$member) { throw 'Reviewed source bytes mismatch' }
    }
    $configName=if ($Job -eq 'calibration') { 'phase14_setc_calibration_v1.yaml' } else { 'phase14_setc_final_test_v1.yaml' }
    $config=Assert-Phase14Path (Join-Path $code ('config\'+$configName)) $repo
    if ((Get-PhaseFileSha256 $config) -cne $auth.configuration_sha256 -or [IO.File]::ReadAllText($config) -notmatch '(?m)^execution_enabled:\s*true\s*$') { throw 'Fixed configuration disabled or mismatched' }
    $root=Assert-Phase14Path (Join-Path $code 'output\phase14') $repo
    $evidence=Assert-Phase14Path (Join-Path $root ('evidence\'+$RunId)) $root
    $lockPath=$null
    if ($Job -eq 'calibration') {
        if ($null -ne $auth.pretest_lock -or $null -ne $auth.primary_evidence) { throw 'Calibration cannot bind final evidence' }
        $destination=Join-Path $root ('phase14-setc-rf-61e8f740-r1\calibration\'+$RunId)
    }
    else {
        Assert-Phase14Fields $auth.pretest_lock @('lock_id','sha256','owner_approval_reference')
        Assert-Phase14Identifier $auth.pretest_lock.lock_id; Assert-Phase14Identifier $auth.pretest_lock.owner_approval_reference; Assert-Phase14Hash $auth.pretest_lock.sha256
        $lockPath=Assert-Phase14Path (Join-Path $root ('model-lock\'+$auth.pretest_lock.lock_id+'.json')) $root
        if ((Get-PhaseFileSha256 $lockPath) -cne $auth.pretest_lock.sha256) { throw 'External lock mismatch' }
        $lock=Read-Phase14Record $lockPath
        Assert-Phase14Lock $lock $auth
        Assert-Phase14Fields $lock @('schema_version','lock_id','created_at_utc','configuration_sha256','evaluation_revision','environment','model_sha256','final_rows_sha256','protected_attestation_sha256','development_attestation_sha256','protocol','calibration_revision','evaluator_source_sha256','phase13','calibration','prediction_contract','reporting')
        if ($lock.schema_version -cne 'phase14a_pretest_lock.v1' -or $lock.lock_id -cne $auth.pretest_lock.lock_id -or $lock.configuration_sha256 -cne $auth.configuration_sha256 -or $lock.evaluation_revision -cne $ExpectedRevision -or (Get-Phase14CanonicalJson $lock.environment) -cne (Get-Phase14CanonicalJson $auth.environment)) { throw 'Lock binding mismatch' }
        if ($Job -eq 'final_primary' -and $null -ne $auth.primary_evidence) { throw 'Primary cannot bind future evidence' }
        if ($Job -eq 'final_verification') {
            Assert-Phase14Fields $auth.primary_evidence @('run_id','report_sha256','receipt_sha256','supervisor_completion_sha256')
            Assert-Phase14Identifier $auth.primary_evidence.run_id
            if ($auth.primary_evidence.run_id -ceq $RunId) { throw 'Distinct primary run required' }
            foreach ($field in @('report_sha256','receipt_sha256','supervisor_completion_sha256')) { Assert-Phase14Hash $auth.primary_evidence.$field }
        }
        $destination=Join-Path $root ('final-test\'+$auth.pretest_lock.sha256.Substring(0,12)+'\'+$Job.Substring(6)+'\'+$RunId)
    }
    $destination=Assert-Phase14Path $destination $root
    if ((Test-Path -LiteralPath $destination) -or (Test-Path -LiteralPath $evidence)) { throw 'Output/evidence already exists' }
    $arguments=Get-Phase14Arguments $Job $ApprovalReference $ExpectedRevision $RunId $auth.configuration_sha256 $code $config $evidence
    return @{job=$Job; run_id=$RunId; approval=$ApprovalReference; revision=$ExpectedRevision; root=$root; code=$code; executable=$python; arguments=$arguments; evidence=$evidence; destination=$destination; lock_path=$lockPath; authorization=$auth; authorization_json=$raw; authorization_sha256=$hash; source_check=@{revision=$head; clean=$true; source_files=$auth.source_files; configuration_sha256=$auth.configuration_sha256; protocol_sha256=$script:Phase14ProtocolHash}}
}
function Wait-Phase14CaptureTasks {
    param([Threading.Tasks.Task[]]$Tasks,[int]$TimeoutMilliseconds)
    return [Threading.Tasks.Task]::WaitAll($Tasks,$TimeoutMilliseconds)
}
function Get-Phase14ProcessInstance {
    param([int]$ProcessId,[switch]$AllowMissing)
    $items=@(Get-CimInstance -ClassName Win32_Process -Filter ("ProcessId = "+$ProcessId) -OperationTimeoutSec 1 -ErrorAction Stop)
    if ($AllowMissing -and $items.Count -eq 0) { return $null }
    if ($items.Count -ne 1 -or $items[0].ProcessId -ne $ProcessId -or -not $items[0].CreationDate -or -not $items[0].ExecutablePath -or -not $items[0].CommandLine) { throw 'Specific OS process instance unavailable' }
    $item=$items[0]
    return @{pid=[int]$item.ProcessId;ppid=[int]$item.ParentProcessId;created_at_utc=$item.CreationDate.ToUniversalTime().ToString('o');executable=[string]$item.ExecutablePath;command_line=[string]$item.CommandLine}
}
function Get-Phase14BindingHash {
    param([object]$Binding,[string]$Evidence)
    $path=Assert-Phase14Path (Join-Path $Evidence 'worker_binding.json') $Evidence
    if ((Get-Phase14CanonicalJson (Read-Phase14Record $path)) -cne (Get-Phase14CanonicalJson $Binding)) { throw 'Immutable worker binding differs' }
    return Get-PhaseFileSha256 $path
}
function Assert-Phase14WorkerBinding {
    param([object]$Process,[object]$Handoff,[string]$Evidence)
    $Process=ConvertFrom-Phase14Json (Get-Phase14CanonicalJson $Process)
    $fields=@('schema_version','run_id','process_id','process_ids','process_started_at_utc','executable','working_directory','worker_binding')
    foreach ($optional in @('stop_reason','updated_at_utc')) { if ($Process.PSObject.Properties.Name -ccontains $optional) { $fields+=$optional } }
    Assert-Phase14Fields $Process $fields
    $binding=$Process.worker_binding
    Assert-Phase14Fields $binding @('schema_version','claim_sha256','claim','observed_at_utc','observation_source','supervisor','launcher','worker')
    $claimPath=Assert-Phase14Path (Join-Path $Evidence 'worker_claim.json') $Evidence
    $claim=Read-Phase14Record $claimPath
    Assert-Phase14Fields $claim @('schema_version','job','run_id','nonce','handoff_sha256','supervisor_pid','worker_pid','worker_ppid','logical_executable','physical_executable','worker_created_at_utc','command_line','claimed_at_utc')
    foreach ($instance in @($binding.supervisor,$binding.launcher,$binding.worker)) {
        Assert-Phase14Fields $instance @('pid','ppid','created_at_utc','executable','command_line')
        if ($instance.pid -isnot [int] -or $instance.pid -le 0 -or $instance.ppid -isnot [int] -or $instance.ppid -le 0 -or $instance.executable -isnot [string] -or -not [IO.Path]::IsPathRooted($instance.executable) -or $instance.command_line -isnot [string] -or -not $instance.command_line) { throw 'OS instance schema mismatch' }
    }
    foreach ($key in @('worker_pid','worker_ppid','supervisor_pid')) { if ($claim.$key -isnot [int] -or $claim.$key -le 0) { throw 'Claim PID schema mismatch' } }
    foreach ($key in @('job','run_id','nonce','supervisor_pid')) { if ($claim.$key -cne $Handoff.$key) { throw 'Claim handoff mismatch' } }
    $started=[DateTimeOffset]::Parse($Handoff.created_at_utc);$deadline=[DateTimeOffset]::Parse($Handoff.deadline_utc)
    $recorded=[DateTimeOffset]::Parse($Process.process_started_at_utc);$observed=[DateTimeOffset]::Parse($binding.observed_at_utc)
    $claimed=[DateTimeOffset]::Parse($claim.claimed_at_utc)
    $claimStart=[DateTimeOffset]::Parse($claim.worker_created_at_utc)
    $superStart=[DateTimeOffset]::Parse($binding.supervisor.created_at_utc);$launchStart=[DateTimeOffset]::Parse($binding.launcher.created_at_utc);$workerStart=[DateTimeOffset]::Parse($binding.worker.created_at_utc)
    foreach ($timestamp in @($started,$deadline,$recorded,$observed,$claimed,$superStart,$launchStart,$workerStart,$claimStart)) { if ($timestamp.Offset -ne [TimeSpan]::Zero) { throw 'Worker UTC identity invalid' } }
    [long]$workerTicks=$workerStart.UtcDateTime.Ticks;[long]$claimTicks=$claimStart.UtcDateTime.Ticks
    if ($Process.schema_version -cne 'phase14a_process.v2' -or $binding.schema_version -cne 'phase14a_worker_binding.v1' -or $claim.schema_version -cne 'phase14a_worker_claim.v1' -or $binding.observation_source -cne 'Get-CimInstance Win32_Process' -or $binding.claim_sha256 -cne (Get-PhaseFileSha256 $claimPath) -or (Get-Phase14CanonicalJson $binding.claim) -cne (Get-Phase14CanonicalJson $claim) -or $claim.handoff_sha256 -cne (Get-PhaseFileSha256 (Join-Path $Evidence 'supervisor_handoff.json')) -or $Process.run_id -cne $Handoff.run_id -or $Process.executable -ne $Handoff.executable -or $Process.working_directory -ne $Handoff.working_directory -or $claim.logical_executable -ne $Handoff.executable -or $Process.process_id -isnot [int] -or $Process.process_id -ne $binding.launcher.pid -or $binding.supervisor.pid -ne $Handoff.supervisor_pid -or $binding.launcher.ppid -ne $binding.supervisor.pid -or $binding.supervisor.pid -in @($binding.launcher.pid,$binding.worker.pid) -or $binding.worker.pid -ne $claim.worker_pid -or $binding.worker.ppid -ne $claim.worker_ppid -or ($binding.worker.pid -ne $binding.launcher.pid -and $binding.worker.ppid -ne $binding.launcher.pid) -or ($binding.worker.pid -eq $binding.launcher.pid -and (Get-Phase14CanonicalJson $binding.worker) -cne (Get-Phase14CanonicalJson $binding.launcher)) -or $binding.launcher.executable -ne $Handoff.executable -or $binding.worker.executable -ne $claim.physical_executable -or $binding.worker.command_line -cne $claim.command_line -or ($workerTicks - ($workerTicks % 10)) -ne ($claimTicks - ($claimTicks % 10)) -or -not ($superStart -le $started -and $started -le $launchStart -and $launchStart -le $recorded -and $recorded -le $observed -and $observed -lt $deadline -and $launchStart -le $workerStart -and $workerStart -le $claimed -and $claimed -le $observed) -or ($observed-$recorded).TotalSeconds -gt 10 -or $Process.process_ids -isnot [array] -or $Process.process_id -notin $Process.process_ids) { throw 'Worker startup binding mismatch' }
    foreach ($idValue in $Process.process_ids) { if ($idValue -isnot [int] -or $idValue -le 0) { throw 'Process inventory invalid' } }
    if (@($Process.process_ids | Select-Object -Unique).Count -ne $Process.process_ids.Count) { throw 'Process inventory duplicate' }
    $null=Get-Phase14BindingHash $binding $Evidence
    return $binding
}
function Get-Phase14WorkerExit {
    param([object]$Binding,[string]$Evidence)
    $worker=$Binding.worker
    $instance=Get-Phase14ProcessInstance -ProcessId $worker.pid -AllowMissing
    if ($null -ne $instance) { throw 'Worker still alive or PID reused' }
    return @{schema_version='phase14a_worker_exit.v1';worker_binding_sha256=(Get-Phase14BindingHash $Binding $Evidence);worker_pid=$worker.pid;worker_created_at_utc=$worker.created_at_utc;observed_at_utc=[DateTime]::UtcNow.ToString('o');observation_source='Get-CimInstance Win32_Process';same_instance_running=$false;pid_reused=$false}
}
function Assert-Phase14WorkerExit {
    param([object]$Exit,[object]$Binding,[DateTimeOffset]$Ended,[string]$Evidence)
    Assert-Phase14Fields $Exit @('schema_version','worker_binding_sha256','worker_pid','worker_created_at_utc','observed_at_utc','observation_source','same_instance_running','pid_reused')
    $observed=[DateTimeOffset]::Parse($Exit.observed_at_utc)
    if ($Exit.schema_version -cne 'phase14a_worker_exit.v1' -or $Exit.worker_binding_sha256 -cne (Get-Phase14BindingHash $Binding $Evidence) -or $Exit.worker_pid -isnot [int] -or $Exit.worker_pid -ne $Binding.worker.pid -or $Exit.worker_created_at_utc -cne $Binding.worker.created_at_utc -or $Exit.observation_source -cne 'Get-CimInstance Win32_Process' -or $Exit.same_instance_running -isnot [bool] -or $Exit.same_instance_running -or $Exit.pid_reused -isnot [bool] -or $Exit.pid_reused -or $observed.Offset -ne [TimeSpan]::Zero -or $observed -lt [DateTimeOffset]::Parse($Binding.observed_at_utc) -or $observed -gt $Ended) { throw 'Worker completion observation invalid' }
}
function Write-Phase14ProcessRecord {
    param([string]$Path,[object]$Record,[object]$Context,[object]$Handoff)
    $value=[ordered]@{schema_version='phase14a_process.v2'}
    foreach ($key in $Record.Keys) { $value[$key]=$Record[$key] }
    if (Test-Path -LiteralPath $Path) {
        $previous=Read-Phase14Record $Path
        $value.worker_binding=Assert-Phase14WorkerBinding $previous $Handoff $Context.evidence
        foreach ($key in @('run_id','process_id','process_started_at_utc','executable','working_directory')) { if ($value[$key] -cne $previous.$key) { throw 'Process update contradicts startup' } }
        if (@($previous.process_ids | Where-Object { $_ -notin $value.process_ids }).Count) { throw 'Process update discards inventory' }
    } else {
        $claimPath=Join-Path $Context.evidence 'worker_claim.json'
        $limit=[DateTime]::UtcNow.AddSeconds(10)
        $operationDeadline=[DateTimeOffset]::Parse($Handoff.deadline_utc).UtcDateTime
        if ($operationDeadline -lt $limit) { $limit=$operationDeadline }
        while (-not (Test-Path -LiteralPath $claimPath -PathType Leaf) -and [DateTime]::UtcNow -lt $limit) { Start-Sleep -Milliseconds 20 }
        if ([DateTime]::UtcNow -ge $limit) { throw 'Worker startup claim deadline reached' }
        $claim=Read-Phase14Record (Assert-Phase14Path $claimPath $Context.evidence)
        if ($claim.worker_pid -isnot [int] -or $claim.worker_pid -le 0) { throw 'Worker claim PID invalid' }
        $supervisor=Get-Phase14ProcessInstance $Handoff.supervisor_pid
        $launcher=Get-Phase14ProcessInstance $Record.process_id
        $worker=if ($claim.worker_pid -eq $Record.process_id) { $launcher } else { Get-Phase14ProcessInstance $claim.worker_pid }
        if ($supervisor.executable -ne ([Diagnostics.Process]::GetCurrentProcess().MainModule.FileName) -or [DateTime]::UtcNow -ge $limit) { throw 'Supervisor identity or startup deadline invalid' }
        $value.worker_binding=@{schema_version='phase14a_worker_binding.v1';claim_sha256=(Get-PhaseFileSha256 $claimPath);claim=$claim;observed_at_utc=[DateTime]::UtcNow.ToString('o');observation_source='Get-CimInstance Win32_Process';supervisor=$supervisor;launcher=$launcher;worker=$worker}
        Write-Phase14JsonNew (Join-Path $Context.evidence 'worker_binding.json') $value.worker_binding
        $null=Assert-Phase14WorkerBinding $value $Handoff $Context.evidence
    }
    # Replacement preserves the immutable binding while the core inventory grows.
    $temporary=Join-Path (Split-Path -Parent $Path) ('.phase14-process-'+[Guid]::NewGuid().ToString('N')+'.json')
    Write-Phase14JsonNew $temporary $value
    if (Test-Path -LiteralPath $Path) { [IO.File]::Replace($temporary,$Path,[System.Management.Automation.Language.NullString]::Value) } else { [IO.File]::Move($temporary,$Path) }
}
function Test-Phase14ChildReceipt {
    param([object]$Context,[object]$Handoff,[switch]$ReviewOnly)
    $path=Join-Path $Context.destination 'execution_receipt.json'
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return @{valid=$false; receipt=@{status='missing'; path=$null; sha256=$null}; outputs=@{}; handoff=$null} }
    $identity=@{status='present'; path=$path; sha256=$null}
    try {
        $identity.sha256=Get-PhaseFileSha256 (Assert-Phase14Path $path $Context.root)
        $receipt=Read-Phase14Record $path
        $fields=@('schema_version','run_id','approval_reference','argv','working_directory','source_revision','source_clean','source_check','supervisor_handoff','worker_binding_sha256','configuration_sha256','environment','inputs','computation_started_at_utc','computation_ended_at_utc','computation_started_at_local','computation_ended_at_local','local_timezone','computation_status','failure_reason','finalized_outputs','partial_output_inventory')
        if ($Context.job -eq 'calibration') { $fields+='execution_scope' } else { $fields+=@('model_sha256','final_rows_sha256') }
        Assert-Phase14Fields $receipt $fields
        foreach ($field in @('schema_version','run_id','approval_reference','working_directory','source_revision','configuration_sha256','computation_status','computation_started_at_utc','computation_ended_at_utc','computation_started_at_local','computation_ended_at_local','local_timezone')) { if ($receipt.$field -isnot [string]) { throw 'Invalid child receipt scalar type' } }
        Assert-Phase14Fields $receipt.source_check @('revision','clean','source_files','configuration_sha256','protocol_sha256')
        Assert-Phase14Fields $receipt.source_check.source_files $script:Phase14Sources
        Assert-Phase14Fields $receipt.supervisor_handoff @('run_id','nonce','sha256','supervisor_pid','child_pid')
        foreach ($member in $script:Phase14Sources) { Assert-Phase14Hash $receipt.source_check.source_files.$member }
        if ($receipt.source_check.clean -isnot [bool] -or -not $receipt.source_check.clean -or $receipt.source_check.revision -isnot [string] -or $receipt.source_check.revision -cnotmatch '^[0-9a-f]{40}$' -or $receipt.source_check.revision -cne $receipt.source_revision -or $receipt.supervisor_handoff.nonce -isnot [string] -or $receipt.supervisor_handoff.nonce -cnotmatch '^[0-9a-f]{32}$') { throw 'Child source/handoff evidence invalid' }
        $expectedEnvironment=if ($Context.job -eq 'calibration') { $Context.authorization.environment.versions } else { $Context.authorization.environment }
        if ((Get-Phase14CanonicalJson $receipt.environment) -cne (Get-Phase14CanonicalJson $expectedEnvironment)) { throw 'Observed child environment mismatch' }
        $process=Read-Phase14Record (Join-Path $Context.evidence 'process.json')
        $workerBinding=Assert-Phase14WorkerBinding $process $Handoff $Context.evidence
        if ($receipt.worker_binding_sha256 -cne (Get-Phase14BindingHash $workerBinding $Context.evidence)) { throw 'Receipt worker binding digest mismatch' }
        foreach ($value in @($receipt.computation_started_at_utc,$receipt.computation_ended_at_utc)) {
            if ($value -cnotmatch '(Z|\+00:00)$' -or [DateTimeOffset]::Parse($value).Offset -ne [TimeSpan]::Zero) { throw 'Child computation UTC timestamp invalid' }
        }
        $computationStart=[DateTimeOffset]::Parse($receipt.computation_started_at_utc); $computationEnd=[DateTimeOffset]::Parse($receipt.computation_ended_at_utc)
        $operationDeadline=[DateTimeOffset]::Parse($Handoff.deadline_utc)
        if ($computationStart -lt [DateTimeOffset]::Parse($workerBinding.observed_at_utc) -or $computationStart -lt [DateTimeOffset]::Parse($process.process_started_at_utc) -or $computationEnd -lt $computationStart -or $computationEnd -ge $operationDeadline) { throw 'Child computation chronology invalid' }
        $binding=@{run_id=$Context.run_id; nonce=$Handoff.nonce; sha256=(Get-PhaseFileSha256 (Join-Path $Context.evidence 'supervisor_handoff.json')); supervisor_pid=$Handoff.supervisor_pid; child_pid=$workerBinding.worker.pid}
        $success=if ($Context.job -eq 'calibration') { 'complete' } else { 'computation_complete' }
        if ($receipt.schema_version -cne 'phase14a_execution_receipt.v2' -or $receipt.run_id -cne $Context.run_id -or $receipt.approval_reference -cne $Context.approval -or $receipt.source_revision -cne $Context.revision -or $receipt.source_clean -isnot [bool] -or -not $receipt.source_clean -or $receipt.configuration_sha256 -cne $Context.authorization.configuration_sha256 -or $receipt.working_directory -ne $Context.code -or $receipt.computation_status -cne $success -or $null -ne $receipt.failure_reason -or (Get-Phase14CanonicalJson $receipt.argv) -cne (Get-Phase14CanonicalJson $Context.arguments) -or (Get-Phase14CanonicalJson $receipt.source_check) -cne (Get-Phase14CanonicalJson $Context.source_check) -or (Get-Phase14CanonicalJson $receipt.supervisor_handoff) -cne (Get-Phase14CanonicalJson $binding)) { throw 'Child computation binding invalid' }
        if ($process.process_id -isnot [int] -or $process.process_id -le 0 -or $Handoff.supervisor_pid -isnot [int] -or $Handoff.supervisor_pid -le 0) { throw 'Invalid observed PID' }
        $names=if ($Context.job -eq 'calibration') { @('base_rf_calibrated.joblib','calibration_manifest.json','calibration_report.json') } elseif ($Context.job -eq 'final_primary') { @('final_test_report.json') } else { @('verification_metrics.json') }
        Assert-Phase14Fields $receipt.finalized_outputs $names
        $outputs=[ordered]@{}
        foreach ($name in $names) {
            Assert-Phase14Hash $receipt.finalized_outputs.$name
            $outputPath=Assert-Phase14Path (Join-Path $Context.destination $name) $Context.destination
            if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf)) { throw 'Finalized output missing' }
            # Review binds the already verified model hash without reading model contents.
            $outputs[$name]=if ($ReviewOnly -and $name -eq 'base_rf_calibrated.joblib') { $receipt.finalized_outputs.$name } else { Get-PhaseFileSha256 $outputPath }
            if ($outputs[$name] -cne $receipt.finalized_outputs.$name) { throw 'Output identity mismatch' }
        }
        if (-not $ReviewOnly) {
            [DateTimeOffset]$validationObserved=Get-Phase14UtcNow
            if ($validationObserved.Offset -ne [TimeSpan]::Zero -or $computationEnd -gt $validationObserved -or $validationObserved -ge $operationDeadline) { throw 'Live child computation chronology invalid' }
        }
        return @{valid=$true; receipt=$identity; outputs=$outputs; handoff=$binding}
    }
    catch { return @{valid=$false; receipt=$identity; outputs=@{}; handoff=$null; error=('child receipt validation ('+$_.Exception.GetType().Name+')')} }
}
function Invoke-Phase14Supervision {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Job,[Parameter(Mandatory)][string]$ApprovalReference,[Parameter(Mandatory)][string]$ExpectedRevision,[Parameter(Mandatory)][string]$RunId)
    $started=Get-Phase14UtcNow; $deadline=$started.AddSeconds(1800)
    $evidenceCreated=$false
    try {
        $context=Get-Phase14JobContext $Job $ApprovalReference $ExpectedRevision $RunId
        $resource=Get-PhaseMemoryAndDisk
        if ($resource.available_physical_memory_bytes -lt 4GB -or $resource.free_c_bytes -lt 2GB -or (Get-Phase14UtcNow) -ge $deadline) { throw 'Fresh resource/time gate failed' }
        $null=[IO.Directory]::CreateDirectory((Split-Path -Parent $context.evidence))
        New-Item -ItemType Directory -Path $context.evidence -ErrorAction Stop | Out-Null
        $evidenceCreated=$true
        $handoff=[ordered]@{schema_version='phase14a_supervisor_handoff.v1'; job=$Job; run_id=$RunId; approval_reference=$ApprovalReference; expected_revision=$ExpectedRevision; authorization_json=$context.authorization_json; authorization_sha256=$context.authorization_sha256; nonce=[Guid]::NewGuid().ToString('N'); supervisor_pid=$PID; executable=$context.executable; argv=$context.arguments; working_directory=$context.code; evidence_directory=$context.evidence; destination=$context.destination; created_at_utc=$started.ToString('o'); deadline_utc=$deadline.ToString('o')}
        $handoffPath=Join-Path $context.evidence 'supervisor_handoff.json'
        $writer=(Get-Command Write-Phase14JsonNew -CommandType Function).ScriptBlock
        $processWriter=(Get-Command Write-Phase14ProcessRecord -CommandType Function).ScriptBlock
        $receiptCheck=(Get-Command Test-Phase14ChildReceipt -CommandType Function).ScriptBlock
        $beforeStart={ & $writer $handoffPath $handoff; return @{PHASE14_HANDOFF_PATH=$handoffPath; PHASE14_SUPERVISOR_NONCE=$handoff.nonce; PYTHONDONTWRITEBYTECODE='1'} }.GetNewClosure()
        $recordProcess={param($record) & $processWriter (Join-Path $context.evidence 'process.json') $record $context $handoff}.GetNewClosure()
        $inspectExit={
            param($exitStatus,$treeState,$stopReason)
            $package=& $receiptCheck $context $handoff
            if ($stopReason -eq 'completed') {
                if ($null -eq $exitStatus) { $stopReason='child exit status could not be observed' }
                elseif ($exitStatus -ne 0) { $stopReason='child exited nonzero' }
                elseif (-not $package.valid) { $stopReason='child computation evidence invalid' }
                elseif (-not $treeState.process_exited -or $treeState.remaining_process_ids.Count) { $stopReason='child process tree remains alive' }
            }
            return @{evidence=$package; stop_reason=$stopReason}
        }.GetNewClosure()
        $remaining=[int][Math]::Floor(($deadline-(Get-Phase14UtcNow)).TotalSeconds)
        if ($remaining -le 0) { throw 'Operation budget exhausted' }
        $core=@{Executable=$context.executable; ChildArguments=$context.arguments; WorkingDirectory=$context.code; EvidenceDirectory=$context.evidence; RunId=$RunId; ResourceLogPath=(Join-Path $context.evidence 'operator_resources.jsonl'); OverallLimitSeconds=$remaining; SampleIntervalSeconds=1; MemoryFloorBytes=4GB; DiskFloorBytes=2GB; ResourceProvider=(Get-Command Get-PhaseMemoryAndDisk).ScriptBlock; Clock={ [DateTime]::UtcNow }; CaptureWait=(Get-Command Wait-Phase14CaptureTasks).ScriptBlock; BeforeStart=$beforeStart; RecordProcess=$recordProcess; InspectExit=$inspectExit}
        $state=Invoke-PhaseSupervisionCore @core
        $package=Test-Phase14ChildReceipt $context $handoff
        $workerExit=$null; $workerExitError=$false
        try { $process=Read-Phase14Record (Join-Path $context.evidence 'process.json'); $workerBinding=Assert-Phase14WorkerBinding $process $handoff $context.evidence; $workerExit=Get-Phase14WorkerExit $workerBinding $context.evidence } catch { $workerExitError=$true }
        $errors=[ordered]@{}
        foreach ($name in @('runnerError','cleanupError','captureError','finalizationError')) { $errors[$name]=if ($state[$name]) { 'supervision '+$name+' (Exception)' } else { $null } }
        if ($workerExitError -and -not $errors.finalizationError) { $errors.finalizationError='worker completion observation (Exception)' }
        $safeStops=@('completed','overall time limit reached','resource floor breached','child exit status could not be observed','child exited nonzero','child computation evidence invalid','child process tree remains alive')
        $stop=if ($state.stopReason -cin $safeStops) { $state.stopReason } else { 'supervision failed' }
        $success=$null -ne $workerExit -and $package.valid -and $state.stopReason -eq 'completed' -and $null -ne $state.exitStatus -and $state.exitStatus -eq 0 -and $state.captureComplete -and $state.captureStatus -eq 'complete' -and $state.treeState.process_exited -and $state.treeState.remaining_process_ids.Count -eq 0 -and -not ($errors.Values | Where-Object { $_ })
        $lockHash=if ($context.authorization.pretest_lock) { $context.authorization.pretest_lock.sha256 } else { $null }
        $completion=[ordered]@{schema_version='phase14a_supervisor_completion.v2'; job=$Job; run_id=$RunId; approval_reference=$ApprovalReference; source_check=$context.source_check; supervisor_handoff=$package.handoff; worker_binding_sha256=$(if ($workerExit) { $workerExit.worker_binding_sha256 } else { $null }); worker_exit=$workerExit; pretest_lock_sha256=$lockHash; child_receipt=$package.receipt; verified_outputs=$package.outputs; actual_exit_status=$state.exitStatus; stop_reason=$stop; capture_complete=$state.captureComplete; capture_status=$state.captureStatus; capture_error=$errors.captureError; stdout_sha256=$state.stdoutHash; stderr_sha256=$state.stderrHash; resource_sha256=$state.resourceHash; capture_drain_timed_out=$state.captureDrainTimedOut; capture_drain_timeout_ms=$state.captureDrainTimeoutMilliseconds; process_ids=@($state.treeState.process_ids); remaining_process_ids=@($state.treeState.remaining_process_ids); process_exited=$state.treeState.process_exited; resource_samples=@($state.samples.ToArray()); runner_error=$errors.runnerError; cleanup_error=$errors.cleanupError; finalization_error=$errors.finalizationError; started_at_utc=$started.ToString('o'); ended_at_utc=$null; deadline_utc=$deadline.ToString('o'); shutdown_limitation='best-effort; synchronous OS operations may exceed deadline'; overall_status=$null}
        # D-067: decide once before synchronous exclusive publication; publication may overrun.
        $decision=Get-Phase14UtcNow
        $success=$success -and $decision -lt $deadline
        $completion.ended_at_utc=$decision.ToString('o')
        $completion.overall_status=if ($success) { 'PASS' } else { 'NEEDS EVIDENCE' }
        Write-Phase14JsonNew (Join-Path $context.evidence 'supervisor_completion.json') $completion
        if (-not $success) { throw 'Supervised operation incomplete' }
        return $completion
    }
    catch {
        $failureType=$_.Exception.GetType().Name
        if ($evidenceCreated -and -not (Test-Path -LiteralPath (Join-Path $context.evidence 'supervisor_completion.json'))) {
            try {
                $failed=[ordered]@{schema_version='phase14a_supervisor_completion.v2'; job=$Job; run_id=$RunId; approval_reference=$ApprovalReference; source_check=$context.source_check; supervisor_handoff=$null; worker_binding_sha256=$null; worker_exit=$null; pretest_lock_sha256=$null; child_receipt=@{status='missing';path=$null;sha256=$null}; verified_outputs=@{}; actual_exit_status=$null; stop_reason='supervision finalization failed'; capture_complete=$false; capture_status='unknown'; capture_error=$null; stdout_sha256=$null; stderr_sha256=$null; resource_sha256=$null; capture_drain_timed_out=$false; capture_drain_timeout_ms=0; process_ids=@(); remaining_process_ids=@(); process_exited=$null; resource_samples=@(); runner_error=('supervision failure ('+$failureType+')'); cleanup_error=$null; finalization_error='supervisor completion unavailable'; started_at_utc=$started.ToString('o'); ended_at_utc=[DateTime]::UtcNow.ToString('o'); deadline_utc=$deadline.ToString('o'); shutdown_limitation='best-effort; synchronous OS operations may exceed deadline'; overall_status='NEEDS EVIDENCE'}
                $childPath=Join-Path $context.destination 'execution_receipt.json'
                if (Test-Path -LiteralPath $childPath -PathType Leaf) {
                    $failed.child_receipt=@{status='present';path=$childPath;sha256=$null}
                    try { $failed.child_receipt.sha256=Get-PhaseFileSha256 (Assert-Phase14Path $childPath $context.root) } catch { }
                }
                Write-Phase14JsonNew (Join-Path $context.evidence 'supervisor_completion.json') $failed
            } catch { }
        }
        throw ('Phase 14 supervision denied or incomplete ('+$failureType+')')
    }
}
function Test-Phase14Number {
    param([object]$Value)
    return ($Value -is [int] -or $Value -is [long] -or $Value -is [double] -or $Value -is [decimal]) -and -not [double]::IsNaN([double]$Value) -and -not [double]::IsInfinity([double]$Value)
}
function Test-Phase14Close {
    param([object]$Left,[object]$Right)
    if ($null -eq $Left -or $null -eq $Right) { return $null -eq $Left -and $null -eq $Right }
    if (-not (Test-Phase14Number $Left) -or -not (Test-Phase14Number $Right)) { return $false }
    return [Math]::Abs([double]$Left-[double]$Right) -le [Math]::Max(1e-12,1e-10*[Math]::Max([Math]::Abs([double]$Left),[Math]::Abs([double]$Right)))
}
function Assert-Phase14ProbabilityReport {
    param([object]$Report,[string]$Mode,[object]$Lock,[string]$LockHash)
    $fields=@('schema_version','metrics','row_count','positive_count','negative_count','prevalence','classification_threshold','reliability_bins')
    if ($Mode -eq 'calibration') { $fields+='diagnostic_scope' } else { $fields+=@('evaluation_scope','evaluation_label','evaluation_mode','identities') }
    Assert-Phase14Fields $Report $fields
    foreach ($key in @('row_count','positive_count','negative_count')) {
        if (($Report.$key -isnot [int] -and $Report.$key -isnot [long]) -or $Report.$key -lt 1) { throw 'Invalid aggregate counts' }
    }
    if ($Report.positive_count+$Report.negative_count -ne $Report.row_count -or -not (Test-Phase14Close $Report.prevalence ($Report.positive_count/[double]$Report.row_count)) -or $null -ne $Report.classification_threshold) { throw 'Invalid prevalence/threshold' }
    if ($Mode -eq 'calibration') {
        if ($Report.schema_version -cne 'phase14a_calibration_report.v1' -or $Report.diagnostic_scope -cne 'in-sample calibration-fit diagnostics' -or $Report.row_count -ne 2060 -or $Report.positive_count -ne 153 -or $Report.negative_count -ne 1907) { throw 'Calibration aggregate population mismatch' }
    } else {
        $schema=if ($Mode -eq 'primary') { 'phase14a_final_test_report.v1' } else { 'phase14a_verification_metrics.v1' }
        $identities=@{rows_sha256=$Lock.final_rows_sha256; model_sha256=$Lock.model_sha256; protected_attestation_sha256=$Lock.protected_attestation_sha256; development_attestation_sha256=$Lock.development_attestation_sha256; configuration_sha256=$Lock.configuration_sha256; evaluation_revision=$Lock.evaluation_revision; pretest_lock_sha256=$LockHash}
        if ($Report.schema_version -cne $schema -or $Report.evaluation_scope -cne 'final_test' -or $Report.evaluation_label -cne 'untouched final test' -or $Report.evaluation_mode -cne $Mode -or (Get-Phase14CanonicalJson $Report.identities) -cne (Get-Phase14CanonicalJson $identities)) { throw 'Final report identity/schema mismatch' }
    }
    Assert-Phase14Fields $Report.metrics @('average_precision','roc_auc','brier_score','log_loss')
    foreach ($name in @('average_precision','roc_auc','brier_score','log_loss')) {
        $metric=$Report.metrics.$name
        Assert-Phase14Fields $metric @('value','status','reason')
        if ($metric.status -ceq 'defined') {
            if (-not (Test-Phase14Number $metric.value) -or $metric.value -lt 0 -or ($name -ne 'log_loss' -and $metric.value -gt 1) -or $null -ne $metric.reason) { throw 'Invalid defined metric' }
        } elseif ($metric.status -ceq 'undefined' -and $name -ne 'brier_score') {
            if ($null -ne $metric.value -or $metric.reason -isnot [string] -or -not $metric.reason -or ($Mode -ne 'calibration' -and $metric.reason -cnotin @('non-finite metric','metric undefined for these labels'))) { throw 'Invalid undefined metric' }
        } else { throw 'Invalid metric state' }
    }
    if ($Report.reliability_bins -isnot [array] -or $Report.reliability_bins.Count -ne 10) { throw 'Invalid bin inventory' }
    $edges=if ($Mode -eq 'calibration') { ConvertFrom-Phase14Json '[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]' } else { ConvertFrom-Phase14Json '[0.0,0.1,0.2,0.30000000000000004,0.4,0.5,0.6000000000000001,0.7000000000000001,0.8,0.9,1.0]' }; $total=0L
    for ($index=0; $index -lt 10; $index++) {
        $bin=$Report.reliability_bins[$index]
        Assert-Phase14Fields $bin @('index','lower','upper','endpoint_rule','count','mean_predicted_probability','observed_positive_fraction','status')
        if (($bin.index -isnot [int] -and $bin.index -isnot [long]) -or $bin.index -ne $index -or ($bin.count -isnot [int] -and $bin.count -isnot [long]) -or $bin.count -lt 0 -or -not (Test-Phase14Number $bin.lower) -or -not (Test-Phase14Number $bin.upper) -or $bin.lower -ne $edges[$index] -or $bin.upper -ne $edges[$index+1] -or $bin.endpoint_rule -cne 'lower inclusive; upper exclusive except bin 9 includes 1.0') { throw 'Invalid bin endpoints/count' }
        $total+=$bin.count
        if ($bin.count -eq 0) {
            if ($bin.status -cne 'empty' -or $null -ne $bin.mean_predicted_probability -or $null -ne $bin.observed_positive_fraction) { throw 'Invalid empty bin' }
        } else {
            if ($bin.status -cne 'defined') { throw 'Invalid populated bin' }
            foreach ($key in @('mean_predicted_probability','observed_positive_fraction')) {
                if (-not (Test-Phase14Number $bin.$key) -or $bin.$key -lt 0 -or $bin.$key -gt 1) { throw 'Invalid bin statistic' }
            }
        }
    }
    if ($total -ne $Report.row_count) { throw 'Bin counts do not cover population' }
}
function Test-Phase14ReportsMatch {
    param([object]$Primary,[object]$Verifier)
    foreach ($name in @('row_count','positive_count','negative_count','classification_threshold','evaluation_scope','evaluation_label','identities')) {
        $left=$Primary.$name; $right=$Verifier.$name
        $leftIsNull=[object]::ReferenceEquals($left,$null); $rightIsNull=[object]::ReferenceEquals($right,$null)
        if ($leftIsNull -or $rightIsNull) {
            if ($leftIsNull -ne $rightIsNull) { return $false }
            continue
        }
        if ((Get-Phase14CanonicalJson $left) -cne (Get-Phase14CanonicalJson $right)) { return $false }
    }
    if (-not (Test-Phase14Close $Primary.prevalence $Verifier.prevalence)) { return $false }
    foreach ($name in @('average_precision','roc_auc','brier_score','log_loss')) {
        $left=$Primary.metrics.$name; $right=$Verifier.metrics.$name
        if ($left.status -cne $right.status -or $left.reason -cne $right.reason) { return $false }
        # The approved Python comparison treats a left integer as exact.
        if ($left.value -is [int] -or $left.value -is [long]) {
            if ($left.value -ne $right.value) { return $false }
        } elseif (-not (Test-Phase14Close $left.value $right.value)) { return $false }
    }
    for ($index=0; $index -lt 10; $index++) {
        $left=$Primary.reliability_bins[$index]; $right=$Verifier.reliability_bins[$index]
        foreach ($name in @('index','lower','upper','endpoint_rule','count','status')) { if ($left.$name -cne $right.$name) { return $false } }
        foreach ($name in @('mean_predicted_probability','observed_positive_fraction')) { if (-not (Test-Phase14Close $left.$name $right.$name)) { return $false } }
    }
    return $true
}
function Assert-Phase14CalibrationMetadata {
    param([object]$Manifest,[object]$Report,[object]$Receipt,[object]$Context,[object]$Outputs)
    Assert-Phase14ProbabilityReport $Report 'calibration' $null $null
    Assert-Phase14Fields $Manifest @('schema_version','protocol_id','protocol_sha256','run_id','source_revision','source_clean','execution_scope','approval_reference','command','configuration_sha256','working_directory','actual_base_model_identity','expected_policy','output_paths','phase13_expected_policy','method','method_decision_reference','feature_schema_version','feature_names','target_name','target_version','positive_label','input_role','input','software','forest_fingerprint','calibrated_model','report','status','started_at_utc','ended_at_utc','computation_started_at_local','computation_ended_at_local','local_timezone')
    $provenance=@{attestation_sha256='4fcf286156892c9da0289e56712210e90cccdcd608641c7ee573a8db1dcea23f'; calibration_csv_sha256='e424865be9143823f5a15fce27a3eb6df74abe802be84f142505f1209e3a77dc'; rows=2060; positives=153; negatives=1907; source_csv_sha256='7f0f1867445141f365687d127e449117f234348cec571990d70bc2b226ba40f6'; source_revision='61e8f74031d61c9be22a9b6408aec3ad74f4e892'; synthetic=$false}
    $base=@{kind='phase13_base_model_file';sha256='7c3b431dcad6554d9ea80e0e6b0d13826245f8fcf3de9534ccfde78bd06b4ce1';manifest_sha256='b0d0d976a51d8559db73f9ee0db8365c0133c4414de5242cd328d0eb7ff96f3c'}
    $policy=@{protocol_id='phase14a_calibration_final_test_protocol.v1';protocol_sha256=$script:Phase14ProtocolHash;phase13_model_sha256=$base.sha256;phase13_manifest_sha256=$base.manifest_sha256}
    $phase13=@{run_id='phase13-setc-rf-61e8f740-r2';source_revision=$provenance.source_revision;protocol_sha256='69c39d1d97430ebb903ec2d856e6205a2c1b8ad77641798955d6ad2bdb671637';config_sha256='3800fe006463392f102a85ca3c491a6386b5340aeb8d2211eb31ae079a63bfc8';manifest_sha256=$base.manifest_sha256;base_model_sha256=$base.sha256}
    $paths=@{calibrated_model='base_rf_calibrated.joblib';manifest='calibration_manifest.json';report='calibration_report.json';execution_receipt='execution_receipt.json'}
    Assert-Phase14Fields $Manifest.forest_fingerprint @('before_sha256','after_sha256','embedded_sha256','positive_class_index')
    foreach ($key in @('before_sha256','after_sha256','embedded_sha256')) { Assert-Phase14Hash $Manifest.forest_fingerprint.$key }
    if ($Manifest.forest_fingerprint.before_sha256 -cne $Manifest.forest_fingerprint.after_sha256 -or $Manifest.forest_fingerprint.before_sha256 -cne $Manifest.forest_fingerprint.embedded_sha256 -or $Manifest.forest_fingerprint.positive_class_index -isnot [int] -or $Manifest.forest_fingerprint.positive_class_index -notin @(0,1)) { throw 'Frozen forest metadata mismatch' }
    $expected=@{schema_version='phase14a_calibrated_model_manifest.v1';protocol_id=$policy.protocol_id;protocol_sha256=$script:Phase14ProtocolHash;run_id=$Context.run_id;source_revision=$Context.revision;source_clean=$true;execution_scope='production_calibration';approval_reference=$Context.approval;command=$Context.arguments;configuration_sha256=$Context.authorization.configuration_sha256;working_directory=$Context.code;actual_base_model_identity=$base;expected_policy=$policy;output_paths=$paths;phase13_expected_policy=$phase13;method='sigmoid';method_decision_reference='D-077/D-079';feature_schema_version='set_c.v1';feature_names=@('slope_risk','proximity_risk','building_presence','material_risk','material_class','wind_speed','wind_sin','wind_cos','neighbor_burning_count','composite_flammability','wind_weighted_score');target_name='Ignited';target_version='ignition_t_plus_1.v1';positive_label=1;input_role='calibration';input=$provenance;software=$Context.authorization.environment.versions;calibrated_model=@{filename='base_rf_calibrated.joblib';sha256=$Outputs.'base_rf_calibrated.joblib'};report=@{filename='calibration_report.json';sha256=$Outputs.'calibration_report.json'};status='complete'}
    foreach ($key in $expected.Keys) { if ((Get-Phase14CanonicalJson $Manifest.$key) -cne (Get-Phase14CanonicalJson $expected[$key])) { throw 'Calibration manifest binding mismatch' } }
    if ($Receipt.execution_scope -cne 'production_calibration' -or (Get-Phase14CanonicalJson $Receipt.inputs) -cne (Get-Phase14CanonicalJson @{role='calibration';synthetic=$false;provenance=$provenance})) { throw 'Calibration receipt input mismatch' }
    foreach ($key in @('started_at_utc','ended_at_utc')) { if ($Manifest.$key -isnot [string] -or $Manifest.$key -cnotmatch '(Z|\+00:00)$' -or [DateTimeOffset]::Parse($Manifest.$key).Offset -ne [TimeSpan]::Zero) { throw 'Invalid manifest timestamp' } }
    if ([DateTimeOffset]::Parse($Manifest.started_at_utc) -gt [DateTimeOffset]::Parse($Manifest.ended_at_utc) -or $Manifest.computation_started_at_local -isnot [string] -or $Manifest.computation_ended_at_local -isnot [string] -or $Manifest.local_timezone -isnot [string] -or -not $Manifest.local_timezone) { throw 'Invalid manifest chronology' }
}
function Write-Phase14ReviewRecord {
    [CmdletBinding()]
    param([Parameter(Mandatory)][ValidateSet('calibration','verification')][string]$ReviewKind,
          [Parameter(Mandatory)][string]$ReviewerId,[Parameter(Mandatory)][string]$ApprovalReference,
          [string]$CalibrationRunId,[string]$PrimaryRunId,[string]$VerificationRunId,[string]$LockId,
          [Parameter(Mandatory)][ValidateSet('PASS','FAIL','NEEDS EVIDENCE')][string]$Disposition,
          [Parameter(Mandatory)][string[]]$Findings)
    try {
        Assert-Phase14Identifier $ReviewerId; Assert-Phase14Identifier $ApprovalReference
        if (@($Findings | Where-Object { $_ -cnotin @('evidence_verified','evidence_missing','evidence_invalid','supervision_incomplete','aggregate_mismatch') }).Count) { throw 'Only redacted finding codes permitted' }
        $raw=[Environment]::GetEnvironmentVariable('PHASE14_REVIEW_AUTHORIZATION_JSON')
        $hash=[Environment]::GetEnvironmentVariable('PHASE14_REVIEW_AUTHORIZATION_SHA256'); Assert-Phase14Hash $hash
        if ((Get-Phase14TextHash $raw) -cne $hash) { throw 'Review authorization mismatch' }
        $auth=ConvertFrom-Phase14Json $raw
        Assert-Phase14Fields $auth @('schema_version','review_kind','reviewer_id','approval_reference','calibration_run_id','primary_run_id','verification_run_id','lock_id','disposition','findings','created_at_utc')
        foreach ($field in @('schema_version','review_kind','reviewer_id','approval_reference','calibration_run_id','primary_run_id','verification_run_id','lock_id','disposition','created_at_utc')) { if ($auth.$field -isnot [string]) { throw 'Invalid review authorization type' } }
        if ($auth.schema_version -cne 'phase14a_review_authorization.v1' -or $auth.review_kind -cne $ReviewKind -or $auth.reviewer_id -cne $ReviewerId -or $auth.approval_reference -cne $ApprovalReference -or $auth.calibration_run_id -cne $CalibrationRunId -or $auth.primary_run_id -cne $PrimaryRunId -or $auth.verification_run_id -cne $VerificationRunId -or $auth.lock_id -cne $LockId -or $auth.disposition -cne $Disposition -or (Get-Phase14CanonicalJson $auth.findings) -cne (Get-Phase14CanonicalJson $Findings)) { throw 'Review authorization fields mismatch' }
        if ($auth.created_at_utc -cnotmatch '(Z|\+00:00)$' -or [DateTimeOffset]::Parse($auth.created_at_utc).Offset -ne [TimeSpan]::Zero) { throw 'UTC review timestamp required' }
        $root=Assert-Phase14Path (Join-Path $PSScriptRoot 'output\phase14') $PSScriptRoot
        $runs=if ($ReviewKind -eq 'calibration') { @($CalibrationRunId) } else { @($PrimaryRunId,$VerificationRunId) }
        $bindings=[ordered]@{}; $reports=@{}; $observedFindings=@(); $lockHash=$null; $lock=$null; $identities=$null; $lockVerified=$false
        if ($ReviewKind -eq 'verification') {
            Assert-Phase14Identifier $LockId
            if ($PrimaryRunId -ceq $VerificationRunId -or $CalibrationRunId) { throw 'Distinct final run identifiers required' }
            $lockPath=Assert-Phase14Path (Join-Path $root ('model-lock\'+$LockId+'.json')) $root
            $lockHash=Get-PhaseFileSha256 $lockPath
            try {
                $lock=Read-Phase14Record $lockPath
                Assert-Phase14Fields $lock @('schema_version','lock_id','created_at_utc','configuration_sha256','evaluation_revision','environment','model_sha256','final_rows_sha256','protected_attestation_sha256','development_attestation_sha256','protocol','calibration_revision','evaluator_source_sha256','phase13','calibration','prediction_contract','reporting')
                if ($lock.schema_version -cne 'phase14a_pretest_lock.v1' -or $lock.lock_id -cne $LockId) { throw 'Reviewed lock schema invalid' }
                $primaryHandoffPath=Assert-Phase14Path (Join-Path $root ('evidence\'+$PrimaryRunId+'\supervisor_handoff.json')) $root
                if (Test-Path -LiteralPath $primaryHandoffPath -PathType Leaf) {
                    $primaryHandoff=Read-Phase14Record $primaryHandoffPath
                    $repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
                    $python=[IO.Path]::GetFullPath((Join-Path $repo '.venv\Scripts\python.exe'))
                    $primaryAuthorization=Assert-Phase14Authorization $primaryHandoff.authorization_json $primaryHandoff.authorization_sha256 'final_primary' $primaryHandoff.approval_reference $primaryHandoff.expected_revision $PrimaryRunId $python
                    Assert-Phase14Lock $lock $primaryAuthorization
                    if ($primaryAuthorization.pretest_lock.sha256 -cne $lockHash) { throw 'Reviewed external lock mismatch' }
                    $lockVerified=$true
                } else { $observedFindings+='evidence_missing' }
            } catch { $observedFindings+='evidence_invalid' }
        }
        elseif ($PrimaryRunId -or $VerificationRunId -or $LockId) { throw 'Calibration review cannot bind future evidence' }
        foreach ($run in $runs) {
            Assert-Phase14Identifier $run
            $job=if ($ReviewKind -eq 'calibration') { 'calibration' } elseif ($run -ceq $PrimaryRunId) { 'final_primary' } else { 'final_verification' }
            $evidence=Assert-Phase14Path (Join-Path $root ('evidence\'+$run)) $root
            $destination=if ($job -eq 'calibration') { Join-Path $root ('phase14-setc-rf-61e8f740-r1\calibration\'+$run) } else { Join-Path $root ('final-test\'+$lockHash.Substring(0,12)+'\'+$job.Substring(6)+'\'+$run) }
            $destination=Assert-Phase14Path $destination $root
            $receiptPath=Join-Path $destination 'execution_receipt.json'
            $completionPath=Join-Path $evidence 'supervisor_completion.json'
            $binding=@{child_receipt=@{status='missing';path=$null;sha256=$null}; supervisor_completion=@{status='missing';path=$null;sha256=$null}; outputs=@{}}
            # Inventory these independently, including a receipt whose completion is absent.
            foreach ($item in @(@('child_receipt',$receiptPath),@('supervisor_completion',$completionPath))) {
                if (Test-Path -LiteralPath $item[1] -PathType Leaf) {
                    $binding[$item[0]]=@{status='present';path=$item[1];sha256=(Get-PhaseFileSha256 (Assert-Phase14Path $item[1] $root))}
                }
            }
            $bindings[$run]=$binding
            if ($binding.child_receipt.status -eq 'missing' -or $binding.supervisor_completion.status -eq 'missing') { $observedFindings+='evidence_missing'; continue }
            try {
                $completion=Read-Phase14Record $completionPath
                $handoff=Read-Phase14Record (Join-Path $evidence 'supervisor_handoff.json')
                Assert-Phase14Fields $handoff @('schema_version','job','run_id','approval_reference','expected_revision','authorization_json','authorization_sha256','nonce','supervisor_pid','executable','argv','working_directory','evidence_directory','destination','created_at_utc','deadline_utc')
                Assert-Phase14Fields $completion @('schema_version','job','run_id','approval_reference','source_check','supervisor_handoff','worker_binding_sha256','worker_exit','pretest_lock_sha256','child_receipt','verified_outputs','actual_exit_status','stop_reason','capture_complete','capture_status','capture_error','stdout_sha256','stderr_sha256','resource_sha256','capture_drain_timed_out','capture_drain_timeout_ms','process_ids','remaining_process_ids','process_exited','resource_samples','runner_error','cleanup_error','finalization_error','started_at_utc','ended_at_utc','deadline_utc','shutdown_limitation','overall_status')
                $repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
                $python=[IO.Path]::GetFullPath((Join-Path $repo '.venv\Scripts\python.exe'))
                $authorization=Assert-Phase14Authorization $handoff.authorization_json $handoff.authorization_sha256 $job $handoff.approval_reference $handoff.expected_revision $run $python
                $config=Join-Path $PSScriptRoot ('config\'+$(if ($job -eq 'calibration') { 'phase14_setc_calibration_v1.yaml' } else { 'phase14_setc_final_test_v1.yaml' }))
                $arguments=Get-Phase14Arguments $job $handoff.approval_reference $handoff.expected_revision $run $authorization.configuration_sha256 $PSScriptRoot $config $evidence
                $source=@{revision=$handoff.expected_revision;clean=$true;source_files=$authorization.source_files;configuration_sha256=$authorization.configuration_sha256;protocol_sha256=$authorization.protocol_sha256}
                if ($ReviewKind -eq 'verification') {
                    Assert-Phase14Lock $lock $authorization
                    if ($authorization.pretest_lock.sha256 -cne $lockHash -or $authorization.pretest_lock.lock_id -cne $LockId) { throw 'Reviewed lock identity mismatch' }
                    $lockVerified=$true
                }
                if ($handoff.schema_version -cne 'phase14a_supervisor_handoff.v1' -or $handoff.job -cne $job -or $handoff.run_id -cne $run -or $handoff.working_directory -ne $PSScriptRoot -or $handoff.evidence_directory -ne $evidence -or $handoff.destination -ne $destination -or $handoff.executable -ne $python -or (Get-Phase14CanonicalJson $handoff.argv) -cne (Get-Phase14CanonicalJson $arguments) -or $handoff.nonce -isnot [string] -or $handoff.nonce -cnotmatch '^[0-9a-f]{32}$' -or $handoff.supervisor_pid -isnot [int] -or $handoff.supervisor_pid -le 0 -or $completion.schema_version -cne 'phase14a_supervisor_completion.v2' -or $completion.job -cne $job -or $completion.run_id -cne $run -or $completion.approval_reference -cne $handoff.approval_reference -or (Get-Phase14CanonicalJson $source) -cne (Get-Phase14CanonicalJson $completion.source_check)) { throw 'Reviewed context mismatch' }
                foreach ($value in @($handoff.created_at_utc,$handoff.deadline_utc,$completion.started_at_utc,$completion.ended_at_utc,$completion.deadline_utc)) {
                    if ($value -isnot [string] -or $value -cnotmatch '(Z|\+00:00)$' -or [DateTimeOffset]::Parse($value).Offset -ne [TimeSpan]::Zero) { throw 'Reviewed UTC timestamp invalid' }
                }
                $started=[DateTimeOffset]::Parse($handoff.created_at_utc); $ended=[DateTimeOffset]::Parse($completion.ended_at_utc); $deadline=[DateTimeOffset]::Parse($handoff.deadline_utc)
                if (($deadline-$started).TotalSeconds -ne 1800 -or $completion.started_at_utc -cne $handoff.created_at_utc -or $completion.deadline_utc -cne $handoff.deadline_utc -or $ended -lt $started -or ($completion.overall_status -ceq 'PASS' -and $ended -ge $deadline)) { throw 'Reviewed chronology mismatch' }
                $context=@{destination=$destination;evidence=$evidence;root=$root;code=$PSScriptRoot;run_id=$run;job=$job;approval=$handoff.approval_reference;revision=$handoff.expected_revision;authorization=$authorization;arguments=$arguments;source_check=$source}
                $package=Test-Phase14ChildReceipt $context $handoff -ReviewOnly
                if (-not $package.valid) { throw 'Reviewed child evidence invalid' }
                $binding.outputs=$package.outputs
                if ((Get-Phase14CanonicalJson $completion.child_receipt) -cne (Get-Phase14CanonicalJson $package.receipt) -or (Get-Phase14CanonicalJson $completion.verified_outputs) -cne (Get-Phase14CanonicalJson $package.outputs) -or (Get-Phase14CanonicalJson $completion.supervisor_handoff) -cne (Get-Phase14CanonicalJson $package.handoff) -or $completion.pretest_lock_sha256 -cne $(if ($job -eq 'calibration') { $null } else { $lockHash })) { throw 'Reviewed supervisory binding mismatch' }
                $process=Read-Phase14Record (Join-Path $evidence 'process.json')
                $processFields=@('schema_version','run_id','process_id','process_ids','process_started_at_utc','executable','working_directory','worker_binding')
                foreach ($optional in @('stop_reason','updated_at_utc')) { if ($process.PSObject.Properties.Name -ccontains $optional) { $processFields+=$optional } }
                Assert-Phase14Fields $process $processFields
                if ($process.schema_version -cne 'phase14a_process.v2' -or $process.run_id -cne $run -or $process.executable -ne $python -or $process.working_directory -ne $PSScriptRoot -or $process.worker_binding.worker.pid -ne $package.handoff.child_pid -or $process.process_ids -isnot [array] -or $completion.process_ids -isnot [array] -or $process.process_ids.Count -eq 0 -or $process.process_id -notin $process.process_ids) { throw 'Reviewed process binding mismatch' }
                foreach ($pidValue in @($process.process_ids)+@($completion.process_ids)) { if ($pidValue -isnot [int] -or $pidValue -le 0) { throw 'Reviewed process IDs invalid' } }
                if (@($process.process_ids | Select-Object -Unique).Count -ne $process.process_ids.Count -or @($completion.process_ids | Select-Object -Unique).Count -ne $completion.process_ids.Count -or @($process.process_ids | Where-Object { $_ -notin $completion.process_ids }).Count) { throw 'Reviewed process inventory mismatch' }
                $workerBinding=Assert-Phase14WorkerBinding $process $handoff $evidence
                if ($completion.worker_binding_sha256 -cne (Get-Phase14BindingHash $workerBinding $evidence)) { throw 'Reviewed worker binding mismatch' }
                Assert-Phase14WorkerExit $completion.worker_exit $workerBinding $ended $evidence
                $processStart=[DateTimeOffset]::Parse($process.process_started_at_utc)
                if ($processStart.Offset -ne [TimeSpan]::Zero -or $processStart -lt $started -or $processStart -gt $ended) { throw 'Reviewed process chronology mismatch' }
                if ($process.PSObject.Properties.Name -ccontains 'updated_at_utc') {
                    $updated=[DateTimeOffset]::Parse($process.updated_at_utc)
                    if ($updated.Offset -ne [TimeSpan]::Zero -or $updated -lt $processStart -or $updated -gt $ended) { throw 'Reviewed process update mismatch' }
                }
                if ($process.PSObject.Properties.Name -ccontains 'stop_reason' -and $process.stop_reason -cne 'completed') { $observedFindings+='supervision_incomplete' }
                foreach ($capture in @(@('stdout.log','stdout_sha256'),@('stderr.log','stderr_sha256'),@('operator_resources.jsonl','resource_sha256'))) {
                    Assert-Phase14Hash $completion.($capture[1])
                    if ((Get-PhaseFileSha256 (Assert-Phase14Path (Join-Path $evidence $capture[0]) $root)) -cne $completion.($capture[1])) { throw 'Reviewed capture digest mismatch' }
                }
                if ($null -ne $completion.actual_exit_status -and $completion.actual_exit_status -isnot [int]) { throw 'Reviewed exit status type invalid' }
                foreach ($key in @('overall_status','stop_reason','capture_status','shutdown_limitation')) { if ($completion.$key -isnot [string] -or -not $completion.$key) { throw 'Reviewed supervisory state type invalid' } }
                if ($completion.overall_status -cnotin @('PASS','FAIL','NEEDS EVIDENCE')) { throw 'Reviewed completion status invalid' }
                foreach ($key in @('runner_error','cleanup_error','capture_error','finalization_error')) { if ($null -ne $completion.$key -and ($completion.$key -isnot [string] -or -not $completion.$key)) { throw 'Reviewed error state type invalid' } }
                if ($completion.capture_complete -isnot [bool] -or $completion.process_exited -isnot [bool] -or $completion.remaining_process_ids -isnot [array] -or $completion.capture_drain_timed_out -isnot [bool] -or $completion.capture_drain_timeout_ms -isnot [int] -or $completion.capture_drain_timeout_ms -lt 0 -or $completion.resource_samples -isnot [array]) { throw 'Reviewed completion scalar schema invalid' }
                if ($completion.overall_status -cne 'PASS' -or $completion.actual_exit_status -isnot [int] -or $completion.actual_exit_status -ne 0 -or $completion.stop_reason -cne 'completed' -or -not $completion.capture_complete -or $completion.capture_status -cne 'complete' -or -not $completion.process_exited -or $completion.remaining_process_ids.Count -or $completion.runner_error -or $completion.cleanup_error -or $completion.capture_error -or $completion.finalization_error -or $completion.capture_drain_timed_out) { $observedFindings+='supervision_incomplete' }
                $receipt=Read-Phase14Record $receiptPath
                foreach ($value in @($receipt.computation_started_at_utc,$receipt.computation_ended_at_utc)) {
                    if ($value -isnot [string] -or $value -cnotmatch '(Z|\+00:00)$' -or [DateTimeOffset]::Parse($value).Offset -ne [TimeSpan]::Zero) { throw 'Reviewed computation timestamp invalid' }
                }
                $computationStart=[DateTimeOffset]::Parse($receipt.computation_started_at_utc); $computationEnd=[DateTimeOffset]::Parse($receipt.computation_ended_at_utc)
                if ($computationStart -lt [DateTimeOffset]::Parse($workerBinding.observed_at_utc) -or $computationStart -lt $processStart -or $computationEnd -lt $computationStart -or $computationEnd -gt $ended) { throw 'Reviewed computation chronology mismatch' }
                if ((Get-Phase14CanonicalJson $receipt.partial_output_inventory) -cne (Get-Phase14CanonicalJson @($package.outputs.Keys | Sort-Object))) { throw 'Reviewed output inventory invalid' }
                if ($job -eq 'calibration') {
                    $report=Read-Phase14Record (Join-Path $destination 'calibration_report.json')
                    $manifest=Read-Phase14Record (Join-Path $destination 'calibration_manifest.json')
                    Assert-Phase14CalibrationMetadata $manifest $report $receipt $context $package.outputs
                    $identities=@{model_sha256=$completion.verified_outputs.'base_rf_calibrated.joblib'}
                } else {
                    $name=if ($job -eq 'final_primary') { 'final_test_report.json' } else { 'verification_metrics.json' }
                    $report=Read-Phase14Record (Join-Path $destination $name)
                    Assert-Phase14ProbabilityReport $report ($job.Substring(6)) $lock $lockHash
                    $inputs=@{rows_sha256=$lock.final_rows_sha256;model_sha256=$lock.model_sha256;protected_attestation_sha256=$lock.protected_attestation_sha256;development_attestation_sha256=$lock.development_attestation_sha256}
                    if ($receipt.model_sha256 -cne $lock.model_sha256 -or $receipt.final_rows_sha256 -cne $lock.final_rows_sha256 -or (Get-Phase14CanonicalJson $receipt.inputs) -cne (Get-Phase14CanonicalJson $inputs)) { throw 'Reviewed final receipt identities invalid' }
                    if ($job -eq 'final_verification') {
                        $primary=$bindings[$PrimaryRunId]
                        $expectedPrimary=@{run_id=$PrimaryRunId;report_sha256=$primary.outputs.'final_test_report.json';receipt_sha256=$primary.child_receipt.sha256;supervisor_completion_sha256=$primary.supervisor_completion.sha256}
                        if ($primary.child_receipt.status -eq 'missing' -or $primary.supervisor_completion.status -eq 'missing') { $observedFindings+='evidence_missing' }
                        elseif ((Get-Phase14CanonicalJson $authorization.primary_evidence) -cne (Get-Phase14CanonicalJson $expectedPrimary)) { throw 'Reviewed primary evidence binding mismatch' }
                    }
                    $reports[$run]=$report
                }
            } catch { $observedFindings+='evidence_invalid' }
        }
        if ($ReviewKind -eq 'verification' -and $lockVerified -and $reports.ContainsKey($PrimaryRunId) -and $reports.ContainsKey($VerificationRunId)) {
            $identities=@{model_sha256=$lock.model_sha256;final_rows_sha256=$lock.final_rows_sha256;configuration_sha256=$lock.configuration_sha256;evaluation_revision=$lock.evaluation_revision;evaluator_source_sha256=$lock.evaluator_source_sha256;environment=$lock.environment;environment_sha256=(Get-Phase14TextHash (Get-Phase14CanonicalJson $lock.environment))}
            if (-not (Test-Phase14ReportsMatch $reports[$PrimaryRunId] $reports[$VerificationRunId])) { $observedFindings+='aggregate_mismatch' }
        }
        $findings=@($Findings)+@($observedFindings) | Select-Object -Unique
        $status=$Disposition
        if ($findings -ccontains 'aggregate_mismatch' -or $findings -ccontains 'evidence_invalid') { $status='FAIL' }
        elseif ($Disposition -eq 'PASS' -and ($findings -ccontains 'evidence_missing' -or $findings -ccontains 'supervision_incomplete')) { $status='NEEDS EVIDENCE' }
        $reviewPath=if ($ReviewKind -eq 'calibration') { Join-Path $root ('calibration-review\'+$CalibrationRunId+'\review_record.json') } else { Join-Path $root ('final-test\'+$lockHash.Substring(0,12)+'\verification\'+$VerificationRunId+'\review_record.json') }
        $reviewPath=Assert-Phase14Path $reviewPath $root; $null=[IO.Directory]::CreateDirectory((Split-Path -Parent $reviewPath))
        $record=[ordered]@{schema_version=$(if ($ReviewKind -eq 'calibration') { 'phase14a_calibration_package_review.v1' } else { 'phase14a_independent_review.v1' });reviewer_id=$ReviewerId;approval_reference=$ApprovalReference;created_at_utc=[DateTime]::UtcNow.ToString('o');disposition=$status;findings=@($findings);pretest_lock_sha256=$lockHash;identities=$identities;evidence=$bindings}
        Write-Phase14JsonNew $reviewPath $record
        return $record
    }
    catch { throw ('Phase 14 review denied ('+$_.Exception.GetType().Name+')') }
}
if ($MyInvocation.InvocationName -ne '.') {
    if ($PSBoundParameters.Count -ne 4) { throw 'Exactly Job, ApprovalReference, ExpectedRevision and RunId required' }
    Invoke-Phase14Supervision -Job $Job -ApprovalReference $ApprovalReference -ExpectedRevision $ExpectedRevision -RunId $RunId
}
