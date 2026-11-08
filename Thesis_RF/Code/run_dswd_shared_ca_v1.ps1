# D-338 source preparation only. Invocation requires a later exact owner release.
# One selected trajectory; no retry, campaign continuation, scoring or reference reads.
# Ordinary library initialization/deserialization execute code; no sandbox claim.
# Limits are cooperative/sampled; pre-binding workers and other descendants are not supervised.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidatePattern('^D-[0-9]+$')][string]$OwnerApproval,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ReviewedLauncherSha256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ExpectedBindingSha256,
    [Parameter(Mandatory=$true)][string]$RunId,
    [Parameter(Mandatory=$true)][ValidateSet('rf','lr')][string]$Arm,
    [Parameter(Mandatory=$true)][switch]$ExecuteApprovedTrajectory,
    [Parameter(Mandatory=$true)][switch]$AdoptCallerLimits
)
Set-StrictMode -Version 2
$ErrorActionPreference='Stop'
if ($PSVersionTable.PSVersion.Major -ne 5 -or $PSVersionTable.PSVersion.Minor -ne 1 -or
    $PSVersionTable.PSEdition -ne 'Desktop') {throw 'Windows PowerShell 5.1 Desktop required.'}
if (-not $ExecuteApprovedTrajectory -or -not $AdoptCallerLimits -or
    [int]$OwnerApproval.Substring(2) -le 338) {throw 'Fresh owner execution release required.'}
# OwnerApproval is an operator assertion, not authority conferred by a configuration label.
$repo='C:\Users\gibgib\Thesis-Cellular-Automata'
$code=Join-Path $repo 'Thesis_RF\Code'
$python=Join-Path $repo '.venv\Scripts\python.exe'
$basePython='C:\Python314\python.exe'
$launcherPath=[IO.Path]::GetFullPath($PSCommandPath)
if ($launcherPath -ine (Join-Path $code 'run_dswd_shared_ca_v1.ps1')) {throw 'Canonical caller path required.'}
$utf8=[Text.UTF8Encoding]::new($false,$true)
$root=Join-Path 'C:\Users\gibgib\AppData\Local\Temp' ('rf-dswd-trajectory-v1-'+[Guid]::NewGuid().ToString('N'))
$attempts=@{}
$s=@{process=$null;worker=$null;started=$false;bound=$false;sent=$false;inputClosed=$false
    launcherWaited=$false;workerWaited=$false;launcherExit=$null;workerExit=$null
    killRequested=$false;killAt=$null;binding=$null;errors=[Collections.Generic.List[string]]::new()
    stdout=$null;stderr=$null;journal=$null;inputTask=$null;inputStream=$null
    stdoutDone=$false;stderrDone=$false;nextSample=0.0;killErrors=@()
    launcherPre=$null;launcherPost=$null;admissionReceipt=$null;identityReceipt=$null
    configPre=$null;configPost=$null}
$clock=[Diagnostics.Stopwatch]::StartNew()
$memoryInfo=$null
$drive=$null
function Is-Integer($v) {
    return ($v -is [int] -or $v -is [long] -or $v -is [uint32] -or $v -is [uint64])
}
function No-Reparse([string]$Path) {
    $item=Get-Item -Force -LiteralPath $Path
    while ($null -ne $item) {
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {throw 'Reparse component.'}
        if ($item -is [IO.DirectoryInfo]) {$item=$item.Parent} else {$item=$item.Directory}
    }
}
function Hash-Bytes([byte[]]$Bytes) {
    $h=[Security.Cryptography.SHA256]::Create()
    try {return ([BitConverter]::ToString($h.ComputeHash($Bytes))).Replace('-','').ToLowerInvariant()}
    finally {$h.Dispose()}
}
function Read-Bytes([string]$Path,[long]$Cap) {
    No-Reparse $Path
    $a=Get-Item -Force -LiteralPath $Path
    if ($a -isnot [IO.FileInfo] -or $a.Length -lt 0 -or $a.Length -gt $Cap) {throw 'Read type/cap.'}
    $before=@($a.Length,$a.LastWriteTimeUtc.Ticks,$a.CreationTimeUtc.Ticks)
    $f=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
    try {
        $n=$f.Length
        if ($n -ne $a.Length) {throw 'Read open length drift.'}
        $bytes=[byte[]]::new([int]$n)
        $offset=0
        while ($offset -lt $bytes.Length) {
            $got=$f.Read($bytes,$offset,[Math]::Min(65536,$bytes.Length-$offset))
            if ($got -eq 0) {throw 'Premature evidence EOF.'}
            $offset+=$got
        }
        if ($f.Length -ne $n) {throw 'Read handle length drift.'}
    } finally {$f.Dispose()}
    $b=Get-Item -Force -LiteralPath $Path
    if ($before[0] -ne $b.Length -or $before[1] -ne $b.LastWriteTimeUtc.Ticks -or
        $before[2] -ne $b.CreationTimeUtc.Ticks) {throw 'Read temporal drift.'}
    return ,$bytes
}
function Receipt([string]$Name,[long]$Cap=65536) {
    $b=Read-Bytes (Join-Path $root $Name) $Cap
    return [ordered]@{filename=$Name;bytes=$b.Length;sha256=(Hash-Bytes $b)}
}
function Publish([string]$Name,$Value,[long]$Cap=65536) {
    if ($attempts.ContainsKey($Name)) {throw 'Publication already attempted.'}
    $attempts[$Name]=$true
    $bytes=$utf8.GetBytes((ConvertTo-Json -InputObject $Value -Depth 70 -Compress))
    if ($bytes.Length -gt $Cap) {throw 'Publication cap.'}
    $f=[IO.File]::Open((Join-Path $root $Name),[IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write,[IO.FileShare]::Read)
    try {$f.Write($bytes,0,$bytes.Length);$f.Flush($true)} finally {$f.Dispose()}
    $r=Receipt $Name $Cap
    if ($r.bytes -ne $bytes.Length -or $r.sha256 -cne (Hash-Bytes $bytes)) {throw 'Publication receipt.'}
    return $r
}
function Marker([string]$Name) {
    if ($attempts.ContainsKey($Name)) {throw 'Marker already attempted.'}
    $attempts[$Name]=$true
    $f=[IO.File]::Open((Join-Path $root $Name),[IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write,[IO.FileShare]::Read)
    try {$f.Flush($true)} finally {$f.Dispose()}
}
function Ready([string]$Name) {
    $p=Join-Path $root $Name
    try {$v=Get-Item -Force -LiteralPath $p -ErrorAction Stop}
    catch [System.Management.Automation.ItemNotFoundException] {return $false}
    No-Reparse $p
    if ($v -isnot [IO.FileInfo] -or $v.Length -ne 0) {throw 'Invalid ready marker.'}
    return $true
}
function Journal($Value) {
    $b=$utf8.GetBytes((ConvertTo-Json -InputObject $Value -Depth 12 -Compress)+[char]10)
    if ($s.journal.Length+$b.Length -gt 8388608) {throw 'Resource journal cap.'}
    $s.journal.Write($b,0,$b.Length);$s.journal.Flush($true)
}
function Sample($Proc,[string]$Role) {
    $id=0;$rss=$null;$exited=$false
    if ($null -ne $Proc) {
        $id=[int]$Proc.Id;$Proc.Refresh();$exited=$Proc.HasExited
        if (-not $exited) {$rss=[long]$Proc.WorkingSet64}
    }
    $available=[System.UInt64]$memoryInfo.AvailablePhysicalMemory
    $free=[long]$drive.AvailableFreeSpace
    $v=[ordered]@{role=$Role;parent_pid=[int]$PID;child_pid=$id
        utc_ms=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        elapsed_ms=$clock.Elapsed.TotalMilliseconds
        child_rss_bytes=$rss;process_has_exited=$exited
        system_available_bytes=$available;evidence_free_bytes=$free
        ok=(($null -eq $rss -or $rss -le 4294967296L) -and $available -ge 2147483648L -and
            $free -ge 268435456L -and $clock.Elapsed.TotalMilliseconds -le 1920000)}
    Journal $v
    if (-not $v.ok) {throw ('Observed resource breach: '+$Role)}
    return $v
}
function Stop-Owned {
    if ($s.killRequested) {return}
    $s.killRequested=$true;$s.killAt=$clock.Elapsed.TotalMilliseconds
    $seen=@{}
    $targets=@()
    if ($s.bound) {$targets+=,$s.worker}
    if ($s.started) {$targets+=,$s.process}
    foreach ($p in $targets) {
        if ($null -eq $p -or $seen.ContainsKey([string]$p.Id)) {continue}
        $seen[[string]$p.Id]=$true
        try {if (-not $p.HasExited) {$p.Kill()}}
        catch {$s.killErrors+=,$_.ToString()}
    }
}
function New-Pump($InputStream,[string]$Name) {
    $out=[IO.File]::Open((Join-Path $root $Name),[IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write,[IO.FileShare]::Read)
    try {
        $b=[byte[]]::new(65536)
        return @{input=$InputStream;output=$out;buffer=$b;task=$InputStream.ReadAsync($b,0,$b.Length)
            eof=$false;overflow=$false;bytes=0L;name=$Name}
    } catch {
        $constructionFailure=$_
        try {$out.Dispose()}
        catch {$s.errors.Add('Pump construction disposal: '+$_.ToString())}
        throw $constructionFailure
    }
}
function Pump($p) {
    if ($null -eq $p -or $p.eof -or $p.overflow -or -not $p.task.IsCompleted) {return}
    $n=$p.task.GetAwaiter().GetResult()
    if ($n -eq 0) {$p.eof=$true;return}
    $room=16777216L-$p.bytes
    $retain=[int][Math]::Min([long]$n,$room)
    if ($retain -gt 0) {$p.output.Write($p.buffer,0,$retain);$p.bytes+=$retain}
    if ($n -gt $room) {$p.overflow=$true;throw 'Raw capture exceeded cap; prefix retained.'}
    $p.task=$p.input.ReadAsync($p.buffer,0,$p.buffer.Length)
}
function Tick {
    Pump $s.stdout;Pump $s.stderr
    if ($clock.Elapsed.TotalMilliseconds -gt 1920000) {throw 'Parent elapsed budget.'}
    if ($clock.Elapsed.TotalMilliseconds -ge $s.nextSample) {
        $null=Sample $s.process 'launcher'
        if ($s.bound) {$null=Sample $s.worker 'verified_worker'}
        $s.nextSample=$clock.Elapsed.TotalMilliseconds+1000
    }
}
function Equal-Receipt($a,$b) {
    return ($a.filename -ceq $b.filename -and (Is-Integer $a.bytes) -and
        $a.bytes -eq $b.bytes -and $a.sha256 -ceq $b.sha256)
}
$configPins=ConvertFrom-Json @'
{
  "p16-dswd-shared-v1-airbase-rf-s43-r1": "404a90489188a1d8f4810676b0f3e533cbc36ee47dbb4f5fe8787affd39af53d",
  "p16-dswd-shared-v1-airbase-rf-s44-r1": "8953cfb837e4e2eda53f363711dda3df48ad8b8d0f89ac45e66e15302337bf3f",
  "p16-dswd-shared-v1-airbase-rf-s45-r1": "550303c0d43d246dc5db36f2fda48d16fdd5f85ae54739741801e6da26598517",
  "p16-dswd-shared-v1-airbase-rf-s46-r1": "8a1dee3168a24e8b90bbe4d4fc58dc737ed19cc834feaa5e0f181f1227ee8a92",
  "p16-dswd-shared-v1-santa-maria-rf-s42-r1": "621323094a69a0cb30108d04577f03df8d7af84decc3e7a0a858c16013e8914a",
  "p16-dswd-shared-v1-santa-maria-rf-s43-r1": "027e09da8a0460c81c4f440b082dfa5d5fc4701ba07968f27e459dd57ecb1ee0",
  "p16-dswd-shared-v1-santa-maria-rf-s44-r1": "aad3b6a724c6ca80e1212d4ccf48c3d424be99c0432e02a6de165fd6aec2c6f4",
  "p16-dswd-shared-v1-santa-maria-rf-s45-r1": "285f12fc9bff787c6383e20587bbcb30c1deb0da127975a120e9a8eea4775481",
  "p16-dswd-shared-v1-santa-maria-rf-s46-r1": "6f319b509a26a563368971535e3360e8db18dc6c5fbb6a86cef20db3d73e4ce3",
  "p16-dswd-shared-v1-santa-maria-lr-s42-r1": "f976eb988d2d73e62e84feac0f2944b8627a37044d93032a29e334529251ac6c",
  "p16-dswd-shared-v1-santa-maria-lr-s43-r1": "9ed39fdff53da74bcbd2c5d101419ff308ee7df9e7ac60bd243e3a4f564188a7",
  "p16-dswd-shared-v1-santa-maria-lr-s44-r1": "ee87ee0fe4c8e43428eb44ee65f58c371e94b4a87793858f67602841cb69353a",
  "p16-dswd-shared-v1-santa-maria-lr-s45-r1": "0e3f3a2a01cb1684c122549c260175e00783753bf814f573ea23effd1cc1e32b",
  "p16-dswd-shared-v1-santa-maria-lr-s46-r1": "82cde436239cfad909c01f1f4317bfe5781f113ea8ebd5def64b38a11fea2971",
  "p16-dswd-shared-v1-airbase-rf-s42-r2": "47fe1b0b688582c6ba0bac0972e44068c698cb4433d10ef7f6d097786b565599",
  "p16-dswd-shared-v1-airbase-lr-s42-r2": "423641e5fc71718899d9dec953d27daf46ca0d4294838103cb8028553e347378",
  "p16-dswd-shared-v1-airbase-lr-s43-r2": "4a1ebf6403a3202983e8e0d87dc8c357ffab1b8043f83c6daf2ef9146a2f15b0",
  "p16-dswd-shared-v1-airbase-lr-s44-r2": "f2f5a3f5ac3f79b6c184763eaf5df457f98bc5f2ac2994a68ce334f37ba2fd54",
  "p16-dswd-shared-v1-airbase-lr-s45-r2": "fd465e1444f22a4212eb3fcfd7dcb18c5ebd80ad08d1e781897b4991a63732e3",
  "p16-dswd-shared-v1-airbase-lr-s46-r2": "e2ec263ddd414bde152e204b23d250950b319ac8112e13e57c3fb6cd5ace071d"
}
'@
$selection=$configPins.PSObject.Properties[$RunId]
if ($null -eq $selection -or $RunId -cnotmatch ('-'+$Arm+'-s[0-9]+-r[12]$')) {throw 'Unselected trajectory/arm.'}
$configSha=[string]$selection.Value
$configPath=Join-Path $code ('config\phase16_dswd_shared_ca_exec_v1\'+$RunId+'.yaml')
$body=@'
# D-338 production caller. No execution approval is embedded in this source.
import os, sys, time, json, hashlib, stat, io, struct, math, warnings, re, traceback
from pathlib import Path
REPO = Path("C:/Users/gibgib/Thesis-Cellular-Automata")
CODE = REPO / "Thesis_RF/Code"
VENV = REPO / ".venv"
SITE = VENV / "Lib/site-packages"
BASE = Path("C:/Python314")
EXE = VENV / "Scripts/python.exe"
E = Path(os.environ["RF_A_ROOT"])
AUTH = os.environ["RF_A_APPROVAL"]
PARENT = int(os.environ["RF_A_PARENT"])
BODY_SHA = os.environ["RF_A_BODY_SHA"]
LAUNCHER_SHA = os.environ["RF_A_LAUNCHER_SHA"]
RUN_ID, ARM = os.environ["DSWD_RUN_ID"], os.environ["DSWD_ARM"]
EXPECTED_BINDING = os.environ["DSWD_EXPECTED_BINDING"]
CALLER = CODE / "run_dswd_shared_ca_v1.ps1"
CONFIG_PATH = CODE / "config/phase16_dswd_shared_ca_exec_v1" / (RUN_ID + ".yaml")
FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
ATTEMPTED = set()
receipts, reads, gates, pre, post = {}, [], [], {}, {}
counts = {"source_hash_success": 0, "journal_read_hash_success": 0}
state = {"dispatch_attempts": 0, "dispatch_completions": 0, "warnings": [],
         "guard_entry": False, "guard_completion": False, "guard_removed": False}
binding, runtime, config_pre, config_post, caller_pre, caller_post = None, {}, None, None, None, None
stage, error, post_error, complete = "BOOTSTRAP", None, None, False
started = time.monotonic()
PINS = {
    "C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/orchestrator.py": "38e3cb58144df53a91a7f85d1f8ea3eb279730a9220935d63b0bb873fc2b2809",
    "C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/dswd_inference_guard.py": "2e69f5a50adc1ca6f69cde0f8bca7758554786bc6ddc3b62feb584b0152f7578",
    "C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/modules/feature_pipeline.py": "b404e82cac952f3111defedf857ecf0b7891214afe4f0876a3bf7617680daeb1",
    "C:/Users/gibgib/Thesis-Cellular-Automata/.venv/Lib/site-packages/sklearn/calibration.py": "4c295444ea50cdd672682cc72e07110abb68c3ce81e02998cec9372449864f35",
    "C:/Users/gibgib/Thesis-Cellular-Automata/.venv/Lib/site-packages/sklearn/preprocessing/_label.py": "a5200ac488bf8b85440783cb3ffd500d805c18e12c2dd86483985dbc5ebc29be",
    "C:/Users/gibgib/Thesis-Cellular-Automata/.venv/Lib/site-packages/sklearn/utils/validation.py": "4e489efc8586dd3e896e6443bffc3620727037b63417d5d9bd3da0529329d51a",
    "C:/Users/gibgib/Thesis-Cellular-Automata/.venv/Lib/site-packages/sklearn/ensemble/_forest.py": "4dd334514b7551590a049548570b824a9cd91bc444959af7a1ac20a104116702"
}
CONFIG_PINS = {
    "p16-dswd-shared-v1-airbase-rf-s43-r1": "404a90489188a1d8f4810676b0f3e533cbc36ee47dbb4f5fe8787affd39af53d",
    "p16-dswd-shared-v1-airbase-rf-s44-r1": "8953cfb837e4e2eda53f363711dda3df48ad8b8d0f89ac45e66e15302337bf3f",
    "p16-dswd-shared-v1-airbase-rf-s45-r1": "550303c0d43d246dc5db36f2fda48d16fdd5f85ae54739741801e6da26598517",
    "p16-dswd-shared-v1-airbase-rf-s46-r1": "8a1dee3168a24e8b90bbe4d4fc58dc737ed19cc834feaa5e0f181f1227ee8a92",
    "p16-dswd-shared-v1-santa-maria-rf-s42-r1": "621323094a69a0cb30108d04577f03df8d7af84decc3e7a0a858c16013e8914a",
    "p16-dswd-shared-v1-santa-maria-rf-s43-r1": "027e09da8a0460c81c4f440b082dfa5d5fc4701ba07968f27e459dd57ecb1ee0",
    "p16-dswd-shared-v1-santa-maria-rf-s44-r1": "aad3b6a724c6ca80e1212d4ccf48c3d424be99c0432e02a6de165fd6aec2c6f4",
    "p16-dswd-shared-v1-santa-maria-rf-s45-r1": "285f12fc9bff787c6383e20587bbcb30c1deb0da127975a120e9a8eea4775481",
    "p16-dswd-shared-v1-santa-maria-rf-s46-r1": "6f319b509a26a563368971535e3360e8db18dc6c5fbb6a86cef20db3d73e4ce3",
    "p16-dswd-shared-v1-santa-maria-lr-s42-r1": "f976eb988d2d73e62e84feac0f2944b8627a37044d93032a29e334529251ac6c",
    "p16-dswd-shared-v1-santa-maria-lr-s43-r1": "9ed39fdff53da74bcbd2c5d101419ff308ee7df9e7ac60bd243e3a4f564188a7",
    "p16-dswd-shared-v1-santa-maria-lr-s44-r1": "ee87ee0fe4c8e43428eb44ee65f58c371e94b4a87793858f67602841cb69353a",
    "p16-dswd-shared-v1-santa-maria-lr-s45-r1": "0e3f3a2a01cb1684c122549c260175e00783753bf814f573ea23effd1cc1e32b",
    "p16-dswd-shared-v1-santa-maria-lr-s46-r1": "82cde436239cfad909c01f1f4317bfe5781f113ea8ebd5def64b38a11fea2971",
    "p16-dswd-shared-v1-airbase-rf-s42-r2": "47fe1b0b688582c6ba0bac0972e44068c698cb4433d10ef7f6d097786b565599",
    "p16-dswd-shared-v1-airbase-lr-s42-r2": "423641e5fc71718899d9dec953d27daf46ca0d4294838103cb8028553e347378",
    "p16-dswd-shared-v1-airbase-lr-s43-r2": "4a1ebf6403a3202983e8e0d87dc8c357ffab1b8043f83c6daf2ef9146a2f15b0",
    "p16-dswd-shared-v1-airbase-lr-s44-r2": "f2f5a3f5ac3f79b6c184763eaf5df457f98bc5f2ac2994a68ce334f37ba2fd54",
    "p16-dswd-shared-v1-airbase-lr-s45-r2": "fd465e1444f22a4212eb3fcfd7dcb18c5ebd80ad08d1e781897b4991a63732e3",
    "p16-dswd-shared-v1-airbase-lr-s46-r2": "e2ec263ddd414bde152e204b23d250950b319ac8112e13e57c3fb6cd5ace071d"
}
VERSIONS = {"numpy":"2.4.4","scipy":"1.17.1","pandas":"3.0.2","scikit-learn":"1.8.0",
    "joblib":"1.5.3","python-dateutil":"2.9.0.post0","rasterio":"1.5.0","PyYAML":"6.0.3"}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)
def lex(path):
    return os.path.normcase(os.path.abspath(os.fspath(path)))
def sig(info):
    return {key:getattr(info,key) for key in FIELDS}
def no_reparse(path):
    path = Path(os.path.abspath(os.fspath(path)))
    for item in reversed((path,*path.parents)):
        info = os.lstat(item)
        require(not stat.S_ISLNK(info.st_mode)
                and not (getattr(info,"st_file_attributes",0)&0x400),"Reparse component")
    return path
def read_fixed(path, cap, expected=None, keep=False, category=None):
    path = no_reparse(path)
    observation = {"path":str(path),"category":category,"requested":None,"returned":0}
    reads.append(observation)
    a = os.lstat(path)
    observation["path_before"] = sig(a)
    require(stat.S_ISREG(a.st_mode) and 0<=a.st_size<=cap,"File type/initial length")
    digest = hashlib.sha256()
    retained = io.BytesIO() if keep else None
    try:
        with open(path,"rb",buffering=0) as stream:
            b = os.fstat(stream.fileno())
            observation["handle_before"] = sig(b)
            require(stat.S_ISREG(b.st_mode) and all(getattr(a,k)==getattr(b,k)
                    for k in FIELDS[:4]),"Path/handle identity")
            remaining = a.st_size
            observation["requested"] = remaining
            while remaining:
                part = stream.read(min(65536,remaining))
                observation["returned"] += len(part)
                require(bool(part),"Premature EOF")
                digest.update(part)
                if retained is not None:
                    require(retained.write(part)==len(part),"Buffer short write")
                remaining -= len(part)
            c = os.fstat(stream.fileno())
            observation["handle_after"] = sig(c)
        d = os.lstat(path)
        observation["path_after"] = sig(d)
        require(sig(a)==sig(d) and sig(b)==sig(c),"Within-API temporal instability")
        require(all(getattr(c,k)==getattr(d,k) for k in FIELDS[:4]),"Final path/handle identity")
        observation["cross_api_differences"] = [k for k in FIELDS if getattr(a,k)!=getattr(b,k)]
        observation["sha256"] = digest.hexdigest()
        require(observation["returned"]==a.st_size,"Read length")
        require(expected is None or digest.hexdigest()==expected,"SHA-256 mismatch")
        if category in ("source","protected"):
            counts[category+"_hash_success"] += 1
        if retained is not None:
            retained.seek(0)
        return retained,observation
    except BaseException:
        if retained is not None:
            retained.close()
        raise
def receipt(observation):
    return {"filename":Path(observation["path"]).name,
            "bytes":observation["returned"],"sha256":observation["sha256"]}
def publish(name,value,cap=65536):
    require(name in ("worker-identity.json","caller.json","python-status.json"),"Publication name")
    require(name not in ATTEMPTED,"Publication already attempted")
    ATTEMPTED.add(name)
    payload = json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False,
                         separators=(",",":")).encode("utf-8")
    require(len(payload)<=cap,"Publication cap")
    no_reparse(E)
    try:
        os.lstat(E/name)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError("Publication collision")
    with open(E/name,"xb",buffering=0) as stream:
        require(stream.write(payload)==len(payload),"Publication short write")
        os.fsync(stream.fileno())
    _,observed = read_fixed(E/name,cap,hashlib.sha256(payload).hexdigest())
    require(observed["returned"]==len(payload),"Publication readback length")
    receipts[name] = receipt(observed)
    return receipts[name]
def marker(name):
    require(name=="worker.ready" and name not in ATTEMPTED,"Marker name/attempt")
    ATTEMPTED.add(name)
    no_reparse(E)
    with open(E/name,"xb",buffering=0) as stream:
        os.fsync(stream.fileno())
def marker_present(name):
    require(name=="admission.ready","Marker name")
    try:
        value = os.lstat(E/name)
    except FileNotFoundError:
        return False
    require(stat.S_ISREG(value.st_mode) and value.st_size==0
            and not(getattr(value,"st_file_attributes",0)&0x400),"Invalid ready marker")
    return True
def evaluate_sample(sample):
    age = time.time()*1000-sample["utc_ms"]
    tests = {
        "role":sample.get("role")=="verified_worker",
        "parent_pid":type(sample.get("parent_pid")) is int and sample["parent_pid"]==PARENT,
        "worker_pid":type(sample.get("child_pid")) is int and sample["child_pid"]==os.getpid(),
        "ok":type(sample.get("ok")) is bool and sample["ok"] is True,
        "fresh":type(sample["utc_ms"]) is int and 0<=age<=5000,
        "rss":type(sample.get("child_rss_bytes")) is int
              and 0<=sample["child_rss_bytes"]<=4294967296,
        "available":type(sample.get("system_available_bytes")) is int
              and sample["system_available_bytes"]>=2147483648,
        "free":type(sample.get("evidence_free_bytes")) is int
              and sample["evidence_free_bytes"]>=268435456,
        "elapsed":type(sample.get("elapsed_ms")) in (int,float)
              and math.isfinite(sample["elapsed_ms"]) and 0<=sample["elapsed_ms"]<=1920000,
        "live":sample.get("process_has_exited") is False,
    }
    return {"sample":sample,"age_ms":age,"checks":tests,"pass":all(tests.values())}
def resource_gate(name):
    record = {"stage":name,"requested":None,"returned":None,"complete":False}
    gates.append(record)
    path = no_reparse(E/"resources.partial.jsonl")
    a = os.lstat(path)
    require(stat.S_ISREG(a.st_mode) and 0<a.st_size<=8388608,"Journal type/length")
    with open(path,"rb",buffering=0) as stream:
        b = os.fstat(stream.fileno())
        require(all(getattr(a,k)==getattr(b,k) for k in FIELDS[:4]),"Journal open identity")
        record["requested"] = a.st_size
        data = stream.read(a.st_size)
        record["returned"] = len(data)
        require(len(data)==a.st_size,"Incomplete journal acquisition")
        c = os.fstat(stream.fileno())
    d = os.lstat(path)
    record["observations"] = {"path_before":sig(a),"handle_before":sig(b),
                             "handle_after":sig(c),"path_after":sig(d)}
    require(c.st_dev==b.st_dev and c.st_ino==b.st_ino and c.st_size>=b.st_size
            and d.st_dev==a.st_dev and d.st_ino==a.st_ino and d.st_size>=a.st_size,
            "Append-only journal identity/length")
    require(data.endswith(b"\n"),"Incomplete journal final record")
    record["sha256"] = hashlib.sha256(data).hexdigest()
    values = [json.loads(line) for line in data.splitlines()]
    selected = [v for v in values if v.get("role")=="verified_worker"]
    require(bool(selected),"No verified-worker sample")
    record["admission"] = evaluate_sample(selected[-1])
    require(record["admission"]["pass"],"Resource admission failed")
    record["complete"] = True
    counts["journal_read_hash_success"] += 1
def source_phase(target):
    for path,digest in PINS.items():
        _,target[path] = read_fixed(path,1048576,digest,category="source")

# BEGIN DSWD CALLER CONTRACT
SELECTED_RUNS = frozenset(
    ["p16-dswd-shared-v1-airbase-rf-s42-r2"]
    + [f"p16-dswd-shared-v1-airbase-lr-s{s}-r2" for s in range(42, 47)]
    + [f"p16-dswd-shared-v1-airbase-rf-s{s}-r1" for s in range(43, 47)]
    + [f"p16-dswd-shared-v1-santa-maria-{a}-s{s}-r1"
       for a in ("rf", "lr") for s in range(42, 47)])
BOUND_REFERENCE = "p16-dswd-shared-ca-v1-execution-r1"
WARNING_LIMIT = 8192


def selected_dispatch(runner, guard, features, config, config_path, *, run_id, arm,
                      authority, execute, expected_config_sha256, disk_sha256,
                      expected_binding_sha256, launcher_sha256, state):
    """Actual one-attempt production dispatch; tests extract this exact block."""
    require(execute is True and type(authority) is str
            and re.fullmatch(r"D-[0-9]+", authority) is not None
            and int(authority[2:]) > 338, "Fresh owner execution authority required")
    require(os.environ.get("JOBLIB_MULTIPROCESSING") == "0", "Serialization environment")
    require(sys.getprofile() is None, "Pre-existing profiler")
    require(runner._phase16_is_shared(config), "Shared trajectory configuration required")
    require(disk_sha256 == expected_config_sha256, "Configuration identity")
    require(config["phase15"]["enabled"] is False, "Disk configuration must be disabled")
    active = runner._phase15_normalize(config)
    actual_arm = active["snapshot_basis"]["arm"]
    require(actual_arm in ("rf", "lr") and arm == actual_arm, "Selected arm mismatch")
    require(run_id in SELECTED_RUNS and active["phase15"]["run_id"] == run_id,
            "Unselected trajectory")
    require(active["phase15"]["authorization_reference"] == BOUND_REFERENCE,
            "Prospective configuration reference")
    require(Path(config_path).name == run_id + ".yaml", "Configuration filename")
    runner._phase15_digest(expected_binding_sha256)
    binding = runner._phase15_binding(active, config_path, disk_sha256)
    require(runner.canonical_metadata_hash(binding) == expected_binding_sha256,
            "Independent binding SHA-256 mismatch")
    caller_key = "Thesis_RF/Code/run_dswd_shared_ca_v1.ps1"
    require(binding["source"]["files"].get(caller_key) == launcher_sha256,
            "Invoked caller differs from execution binding")
    state.update(binding_pre=binding, actual_arm=actual_arm, dispatch_attempts=0,
                 dispatch_completions=0, guard_entry=False, guard_completion=False,
                 guard_removed=False, warnings=[])
    callback = guard.make_inference_guard(runner.PHASE15_CODE_DIR, arm=actual_arm)
    held = (callback.__code__, guard.make_inference_guard,
            guard.make_inference_guard.__code__, guard._temporary_label_encoding_fit,
            guard._temporary_label_encoding_fit.__code__, features.predict_positive_probability,
            features.predict_positive_probability.__code__)

    def intact():
        return (sys.getprofile() is callback and callback.__code__ is held[0]
                and guard.make_inference_guard is held[1]
                and guard.make_inference_guard.__code__ is held[2]
                and guard._temporary_label_encoding_fit is held[3]
                and guard._temporary_label_encoding_fit.__code__ is held[4]
                and features.predict_positive_probability is held[5]
                and features.predict_positive_probability.__code__ is held[6])

    def observed_warning(message, category, filename, lineno, file=None, line=None):
        record = {"category": category.__module__ + "." + category.__name__,
                  "message": str(message)[:2048], "message_truncated": len(str(message)) > 2048,
                  "filename": str(filename),
                  "lineno": lineno, "arm": actual_arm, "accepted": False}
        require(len(state["warnings"]) < WARNING_LIMIT, "Warning observation overflow")
        state["warnings"].append(record)
        require(not record["message_truncated"], "Warning message cap")
        frame, helper_seen = sys._getframe(), False
        while frame is not None:
            if (frame.f_code is held[6] and frame.f_globals is features.__dict__):
                helper_seen = True
                break
            frame = frame.f_back
        estimator = "CalibratedClassifierCV" if actual_arm == "rf" else "LogisticRegression"
        expected = "X does not have valid feature names, but " + estimator + " was fitted with feature names"
        require(category is UserWarning and str(message) == expected
                and lex(filename) == lex(SITE / "sklearn/utils/validation.py")
                and lineno == 2691 and helper_seen and intact(), "Unexpected production warning")
        record["accepted"] = True
        sys.stderr.write("DSWD_WARNING " + json.dumps(record, sort_keys=True) + "\n")
        sys.stderr.flush()

    sys.setprofile(callback)
    try:
        require(intact(), "Canonical callback missing at entry")
        state["guard_entry"] = True
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = observed_warning
            state["dispatch_attempts"] += 1
            runner.run_simulation(config, explicitly_authorized=True,
                approval_reference=BOUND_REFERENCE, expected_binding_sha256=expected_binding_sha256,
                config_path=config_path)
            state["dispatch_completions"] += 1
        require(intact(), "Canonical callback lost or changed")
        state["guard_completion"] = True
    finally:
        # Never reinstall a lost callback or restore acceptance after any exception.
        if sys.getprofile() is callback:
            sys.setprofile(None)
            state["guard_removed"] = sys.getprofile() is None
    require(state["guard_removed"], "Callback cleanup failed")
    state["binding_post"] = runner._phase15_binding(active, config_path, disk_sha256)
    require(runner._phase15_json_bytes(state["binding_post"]) == runner._phase15_json_bytes(binding),
            "Source/runtime/execution binding changed")
    return active
# END DSWD CALLER CONTRACT

try:
    require(os.name=="nt" and sys.version_info[:3]==(3,14,3),"Python basis")
    require(lex(sys.executable)==lex(EXE) and lex(sys.prefix)==lex(VENV)
        and lex(sys.base_prefix)==lex(BASE)
        and lex(getattr(sys,"_base_executable",""))==lex(BASE/"python.exe"),"Interpreter locations")
    require(all((sys.flags.isolated,sys.flags.no_site,sys.flags.dont_write_bytecode,
                 sys.flags.utf8_mode)) and struct.calcsize("P")==8,"Interpreter flags/width")
    require(sys.implementation.name=="cpython" and sys.implementation.cache_tag=="cpython-314"
            and sys.byteorder=="little","Implementation basis")
    identity = {"schema":"dswd_shared_worker.v1","approval":AUTH,"root":str(E),
        "pid":os.getpid(),"ppid":os.getppid(),"parent_pid":PARENT,
        "executable":sys.executable,"base_executable":sys._base_executable,
        "prefix":sys.prefix,"base_prefix":sys.base_prefix,
        "version_info":list(sys.version_info[:3]),"pointer_bits":64,
        "flags":[sys.flags.isolated,sys.flags.no_site,sys.flags.dont_write_bytecode,sys.flags.utf8_mode],
        "body_sha256":BODY_SHA,"launcher_sha256":LAUNCHER_SHA,
        "utc_ms":int(time.time()*1000),"production_authorized":False}
    identity.update(run_id=RUN_ID, arm=ARM, expected_binding_sha256=EXPECTED_BINDING)
    worker_receipt = publish("worker-identity.json",identity)
    marker("worker.ready")
    while not marker_present("admission.ready"):
        require(time.monotonic()-started<=1920,"Handshake deadline")
        time.sleep(0.05)
    data,admission_observation = read_fixed(E/"admission-input.json",65536,keep=True)
    try:
        binding = json.loads(data.getvalue())
    finally:
        data.close()
    require(binding["schema"]=="dswd_shared_process_binding.v1" and binding["approval"]==AUTH
        and binding["worker_pid"]==os.getpid() and type(binding["worker_pid"]) is int
        and binding["worker_ppid"]==os.getppid() and binding["parent_pid"]==PARENT
        and binding["root"]==str(E) and binding["worker_identity_receipt"]==worker_receipt
        and binding["body_sha256"]==BODY_SHA and binding["launcher_sha256"]==LAUNCHER_SHA,
        "Admission identity/receipt conflict")
    require(binding["run_id"] == RUN_ID and binding["arm"] == ARM
            and binding["expected_binding_sha256"] == EXPECTED_BINDING, "Admission selection")
    require(type(binding["checks"]) is dict and len(binding["checks"])==7
            and all(type(v) is bool and v for v in binding["checks"].values()),"Binding checks")
    initial_admission = evaluate_sample(binding["worker_sample"])
    require(initial_admission["pass"],"Initial worker admission")
    stage = "SOURCE_PRE"
    resource_gate("BEFORE_SOURCE")
    source_phase(pre)
    _, caller_pre = read_fixed(CALLER, 1048576, LAUNCHER_SHA)
    require(RUN_ID in SELECTED_RUNS and RUN_ID in CONFIG_PINS, "Unselected trajectory")
    data, config_pre = read_fixed(CONFIG_PATH, 1048576, CONFIG_PINS[RUN_ID], keep=True)
    try:
        config_bytes = data.getvalue()
    finally:
        data.close()
    stage = "ORDINARY_INITIALIZATION"
    # Executable ordinary library initialization is explicitly declared, not sandboxed.
    sys.path.insert(0, str(CODE))
    sys.path.append(str(SITE))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        import importlib.metadata
        import numpy, scipy, pandas, rasterio, sklearn, joblib, yaml
        import sklearn.calibration as calibration
        import sklearn.preprocessing._label as labels
        import sklearn.utils.validation as validation
        import sklearn.ensemble._forest as forest
        import orchestrator as runner
        import dswd_inference_guard as guard
        import modules.feature_pipeline as features
        actual_versions = {k: importlib.metadata.version(k) for k in VERSIONS}
        require(actual_versions == VERSIONS,
                "Package version mismatch")
        require(rasterio.__gdal_version__ == "3.12.1", "GDAL version mismatch")
        for module, path in ((runner, CODE/"orchestrator.py"),
                (guard, CODE/"dswd_inference_guard.py"),
                (features, CODE/"modules/feature_pipeline.py"),
                (calibration, SITE/"sklearn/calibration.py"),
                (labels, SITE/"sklearn/preprocessing/_label.py"),
                (validation, SITE/"sklearn/utils/validation.py"),
                (forest, SITE/"sklearn/ensemble/_forest.py")):
            require(lex(module.__file__) == lex(path), "Imported enforcement source location")
        for module, folder in ((numpy,"numpy"),(scipy,"scipy"),(pandas,"pandas"),
                (rasterio,"rasterio"),(sklearn,"sklearn"),(joblib,"joblib"),(yaml,"yaml")):
            require(lex(module.__file__).startswith(lex(SITE/folder)+os.sep), "Package location")
        runtime = {"execution": runner._phase15_runtime(),
                   "versions": actual_versions, "gdal": rasterio.__gdal_version__,
                   "joblib_multiprocessing": os.environ.get("JOBLIB_MULTIPROCESSING"),
                   "ordinary_initialization": True, "sandbox": False,
                   "n_jobs_policy": "Frozen stored parameters unchanged; no two-worker claim"}
        config = runner._phase15_yaml(config_bytes)
        stage = "GUARDED_TRAJECTORY"
        resource_gate("BEFORE_DISPATCH")
        active = selected_dispatch(runner, guard, features, config, CONFIG_PATH,
            run_id=RUN_ID, arm=ARM, authority=AUTH, execute=os.environ.get("DSWD_EXECUTE") == "1",
            expected_config_sha256=CONFIG_PINS[RUN_ID], disk_sha256=config_pre["sha256"],
            expected_binding_sha256=EXPECTED_BINDING, launcher_sha256=LAUNCHER_SHA, state=state)
        require(config["phase15"]["enabled"] is False, "In-memory disk configuration changed")
    stage = "RUN_RECEIPTS"
    # Inert fixed-name records only; native output acceptance is a later independent read.
    destination = Path(state["binding_pre"]["run"]["destination"])
    run_records = {}
    run_receipts = {}
    for name in ("start.json", "termination.json"):
        data, observed = read_fixed(destination/name, 8388608, keep=True)
        try:
            run_records[name] = json.loads(data.getvalue())
        finally:
            data.close()
        run_receipts[name] = receipt(observed)
    start_record, end = run_records["start.json"], run_records["termination.json"]
    require(os.path.samefile(destination/"termination.json", destination/"termination.json.partial"),
            "Final termination publication link")
    require(start_record["binding"]["payload"] == state["binding_pre"]
            and start_record["binding"]["calculated_sha256"] == EXPECTED_BINDING
            and start_record["binding"]["expected_sha256"] == EXPECTED_BINDING
            and end["binding_sha256"] == EXPECTED_BINDING
            and end["start_record_sha256"] == run_receipts["start.json"]["sha256"]
            and start_record["run_id"] == RUN_ID and end["run_id"] == RUN_ID,
            "Runtime record binding")
    require(end["exit_code"] == 0 and type(end["exit_code"]) is int
            and end["output_valid"] is True and end["error"] is None and not end["violations"]
            and end["termination_reason"] in ("inactive", "timestep_limit"), "Runtime failed/incomplete")
    state["run_receipts"] = run_receipts
    state["termination"] = {k: end[k] for k in ("status", "termination_reason", "last_completed_timestep")}
    complete = True
except BaseException as exc:
    error = {"stage": stage, "type": type(exc).__name__, "message": str(exc)[:8192]}
    try:
        traceback.print_exc(file=sys.stderr)
    except BaseException:
        pass
finally:
    try:
        source_phase(post)
        _, config_post = read_fixed(CONFIG_PATH, 1048576, CONFIG_PINS[RUN_ID])
        _, caller_post = read_fixed(CALLER, 1048576, LAUNCHER_SHA)
        require(pre.keys() == post.keys() and all(pre[p]["sha256"] == post[p]["sha256"] for p in pre),
                "Caller enforcement source drift")
        require(config_pre is not None and config_pre["sha256"] == config_post["sha256"]
                and caller_pre is not None and caller_pre["sha256"] == caller_post["sha256"],
                "Configuration/caller integrity")
    except BaseException as exc:
        complete = False
        post_error = {"type": type(exc).__name__, "message": str(exc)[:8192]}
report = {"schema": "dswd_shared_caller.v1", "complete": complete, "error": error,
    "post_error": post_error, "approval": AUTH, "configuration_reference": BOUND_REFERENCE,
    "run_id": RUN_ID, "arm": ARM, "expected_binding_sha256": EXPECTED_BINDING,
    "worker_pid": os.getpid(), "body_sha256": BODY_SHA, "launcher_sha256": LAUNCHER_SHA,
    "process_binding": binding, "admission_receipt": receipt(admission_observation) if binding else None,
    "runtime": runtime, "source_pre": pre, "source_post": post,
    "config_pre": config_pre, "config_post": config_post,
    "caller_pre": caller_pre, "caller_post": caller_post,
    "counts": counts, "state": state, "reads": reads, "resource_gates": gates,
    "paired_acceptance": "PENDING_INDEPENDENT_NATIVE_READERS",
    "scoring_performed": False, "compatibility_inference": False}
# Each filename has one publication attempt; failures never restart this site.
try:
    report_receipt = publish("caller.json", report, 8388608)
    status = {"schema": "dswd_shared_python.v1", "complete": complete,
        "approval": AUTH, "run_id": RUN_ID, "arm": ARM, "worker_pid": os.getpid(),
        "body_sha256": BODY_SHA, "launcher_sha256": LAUNCHER_SHA,
        "expected_binding_sha256": EXPECTED_BINDING, "report_receipt": report_receipt,
        "dispatch_attempts": state["dispatch_attempts"], "dispatch_completions": state["dispatch_completions"],
        "guard_entry": state["guard_entry"], "guard_completion": state["guard_completion"],
        "guard_removed": state["guard_removed"], "warning_count": len(state["warnings"]), "counts": counts}
    publish("python-status.json", status)
    print("DSWD_FINISHED " + json.dumps({"schema": "dswd_shared_finished.v1",
        "complete": complete, "receipts": receipts}, sort_keys=True), flush=True)
except BaseException:
    try:
        os.write(2, b"Caller publication/print failure; retain partials; no retry.\n")
    except BaseException:
        pass
    raise SystemExit(1)
raise SystemExit(0 if complete else 1)
'@
$body=$body.Replace(([string][char]13+[char]10),[string][char]10).TrimEnd([char[]]@([char]13,[char]10))
$bodyBytes=$utf8.GetBytes($body)
$bodySha=Hash-Bytes $bodyBytes
$child=$null;$report=$null;$finish=$null;$captures=@{};$captureBytes=@{};$finalReceipts=@{}
$journalReceipt=$null;$journalBytes=$null
$candidate=$false
try {
    No-Reparse $launcherPath;No-Reparse $python;No-Reparse $basePython
    $parentDir=[IO.Path]::GetDirectoryName($root)
    No-Reparse $parentDir
    try {$null=Get-Item -Force -LiteralPath $root -ErrorAction Stop;throw 'Fresh root collision.'}
    catch [System.Management.Automation.ItemNotFoundException] {}
    $null=[IO.Directory]::CreateDirectory($root)
    No-Reparse $root
    foreach ($name in @('tmp','cache','pycache')) {$null=[IO.Directory]::CreateDirectory((Join-Path $root $name))}
    [Console]::WriteLine('Retain all evidence: '+$root)
    $selfBytes=Read-Bytes $launcherPath 1048576
    $configPreBytes=Read-Bytes $configPath 1048576
    $s.configPre=[ordered]@{path=$configPath;bytes=$configPreBytes.Length;sha256=(Hash-Bytes $configPreBytes)}
    if ($s.configPre.sha256 -cne $configSha) {throw 'Selected configuration identity.'}
    $s.launcherPre=[ordered]@{path=$launcherPath;bytes=$selfBytes.Length;sha256=(Hash-Bytes $selfBytes)}
    if ($s.launcherPre.sha256 -cne $ReviewedLauncherSha256) {throw 'Reviewed launcher identity mismatch.'}
    $selfText=$utf8.GetString($selfBytes)
    $opening='$body=@'+[char]39+[char]13+[char]10
    $closing=[string][char]13+[char]10+[char]39+'@'
    $start=$selfText.IndexOf($opening,[StringComparison]::Ordinal)
    if ($start -lt 0 -or $start -ne $selfText.LastIndexOf($opening,[StringComparison]::Ordinal)) {throw 'Body delimiter identity.'}
    $start+=$opening.Length
    $end=$selfText.IndexOf($closing,$start,[StringComparison]::Ordinal)
    if ($end -lt $start) {throw 'Body delimiter termination.'}
    $diskBody=$selfText.Substring($start,$end-$start).Replace(([string][char]13+[char]10),[string][char]10)
    if ((Hash-Bytes $utf8.GetBytes($diskBody)) -cne $bodySha) {throw 'Delivered body differs from reviewed caller bytes.'}
    $selfBytes=$null
    $s.journal=[IO.File]::Open((Join-Path $root 'resources.partial.jsonl'),
        [IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    $assembly=[Reflection.Assembly]::Load(
        'Microsoft.VisualBasic, Version=10.0.0.0, Culture=neutral, PublicKeyToken=b03f5f7f11d50a3a')
    $memoryInfo=[Activator]::CreateInstance($assembly.GetType('Microsoft.VisualBasic.Devices.ComputerInfo',$true))
    $drive=[IO.DriveInfo]::new([IO.Path]::GetPathRoot($root))
    $null=Sample $null 'host_admission'
    $info=[Diagnostics.ProcessStartInfo]::new()
    $info.FileName=$python
    $info.Arguments='-B -I -S -X utf8 -'
    $info.UseShellExecute=$false;$info.CreateNoWindow=$true
    $info.WorkingDirectory=$repo
    $info.RedirectStandardInput=$true;$info.RedirectStandardOutput=$true;$info.RedirectStandardError=$true
    $info.EnvironmentVariables['DSWD_RUN_ID']=$RunId
    $info.EnvironmentVariables['DSWD_ARM']=$Arm
    $info.EnvironmentVariables['DSWD_EXPECTED_BINDING']=$ExpectedBindingSha256
    $info.EnvironmentVariables['DSWD_EXECUTE']='1'
    $info.EnvironmentVariables['RF_A_ROOT']=$root
    $info.EnvironmentVariables['RF_A_APPROVAL']=$OwnerApproval
    $info.EnvironmentVariables['RF_A_PARENT']=[string]$PID
    $info.EnvironmentVariables['RF_A_BODY_SHA']=$bodySha
    $info.EnvironmentVariables['RF_A_LAUNCHER_SHA']=$ReviewedLauncherSha256
    foreach ($name in @('TEMP','TMP','TMPDIR')) {$info.EnvironmentVariables[$name]=(Join-Path $root 'tmp')}
    foreach ($name in @('XDG_CACHE_HOME','JOBLIB_TEMP_FOLDER')) {$info.EnvironmentVariables[$name]=(Join-Path $root 'cache')}
    $info.EnvironmentVariables['PYTHONPYCACHEPREFIX']=(Join-Path $root 'pycache')
    $info.EnvironmentVariables['PYTHONDONTWRITEBYTECODE']='1'
    $info.EnvironmentVariables['JOBLIB_MULTIPROCESSING']='0'
    foreach ($name in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')) {
        $info.EnvironmentVariables[$name]='1'
    }
    $info.EnvironmentVariables['PATH']=(Join-Path $repo '.venv\Scripts')+';C:\Windows\System32'
    $s.process=[Diagnostics.Process]::new();$s.process.StartInfo=$info
    if (-not $s.process.Start()) {throw 'Process.Start false.'}
    $s.started=$true;$null=$s.process.Handle
    $launcherStart=$s.process.StartTime.ToUniversalTime().Ticks
    $s.stdout=New-Pump $s.process.StandardOutput.BaseStream 'stdout.log'
    $s.stderr=New-Pump $s.process.StandardError.BaseStream 'stderr.log'
    $s.inputStream=$s.process.StandardInput.BaseStream
    $s.inputTask=$s.inputStream.WriteAsync($bodyBytes,0,$bodyBytes.Length)
    $null=Sample $s.process 'launcher'
    do {
        Tick
        if (-not $s.sent -and $s.inputTask.IsCompleted) {
            $null=$s.inputTask.GetAwaiter().GetResult()
            $s.inputStream.Flush()
            $s.inputClosed=$true
            $s.inputStream.Close()
            $s.sent=$true
        }
        if (-not $s.bound -and (Ready 'worker.ready')) {
            $idBytes=Read-Bytes (Join-Path $root 'worker-identity.json') 65536
            $id=ConvertFrom-Json -InputObject $utf8.GetString($idBytes)
            $s.identityReceipt=[ordered]@{filename='worker-identity.json';bytes=$idBytes.Length;sha256=(Hash-Bytes $idBytes)}
            if ($id.schema -cne 'dswd_shared_worker.v1' -or $id.approval -cne $OwnerApproval -or
                $id.root -cne $root -or -not(Is-Integer $id.pid) -or
                -not(Is-Integer $id.ppid) -or $id.pid -le 0 -or $id.pid -gt [int]::MaxValue -or
                $id.parent_pid -ne $PID -or $id.body_sha256 -cne $bodySha -or
                $id.launcher_sha256 -cne $ReviewedLauncherSha256 -or
                $id.executable -ine $python -or $id.base_executable -ine $basePython -or
                $id.prefix -ine (Join-Path $repo '.venv') -or $id.base_prefix -ine 'C:\Python314' -or
                $id.pointer_bits -ne 64 -or ($id.version_info -join '.') -cne '3.14.3' -or
                $id.flags.Count -ne 4 -or $id.production_authorized -ne $false) {throw 'Worker runtime identity.'}
            if ($id.run_id -cne $RunId -or $id.arm -cne $Arm -or
                $id.expected_binding_sha256 -cne $ExpectedBindingSha256) {throw 'Selected worker identity.'}
            foreach ($v in $id.flags) {if (-not(Is-Integer $v) -or $v -ne 1) {throw 'Worker startup flags.'}}
            if (-not(Is-Integer $id.utc_ms)) {throw 'Worker clock type.'}
            $age=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()-$id.utc_ms
            $workerId=[int]$id.pid;$same=($workerId -eq $s.process.Id)
            $expectedParent=[int]$s.process.Id
            if ($same) {$expectedParent=[int]$PID}
            $cp=$null
            try {
                if ($same) {$cp=$s.process} else {$cp=[Diagnostics.Process]::GetProcessById($workerId)}
                $null=$cp.Handle;$cp.Refresh()
                $image=$cp.MainModule.FileName
                $checks=[ordered]@{pid=($cp.Id -eq $workerId);live=(-not $cp.HasExited)
                    parentage=($id.ppid -eq $expectedParent)
                    start=($cp.StartTime.ToUniversalTime().Ticks -ge $launcherStart)
                    image=($image -ieq $basePython -or ($same -and $image -ieq $python))
                    runtime=$true;fresh=($age -ge 0 -and $age -le 5000)}
                if ($checks.Values -contains $false) {throw 'Held worker binding failed.'}
                $s.worker=$cp;$s.bound=$true
            } finally {if (-not $s.bound -and $null -ne $cp -and -not $same) {$cp.Dispose()}}
            $s.binding=[ordered]@{schema='dswd_shared_process_binding.v1';approval=$OwnerApproval;root=$root
                run_id=$RunId;arm=$Arm;expected_binding_sha256=$ExpectedBindingSha256
                parent_pid=[int]$PID;launcher_pid=[int]$s.process.Id;worker_pid=$workerId
                worker_ppid=$id.ppid;same_process=$same;worker_image=$image
                launcher_start_ticks=$launcherStart;worker_start_ticks=$s.worker.StartTime.ToUniversalTime().Ticks
                checks=$checks;worker_identity_receipt=$s.identityReceipt
                body_sha256=$bodySha;launcher_sha256=$ReviewedLauncherSha256
                launcher_sample=(Sample $s.process 'launcher')
                worker_sample=(Sample $s.worker 'verified_worker')}
            $s.admissionReceipt=Publish 'admission-input.json' $s.binding
            Marker 'admission.ready'
        }
        $ended=$s.process.WaitForExit(100)
    } while (-not $ended)
} catch {
    $s.errors.Add('Attempt: '+$_.ToString())
    Stop-Owned
} finally {
    if ($null -ne $s.inputStream -and -not $s.inputClosed) {
        $s.inputClosed=$true
        try {$s.inputStream.Close()} catch {$s.errors.Add('Input close: '+$_.ToString());Stop-Owned}
    }
    foreach ($role in @('launcher','worker')) {
        $p=if ($role -eq 'launcher') {$s.process} else {$s.worker}
        $owned=if ($role -eq 'launcher') {$s.started} else {$s.bound}
        if (-not $owned -or $null -eq $p) {continue}
        try {
            $waited=$p.WaitForExit(100)
            while (-not $waited) {
                try {Tick} catch {$s.errors.Add('Wait monitor: '+$_.ToString());Stop-Owned}
                if ($s.killRequested -and $clock.Elapsed.TotalMilliseconds-$s.killAt -ge 5000) {
                    $s.errors.Add('Owned wait incomplete after termination window: '+$role)
                    break
                }
                $waited=$p.WaitForExit(100)
            }
            if ($waited) {$s[$role+'Waited']=$true;$s[$role+'Exit']=[int]$p.ExitCode}
        } catch {$s.errors.Add('Wait/exit '+$role+': '+$_.ToString());Stop-Owned}
    }
    try {
        while (($null -ne $s.stdout -and -not $s.stdout.eof -and -not $s.stdout.overflow) -or
               ($null -ne $s.stderr -and -not $s.stderr.eof -and -not $s.stderr.overflow)) {
            try {Tick}
            catch {$s.errors.Add('Drain: '+$_.ToString());Stop-Owned;break}
            Start-Sleep -Milliseconds 20
        }
    } catch {$s.errors.Add('Final drain: '+$_.ToString());Stop-Owned}
    foreach ($p in @($s.stdout,$s.stderr)) {
        if ($null -eq $p) {continue}
        try {$p.output.Flush($true)}
        catch {$s.errors.Add('Capture flush: '+$_.ToString())}
        try {$p.output.Dispose()}
        catch {$s.errors.Add('Capture disposal: '+$_.ToString())}
        try {
            $captureBytes[$p.name]=Read-Bytes (Join-Path $root $p.name) 16777216
            $captures[$p.name]=[ordered]@{filename=$p.name;bytes=$captureBytes[$p.name].Length
                sha256=(Hash-Bytes $captureBytes[$p.name])}
        } catch {$s.errors.Add('Capture receipt: '+$_.ToString())}
    }
    $s.stdoutDone=($null -ne $s.stdout -and $s.stdout.eof -and -not $s.stdout.overflow)
    $s.stderrDone=($null -ne $s.stderr -and $s.stderr.eof -and -not $s.stderr.overflow)
    if ($null -ne $s.journal) {
        try {$s.journal.Flush($true)}
        catch {$s.errors.Add('Journal flush: '+$_.ToString())}
        try {$s.journal.Dispose()}
        catch {$s.errors.Add('Journal disposal: '+$_.ToString())}
    }
}
try {
    $journalBytes=Read-Bytes (Join-Path $root 'resources.partial.jsonl') 8388608
    $journalReceipt=[ordered]@{filename='resources.partial.jsonl';bytes=$journalBytes.Length;sha256=(Hash-Bytes $journalBytes)}
    $selfPost=Read-Bytes $launcherPath 1048576
    $s.launcherPost=[ordered]@{path=$launcherPath;bytes=$selfPost.Length;sha256=(Hash-Bytes $selfPost)}
    $configPost=Read-Bytes $configPath 1048576
    $s.configPost=[ordered]@{path=$configPath;bytes=$configPost.Length;sha256=(Hash-Bytes $configPost)}
    if ($s.launcherPost.sha256 -cne $ReviewedLauncherSha256 -or $s.launcherPost.bytes -ne $s.launcherPre.bytes -or
        $s.configPost.sha256 -cne $configSha -or $s.configPost.bytes -ne $s.configPre.bytes) {throw 'Parent integrity postcheck.'}
    $reportBytes=Read-Bytes (Join-Path $root 'caller.json') 8388608
    $report=ConvertFrom-Json -InputObject $utf8.GetString($reportBytes)
    $childBytes=Read-Bytes (Join-Path $root 'python-status.json') 65536
    $child=ConvertFrom-Json -InputObject $utf8.GetString($childBytes)
    $finalReceipts['caller.json']=[ordered]@{filename='caller.json';bytes=$reportBytes.Length;sha256=(Hash-Bytes $reportBytes)}
    $finalReceipts['python-status.json']=[ordered]@{filename='python-status.json';bytes=$childBytes.Length;sha256=(Hash-Bytes $childBytes)}
    $finalReceipts['worker-identity.json']=Receipt 'worker-identity.json'
    $finalAdmission=Receipt 'admission-input.json'
    if (-not(Equal-Receipt $s.identityReceipt $finalReceipts['worker-identity.json']) -or
        -not(Equal-Receipt $s.admissionReceipt $finalAdmission) -or
        -not(Equal-Receipt $child.report_receipt $finalReceipts['caller.json']) -or
        -not(Equal-Receipt $report.admission_receipt $finalAdmission)) {throw 'Receipt reconciliation.'}
    $finishes=@($utf8.GetString($captureBytes['stdout.log']) -split "\r?\n" | Where-Object {$_ -clike 'DSWD_FINISHED *'})
    if ($finishes.Count -ne 1) {throw 'Missing/duplicate child completion record.'}
    $finish=ConvertFrom-Json -InputObject $finishes[0].Substring(14)
    foreach ($name in @('worker-identity.json','caller.json','python-status.json')) {
        if (-not(Equal-Receipt $finish.receipts.PSObject.Properties[$name].Value $finalReceipts[$name])) {throw 'Print receipt.'}
    }
    if ($report.schema -cne 'dswd_shared_caller.v1' -or $child.schema -cne 'dswd_shared_python.v1' -or
        $finish.schema -cne 'dswd_shared_finished.v1' -or $finish.complete -ne $true -or
        $null -ne $report.error -or $null -ne $report.post_error -or $report.scoring_performed -ne $false -or
        $report.compatibility_inference -ne $false) {throw 'Report status/schema.'}
    foreach ($v in @($report,$child)) {
        if ($v.complete -ne $true -or $v.approval -cne $OwnerApproval -or $v.run_id -cne $RunId -or
            $v.arm -cne $Arm -or $v.worker_pid -ne $s.worker.Id -or $v.body_sha256 -cne $bodySha -or
            $v.launcher_sha256 -cne $ReviewedLauncherSha256 -or
            $v.expected_binding_sha256 -cne $ExpectedBindingSha256) {throw 'Child identity reconciliation.'}
    }
    $st=$report.state
    foreach ($key in @('guard_entry','guard_completion','guard_removed')) {
        if ($st.PSObject.Properties[$key].Value -ne $true -or $child.PSObject.Properties[$key].Value -ne $true) {throw 'Guard evidence.'}
    }
    foreach ($key in @('dispatch_attempts','dispatch_completions')) {
        if ($st.PSObject.Properties[$key].Value -ne 1 -or $child.PSObject.Properties[$key].Value -ne 1) {throw 'Single dispatch.'}
    }
    if ($st.actual_arm -cne $Arm -or $report.configuration_reference -cne 'p16-dswd-shared-ca-v1-execution-r1' -or
        $report.config_pre.sha256 -cne $configSha -or $report.config_post.sha256 -cne $configSha -or
        $report.caller_pre.sha256 -cne $ReviewedLauncherSha256 -or $report.caller_post.sha256 -cne $ReviewedLauncherSha256 -or
        $report.runtime.joblib_multiprocessing -cne '0') {throw 'Configuration/environment evidence.'}
    $bp=$st.binding_pre;$bq=$st.binding_post
    if ((ConvertTo-Json $bp -Depth 70 -Compress) -cne (ConvertTo-Json $bq -Depth 70 -Compress) -or
        $bp.run.run_id -cne $RunId -or $bp.configuration.disk_sha256 -cne $configSha -or
        $bp.source.files.PSObject.Properties['Thesis_RF/Code/run_dswd_shared_ca_v1.ps1'].Value -cne $ReviewedLauncherSha256 -or
        $bp.run.destination -cne ((Join-Path $code ('output\phase15\'+$RunId)).Replace('\','/'))) {throw 'Binding reconciliation.'}
    if ((ConvertTo-Json $report.process_binding -Depth 30 -Compress) -cne
        (ConvertTo-Json $s.binding -Depth 30 -Compress)) {throw 'Held process binding reconciliation.'}
    $expectedPins=ConvertFrom-Json @'
{
  "C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/orchestrator.py": "38e3cb58144df53a91a7f85d1f8ea3eb279730a9220935d63b0bb873fc2b2809",
  "C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/dswd_inference_guard.py": "2e69f5a50adc1ca6f69cde0f8bca7758554786bc6ddc3b62feb584b0152f7578",
  "C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/modules/feature_pipeline.py": "b404e82cac952f3111defedf857ecf0b7891214afe4f0876a3bf7617680daeb1",
  "C:/Users/gibgib/Thesis-Cellular-Automata/.venv/Lib/site-packages/sklearn/calibration.py": "4c295444ea50cdd672682cc72e07110abb68c3ce81e02998cec9372449864f35",
  "C:/Users/gibgib/Thesis-Cellular-Automata/.venv/Lib/site-packages/sklearn/preprocessing/_label.py": "a5200ac488bf8b85440783cb3ffd500d805c18e12c2dd86483985dbc5ebc29be",
  "C:/Users/gibgib/Thesis-Cellular-Automata/.venv/Lib/site-packages/sklearn/utils/validation.py": "4e489efc8586dd3e896e6443bffc3620727037b63417d5d9bd3da0529329d51a",
  "C:/Users/gibgib/Thesis-Cellular-Automata/.venv/Lib/site-packages/sklearn/ensemble/_forest.py": "4dd334514b7551590a049548570b824a9cd91bc444959af7a1ac20a104116702"
}
'@
    foreach ($inventory in @($report.source_pre,$report.source_post)) {
        if (@($inventory.PSObject.Properties).Count -ne 7) {throw 'Source inventory cardinality.'}
        foreach ($p in $expectedPins.PSObject.Properties) {
            if ($inventory.PSObject.Properties[$p.Name].Value.sha256 -cne $p.Value) {throw 'Source identity reconciliation.'}
        }
    }
    if ((ConvertTo-Json $report.runtime.execution -Depth 30 -Compress) -cne
        (ConvertTo-Json $bp.runtime -Depth 30 -Compress) -or $report.runtime.gdal -cne '3.12.1') {throw 'Runtime reconciliation.'}
    foreach ($gate in $report.resource_gates) {
        if ($gate.complete -ne $true -or $gate.admission.pass -ne $true -or
            -not(Is-Integer $gate.requested) -or $gate.requested -le 0 -or $gate.returned -ne $gate.requested -or
            $gate.requested -gt $journalBytes.Length -or $gate.admission.sample.child_pid -ne $s.worker.Id -or
            $gate.admission.sample.role -cne 'verified_worker') {throw 'Resource gate evidence.'}
        $prefix=[byte[]]::new([int]$gate.requested)
        [Array]::Copy($journalBytes,0,$prefix,0,$prefix.Length)
        if ((Hash-Bytes $prefix) -cne $gate.sha256) {throw 'Journal prefix receipt.'}
    }
    if ($report.resource_gates.Count -ne 2 -or $report.counts.source_hash_success -ne 14 -or
        $child.counts.source_hash_success -ne 14 -or $report.counts.journal_read_hash_success -ne 2 -or
        $child.counts.journal_read_hash_success -ne 2) {throw 'Caller operation accounting.'}
    $warningLines=@($utf8.GetString($captureBytes['stderr.log']) -split "\r?\n" | Where-Object {$_ -ne ''})
    $observations=@($st.warnings)
    if ($warningLines.Count -ne $observations.Count -or $child.warning_count -ne $observations.Count -or
        $observations.Count -gt 8192) {throw 'Warning capture accounting.'}
    $estimator=if ($Arm -ceq 'rf') {'CalibratedClassifierCV'} else {'LogisticRegression'}
    $warningMessage='X does not have valid feature names, but '+$estimator+' was fitted with feature names'
    for ($i=0;$i -lt $warningLines.Count;$i++) {
        if (-not $warningLines[$i].StartsWith('DSWD_WARNING ')) {throw 'Unexpected raw stderr.'}
        $w=ConvertFrom-Json -InputObject $warningLines[$i].Substring(13)
        $saved=$observations[$i]
        foreach ($v in @($w,$saved)) {
            if ($v.category -cne 'builtins.UserWarning' -or $v.message -cne $warningMessage -or
                $v.filename -ine (Join-Path $repo '.venv\Lib\site-packages\sklearn\utils\validation.py') -or
                $v.lineno -ne 2691 -or $v.arm -cne $Arm -or $v.accepted -ne $true -or
                $v.message_truncated -ne $false) {throw 'Warning policy.'}
        }
    }
    $runRoot=Join-Path $code ('output\phase15\'+$RunId)
    $startBytes=Read-Bytes (Join-Path $runRoot 'start.json') 8388608
    $endBytes=Read-Bytes (Join-Path $runRoot 'termination.json') 8388608
    $start=ConvertFrom-Json -InputObject $utf8.GetString($startBytes)
    $end=ConvertFrom-Json -InputObject $utf8.GetString($endBytes)
    if ((Hash-Bytes $startBytes) -cne $st.run_receipts.PSObject.Properties['start.json'].Value.sha256 -or
        (Hash-Bytes $endBytes) -cne $st.run_receipts.PSObject.Properties['termination.json'].Value.sha256 -or
        $end.start_record_sha256 -cne (Hash-Bytes $startBytes) -or $end.binding_sha256 -cne $ExpectedBindingSha256 -or
        $start.binding.expected_sha256 -cne $ExpectedBindingSha256 -or
        $start.binding.calculated_sha256 -cne $ExpectedBindingSha256 -or
        (ConvertTo-Json $start.binding.payload -Depth 70 -Compress) -cne (ConvertTo-Json $bp -Depth 70 -Compress) -or
        $start.run_id -cne $RunId -or $end.run_id -cne $RunId -or $end.exit_code -ne 0 -or
        $end.output_valid -ne $true -or $null -ne $end.error -or $end.violations.Count -ne 0 -or
        $end.termination_reason -cnotin @('inactive','timestep_limit')) {throw 'Runtime receipt reconciliation.'}
    $candidate=($s.started -and $s.bound -and $s.sent -and $s.launcherWaited -and $s.workerWaited -and
        $s.launcherExit -eq 0 -and $s.workerExit -eq 0 -and $s.stdoutDone -and $s.stderrDone -and
        -not $s.killRequested -and $s.errors.Count -eq 0)
} catch {$s.errors.Add('Reconciliation: '+$_.ToString());$candidate=$false}
foreach ($p in @($s.worker,$s.process)) {
    if ($null -eq $p) {continue}
    try {$p.Dispose()} catch {$s.errors.Add('Process disposal: '+$_.ToString());$candidate=$false}
}
if ($s.errors.Count -ne 0) {$candidate=$false}
# Publication/print failure below cannot restart publication or restore acceptance.
$parentStatus=[ordered]@{schema='dswd_shared_parent.v1';approval=$OwnerApproval;root=$root
    run_id=$RunId;arm=$Arm;expected_binding_sha256=$ExpectedBindingSha256
    configuration_reference='p16-dswd-shared-ca-v1-execution-r1';candidate_execution_complete=$candidate
    outcome_acceptance='PENDING_INDEPENDENT_NATIVE_READERS';paired_acceptance=$false
    launcher_pre=$s.launcherPre;launcher_post=$s.launcherPost;body_sha256=$bodySha
    config_pre=$s.configPre;config_post=$s.configPost;binding=$s.binding
    resource_journal_receipt=$journalReceipt;input_delivered=$s.sent
    launcher_waited=$s.launcherWaited;worker_waited=$s.workerWaited
    launcher_exit=$s.launcherExit;worker_exit=$s.workerExit
    stdout_eof=$s.stdoutDone;stderr_eof=$s.stderrDone;captures=$captures;child_receipts=$finalReceipts
    termination_requested=$s.killRequested;termination_errors=$s.killErrors;errors=@($s.errors)
    caller_budget_seconds=1920;production_between_step_seconds=1800
    sampled_limits=$true;prebinding_worker_coverage=$false;descendant_supervision=$false
    retry=$false;cleanup=$false}
$parentReceipt=Publish 'exit-status.txt' $parentStatus
[Console]::WriteLine(($parentReceipt|ConvertTo-Json -Compress))
if (-not $candidate) {throw ('Trajectory failed/incomplete; retain '+$root)}
