"""Temp-only Stage 14 adapter contracts; execution requires the parent test release."""

from __future__ import annotations

import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "run_phase14_supervised.ps1"
HOST = shutil.which("powershell.exe")


def _q(value):
    return "'" + str(value).replace("'", "''") + "'"


def _harness(tmp_path, body, *, load=True):
    assert os.name == "nt" and HOST, "Required Windows PowerShell coverage cannot be skipped"
    path = tmp_path / "synthetic_harness.ps1"
    path.write_text(
        "$ErrorActionPreference='Stop'\n"
        "if ($PSVersionTable.PSVersion.ToString() -ne '5.1.26100.9711' -or $PSVersionTable.PSEdition -ne 'Desktop') { throw 'Required host mismatch' }\n"
        + (". " + _q(RUNNER) + "\n" if load else "") + body, encoding="utf-8")
    return subprocess.run([HOST, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(path)],
                          capture_output=True, text=True, timeout=20, check=False)


def _json_expression(value):
    encoded = base64.b64encode(json.dumps(value).encode()).decode()
    return "ConvertFrom-Phase14Json ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String(" + _q(encoded) + ")))"


def test_safe_dot_source_does_not_launch_or_write_review(tmp_path):
    body = (
        "$script:forbidden=0\n"
        "function Start-Process { $script:forbidden++; throw 'forbidden launch' }\n"
        "function New-Object { $script:forbidden++; throw 'forbidden construction' }\n"
        ". " + _q(RUNNER) + "\n"
        "if ($script:forbidden -ne 0) { throw 'load executed' }\n"
        "if (-not (Get-Command Write-Phase14ReviewRecord -CommandType Function)) { throw 'review interface missing' }\n")
    result = _harness(tmp_path, body, load=False)
    assert result.returncode == 0, result.stderr
    assert [path.name for path in tmp_path.iterdir()] == ["synthetic_harness.ps1"]


@pytest.mark.parametrize("arguments", [
    "", "-Job arbitrary", "-Executable arbitrary", "-Config arbitrary", "-Destination arbitrary",
    "-Job arbitrary -ApprovalReference approval -ExpectedRevision " + "a" * 40 + " -RunId run",
    "-Synthetic", "-Job calibration -ApprovalReference approval -ExpectedRevision revision -RunId run -Extra extra",
])
def test_direct_interface_denies_before_child_launch(tmp_path, arguments):
    body = (
        "$script:launches=0\n"
        "function New-Object { $script:launches++; throw 'forbidden construction' }\n"
        "function Start-Process { $script:launches++; throw 'forbidden launch' }\n"
        "$denied=$false; try { & " + _q(RUNNER) + " " + arguments + " } catch { $denied=$true }\n"
        "if (-not $denied) { throw 'accepted invalid entry' }\n"
        "if ($script:launches -ne 0) { throw 'entry launched a child' }\n")
    result = _harness(tmp_path, body, load=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("text", [
    '{"x":1,"x":2}', '{"nested":{"x":1,"x":2}}', '{"x":NaN}', '{"x":Infinity}',
    '{"x":1} trailing', '\ufeff{"x":1}', '{"x":"unterminated}', '{"x":1 /*comment*/}',
])
def test_strict_transport_rejects_duplicates_bom_and_non_json(tmp_path, text):
    encoded = base64.b64encode(text.encode()).decode()
    body = (
        "$text=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String(" + _q(encoded) + "))\n"
        "$rejected=$false; try { ConvertFrom-Phase14Json $text | Out-Null } catch { $rejected=$true }\n"
        "if (-not $rejected) { throw 'invalid JSON accepted' }\n")
    result = _harness(tmp_path, body)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("text", ['{\n  "reviewed": true\n}\n', '{\r\n  "reviewed": true\r\n}\r\n'], ids=["lf", "crlf"])
def test_exact_text_digest_and_exclusive_record_creation(tmp_path, text):
    target = tmp_path / "immutable.json"
    raw = text.encode("utf-8")
    expected = sha256(raw).hexdigest()
    encoded = base64.b64encode(raw).decode("ascii")
    body = (
        "$text=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String(" + _q(encoded) + "))\n"
        "if ((Get-Phase14TextHash $text) -cne " + _q(expected) + ") { throw 'transport normalized' }\n"
        "Write-Phase14JsonNew " + _q(target) + " @{first=$true}\n"
        "$before=Get-PhaseFileSha256 " + _q(target) + "\n"
        "$denied=$false; try { Write-Phase14JsonNew " + _q(target) + " @{second=$true} } catch { $denied=$true }\n"
        "if (-not $denied -or (Get-PhaseFileSha256 " + _q(target) + ") -cne $before) { throw 'record overwritten' }\n")
    result = _harness(tmp_path, body)
    assert result.returncode == 0, result.stderr


def test_canonical_json_preserves_primitives_and_array_structure(tmp_path):
    body = (
        "$empty=[object[]]::new(0)\n"
        "$single=[object[]]@(7492)\n"
        "$multiple=[object[]]@(7492,18412)\n"
        "$primitive=[ordered]@{text='worker';enabled=$true;count=[int]7492;absent=$null}\n"
        "$nested=[ordered]@{z=7;a=[ordered]@{values=[object[]]@(7492,$null,[object[]]@(18412));label='child';enabled=$false}}\n"
        "$rootNested=[object[]]@(1,[object[]]@(2,$null),$false)\n"
        "if ((Get-Phase14CanonicalJson ([int]7492)) -cne '7492') { throw 'integer primitive changed' }\n"
        "if ((Get-Phase14CanonicalJson ([double]1.25)) -cne '1.25') { throw 'fractional primitive changed' }\n"
        "if ((Get-Phase14CanonicalJson $primitive) -cne '{\"absent\":null,\"count\":7492,\"enabled\":true,\"text\":\"worker\"}') { throw 'primitive value changed' }\n"
        "if ((Get-Phase14CanonicalJson $empty) -cne '[]') { throw 'root empty array changed' }\n"
        "if ((Get-Phase14CanonicalJson $single) -cne '[7492]') { throw 'root singleton array changed' }\n"
        "if ((Get-Phase14CanonicalJson $multiple) -cne '[7492,18412]') { throw 'root multiple array changed' }\n"
        "if ((Get-Phase14CanonicalJson $rootNested) -cne '[1,[2,null],false]') { throw 'root nested array changed' }\n"
        "if ((Get-Phase14CanonicalJson @{values=$empty}) -cne '{\"values\":[]}') { throw 'empty array changed' }\n"
        "if ((Get-Phase14CanonicalJson @{process_ids=$single}) -cne '{\"process_ids\":[7492]}') { throw 'singleton process ID changed' }\n"
        "if ((Get-Phase14CanonicalJson @{process_ids=$multiple}) -cne '{\"process_ids\":[7492,18412]}') { throw 'multiple process IDs changed' }\n"
        "if ((Get-Phase14CanonicalJson $nested) -cne '{\"a\":{\"enabled\":false,\"label\":\"child\",\"values\":[7492,null,[18412]]},\"z\":7}') { throw 'nested arrays or key ordering changed' }\n")
    result = _harness(tmp_path, body)
    assert result.returncode == 0, result.stderr


def _synthetic_context(tmp_path, job, receipt_mode="valid", *, run_id="synthetic-adapter", lock_hash="d" * 64):
    """Internal seam only; no production interface exposes command/path overrides."""
    root = tmp_path / "output" / "phase14"
    evidence = root / "evidence" / run_id
    destination = (root / "phase14-setc-rf-61e8f740-r1" / "calibration" / run_id if job == "calibration"
                   else root / "final-test" / lock_hash[:12] / job.removeprefix("final_") / run_id)
    versions = {key: "synthetic" for key in ("python", "scikit_learn", "optuna", "joblib", "numpy", "pandas", "scipy", "pyyaml", "platform")}
    from modules.phase14_calibration import SOURCE_FILES, PROTOCOL_SHA256
    source = {"revision": "a" * 40, "clean": True, "source_files": {name: "c" * 64 for name in SOURCE_FILES},
              "configuration_sha256": "b" * 64, "protocol_sha256": PROTOCOL_SHA256}
    context = {"job": job, "run_id": run_id, "approval": "synthetic-approval", "revision": "a" * 40,
        "root": str(root), "code": str(tmp_path), "executable": sys.executable,
        "evidence": str(evidence), "destination": str(destination), "lock_path": None,
        "authorization": {"configuration_sha256": "b" * 64, "environment": {"versions": versions}, "pretest_lock": None},
        "authorization_json": '{"synthetic":true}', "authorization_sha256": sha256(b'{"synthetic":true}').hexdigest(), "source_check": source}
    context["authorization"].update(job=job, run_id=run_id, expected_revision=context["revision"],
        approval_reference=context["approval"], source_files=source["source_files"], protocol_sha256=PROTOCOL_SHA256)
    if job != "calibration":
        context["authorization"]["pretest_lock"] = {"lock_id": "synthetic-lock", "sha256": lock_hash,
                                                     "owner_approval_reference": "synthetic-lock-approval"}
    context["authorization_json"] = json.dumps(context["authorization"])
    context["authorization_sha256"] = sha256(context["authorization_json"].encode()).hexdigest()
    import inspect
    from modules.phase14_calibration import _current_worker_instance
    worker_source = inspect.getsource(_current_worker_instance)
    child = f"""
import json, os, time, sys
from datetime import datetime, timedelta, timezone
{worker_source}
from hashlib import sha256
from pathlib import Path
evidence=Path({str(evidence)!r})
destination=Path({str(destination)!r})
context=json.loads({json.dumps(context)!r})
handoff=json.loads((evidence/'supervisor_handoff.json').read_text())
if {receipt_mode!r}=='missing_claim': time.sleep(11); raise SystemExit(0)
claim={{'schema_version':'phase14a_worker_claim.v1','job':{job!r},'run_id':context['run_id'],
    'nonce':handoff['nonce'],'handoff_sha256':sha256((evidence/'supervisor_handoff.json').read_bytes()).hexdigest(),
    'supervisor_pid':handoff['supervisor_pid'],'logical_executable':sys.executable,
    **_current_worker_instance(),'claimed_at_utc':datetime.now(timezone.utc).isoformat()}}
pending=evidence/'worker_claim.pending.json'
with pending.open('x',encoding='utf-8') as stream: json.dump(claim,stream)
pending.rename(evidence/'worker_claim.json')
deadline=time.monotonic()+10
while not (evidence/'process.json').is_file() and time.monotonic()<deadline: time.sleep(0.02)
process=json.loads((evidence/'process.json').read_text())
assert process['worker_binding']['worker']['pid']==os.getpid()
assert process['process_id'] in process['process_ids']
binding_hash=sha256((evidence/'worker_binding.json').read_bytes()).hexdigest()
if {receipt_mode!r}=='missing': raise SystemExit(0)
computation_started=datetime.now(timezone.utc)
destination.mkdir(parents=True)
names=(['base_rf_calibrated.joblib','calibration_manifest.json','calibration_report.json'] if {job!r}=='calibration'
       else ['final_test_report.json'] if {job!r}=='final_primary' else ['verification_metrics.json'])
finalized={{}}
for name in names:
    path=destination/name
    path.write_text('{{"synthetic":true}}')
    finalized[name]=sha256(path.read_bytes()).hexdigest()
computation_ended=datetime.now(timezone.utc)
if {receipt_mode!r}=='stale':
    computation_started=computation_ended=datetime.fromisoformat(handoff['created_at_utc'])-timedelta(seconds=1)
if {receipt_mode!r}=='future': computation_ended=datetime.fromisoformat(handoff['deadline_utc'])-timedelta(seconds=1)
if {receipt_mode!r}=='reversed': computation_started=computation_ended+timedelta(microseconds=1)
receipt={{
    'schema_version':'phase14a_execution_receipt.v2', 'run_id':context['run_id'], 'approval_reference':context['approval'],
    'argv':handoff['argv'], 'working_directory':context['code'], 'source_revision':context['revision'], 'source_clean':True,
    'source_check':context['source_check'], 'supervisor_handoff':{{'run_id':context['run_id'],'nonce':handoff['nonce'],
        'sha256':sha256((evidence/'supervisor_handoff.json').read_bytes()).hexdigest(),'supervisor_pid':handoff['supervisor_pid'],'child_pid':os.getpid()}},
    'worker_binding_sha256':binding_hash, 'configuration_sha256':'b'*64,'environment':context['authorization']['environment']['versions'] if {job!r}=='calibration' else context['authorization']['environment'],
    'inputs':{{}},'computation_started_at_utc':computation_started.isoformat(),'computation_ended_at_utc':computation_ended.isoformat(),
    'computation_started_at_local':computation_started.astimezone().isoformat(),'computation_ended_at_local':computation_ended.astimezone().isoformat(),
    'local_timezone':str(computation_started.astimezone().tzinfo),'computation_status':'complete' if {job!r}=='calibration' else 'computation_complete',
    'failure_reason':None,'finalized_outputs':finalized,'partial_output_inventory':list(finalized)}}
if {job!r}=='calibration': receipt['execution_scope']='production_calibration'
else: receipt.update(model_sha256='d'*64,final_rows_sha256='e'*64)
if {receipt_mode!r}=='needs_evidence': receipt.update(computation_status='needs_evidence',failure_reason='verification aggregates differ')
if {receipt_mode!r}=='invalid': receipt['finalized_outputs']['execution_receipt.json']='0'*64
(destination/'execution_receipt.json').write_text(json.dumps(receipt))
print('synthetic stdout',flush=True)
print('synthetic stderr',file=__import__('sys').stderr,flush=True)
if {receipt_mode!r}=='descendant':
    import subprocess, sys
    subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(1.5)'])
if {receipt_mode!r}=='slow': time.sleep(30)
if {receipt_mode!r}=='nonzero': raise SystemExit(7)
"""
    encoded = base64.b64encode(child.encode()).decode()
    context["arguments"] = ["-B", "-c", f"import base64;exec(base64.b64decode('{encoded}'))"]
    return context


def _injected_context(context):
    return (
        "$script:syntheticContext=" + _json_expression(context) + "\n"
        "function Get-Phase14JobContext { param($Job,$ApprovalReference,$ExpectedRevision,$RunId); return $script:syntheticContext }\n"
        "function Get-PhaseMemoryAndDisk { return @{available_physical_memory_bytes=8589934592L;free_c_bytes=4294967296L} }\n")


@pytest.mark.parametrize("job", ["calibration", "final_primary", "final_verification"])
def test_shared_core_with_synthetic_child_publishes_separate_immutable_records(tmp_path, job):
    context = _synthetic_context(tmp_path, job)
    body = _injected_context(context) + (
        "Invoke-Phase14Supervision -Job " + _q(job) + " -ApprovalReference 'synthetic-approval' -ExpectedRevision " + _q("a" * 40) + " -RunId 'synthetic-adapter' | Out-Null\n")
    result = _harness(tmp_path, body)
    assert result.returncode == 0, result.stderr
    evidence = Path(context["evidence"])
    completion = json.loads((evidence / "supervisor_completion.json").read_text())
    receipt_path = Path(context["destination"]) / "execution_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert completion["overall_status"] == "PASS" and completion["actual_exit_status"] == 0
    assert completion["capture_complete"] is True and completion["remaining_process_ids"] == []
    assert completion["child_receipt"]["sha256"] == sha256(receipt_path.read_bytes()).hexdigest()
    from datetime import datetime
    handoff = json.loads((evidence / "supervisor_handoff.json").read_text())
    process = json.loads((evidence / "process.json").read_text())
    parse = datetime.fromisoformat
    assert (parse(process["worker_binding"]["observed_at_utc"])
            <= parse(receipt["computation_started_at_utc"]) <= parse(receipt["computation_ended_at_utc"])
            <= parse(completion["ended_at_utc"]) < parse(handoff["deadline_utc"]))
    assert "execution_receipt.json" not in receipt["finalized_outputs"]
    assert "supervisor_completion.json" not in receipt["finalized_outputs"]
    assert "review_record.json" not in receipt["finalized_outputs"]
    assert "actual_exit_status" not in receipt and "capture_complete" not in receipt
    assert "sha256" not in completion
    assert "synthetic stdout" in (evidence / "stdout.log").read_text()


@pytest.mark.parametrize("receipt_mode", ["missing", "invalid", "needs_evidence"])
def test_missing_invalid_or_mismatched_computation_cannot_be_pass(tmp_path, receipt_mode):
    context = _synthetic_context(tmp_path, "final_verification", receipt_mode)
    body = _injected_context(context) + (
        "$denied=$false; try { Invoke-Phase14Supervision -Job final_verification -ApprovalReference synthetic-approval -ExpectedRevision "
        + _q("a" * 40) + " -RunId synthetic-adapter | Out-Null } catch { $denied=$true }\n"
        "if (-not $denied) { throw 'failed computation promoted' }\n")
    result = _harness(tmp_path, body)
    assert result.returncode == 0, result.stderr
    completion = json.loads((Path(context["evidence"]) / "supervisor_completion.json").read_text())
    assert completion["overall_status"] != "PASS"
    assert completion["child_receipt"]["status"] == ("missing" if receipt_mode == "missing" else "present")
    if receipt_mode == "missing":
        assert not (Path(context["destination"]) / "execution_receipt.json").exists()


@pytest.mark.parametrize("capture", ["fault", "timeout"])
def test_capture_injection_is_reached_and_failure_cannot_promote_valid_receipt(tmp_path, capture):
    context = _synthetic_context(tmp_path, "calibration")
    marker = tmp_path / "capture-calls.txt"
    behavior = "throw 'PRIVATE_CAPTURE'" if capture == "fault" else "return $false"
    body = _injected_context(context) + (
        "function Wait-Phase14CaptureTasks { param($Tasks,$TimeoutMilliseconds); Add-Content -LiteralPath "
        + _q(marker) + " -Value called; " + behavior + " }\n"
        "try { Invoke-Phase14Supervision -Job calibration -ApprovalReference synthetic-approval -ExpectedRevision "
        + _q("a" * 40) + " -RunId synthetic-adapter | Out-Null } catch { if ($_.Exception.Message -like '*PRIVATE_CAPTURE*') { throw 'private diagnostic leaked' } }\n")
    result = _harness(tmp_path, body)
    assert result.returncode == 0, result.stderr
    assert marker.read_text(encoding="utf-8-sig").splitlines() == ["called"]
    completion_text = (Path(context["evidence"]) / "supervisor_completion.json").read_text()
    completion = json.loads(completion_text)
    assert "PRIVATE_CAPTURE" not in completion_text + result.stdout + result.stderr
    assert completion["overall_status"] != "PASS" and completion["capture_complete"] is False
    assert completion["stdout_sha256"] is None and completion["stderr_sha256"] is None


def test_requested_pass_review_with_missing_evidence_is_needs_evidence_and_exclusive(tmp_path):
    adapter = tmp_path / "run_phase14_supervised.ps1"
    adapter.write_bytes(RUNNER.read_bytes())
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "phase_supervision.psm1").write_bytes((RUNNER.parent / "modules" / "phase_supervision.psm1").read_bytes())
    authorization = {"schema_version": "phase14a_review_authorization.v1", "review_kind": "calibration",
        "reviewer_id": "synthetic-reviewer", "approval_reference": "synthetic-review-approval",
        "calibration_run_id": "synthetic-calibration", "primary_run_id": "", "verification_run_id": "", "lock_id": "",
        "disposition": "PASS", "findings": ["evidence_missing"], "created_at_utc": "2026-10-10T00:00:00+00:00"}
    text = json.dumps(authorization)
    body = (
        ". " + _q(adapter) + "\n"
        "$env:PHASE14_REVIEW_AUTHORIZATION_JSON=" + _q(text) + "\n"
        "$env:PHASE14_REVIEW_AUTHORIZATION_SHA256=" + _q(sha256(text.encode()).hexdigest()) + "\n"
        "$result=Write-Phase14ReviewRecord -ReviewKind calibration -ReviewerId synthetic-reviewer -ApprovalReference synthetic-review-approval -CalibrationRunId synthetic-calibration -Disposition PASS -Findings @('evidence_missing')\n"
        "if ($result.disposition -cne 'NEEDS EVIDENCE') { throw 'missing evidence promoted' }\n"
        "$denied=$false; try { Write-Phase14ReviewRecord -ReviewKind calibration -ReviewerId synthetic-reviewer -ApprovalReference synthetic-review-approval -CalibrationRunId synthetic-calibration -Disposition PASS -Findings @('evidence_missing') | Out-Null } catch { $denied=$true }\n"
        "if (-not $denied) { throw 'review overwritten' }\n")
    result = _harness(tmp_path, body, load=False)
    assert result.returncode == 0, result.stderr
    review_path = tmp_path / "output" / "phase14" / "calibration-review" / "synthetic-calibration" / "review_record.json"
    assert json.loads(review_path.read_text())["disposition"] == "NEEDS EVIDENCE"


@pytest.mark.parametrize("failure", ["nonzero", "unknown", "resource", "time", "finalization"])
def test_adapter_retains_failure_evidence_and_does_not_promote_valid_outputs(tmp_path, failure):
    context = _synthetic_context(tmp_path, "calibration", "slow" if failure in {"resource", "time"} else "nonzero" if failure == "nonzero" else "valid")
    body = _injected_context(context)
    if failure == "resource":
        receipt = Path(context["destination"]) / "execution_receipt.json"
        body += "function Get-PhaseMemoryAndDisk { if (Test-Path -LiteralPath " + _q(receipt) + ") { return @{available_physical_memory_bytes=1L;free_c_bytes=4294967296L} }; return @{available_physical_memory_bytes=8589934592L;free_c_bytes=4294967296L} }\n"
        body += (
            "function Invoke-PhaseSupervisionCore { param($Executable,$ChildArguments,$WorkingDirectory,$EvidenceDirectory,$RunId,$ResourceLogPath,$OverallLimitSeconds,$SampleIntervalSeconds,$MemoryFloorBytes,$DiskFloorBytes,$ResourceProvider,$Clock,$CaptureWait,$BeforeStart,$RecordProcess,$InspectExit)\n"
            "$observed=phase_supervision\\Invoke-PhaseSupervisionCore @PSBoundParameters\n"
            "try {\n"
            "  $diagnosticDirectory=[IO.Path]::Combine([IO.Path]::GetTempPath(),('phase14g-resource-stop-'+[Guid]::NewGuid().ToString('N')))\n"
            "  $null=New-Item -ItemType Directory -Path $diagnosticDirectory -ErrorAction Stop\n"
            "  [Console]::Error.WriteLine('RESOURCE_DIAGNOSTIC_DIRECTORY='+$diagnosticDirectory)\n"
            "  $diagnostic=[ordered]@{stopReason=$observed.stopReason;runnerError=$observed.runnerError;cleanupError=$observed.cleanupError;captureError=$observed.captureError;finalizationError=$observed.finalizationError;exitStatus=$observed.exitStatus;samples=@($observed.samples.ToArray());treeState=$observed.treeState;knownIds=@($observed.knownIds);processStarted=$observed.processStarted;processStartedUtc=$observed.processStartedUtc;captureComplete=$observed.captureComplete;captureStatus=$observed.captureStatus;captureDrainTimedOut=$observed.captureDrainTimedOut}\n"
            "  $json=ConvertTo-Json -InputObject $diagnostic -Depth 10\n"
            "  $bytes=(New-Object Text.UTF8Encoding($false)).GetBytes($json)\n"
            "  $stream=[IO.File]::Open([IO.Path]::Combine($diagnosticDirectory,'raw-core-state.json'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)\n"
            "  try { $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }\n"
            "} catch {\n"
            "  try { [Console]::Error.WriteLine('RESOURCE_DIAGNOSTIC_REPORTING_ERROR='+$_.Exception.ToString()) } catch {}\n"
            "}\n"
            "return $observed }\n")
    if failure in {"unknown", "time", "finalization"}:
        body += (
            "function Invoke-PhaseSupervisionCore { param($Executable,$ChildArguments,$WorkingDirectory,$EvidenceDirectory,$RunId,$ResourceLogPath,$OverallLimitSeconds,$SampleIntervalSeconds,$MemoryFloorBytes,$DiskFloorBytes,$ResourceProvider,$Clock,$CaptureWait,$BeforeStart,$RecordProcess,$InspectExit)\n"
            + ("$PSBoundParameters['OverallLimitSeconds']=1\n" if failure == "time" else "")
            + "$observed=phase_supervision\\Invoke-PhaseSupervisionCore @PSBoundParameters\n"
            + ("$observed.exitStatus=$null\n" if failure == "unknown" else "")
            + ("$observed.finalizationError='PRIVATE_FINALIZATION'\n" if failure == "finalization" else "")
            + "return $observed }\n")
    body += ("$denied=$false; try { Invoke-Phase14Supervision -Job calibration -ApprovalReference synthetic-approval -ExpectedRevision "
             + _q("a" * 40) + " -RunId synthetic-adapter | Out-Null } catch { $denied=$true }; if (-not $denied) { throw 'failure promoted' }\n")
    result = _harness(tmp_path, body)
    assert result.returncode == 0, result.stderr
    raw = (Path(context["evidence"]) / "supervisor_completion.json").read_text()
    completion = json.loads(raw)
    assert completion["overall_status"] != "PASS"
    assert "PRIVATE_FINALIZATION" not in raw + result.stdout + result.stderr
    assert completion["remaining_process_ids"] == []
    if failure == "unknown": assert completion["actual_exit_status"] is None
    if failure == "nonzero": assert completion["actual_exit_status"] == 7
    if failure == "resource":
        assert completion["stop_reason"] == "resource floor breached"
        diagnostic_directory = next(line.split("=", 1)[1] for line in result.stderr.splitlines()
                                    if line.startswith("RESOURCE_DIAGNOSTIC_DIRECTORY="))
        diagnostic = json.loads((Path(diagnostic_directory) / "raw-core-state.json").read_text())
        assert diagnostic["runnerError"] is None
    if failure == "time": assert completion["stop_reason"] == "overall time limit reached"


def test_descendant_is_observed_before_resolved_completion(tmp_path):
    context = _synthetic_context(tmp_path, "calibration", "descendant")
    result = _harness(tmp_path, _injected_context(context) +
        "Invoke-Phase14Supervision -Job calibration -ApprovalReference synthetic-approval -ExpectedRevision "
        + _q("a" * 40) + " -RunId synthetic-adapter | Out-Null\n")
    assert result.returncode == 0, result.stderr
    completion = json.loads((Path(context["evidence"]) / "supervisor_completion.json").read_text())
    assert len(completion["process_ids"]) >= 2
    assert completion["process_exited"] is True and completion["remaining_process_ids"] == []
    assert completion["overall_status"] == "PASS"


def _canonical_digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _metadata_environment(code):
    versions = dict.fromkeys(("python", "scikit_learn", "optuna", "joblib", "numpy", "pandas", "scipy", "pyyaml", "platform"), "synthetic")
    return {"status": "complete_locked", "versions": versions, "lock_sha256": _canonical_digest(versions),
            "interpreter": {"path": str(code.parents[1] / ".venv" / "Scripts" / "python.exe")},
            "powershell": {"version": "5.1.26100.9711", "edition": "Desktop"}, "auxiliary_versions": {"rasterio": "synthetic"}}


def _metadata_lock(environment, sources, configuration_hash):
    from modules import phase14_calibration as phase14
    from modules.feature_pipeline import CANONICAL_FEATURE_NAMES
    return {"schema_version": "phase14a_pretest_lock.v1", "lock_id": "synthetic-lock",
        "created_at_utc": "2026-10-10T00:00:00+00:00", "configuration_sha256": configuration_hash,
        "evaluation_revision": "a" * 40, "environment": environment, "model_sha256": "d" * 64,
        "final_rows_sha256": "e" * 64, "protected_attestation_sha256": "f" * 64,
        "development_attestation_sha256": phase14.DEVELOPMENT_ATTESTATION_SHA256,
        "protocol": {"id": phase14.PROTOCOL_ID, "sha256": phase14.PROTOCOL_SHA256}, "calibration_revision": "b" * 40,
        "evaluator_source_sha256": sources["Thesis_RF/Code/evaluate_rf_final_test.py"],
        "phase13": {"run_id": phase14.PHASE13_RUN_ID, "source_revision": phase14.PHASE13_REVISION,
            "protocol_sha256": phase14.PHASE13_PROTOCOL_SHA256, "configuration_sha256": phase14.PHASE13_CONFIG_SHA256,
            "manifest_sha256": phase14.PHASE13_MANIFEST_SHA256, "base_model_sha256": phase14.BASE_MODEL_SHA256,
            "development_attestation_sha256": phase14.DEVELOPMENT_ATTESTATION_SHA256,
            "source_csv_sha256": phase14.SOURCE_CSV_SHA256, "source_manifest_sha256": phase14.SOURCE_MANIFEST_SHA256},
        "calibration": {"run_id": "synthetic-calibration", "approval_reference": "synthetic-calibration-approval",
            "method": "sigmoid", "method_decision_reference": phase14.DECISION_REFERENCE,
            "calibration_csv_sha256": phase14.CALIBRATION_CSV_SHA256,
            **dict.fromkeys(("configuration_sha256", "source_sha256", "manifest_sha256", "report_sha256",
                            "receipt_sha256", "supervisor_completion_sha256", "package_review_sha256", "forest_fingerprint_sha256"), "1" * 64)},
        "prediction_contract": {"feature_schema_version": "set_c.v1", "feature_names": list(CANONICAL_FEATURE_NAMES),
            "feature_dtype": "float32", "target_name": "Ignited", "target_version": "ignition_t_plus_1.v1",
            "target_dtype": "int8", "required_labels": [0, 1], "positive_label": 1},
        "reporting": {"metrics": ["average_precision", "roc_auc", "brier_score", "log_loss"],
            "brier_definition": "mean_squared_probability_error", "log_loss_labels": [0, 1],
            "bin_edges": list(phase14.BIN_EDGES), "endpoint_rule": "exact_internal_edge_starts_next_bin; p1_in_last_bin",
            "empty_bin_statistics": None, "classification_threshold": None, "verification_rel_tol": 1e-10, "verification_abs_tol": 1e-12}}


def _context_tree(tmp_path, job, *, runner_source=None):
    from modules import phase14_calibration as phase14
    runner_source = RUNNER if runner_source is None else runner_source
    repo = tmp_path / "synthetic-checkout"
    code = repo / "Thesis_RF" / "Code"
    for member in phase14.SOURCE_FILES:
        path = repo / member
        path.parent.mkdir(parents=True, exist_ok=True)
        if member == "Thesis_RF/Code/run_phase14_supervised.ps1":
            path.write_bytes(runner_source.read_bytes())
        elif member == "Thesis_RF/Code/modules/phase_supervision.psm1":
            path.write_bytes((runner_source.parent / "modules" / "phase_supervision.psm1").read_bytes())
        elif member == "docs/Phase14A_Calibration_Final_Test_Protocol_Proposal.md":
            path.write_bytes((RUNNER.parents[2] / member).read_bytes())
        else:
            path.write_text("execution_enabled: true\n" if "/config/" in member else "# explicitly synthetic source\n", encoding="utf-8")
    executable = repo / ".venv" / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"synthetic path anchor, never executed")
    sources = {member: sha256((repo / member).read_bytes()).hexdigest() for member in phase14.SOURCE_FILES}
    config_hash = sources["Thesis_RF/Code/config/" + ("phase14_setc_calibration_v1.yaml" if job == "calibration" else "phase14_setc_final_test_v1.yaml")]
    environment = _metadata_environment(code)
    lock = _metadata_lock(environment, sources, config_hash)
    lock_path = code / "output" / "phase14" / "model-lock" / "synthetic-lock.json"
    if job != "calibration":
        lock_path.parent.mkdir(parents=True)
        lock_path.write_text(json.dumps(lock), encoding="utf-8")
    lock_hash = sha256(lock_path.read_bytes()).hexdigest() if job != "calibration" else None
    authorization = {"schema_version": "phase14a_job_authorization.v1", "job": job, "approval_reference": "synthetic-approval",
        "run_id": "synthetic-context", "expected_revision": "a" * 40, "created_at_utc": "2026-10-10T00:00:00+00:00",
        "configuration_sha256": config_hash, "protocol_sha256": phase14.PROTOCOL_SHA256, "source_files": sources,
        "environment": environment, "environment_sha256": _canonical_digest(environment),
        "pretest_lock": None if job == "calibration" else {"lock_id": "synthetic-lock", "sha256": lock_hash, "owner_approval_reference": "synthetic-lock-approval"},
        "primary_evidence": {"run_id": "synthetic-primary", "report_sha256": "1" * 64, "receipt_sha256": "2" * 64,
                            "supervisor_completion_sha256": "3" * 64} if job == "final_verification" else None}
    return code, authorization, lock, lock_hash


def _metadata_report(mode, lock, lock_hash):
    from modules import phase14_calibration as phase14
    from evaluate_rf_final_test import BIN_EDGES
    calibration = mode == "calibration"
    rows, positives = (2060, 153) if calibration else (4, 2)
    edges = phase14.BIN_EDGES if calibration else BIN_EDGES
    bins = [{"index": index, "lower": float(edges[index]), "upper": float(edges[index+1]),
             "endpoint_rule": "lower inclusive; upper exclusive except bin 9 includes 1.0", "count": rows if index == 1 else 0,
             "mean_predicted_probability": 0.15 if index == 1 else None,
             "observed_positive_fraction": positives / rows if index == 1 else None,
             "status": "defined" if index == 1 else "empty"} for index in range(10)]
    report = {"schema_version": "phase14a_calibration_report.v1" if calibration else
              "phase14a_final_test_report.v1" if mode == "primary" else "phase14a_verification_metrics.v1",
              "metrics": {name: {"value": value, "status": "defined", "reason": None}
                          for name, value in (("average_precision", 0.5), ("roc_auc", 0.5), ("brier_score", 0.25), ("log_loss", 0.7))},
              "row_count": rows, "positive_count": positives, "negative_count": rows-positives,
              "prevalence": positives / rows, "classification_threshold": None, "reliability_bins": bins}
    if calibration:
        report["diagnostic_scope"] = "in-sample calibration-fit diagnostics"
    else:
        report.update(evaluation_scope="final_test", evaluation_label="untouched final test", evaluation_mode=mode,
                      identities={"rows_sha256": lock["final_rows_sha256"], "model_sha256": lock["model_sha256"],
                          "protected_attestation_sha256": lock["protected_attestation_sha256"],
                          "development_attestation_sha256": lock["development_attestation_sha256"],
                          "configuration_sha256": lock["configuration_sha256"], "evaluation_revision": lock["evaluation_revision"],
                          "pretest_lock_sha256": lock_hash})
    return report


def _metadata_calibration_manifest(code, authorization, receipt, outputs):
    from modules import phase14_calibration as phase14
    from modules.feature_pipeline import CANONICAL_FEATURE_NAMES
    provenance = {"attestation_sha256": phase14.DEVELOPMENT_ATTESTATION_SHA256,
        "calibration_csv_sha256": phase14.CALIBRATION_CSV_SHA256, "rows": 2060, "positives": 153, "negatives": 1907,
        "source_csv_sha256": phase14.SOURCE_CSV_SHA256,
        "source_revision": "8b750ae733ab0346d5cb9291780310c87c2cc752", "synthetic": False}
    receipt["inputs"] = {"role": "calibration", "synthetic": False, "provenance": provenance}
    return {"schema_version": "phase14a_calibrated_model_manifest.v1", "protocol_id": phase14.PROTOCOL_ID,
        "protocol_sha256": phase14.PROTOCOL_SHA256, "run_id": authorization["run_id"], "source_revision": "a" * 40,
        "source_clean": True, "execution_scope": "production_calibration", "approval_reference": "synthetic-approval",
        "command": receipt["argv"], "configuration_sha256": authorization["configuration_sha256"], "working_directory": str(code),
        "actual_base_model_identity": {"kind": "phase13_base_model_file", "sha256": phase14.BASE_MODEL_SHA256,
                                      "manifest_sha256": phase14.PHASE13_MANIFEST_SHA256},
        "expected_policy": {"protocol_id": phase14.PROTOCOL_ID, "protocol_sha256": phase14.PROTOCOL_SHA256,
                            "phase13_model_sha256": phase14.BASE_MODEL_SHA256, "phase13_manifest_sha256": phase14.PHASE13_MANIFEST_SHA256},
        "output_paths": {"calibrated_model": "base_rf_calibrated.joblib", "manifest": "calibration_manifest.json",
                         "report": "calibration_report.json", "execution_receipt": "execution_receipt.json"},
        "phase13_expected_policy": {"run_id": phase14.PHASE13_RUN_ID, "source_revision": phase14.PHASE13_REVISION,
            "protocol_sha256": phase14.PHASE13_PROTOCOL_SHA256, "config_sha256": phase14.PHASE13_CONFIG_SHA256,
            "manifest_sha256": phase14.PHASE13_MANIFEST_SHA256, "base_model_sha256": phase14.BASE_MODEL_SHA256},
        "method": "sigmoid", "method_decision_reference": phase14.DECISION_REFERENCE, "feature_schema_version": "set_c.v1",
        "feature_names": list(CANONICAL_FEATURE_NAMES), "target_name": "Ignited", "target_version": "ignition_t_plus_1.v1",
        "positive_label": 1, "input_role": "calibration", "input": provenance, "software": authorization["environment"]["versions"],
        "forest_fingerprint": {"before_sha256": "1" * 64, "after_sha256": "1" * 64, "embedded_sha256": "1" * 64, "positive_class_index": 1},
        "calibrated_model": {"filename": "base_rf_calibrated.joblib", "sha256": outputs["base_rf_calibrated.joblib"]},
        "report": {"filename": "calibration_report.json", "sha256": outputs["calibration_report.json"]}, "status": "complete",
        "started_at_utc": "2026-10-10T00:00:02+00:00", "ended_at_utc": "2026-10-10T00:00:59+00:00",
        "computation_started_at_local": "2026-10-10T08:00:02+08:00", "computation_ended_at_local": "2026-10-10T08:00:59+08:00",
        "local_timezone": "UTC+08:00"}


def _finalized_metadata_fixture(tmp_path, review_kind, *, runner_source=None):
    """Production-shaped metadata only, wholly synthetic and under the test's OS-temp path."""
    code, base_authorization, lock, lock_hash = _context_tree(
        tmp_path, "calibration" if review_kind == "calibration" else "final_primary", runner_source=runner_source)
    root = code / "output" / "phase14"
    jobs = [("calibration", "synthetic-calibration")] if review_kind == "calibration" else [
        ("final_primary", "synthetic-primary"), ("final_verification", "synthetic-verification")]
    records = {}
    for job, run in jobs:
        evidence = root / "evidence" / run
        destination = (root / "phase14-setc-rf-61e8f740-r1" / "calibration" / run if job == "calibration" else
                       root / "final-test" / lock_hash[:12] / job.removeprefix("final_") / run)
        evidence.mkdir(parents=True)
        destination.mkdir(parents=True)
        authorization = json.loads(json.dumps(base_authorization))
        authorization.update(job=job, run_id=run)
        if job == "final_verification":
            primary = records["synthetic-primary"]
            authorization["primary_evidence"] = {"run_id": "synthetic-primary", "report_sha256": primary["outputs"]["final_test_report.json"],
                "receipt_sha256": sha256(primary["receipt_path"].read_bytes()).hexdigest(),
                "supervisor_completion_sha256": sha256(primary["completion_path"].read_bytes()).hexdigest()}
        raw = json.dumps(authorization)
        arguments = (["-B", "-m", "modules.phase14_calibration"] if job == "calibration" else
                     ["-B", "evaluate_rf_final_test.py", "--mode", job.removeprefix("final_")])
        arguments += ["--config", str(code / "config" / ("phase14_setc_calibration_v1.yaml" if job == "calibration" else "phase14_setc_final_test_v1.yaml")),
            "--approval-reference", "synthetic-approval", "--expected-revision", "a" * 40, "--run-id", run,
            "--expected-config-sha256", authorization["configuration_sha256"], "--evidence-dir", str(evidence)]
        handoff = {"schema_version": "phase14a_supervisor_handoff.v1", "job": job, "run_id": run,
            "approval_reference": "synthetic-approval", "expected_revision": "a" * 40, "authorization_json": raw,
            "authorization_sha256": sha256(raw.encode()).hexdigest(), "nonce": "a" * 32, "supervisor_pid": 123,
            "executable": authorization["environment"]["interpreter"]["path"], "argv": arguments, "working_directory": str(code),
            "evidence_directory": str(evidence), "destination": str(destination), "created_at_utc": "2026-10-10T00:00:00+00:00",
            "deadline_utc": "2026-10-10T00:30:00+00:00"}
        handoff_path = evidence / "supervisor_handoff.json"
        handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
        from test_phase14_calibration_contract import _worker_metadata
        process, worker_exit = _worker_metadata(evidence, handoff)
        binding_hash = sha256((evidence / "worker_binding.json").read_bytes()).hexdigest()
        binding = {"run_id": run, "nonce": handoff["nonce"], "sha256": sha256(handoff_path.read_bytes()).hexdigest(), "supervisor_pid": 123, "child_pid": 125}
        source = {"revision": "a" * 40, "clean": True, "source_files": authorization["source_files"],
                  "configuration_sha256": authorization["configuration_sha256"], "protocol_sha256": authorization["protocol_sha256"]}
        report = _metadata_report("calibration" if job == "calibration" else job.removeprefix("final_"), lock, lock_hash)
        name = "calibration_report.json" if job == "calibration" else "final_test_report.json" if job == "final_primary" else "verification_metrics.json"
        report_path = destination / name
        report_path.write_text(json.dumps(report), encoding="utf-8")
        outputs = {name: sha256(report_path.read_bytes()).hexdigest()}
        receipt = {"schema_version": "phase14a_execution_receipt.v2", "run_id": run, "approval_reference": "synthetic-approval",
            "argv": arguments, "working_directory": str(code), "source_revision": "a" * 40, "source_clean": True,
            "source_check": source, "supervisor_handoff": binding, "worker_binding_sha256": binding_hash, "configuration_sha256": authorization["configuration_sha256"],
            "environment": authorization["environment"]["versions"] if job == "calibration" else authorization["environment"],
            "inputs": {}, "computation_started_at_utc": "2026-10-10T00:00:02+00:00", "computation_ended_at_utc": "2026-10-10T00:00:59+00:00",
            "computation_started_at_local": "2026-10-10T08:00:02+08:00", "computation_ended_at_local": "2026-10-10T08:00:59+08:00",
            "local_timezone": "UTC+08:00", "computation_status": "complete" if job == "calibration" else "computation_complete",
            "failure_reason": None}
        if job == "calibration":
            model_path = destination / "base_rf_calibrated.joblib"
            model_path.write_bytes(b"synthetic bytes; never loaded as a model")
            outputs[model_path.name] = sha256(model_path.read_bytes()).hexdigest()
            receipt["execution_scope"] = "production_calibration"
            manifest = _metadata_calibration_manifest(code, authorization, receipt, outputs)
            manifest_path = destination / "calibration_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            outputs[manifest_path.name] = sha256(manifest_path.read_bytes()).hexdigest()
        else:
            receipt.update(model_sha256=lock["model_sha256"], final_rows_sha256=lock["final_rows_sha256"],
                inputs={"rows_sha256": lock["final_rows_sha256"], "model_sha256": lock["model_sha256"],
                        "protected_attestation_sha256": lock["protected_attestation_sha256"],
                        "development_attestation_sha256": lock["development_attestation_sha256"]})
        receipt.update(finalized_outputs=outputs, partial_output_inventory=sorted(outputs))
        receipt_path = destination / "execution_receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        completion = {"schema_version": "phase14a_supervisor_completion.v2", "job": job, "run_id": run,
            "approval_reference": "synthetic-approval", "source_check": source, "supervisor_handoff": binding,
            "worker_binding_sha256": binding_hash, "worker_exit": worker_exit,
            "pretest_lock_sha256": lock_hash, "child_receipt": {"status": "present", "path": str(receipt_path), "sha256": sha256(receipt_path.read_bytes()).hexdigest()},
            "verified_outputs": outputs, "actual_exit_status": 0, "stop_reason": "completed", "capture_complete": True,
            "capture_status": "complete", "capture_error": None, "capture_drain_timed_out": False, "capture_drain_timeout_ms": 1000,
            "process_ids": [124], "remaining_process_ids": [], "process_exited": True, "resource_samples": [],
            "runner_error": None, "cleanup_error": None, "finalization_error": None, "started_at_utc": handoff["created_at_utc"],
            "ended_at_utc": "2026-10-10T00:01:00+00:00", "deadline_utc": handoff["deadline_utc"],
            "shutdown_limitation": "best-effort; synchronous OS operations may exceed deadline", "overall_status": "PASS"}
        for filename, field in (("stdout.log", "stdout_sha256"), ("stderr.log", "stderr_sha256"), ("operator_resources.jsonl", "resource_sha256")):
            path = evidence / filename
            path.write_bytes(b"")
            completion[field] = sha256(path.read_bytes()).hexdigest()
        completion_path = evidence / "supervisor_completion.json"
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
        records[run] = {"receipt_path": receipt_path, "completion_path": completion_path, "report_path": report_path,
                        "handoff_path": handoff_path, "outputs": outputs, "destination": destination, "evidence": evidence}
    return code, lock, lock_hash, records


def _review_body(code, review_kind, findings, expected_status):
    authorization = {"schema_version": "phase14a_review_authorization.v1", "review_kind": review_kind,
        "reviewer_id": "synthetic-reviewer", "approval_reference": "synthetic-review-approval",
        "calibration_run_id": "synthetic-calibration" if review_kind == "calibration" else "",
        "primary_run_id": "synthetic-primary" if review_kind == "verification" else "",
        "verification_run_id": "synthetic-verification" if review_kind == "verification" else "",
        "lock_id": "synthetic-lock" if review_kind == "verification" else "", "disposition": "PASS",
        "findings": findings, "created_at_utc": "2026-10-10T00:00:00+00:00"}
    raw = json.dumps(authorization)
    parameters = ("-CalibrationRunId synthetic-calibration" if review_kind == "calibration" else
                  "-PrimaryRunId synthetic-primary -VerificationRunId synthetic-verification -LockId synthetic-lock")
    codes = "@(" + ",".join(_q(value) for value in findings) + ")"
    return (". " + _q(code / RUNNER.name) + "\n"
        "$script:forbidden=@{modelRead=0;rowRead=0;protectedRead=0;launch=0}\n"
        "$script:hashOriginal=(Get-Command Get-PhaseFileSha256).ScriptBlock\n"
        "$script:recordOriginal=(Get-Command Read-Phase14Record).ScriptBlock\n"
        "function Assert-SyntheticRead { param($Path); if ([IO.Path]::GetExtension($Path) -eq '.joblib') { $script:forbidden.modelRead++; throw 'forbidden model content' }; if ([IO.Path]::GetExtension($Path) -eq '.csv') { $script:forbidden.rowRead++; throw 'forbidden row content' }; if ($Path -match '[\\\\/]protected[\\\\/]') { $script:forbidden.protectedRead++; throw 'forbidden protected content' } }\n"
        "function Get-PhaseFileSha256 { param($Path); Assert-SyntheticRead $Path; & $script:hashOriginal $Path }\n"
        "function Read-Phase14Record { param($Path); Assert-SyntheticRead $Path; & $script:recordOriginal $Path }\n"
        "function Start-Process { $script:forbidden.launch++; throw 'forbidden reviewer launch' }\n"
        "function New-Object { $script:forbidden.launch++; throw 'forbidden reviewer construction' }\n"
        "$env:PHASE14_REVIEW_AUTHORIZATION_JSON=" + _q(raw) + "\n"
        "$env:PHASE14_REVIEW_AUTHORIZATION_SHA256=" + _q(sha256(raw.encode()).hexdigest()) + "\n"
        "$result=Write-Phase14ReviewRecord -ReviewKind " + review_kind
        + " -ReviewerId synthetic-reviewer -ApprovalReference synthetic-review-approval " + parameters
        + " -Disposition PASS -Findings " + codes + "\n"
        "if ($env:PHASE14_PRIVATE_DIAGNOSTIC_DIR) { try { $counterRecord=[ordered]@{forbidden_calls=$script:forbidden}; $counterBytes=[Text.UTF8Encoding]::new($false,$true).GetBytes((ConvertTo-Json -InputObject $counterRecord -Depth 10 -Compress)); $counterPath=Join-Path $env:PHASE14_PRIVATE_DIAGNOSTIC_DIR ('forbidden-call-counters-'+[Guid]::NewGuid().ToString('N')+'.json'); $counterStream=[IO.File]::Open($counterPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None); try { $counterStream.Write($counterBytes,0,$counterBytes.Length); $counterStream.Flush($true) } finally { $counterStream.Dispose() } } catch { try { [Console]::Error.WriteLine('PHASE14_DIAGNOSTIC_REPORTING_ERROR=forbidden-call-counters: '+$_.Exception.GetType().FullName+': '+$_.Exception.Message) } catch {} } }\n"
        "if ($result.disposition -cne " + _q(expected_status) + ") { throw ('review status: '+$result.disposition) }\n"
        "if (@($script:forbidden.Values | Where-Object { $_ -ne 0 }).Count) { throw 'forbidden reviewer call reached' }\n")


def _instrument_review_runner(source_path, destination_path):
    text = source_path.read_text(encoding="utf-8")
    helper = r'''function Write-Phase14DiagnosticReportingError {
    param([string]$Stage,[object]$ReportingError)
    try {
        [Console]::Error.WriteLine('PHASE14_DIAGNOSTIC_REPORTING_ERROR='+$Stage+': '+$ReportingError.Exception.GetType().FullName+': '+$ReportingError.Exception.Message)
    } catch {
        # Reporting is best-effort and never participates in validation.
    }
}

function Write-Phase14PrivateOriginalException {
    param([string]$Stage,[System.Management.Automation.ErrorRecord]$ErrorRecord)
    try {
        $base=[Environment]::GetEnvironmentVariable('PHASE14_PRIVATE_DIAGNOSTIC_DIR')
        if (-not $base) { throw 'Private diagnostic directory is not configured' }
        $diagnosticId=[Guid]::NewGuid().ToString('N')
        $inner=@(); $exception=$ErrorRecord.Exception
        while ($null -ne $exception) {
            $inner+=@{type=$exception.GetType().FullName;message=$exception.Message}; $exception=$exception.InnerException
        }
        $record=[ordered]@{diagnostic_id=$diagnosticId;stage=$Stage;
            exception_type=$ErrorRecord.Exception.GetType().FullName;message=$ErrorRecord.Exception.Message;
            inner_exceptions=$inner;script_stack_trace=$ErrorRecord.ScriptStackTrace;
            invocation_position=$ErrorRecord.InvocationInfo.PositionMessage}
        $bytes=[Text.UTF8Encoding]::new($false,$true).GetBytes((ConvertTo-Json -InputObject $record -Depth 20 -Compress))
        $path=Join-Path $base ('original-'+$diagnosticId+'.json')
        $stream=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try { $stream.Write($bytes,0,$bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
        return $diagnosticId
    } catch { Write-Phase14DiagnosticReportingError ('original-exception-'+$Stage) $_ }
    return $null
}

function Write-Phase14PrivateOperands {
    param([string]$DiagnosticId,[string]$Stage,[string]$RunId,[string]$LockId,[object]$Operands)
    try {
        $base=[Environment]::GetEnvironmentVariable('PHASE14_PRIVATE_DIAGNOSTIC_DIR')
        if (-not $base) { throw 'Private diagnostic directory is not configured' }
        $recordId=if ($DiagnosticId) { $DiagnosticId } else { [Guid]::NewGuid().ToString('N') }
        $record=[ordered]@{diagnostic_id=$DiagnosticId;stage=$Stage;run_id=$RunId;lock_id=$LockId;operands=$Operands}
        $bytes=[Text.UTF8Encoding]::new($false,$true).GetBytes((ConvertTo-Json -InputObject $record -Depth 30 -Compress))
        $path=Join-Path $base ('operands-'+$Stage+'-'+$recordId+'.json')
        $stream=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try { $stream.Write($bytes,0,$bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
    } catch { Write-Phase14DiagnosticReportingError ('operands-'+$Stage) $_ }
}

function Get-Phase14PrivateArrayShape {
    param([object]$Value)
    if ($null -eq $Value) { return @{is_null=$true;type=$null;is_array=$false;count=$null;elements=@()} }
    if ($Value -isnot [array]) { return @{is_null=$false;type=$Value.GetType().FullName;is_array=$false;count=$null;value=$Value;elements=@()} }
    $elements=@()
    foreach ($element in $Value) {
        $elements+=@{type=if ($null -eq $element) { $null } else { $element.GetType().FullName };value=$element}
    }
    return @{is_null=$false;type=$Value.GetType().FullName;is_array=$true;count=$Value.Count;elements=$elements}
}

function Get-Phase14PrivatePolicyShape {
    param([object]$Value,[string[]]$ArrayFields)
    if ($null -eq $Value) { return @{is_null=$true;type=$null;array_fields=@{}} }
    $fields=@{}
    foreach ($field in $ArrayFields) { $fields[$field]=Get-Phase14PrivateArrayShape $Value.$field }
    return @{is_null=$false;type=$Value.GetType().FullName;array_fields=$fields}
}

function Get-Phase14PrivateStringDifference {
    param([AllowNull()][string]$Actual,[AllowNull()][string]$Expected)
    $actualLength=if ($null -eq $Actual) { 0 } else { $Actual.Length }
    $expectedLength=if ($null -eq $Expected) { 0 } else { $Expected.Length }
    $limit=[Math]::Min($actualLength,$expectedLength); $index=0
    while ($index -lt $limit -and $Actual[$index] -ceq $Expected[$index]) { $index++ }
    if ($index -eq $limit -and $actualLength -eq $expectedLength) { $index=$null }
    $start=if ($null -eq $index) { 0 } else { [Math]::Max(0,$index-24) }
    $actualContext=$null; $expectedContext=$null
    if ($null -ne $index) {
        if ($null -ne $Actual) { $actualContext=$Actual.Substring($start,[Math]::Min($Actual.Length-$start,48)) }
        if ($null -ne $Expected) { $expectedContext=$Expected.Substring($start,[Math]::Min($Expected.Length-$start,48)) }
    }
    return @{first_differing_character=$index;actual_length=$actualLength;expected_length=$expectedLength;
        actual_context=$actualContext;expected_context=$expectedContext}
}

function Write-Phase14PrivateLockComparison {
    param([string]$RunId,[string]$LockId,[string]$ActualPredictionJson,[string]$ExpectedPredictionJson,
        [bool]$PredictionMismatch,[bool]$ReportingEvaluated,[AllowNull()][string]$ActualReportingJson,
        [AllowNull()][string]$ExpectedReportingJson,[AllowNull()][bool]$ReportingMismatch,
        [object]$ActualPrediction,[object]$ExpectedPrediction,[object]$ActualReporting,[object]$ExpectedReporting)
    try {
        $base=[Environment]::GetEnvironmentVariable('PHASE14_PRIVATE_DIAGNOSTIC_DIR')
        if (-not $base) { throw 'Private diagnostic directory is not configured' }
        $reportingDifference=$null
        if ($ReportingEvaluated) { $reportingDifference=Get-Phase14PrivateStringDifference $ActualReportingJson $ExpectedReportingJson }
        $record=[ordered]@{run_id=$RunId;lock_id=$LockId;actual_prediction_json=$ActualPredictionJson;
            expected_prediction_json=$ExpectedPredictionJson;prediction_mismatch=$PredictionMismatch;
            reporting_evaluated=$ReportingEvaluated;actual_reporting_json=$ActualReportingJson;
            expected_reporting_json=$ExpectedReportingJson;reporting_mismatch=$ReportingMismatch;
            prediction_difference=(Get-Phase14PrivateStringDifference $ActualPredictionJson $ExpectedPredictionJson);
            reporting_difference=$reportingDifference;
            actual_prediction_shape=(Get-Phase14PrivatePolicyShape $ActualPrediction @('feature_names','required_labels'));
            expected_prediction_shape=(Get-Phase14PrivatePolicyShape $ExpectedPrediction @('feature_names','required_labels'));
            actual_reporting_shape=if ($ReportingEvaluated) { Get-Phase14PrivatePolicyShape $ActualReporting @('metrics','log_loss_labels','bin_edges') } else { $null };
            expected_reporting_shape=if ($ReportingEvaluated) { Get-Phase14PrivatePolicyShape $ExpectedReporting @('metrics','log_loss_labels','bin_edges') } else { $null }}
        $bytes=[Text.UTF8Encoding]::new($false,$true).GetBytes((ConvertTo-Json -InputObject $record -Depth 40 -Compress))
        $path=Join-Path $base ('lock-comparison-'+[Guid]::NewGuid().ToString('N')+'.json')
        $stream=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try { $stream.Write($bytes,0,$bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
    } catch { Write-Phase14DiagnosticReportingError 'lock-comparison' $_ }
}

function Write-Phase14PrivateEndpointFailure {
    param([object]$Snapshot)
    try {
        $base=[Environment]::GetEnvironmentVariable('PHASE14_PRIVATE_DIAGNOSTIC_DIR')
        if (-not $base) { throw 'Private diagnostic directory is not configured' }
        $record=[ordered]@{stage='report-bin-endpoint-validation';snapshot=$Snapshot}
        $bytes=[Text.UTF8Encoding]::new($false,$true).GetBytes((ConvertTo-Json -InputObject $record -Depth 30 -Compress))
        $path=Join-Path $base ('endpoint-failure-'+[Guid]::NewGuid().ToString('N')+'.json')
        $stream=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try { $stream.Write($bytes,0,$bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
    } catch { Write-Phase14DiagnosticReportingError 'endpoint-failure' $_ }
}
'''
    marker = "function Test-Phase14ChildReceipt {"
    if text.count(marker) != 1:
        raise AssertionError("Expected one Test-Phase14ChildReceipt definition")
    text = text.replace(marker, helper + marker, 1)

    child_catch = "    catch { return @{valid=$false; receipt=$identity; outputs=@{}; handoff=$null; error=('child receipt validation ('+$_.Exception.GetType().Name+')')} }"
    child_observer = r'''    catch {
        $phase14OriginalError=$_
        $phase14DiagnosticId=Write-Phase14PrivateOriginalException 'child-receipt' $phase14OriginalError
        try {
            $phase14Operands=@{path=$path;path_type=if ($null -eq $path) { $null } else { $path.GetType().FullName };
                review_only=[bool]$ReviewOnly;context_type=if ($null -eq $Context) { $null } else { $Context.GetType().FullName };
                receipt=$receipt;receipt_type=if ($null -eq $receipt) { $null } else { $receipt.GetType().FullName };
                handoff=$Handoff;handoff_type=if ($null -eq $Handoff) { $null } else { $Handoff.GetType().FullName };
                process=$process;process_type=if ($null -eq $process) { $null } else { $process.GetType().FullName };
                worker_binding=$workerBinding;worker_binding_type=if ($null -eq $workerBinding) { $null } else { $workerBinding.GetType().FullName };
                expected_environment=$expectedEnvironment;expected_environment_type=if ($null -eq $expectedEnvironment) { $null } else { $expectedEnvironment.GetType().FullName };
                binding=$binding;binding_type=if ($null -eq $binding) { $null } else { $binding.GetType().FullName };
                computation_start=$computationStart;computation_start_type=if ($null -eq $computationStart) { $null } else { $computationStart.GetType().FullName };
                computation_end=$computationEnd;computation_end_type=if ($null -eq $computationEnd) { $null } else { $computationEnd.GetType().FullName };
                operation_deadline=$operationDeadline;operation_deadline_type=if ($null -eq $operationDeadline) { $null } else { $operationDeadline.GetType().FullName };
                run_id=$Context.run_id;destination=$Context.destination;evidence=$Context.evidence;identity=$identity}
            Write-Phase14PrivateOperands $phase14DiagnosticId 'child-receipt' $Context.run_id $Context.authorization.pretest_lock.lock_id $phase14Operands
        } catch { Write-Phase14DiagnosticReportingError 'child-receipt-operands' $_ }
        return @{valid=$false; receipt=$identity; outputs=@{}; handoff=$null; error=('child receipt validation ('+$phase14OriginalError.Exception.GetType().Name+')')}
    }'''
    if text.count(child_catch) != 1:
        raise AssertionError("Expected one child-receipt validation catch")
    text = text.replace(child_catch, child_observer, 1)

    review_catch = "            } catch { $observedFindings+='evidence_invalid' }"
    lock_observer = r'''            } catch {
                $phase14OriginalError=$_
                $phase14DiagnosticId=Write-Phase14PrivateOriginalException 'verification-lock' $phase14OriginalError
                try {
                    $phase14Operands=@{lock_type=if ($null -eq $lock) { $null } else { $lock.GetType().FullName };
                        authorization_type=if ($null -eq $primaryAuthorization) { $null } else { $primaryAuthorization.GetType().FullName };
                        primary_run_id=$PrimaryRunId;lock_id=$LockId;lock=$lock;authorization=$primaryAuthorization;
                        primary_handoff_path=$primaryHandoffPath;
                        primary_handoff_type=if ($null -eq $primaryHandoff) { $null } else { $primaryHandoff.GetType().FullName };
                        primary_authorization_type=if ($null -eq $primaryAuthorization) { $null } else { $primaryAuthorization.GetType().FullName }}
                    Write-Phase14PrivateOperands $phase14DiagnosticId 'verification-lock' $PrimaryRunId $LockId $phase14Operands
                } catch { Write-Phase14DiagnosticReportingError 'verification-lock-operands' $_ }
                $observedFindings+='evidence_invalid'
            }'''
    run_observer = r'''            } catch {
                $phase14OriginalError=$_
                $phase14DiagnosticId=Write-Phase14PrivateOriginalException 'verification-run' $phase14OriginalError
                try {
                    $phase14Operands=@{job=$job;run_id=$run;lock_id=$LockId;
                        job_type=if ($null -eq $job) { $null } else { $job.GetType().FullName };
                        lock_type=if ($null -eq $lock) { $null } else { $lock.GetType().FullName };
                        authorization_type=if ($null -eq $authorization) { $null } else { $authorization.GetType().FullName };
                        job_value=$job;lock=$lock;authorization=$authorization;
                        completion=$completion;handoff=$handoff;source=$source;arguments=$arguments;context=$context;
                        package=$package;receipt=$receipt;report=$report;expected_primary=$expectedPrimary;
                        completion_type=if ($null -eq $completion) { $null } else { $completion.GetType().FullName };
                        handoff_type=if ($null -eq $handoff) { $null } else { $handoff.GetType().FullName };
                        source_type=if ($null -eq $source) { $null } else { $source.GetType().FullName };
                        arguments_type=if ($null -eq $arguments) { $null } else { $arguments.GetType().FullName };
                        context_type=if ($null -eq $context) { $null } else { $context.GetType().FullName };
                        package_type=if ($null -eq $package) { $null } else { $package.GetType().FullName };
                        receipt_type=if ($null -eq $receipt) { $null } else { $receipt.GetType().FullName };
                        report_type=if ($null -eq $report) { $null } else { $report.GetType().FullName };
                        expected_primary_type=if ($null -eq $expectedPrimary) { $null } else { $expectedPrimary.GetType().FullName }}
                    Write-Phase14PrivateOperands $phase14DiagnosticId 'verification-run' $run $LockId $phase14Operands
                } catch { Write-Phase14DiagnosticReportingError 'verification-run-operands' $_ }
                $observedFindings+='evidence_invalid'
            }'''
    if text.count(review_catch) != 2:
        raise AssertionError("Expected the two reviewed evidence-invalid catches")
    text = text.replace(review_catch, lock_observer, 1)
    text = text.replace(review_catch, run_observer, 1)

    compare = "    if ((Get-Phase14CanonicalJson $Lock.prediction_contract) -cne (Get-Phase14CanonicalJson $prediction) -or (Get-Phase14CanonicalJson $Lock.reporting) -cne (Get-Phase14CanonicalJson $reporting)) { throw 'Lock prediction/reporting policy mismatch' }"
    observed_compare = r'''    $actualPredictionJson=Get-Phase14CanonicalJson $Lock.prediction_contract
    $expectedPredictionJson=Get-Phase14CanonicalJson $prediction
    $predictionMismatch=$actualPredictionJson -cne $expectedPredictionJson
    $reportingEvaluated=$false; $actualReportingJson=$null; $expectedReportingJson=$null; $reportingMismatch=$false
    $actualReportingPolicy=$null; $expectedReportingPolicy=$null
    if (-not $predictionMismatch) {
        $actualReportingJson=Get-Phase14CanonicalJson $Lock.reporting
        $expectedReportingJson=Get-Phase14CanonicalJson $reporting
        $actualReportingPolicy=$Lock.reporting; $expectedReportingPolicy=$reporting
        $reportingMismatch=$actualReportingJson -cne $expectedReportingJson
        $reportingEvaluated=$true
    }
    try {
        Write-Phase14PrivateLockComparison $Authorization.run_id $Lock.lock_id $actualPredictionJson $expectedPredictionJson `
            $predictionMismatch $reportingEvaluated $actualReportingJson $expectedReportingJson $reportingMismatch `
            $Lock.prediction_contract $prediction $actualReportingPolicy $expectedReportingPolicy
    } catch { Write-Phase14DiagnosticReportingError 'lock-comparison-invocation' $_ }
    if ($predictionMismatch -or $reportingMismatch) { throw 'Lock prediction/reporting policy mismatch' }'''
    if text.count(compare) != 1:
        raise AssertionError("Expected one original lock comparison expression")
    text = text.replace(compare, observed_compare, 1)

    endpoint_check = r"""        if (($bin.index -isnot [int] -and $bin.index -isnot [long]) -or $bin.index -ne $index -or ($bin.count -isnot [int] -and $bin.count -isnot [long]) -or $bin.count -lt 0 -or -not (Test-Phase14Number $bin.lower) -or -not (Test-Phase14Number $bin.upper) -or $bin.lower -ne $edges[$index] -or $bin.upper -ne $edges[$index+1] -or $bin.endpoint_rule -cne 'lower inclusive; upper exclusive except bin 9 includes 1.0') { throw 'Invalid bin endpoints/count' }"""
    observed_endpoint = r'''        $endpointChecks=[ordered]@{}; $endpointFailure=$null
        $endpointChecks.index_type=($bin.index -isnot [int] -and $bin.index -isnot [long])
        if ($endpointChecks.index_type) { $endpointFailure='index_type' }
        if ($null -eq $endpointFailure) {
            $endpointChecks.index_value=($bin.index -ne $index)
            if ($endpointChecks.index_value) { $endpointFailure='index_value' }
        }
        if ($null -eq $endpointFailure) {
            $endpointChecks.count_type=($bin.count -isnot [int] -and $bin.count -isnot [long])
            if ($endpointChecks.count_type) { $endpointFailure='count_type' }
        }
        if ($null -eq $endpointFailure) {
            $endpointChecks.count_negative=($bin.count -lt 0)
            if ($endpointChecks.count_negative) { $endpointFailure='count_negative' }
        }
        if ($null -eq $endpointFailure) {
            $lowerIsNumber=Test-Phase14Number $bin.lower
            $endpointChecks.lower_is_number=(-not $lowerIsNumber)
            if ($endpointChecks.lower_is_number) { $endpointFailure='lower_number' }
        }
        if ($null -eq $endpointFailure) {
            $upperIsNumber=Test-Phase14Number $bin.upper
            $endpointChecks.upper_is_number=(-not $upperIsNumber)
            if ($endpointChecks.upper_is_number) { $endpointFailure='upper_number' }
        }
        if ($null -eq $endpointFailure) {
            $endpointChecks.lower_value=($bin.lower -ne $edges[$index])
            if ($endpointChecks.lower_value) { $endpointFailure='lower_value' }
        }
        if ($null -eq $endpointFailure) {
            $endpointChecks.upper_value=($bin.upper -ne $edges[$index+1])
            if ($endpointChecks.upper_value) { $endpointFailure='upper_value' }
        }
        if ($null -eq $endpointFailure) {
            $endpointChecks.endpoint_rule=($bin.endpoint_rule -cne 'lower inclusive; upper exclusive except bin 9 includes 1.0')
            if ($endpointChecks.endpoint_rule) { $endpointFailure='endpoint_rule' }
        }
        if ($null -ne $endpointFailure) {
            try {
                $endpointSnapshot=[ordered]@{mode=$Mode;failing_index=$index;failing_index_type=$index.GetType().FullName;
                    report_bin_count=$Report.reliability_bins.Count;report_bin_collection_type=$Report.reliability_bins.GetType().FullName;
                    failed_subcondition=$endpointFailure;evaluated_conditions=$endpointChecks;
                    actual_bin=@{index=$bin.index;index_type=if ($null -eq $bin.index) { $null } else { $bin.index.GetType().FullName };
                        count=$bin.count;count_type=if ($null -eq $bin.count) { $null } else { $bin.count.GetType().FullName };
                        lower=$bin.lower;lower_type=if ($null -eq $bin.lower) { $null } else { $bin.lower.GetType().FullName };
                        upper=$bin.upper;upper_type=if ($null -eq $bin.upper) { $null } else { $bin.upper.GetType().FullName };
                        endpoint_rule=$bin.endpoint_rule;endpoint_rule_type=if ($null -eq $bin.endpoint_rule) { $null } else { $bin.endpoint_rule.GetType().FullName }};
                    expected_bin=@{index=$index;index_type=$index.GetType().FullName;
                        lower=$edges[$index];lower_type=if ($null -eq $edges[$index]) { $null } else { $edges[$index].GetType().FullName };
                        upper=$edges[$index+1];upper_type=if ($null -eq $edges[$index+1]) { $null } else { $edges[$index+1].GetType().FullName };
                        endpoint_rule='lower inclusive; upper exclusive except bin 9 includes 1.0';endpoint_rule_type='System.String'}}
                Write-Phase14PrivateEndpointFailure $endpointSnapshot
            } catch { Write-Phase14DiagnosticReportingError 'endpoint-failure-snapshot' $_ }
            throw 'Invalid bin endpoints/count'
        }'''
    if text.count(endpoint_check) != 1:
        raise AssertionError("Expected one original endpoint/count compound condition")
    text = text.replace(endpoint_check, observed_endpoint, 1)

    stage_replacements = [
        ("        Assert-Phase14Identifier $ReviewerId; Assert-Phase14Identifier $ApprovalReference",
         "        $phase14DiagnosticStage='review-authorization-identifiers'; $phase14DiagnosticRun=$null; $phase14AggregateMatch=$null; $phase14ReviewParentPath=$null\n        $lockPath=$null; $lockHash=$null; $lockVerified=$false; $reports=$null; $bindings=$null; $reviewPath=$null; $record=$null; $root=$null; $observedFindings=@()\n        Assert-Phase14Identifier $ReviewerId; Assert-Phase14Identifier $ApprovalReference"),
        ("        $auth=ConvertFrom-Phase14Json $raw",
         "        $phase14DiagnosticStage='review-authorization-parse'\n        $auth=ConvertFrom-Phase14Json $raw"),
        ("            try {\n                $lock=Read-Phase14Record $lockPath",
         "            $phase14DiagnosticStage='review-lock-read'\n            try {\n                $lock=Read-Phase14Record $lockPath\n                $phase14DiagnosticStage='review-lock-validation'"),
        ("        foreach ($run in $runs) {\n            Assert-Phase14Identifier $run",
         "        foreach ($run in $runs) {\n            $phase14DiagnosticRun=$run; $phase14DiagnosticStage='review-run-validation'\n            Assert-Phase14Identifier $run"),
        ("            try {\n                $completion=Read-Phase14Record $completionPath",
         "            $phase14DiagnosticStage='review-run-evidence-read'\n            try {\n                $completion=Read-Phase14Record $completionPath"),
        ("                $package=Test-Phase14ChildReceipt $context $handoff -ReviewOnly",
         "                $phase14DiagnosticStage='review-child-receipt-validation'\n                $package=Test-Phase14ChildReceipt $context $handoff -ReviewOnly"),
        ("                    Assert-Phase14ProbabilityReport $report ($job.Substring(6)) $lock $lockHash",
         "                    $phase14DiagnosticStage='review-probability-report-validation'\n                    Assert-Phase14ProbabilityReport $report ($job.Substring(6)) $lock $lockHash"),
        ("                    $reports[$run]=$report",
         "                    $phase14DiagnosticStage='review-run-report-assignment'\n                    $reports[$run]=$report"),
        ("        if ($ReviewKind -eq 'verification' -and $lockVerified -and $reports.ContainsKey($PrimaryRunId) -and $reports.ContainsKey($VerificationRunId)) {",
         "        $phase14DiagnosticStage='review-aggregate-eligibility'\n        if ($ReviewKind -eq 'verification' -and $lockVerified -and $reports.ContainsKey($PrimaryRunId) -and $reports.ContainsKey($VerificationRunId)) {"),
        ("            if (-not (Test-Phase14ReportsMatch $reports[$PrimaryRunId] $reports[$VerificationRunId])) { $observedFindings+='aggregate_mismatch' }",
         "            $phase14DiagnosticStage='review-aggregate-comparison'\n            $phase14AggregateMatch=Test-Phase14ReportsMatch $reports[$PrimaryRunId] $reports[$VerificationRunId]\n            $phase14DiagnosticStage='review-aggregate-comparison-complete'\n            if (-not $phase14AggregateMatch) { $observedFindings+='aggregate_mismatch' }"),
        ("        $findings=@($Findings)+@($observedFindings) | Select-Object -Unique",
         "        $phase14DiagnosticStage='review-final-status'\n        $findings=@($Findings)+@($observedFindings) | Select-Object -Unique"),
        ("        elseif ($Disposition -eq 'PASS' -and ($findings -ccontains 'evidence_missing' -or $findings -ccontains 'supervision_incomplete')) { $status='NEEDS EVIDENCE' }",
         "        elseif ($Disposition -eq 'PASS' -and ($findings -ccontains 'evidence_missing' -or $findings -ccontains 'supervision_incomplete')) { $status='NEEDS EVIDENCE' }\n        $phase14DiagnosticStage='review-final-status-complete'"),
        ("        $reviewPath=if ($ReviewKind -eq 'calibration')",
         "        $phase14DiagnosticStage='review-path-derivation'\n        $reviewPath=if ($ReviewKind -eq 'calibration')"),
        ("        $reviewPath=Assert-Phase14Path $reviewPath $root; $null=[IO.Directory]::CreateDirectory((Split-Path -Parent $reviewPath))",
         "        $phase14DiagnosticStage='review-path-validation'\n        $reviewPath=Assert-Phase14Path $reviewPath $root\n        $phase14DiagnosticStage='review-parent-path-derivation'\n        $phase14ReviewParentPath=Split-Path -Parent $reviewPath\n        $phase14DiagnosticStage='review-parent-directory-create'\n        $null=[IO.Directory]::CreateDirectory($phase14ReviewParentPath)\n        $phase14DiagnosticStage='review-parent-directory-created'"),
        ("        $record=[ordered]@{schema_version=",
         "        $phase14DiagnosticStage='review-record-construction'\n        $record=[ordered]@{schema_version="),
        ("        Write-Phase14JsonNew $reviewPath $record\n        return $record",
         "        $phase14DiagnosticStage='review-record-write'\n        Write-Phase14JsonNew $reviewPath $record\n        $phase14DiagnosticStage='review-record-written'\n        $phase14DiagnosticStage='review-return'\n        return $record"),
        ("    catch { throw ('Phase 14 review denied ('+$_.Exception.GetType().Name+')') }",
         r'''    catch {
        $phase14OriginalReviewError=$_
        $phase14ReviewDiagnosticId=Write-Phase14PrivateOriginalException 'review-outer' $phase14OriginalReviewError
        try {
            $phase14OuterCounters=@{forbidden_calls=$script:forbidden}
            Write-Phase14PrivateOperands $phase14ReviewDiagnosticId 'review-outer-counters' $phase14DiagnosticRun $LockId $phase14OuterCounters
        } catch { Write-Phase14DiagnosticReportingError 'review-outer-counters' $_ }
        try {
            $phase14ReviewSnapshot=@{
                last_reached_stage=$phase14DiagnosticStage;current_run_id=$phase14DiagnosticRun;
                review_kind=$ReviewKind;reviewer_id=$ReviewerId;approval_reference=$ApprovalReference;
                calibration_run_id=$CalibrationRunId;primary_run_id=$PrimaryRunId;verification_run_id=$VerificationRunId;lock_id=$LockId;
                lock_path=$lockPath;lock_path_type=if ($null -eq $lockPath) { $null } else { $lockPath.GetType().FullName };
                lock_hash=$lockHash;lock_verified=$lockVerified;
                report_keys=if ($null -eq $reports) { @() } else { @($reports.Keys) };
                binding_keys=if ($null -eq $bindings) { @() } else { @($bindings.Keys) };
                aggregate_match=$phase14AggregateMatch;aggregate_match_type=if ($null -eq $phase14AggregateMatch) { $null } else { $phase14AggregateMatch.GetType().FullName };
                review_path=$reviewPath;review_path_type=if ($null -eq $reviewPath) { $null } else { $reviewPath.GetType().FullName };
                review_path_length=if ($null -eq $reviewPath) { $null } else { $reviewPath.Length };
                parent_path=$phase14ReviewParentPath;parent_path_type=if ($null -eq $phase14ReviewParentPath) { $null } else { $phase14ReviewParentPath.GetType().FullName };
                parent_path_length=if ($null -eq $phase14ReviewParentPath) { $null } else { $phase14ReviewParentPath.Length };
                record_keys=if ($null -eq $record) { @() } else { @($record.Keys) };
                record_type=if ($null -eq $record) { $null } else { $record.GetType().FullName };
                root=$root;root_type=if ($null -eq $root) { $null } else { $root.GetType().FullName };
                observed_findings=$observedFindings;observed_findings_type=if ($null -eq $observedFindings) { $null } else { $observedFindings.GetType().FullName }
            }
            Write-Phase14PrivateOperands $phase14ReviewDiagnosticId 'review-outer-state' $phase14DiagnosticRun $LockId $phase14ReviewSnapshot
        } catch { Write-Phase14DiagnosticReportingError 'review-outer-state' $_ }
        throw ('Phase 14 review denied ('+$phase14OriginalReviewError.Exception.GetType().Name+')')
    }'''),
    ]
    for original, observed in stage_replacements:
        if text.count(original) != 1:
            raise AssertionError("Expected one source occurrence for review-stage instrumentation: " + original[:100])
        text = text.replace(original, observed, 1)
    destination_path.parent.mkdir(parents=True, exist_ok=False)
    (destination_path.parent / "modules").mkdir()
    shutil.copyfile(source_path.parent / "modules" / "phase_supervision.psm1",
                    destination_path.parent / "modules" / "phase_supervision.psm1")
    destination_path.write_text(text, encoding="utf-8", newline="")
    return destination_path


@pytest.mark.parametrize("review_kind", ["calibration", "verification"])
def test_separate_successful_review_binds_finalized_evidence_without_rewriting_or_model_reads(tmp_path, review_kind):
    diagnostic_path = None
    runner_source = None
    if review_kind == "verification":
        diagnostic_path = tmp_path / "private-diagnostics"
        runner_source = _instrument_review_runner(RUNNER, tmp_path / "instrumented-runner" / RUNNER.name)
    code, lock, lock_hash, records = _finalized_metadata_fixture(tmp_path, review_kind, runner_source=runner_source)
    paths = [record[key] for record in records.values() for key in ("receipt_path", "completion_path")]
    before = {path: sha256(path.read_bytes()).hexdigest() for path in paths}
    body = _review_body(code, review_kind, ["evidence_verified"], "PASS")
    if diagnostic_path is not None:
        diagnostic_path.mkdir()
        body = body.replace("$result=Write-Phase14ReviewRecord",
                            "$env:PHASE14_PRIVATE_DIAGNOSTIC_DIR=" + _q(diagnostic_path) + "\n$result=Write-Phase14ReviewRecord", 1)
    body += "$denied=$false; try { " + body[body.index("$result=Write-Phase14ReviewRecord"):body.index("\nif ($result.disposition")] + " } catch { $denied=$true }; if (-not $denied) { throw 'review overwritten' }\n"
    result = _harness(tmp_path, body, load=False)
    assert result.returncode == 0, result.stderr
    assert {path: sha256(path.read_bytes()).hexdigest() for path in paths} == before
    review_path = (code / "output" / "phase14" / "calibration-review" / "synthetic-calibration" / "review_record.json"
                   if review_kind == "calibration" else records["synthetic-verification"]["destination"] / "review_record.json")
    review = json.loads(review_path.read_text())
    assert review["disposition"] == "PASS"
    for run, record in records.items():
        bound = review["evidence"][run]
        assert bound["child_receipt"] == {"status": "present", "path": str(record["receipt_path"]), "sha256": before[record["receipt_path"]]}
        assert bound["supervisor_completion"] == {"status": "present", "path": str(record["completion_path"]), "sha256": before[record["completion_path"]]}
    if review_kind == "verification":
        assert review["identities"] == {key: lock[key] for key in ("model_sha256", "final_rows_sha256", "configuration_sha256",
            "evaluation_revision", "evaluator_source_sha256", "environment")} | {"environment_sha256": _canonical_digest(lock["environment"])}
        comparison_records = [json.loads(path.read_text()) for path in diagnostic_path.glob("lock-comparison-*.json")]
        assert len(comparison_records) == 6
        assert not list(diagnostic_path.glob("endpoint-failure-*.json"))
        for comparison in comparison_records:
            assert comparison["prediction_mismatch"] is False
            assert comparison["reporting_evaluated"] is True
            assert comparison["reporting_mismatch"] is False
            assert comparison["actual_reporting_json"] == comparison["expected_reporting_json"]
            for policy_key in ("actual_reporting_shape", "expected_reporting_shape"):
                bin_edges = comparison[policy_key]["array_fields"]["bin_edges"]
                assert bin_edges["is_array"] is True
                assert bin_edges["count"] == 11
                assert len(bin_edges["elements"]) == 11
                assert all(not element["type"].endswith("[]") for element in bin_edges["elements"])
    else:
        assert review["identities"]["model_sha256"] == records["synthetic-calibration"]["outputs"]["base_rf_calibrated.joblib"]
        manifest = json.loads((records["synthetic-calibration"]["destination"] / "calibration_manifest.json").read_text())
        assert manifest["input"]["source_revision"] == "8b750ae733ab0346d5cb9291780310c87c2cc752"
        assert manifest["phase13_expected_policy"]["source_revision"] == "61e8f74031d61c9be22a9b6408aec3ad74f4e892"


@pytest.mark.parametrize("change", [
    "export-revision", "training-revision", "swapped-revisions", "receipt-export-revision",
])
def test_calibration_review_rejects_altered_or_conflated_lineage(tmp_path, change):
    code, _, _, records = _finalized_metadata_fixture(tmp_path, "calibration")
    record = records["synthetic-calibration"]
    manifest_path = record["destination"] / "calibration_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(record["receipt_path"].read_text())
    completion = json.loads(record["completion_path"].read_text())
    export_revision = manifest["input"]["source_revision"]
    training_revision = manifest["phase13_expected_policy"]["source_revision"]
    assert export_revision != training_revision
    if change == "export-revision":
        manifest["input"]["source_revision"] = "0" * 40
        receipt["inputs"]["provenance"]["source_revision"] = "0" * 40
    elif change == "training-revision":
        manifest["phase13_expected_policy"]["source_revision"] = export_revision
    elif change == "swapped-revisions":
        manifest["input"]["source_revision"] = training_revision
        receipt["inputs"]["provenance"]["source_revision"] = training_revision
        manifest["phase13_expected_policy"]["source_revision"] = export_revision
    else:
        receipt["inputs"]["provenance"]["source_revision"] = "0" * 40

    # Rebind every affected hash so lineage, not a stale digest, must be rejected.
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt["finalized_outputs"][manifest_path.name] = sha256(manifest_path.read_bytes()).hexdigest()
    record["receipt_path"].write_text(json.dumps(receipt), encoding="utf-8")
    completion["child_receipt"]["sha256"] = sha256(record["receipt_path"].read_bytes()).hexdigest()
    completion["verified_outputs"] = receipt["finalized_outputs"]
    record["completion_path"].write_text(json.dumps(completion), encoding="utf-8")
    paths = [*record["destination"].iterdir(), *record["evidence"].iterdir()]
    before = {path: sha256(path.read_bytes()).hexdigest() for path in paths}
    result = _harness(tmp_path, _review_body(code, "calibration", ["evidence_verified"], "FAIL"), load=False)
    assert result.returncode == 0, result.stderr
    assert {path: sha256(path.read_bytes()).hexdigest() for path in paths} == before
    review_path = code / "output" / "phase14" / "calibration-review" / "synthetic-calibration" / "review_record.json"
    review = json.loads(review_path.read_text())
    assert review["disposition"] == "FAIL"
    assert "evidence_invalid" in review["findings"]


@pytest.mark.parametrize("change,status", [
    ("lock-schema", "FAIL"), ("lock-policy", "FAIL"), ("lock-bin-edge", "FAIL"),
    ("lock-bin-edge-tiny", "FAIL"), ("lock-bin-edge-type", "FAIL"), ("lock-bin-edge-nested", "FAIL"),
    ("report-schema", "FAIL"), ("report-counts", "FAIL"),
    ("report-metric", "FAIL"), ("report-bin", "FAIL"),
    ("report-bin-upper", "FAIL"), ("report-bin-lower", "FAIL"),
    ("report-bin-upper-tiny", "FAIL"), ("report-bin-upper-type", "FAIL"), ("report-bin-count-type", "FAIL"),
    ("report-extra", "FAIL"), ("report-identity", "FAIL"), ("aggregate-mismatch", "FAIL"),
    *[(f"report-identity-{key}", "FAIL") for key in ("rows_sha256", "protected_attestation_sha256",
        "development_attestation_sha256", "configuration_sha256", "evaluation_revision", "pretest_lock_sha256")],
    ("handoff-job", "FAIL"), ("handoff-argv", "FAIL"), ("handoff-nonce", "FAIL"), ("process-id", "FAIL"),
    ("receipt-inputs", "FAIL"), ("source-binding", "FAIL"), ("environment-binding", "FAIL"),
    ("completion-missing", "NEEDS EVIDENCE"), ("receipt-missing", "NEEDS EVIDENCE"),
    ("supervision-incomplete", "NEEDS EVIDENCE"),
    ("finding-mismatch", "FAIL"), ("finding-invalid", "FAIL"), ("finding-missing", "NEEDS EVIDENCE"),
    ("finding-incomplete", "NEEDS EVIDENCE")])
def test_review_reconciles_invalid_missing_and_contradictory_evidence(tmp_path, change, status):
    code, lock, lock_hash, records = _finalized_metadata_fixture(tmp_path, "verification")
    record = records["synthetic-verification"]
    receipt = json.loads(record["receipt_path"].read_text())
    completion = json.loads(record["completion_path"].read_text())
    report = json.loads(record["report_path"].read_text())
    handoff = json.loads(record["handoff_path"].read_text())
    findings = ["evidence_verified"]
    if change.startswith("finding-"):
        findings.append({"finding-mismatch": "aggregate_mismatch", "finding-invalid": "evidence_invalid",
                         "finding-missing": "evidence_missing", "finding-incomplete": "supervision_incomplete"}[change])
    if change == "lock-schema": lock["schema_version"] = "wrong"
    if change == "lock-policy": lock["reporting"]["classification_threshold"] = 0.5
    if change == "lock-bin-edge": lock["reporting"]["bin_edges"][3] = 0.31
    if change == "lock-bin-edge-tiny": lock["reporting"]["bin_edges"][3] = 0.3000000000001
    if change == "lock-bin-edge-type": lock["reporting"]["bin_edges"][3] = "0.3"
    if change == "lock-bin-edge-nested": lock["reporting"]["bin_edges"] = [lock["reporting"]["bin_edges"]]
    if change.startswith("lock-"):
        (code / "output" / "phase14" / "model-lock" / "synthetic-lock.json").write_text(json.dumps(lock), encoding="utf-8")
    if change == "report-schema": report["schema_version"] = "wrong"
    if change == "report-counts": report["positive_count"] = 3
    if change == "report-metric": report["metrics"]["average_precision"]["value"] = None
    if change == "report-bin": report["reliability_bins"][0]["status"] = "defined"
    if change == "report-bin-upper": report["reliability_bins"][2]["upper"] = 0.31
    if change == "report-bin-lower": report["reliability_bins"][2]["lower"] = 0.21
    if change == "report-bin-upper-tiny": report["reliability_bins"][2]["upper"] = 0.3000000000001
    if change == "report-bin-upper-type": report["reliability_bins"][2]["upper"] = "0.30000000000000004"
    if change == "report-bin-count-type": report["reliability_bins"][2]["count"] = 0.0
    if change == "report-identity": report["identities"]["model_sha256"] = "0" * 64
    if change == "report-extra": report["unexpected"] = "PRIVATE_AGGREGATE"
    if change.startswith("report-identity-"):
        key = change.removeprefix("report-identity-")
        report["identities"][key] = "0" * (40 if key == "evaluation_revision" else 64)
    if change == "aggregate-mismatch": report["metrics"]["average_precision"]["value"] = 0.6
    record["report_path"].write_text(json.dumps(report), encoding="utf-8")
    receipt["finalized_outputs"][record["report_path"].name] = sha256(record["report_path"].read_bytes()).hexdigest()
    if change == "receipt-inputs": receipt["inputs"]["model_sha256"] = "0" * 64
    if change == "handoff-job": handoff["job"] = "calibration"
    if change == "handoff-argv": handoff["argv"] = []
    if change == "handoff-nonce": handoff["nonce"] = "b" * 32
    if change in {"source-binding", "environment-binding"}:
        auth = json.loads(handoff["authorization_json"])
        if change == "source-binding": auth["source_files"]["Thesis_RF/Code/modules/feature_pipeline.py"] = "0" * 64
        else:
            auth["environment"]["versions"]["numpy"] = "other"
            auth["environment"]["lock_sha256"] = _canonical_digest(auth["environment"]["versions"])
            auth["environment_sha256"] = _canonical_digest(auth["environment"])
        handoff["authorization_json"] = json.dumps(auth)
        handoff["authorization_sha256"] = sha256(handoff["authorization_json"].encode()).hexdigest()
    record["handoff_path"].write_text(json.dumps(handoff), encoding="utf-8")
    receipt["supervisor_handoff"]["sha256"] = sha256(record["handoff_path"].read_bytes()).hexdigest()
    completion["supervisor_handoff"] = receipt["supervisor_handoff"]
    receipt_path = record["receipt_path"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    completion["child_receipt"]["sha256"] = sha256(receipt_path.read_bytes()).hexdigest()
    completion["verified_outputs"] = receipt["finalized_outputs"]
    if change == "supervision-incomplete": completion["overall_status"] = "NEEDS EVIDENCE"
    if change == "process-id":
        process_path = record["evidence"] / "process.json"
        process = json.loads(process_path.read_text())
        process["process_id"] = 125
        process_path.write_text(json.dumps(process), encoding="utf-8")
    completion_path = record["completion_path"]
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    if change == "completion-missing": completion_path.unlink()  # OS-temp fixture only
    if change == "receipt-missing": receipt_path.unlink()  # OS-temp fixture only
    paths = [item[key] for item in records.values() for key in ("receipt_path", "completion_path") if item[key].exists()]
    before = {path: sha256(path.read_bytes()).hexdigest() for path in paths}
    result = _harness(tmp_path, _review_body(code, "verification", findings, status), load=False)
    assert result.returncode == 0, result.stderr
    assert "PRIVATE_AGGREGATE" not in result.stdout + result.stderr
    assert {path: sha256(path.read_bytes()).hexdigest() for path in paths} == before
    # Malformed locks change the destination prefix; this is metadata-only OS-temp inventory.
    reviews = list((code / "output" / "phase14" / "final-test").glob("*/verification/synthetic-verification/review_record.json"))
    assert len(reviews) == 1
    review = json.loads(reviews[0].read_text())
    assert review["disposition"] == status
    if change == "completion-missing":
        bound = review["evidence"]["synthetic-verification"]
        assert bound["child_receipt"] == {"status": "present", "path": str(receipt_path), "sha256": before[receipt_path]}
        assert bound["supervisor_completion"] == {"status": "missing", "path": None, "sha256": None}

    if change == "receipt-missing":
        bound = review["evidence"]["synthetic-verification"]
        assert bound["child_receipt"] == {"status": "missing", "path": None, "sha256": None}
        assert bound["supervisor_completion"] == {"status": "present", "path": str(completion_path), "sha256": before[completion_path]}


@pytest.mark.parametrize("change", ["schema", "missing-field", "extra-field"])
def test_calibration_review_rejects_malformed_manifest_with_rebound_evidence(tmp_path, change):
    code, _, _, records = _finalized_metadata_fixture(tmp_path, "calibration")
    record = records["synthetic-calibration"]
    manifest_path = record["destination"] / "calibration_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if change == "schema": manifest["schema_version"] = "wrong"
    if change == "missing-field": del manifest["schema_version"]
    if change == "extra-field": manifest["private_extra"] = "PRIVATE_MANIFEST"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path, completion_path = record["receipt_path"], record["completion_path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["finalized_outputs"][manifest_path.name] = sha256(manifest_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["verified_outputs"] = receipt["finalized_outputs"]
    completion["child_receipt"]["sha256"] = sha256(receipt_path.read_bytes()).hexdigest()
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    before = {path: sha256(path.read_bytes()).hexdigest() for path in (manifest_path, receipt_path, completion_path)}
    # The shared review harness asserts zero model/row/protected-content reads and child launches.
    result = _harness(tmp_path, _review_body(code, "calibration", ["evidence_verified"], "FAIL"), load=False)
    assert result.returncode == 0, result.stderr
    assert "PRIVATE_MANIFEST" not in result.stdout + result.stderr
    assert {path: sha256(path.read_bytes()).hexdigest() for path in before} == before
    review_path = code / "output" / "phase14" / "calibration-review" / "synthetic-calibration" / "review_record.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert review["disposition"] == "FAIL" and "evidence_invalid" in review["findings"]
    bound = review["evidence"]["synthetic-calibration"]
    assert bound["child_receipt"] == {"status": "present", "path": str(receipt_path), "sha256": before[receipt_path]}
    assert bound["supervisor_completion"] == {"status": "present", "path": str(completion_path), "sha256": before[completion_path]}




@pytest.mark.parametrize("job,change", [
    ("calibration", None), ("final_primary", None), ("final_verification", None),
    *[("calibration", change) for change in ("authorization-schema", "authorization-job", "authorization-run",
        "authorization-approval", "authorization-revision", "authorization-inventory", "authorization-digest",
        "configuration-binding", "protocol-binding", "environment-digest", "environment-version",
        "environment-interpreter", "environment-host", "source-dirty", "source-revision", "source-bytes",
        "nested-package", "nested-module", "nested-staged", "cwd", "disabled-configuration", "existing-output")],
    *[("final_primary", change) for change in ("lock-schema", "lock-environment", "lock-revision", "lock-source",
        "lock-reporting", "lock-prediction", "lock-phase13", "lock-calibration", "lock-digest", "future-primary")],
    ("final_verification", "primary-inventory"), ("final_verification", "primary-same-run")])
def test_actual_job_context_branches_before_any_computation(tmp_path, job, change):
    code, auth, lock, lock_hash = _context_tree(tmp_path, job)
    expected_revision = auth["expected_revision"]
    original_cwd = code if change != "cwd" else code.parent
    if change == "authorization-schema": auth["schema_version"] = "wrong"
    if change == "authorization-job": auth["job"] = "wrong"
    if change == "authorization-run": auth["run_id"] = "other-run"
    if change == "authorization-approval": auth["approval_reference"] = "other-approval"
    if change == "authorization-revision": auth["expected_revision"] = "b" * 40
    if change == "authorization-inventory": auth["unexpected"] = 1
    if change == "configuration-binding": auth["configuration_sha256"] = "0" * 64
    if change == "protocol-binding": auth["protocol_sha256"] = "0" * 64
    if change == "environment-digest": auth["environment_sha256"] = "0" * 64
    if change == "environment-version": auth["environment"]["versions"]["numpy"] = ""
    if change == "environment-interpreter": auth["environment"]["interpreter"]["path"] = "wrong"
    if change == "environment-host": auth["environment"]["powershell"]["edition"] = "Core"
    if change == "source-bytes": (code / "modules" / "feature_pipeline.py").write_text("# changed source", encoding="utf-8")
    if change in {"nested-package", "nested-module", "nested-staged"}:
        injected = code / ("nested" if change == "nested-package" else "modules/nested") / "injected.py"
        injected.parent.mkdir(parents=True)
        injected.write_text("# unreviewed synthetic runtime source", encoding="utf-8")
    if change == "disabled-configuration":
        member = "Thesis_RF/Code/config/phase14_setc_calibration_v1.yaml"
        path = code.parents[1] / member
        path.write_text("execution_enabled: false\n", encoding="utf-8")
        auth["configuration_sha256"] = auth["source_files"][member] = sha256(path.read_bytes()).hexdigest()
    if change == "existing-output":
        (code / "output" / "phase14" / "phase14-setc-rf-61e8f740-r1" / "calibration" / "synthetic-context").mkdir(parents=True)
    if change and change.startswith("lock-") and change != "lock-digest":
        mutations = {"lock-schema": ("schema_version", "wrong"), "lock-environment": ("environment", {}),
            "lock-revision": ("evaluation_revision", "b" * 40), "lock-source": ("evaluator_source_sha256", "0" * 64),
            "lock-reporting": ("reporting", {}), "lock-prediction": ("prediction_contract", {}),
            "lock-phase13": ("phase13", {}), "lock-calibration": ("calibration", {})}
        key, value = mutations[change]
        lock[key] = value
        lock_path = code / "output" / "phase14" / "model-lock" / "synthetic-lock.json"
        lock_path.write_text(json.dumps(lock), encoding="utf-8")
        auth["pretest_lock"]["sha256"] = sha256(lock_path.read_bytes()).hexdigest()
    if change == "lock-digest": auth["pretest_lock"]["sha256"] = "0" * 64
    if change == "future-primary": auth["primary_evidence"] = {}
    if change == "primary-inventory": auth["primary_evidence"]["unexpected"] = 1
    if change == "primary-same-run": auth["primary_evidence"]["run_id"] = auth["run_id"]
    for name in ("output", "data", "models", "protected", "manuscripts"):
        excluded = code / name
        excluded.mkdir(exist_ok=True)
        (excluded / "never_inspect.py").write_text("# excluded synthetic canary", encoding="utf-8")
    raw = json.dumps(auth)
    digest = "0" * 64 if change == "authorization-digest" else sha256(raw.encode()).hexdigest()
    body = (". " + _q(code / RUNNER.name) + "\n"
        "Set-Location -LiteralPath " + _q(original_cwd) + "\n"
        "$script:forbidden=@{rowResolution=0;rowHash=0;model=0;inference=0;output=0;launch=0;excluded=0}\n"
        "$script:inventoryCalls=0\n"
        "$script:sourceOriginal=(Get-Command Get-Phase14ImportSources).ScriptBlock\n"
        "function Get-Phase14ImportSources { param($Repository); $script:inventoryCalls++; & $script:sourceOriginal $Repository }\n"
        "function Invoke-Phase14Git { param($Repository,$Arguments); if ($Arguments[0] -eq 'rev-parse') { return "
        + _q("b" * 40 if change == "source-revision" else expected_revision)
        + " }; if ($Arguments[0] -eq 'status') { " + ("return ' M synthetic.py'" if change == "source-dirty" else
            "if ($Arguments -contains 'Thesis_RF/Code/modules/nested/injected.py') { return 'A  Thesis_RF/Code/modules/nested/injected.py' }; return" if change == "nested-staged" else "return")
        + " }; " + ("return 'Thesis_RF/Code/modules/nested/injected.py'" if change == "nested-staged" else "return") + " }\n"
        "function Get-ChildItem { param($LiteralPath,[switch]$Force,$ErrorAction); if ($LiteralPath -match '[\\\\/](output|data|models|protected|manuscripts)([\\\\/]|$)') { $script:forbidden.excluded++; throw 'excluded traversal' }; Microsoft.PowerShell.Management\\Get-ChildItem -LiteralPath $LiteralPath -Force -ErrorAction Stop }\n"
        "$script:pathOriginal=(Get-Command Assert-Phase14Path).ScriptBlock\n"
        "function Assert-Phase14Path { param($Path,$Root); if ([IO.Path]::GetExtension($Path) -eq '.csv') { $script:forbidden.rowResolution++; throw 'row resolution' }; if ([IO.Path]::GetExtension($Path) -eq '.joblib') { $script:forbidden.model++; throw 'model resolution' }; & $script:pathOriginal $Path $Root }\n"
        "$script:hashOriginal=(Get-Command Get-PhaseFileSha256).ScriptBlock\n"
        "function Get-PhaseFileSha256 { param($Path); if ([IO.Path]::GetExtension($Path) -eq '.csv') { $script:forbidden.rowHash++; throw 'row hash' }; & $script:hashOriginal $Path }\n"
        "function New-Item { $script:forbidden.output++; throw 'output creation' }\n"
        "function Start-Process { $script:forbidden.launch++; throw 'child launch' }\n"
        "function Invoke-PhaseSupervisionCore { $script:forbidden.launch++; throw 'child core' }\n"
        "function predict_positive_probability { $script:forbidden.inference++; throw 'inference' }\n"
        "$env:PHASE14_AUTHORIZATION_JSON=" + _q(raw) + "\n"
        "$env:PHASE14_AUTHORIZATION_SHA256=" + _q(digest) + "\n"
        "$denied=$false; try { $context=Get-Phase14JobContext " + _q(job) + " synthetic-approval " + _q(expected_revision)
        + " synthetic-context } catch { $denied=$true }\n"
        "if ($denied -ne " + ("$false" if change is None else "$true") + ") { throw 'context denial expectation failed' }\n"
        "if (@($script:forbidden.Values | Where-Object { $_ -ne 0 }).Count) { throw 'forbidden context call reached' }\n")
    if change in {"nested-package", "nested-module", "nested-staged"}:
        body += "if ($script:inventoryCalls -ne 1) { throw 'nested inventory not reached' }\n"
    if change is None:
        body += "if (@($context.source_check.source_files.PSObject.Properties.Name).Count -ne 19 -or $context.arguments[0] -cne '-B') { throw 'valid context incomplete' }\n"
    result = _harness(tmp_path, body, load=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("boundary", ["crossing-publication", "already-expired", "publication-error"])
def test_completion_decision_precedes_exclusive_publication_and_is_sampled_once(tmp_path, boundary):
    context = _synthetic_context(tmp_path, "calibration")
    body = _injected_context(context) + (
        "$script:clockCalls=0;  $script:origin=[DateTime]::UtcNow\n"
        "function Get-Phase14UtcNow { $script:clockCalls++; if ($script:clockCalls -lt 4) { return $script:origin.AddSeconds($script:clockCalls-1) }; if ($script:clockCalls -lt 6) { return [DateTime]::UtcNow }; if ($script:clockCalls -eq 6) { return $script:origin.AddSeconds("
        + ("1800" if boundary == "already-expired" else "1799")
        + ") }; return $script:origin.AddSeconds(1801) }\n"
        "$script:publishOriginal=(Get-Command Write-Phase14JsonNew).ScriptBlock; $script:publicationCalls=0\n"
        "function Write-Phase14JsonNew { param($Path,$Record); if ([IO.Path]::GetFileName($Path) -eq 'supervisor_completion.json') { $script:publicationCalls++; $script:publicationTime=$script:origin.AddSeconds(1801); "
        + ("throw 'PRIVATE_PUBLICATION'" if boundary == "publication-error" else
           "if ($Record.ended_at_utc -cne $script:origin.AddSeconds(" + ("1800" if boundary == "already-expired" else "1799") + ").ToString('o')) { throw 'decision timestamp mismatch' }")
        + " }; & $script:publishOriginal $Path $Record }\n"
        "$denied=$false; try { Invoke-Phase14Supervision calibration synthetic-approval " + _q("a" * 40)
        + " synthetic-adapter | Out-Null } catch { $denied=$true; if ($_.Exception.Message -like '*PRIVATE_PUBLICATION*') { throw 'publication diagnostic leaked' } }\n"
        "if ($denied -ne " + ("$false" if boundary == "crossing-publication" else "$true") + ") { throw 'publication boundary expectation failed' }\n"
        "if ($script:clockCalls -ne 6 -or $script:publicationCalls -lt 1) { throw 'completion decision resampled or publication not reached' }\n"
        "if ($script:publicationTime -le $script:origin.AddSeconds(1800)) { throw 'publication crossing not exercised' }\n")
    result = _harness(tmp_path, body)
    assert result.returncode == 0, result.stderr
    assert (Path(context["destination"]) / "execution_receipt.json").is_file()
    completion_path = Path(context["evidence"]) / "supervisor_completion.json"
    if boundary != "publication-error":
        completion = json.loads(completion_path.read_text())
        assert completion["overall_status"] == ("PASS" if boundary == "crossing-publication" else "NEEDS EVIDENCE")
    assert "PRIVATE_PUBLICATION" not in result.stdout + result.stderr



@pytest.mark.parametrize("case", ["equal", "within-tolerance", "outside-tolerance", "integer-left", "undefined", "bin-statistic", "counts"])
def test_reviewer_comparison_has_the_approved_python_policy(tmp_path, case):
    code, authorization, lock, lock_hash = _context_tree(tmp_path, "final_primary")
    primary = _metadata_report("primary", lock, lock_hash)
    verifier = _metadata_report("verification", lock, lock_hash)
    if case == "within-tolerance": verifier["metrics"]["average_precision"]["value"] += 1e-12
    if case == "outside-tolerance": verifier["metrics"]["average_precision"]["value"] += 1e-8
    if case == "integer-left":
        primary["metrics"]["average_precision"]["value"] = 0
        verifier["metrics"]["average_precision"]["value"] = 1e-13
    if case == "undefined":
        for report in (primary, verifier):
            report["metrics"]["roc_auc"] = {"value": None, "status": "undefined", "reason": "non-finite metric"}
    if case == "bin-statistic": verifier["reliability_bins"][1]["mean_predicted_probability"] += 1e-8
    if case == "counts":
        verifier.update(positive_count=3, negative_count=1, prevalence=0.75)
        verifier["reliability_bins"][1]["observed_positive_fraction"] = 0.75
    from evaluate_rf_final_test import _reports_match
    expected = _reports_match(primary, verifier)
    body = (". " + _q(code / RUNNER.name) + "\n"
        "$primary=" + _json_expression(primary) + "\n$verifier=" + _json_expression(verifier) + "\n$lock=" + _json_expression(lock) + "\n"
        "Assert-Phase14ProbabilityReport $primary primary $lock " + _q(lock_hash) + "\n"
        "Assert-Phase14ProbabilityReport $verifier verification $lock " + _q(lock_hash) + "\n"
        "if ((Test-Phase14ReportsMatch $primary $verifier) -ne " + ("$true" if expected else "$false") + ") { throw 'comparison policy drift' }\n")
    result = _harness(tmp_path, body, load=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("left_threshold,right_threshold,expected", [
    (None, None, True), (None, 0, False), (0, None, False),
    ("null", None, False), (None, "null", False),
])
def test_reviewer_comparison_handles_null_threshold_without_canonicalizing_null(tmp_path, left_threshold, right_threshold, expected):
    code, _, lock, lock_hash = _context_tree(tmp_path, "final_primary")
    primary = _metadata_report("primary", lock, lock_hash)
    verifier = _metadata_report("verification", lock, lock_hash)
    primary["classification_threshold"] = left_threshold
    verifier["classification_threshold"] = right_threshold
    body = (". " + _q(code / RUNNER.name) + "\n"
        "$primary=" + _json_expression(primary) + "\n$verifier=" + _json_expression(verifier) + "\n"
        "if ((Test-Phase14ReportsMatch $primary $verifier) -ne " + ("$true" if expected else "$false")
        + ") { throw 'null-threshold comparison contract failed' }\n")
    result = _harness(tmp_path, body, load=False)
    assert result.returncode == 0, result.stderr

@pytest.mark.parametrize("job", ["calibration", "final_primary", "final_verification"])
@pytest.mark.parametrize("topology", ["direct", "launcher-worker"])
def test_record_callback_binds_specific_worker_and_preserves_fast_worker_inventory(tmp_path, job, topology):
    from test_phase14_calibration_contract import _worker_metadata
    code, _, _, records = _finalized_metadata_fixture(tmp_path, "calibration" if job == "calibration" else "verification")
    run = {"calibration": "synthetic-calibration", "final_primary": "synthetic-primary",
           "final_verification": "synthetic-verification"}[job]
    evidence = records[run]["evidence"]
    handoff = json.loads(records[run]["handoff_path"].read_text())
    process, exited = _worker_metadata(evidence, handoff, worker_pid=124 if topology == "direct" else 125,
                                      worker_ppid=123 if topology == "direct" else 124)
    original_hash = sha256((evidence / "worker_binding.json").read_bytes()).hexdigest()
    record = {key: process[key] for key in ("run_id", "process_id", "process_ids", "process_started_at_utc", "executable", "working_directory")}
    record["process_ids"] = [124, 126]  # worker125 need never appear in a late core scan
    body = "$h=" + _json_expression(handoff) + "\n$c=" + _json_expression({"evidence": str(evidence)}) + "\n"
    body += "$r=" + _json_expression(record) + "\n"
    body += "$r=[ordered]@{run_id=$r.run_id;process_id=$r.process_id;process_ids=$r.process_ids;process_started_at_utc=$r.process_started_at_utc;executable=$r.executable;working_directory=$r.working_directory}\n"
    body += "Write-Phase14ProcessRecord " + _q(evidence / "process.json") + " $r $c $h\n"
    body += "$b=Assert-Phase14WorkerBinding (Read-Phase14Record " + _q(evidence / "process.json") + ") $h " + _q(evidence) + "\n"
    body += "function Get-Phase14ProcessInstance { param($ProcessId,[switch]$AllowMissing); if (-not $AllowMissing) { throw 'missing instance not permitted' }; return $null }\n"
    body += "$exit=Get-Phase14WorkerExit $b " + _q(evidence) + "\nif ($exit.same_instance_running -or $exit.pid_reused) { throw 'fast worker promoted' }\n"
    result = _harness(tmp_path, body)
    assert result.returncode == 0, result.stderr
    assert sha256((evidence / "worker_binding.json").read_bytes()).hexdigest() == original_hash
    updated = json.loads((evidence / "process.json").read_text())
    assert updated["worker_binding"] == process["worker_binding"] and updated["process_ids"] == [124, 126]



def _no_process_backup_check(evidence):
    return ("if (@(Get-ChildItem -LiteralPath " + _q(evidence)
            + " | Where-Object { $_.Name.StartsWith('process.json.',[StringComparison]::OrdinalIgnoreCase) }).Count) { throw 'process backup artifact created' }\n")


@pytest.mark.parametrize("artifact_name", [None, "process.json.bak", "PROCESS.JSON.BAK"])
def test_process_backup_check_ignores_live_record_and_rejects_real_backups(tmp_path, artifact_name):
    (tmp_path / "process.json").write_bytes(b"synthetic process record")
    if artifact_name is not None:
        (tmp_path / artifact_name).write_bytes(b"synthetic backup artifact")
    body = "$denied=$false; try { " + _no_process_backup_check(tmp_path) + " } catch { if ($_.Exception.Message -cne 'process backup artifact created') { throw }; $denied=$true }\n"
    body += "if ($denied -ne " + ("$false" if artifact_name is None else "$true") + ") { throw 'backup detection expectation failed' }\n"
    result = _harness(tmp_path, body, load=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("job", ["calibration", "final_primary", "final_verification"])
@pytest.mark.parametrize("change", [None, "pid", "parent", "start", "image", "nonce", "missing-claim",
                                    "missing-instance", "observer-error", "stale", "reused"])
def test_startup_callback_denies_forged_or_unavailable_claim_before_publication(tmp_path, job, change):
    from test_phase14_calibration_contract import _worker_metadata
    from datetime import datetime, timedelta, timezone
    code, _, _, records = _finalized_metadata_fixture(tmp_path, "calibration" if job == "calibration" else "verification")
    run = {"calibration": "synthetic-calibration", "final_primary": "synthetic-primary",
           "final_verification": "synthetic-verification"}[job]
    evidence = records[run]["evidence"]
    handoff = json.loads(records[run]["handoff_path"].read_text())
    now = datetime.now(timezone.utc)
    handoff["created_at_utc"] = (now - timedelta(seconds=3)).isoformat()
    handoff["deadline_utc"] = (now + timedelta(seconds=30)).isoformat()
    records[run]["handoff_path"].write_text(json.dumps(handoff))
    process, _ = _worker_metadata(evidence, handoff)
    claim_path = evidence / "worker_claim.json"
    claim = json.loads(claim_path.read_text())
    snapshots = process["worker_binding"]
    if change == "pid": claim["worker_pid"] = 7193
    if change == "parent": claim["worker_ppid"] = 7193
    if change == "start": claim["worker_created_at_utc"] = "2000-01-01T00:00:00+00:00"
    if change == "image": claim["physical_executable"] = str(tmp_path / "forged.exe")
    if change == "nonce": claim["nonce"] = "b" * 32
    if change == "stale": snapshots["worker"]["created_at_utc"] = "2000-01-01T00:00:00+00:00"
    if change == "reused": snapshots["worker"]["created_at_utc"] = now.isoformat()
    claim_path.write_text(json.dumps(claim))
    if change == "missing-claim": claim_path.unlink()
    record = {key: process[key] for key in ("run_id", "process_id", "process_ids", "process_started_at_utc", "executable", "working_directory")}
    (evidence / "process.json").unlink()
    (evidence / "worker_binding.json").unlink()
    body = "$h=" + _json_expression(handoff) + "\n$c=" + _json_expression({"evidence": str(evidence)}) + "\n$r=" + _json_expression(record) + "\n"
    body += "$r=[ordered]@{run_id=$r.run_id;process_id=$r.process_id;process_ids=$r.process_ids;process_started_at_utc=$r.process_started_at_utc;executable=$r.executable;working_directory=$r.working_directory}\n"
    body += "$script:snapshots=" + _json_expression(snapshots) + "\n$script:forbidden=@{output=0;model=0;rows=0;inference=0}\n"
    body += "$script:writer=(Get-Command Write-Phase14JsonNew).ScriptBlock\n"
    body += "function Write-Phase14JsonNew { param($Path,$Value); if ([IO.Path]::GetFileName($Path) -notmatch '^(worker_binding|\\.phase14-process-.*)\\.json$') { $script:forbidden.output++; throw 'forbidden computation output' }; & $script:writer $Path $Value }\n"
    body += "function Get-Phase14ProcessInstance { param($ProcessId,[switch]$AllowMissing); "
    if change in {"observer-error", "missing-instance"}:
        body += "throw 'PRIVATE_OS_OBSERVER' }\n"
    else:
        body += "foreach ($name in @('supervisor','launcher','worker')) { if ($script:snapshots.$name.pid -eq $ProcessId) { if ($name -eq 'supervisor') { $script:snapshots.$name.executable=([Diagnostics.Process]::GetCurrentProcess().MainModule.FileName) }; return $script:snapshots.$name } }; throw 'specific instance unavailable' }\n"
    body += "function New-Object { $script:forbidden.model++; throw 'forbidden construction' }\n"
    body += "$denied=$false; try { Write-Phase14ProcessRecord " + _q(evidence / "process.json") + " $r $c $h } catch { $denied=$true }\n"
    body += "if ($denied -ne " + ("$false" if change is None else "$true") + ") { throw 'startup expectation mismatch' }\n"
    if change is None:
        body += (
            "$bindingHash=Get-PhaseFileSha256 " + _q(evidence / "worker_binding.json") + "\n"
            "$r.process_ids=@($r.process_ids)+126\n"
            "Write-Phase14ProcessRecord " + _q(evidence / "process.json") + " $r $c $h\n"
            "$written=Read-Phase14Record " + _q(evidence / "process.json") + "\n"
            "if ($written.process_ids.Count -ne 2 -or $written.process_ids[1] -ne 126) { throw 'process replacement contents invalid' }\n"
            "if ((Get-PhaseFileSha256 " + _q(evidence / "worker_binding.json") + ") -cne $bindingHash) { throw 'worker binding changed' }\n"
            "if (@(Get-ChildItem -LiteralPath " + _q(evidence) + " -Filter '.phase14-process-*.json').Count) { throw 'process staging file remained' }\n"
            + _no_process_backup_check(evidence))
    body += "if (@($script:forbidden.Values | Where-Object { $_ -ne 0 }).Count) { throw 'forbidden startup access' }\n"
    result = _harness(tmp_path, body)
    assert result.returncode == 0, result.stderr
    assert (evidence / "process.json").exists() is (change is None)

@pytest.mark.parametrize("change", ["missing-claim", "mixed-process", "mixed-receipt", "mixed-completion",
                                    "worker-running", "worker-reused", "worker-unavailable", "binding-conflict"])
def test_review_rejects_worker_evidence_with_zero_content_access(tmp_path, change):
    code, _, _, records = _finalized_metadata_fixture(tmp_path, "verification")
    primary = records["synthetic-primary"]
    process_path = primary["evidence"] / "process.json"
    receipt = json.loads(primary["receipt_path"].read_text())
    completion = json.loads(primary["completion_path"].read_text())
    process = json.loads(process_path.read_text())
    if change == "missing-claim": (primary["evidence"] / "worker_claim.json").unlink()
    if change == "mixed-process": process["schema_version"] = "phase14a_process.v1"
    if change == "mixed-receipt": receipt["schema_version"] = "phase14a_execution_receipt.v1"
    if change == "mixed-completion": completion["schema_version"] = "phase14a_supervisor_completion.v1"
    if change == "worker-running": completion["worker_exit"]["same_instance_running"] = True
    if change == "worker-reused": completion["worker_exit"]["pid_reused"] = True
    if change == "worker-unavailable": completion["worker_exit"] = None
    if change == "binding-conflict": process["worker_binding"]["worker"]["pid"] = 7193
    process_path.write_text(json.dumps(process))
    primary["receipt_path"].write_text(json.dumps(receipt))
    completion["child_receipt"]["sha256"] = sha256(primary["receipt_path"].read_bytes()).hexdigest()
    primary["completion_path"].write_text(json.dumps(completion))
    result = _harness(tmp_path, _review_body(code, "verification", ["evidence_verified"], "FAIL"), load=False)
    assert result.returncode == 0, result.stderr

@pytest.mark.parametrize("state", ["alive", "reused", "unavailable"])
def test_completion_observer_cannot_promote_worker_state(tmp_path, state):
    code, _, _, records = _finalized_metadata_fixture(tmp_path, "calibration")
    evidence = records["synthetic-calibration"]["evidence"]
    process = json.loads((evidence / "process.json").read_text())
    body = "$b=" + _json_expression(process["worker_binding"]) + "\n"
    body += "function Get-Phase14ProcessInstance { param($ProcessId,[switch]$AllowMissing); "
    body += ("throw 'PRIVATE_COMPLETION_OBSERVER'" if state == "unavailable" else
             "return @{pid=$ProcessId;created_at_utc=" + _q("2000-01-01T00:00:00+00:00" if state == "reused" else process["worker_binding"]["worker"]["created_at_utc"]) + "}")
    body += " }\n$denied=$false; try { Get-Phase14WorkerExit $b " + _q(evidence) + " | Out-Null } catch { $denied=$true }; if (-not $denied) { throw 'completion uncertainty promoted' }\n"
    result = _harness(tmp_path, body)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("job", ["calibration", "final_primary", "final_verification"])
@pytest.mark.parametrize("case", ["equal", "within-forward", "within-reverse", "adjacent",
                                  "boundary-forward", "boundary-reverse", "stale"])
def test_supervisor_worker_creation_precision_and_chronology(tmp_path, job, case):
    from test_phase14_calibration_contract import _worker_creation_metadata
    code, _, _, records = _finalized_metadata_fixture(tmp_path, "calibration" if job == "calibration" else "verification")
    run = {"calibration": "synthetic-calibration", "final_primary": "synthetic-primary",
           "final_verification": "synthetic-verification"}[job]
    evidence = records[run]["evidence"]
    handoff = json.loads(records[run]["handoff_path"].read_text())
    process, _, accepted = _worker_creation_metadata(evidence, handoff, case)
    body = "$h=" + _json_expression(handoff) + "\n$denied=$false\n"
    body += "try { $binding=Assert-Phase14WorkerBinding (Read-Phase14Record " + _q(evidence / "process.json") + ") $h " + _q(evidence) + " } catch { $denied=$true; if ($_.Exception.Message -notlike '*Worker startup binding mismatch*') { throw } }\n"
    body += "if ($denied -ne " + ("$false" if accepted else "$true") + ") { throw 'creation precision expectation mismatch' }\n"
    result = _harness(tmp_path, body)
    assert result.returncode == 0, result.stderr
    assert json.loads((evidence / "process.json").read_text()) == process


@pytest.mark.parametrize("job", ["calibration", "final_primary", "final_verification"])
@pytest.mark.parametrize("receipt_mode", ["stale", "future", "reversed"])
def test_live_receipt_chronology_rejection_prevents_pass(tmp_path, job, receipt_mode):
    context = _synthetic_context(tmp_path, job, receipt_mode)
    body = _injected_context(context) + (
        "$denied=$false; try { Invoke-Phase14Supervision " + _q(job)
        + " synthetic-approval " + _q("a" * 40) + " synthetic-adapter | Out-Null } catch { $denied=$true }\n"
        "if (-not $denied) { throw 'invalid live chronology promoted' }\n")
    result = _harness(tmp_path, body)
    assert result.returncode == 0, result.stderr
    completion = json.loads((Path(context["evidence"]) / "supervisor_completion.json").read_text())
    assert completion["overall_status"] != "PASS"
    assert completion["child_receipt"]["status"] == "present"
    assert completion["stop_reason"] == "child computation evidence invalid"
    assert completion["verified_outputs"] == {}
