"""Synthetic process-contract tests for the native Phase 13 runner."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


RUNNER = Path(__file__).resolve().parents[1] / "run_phase13_supervised.ps1"
WINDOWS_POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Phase 13 runner requires Windows process APIs")


def _ps_quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _run_harness(tmp_path: Path, body: str, *, load_runner: bool = True) -> subprocess.CompletedProcess[str]:
    if not WINDOWS_POWERSHELL:
        pytest.fail("Windows runner contract requires powershell.exe 5.1; PowerShell Core is not a substitute")
    harness = tmp_path / "runner_harness.ps1"
    harness.write_text(
        "if ($PSVersionTable.PSVersion.ToString() -ne '5.1.26100.9444' -or $PSVersionTable.PSEdition -ne 'Desktop') { "
        "throw 'This contract requires Windows PowerShell 5.1.26100.9444 Desktop.' }\n"
        f"$ErrorActionPreference = 'Stop'\ntrap {{ Write-Error $_; exit 1 }}\n"
        + (f". {_ps_quote(RUNNER)}\n" if load_runner else "")
        + f"{body}\n",
        encoding="utf-8",
    )
    return subprocess.run(
        [WINDOWS_POWERSHELL, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(harness)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def _write_child(tmp_path: Path, body: str) -> Path:
    child = tmp_path / "dummy_child.ps1"
    child.write_text(body, encoding="utf-8")
    return child


def _invoke_arguments(child: Path, destination: Path, evidence: Path, run_id: str, **extras: str) -> str:
    resource_provider = extras.pop(
        "ResourceProvider",
        "{ return @{available_physical_memory_bytes=8589934592L; free_c_bytes=4294967296L} }",
    )
    parts = [
        "-Executable ($PSHOME + '\\powershell.exe')",
        f"-ChildArguments @('-NoProfile', '-File', {_ps_quote(child)})",
        f"-WorkingDirectory {_ps_quote(child.parent)}",
        f"-Destination {_ps_quote(destination)}",
        f"-EvidenceDirectory {_ps_quote(evidence)}",
        f"-RunId {_ps_quote(run_id)}",
        f"-ResourceProvider {resource_provider}",
    ]
    parts.extend(f"-{key} {value}" for key, value in extras.items())
    return "Invoke-Phase13Supervision " + " `\n    ".join(parts)


def _completion(evidence: Path) -> dict:
    return json.loads((evidence / "operator_completion.json").read_text(encoding="utf-8-sig"))



def test_module_and_dot_source_loading_do_not_launch_children(tmp_path: Path):
    module = RUNNER.parent / "modules" / "phase_supervision.psm1"
    body = (
        "$script:launchCalls = 0\n"
        "function New-Object { $script:launchCalls++; throw 'child construction during load' }\n"
        "function Start-Process { $script:launchCalls++; throw 'child launch during load' }\n"
        f"Import-Module {_ps_quote(module)} -ErrorAction Stop\n"
        "if ((Get-Command Invoke-PhaseSupervisionCore).ModuleName -ne 'phase_supervision') { throw 'core not exported' }\n"
        f". {_ps_quote(RUNNER)}\n"
        "if (-not (Get-Command Invoke-Phase13Supervision -CommandType Function)) { throw 'wrapper missing' }\n"
        "if ($script:launchCalls -ne 0) { throw 'load launched a child' }\n"
    )
    result = _run_harness(tmp_path, body, load_runner=False)
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == ["runner_harness.ps1"]


@pytest.mark.parametrize("argument_count", [0, 1, 2, 4])
def test_direct_entry_refuses_wrong_argument_count_before_child_launch(tmp_path: Path, argument_count: int):
    arguments = ", ".join("'synthetic'" for _ in range(argument_count))
    body = (
        "$script:launchCalls = 0\n"
        "function New-Object { $script:launchCalls++; throw 'child construction before refusal' }\n"
        "function Start-Process { $script:launchCalls++; throw 'child launch before refusal' }\n"
        f"try {{ & {_ps_quote(RUNNER)} @({arguments}); throw 'entry accepted invalid arguments' }} "
        "catch { if ($_.Exception.Message -notlike 'Usage: run_phase13_supervised.ps1*') { throw } }\n"
        "if ($script:launchCalls -ne 0) { throw 'child launch preceded argument refusal' }\n"
    )
    result = _run_harness(tmp_path, body, load_runner=False)
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == ["runner_harness.ps1"]


def test_atomic_process_record_temp_stays_outside_child_visible_evidence(tmp_path: Path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    result = _run_harness(
        tmp_path,
        "$target = Join-Path " + _ps_quote(evidence) + " 'process.json'\n"
        + "Write-Phase13JsonAtomic -Path $target -Value @{schema_version='phase13_set_c_rf_process.v1'} "
        + "-TemporaryDirectory " + _ps_quote(tmp_path) + "\n"
        + "if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw 'record missing' }\n"
        + "if (@(Get-ChildItem -LiteralPath " + _ps_quote(evidence) + " -Force).Count -ne 1) { throw 'temp leaked into evidence' }\n",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((evidence / "process.json").read_text(encoding="utf-8-sig"))["schema_version"] == "phase13_set_c_rf_process.v1"


def test_success_requires_valid_receipt_and_retains_separate_captures(tmp_path: Path):
    destination = tmp_path / "package"
    child = _write_child(
        tmp_path,
        "$ErrorActionPreference = 'Stop'\n"
        f"New-Item -ItemType Directory -Path {_ps_quote(destination)} | Out-Null\n"
        "$manifest = @{schema_version='set_c_rf_phase13_run.v1'; artifact_kind='set_c_rf_uncalibrated_candidate_run'; run_id='success'; completion_status='complete'} | ConvertTo-Json\n"
        f"$manifestPath = Join-Path {_ps_quote(destination)} 'manifest.json'\n"
        "$manifest | Set-Content -LiteralPath $manifestPath\n"
        "$hash = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([IO.File]::ReadAllBytes($manifestPath))).Replace('-', '').ToLowerInvariant()\n"
        "$completion = @{schema_version='phase13_set_c_rf_completion.v1'; status='complete'; actual_exit_status=0; run_id='success'; manifest_sha256=$hash} | ConvertTo-Json\n"
        f"$completion | Set-Content -LiteralPath (Join-Path {_ps_quote(destination)} 'completion.json')\n"
        "Write-Output 'synthetic stdout'\nWrite-Error 'synthetic stderr' -ErrorAction Continue\n",
    )
    evidence = tmp_path / "evidence"
    result = _run_harness(tmp_path, _invoke_arguments(child, destination, evidence, "success"))
    assert result.returncode == 0, result.stderr
    receipt = _completion(evidence)
    assert receipt["actual_exit_status"] == 0
    assert receipt["package_verified"] is True
    assert receipt["process_exited"] is True
    assert (evidence / "process.json").is_file()
    assert (evidence / "stdout.log").read_text(encoding="utf-8-sig").find("synthetic stdout") >= 0
    assert (evidence / "stderr.log").read_text(encoding="utf-8-sig").find("synthetic stderr") >= 0
    assert receipt["capture_complete"] is True
    assert receipt["capture_status"] == "complete"
    assert receipt["stdout_sha256"] == hashlib.sha256((evidence / "stdout.log").read_bytes()).hexdigest()
    assert receipt["stderr_sha256"] == hashlib.sha256((evidence / "stderr.log").read_bytes()).hexdigest()


def test_valid_receipt_and_zero_exit_still_reject_capture_task_failure(tmp_path: Path):
    destination = tmp_path / "package"
    child = _write_child(
        tmp_path,
        f"New-Item -ItemType Directory -Path {_ps_quote(destination)} | Out-Null\n"
        f"$manifestPath = Join-Path {_ps_quote(destination)} 'manifest.json'\n"
        "@{schema_version='set_c_rf_phase13_run.v1'; artifact_kind='set_c_rf_uncalibrated_candidate_run'; run_id='capture-fault'; completion_status='complete'} | ConvertTo-Json | Set-Content $manifestPath\n"
        "$hash = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([IO.File]::ReadAllBytes($manifestPath))).Replace('-', '').ToLowerInvariant()\n"
        f"@{{schema_version='phase13_set_c_rf_completion.v1'; status='complete'; actual_exit_status=0; run_id='capture-fault'; manifest_sha256=$hash}} | ConvertTo-Json | Set-Content (Join-Path {_ps_quote(destination)} 'completion.json')\n"
        "Write-Output 'synthetic stdout before injected capture fault'\nWrite-Error 'synthetic stderr before injected capture fault' -ErrorAction Continue\n",
    )
    evidence = tmp_path / "evidence"
    body = (
        "function Wait-Phase13CaptureTasks { param($Tasks, $TimeoutMilliseconds); "
        f"Set-Content -LiteralPath {_ps_quote(tmp_path / 'capture_wait_called.txt')} -Value 'called' ; "
        "throw [System.AggregateException]::new('synthetic capture fault') }\n"
        + _invoke_arguments(child, destination, evidence, "capture-fault")
        + "\n"
    )
    result = _run_harness(tmp_path, body)
    assert result.returncode != 0
    receipt = _completion(evidence)
    assert receipt["actual_exit_status"] == 0
    assert receipt["package_verified"] is True
    assert receipt["capture_complete"] is False
    assert receipt["capture_status"] == "failed"
    assert "synthetic capture fault" in receipt["capture_error"]
    assert (tmp_path / "capture_wait_called.txt").read_text(encoding="utf-8-sig").strip() == "called"
    assert receipt["stdout_sha256"] is None
    assert receipt["stderr_sha256"] is None


def test_noncompleting_capture_drain_stays_within_overall_budget(tmp_path: Path):
    child = _write_child(tmp_path, "Start-Sleep -Milliseconds 100\n")
    evidence = tmp_path / "evidence"
    invocation = _invoke_arguments(
        child, tmp_path / "package", evidence, "capture-timeout",
        OverallLimitSeconds="2", SampleIntervalSeconds="1",
    )
    body = (
        "$script:captureWaitCalls = 0\n"
        "function Wait-Phase13CaptureTasks { param($Tasks, $TimeoutMilliseconds); "
        "$script:captureWaitCalls++; "
        "$source = [System.Threading.Tasks.TaskCompletionSource[object]]::new(); "
        "$script:drainProbeTask = $source.Task; "
        "$all = [System.Collections.Generic.List[System.Threading.Tasks.Task]]::new(); "
        "foreach ($task in $Tasks) { $all.Add($task) }; $all.Add($script:drainProbeTask); "
        "return [System.Threading.Tasks.Task]::WaitAll($all.ToArray(), $TimeoutMilliseconds) }\n"
        + "try { " + invocation + " } catch { if ($_.Exception.Message -notlike 'Phase 13 run incomplete:*') { throw } }; "
        + "if ($script:drainProbeTask.IsCompleted) { throw 'synthetic drain task unexpectedly completed' }; "
        + "if ($script:captureWaitCalls -ne 1) { throw 'capture injection was not reached exactly once' }; "
        + "Write-Output 'SYNTHETIC_DRAIN_TASK_PENDING'\n"
    )
    started = time.monotonic()
    result = _run_harness(tmp_path, body)
    elapsed = time.monotonic() - started
    assert result.returncode == 0, result.stderr
    assert "SYNTHETIC_DRAIN_TASK_PENDING" in result.stdout
    receipt = _completion(evidence)
    assert 0 < receipt["capture_drain_timeout_ms"] <= 2000
    assert 1.0 <= elapsed < 4.0
    assert receipt["actual_exit_status"] == 0
    assert receipt["capture_complete"] is False
    assert receipt["capture_status"] == "drain_timed_out"
    assert "completion receipt/manifest hash" in receipt["stop_reason"]
    assert receipt["process_exited"] is True
    assert receipt["remaining_process_ids"] == []
    assert receipt["stdout_sha256"] is None
    assert receipt["stderr_sha256"] is None


def test_supervisor_evidence_passes_real_python_execution_guard(tmp_path: Path):
    code_dir = tmp_path / "synthetic_code"
    script_path = code_dir / "train_rf_optuna.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("# synthetic path anchor\n", encoding="utf-8")
    output_root = code_dir / "output" / "phase13"
    evidence_parent = output_root / "evidence"
    evidence_parent.mkdir(parents=True)
    destination = output_root / "set_c_rf_v1"
    run_id = "execution-guard"
    evidence = evidence_parent / run_id

    child_source = f"""
import base64, json, os, sys, time
from pathlib import Path
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
import train_rf_optuna as phase13
phase13.__file__ = {str(script_path)!r}
evidence = Path({str(evidence)!r})
destination = Path({str(destination)!r})
deadline = time.monotonic() + 8
while time.monotonic() < deadline:
    if ((evidence / 'supervisor_handoff.json').is_file()
            and (evidence / 'stdout.log').is_file()
            and (evidence / 'stderr.log').is_file()):
        break
    time.sleep(0.02)
else:
    raise RuntimeError('supervisor evidence did not become ready')
assert (evidence / 'stdout.log').stat().st_size == 0
assert (evidence / 'stderr.log').stat().st_size == 0
config = {{
    'destination': str(destination),
    'evidence_parent': str(evidence.parent),
    'resource_limits': {{
        'min_available_physical_memory_bytes': 4294967296,
        'min_free_c_bytes': 2147483648,
    }},
}}
phase13._load_phase13_config = lambda _path: config
phase13._git_state = lambda _revision: {{'revision': 'a' * 40, 'dirty': False}}
phase13._environment_record = lambda _config: {{'versions': {{'python': sys.version}}}}
phase13._resource_snapshot = lambda: {{
    'timestamp_utc': 'synthetic', 'available_physical_memory_bytes': 8 * 1024**3,
    'free_c_bytes': 4 * 1024**3,
}}
phase13._validate_phase13_inputs = lambda _config: ({{}}, {{
    'attestation': {{}}, 'provenance': {{}}, 'observed_role_counts': {{}},
}})
preflight = phase13._preflight(
    Path('synthetic-config.yaml'), 'a' * 40, phase13.PHASE13_CONFIG_SHA256,
    execution=True, evidence_dir=evidence,
)
assert preflight['destination'] == destination.resolve()
assert preflight['evidence_dir'] == evidence.resolve()
destination.mkdir()
manifest_path = destination / 'manifest.json'
manifest_path.write_text(json.dumps({{
    'schema_version': 'set_c_rf_phase13_run.v1',
    'artifact_kind': 'set_c_rf_uncalibrated_candidate_run',
    'run_id': {run_id!r}, 'completion_status': 'complete',
}}), encoding='utf-8')
manifest_hash = __import__('hashlib').sha256(manifest_path.read_bytes()).hexdigest()
(destination / 'completion.json').write_text(json.dumps({{
    'schema_version': 'phase13_set_c_rf_completion.v1', 'status': 'complete',
    'actual_exit_status': 0, 'run_id': {run_id!r}, 'manifest_sha256': manifest_hash,
}}), encoding='utf-8')
print('REAL_EXECUTION_GUARD_ACCEPTED', flush=True)
"""
    encoded_source = base64.b64encode(child_source.encode("utf-8")).decode("ascii")
    child_command = f"import base64;exec(base64.b64decode('{encoded_source}'))"
    body = (
        "Invoke-Phase13Supervision "
        f"-Executable {_ps_quote(sys.executable)} "
        f"-ChildArguments @('-B', '-c', {_ps_quote(child_command)}) "
        f"-WorkingDirectory {_ps_quote(code_dir)} "
        f"-Destination {_ps_quote(destination)} "
        f"-EvidenceDirectory {_ps_quote(evidence)} "
        f"-RunId {_ps_quote(run_id)} "
        "-ResourceProvider { return @{available_physical_memory_bytes=8589934592L; free_c_bytes=4294967296L} }"
    )

    result = _run_harness(tmp_path, body)
    assert result.returncode == 0, result.stderr
    receipt = _completion(evidence)
    assert receipt["actual_exit_status"] == 0
    assert receipt["package_verified"] is True
    assert "REAL_EXECUTION_GUARD_ACCEPTED" in (evidence / "stdout.log").read_text(encoding="utf-8-sig")
    assert (evidence / "supervisor_handoff.json").is_file()
    assert (evidence / "process.json").is_file()
    assert (evidence / "operator_resources.jsonl").is_file()


@pytest.mark.parametrize("receipt_mode", ["missing", "invalid_hash"])
def test_zero_exit_without_valid_receipt_is_failure(tmp_path: Path, receipt_mode: str):
    destination = tmp_path / "package"
    child = _write_child(
        tmp_path,
        f"New-Item -ItemType Directory -Path {_ps_quote(destination)} | Out-Null\n"
        f"$manifestPath = Join-Path {_ps_quote(destination)} 'manifest.json'\n"
        "@{schema_version='set_c_rf_phase13_run.v1'; artifact_kind='set_c_rf_uncalibrated_candidate_run'; run_id='receipt-check'; completion_status='complete'} | ConvertTo-Json | Set-Content $manifestPath\n"
        + (
            "'{}' | Set-Content (Join-Path " + _ps_quote(destination) + " 'completion.json')\n"
            if receipt_mode == "invalid_hash"
            else ""
        ),
    )
    evidence = tmp_path / "evidence"
    result = _run_harness(tmp_path, _invoke_arguments(child, destination, evidence, "receipt-check"))
    assert result.returncode != 0
    receipt = _completion(evidence)
    assert receipt["actual_exit_status"] == 0
    assert receipt["package_verified"] is False
    assert "completion receipt/manifest hash" in receipt["stop_reason"]


def test_nonzero_exit_is_recorded_without_inventing_success(tmp_path: Path):
    child = _write_child(tmp_path, "Write-Output 'before failure'\nexit 7\n")
    evidence = tmp_path / "evidence"
    result = _run_harness(
        tmp_path,
        _invoke_arguments(child, tmp_path / "package", evidence, "nonzero"),
    )
    assert result.returncode != 0
    receipt = _completion(evidence)
    assert receipt["actual_exit_status"] == 7
    assert receipt["package_verified"] is False
    assert "before failure" in (evidence / "stdout.log").read_text(encoding="utf-8-sig")


@pytest.mark.parametrize("exit_code", [0, 7])
def test_python_child_exit_status_is_observed_exactly(tmp_path: Path, exit_code: int):
    evidence = tmp_path / "evidence"
    child_code = (
        "import sys; "
        + ("print('synthetic child output', flush=True); sys.exit(0)" if exit_code == 0 else
           "import traceback; exec(\"try:\\n  raise RuntimeError('synthetic child failure')\\nexcept RuntimeError:\\n  traceback.print_exc()\\n  print('synthetic explicit exit 7', file=sys.stderr)\\n  sys.exit(7)\")")
    )
    body = (
        "Invoke-Phase13Supervision "
        f"-Executable {_ps_quote(sys.executable)} "
        f"-ChildArguments @('-B', '-c', {_ps_quote(child_code)}) "
        f"-WorkingDirectory {_ps_quote(tmp_path)} "
        f"-Destination {_ps_quote(tmp_path / 'package')} "
        f"-EvidenceDirectory {_ps_quote(evidence)} -RunId 'python-exit-{exit_code}' "
        "-ResourceProvider { return @{available_physical_memory_bytes=8589934592L; free_c_bytes=4294967296L} }"
    )
    result = _run_harness(tmp_path, body)
    assert result.returncode != 0  # no package receipt is supplied by this probe
    receipt = _completion(evidence)
    assert receipt["actual_exit_status"] == exit_code
    if exit_code == 7:
        assert receipt["stop_reason"] == "child exited nonzero"
        stderr = (evidence / "stderr.log").read_text(encoding="utf-8-sig")
        assert "Traceback (most recent call last)" in stderr
        assert "synthetic child failure" in stderr
    else:
        assert "completion receipt/manifest hash" in receipt["stop_reason"]


def test_python_unhandled_exception_has_real_traceback_and_exit_one(tmp_path: Path):
    evidence = tmp_path / "evidence"
    child_code = "raise RuntimeError('unhandled synthetic failure')"
    body = (
        "Invoke-Phase13Supervision "
        f"-Executable {_ps_quote(sys.executable)} "
        f"-ChildArguments @('-B', '-c', {_ps_quote(child_code)}) "
        f"-WorkingDirectory {_ps_quote(tmp_path)} "
        f"-Destination {_ps_quote(tmp_path / 'package')} "
        f"-EvidenceDirectory {_ps_quote(evidence)} -RunId 'python-unhandled' "
        "-ResourceProvider { return @{available_physical_memory_bytes=8589934592L; free_c_bytes=4294967296L} }"
    )
    result = _run_harness(tmp_path, body)
    assert result.returncode != 0
    receipt = _completion(evidence)
    assert receipt["actual_exit_status"] == 1
    assert receipt["stop_reason"] == "child exited nonzero"
    stderr = (evidence / "stderr.log").read_text(encoding="utf-8-sig")
    assert "Traceback (most recent call last)" in stderr
    assert "unhandled synthetic failure" in stderr


def test_unobserved_exit_status_remains_null_on_launch_error(tmp_path: Path):
    evidence = tmp_path / "evidence"
    body = (
        "Invoke-Phase13Supervision "
        f"-Executable {_ps_quote(tmp_path / 'missing-child.exe')} "
        "-ChildArguments @('-NoProfile') "
        f"-WorkingDirectory {_ps_quote(tmp_path)} "
        f"-Destination {_ps_quote(tmp_path / 'package')} "
        f"-EvidenceDirectory {_ps_quote(evidence)} -RunId 'launch-error'"
    )
    result = _run_harness(tmp_path, body)
    assert result.returncode != 0
    receipt = _completion(evidence)
    assert receipt["actual_exit_status"] is None
    assert receipt["process_id"] is None
    assert receipt["package_verified"] is False


def test_controlled_time_stop_kills_and_verifies_descendants(tmp_path: Path):
    grandchild = _write_child(tmp_path, "Start-Sleep -Seconds 30\n")
    parent = tmp_path / "dummy_parent.ps1"
    parent.write_text(
        f"$child = Start-Process -FilePath ($PSHOME + '\\powershell.exe') "
        f"-ArgumentList ('-NoProfile -File ' + {_ps_quote(grandchild)}) -PassThru -WindowStyle Hidden\n"
        "Write-Output ('grandchild=' + $child.Id)\n"
        "Start-Sleep -Seconds 30\n",
        encoding="utf-8",
    )
    evidence = tmp_path / "evidence"
    result = _run_harness(
        tmp_path,
        _invoke_arguments(
            parent, tmp_path / "package", evidence, "stopped",
            OverallLimitSeconds="2", SampleIntervalSeconds="1",
        ),
    )
    assert result.returncode != 0
    receipt = _completion(evidence)
    assert receipt["stop_reason"] == "overall time limit reached"
    assert receipt["process_exited"] is True
    assert len(receipt["process_ids"]) >= 2
    assert receipt["remaining_process_ids"] == []
    assert (evidence / "stdout.log").is_file()
    assert (evidence / "operator_resources.jsonl").is_file()


def test_resource_breach_fails_and_kills_descendants(tmp_path: Path):
    sentinel = tmp_path / "resource_child_started.txt"
    grandchild = _write_child(tmp_path, "Start-Sleep -Seconds 30\n")
    parent = tmp_path / "dummy_parent_resource.ps1"
    parent.write_text(
        f"$child = Start-Process -FilePath ($PSHOME + '\\powershell.exe') "
        f"-ArgumentList ('-NoProfile -File ' + {_ps_quote(grandchild)}) -PassThru -WindowStyle Hidden\n"
        "Write-Output ('grandchild=' + $child.Id)\n"
        f"Set-Content -LiteralPath {_ps_quote(sentinel)} -Value 'started'\n"
        "Start-Sleep -Seconds 30\n",
        encoding="utf-8",
    )
    resource_provider = (
        "{ if (Test-Path -LiteralPath " + _ps_quote(sentinel) + ") { "
        "return @{available_physical_memory_bytes=1073741824L; free_c_bytes=4294967296L} }; "
        "return @{available_physical_memory_bytes=8589934592L; free_c_bytes=4294967296L} }"
    )
    evidence = tmp_path / "evidence"
    result = _run_harness(
        tmp_path,
        _invoke_arguments(
            parent, tmp_path / "package", evidence, "resource-stop",
            SampleIntervalSeconds="0", ResourceProvider=resource_provider,
        ),
    )
    assert result.returncode != 0
    receipt = _completion(evidence)
    assert receipt["stop_reason"] == "resource floor breached"
    assert receipt["package_verified"] is False
    assert receipt["process_exited"] is True
    assert len(receipt["process_ids"]) >= 2
    assert receipt["remaining_process_ids"] == []


@pytest.mark.parametrize("stop_mode", ["resource", "time"])
def test_supervisor_stop_is_incomplete_with_valid_zero_exit_receipt(tmp_path: Path, stop_mode: str):
    destination = tmp_path / "package"
    evidence = tmp_path / "evidence"
    grandchild = _write_child(tmp_path, "Start-Sleep -Seconds 30\n")
    parent = tmp_path / "dummy_parent_with_receipt.ps1"
    parent.write_text(
        f"$child = Start-Process -FilePath ($PSHOME + '\\powershell.exe') "
        f"-ArgumentList ('-NoProfile -File ' + {_ps_quote(grandchild)}) -PassThru -WindowStyle Hidden\n"
        "Write-Output ('grandchild=' + $child.Id)\n"
        f"New-Item -ItemType Directory -Path {_ps_quote(destination)} | Out-Null\n"
        f"$manifestPath = Join-Path {_ps_quote(destination)} 'manifest.json'\n"
        f"@{{schema_version='set_c_rf_phase13_run.v1'; artifact_kind='set_c_rf_uncalibrated_candidate_run'; run_id='stopped-valid'; completion_status='complete'}} | ConvertTo-Json | Set-Content -LiteralPath $manifestPath\n"
        "$hash = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([IO.File]::ReadAllBytes($manifestPath))).Replace('-', '').ToLowerInvariant()\n"
        f"@{{schema_version='phase13_set_c_rf_completion.v1'; status='complete'; actual_exit_status=0; run_id='stopped-valid'; manifest_sha256=$hash}} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path {_ps_quote(destination)} 'completion.json')\n"
        "Start-Sleep -Milliseconds 500\n",
        encoding="utf-8",
    )
    resource_provider = "{ return @{available_physical_memory_bytes=8589934592L; free_c_bytes=4294967296L} }"
    extras = {"OverallLimitSeconds": "1", "SampleIntervalSeconds": "0"}
    if stop_mode == "resource":
        resource_provider = (
            "{ if ((Test-Path -LiteralPath " + _ps_quote(destination / "completion.json") + ") -and "
            "(Test-Path -LiteralPath " + _ps_quote(evidence / "process.json") + ")) { "
            "$record = Get-Content -LiteralPath " + _ps_quote(evidence / "process.json") + " -Raw | ConvertFrom-Json; "
            "if (-not (Get-Process -Id ([int]$record.process_id) -ErrorAction SilentlyContinue)) { "
            "return @{available_physical_memory_bytes=1073741824L; free_c_bytes=4294967296L} } }; "
            "return @{available_physical_memory_bytes=8589934592L; free_c_bytes=4294967296L} }"
        )
        extras["OverallLimitSeconds"] = "10"
    extras["ResourceProvider"] = resource_provider

    result = _run_harness(
        tmp_path,
        _invoke_arguments(parent, destination, evidence, "stopped-valid", **extras),
    )
    assert result.returncode != 0
    receipt = _completion(evidence)
    assert receipt["stop_reason"] == (
        "resource floor breached" if stop_mode == "resource" else "overall time limit reached"
    )
    assert receipt["actual_exit_status"] == 0
    assert receipt["package_verified"] is True
    assert receipt["package_manifest_sha256"] == hashlib.sha256(
        (destination / "manifest.json").read_bytes()
    ).hexdigest()
    assert receipt["process_exited"] is True
    assert len(receipt["process_ids"]) >= 2
    assert receipt["remaining_process_ids"] == []


def test_controlled_runner_error_kills_descendants_and_preserves_error(tmp_path: Path):
    sentinel = tmp_path / "descendant_started.txt"
    grandchild = _write_child(tmp_path, "Start-Sleep -Seconds 30\n")
    parent = tmp_path / "dummy_parent_error.ps1"
    parent.write_text(
        f"$child = Start-Process -FilePath ($PSHOME + '\\powershell.exe') "
        f"-ArgumentList ('-NoProfile -File ' + {_ps_quote(grandchild)}) -PassThru -WindowStyle Hidden\n"
        "Write-Output ('grandchild=' + $child.Id)\n"
        f"Set-Content -LiteralPath {_ps_quote(sentinel)} -Value 'started'\n"
        "Start-Sleep -Seconds 30\n",
        encoding="utf-8",
    )
    provider = (
        "{ if (Test-Path -LiteralPath " + _ps_quote(sentinel) + ") { throw 'synthetic runner provider failure' }; "
        "return @{available_physical_memory_bytes=8589934592L; free_c_bytes=4294967296L} }"
    )
    evidence = tmp_path / "evidence"
    result = _run_harness(
        tmp_path,
        _invoke_arguments(
            parent, tmp_path / "package", evidence, "runner-error",
            SampleIntervalSeconds="0", ResourceProvider=provider,
        ),
    )
    assert result.returncode != 0
    receipt = _completion(evidence)
    assert "synthetic runner provider failure" in receipt["runner_error"]
    assert receipt["process_exited"] is True
    assert len(receipt["process_ids"]) >= 2
    assert receipt["remaining_process_ids"] == []


def test_interrupted_staging_refusal_does_not_launch_child(tmp_path: Path):
    stage = tmp_path / ".set_c_rf_v1.staging-old-run"
    stage.mkdir()
    sentinel = tmp_path / "child_launched.txt"
    child = _write_child(tmp_path, f"Set-Content -LiteralPath {_ps_quote(sentinel)} -Value 'launched'\n")
    result = _run_harness(
        tmp_path,
        _invoke_arguments(child, tmp_path / "package", tmp_path / "evidence", "staging-refused"),
    )
    assert result.returncode != 0
    assert stage.is_dir()
    assert not sentinel.exists()
    assert not (tmp_path / "evidence").exists()


def test_existing_output_refusal_does_not_overwrite(tmp_path: Path):
    destination = tmp_path / "package"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    child = _write_child(tmp_path, "exit 0\n")
    result = _run_harness(
        tmp_path,
        _invoke_arguments(child, destination, tmp_path / "evidence", "refused"),
    )
    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "evidence").exists()
