"""Synthetic process-contract tests for the native Phase 13 runner."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


RUNNER = Path(__file__).resolve().parents[1] / "run_phase13_supervised.ps1"
PWSH = shutil.which("pwsh")


pytestmark = pytest.mark.skipif(
    os.name != "nt" or not PWSH,
    reason="Phase 13 runner contract requires Windows PowerShell process APIs",
)


def _ps_quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _run_harness(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    harness = tmp_path / "runner_harness.ps1"
    harness.write_text(
        f"$ErrorActionPreference = 'Stop'\ntrap {{ Write-Error $_; exit 1 }}\n"
        f". {_ps_quote(RUNNER)}\n{body}\n",
        encoding="utf-8",
    )
    return subprocess.run(
        [PWSH, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
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
        "-Executable ($PSHOME + '\\pwsh.exe')",
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


def test_success_requires_valid_receipt_and_retains_separate_captures(tmp_path: Path):
    destination = tmp_path / "package"
    child = _write_child(
        tmp_path,
        "$ErrorActionPreference = 'Stop'\n"
        f"New-Item -ItemType Directory -Path {_ps_quote(destination)} | Out-Null\n"
        "$manifest = @{schema_version='set_c_rf_phase13_run.v1'; artifact_kind='set_c_rf_uncalibrated_candidate_run'; run_id='success'; completion_status='complete'} | ConvertTo-Json\n"
        f"$manifestPath = Join-Path {_ps_quote(destination)} 'manifest.json'\n"
        "$manifest | Set-Content -LiteralPath $manifestPath\n"
        "$hash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()\n"
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
    assert receipt["stdout_sha256"] and receipt["stderr_sha256"]


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
        f"$child = Start-Process -FilePath ($PSHOME + '\\pwsh.exe') "
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
        f"$child = Start-Process -FilePath ($PSHOME + '\\pwsh.exe') "
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
        f"$child = Start-Process -FilePath ($PSHOME + '\\pwsh.exe') "
        f"-ArgumentList ('-NoProfile -File ' + {_ps_quote(grandchild)}) -PassThru -WindowStyle Hidden\n"
        "Write-Output ('grandchild=' + $child.Id)\n"
        f"New-Item -ItemType Directory -Path {_ps_quote(destination)} | Out-Null\n"
        f"$manifestPath = Join-Path {_ps_quote(destination)} 'manifest.json'\n"
        f"@{{schema_version='set_c_rf_phase13_run.v1'; artifact_kind='set_c_rf_uncalibrated_candidate_run'; run_id='stopped-valid'; completion_status='complete'}} | ConvertTo-Json | Set-Content -LiteralPath $manifestPath\n"
        "$hash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()\n"
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
        f"$child = Start-Process -FilePath ($PSHOME + '\\pwsh.exe') "
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
