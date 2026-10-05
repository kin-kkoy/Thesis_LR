"""Synthetic-only contract tests for the disabled production supervisor."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import run_set_c_split as runner
import run_set_c_supervised as supervisor


def _healthy(_child, disk_path):
    return {
        "available_physical_memory_bytes": supervisor.MEMORY_FLOOR + 1,
        "child_working_set_bytes": 123456,
        "resource_sample_kind": "synthetic_point_sample",
        "free_disk_bytes": supervisor.DISK_FLOOR + 1,
        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
        "disk_scope": "synthetic_temp_volume",
    }


def _run(tmp_path, code, *, timeout=5.0, resource_reader=_healthy, monkeypatch=None):
    script = tmp_path / "synthetic_child.py"
    script.write_text(code, encoding="utf-8")
    log_path = tmp_path / "supervisor.jsonl"
    log = supervisor._JsonlLog(log_path)
    result = supervisor._supervise_child(
        [sys.executable, "-B", str(script)], cwd=tmp_path, log=log,
        overall_seconds=timeout, sample_seconds=0.05, resource_reader=resource_reader,
    )
    log.close()
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    return result, records, log_path


def _diagnostic(**exception_changes):
    exception = {
        "type": "RuntimeError",
        "message": "D-035 allocation did not prove an exact optimum",
        "stage": "allocation",
        "frames": [{"file": "set_c_split_release.py", "function": "solve", "line": 1115}],
        "objective_phase": "positive_deviation",
        "objective_index": 1,
        "objective_ordinal": 1,
        "solver_status": 1,
        "solver_success": False,
        "has_solution": False,
        "elapsed_seconds": 3.1,
        "time_limit": 3,
        "mip_rel_gap": 0,
        "failure_reason": "exact_optimum_not_proven",
    }
    exception.update(exception_changes)
    return json.dumps({
        "schema_version": "set_c_failure_diagnostic.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": "3.14.3",
        "exceptions": [exception],
    }, separators=(",", ":"))


def test_default_denial_precedes_revision_and_log_path_access(monkeypatch, capsys):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("preflight must stop before repository or path inspection")

    monkeypatch.setattr(supervisor, "_verify_clean_revision", unexpected)
    monkeypatch.setattr(supervisor, "_validate_log_path", unexpected)
    assert supervisor._main([]) == 1
    record = json.loads(capsys.readouterr().out)
    assert record["stop_reason"] == "approval_required"
    assert record["stdout_drain"] == record["stderr_drain"] == "not_started"
    assert record["automatic_retry"] is False


@pytest.mark.parametrize("argv,reason", [
    (["--execute-approved-run", "--expected-revision", "a" * 40], "approval_required"),
    (["--execute-approved-run", "--owner-approval-reference", "ok",
      "--expected-revision", "A" * 40, "--diagnostic-log", "ignored"], "revision_invalid"),
    (["--execute-approved-run", "--owner-approval-reference", "ok",
      "--expected-revision", "a" * 40, "--diagnostic-log", "ignored",
      "--arbitrary-child-command", "anything"], "preflight_failed"),
])
def test_request_rejects_missing_or_malformed_approval_before_git(argv, reason, monkeypatch, capsys):
    monkeypatch.setattr(supervisor, "_verify_clean_revision",
                        lambda *_: (_ for _ in ()).throw(AssertionError("must reject first")))
    assert supervisor._main(argv) == 1
    assert json.loads(capsys.readouterr().out)["stop_reason"] == reason


def test_revision_and_log_validation_are_before_child_launch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(supervisor, "_verify_clean_revision",
                        lambda value: calls.append(("revision", value)))
    with pytest.raises(supervisor.SupervisorError, match="log_path_invalid"):
        supervisor._validate_log_path(str(supervisor._REPO / "synthetic-log.jsonl"))
    assert calls == []
    log = tmp_path / "already-there.jsonl"
    log.write_text("preserve", encoding="utf-8")
    with pytest.raises(supervisor.SupervisorError, match="log_exists"):
        supervisor._validate_log_path(str(log))
    assert log.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("head,status,reason", [
    ("a" * 40, "", None), ("b" * 40, "", "revision_mismatch"),
    ("a" * 40, " M source.py", "checkout_dirty"),
])
def test_clean_revision_gate_uses_only_fixed_git_queries(monkeypatch, head, status, reason):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        stdout = head if argv[-3:] == ["rev-parse", "--verify", "HEAD"] else status
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    if reason:
        with pytest.raises(supervisor.SupervisorError, match=reason):
            supervisor._verify_clean_revision("a" * 40)
    else:
        supervisor._verify_clean_revision("a" * 40)
    assert len(calls) == 2
    assert all(call[1]["shell"] is False for call in calls)


def test_main_constructs_only_fixed_runner_argv_without_launching_it(monkeypatch, tmp_path, capsys):
    calls = []

    class MemoryLog:
        def __init__(self, _path):
            self.records = []
        def write(self, record):
            self.records.append(record)
        def close(self):
            pass

    monkeypatch.setattr(supervisor, "_verify_clean_revision", lambda _revision: None)
    monkeypatch.setattr(supervisor, "_validate_log_path", lambda _path: tmp_path / "outside.jsonl")
    monkeypatch.setattr(supervisor, "_JsonlLog", MemoryLog)
    monkeypatch.setattr(supervisor, "_resource_sample", lambda *_: {
        "available_physical_memory_bytes": supervisor.MEMORY_FLOOR + 1,
        "free_disk_bytes": supervisor.DISK_FLOOR + 1,
    })

    def fake_supervisor(argv, *, cwd, log, overall_seconds):
        assert 0 < overall_seconds <= supervisor.WALL_SECONDS
        calls.append((argv, cwd))
        return {**supervisor._safe_record("child_exit_zero", logged=True), "logging_outcome": "complete"}

    monkeypatch.setattr(supervisor, "_supervise_child", fake_supervisor)
    reference = "owner-approved-in-a-separate-record"
    revision = "a" * 40
    argv = ["--execute-approved-run", "--owner-approval-reference", reference,
            "--expected-revision", revision, "--diagnostic-log", str(tmp_path / "outside.jsonl")]
    assert supervisor._main(argv) == 0
    assert calls == [([sys.executable, "-B", str(supervisor._RUNNER),
                       "--execute-approved-run", "--owner-approval-reference", reference,
                       "--expected-revision", revision], supervisor._CODE)]
    assert json.loads(capsys.readouterr().out)["logged"] is True


def test_clean_revision_failure_precedes_diagnostic_path_access(monkeypatch, capsys):
    monkeypatch.setattr(supervisor, "_verify_clean_revision",
                        lambda _revision: (_ for _ in ()).throw(
                            supervisor.SupervisorError("revision_mismatch")))
    monkeypatch.setattr(supervisor, "_validate_log_path",
                        lambda _path: (_ for _ in ()).throw(AssertionError("path touched too early")))
    argv = ["--execute-approved-run", "--owner-approval-reference", "approved",
            "--expected-revision", "a" * 40, "--diagnostic-log", "outside.jsonl"]
    assert supervisor._main(argv) == 1
    assert json.loads(capsys.readouterr().out)["stop_reason"] == "revision_mismatch"


def test_reparse_log_parent_is_rejected_before_create(tmp_path, monkeypatch):
    parent = tmp_path / "reparse"
    parent.mkdir()
    original = supervisor._is_reparse
    monkeypatch.setattr(supervisor, "_is_reparse",
                        lambda path: True if path == parent else original(path))
    with pytest.raises(supervisor.SupervisorError, match="log_path_invalid"):
        supervisor._validate_log_path(str(parent / "new.jsonl"))
    assert list(parent.iterdir()) == []


def test_safe_success_uses_synthetic_child_and_records_supervisory_metadata(tmp_path):
    result, records, _ = _run(tmp_path, "pass\n")
    assert result["child_exit_code"] == 0
    assert result["stop_reason"] == "child_exit_zero"
    assert result["process_tree_terminated"] is True
    assert result["termination_method"] == "natural_exit"
    assert result["logging_outcome"] == "final_write_flushed_close_pending"
    assert records[-1]["event"] == "final"
    assert records[-1]["artifact_disposition"].endswith("manual_review_on_failure")
    assert records[-1]["automatic_retry"] is False
    assert records[-1]["rollback_claimed"] is False


def test_unsuccessful_child_retains_only_validated_redacted_diagnostic(tmp_path):
    code = "import sys; sys.stderr.write(" + repr(_diagnostic()) + "); raise SystemExit(7)\n"
    result, records, log_path = _run(tmp_path, code)
    assert result["child_exit_code"] == 7
    assert result["stop_reason"] == "child_exit_nonzero"
    assert result["diagnostic_outcome"] == "validated"
    assert result["child_diagnostic"]["exceptions"][0]["objective_index"] == 1
    assert "SENSITIVE" not in log_path.read_text(encoding="utf-8")
    assert records[-1]["child_exit_code"] == 7


@pytest.mark.parametrize("raw", [
    b"{partial", b"\xff", b"{}", b"[]", b"null", b"NaN", b"{}{}",
])
def test_malformed_partial_or_unknown_diagnostic_is_explicitly_rejected(raw):
    with pytest.raises(ValueError):
        supervisor._validate_child_diagnostic(raw)


def test_sensitive_text_unknown_fields_and_untrusted_frames_never_persist(tmp_path):
    unsafe = _diagnostic(message="SENSITIVE_ROWS")
    code = "import sys; sys.stderr.write(" + repr(unsafe) + "); raise SystemExit(1)\n"
    result, _, log_path = _run(tmp_path, code)
    assert result["stop_reason"] == "diagnostic_invalid"
    assert "SENSITIVE_ROWS" not in log_path.read_text(encoding="utf-8")
    unsafe_frame = _diagnostic(frames=[{"file": "set_c_split_release.py",
                                        "function": "arbitrary_secret_name", "line": 1}])
    clean = supervisor._validate_child_diagnostic(unsafe_frame.encode())
    assert clean["exceptions"][0]["frames"] == [
        {"file": "set_c_split_release.py", "function": "[external]", "line": 1}
    ]


@pytest.mark.parametrize("frame", [
    {"file": "set_c_split_release.py", "function": 7, "line": 1},
    {"file": "SENSITIVE.py", "function": "unknown", "line": 1},
    {"file": "set_c_split_release.py", "function": "unknown", "line": -1},
    {"file": "set_c_split_release.py", "function": "unknown", "line": 1, "extra": "x"},
])
def test_untrusted_frame_shape_remains_strict(frame):
    with pytest.raises(ValueError):
        supervisor._validate_child_diagnostic(_diagnostic(frames=[frame]).encode())


@pytest.mark.parametrize("location", ["diagnostic", "exception"])
def test_unknown_json_fields_remain_rejected(location):
    record = json.loads(_diagnostic())
    if location == "diagnostic":
        record["unknown_field"] = "SENSITIVE_UNKNOWN_FIELD"
    else:
        record["exceptions"][0]["unknown_field"] = "SENSITIVE_UNKNOWN_FIELD"
    with pytest.raises(ValueError):
        supervisor._validate_child_diagnostic(json.dumps(record).encode())


def test_actual_runner_diagnostic_redacts_unknown_frames_and_retains_solver_context(
    monkeypatch, capsys, tmp_path,
):
    source = """
def SENSITIVE_FUNCTION():
    try:
        raise ValueError('SENSITIVE_CAUSE')
    except ValueError as cause:
        error = RuntimeError('SENSITIVE_MESSAGE')
        error.set_c_diagnostic = {
            'stage': 'allocation', 'objective_phase': 'positive_deviation',
            'objective_index': 2, 'objective_ordinal': 2, 'solver_status': 1,
            'solver_success': False, 'has_solution': False, 'elapsed_seconds': 4.5,
            'time_limit': 3, 'mip_rel_gap': 0,
            'failure_reason': 'exact_optimum_not_proven',
        }
        raise error from cause
"""
    namespace = {}
    exec(compile(source, runner.__file__, "exec"), namespace)
    monkeypatch.setattr(runner.argparse, "ArgumentParser",
                        lambda **_kwargs: namespace["SENSITIVE_FUNCTION"]())

    assert runner._cli([]) == 1
    emitted = capsys.readouterr().err.strip()
    raw = json.loads(emitted)
    frames = [frame for item in raw["exceptions"] for frame in item["frames"]]
    assert any(frame["function"] == "SENSITIVE_FUNCTION" for frame in frames)
    assert any(frame["file"] == "run_set_c_split.py" and frame["function"] == "_parser"
               for frame in frames)
    assert [item["type"] for item in raw["exceptions"]] == ["RuntimeError", "ValueError"]
    assert raw["exceptions"][0]["objective_index"] == 2
    assert all(item["message"] == "[redacted]" for item in raw["exceptions"])

    child_code = "import sys; sys.stderr.write(" + repr(emitted) + "); raise SystemExit(1)\n"
    result, _, log_path = _run(tmp_path, child_code)
    assert result["stop_reason"] == "child_exit_nonzero"
    assert result["diagnostic_outcome"] == "validated"
    exceptions = result["child_diagnostic"]["exceptions"]
    assert exceptions[0]["frames"]
    assert all(frame["function"] != "SENSITIVE_FUNCTION"
               for item in exceptions for frame in item["frames"])
    assert exceptions[0]["frames"][-1]["function"] == "[external]"
    assert [item["type"] for item in exceptions] == ["RuntimeError", "ValueError"]
    assert exceptions[0]["stage"] == "allocation"
    assert exceptions[0]["objective_index"] == 2
    assert exceptions[0]["solver_status"] == 1
    assert exceptions[0]["solver_success"] is False
    assert exceptions[0]["has_solution"] is False
    assert exceptions[0]["objective_phase"] == "positive_deviation"
    assert exceptions[0]["time_limit"] == 3
    assert exceptions[0]["mip_rel_gap"] == 0
    assert exceptions[0]["failure_reason"] == "exact_optimum_not_proven"
    retained = log_path.read_text(encoding="utf-8")
    assert "SENSITIVE_FUNCTION" not in retained
    assert "SENSITIVE_MESSAGE" not in retained
    assert "SENSITIVE_CAUSE" not in retained


def test_concurrent_large_stdout_and_stderr_are_bounded_and_drained(tmp_path):
    code = (
        "import os, time\n"
        "for i in range(64):\n"
        " os.write(1, b'o' * 8192); os.write(2, b'e' * 8192)\n"
        "time.sleep(5)\n"
    )
    result, _, log_path = _run(tmp_path, code, timeout=3)
    assert result["stop_reason"] in {"stdout_oversized", "stderr_oversized"}
    assert result["stream_bytes_seen"]["stdout"] > supervisor.STREAM_LIMIT or result["stream_bytes_seen"]["stderr"] > supervisor.STREAM_LIMIT
    assert max(result["stream_bytes_retained"].values()) == supervisor.STREAM_LIMIT
    assert result["process_tree_terminated"] is True
    assert "oooo" not in log_path.read_text(encoding="utf-8")
    assert "eeee" not in log_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("body,reason", [
    ("", "diagnostic_missing"),
    ("import sys; sys.stderr.write('{partial')", "diagnostic_invalid"),
    ("import sys; sys.stderr.write('x' * 33000)", "diagnostic_invalid"),
])
def test_missing_partial_and_oversized_diagnostic_output_is_explicit(tmp_path, body, reason):
    code = body + "\nraise SystemExit(2)\n"
    result, _, log_path = _run(tmp_path, code)
    assert result["stop_reason"] == reason
    assert result["diagnostic_outcome"] in {"missing", "invalid_or_partial"}
    assert "xxxxxxxx" not in log_path.read_text(encoding="utf-8")


def test_timeout_terminates_descendants_and_records_post_stop_exit(tmp_path):
    descendant = tmp_path / "descendant.py"
    marker = tmp_path / "escaped.txt"
    descendant.write_text(
        "import time; time.sleep(1.0); open(" + repr(str(marker)) + ", 'w').write('escaped')\n",
        encoding="utf-8",
    )
    code = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-B', " + repr(str(descendant)) + "])\n"
        "time.sleep(10)\n"
    )
    result, _, _ = _run(tmp_path, code, timeout=0.4)
    time.sleep(1.2)
    assert result["stop_reason"] == "timeout"
    assert result["process_tree_terminated"] is True
    assert result["child_exit_code"] is not None
    assert not marker.exists()


def test_native_resource_sampler_observes_only_synthetic_child_and_temp_volume(tmp_path):
    script = tmp_path / "sampled_child.py"
    script.write_text("import time; time.sleep(10)\n", encoding="utf-8")
    api = supervisor._WinApi()
    child = api.launch([sys.executable, "-B", str(script)], tmp_path)
    try:
        sample = supervisor._resource_sample(child, tmp_path)
        assert sample["available_physical_memory_bytes"] > 0
        assert sample["child_working_set_bytes"] is not None
        assert sample["free_disk_bytes"] > 0
        assert sample["resource_sample_kind"] == "windows_point_sample"
    finally:
        child.kill()
        child.close()


@pytest.mark.parametrize("resource,reason", [
    ({"available_physical_memory_bytes": 1, "free_disk_bytes": supervisor.DISK_FLOOR + 1}, "memory_breach"),
    ({"available_physical_memory_bytes": supervisor.MEMORY_FLOOR + 1, "free_disk_bytes": 1}, "disk_breach"),
])
def test_memory_and_disk_breaches_stop_child_tree(tmp_path, resource, reason):
    def breached(_child, _disk):
        return {**resource, "child_working_set_bytes": 1, "resource_sample_kind": "synthetic"}

    result, _, _ = _run(tmp_path, "import time; time.sleep(10)\n", resource_reader=breached)
    assert result["stop_reason"] == reason
    assert result["process_tree_terminated"] is True


def test_monitoring_failure_stops_tree_without_raw_error_text(tmp_path):
    def fail(_child, _disk):
        raise OSError("SENSITIVE_PATH_AND_ROWS")

    result, _, log_path = _run(tmp_path, "import time; time.sleep(10)\n", resource_reader=fail)
    assert result["stop_reason"] == "monitoring_failure"
    assert result["process_tree_terminated"] is True
    assert "SENSITIVE" not in log_path.read_text(encoding="utf-8")


def test_stream_drain_failure_stops_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor._WinApi, "read_pipe",
                        lambda *_args: (_ for _ in ()).throw(OSError("SENSITIVE_PIPE")))
    result, _, log_path = _run(tmp_path, "import time; time.sleep(10)\n")
    assert result["stop_reason"] == "stream_drain_failure"
    assert result["process_tree_terminated"] is True
    assert "SENSITIVE" not in log_path.read_text(encoding="utf-8")


def test_log_write_failure_stops_tree_and_records_failure_outcome(tmp_path):
    class FailedLog:
        def write(self, _record):
            raise OSError("SENSITIVE_LOG_PATH")

    script = tmp_path / "sleep.py"
    script.write_text("import time; time.sleep(10)\n", encoding="utf-8")
    result = supervisor._supervise_child(
        [sys.executable, "-B", str(script)], cwd=tmp_path, log=FailedLog(),
        overall_seconds=2, sample_seconds=0.05, resource_reader=_healthy,
    )
    assert result["stop_reason"] == "logging_failure"
    assert result["process_tree_terminated"] is True
    assert result["logging_outcome"] == "write_failed"
    assert "SENSITIVE" not in json.dumps(result)


def test_log_failure_after_synthetic_publication_holds_and_preserves_artifact(tmp_path):
    destination = tmp_path / "synthetic-protected-bundle"
    destination.mkdir()
    marker = destination / "published-marker.txt"
    code = (
        "from pathlib import Path; import time; Path(" + repr(str(marker)) +
        ").write_text('synthetic publication evidence'); time.sleep(10)\n"
    )

    class LateFailureLog:
        def write(self, _record):
            deadline = time.monotonic() + 2.0
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not marker.exists():
                raise AssertionError("synthetic child did not publish within the bounded wait")
            raise OSError("SENSITIVE_LOG_PATH")

    script = tmp_path / "publish_then_wait.py"
    script.write_text(code, encoding="utf-8")
    result = supervisor._supervise_child(
        [sys.executable, "-B", str(script)], cwd=tmp_path, log=LateFailureLog(),
        overall_seconds=5.0, sample_seconds=0.05, resource_reader=_healthy,
    )

    assert result["stop_reason"] == "logging_failure"
    assert result["process_tree_terminated"] is True
    assert result["logging_outcome"] == "write_failed"
    assert result["artifact_disposition"] == "preserve_all_outputs; manual_review_on_failure"
    assert result["automatic_retry"] is False
    assert result["rollback_claimed"] is False
    assert marker.read_text(encoding="utf-8") == "synthetic publication evidence"
    assert "SENSITIVE" not in json.dumps(result)


def test_log_fsync_failure_stops_tree_and_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor.os, "fsync",
                        lambda _fd: (_ for _ in ()).throw(OSError("SENSITIVE_FSYNC")))
    script = tmp_path / "sleep.py"
    script.write_text("import time; time.sleep(10)\n", encoding="utf-8")
    log = supervisor._JsonlLog(tmp_path / "failed-fsync.jsonl")
    result = supervisor._supervise_child(
        [sys.executable, "-B", str(script)], cwd=tmp_path, log=log,
        overall_seconds=2, sample_seconds=0.05, resource_reader=_healthy,
    )
    log.close()
    assert result["stop_reason"] == "logging_failure"
    assert result["process_tree_terminated"] is True
    assert result["logging_outcome"] == "write_failed"
    assert "SENSITIVE" not in json.dumps(result)


def test_exclusive_log_create_preserves_existing_destination(tmp_path):
    path = tmp_path / "diagnostic.jsonl"
    path.write_bytes(b"existing synthetic evidence")
    with pytest.raises(FileExistsError):
        supervisor._JsonlLog(path)
    assert path.read_bytes() == b"existing synthetic evidence"


def test_postpublication_timeout_holds_and_preserves_synthetic_destination(tmp_path):
    destination = tmp_path / "synthetic-protected-bundle"
    destination.mkdir()
    marker = destination / "preserve.txt"
    code = "from pathlib import Path; import time; Path(" + repr(str(marker)) + ").write_text('synthetic hold evidence'); time.sleep(10)\n"
    result, _, _ = _run(tmp_path, code, timeout=0.4)
    assert result["stop_reason"] == "timeout"
    assert result["artifact_disposition"] == "preserve_all_outputs; manual_review_on_failure"
    assert result["automatic_retry"] is False and result["rollback_claimed"] is False
    assert marker.read_text(encoding="utf-8") == "synthetic hold evidence"


@pytest.mark.parametrize("resource,reason", [
    ({"available_physical_memory_bytes": 1, "free_disk_bytes": supervisor.DISK_FLOOR + 1}, "memory_floor"),
    ({"available_physical_memory_bytes": supervisor.MEMORY_FLOOR + 1, "free_disk_bytes": 1}, "disk_floor"),
    (None, "monitoring_failure"),
])
def test_prelaunch_resource_failure_is_recorded_without_child_launch(
    monkeypatch, tmp_path, capsys, resource, reason,
):
    class MemoryLog:
        instances = []
        def __init__(self, _path):
            self.records = []
            self.closed = False
            self.instances.append(self)
        def write(self, record):
            self.records.append(record)
        def close(self):
            self.closed = True

    monkeypatch.setattr(supervisor, "_verify_clean_revision", lambda _revision: None)
    monkeypatch.setattr(supervisor, "_validate_log_path", lambda _path: tmp_path / "log.jsonl")
    monkeypatch.setattr(supervisor, "_JsonlLog", MemoryLog)
    def sample(*_args):
        if resource is None:
            raise OSError("SENSITIVE_MONITOR")
        return {**resource, "child_working_set_bytes": None,
                "resource_sample_kind": "synthetic"}
    monkeypatch.setattr(supervisor, "_resource_sample", sample)
    monkeypatch.setattr(supervisor, "_supervise_child",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("prelaunch failure must block child")))
    argv = ["--execute-approved-run", "--owner-approval-reference", "separate-owner-record",
            "--expected-revision", "a" * 40, "--diagnostic-log", str(tmp_path / "log.jsonl")]
    assert supervisor._main(argv) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["stop_reason"] == reason
    assert result["logged"] is True
    if resource is not None:
        assert result["resource_observations"] == [
            {key: value for key, value in MemoryLog.instances[0].records[1].items()
             if key != "event"}
        ]
    else:
        assert result["resource_observations"] == []
    assert MemoryLog.instances[0].closed
    assert MemoryLog.instances[0].records[-1]["event"] == "supervisor_final"


def test_log_close_failure_is_not_reported_as_logged(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(supervisor, "_verify_clean_revision", lambda _revision: None)
    monkeypatch.setattr(supervisor, "_validate_log_path", lambda _raw: tmp_path / "log.jsonl")

    class CloseFailure(supervisor._JsonlLog):
        def close(self):
            raise OSError("SENSITIVE_CLOSE")

    monkeypatch.setattr(supervisor, "_JsonlLog", CloseFailure)
    monkeypatch.setattr(supervisor, "_resource_sample", lambda *_: {
        "available_physical_memory_bytes": 1, "free_disk_bytes": 1,
        "child_working_set_bytes": None, "resource_sample_kind": "synthetic",
    })
    argv = ["--execute-approved-run", "--owner-approval-reference", "approved-separately",
            "--expected-revision", "a" * 40, "--diagnostic-log", str(tmp_path / "log.jsonl")]
    assert supervisor._main(argv) == 1
    record = json.loads(capsys.readouterr().out)
    assert record["logged"] is False
    assert record["logging_outcome"] == "close_failed"
    assert record["stop_reason"] == "logging_failure"
    assert "SENSITIVE" not in json.dumps(record)


def test_timeout_contains_descendant_after_root_has_exited(tmp_path):
    marker = tmp_path / "descendant-escaped.txt"
    started = tmp_path / "root-started.txt"
    descendant = tmp_path / "long-descendant.py"
    descendant.write_text(
        "from pathlib import Path; import time; time.sleep(1); Path(" + repr(str(marker)) +
        ").write_text('escaped')\n", encoding="utf-8",
    )
    code = (
        "import subprocess, sys; from pathlib import Path\n"
        "subprocess.Popen([sys.executable, '-B', " + repr(str(descendant)) + "])\n"
        "Path(" + repr(str(started)) + ").write_text('started')\n"
    )
    result, _, _ = _run(tmp_path, code, timeout=0.4)
    time.sleep(1.1)
    assert started.exists()
    assert result["stop_reason"] == "timeout"
    assert result["child_exit_code"] == 0
    assert result["process_tree_terminated"] is True
    assert not marker.exists()


def test_interrupt_produces_safe_record_and_stops_tree(tmp_path):
    def interrupt(*_args):
        raise KeyboardInterrupt("SENSITIVE_INTERRUPT")

    result, records, log_path = _run(
        tmp_path, "import time; time.sleep(10)\n", resource_reader=interrupt,
    )
    assert result["stop_reason"] == "interrupted"
    assert result["process_tree_terminated"] is True
    assert result["child_exit_code"] is not None
    assert records[-1]["stop_reason"] == "interrupted"
    assert "SENSITIVE" not in log_path.read_text(encoding="utf-8")


def test_failed_explicit_termination_uses_close_guard_without_claiming_confirmation(tmp_path, monkeypatch):
    def fail(*_args):
        raise OSError("SENSITIVE_TERMINATION")

    monkeypatch.setattr(supervisor._WinApi, "kill", fail)
    result, _, log_path = _run(tmp_path, "import time; time.sleep(10)\n", timeout=0.2)
    assert result["stop_reason"] == "tree_termination_failure"
    assert result["stop_trigger"] == "timeout"
    assert result["termination_method"] == "kill_on_job_close_fallback_unconfirmed"
    assert result["process_tree_terminated"] is False
    assert result["stdout_drain"] == result["stderr_drain"] == "complete"
    assert "SENSITIVE" not in log_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("bad", [
    {"extra": "SENSITIVE_METADATA"},
    {"free_disk_bytes": "SENSITIVE_DISK"},
    {"available_physical_memory_bytes": True},
])
def test_malformed_monitor_values_fail_before_logging(tmp_path, bad):
    def malformed(child, disk):
        return {**_healthy(child, disk), **bad}

    result, _, log_path = _run(tmp_path, "import time; time.sleep(10)\n", resource_reader=malformed)
    assert result["stop_reason"] == "monitoring_failure"
    assert result["process_tree_terminated"] is True
    assert result["resource_observations"] == []
    assert "SENSITIVE" not in log_path.read_text(encoding="utf-8")


def test_duplicate_keys_and_sensitive_timestamp_cannot_enter_retained_diagnostic(tmp_path):
    duplicate = _diagnostic().replace('"schema_version":', '"schema_version":"secret","schema_version":', 1)
    unsafe_time = json.loads(_diagnostic())
    unsafe_time["timestamp_utc"] = "2026-10-03TSENSITIVE+00:00"
    for raw in (duplicate, json.dumps(unsafe_time)):
        with pytest.raises(ValueError):
            supervisor._validate_child_diagnostic(raw.encode())
    result, _, log_path = _run(
        tmp_path, "import sys; sys.stderr.write(" + repr(duplicate) + "); sys.exit(1)\n",
    )
    assert result["stop_reason"] == "diagnostic_invalid"
    assert "secret" not in log_path.read_text(encoding="utf-8")


def test_supervisory_output_failure_falls_back_without_raw_traceback(monkeypatch, capsys):
    class BrokenOutput:
        def write(self, _text):
            raise OSError("SENSITIVE_OUTPUT")

    monkeypatch.setattr(supervisor.sys, "stdout", BrokenOutput())
    assert supervisor._main([]) == 1
    record = json.loads(capsys.readouterr().err)
    assert record["stop_reason"] == "approval_required"
    assert "SENSITIVE" not in json.dumps(record)
