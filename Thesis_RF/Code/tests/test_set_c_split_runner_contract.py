"""The one-run split runner is procedural and deny-by-default."""

import json

import pytest

import run_set_c_split as runner


def test_runner_denies_before_import_or_package_access_without_opt_in(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "modules.set_c_split_release":
            raise AssertionError("release exporter imported before default-deny gate")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(PermissionError, match="denied by default"):
        runner.main([])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--execute-approved-run", "--expected-revision", "a" * 40],
        ["--execute-approved-run", "--owner-approval-reference", "approval-1"],
    ],
)
def test_runner_requires_reference_and_full_revision_before_export(arguments, monkeypatch):
    called = []

    def unexpected_export(*args, **kwargs):
        called.append((args, kwargs))

    monkeypatch.setitem(
        __import__("sys").modules,
        "modules.set_c_split_release",
        type("ReleaseModule", (), {"export_set_c_split": unexpected_export}),
    )
    with pytest.raises(PermissionError):
        runner.main(arguments)
    assert called == []


def test_runner_invocation_is_complete_and_uses_fixed_destinations(monkeypatch):
    calls = []

    def capture_export(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setitem(
        __import__("sys").modules,
        "modules.set_c_split_release",
        type("ReleaseModule", (), {"export_set_c_split": capture_export}),
    )
    runner.main([
        "--execute-approved-run",
        "--owner-approval-reference", "approval-record-2026-10-02",
        "--expected-revision", "a" * 40,
    ])
    assert len(calls) == 1
    (source,), kwargs = calls[0]
    repository = runner.Path(runner.__file__).resolve().parents[2]
    assert source == (
        repository / "Thesis_RF" / "Code" / "output" / "datasets" /
        "authoritative_set_c" / "v1" / "set_c.v1.csv"
    )
    assert kwargs == {
        "protected_destination": repository / "SetC_Protected" / "set_c_v1",
        "development_destination": repository / "SetC_Development" / "set_c_v1",
        "production_revision": "a" * 40,
        "authorization_reference": "approval-record-2026-10-02",
        "operator_opt_in": True,
    }


def test_cli_reports_default_denial_without_importing_exporter(monkeypatch, capsys):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "modules.set_c_split_release":
            raise AssertionError("exporter imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert runner._cli([]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    record = json.loads(output.err)
    assert record["exceptions"][0]["type"] == "PermissionError"
    assert record["exceptions"][0]["message"] in runner._SAFE_MESSAGES
    assert record["exceptions"][0]["stage"] == "runner_gate"
    assert "Traceback (most recent call last)" not in output.err


def test_cli_retains_sanitized_chain_and_excludes_sensitive_data(monkeypatch, capsys):
    def failure(argv):
        try:
            exec(compile("raise OSError('SENSITIVE_ROWS_AND_MEMBERSHIP')", "C:/SENSITIVE_PATH/data.csv", "exec"))
        except OSError as cause:
            cause.set_c_diagnostic = {"stage": "development_publication"}
            error = RuntimeError("Development publication failed; protected publication remains incomplete and requires human recovery")
            error.set_c_diagnostic = {
                "stage": "development_publication", "objective_phase": "tie_break",
                "objective_index": 9, "objective_ordinal": 1, "solver_status": 1,
                "solver_success": False, "has_solution": True, "mip_node_count": 17,
                "elapsed_seconds": 3.1, "time_limit": 60, "mip_rel_gap": 0,
                "failure_reason": "exact_optimum_not_proven",
                "x": ["SENSITIVE_SOLUTION"], "message": "SENSITIVE_SOLVER_TEXT",
                "fun": 3, "mip_dual_bound": 2, "mip_gap": 0.25,
                "component_id": "SENSITIVE_COMPONENT", "authorization_reference": "SENSITIVE_APPROVAL",
            }
            raise error from cause

    monkeypatch.setattr(runner, "main", failure)
    assert runner._cli([]) == 1
    output = capsys.readouterr()
    record = json.loads(output.err)
    outer, inner = record["exceptions"]
    assert (outer["type"], inner["type"]) == ("RuntimeError", "OSError")
    assert inner["message"] == "[redacted]"
    assert outer["stage"] == inner["stage"] == "development_publication"
    assert outer["solver_status"] == 1 and outer["objective_index"] == 9
    assert outer["mip_node_count"] == 17 and outer["has_solution"] is True
    assert "SENSITIVE" not in output.err
    assert all("locals" not in frame and "source" not in frame for item in record["exceptions"] for frame in item["frames"])
    assert not {"x", "fun", "mip_dual_bound", "mip_gap", "component_id"} & outer.keys()


def test_failure_record_rejects_untyped_out_of_bounds_and_unknown_metadata():
    error = ValueError("SENSITIVE_UNKNOWN_ERROR")
    error.set_c_diagnostic = {
        "stage": "SENSITIVE_STAGE", "objective_phase": ["tie_break"],
        "failure_reason": "SENSITIVE_REASON", "objective_index": True,
        "objective_ordinal": 27, "solver_status": "1", "mip_node_count": -1,
        "solver_success": "False", "has_solution": 1,
        "elapsed_seconds": float("nan"), "time_limit": 30, "mip_rel_gap": 0.1,
    }
    item = runner._failure_record(error)["exceptions"][0]
    assert set(item) == {"type", "message", "stage", "frames"}
    assert item["message"] == "[redacted]"


@pytest.mark.parametrize("time_limit", [3, 30, 61, 0, 60.0, True])
def test_failure_record_rejects_wrong_time_limit_alone(time_limit):
    error = ValueError("SENSITIVE_UNKNOWN_ERROR")
    error.set_c_diagnostic = {
        "stage": "allocation", "objective_phase": "row_deviation",
        "failure_reason": "exact_optimum_not_proven", "objective_index": 7,
        "objective_ordinal": 3, "solver_status": 1, "mip_node_count": 1,
        "solver_success": False, "has_solution": True, "elapsed_seconds": 60.1,
        "time_limit": time_limit, "mip_rel_gap": 0,
    }
    item = runner._failure_record(error)["exceptions"][0]
    assert item["objective_index"] == 7 and item["mip_rel_gap"] == 0
    assert "time_limit" not in item


def test_failure_record_bounds_cyclic_exception_chain_and_redacts_unknown_type():
    error = type("SENSITIVE_TYPE", (Exception,), {})("SENSITIVE_MESSAGE")
    error.__cause__ = error
    record = runner._failure_record(error)
    assert len(record["exceptions"]) == 1
    assert record["exceptions"][0]["type"] == "Exception"
    assert "SENSITIVE" not in json.dumps(record)


def test_cli_reporter_failure_preserves_nonzero_status(monkeypatch, capsys):
    def failure(argv):
        raise RuntimeError("SENSITIVE_ORIGINAL")

    def broken_reporter(error):
        raise ValueError("SENSITIVE_REPORTER")

    monkeypatch.setattr(runner, "main", failure)
    monkeypatch.setattr(runner, "_failure_record", broken_reporter)
    assert runner._cli([]) == 1
    output = capsys.readouterr()
    assert json.loads(output.err)["reporting_failed"] is True
    assert "SENSITIVE" not in output.err


def test_cli_broken_stderr_preserves_failure_status(monkeypatch):
    class BrokenStream:
        def write(self, value):
            raise OSError("SENSITIVE_STREAM")

    def failure(argv):
        raise RuntimeError("SENSITIVE_ORIGINAL")

    monkeypatch.setattr(runner, "main", failure)
    monkeypatch.setattr(runner.sys, "stderr", BrokenStream())
    assert runner._cli([]) == 1


def test_cli_success_keeps_zero_status_and_silent_output(monkeypatch, capsys):
    monkeypatch.setattr(runner, "main", lambda argv: None)
    assert runner._cli([]) == 0
    output = capsys.readouterr()
    assert output.out == output.err == ""
