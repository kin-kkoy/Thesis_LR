"""The one-run split runner is procedural and deny-by-default."""

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
