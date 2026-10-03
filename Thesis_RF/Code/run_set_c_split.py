"""Procedural one-run Set C export command; execution is denied by default."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys


_SAFE_MESSAGES = {
    "Set C allocation is denied by default; pass --execute-approved-run",
    "A nonempty separate owner approval reference is required",
    "A full reviewed source revision is required",
    "D-035 allocation did not prove an exact optimum",
    "D-035 allocation has a fractional solution",
    "D-035 allocation violates a hard constraint",
    "D-035 optimizer returned an invalid assignment",
    "Development publication failed; protected publication remains incomplete and requires human recovery",
}
_SAFE_STAGES = {
    "unknown", "runner_gate", "export_preflight", "allocation", "protected_staging",
    "development_staging", "protected_bundle", "development_bundle",
    "protected_validation", "protected_publication", "pair_receipt",
    "development_validation", "pair_validation", "development_publication",
}
_SAFE_TYPES = {
    "RuntimeError", "ValueError", "PermissionError", "FileExistsError",
    "FileNotFoundError", "OSError", "TypeError", "AttributeError", "AssertionError",
}


def _failure_record(error: Exception) -> dict:
    """Keep only allowlisted metadata; never serialize exceptions or locals."""
    chain = []
    seen = set()
    while error is not None and id(error) not in seen and len(chain) < 8:
        seen.add(id(error))
        name = type(error).__name__
        message = str(error)
        item = {
            "type": name if name in _SAFE_TYPES else "Exception",
            "message": message if message in _SAFE_MESSAGES else "[redacted]",
            "stage": "unknown", "frames": [],
        }
        trace = error.__traceback__
        while trace is not None and len(item["frames"]) < 32:
            code = trace.tb_frame.f_code
            filename = Path(code.co_filename).name
            known = filename in {"run_set_c_split.py", "set_c_split_release.py"}
            item["frames"].append({
                "file": filename if known else "[external]",
                "function": code.co_name if known else "[external]",
                "line": trace.tb_lineno,
            })
            if filename == "run_set_c_split.py" and code.co_name == "main":
                item["stage"] = "runner_gate"
            if filename == "set_c_split_release.py":
                item["stage"] = "export_preflight"
                if code.co_name in {"_allocate_components", "solve"}:
                    item["stage"] = "allocation"
            trace = trace.tb_next
        context = getattr(error, "set_c_diagnostic", None)
        if isinstance(context, dict):
            for key, choices in (
                ("stage", _SAFE_STAGES),
                ("objective_phase", {"positive_deviation", "row_deviation", "tie_break"}),
                ("failure_reason", {"solver_exception", "exact_optimum_not_proven", "fractional_solution", "hard_constraint_violation"}),
            ):
                value = context.get(key)
                if type(value) is str and value in choices:
                    item[key] = value
            for key, minimum, maximum in (
                ("objective_index", 1, 34), ("objective_ordinal", 1, 26),
                ("solver_status", 0, 4), ("mip_node_count", 0, 10**12),
            ):
                value = context.get(key)
                if type(value) is int and minimum <= value <= maximum:
                    item[key] = value
            for key in ("solver_success", "has_solution"):
                if type(context.get(key)) is bool:
                    item[key] = context[key]
            elapsed = context.get("elapsed_seconds")
            if type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= 10**9:
                item["elapsed_seconds"] = elapsed
            if type(context.get("time_limit")) is int and context["time_limit"] == 3:
                item["time_limit"] = 3
            if type(context.get("mip_rel_gap")) is int and context["mip_rel_gap"] == 0:
                item["mip_rel_gap"] = 0
        chain.append(item)
        error = error.__cause__ or (error.__context__ if not error.__suppress_context__ else None)
    return {
        "schema_version": "set_c_failure_diagnostic.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "exceptions": chain,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-approved-run",
        action="store_true",
        help="explicitly opt in to the separately approved one-run allocation/export",
    )
    parser.add_argument(
        "--owner-approval-reference",
        help="nonempty owner approval reference for audit evidence; not machine-verifiable proof",
    )
    parser.add_argument(
        "--expected-revision",
        help="full lowercase Git commit SHA approved for this one-run invocation",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if not args.execute_approved_run:
        raise PermissionError("Set C allocation is denied by default; pass --execute-approved-run")
    if not isinstance(args.owner_approval_reference, str) or not args.owner_approval_reference.strip():
        raise PermissionError("A nonempty separate owner approval reference is required")
    if not isinstance(args.expected_revision, str) or len(args.expected_revision) != 40:
        raise PermissionError("A full reviewed source revision is required")

    # Import and touch package/output paths only after the procedural gate.
    from modules.set_c_split_release import export_set_c_split

    repository = Path(__file__).resolve().parents[2]
    source = (
        repository / "Thesis_RF" / "Code" / "output" / "datasets" /
        "authoritative_set_c" / "v1" / "set_c.v1.csv"
    )
    export_set_c_split(
        source,
        protected_destination=repository / "SetC_Protected" / "set_c_v1",
        development_destination=repository / "SetC_Development" / "set_c_v1",
        production_revision=args.expected_revision,
        authorization_reference=args.owner_approval_reference.strip(),
        operator_opt_in=True,
    )


def _cli(argv: list[str] | None = None) -> int:
    """Report failures without Python's unrestricted traceback rendering."""
    try:
        main(argv)
    except Exception as error:
        try:
            record = _failure_record(error)
            print(json.dumps(record, sort_keys=True, allow_nan=False), file=sys.stderr, flush=True)
        except Exception:
            # Even a failed reporter must preserve the command's failure status.
            try:
                print('{"schema_version":"set_c_failure_diagnostic.v1","reporting_failed":true}', file=sys.stderr, flush=True)
            except Exception:
                pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
