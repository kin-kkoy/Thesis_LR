"""Procedural one-run Set C export command; execution is denied by default."""

from __future__ import annotations

import argparse
from pathlib import Path


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


if __name__ == "__main__":
    main()
