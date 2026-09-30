"""Explicit entry point for the separately authorized Gate 6 v3 policy replay."""

from __future__ import annotations

import argparse
from typing import Sequence

from modules.set_c_population_replay import (
    approved_destination,
    approved_source,
    replay_policy_v3,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay Gate 6 adequacy from the immutable aggregate report."
    )
    parser.add_argument(
        "--execute-approved-policy-replay",
        action="store_true",
        help="Required one-run authorization gate; absent by default.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.execute_approved_policy_replay:
        parser.error(
            "--execute-approved-policy-replay is required; production replay is disabled"
        )
    result = replay_policy_v3(
        approved_source(),
        approved_destination(),
        production_authorized=True,
    )
    print(f"Gate 6 replay path: {result['path']}")
    print(f"Gate 6 replay payload SHA-256: {result['payload_sha256']}")
    print(f"Gate 6 replay component count: {result['component_count']}")
    print(f"Gate 6 replay adequacy status: {result['adequacy_status']}")
    return 0 if result["adequacy_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
