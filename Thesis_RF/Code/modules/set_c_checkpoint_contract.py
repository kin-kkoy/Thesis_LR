"""Source-only Gate 6 v3 checkpoint and supervisor validation contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
import json
import os
from hashlib import sha256
from pathlib import Path

from .set_c_population_adequacy import (
    EXPECTED_RUN_COUNT, EXPANSION_RUNTIME_LIMIT_SECONDS, FREE_DISK_FLOOR_BYTES,
    LAUNCH_MEMORY_FLOOR_BYTES, RSS_CEILING_BYTES, RUN_RUNTIME_LIMIT_SECONDS,
    RUNTIME_MEMORY_FLOOR_BYTES, PRODUCTION_SOURCE_RELATIVE_PATHS,
    _canonical_json, _require_temp_descendant,
    PopulationAdequacyError,
)


CHECKPOINT_RELATIVE_PATH = Path(
    "output/phase9/.checkpoints/phase9_set_c_population_adequacy_v3.checkpoint.sqlite3")
TELEMETRY_RELATIVE_PATH = Path(
    "output/phase9/.checkpoints/phase9_set_c_population_adequacy_v3.resource.jsonl")
CHECKPOINT_SCHEMA = "phase9_set_c_population_adequacy_checkpoint.v3"
CHECKPOINT_PACKAGE_ID = "phase9_set_c_population_adequacy_v3"
RETENTION_DAYS = 30
SAMPLE_GAP_LIMIT_SECONDS = 5.0
REQUIRED_BINDING_HASHES = frozenset({
    "configuration_sha256", "environment_sha256",
    "inventory_payload_sha256", "inventory_file_sha256",
    "ordered_family_manifest_sha256", "run_order_sha256", "rng_seed_contract_sha256",
})
REQUIRED_TELEMETRY = frozenset({
    "execution_id", "attempt_id", "process_id", "process_started_at_utc",
    "captured_at_utc", "monotonic_elapsed_seconds", "phase", "run_order",
    "available_memory_bytes", "process_rss_bytes", "process_peak_rss_bytes",
    "free_disk_bytes", "completed_run_order", "process_exit_status",
})


def _reject(message: str) -> None:
    raise PopulationAdequacyError(message)


def _utc(value: object) -> datetime:
    if not isinstance(value, str):
        _reject("Checkpoint timestamp is missing")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PopulationAdequacyError("Checkpoint timestamp is invalid") from exc
    if result.tzinfo is None:
        _reject("Checkpoint timestamp must include UTC offset")
    return result.astimezone(timezone.utc)


def _digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value):
        _reject("Checkpoint binding hash is invalid")
    return value


def _state_digest(value: object, name: str) -> str:
    """Recompute a state digest; a correctly-shaped hash alone is insufficient."""
    if not isinstance(value, Mapping):
        _reject(f"Checkpoint {name} state is missing")
    return sha256(_canonical_json(value)).hexdigest()


def validate_storage_proof(
    directory: Path, verifier: Callable[[Path], Mapping[str, object]], *,
    executing_user_sid: str,
) -> None:
    """The injected Windows verifier supplies NTFS/reparse/owner/ACL evidence."""
    proof = verifier(directory)
    if not isinstance(proof, Mapping) or set(proof) != {
        "filesystem", "is_reparse_point", "inheritance_enabled", "owner_sid",
        "allowed_principals", "denied_principals",
    }:
        _reject("Checkpoint storage proof is incomplete")
    if (proof["filesystem"] != "NTFS" or proof["is_reparse_point"] is not False
            or proof["inheritance_enabled"] is not False
            or proof["owner_sid"] != executing_user_sid
            or set(proof["allowed_principals"]) != {
                executing_user_sid, "NT AUTHORITY\\SYSTEM", "BUILTIN\\Administrators"}
            or proof["denied_principals"]):
        _reject("Checkpoint NTFS owner or ACL policy failed")


def validate_binding(binding: Mapping[str, object], expected: Mapping[str, object]) -> None:
    if set(binding) != set(expected) or binding != expected:
        _reject("Checkpoint provenance binding drifted")
    if (binding.get("schema") != CHECKPOINT_SCHEMA
            or binding.get("package_id") != CHECKPOINT_PACKAGE_ID
            or binding.get("expansion_id") != CHECKPOINT_PACKAGE_ID
            or binding.get("source_dirty") is not False):
        _reject("Checkpoint schema or clean revision invalid")
    revision = binding.get("source_revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision):
        _reject("Checkpoint source revision invalid")
    if set(binding.get("hashes", {})) != REQUIRED_BINDING_HASHES:
        _reject("Checkpoint hash inventory incomplete")
    for value in binding["hashes"].values():
        _digest(value)
    source_hashes = binding.get("source_file_hashes")
    input_hashes = binding.get("input_hashes")
    if (not isinstance(source_hashes, Mapping)
            or set(source_hashes) != set(PRODUCTION_SOURCE_RELATIVE_PATHS)
            or not isinstance(input_hashes, Mapping) or not input_hashes):
        _reject("Checkpoint source-file or input hash inventory incomplete")
    for value in (*source_hashes.values(), *input_hashes.values()):
        _digest(value)
    if binding.get("run_count") != EXPECTED_RUN_COUNT:
        _reject("Checkpoint run count drifted")


def append_durable_telemetry(path: Path, sample: Mapping[str, object], *,
                             synthetic_test: bool = False) -> None:
    """Append one complete supervisor sample and force it to stable storage."""
    if not synthetic_test:
        raise PermissionError("Production telemetry requires separate execution approval")
    _require_temp_descendant(path, "telemetry")
    if set(sample) != REQUIRED_TELEMETRY:
        _reject("Resource telemetry fields are incomplete")
    with path.open("ab") as handle:
        handle.write(_canonical_json(sample) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_telemetry(path: Path, *, execution_id: str,
                       completed_run_order: int) -> dict[str, object]:
    """Reject gaps, missing exit, resource breaches, and uncharged attempts."""
    try:
        lines = path.read_bytes().splitlines()
        samples = [json.loads(line) for line in lines]
    except (OSError, ValueError) as exc:
        raise PopulationAdequacyError("Resource telemetry cannot be read") from exc
    if not samples or any(set(sample) != REQUIRED_TELEMETRY for sample in samples):
        _reject("Resource telemetry incomplete")
    attempts: dict[str, list[dict]] = {}
    last_completed = 0
    for sample in samples:
        if sample["execution_id"] != execution_id:
            _reject("Resource telemetry execution drifted")
        if type(sample["completed_run_order"]) is not int or not (
                last_completed <= sample["completed_run_order"] <= completed_run_order):
            _reject("Resource telemetry completed-run order drifted")
        last_completed = sample["completed_run_order"]
        if (type(sample["available_memory_bytes"]) is not int
                or type(sample["process_rss_bytes"]) is not int
                or type(sample["process_peak_rss_bytes"]) is not int
                or type(sample["free_disk_bytes"]) is not int
                or sample["available_memory_bytes"] < RUNTIME_MEMORY_FLOOR_BYTES
                or max(sample["process_rss_bytes"], sample["process_peak_rss_bytes"])
                    > RSS_CEILING_BYTES
                or sample["free_disk_bytes"] < FREE_DISK_FLOOR_BYTES):
            _reject("Resource telemetry breach")
        attempts.setdefault(sample["attempt_id"], []).append(sample)
    if last_completed != completed_run_order:
        _reject("Resource telemetry missing completed-run record")
    total_elapsed = 0.0
    observed_after = set()
    for records in attempts.values():
        first, last = records[0], records[-1]
        if first["phase"] != "launch" or last["phase"] != "exit":
            _reject("Crash attempt has no durable exit record")
        if first["available_memory_bytes"] < LAUNCH_MEMORY_FLOOR_BYTES:
            _reject("Launch memory floor breached")
        if type(last["process_exit_status"]) is not int:
            _reject("Crash attempt exit status missing")
        started = _utc(first["process_started_at_utc"])
        if any(_utc(item["process_started_at_utc"]) != started or
               item["process_id"] != first["process_id"] for item in records):
            _reject("Process identity drifted")
        prior_elapsed = -1.0
        prior_utc = None
        before = {}
        after = set()
        for item in records:
            elapsed = item["monotonic_elapsed_seconds"]
            captured = _utc(item["captured_at_utc"])
            if (type(elapsed) not in (int, float) or elapsed < 0
                    or elapsed < prior_elapsed
                    or (prior_elapsed >= 0
                        and elapsed - prior_elapsed > SAMPLE_GAP_LIMIT_SECONDS)
                    or (prior_utc is not None and
                        (captured - prior_utc).total_seconds() > SAMPLE_GAP_LIMIT_SECONDS)):
                _reject("Resource telemetry has a sampling gap")
            prior_elapsed = elapsed
            prior_utc = captured
            if item["phase"] == "before_run":
                before[item["run_order"]] = elapsed
            if item["phase"] == "after_run":
                if (item["run_order"] not in before
                        or elapsed - before[item["run_order"]] > RUN_RUNTIME_LIMIT_SECONDS):
                    _reject("Resource telemetry misses run start or exceeds per-run limit")
                after.add(item["run_order"])
        if observed_after.intersection(after):
            _reject("Resource telemetry duplicates completed runs")
        observed_after.update(after)
        total_elapsed += last["monotonic_elapsed_seconds"]
    if observed_after != set(range(1, completed_run_order + 1)):
        _reject("Resource telemetry lacks complete run boundaries")
    if total_elapsed > EXPANSION_RUNTIME_LIMIT_SECONDS:
        _reject("Cumulative runtime limit exceeded")
    return {"attempt_count": len(attempts), "elapsed_seconds": total_elapsed,
            "completed_run_order": last_completed}


def inspect_resume_contract(
    *, checkpoint_path: Path, telemetry_path: Path,
    checkpoint_record: Mapping[str, object], expected_binding: Mapping[str, object],
    storage_verifier: Callable[[Path], Mapping[str, object]],
    executing_user_sid: str, report_exists: bool,
    report_verifier: Callable[[], Mapping[str, object]] | None,
    sqlite_integrity_ok: bool, synthetic_test: bool = False,
) -> str:
    """Classify a future recovery without creating, resuming, or deleting files."""
    if not synthetic_test:
        raise PermissionError("Production checkpoint recovery requires separate approval")
    checkpoint_path = _require_temp_descendant(checkpoint_path, "checkpoint")
    telemetry_path = _require_temp_descendant(telemetry_path, "telemetry")
    if checkpoint_path.parent != telemetry_path.parent:
        _reject("Checkpoint and telemetry must share protected directory")
    if report_exists:
        if report_verifier is None:
            _reject("Existing report requires an independent verifier")
        try:
            evidence = report_verifier()
        except Exception as exc:
            raise PopulationAdequacyError(
                "Existing report verifier failed; preserve evidence") from exc
        if (not isinstance(evidence, Mapping)
                or set(evidence) != {
                    "status", "whole_file_sha256", "payload_sha256",
                    "complete_run_count", "provenance_verified", "integrity_verified",
                }
                or evidence["status"] != "complete"
                or evidence["complete_run_count"] != EXPECTED_RUN_COUNT
                or evidence["provenance_verified"] is not True
                or evidence["integrity_verified"] is not True):
            _reject("Existing report invalid; preserve evidence")
        _digest(evidence["whole_file_sha256"])
        _digest(evidence["payload_sha256"])
        return "published_validate_then_owner_approved_cleanup"
    if not checkpoint_path.is_file() or not telemetry_path.is_file():
        _reject("Checkpoint or telemetry missing")
    validate_storage_proof(checkpoint_path.parent, storage_verifier,
                           executing_user_sid=executing_user_sid)
    if not sqlite_integrity_ok:
        _reject("SQLite integrity failed")
    if set(checkpoint_record) != {
        "binding", "execution_id", "created_at_utc", "expires_at_utc",
        "completed_run_order", "aggregate_counts_sha256", "identity_totals_sha256",
        "cumulative_resource_sha256", "checkpoint_state",
    }:
        _reject("Checkpoint metadata incomplete")
    validate_binding(checkpoint_record["binding"], expected_binding)
    created = _utc(checkpoint_record["created_at_utc"])
    expires = _utc(checkpoint_record["expires_at_utc"])
    if (expires - created != timedelta(days=RETENTION_DAYS)
            or datetime.now(timezone.utc) > expires):
        _reject("Checkpoint retention expired or drifted")
    state = checkpoint_record["checkpoint_state"]
    if not isinstance(state, Mapping) or set(state) != {
            "aggregate_counts", "identity_totals", "cumulative_resource"}:
        _reject("Checkpoint state is incomplete")
    expected_state_digests = {
        "aggregate_counts_sha256": _state_digest(state["aggregate_counts"], "aggregate_counts"),
        "identity_totals_sha256": _state_digest(state["identity_totals"], "identity_totals"),
        "cumulative_resource_sha256": _state_digest(state["cumulative_resource"], "cumulative_resource"),
    }
    for key, expected_digest in expected_state_digests.items():
        if checkpoint_record[key] != expected_digest:
            _reject(f"Checkpoint {key} does not match checkpoint state")
    completed = checkpoint_record["completed_run_order"]
    if type(completed) is not int or not 0 <= completed < EXPECTED_RUN_COUNT:
        _reject("Checkpoint completed-run prefix invalid")
    validate_telemetry(telemetry_path, execution_id=checkpoint_record["execution_id"],
                       completed_run_order=completed)
    return "eligible_for_separately_authorized_resume"
