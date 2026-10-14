"""Build the approved Set C split contract; production execution stays disabled."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
import time

import pandas as pd
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .feature_pipeline import (
    AUTHORITATIVE_CSV_COLUMNS,
    AUTHORITATIVE_SPLIT_ASSIGNMENT_KEY_NAMES,
    CANONICAL_FEATURE_NAMES,
    DEVELOPMENT_SPLIT_ROLES,
    SET_C_SCHEMA_VERSION,
    TARGET_NAME,
    TARGET_VERSION,
    validate_feature_values,
)
from .set_c_leakage import LEAKAGE_GRAPH_VERSION, build_joint_leakage_components
from .set_c_publication import PACKAGE_SCHEMA_VERSION, validate_published_package


RELEASE_SCHEMA_VERSION = "set_c_development_projection.v1"
ATTESTATION_SCHEMA_VERSION = "set_c_development_attestation.v1"
PHASE13_SET_C_RF_PROTOCOL_ID = "phase13_set_c_rf_protocol.v1"
PHASE13_SET_C_RF_PROTOCOL_SHA256 = "69c39d1d97430ebb903ec2d856e6205a2c1b8ad77641798955d6ad2bdb671637"
APPROVED_CSV_SHA256 = "7f0f1867445141f365687d127e449117f234348cec571990d70bc2b226ba40f6"
APPROVED_MANIFEST_SHA256 = "b700f1b9f196893f23e7190daf1bc3c68b0f569f99a977ec9e919f0de73e9e3c"
APPROVED_CALIBRATION_ATTESTATION_SHA256 = "4fcf286156892c9da0289e56712210e90cccdcd608641c7ee573a8db1dcea23f"
APPROVED_CALIBRATION_CSV_SHA256 = "e424865be9143823f5a15fce27a3eb6df74abe802be84f142505f1209e3a77dc"
APPROVED_CALIBRATION_COUNTS = (2060, 153, 1907)
RELEASE_COLUMNS = (*CANONICAL_FEATURE_NAMES, TARGET_NAME)
_FORBIDDEN_PATH_PARTS = {"setc_protected", "authoritative_set_c"}
APPROVED_POLICY_VERSION = "set_c_deterministic_allocation_policy.v1"
FOUR_ROLES = (*DEVELOPMENT_SPLIT_ROLES, "final_test")
ROLE_COMPONENT_TARGETS = (11, 5, 5, 5)
ROLE_POSITIVE_MINIMA = (200, 100, 100, 100)
ROLE_NEGATIVE_MINIMA = (800, 400, 400, 400)
ROLE_POSITIVE_BALANCE = (338, 153, 153, 153)
ROLE_ROW_BALANCE = (4541, 2064, 2064, 2064)
MATERIAL_COLUMNS = tuple(f"material_{number}" for number in range(1, 6))
PROTECTED_SCHEMA_VERSION = "set_c_protected_package.v1"
MAPPING_SCHEMA_VERSION = "set_c_complete_split_manifest.v1"
FINAL_TEST_SCHEMA_VERSION = "set_c_final_test_rows.v1"
PROTECTED_ATTESTATION_SCHEMA_VERSION = "set_c_protected_attestation.v1"
PAIR_PUBLICATION_SCHEMA_VERSION = "set_c_pair_publication.v1"
PROTECTED_DIRECTORY_NAME = "SetC_Protected"
DEVELOPMENT_DIRECTORY_NAME = "SetC_Development"
SOURCE_MANIFEST_VERSION = "set_c_source_manifest.v1"
ALLOCATION_CONFIG_VERSION = "set_c_allocation_config.v1"
SOURCE_REVISION_VERSION = "set_c_source_revision.v1"
AUTHORIZATION_VERSION = "set_c_export_authorization.v1"
SPLIT_MANIFEST_COLUMNS = (*AUTHORITATIVE_SPLIT_ASSIGNMENT_KEY_NAMES, "leakage_component_id", "split_role")


def _attach_failure_context(error: Exception, context: dict) -> None:
    """Diagnostics must never replace an allocation/publication failure."""
    try:
        error.set_c_diagnostic = context
    except Exception:
        pass


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_reparse_point(path: Path) -> bool:
    attributes = os.lstat(path)
    return path.is_symlink() or bool(getattr(attributes, "st_file_attributes", 0) & 0x400)


def _verify_clean_source_revision(repo: Path, expected_revision: str) -> None:
    """Require the approved source commit and a clean tracked/untracked checkout."""
    if len(expected_revision) != 40 or any(char not in "0123456789abcdef" for char in expected_revision):
        raise PermissionError("A full lowercase Git source revision is required")

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *arguments],
            check=True, capture_output=True, text=True, encoding="utf-8", timeout=10,
        )
        return result.stdout.strip()

    try:
        actual_revision = git("rev-parse", "--verify", "HEAD")
        dirty_paths = git("status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.SubprocessError) as error:
        raise PermissionError("Unable to verify the production source checkout") from error
    if actual_revision != expected_revision or dirty_paths:
        raise PermissionError("Production source revision differs or the checkout is not clean")


def _validate_output_path(path: Path, *, synthetic: bool, kind: str) -> Path:
    """Constrain production paths to D-038 and synthetic paths to temp only."""
    absolute = path.absolute()
    if synthetic:
        temp_root = Path(tempfile.gettempdir()).resolve()
        if not absolute.is_relative_to(temp_root) or absolute == temp_root:
            raise PermissionError("Synthetic publication is limited to the OS temp tree")
    else:
        repo = Path(__file__).resolve().parents[3]
        expected_parent = repo / (PROTECTED_DIRECTORY_NAME if kind == "protected" else DEVELOPMENT_DIRECTORY_NAME)
        if absolute != expected_parent / "set_c_v1":
            raise PermissionError("Production output must use the approved D-038 path")
    if not absolute.parent.is_dir():
        raise FileNotFoundError("Publication parent must already exist")
    stage_prefix = ".set-c-protected-" if kind == "protected" else ".set-c-development-"
    if any(absolute.parent.glob(f"{stage_prefix}*")):
        raise FileExistsError("Interrupted staging artifact exists; manual recovery is required")
    for part in (absolute.parent, *absolute.parent.parents):
        if _is_reparse_point(part):
            raise PermissionError("Publication path contains a reparse point")
    absolute = absolute.parent.resolve(strict=True) / absolute.name
    if synthetic:
        temp_root = Path(tempfile.gettempdir()).resolve()
        if not absolute.is_relative_to(temp_root) or absolute == temp_root:
            raise PermissionError("Synthetic publication is limited to the OS temp tree")
    if absolute.exists():
        raise FileExistsError("Refusing to overwrite an existing publication")
    if sys.platform != "win32":
        raise OSError("Atomic no-replace directory publication is supported only on Windows")
    return absolute


def _atomic_publish_directory(stage: Path, target: Path) -> None:
    """Publish a closed same-volume directory with Windows no-replace rename."""
    if target.exists():
        raise FileExistsError("Refusing to overwrite an existing publication")
    os.rename(stage, target)


def _canonical_rows(source_csv: Path, *, allow_approved: bool = False) -> tuple[pd.DataFrame, str]:
    canonical_path = (
        Path(__file__).resolve().parents[3] / "Thesis_RF" / "Code" / "output" /
        "datasets" / "authoritative_set_c" / "v1" / "set_c.v1.csv"
    ).resolve()
    source_csv = source_csv.resolve(strict=True)
    if source_csv == canonical_path and not allow_approved:
        raise PermissionError("Reading the accepted Set C package is disabled in source-only mode")
    digest = _sha256(source_csv)
    if allow_approved and digest != APPROVED_CSV_SHA256:
        raise ValueError("Production CSV does not match the accepted Set C identity")
    if digest == APPROVED_CSV_SHA256 and not allow_approved:
        raise PermissionError("Production Set C source cannot be read by this source-only entry point")
    frame = pd.read_csv(source_csv)
    if tuple(frame.columns) != AUTHORITATIVE_CSV_COLUMNS:
        raise ValueError("Source CSV does not match the authoritative Set C schema")
    if frame.empty or frame.isna().any().any():
        raise ValueError("Source CSV is empty or has missing values")
    if frame.loc[:, list(AUTHORITATIVE_SPLIT_ASSIGNMENT_KEY_NAMES)].astype(str).apply(
        lambda column: column.str.strip().eq("")
    ).any().any():
        raise ValueError("Source CSV has empty observation identity fields")
    if frame.duplicated(list(AUTHORITATIVE_SPLIT_ASSIGNMENT_KEY_NAMES)).any():
        raise ValueError("Source CSV has duplicate observation identities")
    if tuple(frame["feature_schema_version"].unique()) != (SET_C_SCHEMA_VERSION,):
        raise ValueError("Source CSV has an unsupported feature schema")
    if tuple(frame["target_version"].unique()) != (TARGET_VERSION,):
        raise ValueError("Source CSV has an unsupported target schema")
    labels = frame[TARGET_NAME]
    if (not pd.api.types.is_integer_dtype(labels.dtype) or not labels.isin((0, 1)).all()):
        raise ValueError("Source CSV target must contain integer labels 0/1")
    validate_feature_values(frame.loc[:, list(CANONICAL_FEATURE_NAMES)].to_numpy(dtype="float32"))
    return frame, digest


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _graph_inventory(
    frame: pd.DataFrame,
    *,
    complete_event_ids: tuple[str, ...] = (),
    synthetic: bool,
) -> tuple[list[str], pd.DataFrame]:
    """Rebuild the versioned leakage graph and summarize its components."""
    graph_ids = build_joint_leakage_components(frame, complete_event_ids=complete_event_ids)
    component_ids = [f"synthetic-{value}" if synthetic else value for value in graph_ids.astype(str)]
    inventory_rows = []
    component_series = pd.Series(component_ids, index=frame.index)
    for component_id in sorted(set(component_ids)):
        rows = frame.loc[component_series.eq(component_id)]
        material_counts = rows["material_class"].astype(int).value_counts()
        inventory_rows.append({
            "component_id": component_id,
            "positives": int(rows[TARGET_NAME].sum()),
            "negatives": int(len(rows) - rows[TARGET_NAME].sum()),
            **{column: int(material_counts.get(index, 0)) for index, column in enumerate(MATERIAL_COLUMNS, 1)},
        })
    inventory = pd.DataFrame(
        inventory_rows,
        columns=("component_id", "positives", "negatives", *MATERIAL_COLUMNS),
    )
    return component_ids, inventory


def _bind_synthetic_provenance(
    source_manifest: dict | None,
    configuration: dict | None,
    source_revision: dict | None,
    authorization: dict | None,
    *,
    source_digest: str,
    row_count: int,
) -> dict:
    if not all(isinstance(item, dict) for item in (
        source_manifest, configuration, source_revision, authorization
    )):
        raise ValueError("Versioned synthetic provenance records are required")
    if set(source_manifest) != {
        "schema_version", "manifest_id", "source_csv_sha256", "row_count", "columns_sha256"
    } or source_manifest["schema_version"] != SOURCE_MANIFEST_VERSION:
        raise ValueError("Synthetic source manifest schema is invalid")
    if (not str(source_manifest["manifest_id"]).startswith("synthetic-")
            or source_manifest["source_csv_sha256"] != source_digest
            or source_manifest["row_count"] != row_count
            or source_manifest["columns_sha256"] != _canonical_json_sha256(list(AUTHORITATIVE_CSV_COLUMNS))):
        raise ValueError("Synthetic source manifest does not bind this CSV")
    if set(configuration) != {
        "schema_version", "graph_version", "policy_version", "dataset_sha256", "complete_event_ids"
    } or configuration["schema_version"] != ALLOCATION_CONFIG_VERSION:
        raise ValueError("Synthetic allocation configuration schema is invalid")
    if (configuration["graph_version"] != LEAKAGE_GRAPH_VERSION
            or configuration["policy_version"] != APPROVED_POLICY_VERSION
            or configuration["dataset_sha256"] != "0" * 64
            or configuration["complete_event_ids"] != []):
        raise ValueError("Synthetic configuration has an unapproved identity or event claim")
    if set(source_revision) != {"schema_version", "revision"} or source_revision["schema_version"] != SOURCE_REVISION_VERSION:
        raise ValueError("Synthetic source revision schema is invalid")
    revision = source_revision["revision"]
    revision_digest = revision.removeprefix("synthetic:") if isinstance(revision, str) else ""
    if len(revision_digest) != 40 or any(character not in "0123456789abcdef" for character in revision_digest):
        raise ValueError("Synthetic source revision identity is invalid")
    if not revision.startswith("synthetic:"):
        raise ValueError("Synthetic source revision identity is invalid")
    if set(authorization) != {"schema_version", "authorization_id", "scope", "production_enabled"}:
        raise ValueError("Synthetic authorization schema is invalid")
    if (authorization["schema_version"] != AUTHORIZATION_VERSION
            or authorization["authorization_id"] != "synthetic-only"
            or authorization["scope"] != "temporary synthetic allocation/export tests"
            or authorization["production_enabled"] is not False):
        raise PermissionError("Authorization record does not limit this operation to synthetic tests")
    records = {
        "source_manifest": source_manifest,
        "configuration": configuration,
        "source_revision": source_revision,
        "authorization": authorization,
    }
    return {
        **records,
        "hashes": {f"{key}_sha256": _canonical_json_sha256(value) for key, value in records.items()},
    }


def _validate_bound_provenance(provenance: dict) -> None:
    keys = ("source_manifest", "configuration", "source_revision", "authorization")
    if (not isinstance(provenance, dict) or set(provenance) != {*keys, "hashes"}
            or not isinstance(provenance.get("hashes"), dict)
            or any(not isinstance(provenance.get(key), dict) for key in keys)
            or set(provenance["hashes"]) != {
        f"{key}_sha256" for key in keys
    }):
        raise ValueError("Attestation provenance inventory is invalid")
    for key in keys:
        if provenance["hashes"][f"{key}_sha256"] != _canonical_json_sha256(provenance[key]):
            raise ValueError(f"Attestation {key} hash mismatch")
    manifest = provenance["source_manifest"]
    configuration = provenance["configuration"]
    revision = provenance["source_revision"]
    authorization = provenance["authorization"]
    synthetic = authorization.get("production_enabled") is False
    source_manifest_fields = (
        {"schema_version", "manifest_id", "source_csv_sha256", "row_count", "columns_sha256"}
        if synthetic else {
            "schema_version", "manifest_sha256", "manifest_payload_sha256",
            "source_csv_sha256", "row_count", "columns_sha256",
        }
    )
    revision_value = revision.get("revision")
    revision_hex = revision_value.removeprefix("synthetic:") if isinstance(revision_value, str) else ""
    expected_authorization = (
        {"schema_version", "authorization_id", "scope", "production_enabled"}
        if synthetic else {"schema_version", "authorization_reference", "scope", "production_enabled"}
    )
    valid_manifest_identity = (
        str(manifest.get("manifest_id", "")).startswith("synthetic-")
        if synthetic else (
            manifest.get("manifest_sha256") == APPROVED_MANIFEST_SHA256
            and manifest.get("source_csv_sha256") == APPROVED_CSV_SHA256
            and _is_sha256(manifest.get("manifest_payload_sha256"))
        )
    )
    valid_revision = isinstance(revision_value, str) and (
        revision_value.startswith("synthetic:") if synthetic else revision_value == revision_hex
    )
    valid_authorization = (
        authorization.get("authorization_id") == "synthetic-only"
        and authorization.get("scope") == "temporary synthetic allocation/export tests"
        and authorization.get("production_enabled") is False
        if synthetic else (
            isinstance(authorization.get("authorization_reference"), str)
            and bool(authorization["authorization_reference"].strip())
            and authorization.get("scope") == "one Set C allocation/export"
            and authorization.get("production_enabled") is True
        )
    )
    if (set(manifest) != source_manifest_fields
            or manifest.get("schema_version") != SOURCE_MANIFEST_VERSION
            or not valid_manifest_identity
            or type(manifest.get("row_count")) is not int
            or manifest["row_count"] < 1
            or not _is_sha256(manifest.get("source_csv_sha256"))
            or not _is_sha256(manifest.get("columns_sha256"))
            or set(configuration) != {
                "schema_version", "graph_version", "policy_version", "dataset_sha256", "complete_event_ids"
            }
            or configuration.get("schema_version") != ALLOCATION_CONFIG_VERSION
            or configuration.get("graph_version") != LEAKAGE_GRAPH_VERSION
            or configuration.get("policy_version") != APPROVED_POLICY_VERSION
            or configuration.get("dataset_sha256") != ("0" * 64 if synthetic else APPROVED_CSV_SHA256)
            or configuration.get("complete_event_ids") != []
            or set(revision) != {"schema_version", "revision"}
            or revision.get("schema_version") != SOURCE_REVISION_VERSION
            or not valid_revision or len(revision_hex) != 40
            or any(character not in "0123456789abcdef" for character in revision_hex)
            or set(authorization) != expected_authorization
            or authorization.get("schema_version") != AUTHORIZATION_VERSION
            or not valid_authorization):
        raise ValueError("Attestation provenance contains an unsupported source identity")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _synthetic_test_provenance(source_digest: str, row_count: int) -> dict:
    return _bind_synthetic_provenance(
        {
            "schema_version": SOURCE_MANIFEST_VERSION,
            "manifest_id": "synthetic-release-test",
            "source_csv_sha256": source_digest,
            "row_count": row_count,
            "columns_sha256": _canonical_json_sha256(list(AUTHORITATIVE_CSV_COLUMNS)),
        },
        {
            "schema_version": ALLOCATION_CONFIG_VERSION,
            "graph_version": LEAKAGE_GRAPH_VERSION,
            "policy_version": APPROVED_POLICY_VERSION,
            "dataset_sha256": "0" * 64,
            "complete_event_ids": [],
        },
        {"schema_version": SOURCE_REVISION_VERSION, "revision": "synthetic:" + "d" * 40},
        {
            "schema_version": AUTHORIZATION_VERSION,
            "authorization_id": "synthetic-only",
            "scope": "temporary synthetic allocation/export tests",
            "production_enabled": False,
        },
        source_digest=source_digest,
        row_count=row_count,
    )


def _production_provenance(
    manifest: dict,
    *,
    manifest_digest: str,
    source_digest: str,
    row_count: int,
    source_revision: str,
    authorization_reference: str,
) -> dict:
    """Bind the future production release to the accepted package and one-run record."""
    if (manifest_digest != APPROVED_MANIFEST_SHA256
            or source_digest != APPROVED_CSV_SHA256
            or manifest.get("manifest_schema_version") != PACKAGE_SCHEMA_VERSION
            or manifest.get("artifact_kind") != "authoritative_set_c_observation_package"
            or manifest.get("split_status") != "unassigned"
            or manifest.get("dataset_sha256") != APPROVED_CSV_SHA256
            or manifest.get("row_count") != row_count
            or manifest.get("canonical_csv_columns") != list(AUTHORITATIVE_CSV_COLUMNS)
            or not _is_sha256(manifest.get("manifest_payload_sha256"))):
        raise ValueError("Production source package does not match the approved Set C identity")
    if (not isinstance(source_revision, str) or len(source_revision) != 40
            or any(character not in "0123456789abcdef" for character in source_revision)
            or not isinstance(authorization_reference, str) or not authorization_reference.strip()):
        raise PermissionError("A pinned source revision and one-run authorization reference are required")
    records = {
        "source_manifest": {
            "schema_version": SOURCE_MANIFEST_VERSION,
            "manifest_sha256": manifest_digest,
            "manifest_payload_sha256": manifest["manifest_payload_sha256"],
            "source_csv_sha256": source_digest,
            "row_count": row_count,
            "columns_sha256": _canonical_json_sha256(list(AUTHORITATIVE_CSV_COLUMNS)),
        },
        "configuration": {
            "schema_version": ALLOCATION_CONFIG_VERSION,
            "graph_version": LEAKAGE_GRAPH_VERSION,
            "policy_version": APPROVED_POLICY_VERSION,
            "dataset_sha256": source_digest,
            "complete_event_ids": [],
        },
        "source_revision": {"schema_version": SOURCE_REVISION_VERSION, "revision": source_revision},
        "authorization": {
            "schema_version": AUTHORIZATION_VERSION,
            "authorization_reference": authorization_reference.strip(),
            "scope": "one Set C allocation/export",
            "production_enabled": True,
        },
    }
    provenance = {
        **records,
        "hashes": {f"{key}_sha256": _canonical_json_sha256(value) for key, value in records.items()},
    }
    _validate_bound_provenance(provenance)
    return provenance


def export_set_c_split(
    source_csv: str | Path,
    *,
    protected_destination: str | Path,
    development_destination: str | Path,
    synthetic: bool = False,
    source_manifest: dict | None = None,
    configuration: dict | None = None,
    source_revision: dict | None = None,
    authorization: dict | None = None,
    production_revision: str | None = None,
    authorization_reference: str | None = None,
    operator_opt_in: bool = False,
) -> tuple[Path, Path]:
    """Derive and publish D-035 roles; production requires per-run opt-in and audit reference."""
    if not synthetic:
        if not operator_opt_in:
            raise PermissionError("Production Set C allocation requires explicit operator opt-in")
        if not isinstance(authorization_reference, str) or not authorization_reference.strip():
            raise PermissionError("A nonempty separate owner approval reference is required")
    source_path = Path(source_csv)
    protected_target = _validate_output_path(Path(protected_destination), synthetic=synthetic, kind="protected")
    development_target = _validate_output_path(Path(development_destination), synthetic=synthetic, kind="development")
    if protected_target == development_target or protected_target in development_target.parents or development_target in protected_target.parents:
        raise ValueError("Protected and development publications must be separate directories")
    if synthetic:
        source_path = source_path.resolve(strict=True)
        if not source_path.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            raise PermissionError("Synthetic source CSV must be inside the OS temp tree")
        source_frame, source_digest = _canonical_rows(source_path)
        provenance = _bind_synthetic_provenance(
            source_manifest, configuration, source_revision, authorization,
            source_digest=source_digest,
            row_count=len(source_frame),
        )
        complete_event_ids: tuple[str, ...] = tuple(configuration["complete_event_ids"])
    else:
        repo = Path(__file__).resolve().parents[3]
        expected_source = repo / "Thesis_RF" / "Code" / "output" / "datasets" / "authoritative_set_c" / "v1" / "set_c.v1.csv"
        if source_path.absolute() != expected_source:
            raise PermissionError("Production source must be the approved Set C v1 package")
        _verify_clean_source_revision(repo, production_revision or "")
        manifest_path = source_path.parent / "manifest.json"
        if any(_is_reparse_point(part) for path in (source_path, manifest_path) for part in (path, *path.parents)):
            raise PermissionError("Production package contains a reparse-point path")
        manifest_digest = _sha256(manifest_path)
        if manifest_digest != APPROVED_MANIFEST_SHA256:
            raise ValueError("Production manifest does not match the accepted Set C identity")
        manifest = validate_published_package(manifest_path)
        source_frame, source_digest = _canonical_rows(source_path, allow_approved=True)
        if (manifest.get("dataset_path") != source_path.name
                or manifest.get("dataset_sha256") != source_digest):
            raise ValueError("Production manifest does not bind the accepted CSV")
        provenance = _production_provenance(
            manifest, manifest_digest=manifest_digest, source_digest=source_digest,
            row_count=len(source_frame), source_revision=production_revision or "",
            authorization_reference=authorization_reference or "",
        )
        complete_event_ids = ()
    component_ids, inventory = _graph_inventory(
        source_frame, complete_event_ids=complete_event_ids, synthetic=synthetic,
    )
    if len(inventory) != 26:
        raise ValueError("Set C leakage graph must contain exactly 26 components")
    assignment_identity = configuration["dataset_sha256"] if synthetic else source_digest
    assignments = _allocate_components(inventory, dataset_sha256=assignment_identity)
    component_set = set(component_ids)
    if set(assignments) != component_set or set(assignments.values()) - set(FOUR_ROLES):
        raise ValueError("Derived assignments must cover every component exactly once")
    if set(assignments.values()) != set(FOUR_ROLES):
        raise ValueError("Derived assignments must include all four roles")
    if [list(assignments.values()).count(role) for role in FOUR_ROLES] != list(ROLE_COMPONENT_TARGETS):
        raise ValueError("Derived assignments violate the D-035 11/5/5/5 component targets")
    labels = source_frame[TARGET_NAME]
    if (labels.isna().any() or not pd.api.types.is_integer_dtype(labels.dtype)
            or not labels.isin((0, 1)).all()):
        raise ValueError("Source target must contain only integer labels 0/1")
    validate_feature_values(source_frame.loc[:, list(CANONICAL_FEATURE_NAMES)].to_numpy(dtype="float32"))
    protected_stage = None
    development_stage = None
    failure_stage = "protected_staging"
    try:
        protected_stage = Path(tempfile.mkdtemp(prefix=".set-c-protected-", dir=protected_target.parent))
        failure_stage = "development_staging"
        development_stage = Path(tempfile.mkdtemp(prefix=".set-c-development-", dir=development_target.parent))
        failure_stage = "protected_bundle"
        shutil.copyfile(source_path, protected_stage / "canonical.csv")
        row_roles = pd.Series(component_ids, index=source_frame.index).map(assignments)
        component_series = pd.Series(component_ids, index=source_frame.index)
        split_manifest = source_frame.loc[:, list(AUTHORITATIVE_SPLIT_ASSIGNMENT_KEY_NAMES)].copy()
        split_manifest["leakage_component_id"] = component_series
        split_manifest["split_role"] = row_roles
        split_manifest_path = protected_stage / "split_manifest.csv"
        split_manifest.to_csv(split_manifest_path, index=False)
        for index, role in enumerate(FOUR_ROLES):
            role_rows = source_frame.loc[row_roles.eq(role)]
            role_labels = role_rows[TARGET_NAME].astype(int)
            positives = int(role_labels.sum())
            negatives = len(role_labels) - positives
            if positives < ROLE_POSITIVE_MINIMA[index] or negatives < ROLE_NEGATIVE_MINIMA[index]:
                raise ValueError(f"{role} rows violate D-015 positive/negative minima")
            if not {1, 2, 3, 4, 5}.issubset(set(role_rows["material_class"].astype(int))):
                raise ValueError(f"{role} rows do not cover material classes 1–5")
            component_positives = role_labels.groupby(component_series.loc[role_rows.index]).sum()
            if (component_positives * 2 > positives).any():
                raise ValueError(f"A {role} component exceeds 50% of its positives")
        final_rows = source_frame.loc[row_roles.eq("final_test")].copy()
        if final_rows.empty or len(final_rows) == len(source_frame):
            raise ValueError("Synthetic fixture must contain both development and final-test rows")
        final_rows["leakage_component_id"] = component_series.loc[final_rows.index]
        final_path = protected_stage / "final_test_rows.csv"
        final_rows.to_csv(final_path, index=False)
        protected_inventory = {
            "schema_version": PROTECTED_ATTESTATION_SCHEMA_VERSION,
            "package_schema_version": PROTECTED_SCHEMA_VERSION,
            "mapping_schema_version": MAPPING_SCHEMA_VERSION,
            "final_test_schema_version": FINAL_TEST_SCHEMA_VERSION,
            "policy_version": APPROVED_POLICY_VERSION,
            "feature_schema_version": SET_C_SCHEMA_VERSION,
            "target_version": TARGET_VERSION,
            "source_csv_sha256": source_digest,
            "split_manifest_sha256": _sha256(split_manifest_path),
            "final_test_rows_sha256": _sha256(final_path),
            "source_rows": len(source_frame),
            "component_count": len(assignments),
            "final_test_rows": len(final_rows),
            "role_counts": {
                role: {
                    "components": sum(value == role for value in assignments.values()),
                    "rows": int(row_roles.eq(role).sum()),
                    "positives": int(source_frame.loc[row_roles.eq(role), TARGET_NAME].sum()),
                    "negatives": int((row_roles.eq(role).sum()) - source_frame.loc[row_roles.eq(role), TARGET_NAME].sum()),
                }
                for role in FOUR_ROLES
            },
            "source_provenance": provenance,
            "synthetic": synthetic,
            # This record is never edited in place. A matching development receipt
            # is the only proof that the two-directory publication completed.
            "publication_state": "protected_only_incomplete_until_paired_release",
        }
        protected_attestation = protected_stage / "attestation.json"
        with protected_attestation.open("x", encoding="utf-8") as stream:
            json.dump(protected_inventory, stream, sort_keys=True, separators=(",", ":"))

        failure_stage = "development_bundle"
        role_files = {}
        for role in DEVELOPMENT_SPLIT_ROLES:
            role_rows = source_frame.loc[row_roles.eq(role), list(RELEASE_COLUMNS)]
            if role_rows.empty:
                raise ValueError(f"Synthetic assignment has no {role} rows")
            role_path = development_stage / f"{role}.csv"
            role_rows.to_csv(role_path, index=False)
            positives = int(role_rows[TARGET_NAME].sum())
            role_files[role] = {
                "sha256": _sha256(role_path), "rows": len(role_rows),
                "positives": positives, "negatives": len(role_rows) - positives,
            }
        development_attestation = {
            "schema_version": ATTESTATION_SCHEMA_VERSION,
            "release_schema_version": RELEASE_SCHEMA_VERSION,
            "source_csv_sha256": source_digest,
            "policy_version": APPROVED_POLICY_VERSION,
            "feature_schema_version": SET_C_SCHEMA_VERSION,
            "target_version": TARGET_VERSION,
            "feature_names": list(CANONICAL_FEATURE_NAMES),
            "roles": list(DEVELOPMENT_SPLIT_ROLES),
            "role_files": role_files,
            "synthetic": synthetic,
            "final_test_used": False,
            "source_provenance": provenance,
        }

        failure_stage = "protected_validation"
        _validate_protected_stage(protected_stage, source_digest, source_frame)
        # Stage on each destination volume so publication is a same-volume rename.
        if protected_stage.stat().st_dev != protected_target.parent.stat().st_dev or development_stage.stat().st_dev != development_target.parent.stat().st_dev:
            raise OSError("Staging and destination must be on the same volume")
        failure_stage = "protected_publication"
        _atomic_publish_directory(protected_stage, protected_target)
        try:
            failure_stage = "pair_receipt"
            protected_attestation_path = protected_target / "attestation.json"
            development_attestation["publication_receipt"] = {
                "schema_version": PAIR_PUBLICATION_SCHEMA_VERSION,
                "state": "complete",
                "protected_attestation_sha256": _sha256(protected_attestation_path),
                "source_csv_sha256": source_digest,
                "policy_version": APPROVED_POLICY_VERSION,
                "source_revision": provenance["source_revision"]["revision"],
            }
            dev_attestation_path = development_stage / "attestation.json"
            with dev_attestation_path.open("x", encoding="utf-8") as stream:
                json.dump(development_attestation, stream, sort_keys=True, separators=(",", ":"))
            dev_hash = _sha256(dev_attestation_path)
            failure_stage = "development_validation"
            _load_development_release(
                development_stage,
                expected_attestation_sha256=dev_hash,
                allow_synthetic=True,
                allow_staging=True,
            )
            failure_stage = "pair_validation"
            _verify_publication_pair(
                protected_target,
                development_stage,
                expected_development_attestation_sha256=dev_hash,
                allow_synthetic=synthetic,
                allow_development_staging=True,
            )
            # Final fallible operation: only publish after both staged attestations
            # and protected artifact hashes verify. This no-replace rename is the
            # paired-publication completion signal.
            failure_stage = "development_publication"
            _atomic_publish_directory(development_stage, development_target)
        except Exception as error:
            _attach_failure_context(error, {"stage": failure_stage})
            failure = RuntimeError(
                "Development publication failed; protected publication remains incomplete and requires human recovery"
            )
            _attach_failure_context(failure, {"stage": failure_stage})
            raise failure from error
        return protected_target, development_target
    except Exception as error:
        _attach_failure_context(error, {"stage": failure_stage})
        if protected_stage is not None and protected_stage.exists():
            shutil.rmtree(protected_stage)
        if development_stage is not None and development_stage.exists():
            shutil.rmtree(development_stage)
        raise


def _validate_protected_stage(
    stage: Path, source_digest: str, source_frame: pd.DataFrame
) -> None:
    expected = {"canonical.csv", "split_manifest.csv", "final_test_rows.csv", "attestation.json"}
    if {item.name for item in stage.iterdir()} != expected:
        raise ValueError("Protected package inventory is incomplete")
    record = json.loads((stage / "attestation.json").read_text(encoding="utf-8"))
    if set(record) != {
        "schema_version", "package_schema_version", "mapping_schema_version", "final_test_schema_version",
        "policy_version", "feature_schema_version", "target_version", "role_counts",
        "source_csv_sha256", "split_manifest_sha256", "final_test_rows_sha256", "source_provenance",
        "source_rows", "component_count", "final_test_rows", "synthetic",
        "publication_state",
    }:
        raise ValueError("Protected attestation has an unexpected schema")
    if (record["schema_version"] != PROTECTED_ATTESTATION_SCHEMA_VERSION
            or record["package_schema_version"] != PROTECTED_SCHEMA_VERSION
            or record["mapping_schema_version"] != MAPPING_SCHEMA_VERSION
            or record["final_test_schema_version"] != FINAL_TEST_SCHEMA_VERSION
            or record["policy_version"] != APPROVED_POLICY_VERSION
            or record["feature_schema_version"] != SET_C_SCHEMA_VERSION
            or record["target_version"] != TARGET_VERSION
            or record["source_csv_sha256"] != source_digest
            or _sha256(stage / "canonical.csv") != source_digest
            or _sha256(stage / "split_manifest.csv") != record["split_manifest_sha256"]
            or _sha256(stage / "final_test_rows.csv") != record["final_test_rows_sha256"]
            or type(record["synthetic"]) is not bool
            or record["publication_state"] != "protected_only_incomplete_until_paired_release"
            or set(record["role_counts"]) != set(FOUR_ROLES)):
        raise ValueError("Protected package provenance or hashes do not verify")
    _validate_bound_provenance(record["source_provenance"])
    if record["source_provenance"]["source_manifest"]["source_csv_sha256"] != source_digest:
        raise ValueError("Protected source manifest does not match the canonical copy")
    mapping = pd.read_csv(stage / "split_manifest.csv", dtype={key: str for key in AUTHORITATIVE_SPLIT_ASSIGNMENT_KEY_NAMES})
    if (tuple(mapping.columns) != SPLIT_MANIFEST_COLUMNS
            or mapping.loc[:, list(AUTHORITATIVE_SPLIT_ASSIGNMENT_KEY_NAMES)].duplicated().any()
            or len(mapping) != len(source_frame)):
        raise ValueError("Protected row split manifest schema or identity inventory is invalid")
    source_keys = source_frame.loc[:, list(AUTHORITATIVE_SPLIT_ASSIGNMENT_KEY_NAMES)].astype(str).reset_index(drop=True)
    if not mapping.loc[:, list(AUTHORITATIVE_SPLIT_ASSIGNMENT_KEY_NAMES)].astype(str).reset_index(drop=True).equals(source_keys):
        raise ValueError("Protected split manifest does not cover canonical rows in order")
    expected_components, _ = _graph_inventory(
        source_frame, complete_event_ids=(), synthetic=record["synthetic"],
    )
    if mapping["leakage_component_id"].astype(str).tolist() != expected_components:
        raise ValueError("Protected split manifest component identities do not rebuild")
    if mapping.groupby("leakage_component_id")["split_role"].nunique().max() != 1:
        raise ValueError("Protected split manifest divides a leakage component")
    if set(mapping["split_role"]) != set(FOUR_ROLES):
        raise ValueError("Protected split manifest omits an allocation role")
    component_roles = mapping.loc[:, ["leakage_component_id", "split_role"]].drop_duplicates()
    if (len(component_roles) != record["component_count"]
            or component_roles["split_role"].value_counts().to_dict() != {
                role: record["role_counts"][role]["components"] for role in FOUR_ROLES
            }):
        raise ValueError("Protected component allocation totals disagree with attestation")
    row_roles = mapping["split_role"].astype(str)
    for role_index, role in enumerate(FOUR_ROLES):
        selected = source_frame.loc[row_roles.eq(role)]
        positives = int(selected[TARGET_NAME].sum())
        negatives = len(selected) - positives
        role_record = record["role_counts"][role]
        if ((len(selected), positives, negatives) != (
                role_record["rows"], role_record["positives"], role_record["negatives"]
        ) or positives < ROLE_POSITIVE_MINIMA[role_index]
                or negatives < ROLE_NEGATIVE_MINIMA[role_index]
                or not {1, 2, 3, 4, 5}.issubset(set(selected["material_class"].astype(int)))):
            raise ValueError("Protected split manifest violates role counts or D-015 minima")
        component_positive_counts = selected[TARGET_NAME].groupby(
            mapping.loc[selected.index, "leakage_component_id"]
        ).sum()
        if (component_positive_counts * 2 > positives).any():
            raise ValueError("Protected split manifest violates the D-035 concentration limit")
    final_mapping = mapping.loc[mapping["split_role"].eq("final_test")]
    expected_final = source_frame.loc[final_mapping.index].copy()
    expected_final["leakage_component_id"] = final_mapping["leakage_component_id"].to_numpy()
    final_rows = pd.read_csv(
        stage / "final_test_rows.csv",
        dtype=expected_final.dtypes.to_dict(),
        float_precision="round_trip",
    )
    if tuple(final_rows.columns) != (*AUTHORITATIVE_CSV_COLUMNS, "leakage_component_id"):
        raise ValueError("Protected final-test row schema is invalid")
    if not final_rows.reset_index(drop=True).equals(expected_final.reset_index(drop=True)):
        raise ValueError("Protected final-test rows disagree with the complete split manifest")
    if len(final_rows) != record["final_test_rows"] or len(final_rows) != record["role_counts"]["final_test"]["rows"]:
        raise ValueError("Protected final-test row totals disagree with attestation")


def _validate_component_inventory(inventory: pd.DataFrame) -> None:
    required = ("component_id", "positives", "negatives", *MATERIAL_COLUMNS)
    if tuple(inventory.columns) != required or len(inventory) != 26:
        raise ValueError("Allocation requires exactly 26 components and the versioned inventory schema")
    if inventory.isna().any().any() or inventory["component_id"].duplicated().any():
        raise ValueError("Component identities must be unique and complete")
    if any(not isinstance(value, str) or not value.strip() for value in inventory["component_id"]):
        raise ValueError("Component identities must be nonempty text")
    counts = inventory.loc[:, list(required[1:])]
    if any(not pd.api.types.is_integer_dtype(counts[column].dtype) for column in counts):
        raise ValueError("Component counts must be integers")
    if (counts < 0).any().any() or (counts["positives"] < 5).any() or (counts["negatives"] < 5).any():
        raise ValueError("Every component must meet the five-positive/five-negative minimum")


def _check_development_path(path: Path, *, allow_staging: bool = False) -> Path:
    """Reject canonical/vault paths and reparse points before reading a release."""
    absolute = path.absolute()
    if not allow_staging and any(
        part.lower().startswith(".set-c-development-") for part in absolute.parts
    ):
        raise PermissionError("Unpublished development staging is not a consumable release")
    if _FORBIDDEN_PATH_PARTS.intersection(part.lower() for part in absolute.parts):
        raise PermissionError("Development loader cannot read a canonical package or vault")
    for candidate in (absolute, absolute.resolve(strict=True)):
        if _FORBIDDEN_PATH_PARTS.intersection(part.lower() for part in candidate.parts):
            raise PermissionError("Development loader cannot read a canonical package or vault")
        for part in (candidate, *candidate.parents):
            attributes = os.lstat(part)
            if part.is_symlink() or (getattr(attributes, "st_file_attributes", 0) & 0x400):
                raise PermissionError("Development release path contains a reparse point")
    if not absolute.is_dir():
        raise ValueError("Development release must be a directory")
    return absolute


def verify_publication_pair(
    protected_dir: str | Path,
    development_dir: str | Path,
    *,
    expected_development_attestation_sha256: str,
    allow_synthetic: bool = False,
) -> None:
    """Verify only a final published pair; staging is private to the exporter."""
    _verify_publication_pair(
        protected_dir,
        development_dir,
        expected_development_attestation_sha256=expected_development_attestation_sha256,
        allow_synthetic=allow_synthetic,
    )


def _verify_publication_pair(
    protected_dir: str | Path,
    development_dir: str | Path,
    *,
    expected_development_attestation_sha256: str,
    allow_synthetic: bool = False,
    allow_development_staging: bool = False,
) -> None:
    """Verify the completion receipt using attestations, without opening protected rows."""
    protected = Path(protected_dir).absolute()
    development = Path(development_dir).absolute()
    if protected == development or protected in development.parents or development in protected.parents:
        raise ValueError("Protected and development bundles must be separate directories")
    if allow_synthetic:
        temp_root = Path(tempfile.gettempdir()).resolve()
        if not protected.resolve(strict=True).is_relative_to(temp_root) or not development.resolve(strict=True).is_relative_to(temp_root):
            raise PermissionError("Synthetic paired verification is limited to the OS temp tree")
    else:
        repository = Path(__file__).resolve().parents[3]
        approved_protected = repository / PROTECTED_DIRECTORY_NAME / "set_c_v1"
        approved_development = repository / DEVELOPMENT_DIRECTORY_NAME / "set_c_v1"
        development_stage = (
            allow_development_staging
            and
            development.parent == approved_development.parent
            and development.name.startswith(".set-c-development-")
        )
        if protected != approved_protected or (development != approved_development and not development_stage):
            raise PermissionError("Paired verification requires the approved D-038 paths")
    for root in (protected, development):
        for candidate in (root, *root.parents):
            attributes = os.lstat(candidate)
            if candidate.is_symlink() or getattr(attributes, "st_file_attributes", 0) & 0x400:
                raise PermissionError("Paired bundle path contains a reparse point")
    expected_protected_files = {
        "canonical.csv", "split_manifest.csv", "final_test_rows.csv", "attestation.json",
    }
    if {item.name for item in protected.iterdir()} != expected_protected_files:
        raise ValueError("Protected bundle inventory is incomplete")
    protected_attestation_path = protected / "attestation.json"
    protected_record = json.loads(protected_attestation_path.read_text(encoding="utf-8"))
    if (
        protected_record.get("schema_version") != PROTECTED_ATTESTATION_SCHEMA_VERSION
        or protected_record.get("package_schema_version") != PROTECTED_SCHEMA_VERSION
        or protected_record.get("mapping_schema_version") != MAPPING_SCHEMA_VERSION
        or protected_record.get("final_test_schema_version") != FINAL_TEST_SCHEMA_VERSION
        or protected_record.get("policy_version") != APPROVED_POLICY_VERSION
        or not _is_sha256(protected_record.get("source_csv_sha256"))
        or not _is_sha256(protected_record.get("split_manifest_sha256"))
        or not _is_sha256(protected_record.get("final_test_rows_sha256"))
        or _sha256(protected / "canonical.csv") != protected_record.get("source_csv_sha256")
        or _sha256(protected / "split_manifest.csv") != protected_record.get("split_manifest_sha256")
        or _sha256(protected / "final_test_rows.csv") != protected_record.get("final_test_rows_sha256")
    ):
        raise ValueError("Protected artifacts do not match their versioned attestation hashes")
    _validate_bound_provenance(protected_record["source_provenance"])
    development_data, development_record = _load_development_release(
        development,
        expected_attestation_sha256=expected_development_attestation_sha256,
        allow_synthetic=allow_synthetic,
        allow_staging=allow_development_staging,
    )
    del development_data
    receipt = development_record["publication_receipt"]
    if (
        protected_record.get("publication_state") != "protected_only_incomplete_until_paired_release"
        or receipt["protected_attestation_sha256"] != _sha256(protected_attestation_path)
        or protected_record.get("source_csv_sha256") != development_record.get("source_csv_sha256")
        or protected_record.get("policy_version") != development_record.get("policy_version")
        or protected_record.get("source_provenance") != development_record.get("source_provenance")
    ):
        raise ValueError("Protected and development attestations do not prove the same completed publication")


def load_development_release(
    release_dir: str | Path,
    *,
    expected_attestation_sha256: str,
    allow_synthetic: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Load only the three projected roles, bound to a pinned attestation hash."""
    return _load_development_release(
        release_dir,
        expected_attestation_sha256=expected_attestation_sha256,
        allow_synthetic=allow_synthetic,
    )


def load_development_calibration_release(
    release_dir: str | Path,
    *,
    expected_attestation_sha256: str = APPROVED_CALIBRATION_ATTESTATION_SHA256,
    allow_synthetic: bool = False,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load only the attested calibration role, without opening other role contents."""
    supplied = Path(release_dir)
    if ".." in supplied.parts:
        raise PermissionError("Development release paths cannot contain parent traversal")
    root = supplied.absolute()
    if any(part.lower() in {"canonical", "canonical.csv"} for part in root.parts):
        raise PermissionError("Calibration loader cannot read a canonical package")
    if allow_synthetic:
        temp_root = Path(tempfile.gettempdir()).resolve()
        resolved = root.resolve(strict=True)
        if resolved == temp_root or not resolved.is_relative_to(temp_root):
            raise PermissionError("Synthetic calibration releases are limited to the OS temp tree")
    else:
        approved = Path(__file__).resolve().parents[3] / DEVELOPMENT_DIRECTORY_NAME / "set_c_v1"
        if root != approved:
            raise PermissionError("Calibration loader accepts only the approved D-038 development release")
        if expected_attestation_sha256 != APPROVED_CALIBRATION_ATTESTATION_SHA256:
            raise PermissionError("Calibration loader requires the accepted development attestation")

    data, attestation = _load_development_release(
        root,
        expected_attestation_sha256=expected_attestation_sha256,
        allow_synthetic=allow_synthetic,
        selected_roles=("calibration",),
    )
    if allow_synthetic and attestation["synthetic"] is not True:
        raise PermissionError("Synthetic calibration fixtures must carry explicit synthetic provenance")

    record = attestation["role_files"]["calibration"]
    if not attestation["synthetic"] and (
        record["sha256"] != APPROVED_CALIBRATION_CSV_SHA256
        or (record["rows"], record["positives"], record["negatives"])
        != APPROVED_CALIBRATION_COUNTS
    ):
        raise ValueError("Calibration role identity or accepted counts do not match D-053/D-054")

    frame = data.loc[:, list(RELEASE_COLUMNS)]
    features = frame.loc[:, list(CANONICAL_FEATURE_NAMES)].astype(np.float32)
    labels = frame[TARGET_NAME].copy()
    if tuple(features.columns) != CANONICAL_FEATURE_NAMES or features.dtypes.ne(np.dtype("float32")).any():
        raise ValueError("Calibration features do not match the ordered float32 schema")
    if labels.dtype != np.dtype("int8"):
        raise ValueError("Calibration labels must use int8")
    provenance = {
        "attestation_sha256": expected_attestation_sha256,
        "calibration_csv_sha256": record["sha256"],
        "rows": record["rows"],
        "positives": record["positives"],
        "negatives": record["negatives"],
        "source_csv_sha256": attestation["source_csv_sha256"],
        "source_revision": attestation["source_provenance"]["source_revision"]["revision"],
        "synthetic": attestation["synthetic"],
    }
    return features, labels, provenance


def load_development_train_validation_release(
    release_dir: str | Path,
    *,
    expected_attestation_sha256: str,
    allow_synthetic: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Verify and load only the pinned train and validation role files."""
    supplied = Path(release_dir)
    if ".." in supplied.parts:
        raise PermissionError("Development release paths cannot contain parent traversal")
    root = supplied.absolute()
    if allow_synthetic:
        temp_root = Path(tempfile.gettempdir()).resolve()
        if not root.resolve(strict=True).is_relative_to(temp_root) or root.resolve() == temp_root:
            raise PermissionError("Synthetic training releases are limited to the OS temp tree")
    else:
        approved = Path(__file__).resolve().parents[3] / DEVELOPMENT_DIRECTORY_NAME / "set_c_v1"
        if root != approved:
            raise PermissionError("Training loader accepts only the approved development release")
    data, attestation = _load_development_release(
        root,
        expected_attestation_sha256=expected_attestation_sha256,
        allow_synthetic=allow_synthetic,
        selected_roles=("train", "validation"),
    )
    roles = {
        role: data.loc[data["split_role"].eq(role), list(RELEASE_COLUMNS)].reset_index(drop=True)
        for role in ("train", "validation")
    }
    dtype_map = {**dict.fromkeys(CANONICAL_FEATURE_NAMES, np.float32), TARGET_NAME: np.int8}
    roles = {role: frame.astype(dtype_map) for role, frame in roles.items()}
    return roles, attestation


def _load_development_release(
    release_dir: str | Path,
    *,
    expected_attestation_sha256: str,
    allow_synthetic: bool = False,
    allow_staging: bool = False,
    selected_roles: tuple[str, ...] = DEVELOPMENT_SPLIT_ROLES,
) -> tuple[pd.DataFrame, dict]:
    """Private validator may inspect a pre-rename stage during exporter preflight."""
    if len(expected_attestation_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_attestation_sha256
    ):
        raise ValueError("A pinned lowercase attestation SHA-256 is required")
    root = _check_development_path(Path(release_dir), allow_staging=allow_staging)
    expected_files = {"attestation.json", *(f"{role}.csv" for role in DEVELOPMENT_SPLIT_ROLES)}
    if {item.name for item in root.iterdir()} != expected_files:
        raise PermissionError("Development release contains missing or unexpected files")
    entries = {item.name: item for item in root.iterdir()}
    if any(item.is_symlink() or (getattr(os.lstat(item), "st_file_attributes", 0) & 0x400)
           or not item.is_file() for item in entries.values()):
        raise PermissionError("Development release contains a reparse point")
    attestation_path = root / "attestation.json"
    if _sha256(attestation_path) != expected_attestation_sha256:
        raise ValueError("Development attestation hash mismatch")
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    current_fields = {
        "schema_version", "release_schema_version", "source_csv_sha256", "policy_version",
        "feature_schema_version", "target_version", "feature_names", "roles",
        "role_files", "synthetic", "final_test_used", "source_provenance", "publication_receipt",
    }
    if set(attestation) != current_fields:
        raise ValueError("Development attestation fields are incomplete or unexpected")
    if (
        attestation["schema_version"] != ATTESTATION_SCHEMA_VERSION
        or attestation["release_schema_version"] != RELEASE_SCHEMA_VERSION
        or attestation["feature_schema_version"] != SET_C_SCHEMA_VERSION
        or attestation["target_version"] != TARGET_VERSION
        or tuple(attestation["feature_names"]) != CANONICAL_FEATURE_NAMES
        or tuple(attestation["roles"]) != DEVELOPMENT_SPLIT_ROLES
        or attestation["final_test_used"] is not False
        or type(attestation["synthetic"]) is not bool
    ):
        raise ValueError("Development release violates its versioned schema")
    if attestation["synthetic"]:
        if not allow_synthetic:
            raise PermissionError("Synthetic release is forbidden in production loading")
    elif attestation["source_csv_sha256"] != APPROVED_CSV_SHA256:
        raise ValueError("Development release is not bound to approved Set C v1")
    if not attestation["synthetic"] and attestation["policy_version"] != APPROVED_POLICY_VERSION:
        raise ValueError("Development release has an unapproved allocation policy")
    if not isinstance(attestation["policy_version"], str) or not attestation["policy_version"].strip():
        raise ValueError("Development release lacks a policy version")
    _validate_bound_provenance(attestation["source_provenance"])
    if attestation["source_provenance"]["source_manifest"]["source_csv_sha256"] != attestation["source_csv_sha256"]:
        raise ValueError("Development source manifest hash does not match its attestation")
    receipt = attestation.get("publication_receipt")
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {
            "schema_version", "state", "protected_attestation_sha256",
            "source_csv_sha256", "policy_version", "source_revision",
        }
        or receipt.get("schema_version") != PAIR_PUBLICATION_SCHEMA_VERSION
        or receipt.get("state") != "complete"
        or not _is_sha256(receipt.get("protected_attestation_sha256"))
        or receipt.get("source_csv_sha256") != attestation["source_csv_sha256"]
        or receipt.get("policy_version") != attestation["policy_version"]
        or receipt.get("source_revision") != attestation["source_provenance"]["source_revision"]["revision"]
    ):
        raise ValueError("Development release lacks a valid paired-publication receipt")
    role_files = attestation["role_files"]
    if not isinstance(role_files, dict) or set(role_files) != set(DEVELOPMENT_SPLIT_ROLES):
        raise ValueError("Development release role inventory is invalid")

    for role in DEVELOPMENT_SPLIT_ROLES:
        record = role_files[role]
        if (not isinstance(record, dict)
                or set(record) != {"sha256", "rows", "positives", "negatives"}
                or not _is_sha256(record.get("sha256"))
                or any(type(record.get(key)) is not int or record[key] <= 0
                       for key in ("rows", "positives", "negatives"))
                or record["positives"] + record["negatives"] != record["rows"]):
            raise ValueError(f"Development release {role} inventory is invalid")

    frames = []
    for role in selected_roles:
        path = root / f"{role}.csv"
        record = role_files[role]
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"Development release {role} hash mismatch")
        with path.open(newline="", encoding="utf-8") as stream:
            if tuple(next(csv.reader(stream), ())) != RELEASE_COLUMNS:
                raise ValueError(f"Development release {role} has forbidden or reordered columns")
        frame = pd.read_csv(path)
        if tuple(frame.columns) != RELEASE_COLUMNS or frame.empty:
            raise ValueError(f"Development release {role} has invalid columns or no rows")
        values = frame.loc[:, list(CANONICAL_FEATURE_NAMES)].to_numpy(dtype="float32")
        validate_feature_values(values)
        labels = frame[TARGET_NAME]
        if (labels.isna().any() or not pd.api.types.is_integer_dtype(labels.dtype)
                or not labels.isin((0, 1)).all()):
            raise ValueError(f"Development release {role} has invalid labels")
        frame[TARGET_NAME] = labels.astype("int8")
        if {int(value) for value in labels} != {0, 1}:
            raise ValueError(f"Development release {role} needs both labels")
        positives = int(labels.sum())
        if (record["rows"], record["positives"], record["negatives"]) != (
            len(frame), positives, len(frame) - positives
        ):
            raise ValueError(f"Development release {role} counts disagree")
        frame["split_role"] = role
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), attestation


def publish_synthetic_development_release(
    role_frames: dict[str, pd.DataFrame],
    destination: str | Path,
) -> tuple[Path, str]:
    """Exercise three-role, no-overwrite publication in the OS temp tree only."""
    target = _validate_output_path(Path(destination), synthetic=True, kind="development")
    if set(role_frames) != set(DEVELOPMENT_SPLIT_ROLES):
        raise ValueError("Synthetic release requires exactly three development roles")
    _check_development_path(target.parent)
    stage = Path(tempfile.mkdtemp(prefix=".set-c-release-", dir=target.parent))
    try:
        role_files = {}
        for role in DEVELOPMENT_SPLIT_ROLES:
            frame = role_frames[role]
            if tuple(frame.columns) != RELEASE_COLUMNS or frame.empty:
                raise ValueError("Synthetic release has missing or forbidden columns")
            validate_feature_values(frame.loc[:, list(CANONICAL_FEATURE_NAMES)].to_numpy(dtype="float32"))
            labels = frame[TARGET_NAME]
            if (labels.isna().any() or not pd.api.types.is_integer_dtype(labels.dtype)
                    or set(labels.astype(int)) != {0, 1}):
                raise ValueError("Synthetic release requires both integer labels in each role")
            path = stage / f"{role}.csv"
            with path.open("x", newline="", encoding="utf-8") as stream:
                frame.to_csv(stream, index=False)
            positives = int(labels.sum())
            role_files[role] = {
                "sha256": _sha256(path), "rows": len(frame),
                "positives": positives, "negatives": len(frame) - positives,
            }
        attestation = {
            "schema_version": ATTESTATION_SCHEMA_VERSION,
            "release_schema_version": RELEASE_SCHEMA_VERSION,
            "source_csv_sha256": "0" * 64,
            "policy_version": "synthetic-policy.v1",
            "feature_schema_version": SET_C_SCHEMA_VERSION,
            "target_version": TARGET_VERSION,
            "feature_names": list(CANONICAL_FEATURE_NAMES),
            "roles": list(DEVELOPMENT_SPLIT_ROLES),
            "role_files": role_files,
            "synthetic": True,
            "final_test_used": False,
            "source_provenance": _synthetic_test_provenance(
                "0" * 64, sum(record["rows"] for record in role_files.values())
            ),
            "publication_receipt": {
                "schema_version": PAIR_PUBLICATION_SCHEMA_VERSION,
                "state": "complete",
                "protected_attestation_sha256": "0" * 64,
                "source_csv_sha256": "0" * 64,
                "policy_version": "synthetic-policy.v1",
                "source_revision": "synthetic:" + "d" * 40,
            },
        }
        attestation_path = stage / "attestation.json"
        with attestation_path.open("x", encoding="utf-8") as stream:
            json.dump(attestation, stream, sort_keys=True, separators=(",", ":"))
        digest = _sha256(attestation_path)
        _atomic_publish_directory(stage, target)
        return target, digest
    except Exception:
        shutil.rmtree(stage)
        raise


def _allocate_components(
    inventory: pd.DataFrame,
    *,
    dataset_sha256: str,
) -> dict[str, str]:
    """Apply the approved deterministic D-035 lexicographic allocation."""
    if not _is_sha256(dataset_sha256):
        raise ValueError("dataset_sha256 must be a lowercase SHA-256")
    _validate_component_inventory(inventory)
    ordered = inventory.assign(
        digest=inventory["component_id"].map(
            lambda value: sha256(f"{dataset_sha256}:{value}".encode("utf-8")).hexdigest()
        )
    ).sort_values(["digest", "component_id"]).reset_index(drop=True)
    positives = ordered["positives"].to_numpy(dtype="int64")
    negatives = ordered["negatives"].to_numpy(dtype="int64")
    rows = positives + negatives
    component_count, role_count = 26, 4
    binary_count = component_count * role_count
    variable_count = binary_count + 2 * role_count
    constraints: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coefficients: np.ndarray, minimum: float, maximum: float) -> None:
        constraints.append(coefficients)
        lower.append(minimum)
        upper.append(maximum)

    def role_total(values: np.ndarray, role: int) -> np.ndarray:
        coefficients = np.zeros(variable_count)
        coefficients[role:binary_count:role_count] = values
        return coefficients

    for component in range(component_count):
        coefficients = np.zeros(variable_count)
        coefficients[component * role_count:(component + 1) * role_count] = 1
        add(coefficients, 1, 1)
    for role in range(role_count):
        add(role_total(np.ones(component_count), role), ROLE_COMPONENT_TARGETS[role], ROLE_COMPONENT_TARGETS[role])
        positive_total = role_total(positives, role)
        negative_total = role_total(negatives, role)
        add(positive_total, ROLE_POSITIVE_MINIMA[role], np.inf)
        add(negative_total, ROLE_NEGATIVE_MINIMA[role], np.inf)
        for material in MATERIAL_COLUMNS:
            add(role_total(ordered[material].to_numpy(dtype="int64"), role), 1, np.inf)
        for component in range(component_count):
            concentration = positive_total.copy()
            concentration[component * role_count + role] -= 2 * positives[component]
            add(concentration, 0, np.inf)
        for values, target, deviation_index in (
            (positives, ROLE_POSITIVE_BALANCE[role], binary_count + role),
            (rows, ROLE_ROW_BALANCE[role], binary_count + role_count + role),
        ):
            total = role_total(values, role)
            above = total.copy()
            above[deviation_index] = -1
            add(above, -np.inf, target)
            below = -total
            below[deviation_index] = -1
            add(below, -np.inf, -target)

    bounds = Bounds(
        np.zeros(variable_count),
        np.array([1] * binary_count + [np.inf] * (2 * role_count)),
    )
    integrality = np.ones(variable_count, dtype="int8")

    solve_index = 0

    def solve(objective: np.ndarray, *, phase: str, ordinal: int) -> np.ndarray:
        nonlocal solve_index
        solve_index += 1
        context = {
            "stage": "allocation", "objective_phase": phase,
            "objective_index": solve_index, "objective_ordinal": ordinal,
            "time_limit": 60, "mip_rel_gap": 0, "failure_reason": "solver_exception",
        }
        started = time.monotonic()
        try:
            result = milp(
                objective,
                integrality=integrality,
                bounds=bounds,
                constraints=LinearConstraint(np.vstack(constraints), lower, upper),
                options={"time_limit": 60, "mip_rel_gap": 0},
            )
            status = getattr(result, "status", None)
            nodes = getattr(result, "mip_node_count", None)
            success = getattr(result, "success", None)
            context.update(
                solver_status=int(status) if isinstance(status, (int, np.integer)) else None,
                solver_success=bool(success) if isinstance(success, (bool, np.bool_)) else None,
                has_solution=getattr(result, "x", None) is not None,
                mip_node_count=int(nodes) if isinstance(nodes, (int, np.integer)) else None,
            )
            if not result.success or result.status != 0 or result.x is None:
                context["failure_reason"] = "exact_optimum_not_proven"
                raise RuntimeError("D-035 allocation did not prove an exact optimum")
            rounded = np.rint(result.x).astype("int64")
            if not np.allclose(result.x, rounded, atol=1e-6):
                context["failure_reason"] = "fractional_solution"
                raise RuntimeError("D-035 allocation has a fractional solution")
            observed = np.vstack(constraints) @ rounded
            if (np.any(observed < np.asarray(lower) - 1e-6)
                    or np.any(observed > np.asarray(upper) + 1e-6)):
                context["failure_reason"] = "hard_constraint_violation"
                raise RuntimeError("D-035 allocation violates a hard constraint")
            return rounded
        except Exception as error:
            context["elapsed_seconds"] = round(time.monotonic() - started, 6)
            _attach_failure_context(error, context)
            raise

    # Sequential integer objectives implement D-035's lexicographic order.
    solution = None
    for index in range(binary_count, variable_count):
        objective = np.zeros(variable_count)
        objective[index] = 1
        deviation = index - binary_count
        phase = "positive_deviation" if deviation < role_count else "row_deviation"
        solution = solve(objective, phase=phase, ordinal=deviation % role_count + 1)
        fixed = np.zeros(variable_count)
        fixed[index] = 1
        add(fixed, int(solution[index]), int(solution[index]))
    for component in range(component_count):
        objective = np.zeros(variable_count)
        objective[component * role_count:(component + 1) * role_count] = range(role_count)
        solution = solve(objective, phase="tie_break", ordinal=component + 1)
        add(objective, int(objective @ solution), int(objective @ solution))
    assert solution is not None
    assigned = solution[:binary_count].reshape(component_count, role_count)
    if not np.isin(assigned, (0, 1)).all() or not (assigned.sum(axis=1) == 1).all():
        raise RuntimeError("D-035 optimizer returned an invalid assignment")
    return {
        str(ordered.iloc[index]["component_id"]): FOUR_ROLES[int(assigned[index].argmax())]
        for index in range(component_count)
    }


def allocate_synthetic_components(
    inventory: pd.DataFrame,
    *,
    dataset_sha256: str,
) -> dict[str, str]:
    """Run the same policy optimizer on explicitly synthetic component IDs only."""
    if dataset_sha256 != "0" * 64:
        raise PermissionError("Only the fixed synthetic fixture identity may be allocated")
    if any(not str(value).startswith("synthetic-") for value in inventory["component_id"]):
        raise PermissionError("Only synthetic component identities may be allocated")
    return _allocate_components(inventory, dataset_sha256=dataset_sha256)
