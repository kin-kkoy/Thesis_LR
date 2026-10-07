"""Tune, calibrate, and evaluate Set C RF models without touching final-test data.

Artifact creation is disabled by default and requires a separately approved,
versioned configuration.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from dataset_generator import SyntheticDatasetGenerator
from modules.feature_pipeline import (
    AUTHORITATIVE_CSV_COLUMNS,
    AUTHORITATIVE_OBSERVATION_METADATA_NAMES,
    AUTHORITATIVE_SPLIT_ASSIGNMENT_KEY_NAMES,
    CALIBRATION_SELECTION_VERSION,
    CANONICAL_FEATURE_NAMES,
    COMBINED_MANIFEST_VERSION,
    DEVELOPMENT_SPLIT_ROLES,
    DEVELOPMENT_SPLIT_MANIFEST_VERSION,
    MODEL_MANIFEST_VERSION,
    OBSERVATION_MANIFEST_VERSION,
    OBSERVATION_METADATA_NAMES,
    POSITIVE_LABEL,
    SET_C_SCHEMA_VERSION,
    SPLIT_MANIFEST_VERSION,
    TARGET_NAME,
    TARGET_VERSION,
    predict_positive_probability,
)
from modules.set_c_publication import PACKAGE_SCHEMA_VERSION, validate_published_package
from modules.model_trainer import (
    ModelTrainer,
    SUPPORTED_CLASS_WEIGHT_STRATEGIES,
    class_weight_for_estimator,
    load_verified_development_release,
    load_verified_train_validation_release,
)
from modules.wind_convention import WindContract


SEED = 42
PHASE13_PROTOCOL_ID = "phase13_set_c_rf_protocol.v1"
PHASE13_PROTOCOL_SHA256 = "69c39d1d97430ebb903ec2d856e6205a2c1b8ad77641798955d6ad2bdb671637"
PHASE13_CONFIG_SHA256 = "3800fe006463392f102a85ca3c491a6386b5340aeb8d2211eb31ae079a63bfc8"
PHASE13_MANIFEST_SCHEMA = "set_c_rf_phase13_run.v1"
PHASE13_ARTIFACT_KIND = "set_c_rf_uncalibrated_candidate_run"
PHASE13_COMPLETION = "phase13_set_c_rf_completion.v1"
PHASE13_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "phase13_set_c_rf_v1.yaml"


def validate_class_weight_candidates(configured: object) -> tuple[str, ...]:
    """Allow data-derived class weighting, never unapproved numeric cost ratios."""
    if not isinstance(configured, list) or not configured:
        raise ValueError("ml_model.class_weight_candidates requires an approved non-empty list")
    candidates = tuple(str(value) for value in configured)
    if (
        len(set(candidates)) != len(candidates)
        or not set(candidates).issubset(SUPPORTED_CLASS_WEIGHT_STRATEGIES)
    ):
        raise ValueError(
            "Class-weight candidates must be unique 'none', 'balanced', or "
            "'balanced_subsample' tokens; fixed numeric cost ratios are not approved"
        )
    return candidates


def load_grouped_data(
    release_dir: Path,
    *,
    expected_attestation_sha256: str,
    allow_synthetic: bool = False,
) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, pd.Series]]]:
    data, _ = load_verified_development_release(
        release_dir,
        expected_attestation_sha256=expected_attestation_sha256,
        allow_synthetic=allow_synthetic,
    )
    partitions: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for role in DEVELOPMENT_SPLIT_ROLES:
        selected = data[data["split_role"] == role]
        partitions[role] = (
            selected.loc[:, list(CANONICAL_FEATURE_NAMES)].copy(),
            selected[TARGET_NAME].astype(int).copy(),
        )
    return data, partitions


def create_objective(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    class_weight_candidates: tuple[str, ...],
    seed: int = SEED,
):
    """Select hyperparameters and class weighting on non-test validation data."""

    def objective(trial: optuna.Trial) -> float:
        class_weight_strategy = trial.suggest_categorical(
            "class_weight", list(class_weight_candidates)
        )
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 10, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "class_weight": class_weight_for_estimator(class_weight_strategy),
            "random_state": int(seed),
            "n_jobs": -1,
        }
        model = RandomForestClassifier(**params)
        model.fit(x_train, y_train)
        probabilities = predict_positive_probability(model, x_validation)
        return float(average_precision_score(y_validation, probabilities))

    return objective


def fit_calibrated_model(
    best_params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    x_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    calibration_method: str,
    seed: int = SEED,
) -> object:
    """Fit on train+validation, then calibrate only on the calibration role."""
    if calibration_method not in {"sigmoid", "isotonic"}:
        raise ValueError("calibration_method must be explicitly 'sigmoid' or 'isotonic'")
    params = dict(best_params)
    params["class_weight"] = class_weight_for_estimator(str(params["class_weight"]))
    params.update({"random_state": int(seed), "n_jobs": -1})
    x_fit = pd.concat([x_train, x_validation], ignore_index=True)
    y_fit = pd.concat([y_train, y_validation], ignore_index=True)
    base_model = RandomForestClassifier(**params)
    base_model.fit(x_fit, y_fit)

    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator
    except ImportError as exc:
        raise RuntimeError(
            "Current scikit-learn must provide FrozenEstimator for disjoint calibration"
        ) from exc
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(base_model), method=calibration_method
    )
    calibrated.fit(x_calibration, y_calibration)
    return calibrated


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_membership_hash(data: pd.DataFrame) -> str:
    columns = [*AUTHORITATIVE_SPLIT_ASSIGNMENT_KEY_NAMES, "split_role"]
    ordered = data.loc[:, columns].astype(str).sort_values(columns).to_csv(index=False)
    return sha256(ordered.encode("utf-8")).hexdigest()


def _require_new_paths(paths: list[Path]) -> None:
    collisions = [str(path) for path in paths if path.exists()]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite immutable artifacts: " + ", ".join(collisions)
        )


def _validate_observation_wind_manifest(wind: object) -> dict[str, object]:
    """Require the complete canonical simulated-wind provenance mapping."""
    if not isinstance(wind, dict):
        raise ValueError("Observation dataset wind provenance is incomplete")
    try:
        canonical = WindContract.from_config(wind).to_manifest()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Observation dataset wind provenance is invalid") from exc
    if wind != canonical:
        raise ValueError(
            "Observation dataset wind provenance must match the canonical wind contract"
        )
    return canonical


def load_dataset_manifest(manifest_path: Path, csv_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_schema_version") == PACKAGE_SCHEMA_VERSION:
        manifest = validate_published_package(manifest_path)
        required_package = {
            "artifact_kind", "package_id", "package_version", "canonical_csv_columns",
            "predictor_dtype", "label_dtype", "state_dtype", "sampling_policy",
            "split_status", "row_count", "target_counts", "dataset_path",
            "dataset_sha256", "identity", "runs", "manifest_payload_sha256",
        }
        missing_package = sorted(required_package.difference(manifest))
        if missing_package:
            raise ValueError(f"Authoritative package manifest is incomplete: {missing_package}")
        if (
            manifest["artifact_kind"] != "authoritative_set_c_observation_package"
            or tuple(manifest["canonical_csv_columns"]) != AUTHORITATIVE_CSV_COLUMNS
            or manifest["predictor_dtype"] != "float32"
            or manifest["label_dtype"] != "int8"
            or manifest["state_dtype"] != "int8"
            or manifest["sampling_policy"] != "all_eligible"
            or manifest["split_status"] != "unassigned"
        ):
            raise ValueError("Authoritative package manifest violates the Set C contract")
        recorded_dataset = manifest_path.parent / str(manifest["dataset_path"])
        if recorded_dataset.resolve() != csv_path.resolve():
            raise ValueError("Authoritative package manifest identifies another dataset")
        if manifest["dataset_sha256"] != _file_sha256(csv_path):
            raise ValueError("Authoritative package dataset hash mismatch")
        if not isinstance(manifest["row_count"], int) or manifest["row_count"] <= 0:
            raise ValueError("Authoritative package row count must be positive")
        counts = {str(key): int(value) for key, value in manifest["target_counts"].items()}
        if set(counts) != {"0", "1"} or sum(counts.values()) != manifest["row_count"]:
            raise ValueError("Authoritative package class totals are inconsistent")
        identity = manifest["identity"]
        if not isinstance(identity, dict) or not str(identity.get("experiment_id", "")).strip():
            raise ValueError("Authoritative package experiment identity is missing")
        normalized = dict(manifest)
        normalized.update(
            set_id="SetC",
            experiment_id=identity["experiment_id"],
            feature_schema_version=SET_C_SCHEMA_VERSION,
            feature_names=list(CANONICAL_FEATURE_NAMES),
            target_name=TARGET_NAME,
            target_version=TARGET_VERSION,
            positive_label=POSITIVE_LABEL,
            metadata_names=list(AUTHORITATIVE_OBSERVATION_METADATA_NAMES),
        )
        return normalized
    required = {
        "manifest_schema_version",
        "artifact_kind",
        "created_at_utc",
        "set_id",
        "experiment_id",
        "feature_schema_version",
        "feature_names",
        "target_name",
        "target_version",
        "positive_label",
        "metadata_names",
        "dataset_path",
        "dataset_sha256",
        "row_count",
        "target_counts",
        "source_revision",
        "source_dirty_state",
        "generation_command",
        "environment_record",
        "source_rasters",
        "spatial_block",
        "duplicate_policy",
        "sampling_policy",
        "roi",
        "transition_source_type",
        "transition_source_provenance",
        "split_status",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"Dataset manifest is incomplete: {missing}")
    if manifest["set_id"] != "SetC":
        raise ValueError("Dataset manifest must identify SetC")
    for field in (
        "experiment_id",
        "source_revision",
        "generation_command",
        "environment_record",
        "duplicate_policy",
        "split_status",
    ):
        if not str(manifest[field]).strip():
            raise ValueError(f"Dataset manifest field {field!r} cannot be empty")
    if manifest["source_dirty_state"] is None:
        raise ValueError("Dataset manifest source_dirty_state cannot be null")
    if manifest["sampling_policy"] != "all_eligible":
        raise ValueError(
            "Set C probability training requires an explicit all-eligible sampling policy; "
            "case-control sampling needs separate correction evidence and approval"
        )
    if manifest["roi"] != {"mode": "full_aligned_raster_domain"}:
        raise ValueError("Set C dataset manifest must record the full aligned raster domain")
    block = manifest["spatial_block"]
    if not isinstance(block, dict) or not {
        "origin_row", "origin_col", "size_rows", "size_cols"
    }.issubset(block):
        raise ValueError("Dataset manifest spatial_block is incomplete")
    if int(block["size_rows"]) <= 0 or int(block["size_cols"]) <= 0:
        raise ValueError("Dataset manifest spatial block size must be positive")
    kinds = {
        "set_c_observation_dataset": OBSERVATION_MANIFEST_VERSION,
        "set_c_combined_dataset": COMBINED_MANIFEST_VERSION,
    }
    expected_version = kinds.get(manifest["artifact_kind"])
    if expected_version is None or manifest["manifest_schema_version"] != expected_version:
        raise ValueError("Dataset manifest kind or schema version is unsupported")
    if manifest["feature_schema_version"] != SET_C_SCHEMA_VERSION:
        raise ValueError("Dataset manifest feature schema does not match Set C")
    if tuple(manifest["feature_names"]) != CANONICAL_FEATURE_NAMES:
        raise ValueError("Dataset manifest feature order does not match Set C")
    if (
        manifest["target_name"] != TARGET_NAME
        or manifest["target_version"] != TARGET_VERSION
        or manifest["positive_label"] != POSITIVE_LABEL
    ):
        raise ValueError("Dataset manifest target semantics do not match ignition at t+1")
    if tuple(manifest["metadata_names"]) != OBSERVATION_METADATA_NAMES:
        raise ValueError("Dataset manifest metadata schema does not match Set C")
    recorded_dataset = Path(manifest["dataset_path"])
    if not recorded_dataset.is_absolute():
        recorded_dataset = manifest_path.parent / recorded_dataset
    if recorded_dataset.resolve() != csv_path.resolve():
        raise ValueError("Dataset manifest path does not identify the configured dataset")
    if not isinstance(manifest["row_count"], int) or manifest["row_count"] <= 0:
        raise ValueError("Dataset manifest row_count must be a positive integer")
    if not isinstance(manifest["target_counts"], dict) or set(
        manifest["target_counts"]
    ) != {"0", str(POSITIVE_LABEL)}:
        raise ValueError("Dataset manifest target_counts must record both classes")
    if any(int(value) <= 0 for value in manifest["target_counts"].values()):
        raise ValueError("Dataset manifest must record a positive count for both classes")
    if sum(int(value) for value in manifest["target_counts"].values()) != manifest["row_count"]:
        raise ValueError("Dataset manifest target_counts do not sum to row_count")
    if manifest["dataset_sha256"] != _file_sha256(csv_path):
        raise ValueError("Dataset hash does not match its immutable manifest")
    if manifest["artifact_kind"] == "set_c_observation_dataset":
        SyntheticDatasetGenerator.validate_transition_source(
            manifest["transition_source_type"],
            manifest["transition_source_provenance"],
        )
        observation_required = {
            "seed", "event_id", "scenario_id", "run_id", "timestep_t", "wind", "grid", "config_sha256"
        }
        missing_observation = sorted(observation_required.difference(manifest))
        if missing_observation:
            raise ValueError(
                f"Observation dataset manifest is incomplete: {missing_observation}"
            )
        for field in ("event_id", "scenario_id", "run_id"):
            if not str(manifest[field]).strip():
                raise ValueError(f"Observation manifest field {field!r} cannot be empty")
        grid = manifest["grid"]
        if not isinstance(grid, dict) or not {
            "grid_id", "shape", "crs", "transform"
        }.issubset(grid):
            raise ValueError("Observation dataset grid provenance is incomplete")
        _validate_observation_wind_manifest(manifest["wind"])
        rasters = manifest["source_rasters"]
        expected_rasters = {"slope", "proximity", "buildings", "materials", "state_t", "state_t1"}
        if not isinstance(rasters, dict) or set(rasters) != expected_rasters:
            raise ValueError("Observation dataset source-raster provenance is incomplete")
        for name, raster in rasters.items():
            if not isinstance(raster, dict) or not {"path", "sha256"}.issubset(raster):
                raise ValueError(f"Source-raster provenance for {name!r} is incomplete")
    else:
        source_records = manifest["transition_source_provenance"]
        if not isinstance(source_records, list) or not source_records:
            raise ValueError("Combined dataset requires transition-source provenance records")
        for source_record in source_records:
            SyntheticDatasetGenerator.validate_transition_source(
                manifest["transition_source_type"], source_record
            )
        entries = manifest.get("scenario_manifests")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Combined dataset manifest requires scenario_manifests")
        nested_rows = 0
        nested_counts: dict[str, int] = {}
        identities: set[tuple[str, str, str, int]] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or set(("path", "sha256", "content")).difference(entry):
                raise ValueError(f"Combined scenario manifest entry {index} is incomplete")
            nested_path = Path(entry["path"])
            if not nested_path.is_absolute():
                nested_path = manifest_path.parent / nested_path
            if _file_sha256(nested_path) != entry["sha256"]:
                raise ValueError(f"Combined scenario manifest {index} hash mismatch")
            nested = entry["content"]
            if json.loads(nested_path.read_text(encoding="utf-8")) != nested:
                raise ValueError(f"Combined scenario manifest {index} content mismatch")
            if (
                not isinstance(nested, dict)
                or nested.get("manifest_schema_version") != OBSERVATION_MANIFEST_VERSION
                or nested.get("artifact_kind") != "set_c_observation_dataset"
                or not nested.get("dataset_path")
            ):
                raise ValueError(
                    f"Combined scenario manifest {index} is not an observation manifest"
                )
            nested_dataset = Path(nested["dataset_path"])
            if not nested_dataset.is_absolute():
                nested_dataset = nested_path.parent / nested_dataset
            nested_validated = load_dataset_manifest(nested_path, nested_dataset)
            if (
                nested_validated["artifact_kind"] != "set_c_observation_dataset"
                or nested_validated["experiment_id"] != manifest["experiment_id"]
            ):
                raise ValueError(f"Combined scenario manifest {index} violates Set C lineage")
            identity = (
                str(nested.get("event_id", "")),
                str(nested.get("scenario_id", "")),
                str(nested.get("run_id", "")),
                int(nested.get("timestep_t")),
            )
            if identity in identities:
                raise ValueError(f"Combined scenario identity is duplicated: {identity}")
            identities.add(identity)
            nested_rows += int(nested["row_count"])
            for label, count in nested["target_counts"].items():
                nested_counts[str(label)] = nested_counts.get(str(label), 0) + int(count)
        if nested_rows != manifest["row_count"] or nested_counts != {
            str(key): int(value) for key, value in manifest["target_counts"].items()
        }:
            raise ValueError("Combined manifest totals disagree with scenario manifests")
    return manifest


def load_split_manifest_metadata(
    metadata_path: Path,
    split_manifest_path: Path,
    csv_path: Path,
    dataset_manifest_path: Path,
) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "manifest_schema_version",
        "artifact_kind",
        "set_id",
        "experiment_id",
        "feature_schema_version",
        "target_version",
        "assignment_path",
        "assignment_sha256",
        "assignment_rows",
        "split_roles",
        "source_dataset_path",
        "source_dataset_sha256",
        "source_dataset_manifest_sha256",
        "split_strategy",
        "block_algorithm",
        "block_scale_evidence",
        "allocation_policy",
        "group_inventory",
        "seed",
        "created_at_utc",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"Split-manifest metadata is incomplete: {missing}")
    if (
        metadata["manifest_schema_version"] != DEVELOPMENT_SPLIT_MANIFEST_VERSION
        or metadata["artifact_kind"] != "set_c_development_split_assignments"
        or metadata["set_id"] != "SetC"
        or metadata["feature_schema_version"] != SET_C_SCHEMA_VERSION
        or metadata["target_version"] != TARGET_VERSION
        or tuple(metadata["split_roles"]) != DEVELOPMENT_SPLIT_ROLES
    ):
        raise ValueError("Split-manifest metadata violates the Set C contract")
    if metadata["assignment_sha256"] != _file_sha256(split_manifest_path):
        raise ValueError("Split assignment hash does not match its metadata")
    recorded_assignment = Path(metadata["assignment_path"])
    if not recorded_assignment.is_absolute():
        recorded_assignment = metadata_path.parent / recorded_assignment
    if recorded_assignment.resolve() != split_manifest_path.resolve():
        raise ValueError("Split metadata path does not identify the configured assignment")
    recorded_dataset = Path(metadata["source_dataset_path"])
    if not recorded_dataset.is_absolute():
        recorded_dataset = metadata_path.parent / recorded_dataset
    if recorded_dataset.resolve() != csv_path.resolve():
        raise ValueError("Split metadata path does not identify the configured dataset")
    if metadata["source_dataset_sha256"] != _file_sha256(csv_path):
        raise ValueError("Split manifest does not belong to the configured dataset")
    if metadata["source_dataset_manifest_sha256"] != _file_sha256(dataset_manifest_path):
        raise ValueError("Split manifest does not bind the configured dataset manifest")
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if metadata["experiment_id"] != dataset_manifest.get("experiment_id"):
        raise ValueError("Split manifest experiment_id disagrees with the dataset manifest")
    if metadata["block_algorithm"] != "deterministic_raster_anchored":
        raise ValueError("Split manifest must use deterministic raster-anchored blocking")
    if metadata["split_strategy"] not in {"group_first_four_way", "nested_grouped_cv"}:
        raise ValueError("Split manifest uses an unsupported grouped split strategy")
    inventory = metadata["group_inventory"]
    if not isinstance(inventory, dict) or "four_way_feasible" not in inventory:
        raise ValueError("Split manifest requires a documented group-feasibility inventory")
    if inventory["four_way_feasible"] is not True:
        raise ValueError(
            "Independent groups do not support this four-way trainer; use an approved "
            "nested grouped-CV workflow"
        )
    if metadata["split_strategy"] != "group_first_four_way":
        raise ValueError("This trainer requires a feasible group-first four-way split")
    assignments = pd.read_csv(split_manifest_path)
    if int(metadata["assignment_rows"]) != len(assignments):
        raise ValueError("Split assignment row count disagrees with its metadata")
    return metadata


def load_calibration_selection_record(
    record_path: Path,
    *,
    configured_method: str,
    csv_path: Path,
    split_manifest_path: Path,
    split_membership_sha256: str,
    experiment_id: str,
) -> dict:
    """Verify immutable evidence for grouped non-test calibration-method selection."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    required = {
        "record_schema_version",
        "artifact_kind",
        "created_at_utc",
        "set_id",
        "experiment_id",
        "source_dataset_sha256",
        "split_assignment_sha256",
        "split_membership_sha256",
        "candidate_methods",
        "selection_metric",
        "grouped_selection_design",
        "selected_method",
        "selected_on_roles",
        "final_test_used",
        "source_revision",
        "selection_command",
        "environment_record",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"Calibration-selection record is incomplete: {missing}")
    if (
        record["record_schema_version"] != CALIBRATION_SELECTION_VERSION
        or record["artifact_kind"] != "set_c_calibration_method_selection"
        or record["set_id"] != "SetC"
        or record["experiment_id"] != experiment_id
    ):
        raise ValueError("Calibration-selection record violates the Set C contract")
    candidates = set(record["candidate_methods"])
    selected_roles = set(record["selected_on_roles"])
    if (
        configured_method not in {"sigmoid", "isotonic"}
        or record["selected_method"] != configured_method
        or configured_method not in candidates
    ):
        raise ValueError("Configured calibration method lacks matching selection evidence")
    if not selected_roles or not selected_roles.issubset({"train", "validation"}):
        raise ValueError(
            "Calibration method selection must use grouped train/validation evidence only"
        )
    if record["final_test_used"] is not False:
        raise ValueError("Calibration-selection evidence must confirm final test was untouched")
    if not str(record["selection_metric"]).strip() or not str(
        record["grouped_selection_design"]
    ).strip():
        raise ValueError("Calibration-selection criterion and grouped design are required")
    if record["source_dataset_sha256"] != _file_sha256(csv_path):
        raise ValueError("Calibration-selection record belongs to another dataset")
    if record["split_assignment_sha256"] != _file_sha256(split_manifest_path):
        raise ValueError("Calibration-selection record belongs to another split assignment")
    if record["split_membership_sha256"] != split_membership_sha256:
        raise ValueError("Calibration-selection record has stale split membership")
    return record


def build_summary(
    *,
    data: pd.DataFrame,
    csv_path: Path,
    config_path: Path,
    source_revision: str,
    source_dirty_state: str,
    artifact_version: str,
    calibration_method: str,
    dataset_manifest_path: Path,
    dataset_manifest: dict,
    split_manifest_path: Path,
    split_manifest_metadata_path: Path,
    split_manifest_metadata: dict,
    calibration_selection_path: Path,
    calibration_selection_record: dict,
    training_command: str,
    environment_record: str,
    class_weight_candidates: tuple[str, ...],
    best_params: dict,
    best_score: float,
    model: object,
) -> dict:
    return {
        "manifest_schema_version": MODEL_MANIFEST_VERSION,
        "artifact_kind": "set_c_rf_training_run",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "set_id": "SetC",
        "artifact_version": artifact_version,
        "dataset_path": str(csv_path),
        "dataset_sha256": _file_sha256(csv_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": _file_sha256(dataset_manifest_path),
        "dataset_manifest": dataset_manifest,
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": _file_sha256(split_manifest_path),
        "split_manifest_metadata_path": str(split_manifest_metadata_path),
        "split_manifest_metadata_sha256": _file_sha256(split_manifest_metadata_path),
        "split_manifest_metadata": split_manifest_metadata,
        "calibration_selection_record_path": str(calibration_selection_path),
        "calibration_selection_record_sha256": _file_sha256(calibration_selection_path),
        "calibration_selection_record": calibration_selection_record,
        "config_path": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "source_revision": source_revision,
        "source_dirty_state": source_dirty_state,
        "training_command": training_command,
        "environment_record": environment_record,
        "feature_schema_version": SET_C_SCHEMA_VERSION,
        "feature_names": list(CANONICAL_FEATURE_NAMES),
        "target_name": TARGET_NAME,
        "target_version": TARGET_VERSION,
        "positive_label": POSITIVE_LABEL,
        "metadata_names": list(AUTHORITATIVE_OBSERVATION_METADATA_NAMES),
        "development_role_column": "split_role",
        "split_membership_sha256": _split_membership_hash(data),
        "split_counts": data["split_role"].value_counts().sort_index().to_dict(),
        "class_weight_candidates": list(class_weight_candidates),
        "class_weight_and_hyperparameters": best_params,
        "validation_pr_auc": float(best_score),
        "calibration_method": calibration_method,
        "model_feature_names": list(getattr(model, "feature_names_in_", ())),
        "model_classes": [int(value) for value in getattr(model, "classes_", ())],
        "classification_threshold": None,
        "ca_decision_policy": "stochastic_calibrated_probability",
        "final_test_status": "not_loaded_not_evaluated",
    }


def _load_phase13_config(path: Path) -> dict:
    if path.resolve() != PHASE13_CONFIG_PATH.resolve():
        raise PermissionError("Phase 13 accepts only its assigned pinned configuration")
    if _file_sha256(path) != PHASE13_CONFIG_SHA256:
        raise ValueError("Phase 13 configuration SHA-256 mismatch")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Phase 13 configuration must be a mapping")
    if (config.get("protocol_id") != PHASE13_PROTOCOL_ID
            or config.get("protocol_sha256") != PHASE13_PROTOCOL_SHA256
            or config.get("manifest_schema") != PHASE13_MANIFEST_SCHEMA
            or config.get("artifact_kind") != PHASE13_ARTIFACT_KIND
            or tuple(config.get("feature_names", ())) != CANONICAL_FEATURE_NAMES
            or config.get("feature_schema_version") != SET_C_SCHEMA_VERSION
            or config.get("target_version") != TARGET_VERSION
            or config.get("target_name") != TARGET_NAME
            or config.get("positive_label") != POSITIVE_LABEL):
        raise ValueError("Phase 13 configuration violates the accepted protocol/schema")
    return config


def _git_state(expected_revision: str) -> dict:
    repo = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args],
            check=True, capture_output=True, text=True, encoding="utf-8", timeout=20,
        ).stdout.strip()

    revision = git("rev-parse", "--verify", "HEAD")
    dirty = git("status", "--porcelain=v1", "--untracked-files=all")
    if (not re.fullmatch(r"[0-9a-f]{40}", expected_revision)
            or revision != expected_revision or dirty):
        raise PermissionError("Training source must match the supplied clean full Git revision")
    return {"revision": revision, "dirty": False}


def _environment_record(config: dict) -> dict:
    versions = {
        "python": platform.python_version(),
        "scikit_learn": __import__("sklearn").__version__,
        "optuna": optuna.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "joblib": joblib.__version__,
        "pyyaml": yaml.__version__,
    }
    for key, expected in config["environment"].items():
        if versions.get(key) != str(expected):
            raise RuntimeError(f"Required Phase 13 environment {key}={expected}, found {versions.get(key)}")
    return {
        "versions": versions,
        "platform": platform.platform(),
        "processor_count": os.cpu_count(),
        "study_workers": config["study"]["n_jobs"],
        "forest_workers": config["fixed_estimator"]["n_jobs"],
        "warnings": ["Python executable location warning was observed during prior validation; origin unresolved."],
    }


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _resource_snapshot() -> dict:
    if os.name != "nt":
        raise RuntimeError("Phase 13 resource preflight requires Windows physical-memory reporting")
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "available_physical_memory_bytes": int(status.ullAvailPhys),
        "free_c_bytes": int(shutil.disk_usage("C:\\").free),
    }


def _make_sampler_pruner(config: dict) -> tuple[optuna.samplers.TPESampler, optuna.pruners.NopPruner]:
    sampler_config = config["sampler"]
    if sampler_config.get("name") != "TPESampler":
        raise ValueError("Phase 13 requires the configured TPESampler")
    sampler = optuna.samplers.TPESampler(**{
        key: value for key, value in sampler_config.items() if key != "name"
    })
    return sampler, optuna.pruners.NopPruner()


def _callable_identity(value: object) -> str | None:
    if not callable(value):
        return None
    return f"{value.__module__}.{value.__qualname__}"


def _sampler_provenance(sampler: optuna.samplers.TPESampler, config: dict) -> dict:
    configured = config["sampler"]
    constructor_options = {
        "n_ei_candidates": 24,
        "gamma": None,
        "weights": None,
        "consider_prior": None,
        "prior_weight": 1.0,
        "consider_magic_clip": None,
        "consider_endpoints": False,
        "n_startup_trials": configured["n_startup_trials"],
        "seed": configured["seed"],
        "multivariate": configured["multivariate"],
        "group": configured["group"],
        "warn_independent_sampling": None,
        "constant_liar": configured["constant_liar"],
        "constraints_func": None,
        "categorical_distance_func": None,
    }
    resolved = {
        "n_ei_candidates": getattr(sampler, "_n_ei_candidates", 24),
        "gamma": _callable_identity(getattr(sampler, "_gamma", None)),
        "weights": _callable_identity(
            getattr(getattr(sampler, "_parzen_estimator_parameters", None), "weights", None)
        ),
        "consider_prior": getattr(sampler, "_consider_prior", True),
        "prior_weight": getattr(sampler, "_prior_weight", 1.0),
        "consider_magic_clip": getattr(sampler, "_consider_magic_clip", True),
        "consider_endpoints": getattr(sampler, "_consider_endpoints", False),
        "n_startup_trials": getattr(sampler, "_n_startup_trials", configured["n_startup_trials"]),
        "seed": configured["seed"],
        "multivariate": getattr(sampler, "_multivariate", configured["multivariate"]),
        "group": getattr(sampler, "_group", configured["group"]),
        "warn_independent_sampling": getattr(sampler, "_warn_independent_sampling", True),
        "constant_liar": getattr(sampler, "_constant_liar", configured["constant_liar"]),
        "constraints_func": _callable_identity(getattr(sampler, "_constraints_func", None)),
        "categorical_distance_func": _callable_identity(
            getattr(sampler, "_categorical_distance_func", None)
        ),
    }
    return {
        "name": "TPESampler",
        "optuna_version": optuna.__version__,
        "constructor_options": constructor_options,
        "resolved_options": resolved,
        "default_behavior_version": optuna.__version__,
    }


def _write_diagnostics(path: Path, payload: dict, *, fallback_to_stderr: bool) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as error:
        if not fallback_to_stderr:
            raise
        try:
            sys.stderr.write(
                f"Phase 13 diagnostics file write failed at {path}: "
                f"{type(error).__name__}: {error}; diagnostics={json.dumps(payload, sort_keys=True)}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass


def _write_selection_freeze(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        stream.flush()
        os.fsync(stream.fileno())


def _assert_resource_floors(sample: dict, config: dict) -> None:
    limits = config["resource_limits"]
    if sample["available_physical_memory_bytes"] < limits["min_available_physical_memory_bytes"]:
        raise RuntimeError("Available physical memory is below the Phase 13 floor")
    if sample["free_c_bytes"] < limits["min_free_c_bytes"]:
        raise RuntimeError("Free C: disk space is below the Phase 13 floor")


def _validate_phase13_inputs(config: dict) -> tuple[dict, dict]:
    train_x, train_y, validation_x, validation_y, attestation, provenance = (
        load_verified_train_validation_release(
            config["release_dir"],
            expected_attestation_sha256=config["release_attestation_sha256"],
        )
    )
    for role, x, y in (("train", train_x, train_y), ("validation", validation_x, validation_y)):
        if (tuple(x.columns) != CANONICAL_FEATURE_NAMES
                or any(dtype != np.dtype("float32") for dtype in x.dtypes)
                or y.dtype != np.dtype("int8") or set(y.astype(int)) != {0, 1}):
            raise ValueError(f"{role} data violates the Phase 13 feature/label contract")
        counts = config["expected_role_counts"][role]
        positives = int(y.sum())
        if (len(y), positives, len(y) - positives) != (
            counts["rows"], counts["positives"], counts["negatives"]
        ):
            raise ValueError(f"{role} row/class counts do not match the accepted release")
        if _file_sha256(Path(config["release_dir"]) / f"{role}.csv") != config["role_sha256"][role]:
            raise ValueError(f"{role} CSV SHA-256 differs from the accepted protocol")
    if attestation.get("source_csv_sha256") != config["source_provenance"]["source_csv_sha256"]:
        raise ValueError("Release source CSV identity differs from accepted provenance")
    source_provenance = attestation["source_provenance"]
    manifest = source_provenance["source_manifest"]
    revision = source_provenance["source_revision"]
    if (manifest.get("source_csv_sha256") != config["source_provenance"]["source_csv_sha256"]
            or manifest.get("manifest_sha256") != config["source_provenance"]["source_manifest_sha256"]
            or revision.get("revision") != config["source_provenance"]["export_revision"]):
        raise ValueError("Release provenance differs from accepted Phase 12 evidence")
    return {
        "X_train": train_x, "y_train": train_y,
        "X_validation": validation_x, "y_validation": validation_y,
    }, {
        "attestation": attestation,
        "provenance": provenance,
        "observed_role_counts": {
            "train": {"rows": int(len(train_y)), "positives": int(train_y.sum()),
                      "negatives": int(len(train_y) - train_y.sum())},
            "validation": {"rows": int(len(validation_y)), "positives": int(validation_y.sum()),
                           "negatives": int(len(validation_y) - validation_y.sum())},
            "refit": {"rows": int(len(train_y) + len(validation_y)),
                      "positives": int(train_y.sum() + validation_y.sum()),
                      "negatives": int(len(train_y) + len(validation_y)
                                       - train_y.sum() - validation_y.sum())},
        },
    }


def _validate_destination(config: dict, *, execution: bool, evidence_dir: Path | None) -> tuple[Path, Path]:
    destination = Path(config["destination"]).resolve()
    expected = (Path(__file__).resolve().parent / "output" / "phase13" / "set_c_rf_v1").resolve()
    if destination != expected or destination.exists():
        raise FileExistsError("Phase 13 destination must be the absent approved versioned path")
    parent = destination.parent
    for candidate in (destination, parent, *parent.parents):
        if candidate.is_symlink() or (candidate.exists()
                                      and getattr(candidate.lstat(), "st_file_attributes", 0) & 0x400):
            raise PermissionError("Phase 13 destination contains a reparse point")
    if parent.exists() and not parent.is_dir():
        raise NotADirectoryError("Approved Phase 13 destination parent is not a directory")
    if parent.exists() and any(parent.glob(".set_c_rf_v1.staging-*")):
        raise FileExistsError("Interrupted Phase 13 staging exists; preserve it and stop")
    evidence_parent = Path(config["evidence_parent"]).resolve()
    if evidence_parent != parent / "evidence":
        raise PermissionError("Phase 13 evidence path differs from the approved sibling directory")
    if evidence_dir is None:
        raise PermissionError("Phase 13 evidence directory identity is required")
    supplied_evidence = evidence_dir.absolute()
    if (supplied_evidence.parent != evidence_parent
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", supplied_evidence.name)):
        raise PermissionError("Phase 13 evidence directory must be a safe versioned child")
    for candidate in (supplied_evidence, *supplied_evidence.parents):
        if candidate.is_symlink() or (candidate.exists()
                                      and getattr(candidate.lstat(), "st_file_attributes", 0) & 0x400):
            raise PermissionError("Phase 13 evidence path contains a reparse point")
    evidence_dir = supplied_evidence.resolve()
    if evidence_dir != supplied_evidence:
        raise PermissionError("Phase 13 evidence path resolves outside its approved identity")
    if execution and not evidence_dir.is_dir():
        raise FileNotFoundError("Phase 13 execution requires a fresh supervisor evidence directory")
    if evidence_dir.exists():
        entries = list(evidence_dir.iterdir())
        if not execution:
            raise FileExistsError("Phase 13 evidence directory must be absent for preflight")
        allowed_entries = {
            "stdout.log", "stderr.log", "supervisor_handoff.json",
            "process.json", "operator_resources.jsonl",
        }
        if any(item.name not in allowed_entries for item in entries):
            raise FileExistsError("Phase 13 evidence directory contains an unexpected entry")
        for item in entries:
            item_stat = item.lstat()
            if (item.is_symlink() or getattr(item_stat, "st_file_attributes", 0) & 0x400
                    or not item.is_file()):
                raise PermissionError("Phase 13 evidence contains a non-regular or reparse entry")
        for redirect in (evidence_dir / "stdout.log", evidence_dir / "stderr.log"):
            if not redirect.is_file() or redirect.stat().st_size != 0:
                raise FileExistsError("Phase 13 redirected logs must be freshly created and empty at preflight")

        handoff_path = evidence_dir / "supervisor_handoff.json"
        process_nonce = os.environ.get("PHASE13_SUPERVISOR_NONCE", "")
        if not process_nonce or not re.fullmatch(r"[0-9a-f]{32}", process_nonce):
            raise PermissionError("Phase 13 execution requires the supervisor's per-process handoff")
        try:
            handoff = json.loads(handoff_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Phase 13 supervisor handoff is missing or malformed") from error
        handoff_fields = {
            "schema_version", "run_id", "nonce", "created_at_utc", "executable",
        }
        if (not isinstance(handoff, dict) or set(handoff) != handoff_fields
                or handoff.get("schema_version") != "phase13_set_c_rf_supervisor_handoff.v1"
                or handoff.get("run_id") != evidence_dir.name
                or handoff.get("nonce") != process_nonce):
            raise PermissionError("Phase 13 supervisor handoff identity is foreign or unexpected")

        def parse_utc(value: object, description: str) -> datetime:
            if not isinstance(value, str):
                raise ValueError(f"Phase 13 {description} timestamp is malformed")
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(f"Phase 13 {description} timestamp is malformed") from error
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError(f"Phase 13 {description} timestamp must include a timezone")
            return parsed.astimezone(timezone.utc)

        now_utc = datetime.now(timezone.utc)
        handoff_time = parse_utc(handoff["created_at_utc"], "handoff")
        if handoff_time > now_utc + timedelta(seconds=30) or now_utc - handoff_time > timedelta(minutes=15):
            raise PermissionError("Phase 13 supervisor handoff is stale or future-dated")
        expected_executable = os.path.normcase(str(Path(sys.executable).resolve()))
        handoff_executable = os.path.normcase(str(Path(handoff["executable"]).resolve()))
        if handoff_executable != expected_executable:
            raise PermissionError("Phase 13 supervisor handoff names a different Python executable")

        process_path = evidence_dir / "process.json"
        if process_path.exists():
            try:
                process_record = json.loads(process_path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("Phase 13 supervisor process record is malformed") from error
            required_process_fields = {
                "schema_version", "run_id", "process_id", "process_ids",
                "process_started_at_utc", "executable", "working_directory",
            }
            optional_process_fields = {"stop_reason", "updated_at_utc"}
            if (not isinstance(process_record, dict)
                    or not required_process_fields.issubset(process_record)
                    or set(process_record) - required_process_fields - optional_process_fields
                    or process_record.get("schema_version") != "phase13_set_c_rf_process.v1"
                    or process_record.get("run_id") != evidence_dir.name):
                raise PermissionError("Phase 13 supervisor process record is foreign or unexpected")
            process_ids = process_record["process_ids"]
            process_id = process_record["process_id"]
            if (type(process_id) is not int or process_id not in {os.getpid(), os.getppid()}
                    or not isinstance(process_ids, list)
                    or not process_ids or any(type(pid) is not int or pid <= 0 for pid in process_ids)
                    or len(set(process_ids)) != len(process_ids) or process_id not in process_ids):
                raise PermissionError("Phase 13 supervisor process record does not bind this child")
            process_started = parse_utc(process_record["process_started_at_utc"], "process")
            if (process_started < handoff_time - timedelta(seconds=2)
                    or process_started > now_utc + timedelta(seconds=30)
                    or now_utc - process_started > timedelta(minutes=15)):
                raise PermissionError("Phase 13 supervisor process record is stale or mismatched")
            process_executable = os.path.normcase(str(Path(process_record["executable"]).resolve()))
            process_working_directory = os.path.normcase(
                str(Path(process_record["working_directory"]).resolve())
            )
            expected_working_directory = os.path.normcase(str(Path(__file__).resolve().parent))
            if (process_executable != expected_executable
                    or process_working_directory != expected_working_directory):
                raise PermissionError("Phase 13 supervisor process record names an unexpected child")
            if "stop_reason" in process_record and not isinstance(process_record["stop_reason"], str):
                raise ValueError("Phase 13 supervisor stop reason is malformed")
            if "updated_at_utc" in process_record:
                update_time = parse_utc(process_record["updated_at_utc"], "process update")
                if update_time < process_started or update_time > now_utc + timedelta(seconds=30):
                    raise PermissionError("Phase 13 supervisor process update is stale or future-dated")

        resources_path = evidence_dir / "operator_resources.jsonl"
        if resources_path.exists():
            raw_resources = None
            for _attempt in range(20):
                size_before = resources_path.stat().st_size
                raw_candidate = resources_path.read_bytes()
                size_after = resources_path.stat().st_size
                if size_before == size_after and (not raw_candidate or raw_candidate.endswith(b"\n")):
                    raw_resources = raw_candidate
                    break
                time.sleep(0.01)
            if raw_resources is None:
                raise ValueError("Phase 13 supervisor resource log has an incomplete append")
            try:
                resource_lines = raw_resources.decode("utf-8-sig").splitlines()
                resource_records = [json.loads(line) for line in resource_lines if line.strip()]
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("Phase 13 supervisor resource log is malformed") from error
            if resource_lines and not resource_records:
                raise ValueError("Phase 13 supervisor resource log has no complete samples")
            resource_fields = {
                "run_id", "timestamp_utc", "elapsed_seconds",
                "available_physical_memory_bytes", "free_c_bytes",
            }
            limits = config["resource_limits"]
            for record in resource_records:
                if not isinstance(record, dict) or set(record) != resource_fields or record["run_id"] != evidence_dir.name:
                    raise PermissionError("Phase 13 supervisor resource sample is foreign or unexpected")
                sample_time = parse_utc(record["timestamp_utc"], "resource")
                elapsed = record["elapsed_seconds"]
                memory = record["available_physical_memory_bytes"]
                free_c = record["free_c_bytes"]
                if (sample_time < handoff_time - timedelta(seconds=2)
                        or sample_time > now_utc + timedelta(seconds=30)
                        or type(elapsed) not in (int, float) or not math.isfinite(float(elapsed)) or elapsed < 0
                        or type(memory) is not int or type(free_c) is not int
                        or memory < limits["min_available_physical_memory_bytes"]
                        or free_c < limits["min_free_c_bytes"]):
                    raise PermissionError("Phase 13 supervisor resource sample is stale or below the approved floor")
    return destination, evidence_dir


def _preflight(config_path: Path, expected_revision: str, expected_config_sha256: str,
               *, execution: bool, evidence_dir: Path | None) -> dict:
    config = _load_phase13_config(config_path)
    if expected_config_sha256 != PHASE13_CONFIG_SHA256:
        raise ValueError("Supplied Phase 13 configuration pin is incorrect")
    source = _git_state(expected_revision)
    environment = _environment_record(config)
    resource = _resource_snapshot()
    _assert_resource_floors(resource, config)
    destination, evidence = _validate_destination(config, execution=execution, evidence_dir=evidence_dir)
    partitions, verified = _validate_phase13_inputs(config)
    return {
        "config": config, "source": source, "environment": environment,
        "resource": resource, "destination": destination, "evidence_dir": evidence,
        "partitions": partitions, **verified,
    }


def _estimator(params: dict, config: dict) -> RandomForestClassifier:
    token = params["class_weight"]
    if token not in config["search"]["class_weight"]:
        raise ValueError("Invalid Phase 13 class-weight token")
    estimator_params = dict(config["fixed_estimator"])
    estimator_params.update({key: value for key, value in params.items() if key != "class_weight"})
    estimator_params["class_weight"] = class_weight_for_estimator(token)
    model = RandomForestClassifier(**estimator_params)
    model.phase13_calibrated = False
    return model


def _phase13_trial_params(trial: optuna.Trial, config: dict) -> dict:
    return {
        name: trial.suggest_categorical(name, values)
        for name, values in config["search"].items()
    }


def _trial_record(trial: optuna.trial.FrozenTrial) -> dict:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "params": trial.params,
        "value": trial.value,
        "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
        "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
        "user_attrs": trial.user_attrs,
        "system_attrs": trial.system_attrs,
    }


def _select_best_completed_trial(trials: list[object]) -> object:
    completed = [trial for trial in trials
                 if trial.state is optuna.trial.TrialState.COMPLETE and trial.value is not None]
    if not completed:
        raise ValueError("No completed Phase 13 trials are eligible for selection")
    return min(completed, key=lambda trial: (-float(trial.value), trial.number))


def _write_trial_records(path: Path, study: optuna.Study) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for trial in study.trials:
            stream.write(json.dumps(_trial_record(trial), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _package_files(stage: Path) -> dict:
    return {
        path.name: {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
        for path in sorted(stage.iterdir()) if path.is_file() and path.name not in {"manifest.json", "completion.json"}
    }


def _execute_phase13(preflight: dict, *, approval_reference: str, run_id: str,
                     package_version: str, command: list[str]) -> Path:
    config = preflight["config"]
    if (not approval_reference.strip()
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id)
            or package_version != Path(config["destination"]).name):
        raise PermissionError("Execution requires approval reference and approved run/package IDs")
    destination = preflight["destination"]
    evidence_dir = preflight["evidence_dir"]
    if destination.exists():
        raise FileExistsError("Approved Phase 13 destination appeared after preflight")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".set_c_rf_v1.staging-{run_id}"
    if any(destination.parent.glob(".set_c_rf_v1.staging-*")):
        raise FileExistsError("Phase 13 staging already exists; preserve it and stop")
    stage.mkdir()
    partitions = preflight["partitions"]
    x_train, y_train = partitions["X_train"], partitions["y_train"]
    x_validation, y_validation = partitions["X_validation"], partitions["y_validation"]
    resource_samples = [preflight["resource"]]
    stop_monitor = threading.Event()
    resource_error: list[str] = []

    def monitor() -> None:
        interval = config["resource_limits"]["monitor_interval_seconds"]
        while not stop_monitor.wait(interval):
            try:
                sample = _resource_snapshot()
                resource_samples.append(sample)
                _assert_resource_floors(sample, config)
                with (evidence_dir / "resources.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(sample, sort_keys=True) + "\n")
            except Exception as error:
                resource_error.append(str(error))
                stop_monitor.set()

    with (evidence_dir / "resources.jsonl").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(resource_samples[0], sort_keys=True) + "\n")
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    study_started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    selected: dict = {"score": -float("inf"), "number": None, "model": None}
    study: optuna.Study | None = None
    sampler_provenance: dict | None = None
    trials_path = stage / "trials.jsonl"
    diagnostics_path = stage / "run_diagnostics.json"
    freeze_path = stage / "selection_freeze.json"

    def objective(trial: optuna.Trial) -> float:
        if resource_error:
            raise RuntimeError(f"Resource floor breached: {resource_error[-1]}")
        params = _phase13_trial_params(trial, config)
        model = _estimator(params, config)
        model.fit(x_train, y_train)
        probabilities = predict_positive_probability(model, x_validation)
        score = float(average_precision_score(y_validation, probabilities))
        if not np.isfinite(score):
            raise ValueError("Phase 13 validation AP is not finite")
        trial.set_user_attr("class_weight_token", params["class_weight"])
        trial.set_user_attr("estimator_params", model.get_params(deep=False))
        if (score > selected["score"]
                or (score == selected["score"] and trial.number < selected["number"])):
            selected.update(score=score, number=trial.number, model=model)
        return score

    try:
        sampler, pruner = _make_sampler_pruner(config)
        sampler_provenance = _sampler_provenance(sampler, config)
        study = optuna.create_study(
            direction=optuna.study.StudyDirection.MAXIMIZE,
            sampler=sampler,
            pruner=pruner,
        )
        for baseline in config["baselines"]:
            study.enqueue_trial(baseline)

        def persist(study_obj: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
            _write_trial_records(trials_path, study_obj)
            if resource_error:
                raise RuntimeError(f"Resource floor breached: {resource_error[-1]}")

        study.optimize(
            objective,
            n_trials=config["study"]["n_trials"],
            timeout=config["study"]["timeout_seconds"],
            n_jobs=config["study"]["n_jobs"],
            callbacks=[persist],
        )
        completed = [trial for trial in study.trials if trial.state is optuna.trial.TrialState.COMPLETE]
        if len(completed) != config["study"]["n_trials"]:
            raise RuntimeError("Phase 13 study incomplete; selection and final refit are forbidden")
        best = _select_best_completed_trial(study.trials)
        if best.number != selected["number"] or selected["model"] is None:
            raise RuntimeError("Selected trial model was not retained consistently")
        selection_model = selected["model"]
        best_params = dict(best.params)
        freeze_record = {
            "schema_version": "phase13_set_c_selection_freeze.v1",
            "selected_trial_number": best.number,
            "validation_average_precision_full_precision": float(best.value),
            "parameters": best_params,
            "class_weight_token": best_params["class_weight"],
            "tie_policy": "exact equality; smallest trial number",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_model_identity": f"trial-{best.number}-selection",
        }
        _write_selection_freeze(freeze_path, freeze_record)
        refit_x = pd.concat((x_train, x_validation), ignore_index=True)
        refit_y = pd.concat((y_train, y_validation), ignore_index=True)
        refit_model = _estimator(best_params, config)
        refit_model.fit(refit_x, refit_y)
        selection_model.phase13_calibrated = False
        refit_model.phase13_calibrated = False
        joblib.dump(selection_model, stage / "selection_rf.joblib")
        joblib.dump(refit_model, stage / "base_rf.joblib")
        _write_trial_records(trials_path, study)
        metrics = {
            "selection_trial_number": best.number,
            "selection_validation_average_precision": float(best.value),
            "validation_positive_fraction": float(y_validation.mean()),
            "selection_validation_roc_auc": float(roc_auc_score(
                y_validation, predict_positive_probability(selection_model, x_validation)
            )),
            "selection_validation_brier_score": float(brier_score_loss(
                y_validation, predict_positive_probability(selection_model, x_validation), pos_label=POSITIVE_LABEL
            )),
            "selection_validation_log_loss": float(log_loss(
                y_validation, predict_positive_probability(selection_model, x_validation), labels=[0, 1]
            )),
            "refit_validation_score": None,
            "limitation": "Selection validation metrics describe the train-only selected model; refit model used validation labels.",
        }
        (stage / "selection_validation_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        shutil.copyfile(PHASE13_CONFIG_PATH, stage / PHASE13_CONFIG_PATH.name)
        stop_monitor.set()
        monitor_thread.join()
        if resource_error:
            raise RuntimeError(f"Resource floor breached: {resource_error[-1]}")
        ended_at = datetime.now(timezone.utc).isoformat()
        diagnostics = {
            "run_id": run_id, "started_at_utc": started_at, "ended_at_utc": ended_at,
            "study_elapsed_seconds": time.monotonic() - study_started,
            "resource_samples": resource_samples,
            "minimum_sampled_available_physical_memory_bytes": min(
                row["available_physical_memory_bytes"] for row in resource_samples
            ),
            "minimum_sampled_free_c_bytes": min(row["free_c_bytes"] for row in resource_samples),
            "resource_policy": config["resource_limits"], "stop_reason": "completed",
            "approval_reference": approval_reference,
        }
        _write_diagnostics(diagnostics_path, diagnostics, fallback_to_stderr=False)
        environment = preflight["environment"]
        trial_manifest = [
            {**_trial_record(trial), "validation_average_precision_full_precision": trial.value}
            for trial in study.trials
        ]
        source_provenance = preflight["attestation"]["source_provenance"]
        manifest = {
            "schema_version": PHASE13_MANIFEST_SCHEMA, "artifact_kind": PHASE13_ARTIFACT_KIND,
            "protocol_id": PHASE13_PROTOCOL_ID, "protocol_sha256": PHASE13_PROTOCOL_SHA256,
            "run_id": run_id, "approval_reference": approval_reference,
            "package_version": package_version, "completion_status": "complete",
            "output_path": str(destination), "staging_path": str(stage),
            "inputs": {
                "development_path": config["release_dir"],
                "attestation_sha256": config["release_attestation_sha256"],
                "train_sha256": config["role_sha256"]["train"],
                "validation_sha256": config["role_sha256"]["validation"],
                "source_csv_sha256": source_provenance["source_manifest"]["source_csv_sha256"],
                "source_manifest_sha256": config["source_provenance"]["source_manifest_sha256"],
                "export_revision": config["source_provenance"]["export_revision"],
            },
            "training_source": {
                **preflight["source"], "command": command,
                "working_directory": str(Path.cwd()), "configuration_sha256": PHASE13_CONFIG_SHA256,
                "resolved_configuration": config,
            },
            "environment": environment,
            "schema_population": {
                "feature_names": list(CANONICAL_FEATURE_NAMES), "feature_dtype": "float32",
                "target_name": TARGET_NAME, "target_dtype": "int8", "positive_label": POSITIVE_LABEL,
                "feature_schema_version": SET_C_SCHEMA_VERSION, "target_version": TARGET_VERSION,
                "expected_role_counts": config["expected_role_counts"],
                "observed_role_counts": preflight["observed_role_counts"],
                "trial_fit_role": "train", "selection_role": "validation",
                "refit_roles_ordered": ["train", "validation"],
            },
            "study": {
                "direction": "maximize", "objective": "validation_average_precision",
                "sampler": sampler_provenance,
                "pruner": {"name": "NopPruner"}, "seed": SEED,
                "search_choices": config["search"], "baseline_order": config["baselines"],
                "n_trials": config["study"]["n_trials"], "timeout_seconds": config["study"]["timeout_seconds"],
                "study_n_jobs": config["study"]["n_jobs"], "tie_policy": "exact score equality; smallest trial number",
                "trials": trial_manifest, "selected_trial_number": best.number,
            },
            "selection_freeze": {
                "path": freeze_path.name,
                "sha256": _file_sha256(freeze_path),
                "record": freeze_record,
            },
            "models": {
                "selection_model": {"identity": f"trial-{best.number}-selection", "filename": "selection_rf.joblib",
                    "training_roles": ["train"], "params": selection_model.get_params(deep=False),
                    "feature_names_in": list(selection_model.feature_names_in_), "classes": [int(v) for v in selection_model.classes_],
                    "calibrated": False},
                "base_model": {"identity": f"{run_id}-train-validation-refit", "filename": "base_rf.joblib",
                    "training_roles_ordered": ["train", "validation"], "params": refit_model.get_params(deep=False),
                    "feature_names_in": list(refit_model.feature_names_in_), "classes": [int(v) for v in refit_model.classes_],
                    "calibrated": False, "heldout_performance_claim": None},
            },
            "metrics": metrics,
            "excluded_operations": {"calibration_method": None, "threshold": None,
                "calibration_contents_opened": False, "final_test_loaded_or_evaluated": False, "ca_run": False},
            "execution": {**diagnostics, "actual_exit_status": 0,
                "evidence_files": {
                    "stdout": {"path": str(evidence_dir / "stdout.log"), "sha256": None,
                               "hash_owner": "PowerShell operator completion record after process exit"},
                    "stderr": {"path": str(evidence_dir / "stderr.log"), "sha256": None,
                               "hash_owner": "PowerShell operator completion record after process exit"},
                    "python_resources": {"path": str(evidence_dir / "resources.jsonl"),
                        "sha256": _file_sha256(evidence_dir / "resources.jsonl"),
                        "bytes": (evidence_dir / "resources.jsonl").stat().st_size},
                },
                "completion_verification": "all required trials/refit/package checks passed"},
            "files": _package_files(stage),
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        manifest_sha = _file_sha256(manifest_path)
        (stage / "completion.json").write_text(json.dumps({
            "schema_version": PHASE13_COMPLETION, "status": "complete",
            "manifest_sha256": manifest_sha, "run_id": run_id, "actual_exit_status": 0,
        }, indent=2, sort_keys=True), encoding="utf-8")
        if destination.exists() or stage.exists() is False:
            raise FileExistsError("Phase 13 destination changed or staging disappeared before publication")
        os.rename(stage, destination)
        return destination
    except BaseException as error:
        original_traceback = traceback.format_exc()
        stop_monitor.set()
        monitor_thread.join(timeout=2)
        trial_record_write_error = None
        if study is not None:
            try:
                _write_trial_records(trials_path, study)
            except BaseException as write_error:
                trial_record_write_error = f"{type(write_error).__name__}: {write_error}"
                try:
                    sys.stderr.write(f"Phase 13 trial-record retry failed: {trial_record_write_error}\n")
                    sys.stderr.flush()
                except BaseException:
                    pass
        partial = {
            "run_id": run_id, "started_at_utc": started_at,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - study_started,
            "stop_reason": f"{type(error).__name__}: {error}",
            "traceback": original_traceback,
            "resource_samples": resource_samples,
            "resource_error": resource_error,
            "trial_record_write_error": trial_record_write_error,
            "actual_exit_status": 1,
            "preserved_partial_staging": str(stage),
        }
        _write_diagnostics(diagnostics_path, partial, fallback_to_stderr=True)
        raise
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=2)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 13 Set C RF tune/refit with default-deny execution")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=PHASE13_CONFIG_PATH)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--approval-reference")
    parser.add_argument("--run-id")
    parser.add_argument("--package-version")
    parser.add_argument("--evidence-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _load_phase13_config(args.config)
    if args.execute and (
        not args.approval_reference or not args.approval_reference.strip()
        or not args.run_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.run_id)
        or args.package_version != Path(config["destination"]).name
        or args.evidence_dir is None
    ):
        raise PermissionError("Execution requires approval reference, safe run ID, approved package version, and evidence directory")
    preflight = _preflight(
        args.config, args.source_revision, args.expected_config_sha256,
        execution=args.execute, evidence_dir=args.evidence_dir,
    )
    print(json.dumps({"preflight": "PASS", "source": preflight["source"],
                      "resource": preflight["resource"], "destination": str(preflight["destination"])},
                     sort_keys=True))
    if args.preflight_only:
        return
    _execute_phase13(
        preflight, approval_reference=args.approval_reference, run_id=args.run_id,
        package_version=args.package_version, command=[sys.executable, *sys.argv],
    )

if __name__ == "__main__":
    main()
