"""Tune, calibrate, and evaluate Set C RF models without touching final-test data.

Artifact creation is disabled by default and requires a separately approved,
versioned configuration.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import tempfile

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from dataset_generator import SyntheticDatasetGenerator
from modules.feature_pipeline import (
    CALIBRATION_SELECTION_VERSION,
    CANONICAL_FEATURE_NAMES,
    COMBINED_MANIFEST_VERSION,
    MODEL_MANIFEST_VERSION,
    OBSERVATION_MANIFEST_VERSION,
    OBSERVATION_METADATA_NAMES,
    POSITIVE_LABEL,
    PROVENANCE_METADATA_NAMES,
    SET_C_SCHEMA_VERSION,
    SPLIT_ROLES,
    SPLIT_MANIFEST_VERSION,
    TARGET_NAME,
    TARGET_VERSION,
    predict_positive_probability,
)
from modules.model_trainer import (
    ModelTrainer,
    SUPPORTED_CLASS_WEIGHT_STRATEGIES,
    class_weight_for_estimator,
)
from modules.wind_convention import WindContract
from validation_engine import evaluate_estimator_probabilities


SEED = 42


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
    csv_path: Path,
    split_manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, pd.Series]]]:
    observations = pd.read_csv(csv_path)
    assignments = pd.read_csv(split_manifest_path)
    data = ModelTrainer.merge_split_assignments(observations, assignments)
    ModelTrainer._validate_dataset_contract(data)
    partitions: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for role in SPLIT_ROLES:
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
    columns = [
        "experiment_id",
        "event_id",
        "scenario_id",
        "run_id",
        "timestep_t",
        "grid_id",
        "cell_row",
        "cell_col",
        "duplicate_group_id",
        "spatial_block_id",
        "split_role",
    ]
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
        metadata["manifest_schema_version"] != SPLIT_MANIFEST_VERSION
        or metadata["artifact_kind"] != "set_c_split_assignments"
        or metadata["set_id"] != "SetC"
        or metadata["feature_schema_version"] != SET_C_SCHEMA_VERSION
        or metadata["target_version"] != TARGET_VERSION
        or tuple(metadata["split_roles"]) != SPLIT_ROLES
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
    metrics: dict,
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
        "metadata_names": list(PROVENANCE_METADATA_NAMES),
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
        "final_test_probability_metrics": metrics,
    }


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    config_path = code_dir / "config" / "default_experiment.yaml"
    with config_path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    ml_cfg = dict(config.get("ml_model", {}))
    if not bool(ml_cfg.get("artifact_write_enabled", False)):
        raise PermissionError(
            "ml_model.artifact_write_enabled is false; artifact-producing training is not authorized"
        )

    required_text = (
        "training_csv",
        "artifact_version",
        "source_revision",
        "source_dirty_state",
        "calibration_method",
        "dataset_manifest_path",
        "split_manifest_path",
        "training_command",
        "environment_record",
        "calibration_selection_record_path",
    )
    missing = [name for name in required_text if ml_cfg.get(name) in (None, "")]
    if missing:
        raise ValueError(f"Missing approved training provenance configuration: {missing}")
    if ml_cfg.get("classification_threshold") is not None:
        raise ValueError("Numeric classification threshold is not approved")
    if ml_cfg.get("positive_label") != POSITIVE_LABEL:
        raise ValueError("ml_model.positive_label must be integer 1")
    if ml_cfg.get("inference_mode") != "stochastic_probability":
        raise ValueError("Only threshold-free stochastic probability inference is approved")
    class_weight_candidates = validate_class_weight_candidates(
        ml_cfg.get("class_weight_candidates")
    )

    version = str(ml_cfg["artifact_version"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
        raise ValueError("artifact_version must be a safe, path-free identifier")

    csv_path = Path(ml_cfg["training_csv"])
    if not csv_path.is_absolute():
        csv_path = code_dir / csv_path
    manifest_path = Path(ml_cfg["dataset_manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = code_dir / manifest_path
    dataset_manifest = load_dataset_manifest(manifest_path, csv_path)
    split_manifest_path = Path(ml_cfg["split_manifest_path"])
    if not split_manifest_path.is_absolute():
        split_manifest_path = code_dir / split_manifest_path
    split_manifest_metadata_path = split_manifest_path.with_suffix(
        split_manifest_path.suffix + ".manifest.json"
    )
    split_manifest_metadata = load_split_manifest_metadata(
        split_manifest_metadata_path, split_manifest_path, csv_path, manifest_path
    )
    data, partitions = load_grouped_data(csv_path, split_manifest_path)
    observed_counts = {
        str(key): int(value)
        for key, value in data[TARGET_NAME].value_counts().sort_index().items()
    }
    if len(data) != dataset_manifest["row_count"] or observed_counts != {
        str(key): int(value) for key, value in dataset_manifest["target_counts"].items()
    }:
        raise ValueError("Dataset contents disagree with manifest row/class counts")
    if set(data["experiment_id"].astype(str)) != {
        str(dataset_manifest["experiment_id"])
    }:
        raise ValueError("Dataset experiment_id disagrees with its manifest")
    split_membership_sha256 = _split_membership_hash(data)
    calibration_selection_path = Path(ml_cfg["calibration_selection_record_path"])
    if not calibration_selection_path.is_absolute():
        calibration_selection_path = code_dir / calibration_selection_path
    calibration_selection_record = load_calibration_selection_record(
        calibration_selection_path,
        configured_method=str(ml_cfg["calibration_method"]),
        csv_path=csv_path,
        split_manifest_path=split_manifest_path,
        split_membership_sha256=split_membership_sha256,
        experiment_id=str(dataset_manifest["experiment_id"]),
    )
    x_train, y_train = partitions["train"]
    x_validation, y_validation = partitions["validation"]
    x_calibration, y_calibration = partitions["calibration"]
    x_test, y_test = partitions["test"]

    output_dir = code_dir / "models"
    final_dir = output_dir / version
    _require_new_paths([final_dir])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name=f"rf_fire_{ml_cfg['artifact_version']}",
    )
    study.optimize(
        create_objective(
            x_train,
            y_train,
            x_validation,
            y_validation,
            class_weight_candidates,
        ),
        n_trials=int(ml_cfg.get("n_trials", 100)),
        show_progress_bar=True,
    )
    model = fit_calibrated_model(
        study.best_params,
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_calibration,
        y_calibration,
        str(ml_cfg["calibration_method"]),
    )
    probabilities = predict_positive_probability(model, x_test)
    metrics = evaluate_estimator_probabilities(y_test.to_numpy(), probabilities)

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{version}.staging-", dir=output_dir))
    try:
        staged_model = staging_dir / f"fire_rf_model_{version}.joblib"
        staged_study = staging_dir / f"optuna_study_rf_{version}.joblib"
        staged_summary = staging_dir / f"rf_provenance_{version}.json"
        final_model = final_dir / staged_model.name
        final_study = final_dir / staged_study.name
        joblib.dump(model, staged_model)
        joblib.dump(study, staged_study)
        summary = build_summary(
            data=data,
            csv_path=csv_path,
            config_path=config_path,
            source_revision=str(ml_cfg["source_revision"]),
            source_dirty_state=str(ml_cfg["source_dirty_state"]),
            artifact_version=version,
            calibration_method=str(ml_cfg["calibration_method"]),
            dataset_manifest_path=manifest_path,
            dataset_manifest=dataset_manifest,
            split_manifest_path=split_manifest_path,
            split_manifest_metadata_path=split_manifest_metadata_path,
            split_manifest_metadata=split_manifest_metadata,
            calibration_selection_path=calibration_selection_path,
            calibration_selection_record=calibration_selection_record,
            training_command=str(ml_cfg["training_command"]),
            environment_record=str(ml_cfg["environment_record"]),
            class_weight_candidates=class_weight_candidates,
            best_params=study.best_params,
            best_score=study.best_value,
            metrics=metrics,
            model=model,
        )
        summary.update(
            model_path=str(final_model),
            model_sha256=_file_sha256(staged_model),
            study_path=str(final_study),
            study_sha256=_file_sha256(staged_study),
        )
        staged_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if final_dir.exists():
            raise FileExistsError(f"Refusing to overwrite immutable artifact set: {final_dir}")
        staging_dir.rename(final_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


if __name__ == "__main__":
    main()
