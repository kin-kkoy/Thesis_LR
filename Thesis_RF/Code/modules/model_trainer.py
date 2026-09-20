"""Train and evaluate RF models under the approved grouped-split contract."""

from __future__ import annotations

from hashlib import sha256
import json
import pickle
from pathlib import Path
import re
import shutil
import tempfile
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    jaccard_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .feature_pipeline import (
    CANONICAL_FEATURE_NAMES,
    MODEL_MANIFEST_VERSION,
    OBSERVATION_METADATA_NAMES,
    POSITIVE_LABEL,
    PROVENANCE_METADATA_NAMES,
    SET_C_SCHEMA_VERSION,
    SPLIT_ROLES,
    SPLIT_ASSIGNMENT_KEY_NAMES,
    TARGET_NAME,
    TARGET_VERSION,
    positive_class_index,
    predict_positive_probability,
    validate_feature_columns,
    validate_feature_values,
    validate_model_feature_schema,
)


SUPPORTED_CLASS_WEIGHT_STRATEGIES = ("none", "balanced", "balanced_subsample")


def class_weight_for_estimator(configured: str | None) -> str | None:
    """Map an approved explicit token to the scikit-learn parameter value."""
    if configured not in SUPPORTED_CLASS_WEIGHT_STRATEGIES:
        raise ValueError(
            "class_weight_strategy must be explicitly 'none', 'balanced', or "
            "'balanced_subsample'"
        )
    return None if configured == "none" else configured


class ModelTrainer:
    """Consume an explicitly grouped dataset without constructing row-level splits."""

    SUPPORTED_MODEL_EXTENSIONS = {".joblib", ".pkl"}
    GROUP_FIELDS = ("scenario_id", "spatial_block_id", "duplicate_group_id")

    def __init__(
        self,
        csv_path: str,
        split_manifest_path: str,
        output_dir: str,
        seed: int = 42,
    ):
        self.csv_path = str(csv_path)
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        observations = pd.read_csv(self.csv_path)
        assignments = pd.read_csv(split_manifest_path)
        data = self.merge_split_assignments(observations, assignments)
        self._validate_dataset_contract(data)
        self.data = data
        self.feature_names = list(CANONICAL_FEATURE_NAMES)
        self.model: object | None = None
        self.model_name: str | None = None

        self.X_train, self.y_train = self.get_partition("train")
        self.X_validation, self.y_validation = self.get_partition("validation")
        self.X_calibration, self.y_calibration = self.get_partition("calibration")
        self.X_test, self.y_test = self.get_partition("test")

    @staticmethod
    def merge_split_assignments(
        observations: pd.DataFrame,
        assignments: pd.DataFrame,
    ) -> pd.DataFrame:
        """Join a separate immutable split manifest to immutable observations."""
        if "split_role" in observations.columns:
            raise ValueError("Observation dataset must not embed split_role")
        observation_required = (
            set(OBSERVATION_METADATA_NAMES)
            | set(CANONICAL_FEATURE_NAMES)
            | {TARGET_NAME, "feature_schema_version", "target_version"}
        )
        missing_observations = sorted(observation_required.difference(observations.columns))
        if missing_observations:
            raise ValueError(f"Observation dataset is missing columns: {missing_observations}")
        assignment_required = set(SPLIT_ASSIGNMENT_KEY_NAMES) | {"split_role"}
        missing_assignments = sorted(assignment_required.difference(assignments.columns))
        if missing_assignments:
            raise ValueError(f"Split manifest is missing columns: {missing_assignments}")
        keys = list(SPLIT_ASSIGNMENT_KEY_NAMES)
        if observations.duplicated(keys).any():
            raise ValueError("Observation identity keys are not unique")
        if assignments.duplicated(keys).any():
            raise ValueError("Split-manifest identity keys are not unique")
        key_membership = observations.loc[:, keys].merge(
            assignments.loc[:, keys],
            on=keys,
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        if not key_membership["_merge"].eq("both").all():
            observation_only = int(key_membership["_merge"].eq("left_only").sum())
            assignment_only = int(key_membership["_merge"].eq("right_only").sum())
            raise ValueError(
                "Split manifest and observation identities must match exactly "
                f"(observation_only={observation_only}, assignment_only={assignment_only})"
            )
        merged = observations.merge(
            assignments.loc[:, keys + ["split_role"]],
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(observations) or merged["split_role"].isna().any():
            raise ValueError("Split manifest does not assign every observation exactly once")
        return merged

    @classmethod
    def _validate_dataset_contract(cls, data: pd.DataFrame) -> None:
        required = set(PROVENANCE_METADATA_NAMES) | set(CANONICAL_FEATURE_NAMES) | {
            TARGET_NAME,
            "feature_schema_version",
            "target_version",
        }
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"Dataset is missing contract columns: {', '.join(missing)}")
        predictor_set = set(CANONICAL_FEATURE_NAMES)
        ordered_predictors = [name for name in data.columns if name in predictor_set]
        validate_feature_columns(ordered_predictors)
        validate_feature_values(data.loc[:, list(CANONICAL_FEATURE_NAMES)].to_numpy())

        if data.empty:
            raise ValueError("Training dataset is empty")
        if set(data["set_id"].astype(str)) != {"SetC"}:
            raise ValueError("Training data must belong only to the SetC provenance chain")
        if set(data["feature_schema_version"].astype(str)) != {SET_C_SCHEMA_VERSION}:
            raise ValueError("Dataset feature_schema_version does not match Set C")
        if set(data["target_version"].astype(str)) != {TARGET_VERSION}:
            raise ValueError("Dataset target_version does not match next-timestep ignition")
        target = data[TARGET_NAME]
        if (
            target.isna().any()
            or not pd.api.types.is_integer_dtype(target.dtype)
            or not target.isin((0, POSITIVE_LABEL)).all()
        ):
            raise ValueError(f"{TARGET_NAME} must contain only integer labels 0 and 1")

        roles = data["split_role"].astype(str)
        invalid_roles = sorted(set(roles).difference(SPLIT_ROLES))
        if invalid_roles:
            raise ValueError(f"Invalid or unassigned split_role values: {invalid_roles}")
        missing_roles = [role for role in SPLIT_ROLES if role not in set(roles)]
        if missing_roles:
            raise ValueError(f"Dataset is missing required split roles: {missing_roles}")
        for role in SPLIT_ROLES:
            role_classes = set(data.loc[roles.eq(role), TARGET_NAME].astype(int))
            if role_classes != {0, POSITIVE_LABEL}:
                raise ValueError(
                    f"Split role {role!r} must contain both target classes 0 and 1"
                )

        metadata = data.loc[:, list(PROVENANCE_METADATA_NAMES)]
        if metadata.isna().any().any():
            raise ValueError("Grouping/provenance metadata cannot contain missing values")
        empty_text = metadata.select_dtypes(include=["object", "string"]).apply(
            lambda column: column.astype(str).str.strip().eq("").any()
        )
        if bool(empty_text.any()):
            raise ValueError("Grouping/provenance metadata cannot contain empty values")
        for field in ("seed", "timestep_t", "cell_row", "cell_col"):
            if not pd.api.types.is_integer_dtype(data[field].dtype):
                raise ValueError(f"Grouping/provenance field {field!r} must be integer")

        for field in cls.GROUP_FIELDS:
            role_counts = data.groupby(field, dropna=False)["split_role"].nunique()
            if bool((role_counts > 1).any()):
                raise ValueError(f"{field} crosses split roles")

        conflicting_duplicates = data.groupby(
            "duplicate_group_id", dropna=False
        )[TARGET_NAME].nunique()
        if bool((conflicting_duplicates > 1).any()):
            raise ValueError(
                "Conflicting-label duplicate feature groups require investigation"
            )

        cell_roles = data.groupby(
            ["grid_id", "cell_row", "cell_col"], dropna=False
        )["split_role"].nunique()
        if bool((cell_roles > 1).any()):
            raise ValueError("A spatial cell crosses split roles")

        if data["event_id"].nunique(dropna=False) > 1:
            event_roles = data.groupby("event_id", dropna=False)["split_role"].nunique()
            if bool((event_roles > 1).any()):
                raise ValueError("event_id crosses split roles")

    def get_partition(self, role: str) -> tuple[pd.DataFrame, pd.Series]:
        if role not in SPLIT_ROLES:
            raise ValueError(f"Unknown split role: {role}")
        partition = self.data[self.data["split_role"] == role]
        if partition.empty:
            raise ValueError(f"Split role {role!r} is empty")
        x = partition.loc[:, list(CANONICAL_FEATURE_NAMES)].copy()
        y = partition[TARGET_NAME].astype(int).copy()
        return x, y

    def train_model(
        self,
        algorithm: str = "random_forest",
        class_weight_strategy: Literal[
            "none", "balanced", "balanced_subsample"
        ]
        | None = None,
        **model_kwargs,
    ) -> object:
        algorithm_name = str(algorithm).strip().lower()
        class_weight = class_weight_for_estimator(class_weight_strategy)
        if algorithm_name != "random_forest":
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        params = {
            "n_estimators": 200,
            "max_depth": 15,
            "random_state": self.seed,
            "class_weight": class_weight,
            "n_jobs": -1,
        }
        params.update(model_kwargs)
        model = RandomForestClassifier(**params)
        model.fit(self.X_train, self.y_train)
        self._validate_model_contract(model, context="Trained model")
        self.model = model
        self.model_name = algorithm_name
        return model

    def train_random_forest(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 15,
        class_weight_strategy: Literal[
            "none", "balanced", "balanced_subsample"
        ]
        | None = None,
    ) -> object:
        return self.train_model(
            algorithm="random_forest",
            class_weight_strategy=class_weight_strategy,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )

    def set_model(self, model: object, model_name: str = "calibrated_model") -> None:
        """Attach a fitted model or calibrator after external non-test calibration."""
        self._validate_model_contract(model, context="Configured model")
        self.model = model
        self.model_name = str(model_name)

    def evaluate(
        self,
        split_role: Literal["validation", "calibration", "test"] = "test",
        threshold_record: dict | None = None,
    ) -> dict:
        """Evaluate without selecting a threshold on the evaluated partition."""
        if self.model is None:
            raise RuntimeError("No trained model found. Call train_model() or set_model() first.")
        x, y = self.get_partition(split_role)
        y_proba = predict_positive_probability(self.model, x)
        if np.unique(y).size != 2:
            raise ValueError(f"{split_role} evaluation requires both target classes")

        metrics = {
            "split_role": split_role,
            "observation_count": int(len(y)),
            "roc_auc": float(roc_auc_score(y, y_proba)),
            "pr_auc": float(average_precision_score(y, y_proba)),
            "brier_score": float(brier_score_loss(y, y_proba, pos_label=POSITIVE_LABEL)),
            "log_loss": float(log_loss(y, y_proba, labels=[0, POSITIVE_LABEL])),
        }
        if threshold_record is None:
            return metrics
        threshold = self._validate_threshold_record(threshold_record)
        y_pred = (y_proba >= threshold).astype(int)
        metrics.update(
            classification_threshold=threshold,
            confusion_matrix=confusion_matrix(y, y_pred, labels=[0, 1]).tolist(),
            precision=float(precision_score(y, y_pred, zero_division=0)),
            recall=float(recall_score(y, y_pred, zero_division=0)),
            f1=float(f1_score(y, y_pred, zero_division=0)),
            jaccard=float(jaccard_score(y, y_pred, zero_division=0)),
        )
        return metrics

    @staticmethod
    def _validate_threshold_record(record: dict) -> float:
        if record.get("selected_on") != "calibration":
            raise ValueError("Classification threshold must be selected on calibration data")
        if not str(record.get("objective", "")).strip():
            raise ValueError("Classification threshold requires an approved objective record")
        if record.get("value") is None:
            raise ValueError("Classification threshold record is missing value")
        threshold = float(record["value"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("classification_threshold must be within [0, 1]")
        return threshold

    def save_model(
        self,
        filename: str,
        artifact_version: str,
        provenance_manifest: dict,
    ) -> str:
        if self.model is None:
            raise RuntimeError("No trained model found. Call train_model() or set_model() first.")
        self._validate_model_contract(self.model, context="Model to export")
        if Path(filename).name != filename:
            raise ValueError("Model filename must not contain directory components")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", artifact_version):
            raise ValueError("artifact_version must be a safe, path-free identifier")
        final_dir = self.output_dir / artifact_version
        model_path = final_dir / filename
        self._validate_model_path(model_path)
        if not artifact_version or artifact_version not in model_path.stem:
            raise ValueError("Versioned model filename must include artifact_version")
        required_manifest = {
            "manifest_schema_version",
            "artifact_kind",
            "set_id",
            "experiment_id",
            "dataset_sha256",
            "dataset_manifest_sha256",
            "split_assignment_sha256",
            "split_membership_sha256",
            "calibration_selection_record_sha256",
            "source_revision",
            "source_dirty_state",
            "calibration_method",
            "feature_schema_version",
            "target_version",
            "positive_label",
        }
        missing = sorted(required_manifest.difference(provenance_manifest))
        if missing:
            raise ValueError(f"Model provenance manifest is incomplete: {missing}")
        if (
            provenance_manifest["manifest_schema_version"] != MODEL_MANIFEST_VERSION
            or provenance_manifest["artifact_kind"] != "set_c_rf_model"
            or provenance_manifest["set_id"] != "SetC"
            or provenance_manifest["feature_schema_version"] != SET_C_SCHEMA_VERSION
            or provenance_manifest["target_version"] != TARGET_VERSION
            or provenance_manifest["positive_label"] != POSITIVE_LABEL
        ):
            raise ValueError("Model provenance manifest violates the Set C contract")
        if final_dir.exists():
            raise FileExistsError(f"Refusing to overwrite immutable artifact set: {final_dir}")
        manifest_path = model_path.with_suffix(model_path.suffix + ".manifest.json")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{artifact_version}.staging-", dir=self.output_dir)
        )
        try:
            staged_model = staging_dir / filename
            staged_manifest = staged_model.with_suffix(staged_model.suffix + ".manifest.json")
            self._save_serialized_model(self.model, staged_model)
            digest = sha256()
            with staged_model.open("rb") as file_obj:
                for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                    digest.update(chunk)
            complete_manifest = dict(provenance_manifest)
            complete_manifest.update(
                artifact_version=artifact_version,
                model_path=str(model_path),
                model_sha256=digest.hexdigest(),
                feature_names=list(CANONICAL_FEATURE_NAMES),
                positive_label=POSITIVE_LABEL,
                model_classes=[int(value) for value in getattr(self.model, "classes_")],
            )
            staged_manifest.write_text(
                json.dumps(complete_manifest, indent=2), encoding="utf-8"
            )
            if final_dir.exists():
                raise FileExistsError(
                    f"Refusing to overwrite immutable artifact set: {final_dir}"
                )
            staging_dir.rename(final_dir)
        except Exception:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            raise
        return str(model_path.resolve())

    @classmethod
    def _validate_model_path(cls, model_path: Path) -> None:
        if model_path.suffix.lower() not in cls.SUPPORTED_MODEL_EXTENSIONS:
            supported = ", ".join(sorted(cls.SUPPORTED_MODEL_EXTENSIONS))
            raise ValueError(f"Model file must use a supported extension: {supported}")

    @staticmethod
    def _save_serialized_model(model: object, model_path: Path) -> None:
        if model_path.suffix.lower() == ".joblib":
            joblib.dump(model, model_path)
            return
        with model_path.open("wb") as file_obj:
            pickle.dump(model, file_obj)

    @staticmethod
    def _validate_model_contract(model: object, context: str) -> None:
        predict_proba = getattr(model, "predict_proba", None)
        if predict_proba is None or not callable(predict_proba):
            raise TypeError(f"{context} must provide a callable predict_proba() method")
        validate_model_feature_schema(model)
        classes = getattr(model, "classes_", None)
        if classes is None:
            raise ValueError(f"{context} must expose classes_")
        positive_class_index(classes)

    def feature_importance_report(self) -> None:
        if self.model is None:
            raise RuntimeError("No trained model found")
        if not hasattr(self.model, "feature_importances_"):
            raise TypeError("Current model does not expose feature_importances_")
        pairs = sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda pair: pair[1],
            reverse=True,
        )
        print("Feature Importances:")
        for feature_name, score in pairs:
            print(f"{feature_name:<30}{score:.6f}")
