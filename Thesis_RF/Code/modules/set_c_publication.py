"""Disabled-by-default, bounded Set C package staging and publication."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from .ca_state import STATE_ENCODING
from .feature_pipeline import (
    AUTHORITATIVE_CSV_COLUMNS,
    CANONICAL_FEATURE_NAMES,
    POSITIVE_LABEL,
    SET_C_SCHEMA_VERSION,
    TARGET_NAME,
    TARGET_VERSION,
    validate_feature_values,
)
from .set_c_collector import SET_C_ROW_BATCH_SCHEMA_VERSION
from .transition_observer import (
    AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
    TRANSITION_OBSERVATION_SCHEMA_VERSION,
    canonical_metadata_hash,
)


PACKAGE_SCHEMA_VERSION = "set_c_authoritative_package.v1"
RUN_MANIFEST_VERSION = "set_c_authoritative_run_manifest.v1"
BATCH_MANIFEST_VERSION = "set_c_authoritative_batch_manifest.v1"
PACKAGE_VERSION = "1"
DESTINATION_PATTERN = "output/datasets/{package_id}/v{package_version}"
AUTHORITATIVE_VERSIONED_MODE = "authoritative_versioned"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Unsupported manifest value: {type(value).__name__}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_cell_id(grid_id: str, row: int, col: int) -> str:
    return sha256(f"stable_cell.v1|{grid_id}|{row}|{col}".encode()).hexdigest()


def validate_published_package(manifest_path: Path) -> dict[str, object]:
    """Revalidate every immutable package layer before development loading."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_payload_hash = manifest.get("manifest_payload_sha256")
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    if recorded_payload_hash != sha256(_canonical_bytes(payload)).hexdigest():
        raise ValueError("Package manifest payload hash mismatch")
    dataset = manifest_path.parent / str(manifest.get("dataset_path", ""))
    if not dataset.is_file() or manifest.get("dataset_sha256") != _file_sha256(dataset):
        raise ValueError("Package dataset hash mismatch")
    total_rows = 0
    totals = {"0": 0, "1": 0}
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Package must contain run manifests")
    for entry in runs:
        run_hash = str(entry.get("run_identity_sha256", ""))
        run_path = manifest_path.parent / "runs" / run_hash / "run.manifest.json"
        if not run_path.is_file() or entry.get("manifest_sha256") != _file_sha256(run_path):
            raise ValueError("Run manifest hash mismatch")
        if json.loads(run_path.read_text(encoding="utf-8")) != entry.get("content"):
            raise ValueError("Run manifest content mismatch")
        run = entry["content"]
        total_rows += int(run["row_count"])
        for label in totals:
            totals[label] += int(run["target_counts"][label])
        for batch in run["batches"]:
            batch_path = manifest_path.parent / batch["path"]
            batch_manifest = batch_path.with_suffix(".manifest.json")
            if (
                not batch_path.is_file()
                or batch.get("sha256") != _file_sha256(batch_path)
                or not batch_manifest.is_file()
                or batch.get("manifest_sha256") != _file_sha256(batch_manifest)
            ):
                raise ValueError("Batch artifact hash mismatch")
    if total_rows != manifest.get("row_count") or totals != manifest.get("target_counts"):
        raise ValueError("Package run totals do not match its manifest")
    return manifest


def _text(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Publication identity {key!r} is required")
    return value.strip()


def _digest(mapping: Mapping[str, object], key: str) -> str:
    value = _text(mapping, key)
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"Publication identity {key!r} must be lowercase SHA-256")
    return value


class SetCPublicationSession:
    """Stream complete batches to per-run staging, then atomically promote once."""

    def __init__(
        self,
        config: Mapping[str, object],
        package_identity: Mapping[str, object],
        *,
        destination_root: Path | None = None,
    ) -> None:
        dataset = config.get("dataset_generation")
        publisher = dataset.get("set_c_publisher") if isinstance(dataset, Mapping) else None
        if not isinstance(publisher, Mapping):
            raise ValueError("dataset_generation.set_c_publisher is required")
        if publisher.get("enabled") is not True:
            raise PermissionError("Authoritative Set C publication is disabled by default")
        if publisher.get("mode") != AUTHORITATIVE_VERSIONED_MODE:
            raise PermissionError("Publisher mode must be authoritative_versioned")
        if publisher.get("destination_pattern") != DESTINATION_PATTERN:
            raise ValueError("Publisher destination pattern is not canonical")
        if not isinstance(publisher.get("authorization_reference"), str) or not str(publisher["authorization_reference"]).strip():
            raise PermissionError("Publication requires an explicit authorization reference")
        if destination_root is None:
            raise PermissionError("Publication requires an explicit destination root")

        self.package_id = _text(publisher, "package_id")
        self.package_version = _text(publisher, "package_version")
        if _SAFE_ID.fullmatch(self.package_id) is None or _SAFE_ID.fullmatch(self.package_version) is None:
            raise ValueError("Package identity must be path-free")
        if self.package_version != PACKAGE_VERSION:
            raise ValueError(f"Only package version {PACKAGE_VERSION} is implemented")
        self.identity = dict(package_identity)
        self._validate_package_identity(self.identity)
        self.final_path = Path(destination_root) / self.package_id / f"v{self.package_version}"
        if self.final_path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable package: {self.final_path}")
        parent = self.final_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        self.staging_path = Path(tempfile.mkdtemp(prefix=".set-c-staging-", dir=parent))
        self._runs: dict[str, dict[str, object]] = {}
        self._active_run_hash: str | None = None
        self._published = False

    @staticmethod
    def _validate_package_identity(identity: Mapping[str, object]) -> None:
        for key in ("experiment_id", "source_revision", "environment_id", "rng_algorithm", "teacher_id", "teacher_version", "authorization_reference"):
            _text(identity, key)
        if not isinstance(identity.get("source_dirty"), bool):
            raise ValueError("source_dirty must be boolean")
        for key in ("configuration_sha256", "environment_sha256", "simulation_valid_mask_sha256", "mapped_building_mask_sha256", "feature_schema_sha256", "target_schema_sha256"):
            _digest(identity, key)
        input_hashes = identity.get("input_hashes")
        if not isinstance(input_hashes, Mapping) or not input_hashes:
            raise ValueError("input_hashes must be a non-empty mapping")
        for name, value in input_hashes.items():
            if not str(name).strip() or not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError("input_hashes must contain named lowercase SHA-256 values")

    def _validate_run_identity(self, identity: Mapping[str, object]) -> None:
        expected = {
            "set_id": "SetC",
            "experiment_id": self.identity["experiment_id"],
            "source_revision": self.identity["source_revision"],
            "source_dirty": self.identity["source_dirty"],
            "configuration_hash": self.identity["configuration_sha256"],
            "environment_id": self.identity["environment_id"],
            "rng_bit_generator": self.identity["rng_algorithm"],
            "input_hashes": self.identity["input_hashes"],
            "simulation_valid_mask_sha256": self.identity["simulation_valid_mask_sha256"],
            "mapped_building_mask_sha256": self.identity["mapped_building_mask_sha256"],
            "simulator_id": self.identity["teacher_id"],
            "simulator_version": self.identity["teacher_version"],
            "authorization_reference": self.identity["authorization_reference"],
        }
        for key, value in expected.items():
            if identity.get(key) != value:
                raise PermissionError(f"Run identity disagrees with package identity: {key}")
        for key in (
            "event_id", "scenario_family_id", "scenario_id", "run_id", "grid_id",
            "wind_manifest", "ignition_coordinates",
        ):
            if key not in identity or identity[key] in (None, "", (), []):
                raise PermissionError(f"Run identity is incomplete: {key}")
        _digest(identity, "ignition_set_sha256")

    def __enter__(self) -> "SetCPublicationSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self._published:
            self.abort()

    def abort(self) -> None:
        if self.staging_path.exists():
            shutil.rmtree(self.staging_path)

    @staticmethod
    def _validate_batch(batch: object) -> int:
        features = getattr(batch, "features")
        labels = getattr(batch, "labels")
        rows = getattr(batch, "cell_rows")
        cols = getattr(batch, "cell_cols")
        duplicates = getattr(batch, "duplicate_group_ids")
        blocks = getattr(batch, "spatial_block_ids")
        if tuple(getattr(batch, "feature_names")) != CANONICAL_FEATURE_NAMES:
            raise ValueError("Batch feature order is not set_c.v1")
        if (
            getattr(batch, "schema_version") != SET_C_ROW_BATCH_SCHEMA_VERSION
            or getattr(batch, "target_name") != TARGET_NAME
            or getattr(batch, "positive_label") != POSITIVE_LABEL
            or getattr(batch, "state_encoding") != STATE_ENCODING
            or getattr(batch, "source_observation_schema_version")
            != TRANSITION_OBSERVATION_SCHEMA_VERSION
            or getattr(batch, "transition_source_type")
            != AUTHORIZED_SIMULATED_TRANSITION_SOURCE
        ):
            raise ValueError("Batch source, state, feature, or target contract is invalid")
        if features.dtype != np.dtype(np.float32) or features.ndim != 2 or features.shape[1] != len(CANONICAL_FEATURE_NAMES):
            raise TypeError("Batch predictors must remain a two-dimensional np.float32 matrix")
        validate_feature_values(features)
        count = int(features.shape[0])
        expected = ((labels, np.int8), (rows, np.int64), (cols, np.int64), (duplicates, "S64"), (blocks, "S64"))
        for values, dtype in expected:
            if values.dtype != np.dtype(dtype) or values.shape != (count,):
                raise TypeError("Batch payload dtype or row count violates the Set C contract")
        if not np.isin(labels, (0, POSITIVE_LABEL)).all():
            raise ValueError("Batch labels must be np.int8 values 0 or 1")
        if int(getattr(batch, "timestep_t1")) != int(getattr(batch, "timestep_t")) + 1:
            raise ValueError("Batch timesteps must be adjacent")
        if getattr(batch, "feature_schema_version") != SET_C_SCHEMA_VERSION or getattr(batch, "target_version") != TARGET_VERSION:
            raise ValueError("Batch schema version is not authoritative Set C")
        return count

    def stage_batch(self, batch: object) -> Path:
        count = self._validate_batch(batch)
        run_hash = str(getattr(batch, "run_identity_sha256"))
        provenance_hash = str(getattr(batch, "provenance_sha256"))
        if _SHA256.fullmatch(run_hash) is None or _SHA256.fullmatch(provenance_hash) is None:
            raise ValueError("Batch run/provenance hashes are invalid")
        if self._active_run_hash not in (None, run_hash):
            raise PermissionError("Model-free teacher runs must be staged sequentially")
        self._active_run_hash = run_hash
        identity = dict(getattr(batch, "provenance"))
        run = self._runs.setdefault(
            run_hash,
            {
                "batches": [],
                "rows": 0,
                "counts": {"0": 0, "1": 0},
                "sealed": False,
                "provenance_sha256": provenance_hash,
                "provenance": identity,
            },
        )
        if (
            run["provenance_sha256"] != provenance_hash
            or run["provenance"] != identity
        ):
            raise PermissionError("Mixed batch provenance within one run is prohibited")
        if run["sealed"]:
            raise RuntimeError("Cannot append to a sealed run")
        if canonical_metadata_hash(identity) != provenance_hash:
            raise ValueError("Batch provenance hash mismatch")
        if _text(identity, "experiment_id") != self.identity["experiment_id"]:
            raise ValueError("Batch belongs to another experiment")
        if dict(getattr(batch, "wind_manifest")) != identity["wind_manifest"]:
            raise ValueError("Batch wind identity is inconsistent")
        masks = dict(getattr(batch, "domain_mask_hashes"))
        if masks != {
            "simulation_valid_mask_sha256": self.identity["simulation_valid_mask_sha256"],
            "mapped_building_mask_sha256": self.identity["mapped_building_mask_sha256"],
        }:
            raise ValueError("Batch domain-mask identity is inconsistent")
        run_dir = self.staging_path / "runs" / run_hash / "batches"
        run_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(getattr(batch, 'observation_index')):06d}-{int(getattr(batch, 'batch_index')):06d}.csv"
        path = run_dir / name
        if path.exists():
            raise FileExistsError(f"Duplicate staged batch: {name}")

        grid_id = _text(identity, "grid_id")
        values = zip(
            getattr(batch, "cell_rows"), getattr(batch, "cell_cols"),
            getattr(batch, "duplicate_group_ids"), getattr(batch, "spatial_block_ids"),
            getattr(batch, "features"), getattr(batch, "labels"), strict=True,
        )
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(AUTHORITATIVE_CSV_COLUMNS)
            for row, col, duplicate, block, features, label in values:
                metadata = (
                    "SetC", _text(identity, "experiment_id"), _text(identity, "event_id"),
                    _text(identity, "scenario_family_id"), _text(identity, "scenario_id"),
                    _text(identity, "run_id"), int(getattr(batch, "seed")),
                    int(getattr(batch, "timestep_t")), int(getattr(batch, "timestep_t1")),
                    int(row), int(col), grid_id, _stable_cell_id(grid_id, int(row), int(col)),
                    duplicate.decode("ascii"), block.decode("ascii"),
                    _digest(identity, "ignition_set_sha256"), run_hash, provenance_hash,
                )
                writer.writerow((*metadata, SET_C_SCHEMA_VERSION, TARGET_VERSION, *map(float, features), int(label)))
        entry = {
            "manifest_schema_version": BATCH_MANIFEST_VERSION,
            "path": str(path.relative_to(self.staging_path)).replace("\\", "/"),
            "sha256": _file_sha256(path),
            "row_count": count,
            "target_counts": {str(label): int(np.count_nonzero(getattr(batch, "labels") == label)) for label in (0, 1)},
            "observation_index": int(getattr(batch, "observation_index")),
            "batch_index": int(getattr(batch, "batch_index")),
            "timestep_t": int(getattr(batch, "timestep_t")),
            "timestep_t1": int(getattr(batch, "timestep_t1")),
        }
        entry_path = path.with_suffix(".manifest.json")
        entry_path.write_bytes(_canonical_bytes(entry))
        entry["manifest_sha256"] = _file_sha256(entry_path)
        run["batches"].append(entry)
        run["rows"] += count
        for label in ("0", "1"):
            run["counts"][label] += entry["target_counts"][label]
        return path

    def seal_run(self, collected_run: object) -> Path:
        run_hash = str(getattr(collected_run, "run_identity_sha256"))
        run = self._runs.get(run_hash)
        if run is None or run["sealed"]:
            raise ValueError("Run has no unsealed staged batches")
        record = getattr(collected_run, "termination_record")
        termination = asdict(record) if is_dataclass(record) else dict(vars(record))
        if (
            termination.get("authoritative") is not True
            or termination.get("completeness_status") != "complete"
            or termination.get("termination_reason") != "inactive"
            or int(termination.get("final_ignited_count", -1)) != 0
            or int(termination.get("final_blazing_count", -1)) != 0
        ):
            raise PermissionError("Failed, censored, or partial runs cannot be promoted")
        if int(termination.get("expected_transition_count", -1)) != int(termination.get("captured_transition_count", -2)):
            raise PermissionError("Count-mismatched runs cannot be promoted")
        if int(getattr(collected_run, "row_count")) != run["rows"]:
            raise PermissionError("Staged row count does not match the complete run")
        if int(getattr(collected_run, "released_row_count")) != run["rows"] or tuple(getattr(collected_run, "batches")):
            raise PermissionError("Production staging requires bounded draining of every complete batch")
        run_identity = dict(getattr(collected_run, "run_identity"))
        run_provenance = dict(getattr(collected_run, "provenance"))
        self._validate_run_identity(run_identity)
        if canonical_metadata_hash(run_identity) != run_hash:
            raise PermissionError("Sealed run identity hash mismatch")
        if (
            str(getattr(collected_run, "provenance_sha256"))
            != run["provenance_sha256"]
            or canonical_metadata_hash(run_provenance) != run["provenance_sha256"]
            or run_provenance != run["provenance"]
        ):
            raise PermissionError("Sealed run provenance does not match staged batches")
        manifest = {
            "manifest_schema_version": RUN_MANIFEST_VERSION,
            "run_identity_sha256": run_hash,
            "provenance_sha256": str(getattr(collected_run, "provenance_sha256")),
            "run_identity": run_identity,
            "row_count": run["rows"],
            "target_counts": run["counts"],
            "batch_count": len(run["batches"]),
            "batches": run["batches"],
            "termination_record": termination,
        }
        path = self.staging_path / "runs" / run_hash / "run.manifest.json"
        path.write_bytes(_canonical_bytes(manifest))
        run.update(sealed=True, manifest=manifest, manifest_sha256=_file_sha256(path))
        self._active_run_hash = None
        return path

    def publish(self, *, expected_run_count: int) -> Path:
        try:
            if expected_run_count <= 0 or len(self._runs) != expected_run_count:
                raise PermissionError("Package run count does not match the approved expectation")
            if any(not run["sealed"] for run in self._runs.values()):
                raise PermissionError("Every run must be sealed before publication")
            dataset_path = self.staging_path / "set_c.v1.csv"
            totals = {"0": 0, "1": 0}
            row_count = 0
            with dataset_path.open("x", newline="", encoding="utf-8") as output:
                for run_hash in sorted(self._runs):
                    for batch in sorted(self._runs[run_hash]["batches"], key=lambda item: (item["observation_index"], item["batch_index"])):
                        source = self.staging_path / batch["path"]
                        if _file_sha256(source) != batch["sha256"]:
                            raise PermissionError("Staged batch hash mismatch")
                        with source.open("r", newline="", encoding="utf-8") as handle:
                            header = handle.readline()
                            if output.tell() == 0:
                                output.write(header)
                            output.writelines(handle)
                        row_count += batch["row_count"]
                        for label in totals:
                            totals[label] += batch["target_counts"][label]
            run_manifests = [
                {"run_identity_sha256": key, "manifest_sha256": self._runs[key]["manifest_sha256"], "content": self._runs[key]["manifest"]}
                for key in sorted(self._runs)
            ]
            manifest = {
                "manifest_schema_version": PACKAGE_SCHEMA_VERSION,
                "artifact_kind": "authoritative_set_c_observation_package",
                "package_id": self.package_id,
                "package_version": self.package_version,
                "destination_pattern": DESTINATION_PATTERN,
                "canonical_csv_columns": list(AUTHORITATIVE_CSV_COLUMNS),
                "predictor_dtype": "float32",
                "label_dtype": "int8",
                "state_dtype": "int8",
                "state_encoding": STATE_ENCODING,
                "sampling_policy": "all_eligible",
                "split_status": "unassigned",
                "row_count": row_count,
                "target_counts": totals,
                "dataset_path": dataset_path.name,
                "dataset_sha256": _file_sha256(dataset_path),
                "identity": self.identity,
                "runs": run_manifests,
            }
            manifest["manifest_payload_sha256"] = sha256(_canonical_bytes(manifest)).hexdigest()
            manifest_path = self.staging_path / "manifest.json"
            manifest_path.write_bytes(_canonical_bytes(manifest))
            if self.final_path.exists():
                raise FileExistsError(f"Refusing to overwrite immutable package: {self.final_path}")
            os.rename(self.staging_path, self.final_path)
            self._published = True
            return self.final_path
        except Exception:
            self.abort()
            raise
