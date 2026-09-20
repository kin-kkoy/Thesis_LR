"""Build Set C next-timestep ignition rows from simulated adjacent states.

Running this module creates a new immutable dataset and therefore requires
separate artifact-generation approval. The state-at-t and state-at-t+1 inputs
are simulator-produced binary CA states, not observed temporal fire rasters.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml

from modules.feature_pipeline import (
    CANONICAL_FEATURE_NAMES as FEATURE_NAMES,
    OBSERVATION_MANIFEST_VERSION,
    OBSERVATION_METADATA_NAMES as METADATA_NAMES,
    POSITIVE_LABEL,
    SET_C_SCHEMA_VERSION as FEATURE_SCHEMA_VERSION,
    TARGET_NAME,
    TARGET_VERSION,
    FeatureAssembler,
    compute_moore_neighbor_count,
    normalize_binary_ca_state,
    validate_binary_ca_state,
)
from modules.wind_convention import WindContract, compute_wind_weighted_score


class SyntheticDatasetGenerator:
    """Generate immutable, provenance-bearing Set C transition observations."""

    REQUIRED_PROVENANCE = (
        "set_id",
        "experiment_id",
        "event_id",
        "scenario_id",
        "run_id",
        "grid_id",
        "timestep_t",
        "source_revision",
        "source_dirty_state",
        "generation_command",
        "environment_record",
        "transition_source_type",
        "transition_source_provenance",
    )

    TRANSITION_SOURCE_TYPES = ("authorized_simulated",)

    def __init__(self, config: dict):
        self.config = config
        self.base_dir = Path(__file__).resolve().parent
        self.dataset_cfg = dict(config["dataset_generation"])
        if not bool(self.dataset_cfg.get("artifact_write_enabled", False)):
            raise PermissionError(
                "dataset_generation.artifact_write_enabled is false; "
                "dataset generation requires separate approval"
            )
        self.target_cfg = dict(self.dataset_cfg.get("target", {}))
        if self.target_cfg.get("mode") != "next_timestep_ignition":
            raise ValueError(
                "dataset_generation.target.mode must be 'next_timestep_ignition'; "
                "final-burn labels are not valid transition targets"
            )
        if self.target_cfg.get("state_encoding") != "binary_burning_0_1":
            raise ValueError(
                "dataset_generation.target.state_encoding must be 'binary_burning_0_1'"
            )
        self.state_t_file = self._required_text(self.target_cfg, "state_t_file")
        self.state_t1_file = self._required_text(self.target_cfg, "state_t1_file")
        self.output_csv = self._required_text(self.dataset_cfg, "output_csv")

        self.provenance = dict(self.dataset_cfg.get("provenance", {}))
        for field in self.REQUIRED_PROVENANCE:
            if field == "timestep_t":
                if self.provenance.get(field) is None:
                    raise ValueError(f"dataset_generation.provenance.{field} is required")
            else:
                self._required_text(self.provenance, field)
        if str(self.provenance["set_id"]).strip() != "SetC":
            raise ValueError("dataset_generation.provenance.set_id must be 'SetC'")
        if str(self.provenance["experiment_id"]) not in Path(self.output_csv).stem:
            raise ValueError("output_csv filename must include the experiment_id")
        self.wind = WindContract.from_config(config["wind"])
        self.wind_cfg = self.wind.to_config()
        self.transition_source_type, self.transition_source_provenance = (
            self.validate_transition_source(
                self.provenance["transition_source_type"],
                self.provenance["transition_source_provenance"],
            )
        )
        if self.transition_source_provenance["wind"] != self.wind.to_manifest():
            raise ValueError(
                "Simulated transition source wind must exactly match configured wind"
            )
        if (
            int(self.transition_source_provenance["seed"])
            != int(self.provenance.get("seed", config["simulation"].get("seed")))
        ):
            raise ValueError("Simulated transition source seed must match provenance.seed")

        block_cfg = dict(self.dataset_cfg.get("spatial_block", {}))
        self.block_rows = self._required_positive_int(block_cfg, "size_rows")
        self.block_cols = self._required_positive_int(block_cfg, "size_cols")
        self.block_origin_row = int(block_cfg.get("origin_row", 0))
        self.block_origin_col = int(block_cfg.get("origin_col", 0))

        self.seed = int(config["simulation"].get("seed", 42))
        configured_seed = self.provenance.get("seed")
        if configured_seed is not None and int(configured_seed) != self.seed:
            raise ValueError("simulation.seed and provenance.seed must match")
        self.provenance["seed"] = self.seed
        self.rng = np.random.default_rng(self.seed)
        self.roi = self.validate_roi(self.dataset_cfg.get("roi"))
        self.sampling_policy = self.dataset_cfg.get("sampling_policy")
        if self.sampling_policy != "all_eligible":
            raise ValueError(
                "dataset_generation.sampling_policy must be explicitly approved as "
                "'all_eligible'; uncorrected case-control sampling is not permitted"
            )
        environment_cfg = dict(config["environment"])
        raster_dir = Path(environment_cfg["raster_dir"])
        if not raster_dir.is_absolute():
            raster_dir = (self.base_dir / raster_dir).resolve()
        self.raster_dir = raster_dir
        self.slope_file = environment_cfg["slope_file"]
        self.proximity_file = environment_cfg["proximity_file"]
        self.buildings_file = environment_cfg["buildings_file"]
        self.materials_file = environment_cfg.get("materials_file", "stack_materials.tif")
        self.nodata_value = float(environment_cfg.get("nodata_value", -9999))

    @staticmethod
    def _required_text(mapping: dict, field: str) -> str:
        value = mapping.get(field)
        if value is None or not str(value).strip():
            raise ValueError(f"{field} must be explicitly configured")
        return str(value).strip()

    @staticmethod
    def _required_positive_int(mapping: dict, field: str) -> int:
        value = mapping.get(field)
        if value is None:
            raise ValueError(
                f"spatial_block.{field} is unset; block scale requires separate approval"
            )
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"spatial_block.{field} must be positive")
        return parsed

    @staticmethod
    def validate_roi(configured: object) -> None:
        """Use the full aligned raster domain until a crop contract is approved."""
        if configured is not None:
            raise ValueError(
                "dataset_generation.roi must remain null until a crop is explicitly approved; "
                "null uses the full aligned raster domain"
            )
        return None

    @classmethod
    def validate_transition_source(
        cls, source_type: object, source_provenance: object
    ) -> tuple[str, dict]:
        """Validate the authorized-simulation provenance required by active Set C."""
        if source_type not in cls.TRANSITION_SOURCE_TYPES:
            raise ValueError(
                "provenance.transition_source_type must be 'authorized_simulated'; "
                "observed temporal fire rasters are not part of the active Set C contract"
            )
        if not isinstance(source_provenance, dict):
            raise ValueError("transition_source_provenance must be a mapping")
        required = (
            "simulator",
            "simulator_version",
            "simulator_code_revision",
            "config_sha256",
            "seed",
            "wind",
            "input_hashes",
            "authorization_reference",
        )
        missing = [
            name for name in required if source_provenance.get(name) in (None, "")
        ]
        if missing:
            raise ValueError(
                f"transition_source_provenance for {source_type} is incomplete: {missing}"
            )
        if isinstance(source_provenance["seed"], bool):
            raise ValueError("authorized_simulated seed must be an integer")
        int(source_provenance["seed"])
        source_wind = source_provenance["wind"]
        if not isinstance(source_wind, dict):
            raise ValueError("authorized_simulated wind must be a mapping")
        canonical_wind = WindContract.from_config(source_wind).to_manifest()
        if source_wind != canonical_wind:
            raise ValueError(
                "authorized_simulated wind must be the exact canonical "
                "seven-field simulated_wind.v1 mapping"
            )
        hashes = source_provenance["input_hashes"]
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("authorized_simulated input_hashes must be non-empty")
        digest_values = [
            str(source_provenance["config_sha256"]),
            *(str(value) for value in hashes.values()),
        ]
        if any(
            len(value) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in value)
            for value in digest_values
        ):
            raise ValueError("Simulation config and input hashes must be SHA-256 hex digests")
        validated_provenance = dict(source_provenance)
        validated_provenance["wind"] = canonical_wind
        return str(source_type), validated_provenance

    def _read_raster(
        self,
        name: str,
        filename: str,
        reference: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        raster_path = self.raster_dir / filename
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster file not found for {name}: {raster_path}")
        with rasterio.open(raster_path) as src:
            arr = src.read(1)
            valid_mask = src.read_masks(1) > 0
            if src.nodata is not None:
                valid_mask &= arr != src.nodata
            current = {"shape": arr.shape, "crs": src.crs, "transform": src.transform}
        if reference is None:
            reference = current
        else:
            for key in ("shape", "crs", "transform"):
                if current[key] != reference[key]:
                    raise ValueError(
                        f"Alignment mismatch for {name}: {key} "
                        f"{current[key]} != {reference[key]}"
                    )
        return np.asarray(arr, dtype=np.float32), valid_mask, reference

    def _load_aligned_stack(self) -> dict[str, Any]:
        reference: dict[str, Any] | None = None
        layers: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        for name, filename in (
            ("slope", self.slope_file),
            ("proximity", self.proximity_file),
            ("buildings", self.buildings_file),
            ("materials", self.materials_file),
            ("state_t", self.state_t_file),
            ("state_t1", self.state_t1_file),
        ):
            layers[name], valid, reference = self._read_raster(name, filename, reference)
            masks[name] = valid
        valid_mask = np.logical_and.reduce(list(masks.values()))
        for layer in layers.values():
            valid_mask &= np.isfinite(layer)
        valid_mask &= layers["slope"] != self.nodata_value
        valid_mask &= layers["slope"] != -9999.0
        rounded_materials = np.rint(layers["materials"])
        valid_mask &= np.isclose(layers["materials"], rounded_materials, atol=1e-6)
        valid_mask &= np.isin(rounded_materials.astype(np.int16), np.arange(6))
        for state_name in ("state_t", "state_t1"):
            layers[state_name] = normalize_binary_ca_state(
                layers[state_name], masks[state_name], state_name
            )
            layers[state_name] = np.where(
                valid_mask, layers[state_name], np.int8(0)
            ).astype(np.int8)
        return {
            **layers,
            "valid_mask": valid_mask,
            "shape": reference["shape"],
            "crs": reference["crs"],
            "transform": reference["transform"],
            "row_offset": 0,
            "col_offset": 0,
        }

    def _crop_stack(self, stack: dict[str, Any], roi: dict | None) -> dict[str, Any]:
        if roi is None:
            return stack
        row_start, row_end = int(roi["row_start"]), int(roi["row_end"])
        col_start, col_end = int(roi["col_start"]), int(roi["col_end"])
        row_slice, col_slice = slice(row_start, row_end), slice(col_start, col_end)
        cropped = {
            key: value[row_slice, col_slice]
            for key, value in stack.items()
            if isinstance(value, np.ndarray)
        }
        cropped.update(
            shape=cropped["slope"].shape,
            crs=stack["crs"],
            transform=stack["transform"],
            row_offset=row_start,
            col_offset=col_start,
        )
        return cropped

    def _compute_wind_layers(self, state_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = validate_binary_ca_state(state_t, "state_t")
        neighbor_count = compute_moore_neighbor_count(state)
        wind_weight = float(
            self.config.get("placeholder_transition", {}).get("wind_weight", 0.20)
        )
        weighted, _ = compute_wind_weighted_score(
            state,
            self.wind,
            wind_weight,
        )
        return neighbor_count, weighted

    @staticmethod
    def construct_transition_target(
        state_t: np.ndarray,
        state_t1: np.ndarray,
        eligible_mask: np.ndarray,
    ) -> np.ndarray:
        if state_t.shape != state_t1.shape or state_t.shape != eligible_mask.shape:
            raise ValueError("state_t, state_t1, and eligible_mask must have equal shapes")
        state_t_values = validate_binary_ca_state(state_t, "state_t")
        state_t1_values = validate_binary_ca_state(state_t1, "state_t1")
        return (
            eligible_mask.astype(bool)
            & (state_t_values == 0)
            & (state_t1_values == 1)
        ).astype(np.int8)

    @staticmethod
    def _require_burning_states_within_burnable_domain(
        state_t: np.ndarray,
        state_t1: np.ndarray,
        valid_mask: np.ndarray,
        burnable_mask: np.ndarray,
    ) -> None:
        """Reject raster-valid burning states outside the building domain."""
        expected_shape = state_t.shape
        if any(
            array.shape != expected_shape
            for array in (state_t1, valid_mask, burnable_mask)
        ):
            raise ValueError(
                "state_t, state_t1, valid_mask, and burnable_mask must have "
                "equal shapes"
            )
        valid = np.asarray(valid_mask, dtype=bool)
        burnable = np.asarray(burnable_mask, dtype=bool)
        outside_burnable = valid & (~burnable)
        state_t_count = int(
            np.count_nonzero(outside_burnable & (state_t == 1))
        )
        state_t1_count = int(
            np.count_nonzero(outside_burnable & (state_t1 == 1))
        )
        if state_t_count or state_t1_count:
            raise ValueError(
                "Burning states outside the approved building-burnable domain: "
                f"state_t={state_t_count}, state_t1={state_t1_count}"
            )

    def _build_feature_layers(self, stack: dict[str, Any]) -> dict[str, np.ndarray]:
        slope_risk = np.clip(stack["slope"], 0.0, 10.0).astype(np.float32) / 10.0
        proximity_risk = stack["proximity"].astype(np.float32) / 10.0
        building_presence = np.where(stack["buildings"] == 10.0, 1.0, 0.0).astype(np.float32)
        valid_mask = stack["valid_mask"].astype(bool)
        burnable_mask = valid_mask & (building_presence > 0)
        material_class = np.where(valid_mask, stack["materials"], 0.0).astype(np.float32)
        assembler = FeatureAssembler(
            {
                "slope_risk": slope_risk,
                "proximity_risk": proximity_risk,
                "building_presence": building_presence,
                "material_class": material_class,
                "burnable_mask": burnable_mask,
                "valid_mask": valid_mask,
                "grid_shape": stack["shape"],
            },
            self.wind_cfg,
            self.config.get("flammability_weights"),
        )
        state_t = validate_binary_ca_state(stack["state_t"], "state_t")
        state_t1 = validate_binary_ca_state(stack["state_t1"], "state_t1")
        self._require_burning_states_within_burnable_domain(
            state_t, state_t1, valid_mask, burnable_mask
        )
        neighbor_count, wind_weighted = self._compute_wind_layers(state_t)
        eligible = burnable_mask & (state_t == 0) & (neighbor_count > 0)
        labels = self.construct_transition_target(state_t, state_t1, eligible)
        features = assembler.assemble_grid_features(neighbor_count, wind_weighted)
        return {"features": features, "eligible_mask": eligible, "labels": labels}

    @staticmethod
    def _eligible_indices(
        eligible_mask: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, int, int, int]:
        flat_eligible, flat_labels = eligible_mask.ravel(), labels.ravel().astype(bool)
        selected = np.flatnonzero(flat_eligible).astype(np.int64)
        positive_count = int(np.count_nonzero(flat_eligible & flat_labels))
        negative_count = int(np.count_nonzero(flat_eligible & (~flat_labels)))
        return selected, positive_count, negative_count, int(flat_eligible.sum())

    @staticmethod
    def _require_eligible_class_coverage(
        eligible_count: int, positive_count: int, negative_count: int
    ) -> None:
        """Fail before writing unless the eligible population contains both labels."""
        if eligible_count == 0:
            raise ValueError("No eligible transition cells exist in the building domain")
        if positive_count == 0:
            raise ValueError(
                "No positive t-to-t+1 ignition transitions exist in the eligible "
                "building domain"
            )
        if negative_count == 0:
            raise ValueError(
                "No eligible negative transitions exist; both target classes are required"
            )

    @staticmethod
    def _duplicate_group_ids(feature_matrix: np.ndarray) -> list[str]:
        canonical = np.asarray(feature_matrix, dtype="<f4")
        return [sha256(row.tobytes()).hexdigest() for row in canonical]

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _dataset_manifest(self, output_path: Path, dataframe: pd.DataFrame) -> dict:
        raster_files = {
            "slope": self.slope_file,
            "proximity": self.proximity_file,
            "buildings": self.buildings_file,
            "materials": self.materials_file,
            "state_t": self.state_t_file,
            "state_t1": self.state_t1_file,
        }
        raster_provenance = {}
        for name, filename in raster_files.items():
            path = self.raster_dir / filename
            raster_provenance[name] = {
                "path": str(path),
                "sha256": self._file_sha256(path),
            }
        config_text = yaml.safe_dump(self.config, sort_keys=True)
        target_counts = dataframe[TARGET_NAME].value_counts().sort_index().to_dict()
        return {
            "manifest_schema_version": OBSERVATION_MANIFEST_VERSION,
            "artifact_kind": "set_c_observation_dataset",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "set_id": "SetC",
            "experiment_id": self.provenance["experiment_id"],
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "target_name": TARGET_NAME,
            "target_version": TARGET_VERSION,
            "positive_label": POSITIVE_LABEL,
            "dataset_path": str(output_path),
            "dataset_sha256": self._file_sha256(output_path),
            "row_count": int(len(dataframe)),
            "target_counts": {str(key): int(value) for key, value in target_counts.items()},
            "metadata_names": list(METADATA_NAMES),
            "source_revision": self.provenance["source_revision"],
            "source_dirty_state": self.provenance["source_dirty_state"],
            "generation_command": self.provenance["generation_command"],
            "environment_record": self.provenance["environment_record"],
            "seed": self.seed,
            "event_id": self.provenance["event_id"],
            "scenario_id": self.provenance["scenario_id"],
            "run_id": self.provenance["run_id"],
            "timestep_t": int(self.provenance["timestep_t"]),
            "transition_state_pair": {
                "kind": "simulator_produced_adjacent_states",
                "timestep_t": int(self.provenance["timestep_t"]),
                "timestep_t1": int(self.provenance["timestep_t"]) + 1,
            },
            "wind": self.wind.to_manifest(),
            "grid": {
                "grid_id": self.provenance["grid_id"],
                "shape": list(self._working_stack["shape"]),
                "crs": str(self._working_stack["crs"]),
                "transform": list(self._working_stack["transform"]),
            },
            "spatial_block": {
                "origin_row": self.block_origin_row,
                "origin_col": self.block_origin_col,
                "size_rows": self.block_rows,
                "size_cols": self.block_cols,
            },
            "duplicate_policy": "sha256_of_canonical_float32_feature_row",
            "sampling_policy": self.sampling_policy,
            "roi": {"mode": "full_aligned_raster_domain"},
            "transition_source_type": self.transition_source_type,
            "transition_source_provenance": self.transition_source_provenance,
            "config_sha256": sha256(config_text.encode("utf-8")).hexdigest(),
            "source_rasters": raster_provenance,
            "split_status": "unassigned; split_role is stored in a separate immutable split manifest",
        }

    def _metadata_frame(self, selected: np.ndarray, shape: tuple[int, int]) -> pd.DataFrame:
        local_rows, local_cols = np.unravel_index(selected, shape)
        rows = local_rows + int(self._working_stack["row_offset"])
        cols = local_cols + int(self._working_stack["col_offset"])
        block_rows = np.floor_divide(rows - self.block_origin_row, self.block_rows)
        block_cols = np.floor_divide(cols - self.block_origin_col, self.block_cols)
        count = len(selected)
        return pd.DataFrame(
            {
                "set_id": [self.provenance["set_id"]] * count,
                "experiment_id": [self.provenance["experiment_id"]] * count,
                "event_id": [self.provenance["event_id"]] * count,
                "scenario_id": [self.provenance["scenario_id"]] * count,
                "run_id": [self.provenance["run_id"]] * count,
                "seed": np.full(count, self.seed, dtype=np.int64),
                "timestep_t": np.full(count, int(self.provenance["timestep_t"]), dtype=np.int64),
                "cell_row": rows.astype(np.int64),
                "cell_col": cols.astype(np.int64),
                "grid_id": [self.provenance["grid_id"]] * count,
                "spatial_block_id": [
                    f"{self.provenance['grid_id']}:r{r}:c{c}"
                    for r, c in zip(block_rows, block_cols)
                ],
            }
        )

    def generate(self) -> str:
        full_stack = self._load_aligned_stack()
        self._working_stack = self._crop_stack(full_stack, self.roi)
        built = self._build_feature_layers(self._working_stack)
        selected, positives, negatives, eligible_count = self._eligible_indices(
            built["eligible_mask"], built["labels"]
        )
        self._require_eligible_class_coverage(
            eligible_count, positives, negatives
        )
        selected_features = built["features"][selected]
        dataframe = self._metadata_frame(selected, self._working_stack["shape"])
        dataframe["duplicate_group_id"] = self._duplicate_group_ids(selected_features)
        dataframe = dataframe.loc[:, list(METADATA_NAMES)]
        for index, name in enumerate(FEATURE_NAMES):
            dataframe[name] = selected_features[:, index]
        dataframe[TARGET_NAME] = built["labels"].ravel()[selected].astype(np.int8)
        dataframe["feature_schema_version"] = FEATURE_SCHEMA_VERSION
        dataframe["target_version"] = TARGET_VERSION
        if not set(METADATA_NAMES).issubset(dataframe.columns):
            raise RuntimeError("Generated dataset is missing approved provenance metadata")

        output_path = Path(self.output_csv)
        if not output_path.is_absolute():
            output_path = self.base_dir / output_path
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable dataset: {output_path}")
        manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
        if manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable manifest: {manifest_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)
        manifest = self._dataset_manifest(output_path, dataframe)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Saved immutable Set C transition dataset: {output_path}")
        print(f"Rows recorded: {len(dataframe)}")
        print(f"Eligible building cells: {eligible_count}")
        print(f"All positives kept: {positives}")
        print(f"All eligible negatives kept: {negatives}")
        return str(output_path)


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "config" / "default_experiment.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    SyntheticDatasetGenerator(config).generate()
