"""Build Set C next-timestep ignition rows from simulated adjacent states.

The deprecated binary-raster publisher remains fail-closed. Active Set C uses
the authoritative five-state, in-memory transition observations defined by
D-013.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np

from modules.feature_pipeline import (
    CANONICAL_FEATURE_NAMES as FEATURE_NAMES,
    FeatureAssembler,
    compute_moore_neighbor_count,
    validate_binary_ca_state,
)
from modules.set_c_collector import (
    SET_C_ROW_BATCH_SCHEMA_VERSION,
    SetCCollector,
    SetCRowBatch,
)
from modules.wind_convention import WindContract, compute_wind_weighted_score
from modules.transition_observer import (
    TransitionObservation,
)


TRANSITION_ROW_BATCH_SCHEMA_VERSION = SET_C_ROW_BATCH_SCHEMA_VERSION
TRANSITION_FEATURE_NAMES = FEATURE_NAMES
TRANSITION_TARGET_VERSION = "ignition_t_plus_1.v1"
LEGACY_BINARY_STATE_ENCODING = "binary_burning_0_1"
LEGACY_BINARY_STATE_CONTRACT_STATUS = "deprecated_non_authoritative"


def build_in_memory_transition_rows(
    observation: TransitionObservation,
    *,
    environment: dict,
    config: dict,
) -> tuple[SetCRowBatch, ...]:
    """Construct bounded canonical batches from one five-state observation."""
    collector = SetCCollector(environment, config)
    collector(observation)
    return collector.pending_batches()


class SyntheticDatasetGenerator:
    """Legacy binary-raster publisher; non-authoritative for active Set C."""

    LEGACY_STATE_ENCODING = LEGACY_BINARY_STATE_ENCODING
    LEGACY_STATE_CONTRACT_STATUS = LEGACY_BINARY_STATE_CONTRACT_STATUS

    TRANSITION_SOURCE_TYPES = ("authorized_simulated",)

    def __init__(self, config: dict):
        dataset_cfg = dict(config["dataset_generation"])
        if not bool(dataset_cfg.get("artifact_write_enabled", False)):
            raise PermissionError(
                "dataset_generation.artifact_write_enabled is false; "
                "dataset generation requires separate approval"
            )
        target_cfg = dict(dataset_cfg.get("target", {}))
        if target_cfg.get("mode") != "next_timestep_ignition":
            raise ValueError(
                "dataset_generation.target.mode must be 'next_timestep_ignition'; "
                "final-burn labels are not valid transition targets"
            )
        if target_cfg.get("state_encoding") != LEGACY_BINARY_STATE_ENCODING:
            raise ValueError(
                "dataset_generation.target.state_encoding must identify the explicit "
                "legacy 'binary_burning_0_1' contract"
            )
        raise PermissionError(
            "Legacy binary_burning_0_1 artifact publication is disabled because "
            "it is non-authoritative for active Set C. Use SetCCollector with "
            "validated five-state TransitionObservation objects; no production "
            "artifact publisher is authorized."
        )

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
