import numpy as np
import pytest

from modules.feature_pipeline import (
    CANONICAL_FEATURE_NAMES,
    POSITIVE_LABEL,
    PROVENANCE_METADATA_NAMES,
    SET_C_SCHEMA_VERSION,
    SPLIT_ROLES,
    TARGET_NAME,
    TARGET_VERSION,
    FeatureAssembler,
    compute_moore_neighbor_count,
    normalize_binary_ca_state,
    positive_class_index,
    validate_feature_columns,
    validate_feature_values,
)
from modules.wind_convention import (
    CALM_REPRESENTATION,
    DIRECTION_CONVENTION,
    DIRECTION_UNITS,
    NORTH_REFERENCE,
    WIND_SCHEMA_VERSION,
)


def _wind_config(speed_kmh: float = 10.0, direction_deg: float = 0.0) -> dict:
    return {
        "speed_kmh": speed_kmh,
        "direction_deg": direction_deg,
        "direction_convention": DIRECTION_CONVENTION,
        "direction_units": DIRECTION_UNITS,
        "north_reference": NORTH_REFERENCE,
        "schema_version": WIND_SCHEMA_VERSION,
        "calm_representation": CALM_REPRESENTATION,
    }


def test_set_c_contract_is_exact_and_metadata_is_not_a_predictor():
    assert CANONICAL_FEATURE_NAMES == (
        "slope_risk",
        "proximity_risk",
        "building_presence",
        "material_risk",
        "material_class",
        "wind_speed",
        "wind_sin",
        "wind_cos",
        "neighbor_burning_count",
        "composite_flammability",
        "wind_weighted_score",
    )
    assert not set(CANONICAL_FEATURE_NAMES) & set(PROVENANCE_METADATA_NAMES)
    assert SET_C_SCHEMA_VERSION == "set_c.v1"
    assert TARGET_NAME == "Ignited"
    assert TARGET_VERSION == "ignition_t_plus_1.v1"
    assert POSITIVE_LABEL == 1
    assert SPLIT_ROLES == ("train", "validation", "calibration", "test")
    assert validate_feature_columns(CANONICAL_FEATURE_NAMES) == CANONICAL_FEATURE_NAMES

    with pytest.raises(ValueError, match="Feature schema mismatch"):
        validate_feature_columns(reversed(CANONICAL_FEATURE_NAMES))


def test_positive_class_index_follows_class_order_and_rejects_missing_label():
    assert positive_class_index(np.array([1, 0], dtype=np.int64)) == 0
    assert positive_class_index(np.array([0, 1], dtype=np.int64)) == 1

    with pytest.raises(ValueError, match="positive label 1"):
        positive_class_index(np.array([0, 2], dtype=np.int64))


def test_feature_assembler_limits_burnability_to_valid_buildings():
    environment = {
        "slope_risk": np.zeros((1, 3), dtype=np.float32),
        "proximity_risk": np.zeros((1, 3), dtype=np.float32),
        "building_presence": np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        "material_class": np.array([[1, 0, 1]], dtype=np.int8),
        "burnable_mask": np.array([[True, True, False]]),
        "grid_shape": (1, 3),
    }
    assembler = FeatureAssembler(
        environment,
        _wind_config(),
    )

    assert assembler.burnable_mask.tolist() == [[True, False, False]]


def test_feature_value_contract_enforces_dynamic_and_derived_fields():
    row = np.array(
        [[0.5, 2.0, 1.0, 0.95, 1.0, 10.0, 0.0, 1.0, 2.0, 0.95, 0.4]],
        dtype=np.float32,
    )
    assert validate_feature_values(row).shape == (1, 11)

    row[0, 8] = 9.0
    with pytest.raises(ValueError, match="within 0..8"):
        validate_feature_values(row)


def test_moore_neighbor_count_preserves_int8_and_does_not_wrap_edges():
    state_t = np.zeros((3, 3), dtype=np.int8)
    state_t[0, 0] = 1

    counts = compute_moore_neighbor_count(state_t)

    assert state_t.dtype == np.int8
    assert counts.dtype == np.int8
    assert counts.tolist() == [[0, 1, 0], [1, 1, 0], [0, 0, 0]]

    with pytest.raises(TypeError, match="np.int8"):
        compute_moore_neighbor_count(state_t.astype(np.float32))


def test_invalid_state_nodata_normalizes_without_weakening_valid_cell_checks():
    state = np.array([[1.0, -9999.0], [0.0, 1.0]], dtype=np.float32)
    raster_valid = np.array([[True, False], [True, True]])

    normalized = normalize_binary_ca_state(state, raster_valid, "state_t")

    assert normalized.dtype == np.int8
    assert normalized.tolist() == [[1, 0], [0, 1]]
    with pytest.raises(ValueError, match="Raster-valid state_t"):
        normalize_binary_ca_state(state, np.ones_like(raster_valid), "state_t")
