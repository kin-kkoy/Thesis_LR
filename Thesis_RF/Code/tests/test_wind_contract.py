import numpy as np
import pytest

from dataset_generator import SyntheticDatasetGenerator
from modules.automata_engine import FireAutomata
from modules.feature_pipeline import CANONICAL_FEATURE_NAMES, FeatureAssembler
from modules.wind_convention import (
    CALM_REPRESENTATION,
    DIRECTION_CONVENTION,
    DIRECTION_UNITS,
    NORTH_REFERENCE,
    WIND_SCHEMA_VERSION,
    WindContract,
    compute_wind_weighted_score,
)


def _wind_config(speed_kmh: float, direction_deg: float) -> dict:
    return {
        "speed_kmh": speed_kmh,
        "direction_deg": direction_deg,
        "direction_convention": DIRECTION_CONVENTION,
        "direction_units": DIRECTION_UNITS,
        "north_reference": NORTH_REFERENCE,
        "schema_version": WIND_SCHEMA_VERSION,
        "calm_representation": CALM_REPRESENTATION,
    }


def _environment(shape: tuple[int, int] = (3, 3)) -> dict:
    return {
        "slope_risk": np.zeros(shape, dtype=np.float32),
        "proximity_risk": np.zeros(shape, dtype=np.float32),
        "building_presence": np.ones(shape, dtype=np.float32),
        "material_class": np.ones(shape, dtype=np.int8),
        "burnable_mask": np.ones(shape, dtype=bool),
        "nodata_mask": np.zeros(shape, dtype=bool),
        "grid_shape": shape,
        "transform": None,
        "crs": None,
    }


@pytest.mark.parametrize(
    ("bearing", "expected_propagation", "strongest_source_offset"),
    [
        (0.0, (1.0, 0.0), (-1, 0)),
        (90.0, (0.0, -1.0), (0, 1)),
        (180.0, (-1.0, 0.0), (1, 0)),
        (270.0, (0.0, 1.0), (0, -1)),
    ],
)
def test_cardinal_from_bearings_fix_row_column_signs(
    bearing, expected_propagation, strongest_source_offset
):
    wind = WindContract.from_config(_wind_config(10.0, bearing))
    assert wind.propagation_direction_rc == pytest.approx(expected_propagation)

    kernel = wind.directional_kernel(0.5)
    dr, dc = strongest_source_offset
    assert kernel[dr + 1, dc + 1] == pytest.approx(float(kernel.max()))


@pytest.mark.parametrize(
    (
        "bearing",
        "expected_propagation",
        "strongest_source_offset",
        "weakest_source_offset",
    ),
    [
        (45.0, (np.sqrt(0.5), -np.sqrt(0.5)), (-1, 1), (1, -1)),
        (135.0, (-np.sqrt(0.5), -np.sqrt(0.5)), (1, 1), (-1, -1)),
        (225.0, (-np.sqrt(0.5), np.sqrt(0.5)), (1, -1), (-1, 1)),
        (315.0, (np.sqrt(0.5), np.sqrt(0.5)), (-1, -1), (1, 1)),
    ],
)
def test_diagonal_from_bearings_fix_row_column_signs(
    bearing,
    expected_propagation,
    strongest_source_offset,
    weakest_source_offset,
):
    wind = WindContract.from_config(_wind_config(10.0, bearing))
    kernel = wind.directional_kernel(0.5)

    assert wind.propagation_direction_rc == pytest.approx(expected_propagation)
    strongest_dr, strongest_dc = strongest_source_offset
    assert kernel[strongest_dr + 1, strongest_dc + 1] == pytest.approx(
        float(kernel.max())
    )
    neighbor_mask = np.ones((3, 3), dtype=bool)
    neighbor_mask[1, 1] = False
    weakest_dr, weakest_dc = weakest_source_offset
    assert kernel[weakest_dr + 1, weakest_dc + 1] == pytest.approx(
        float(kernel[neighbor_mask].min())
    )


def test_calm_representation_is_explicit_and_directionally_uniform():
    calm = WindContract.from_config(_wind_config(0.0, 0.0))
    kernel = calm.directional_kernel(50.0)

    assert calm.is_calm
    assert (calm.from_sin, calm.from_cos) == pytest.approx((0.0, 1.0))
    neighbor_mask = np.ones((3, 3), dtype=bool)
    neighbor_mask[1, 1] = False
    assert np.all(kernel[neighbor_mask] == 1.0)

    with pytest.raises(ValueError, match="Calm wind"):
        WindContract.from_config(_wind_config(0.0, 90.0))


def test_weighted_neighborhood_is_constant_boundary_and_does_not_wrap():
    blazing = np.zeros((3, 3), dtype=np.int8)
    blazing[0, 0] = 1
    calm = WindContract.from_config(_wind_config(0.0, 0.0))

    score, _ = compute_wind_weighted_score(blazing, calm, 0.5)

    assert score[0, 1] == pytest.approx(1.0 / 8.0)
    assert score[1, 0] == pytest.approx(1.0 / 8.0)
    assert score[2, 2] == 0.0


def test_from_bearing_features_normalize_wraparound_without_changing_meaning():
    assembler = FeatureAssembler(_environment(), _wind_config(10.0, 360.0))
    zeros = np.zeros((3, 3), dtype=np.float32)
    features = assembler.assemble_grid_features(zeros, zeros)

    assert assembler.wind.direction_deg == 0.0
    assert features[:, CANONICAL_FEATURE_NAMES.index("wind_sin")] == pytest.approx(0.0)
    assert features[:, CANONICAL_FEATURE_NAMES.index("wind_cos")] == pytest.approx(1.0)


def test_dataset_and_ca_share_identical_kernel_and_directional_score():
    wind_config = _wind_config(20.0, 315.0)
    transition = {"wind_weight": 0.4}
    config = {
        "simulation": {"seed": 9},
        "wind": wind_config,
        "placeholder_transition": transition,
        "ml_model": {"inference_mode": "stochastic_probability", "threshold": None},
    }
    automata = FireAutomata(_environment(), config)

    generator = SyntheticDatasetGenerator.__new__(SyntheticDatasetGenerator)
    generator.config = config
    generator.wind_cfg = wind_config
    generator.wind = WindContract.from_config(wind_config)

    blazing = np.zeros((3, 3), dtype=np.int8)
    blazing[0, 2] = 1
    _, dataset_score = generator._compute_wind_layers(blazing)
    direct_score, direct_kernel = compute_wind_weighted_score(
        blazing,
        generator.wind,
        transition["wind_weight"],
    )

    assert automata.wind_kernel == pytest.approx(direct_kernel)
    assert dataset_score == pytest.approx(direct_score)
    assert automata.grid.dtype == np.int8
