import numpy as np
import pytest

from dataset_generator import SyntheticDatasetGenerator
from generate_multi_scenario import build_scenarios
from modules.feature_pipeline import CANONICAL_FEATURE_NAMES
from modules.wind_convention import (
    CALM_REPRESENTATION,
    DIRECTION_CONVENTION,
    DIRECTION_UNITS,
    NORTH_REFERENCE,
    WIND_SCHEMA_VERSION,
    WindContract,
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


def _simulated_source(
    seed: int = 7,
    wind: dict | None = None,
) -> dict:
    digest = "a" * 64
    source_wind = _wind_config(10.0, 315.0) if wind is None else wind
    return {
        "transition_source_type": "authorized_simulated",
        "transition_source_provenance": {
            "simulator": "thesis-ca-orchestrator",
            "simulator_version": "set-c-contract-v1",
            "simulator_code_revision": "revision-1",
            "config_sha256": digest,
            "seed": seed,
            "wind": source_wind,
            "input_hashes": {"environment_stack": digest},
            "authorization_reference": "D-012",
        },
    }


def test_legacy_binary_artifact_publisher_remains_fail_closed():
    disabled = {"dataset_generation": {"artifact_write_enabled": False}}
    with pytest.raises(PermissionError, match="requires separate approval"):
        SyntheticDatasetGenerator(disabled)

    legacy_enabled = {
        "dataset_generation": {
            "artifact_write_enabled": True,
            "target": {
                "mode": "next_timestep_ignition",
                "state_encoding": "binary_burning_0_1",
            },
        }
    }
    with pytest.raises(PermissionError, match="non-authoritative for active Set C"):
        SyntheticDatasetGenerator(legacy_enabled)


def test_transition_target_uses_state_t_to_state_t_plus_one():
    state_t = np.array([[1, 0, 0]], dtype=np.int8)
    state_t1 = np.array([[1, 1, 0]], dtype=np.int8)
    eligible = np.array([[False, True, True]])

    target = SyntheticDatasetGenerator.construct_transition_target(
        state_t, state_t1, eligible
    )

    assert target.tolist() == [[0, 1, 0]]


def test_transition_features_include_dynamic_neighbor_and_wind_score():
    generator = SyntheticDatasetGenerator.__new__(SyntheticDatasetGenerator)
    generator.wind_cfg = _wind_config()
    generator.wind = WindContract.from_config(generator.wind_cfg)
    generator.config = {}
    stack = {
        "slope": np.zeros((3, 3), dtype=np.float32),
        "proximity": np.zeros((3, 3), dtype=np.float32),
        "buildings": np.full((3, 3), 10.0, dtype=np.float32),
        "materials": np.ones((3, 3), dtype=np.float32),
        "state_t": np.array(
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8
        ),
        "state_t1": np.array(
            [[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8
        ),
        "valid_mask": np.ones((3, 3), dtype=bool),
        "shape": (3, 3),
    }

    built = generator._build_feature_layers(stack)

    assert built["features"].shape == (9, len(CANONICAL_FEATURE_NAMES))
    assert built["features"][1, CANONICAL_FEATURE_NAMES.index("neighbor_burning_count")] == 1
    assert built["labels"].tolist()[0][1] == 1
    assert not built["eligible_mask"][1, 1]


def test_burning_states_must_remain_within_valid_building_domain():
    valid = np.array([[True, True], [True, False]])
    burnable = np.array([[True, False], [True, False]])
    empty = np.zeros((2, 2), dtype=np.int8)

    SyntheticDatasetGenerator._require_burning_states_within_burnable_domain(
        empty, empty, valid, burnable
    )

    state_t = empty.copy()
    state_t[0, 1] = 1
    with pytest.raises(ValueError, match=r"state_t=1, state_t1=0"):
        SyntheticDatasetGenerator._require_burning_states_within_burnable_domain(
            state_t, empty, valid, burnable
        )

    state_t1 = empty.copy()
    state_t1[0, 1] = 1
    with pytest.raises(ValueError, match=r"state_t=0, state_t1=1"):
        SyntheticDatasetGenerator._require_burning_states_within_burnable_domain(
            empty, state_t1, valid, burnable
        )

    mask_invalid_state = empty.copy()
    mask_invalid_state[1, 1] = 1
    SyntheticDatasetGenerator._require_burning_states_within_burnable_domain(
        mask_invalid_state, mask_invalid_state, valid, burnable
    )


def test_block_scale_is_required_and_duplicate_ids_are_deterministic():
    with pytest.raises(ValueError, match="requires separate approval"):
        SyntheticDatasetGenerator._required_positive_int({"size_rows": None}, "size_rows")

    matrix = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    identifiers = SyntheticDatasetGenerator._duplicate_group_ids(matrix)
    assert identifiers[0] == identifiers[1]
    assert identifiers[0] != identifiers[2]


def test_all_eligible_sampling_preserves_population_rows():
    eligible = np.array([[True, True, False, True]])
    labels = np.array([[1, 0, 0, 0]], dtype=np.int8)

    selected, positives, negatives, eligible_count = (
        SyntheticDatasetGenerator._eligible_indices(eligible, labels)
    )

    assert selected.tolist() == [0, 1, 3]
    assert (positives, negatives, eligible_count) == (1, 2, 3)


def test_eligible_population_requires_both_target_classes_before_writing():
    SyntheticDatasetGenerator._require_eligible_class_coverage(3, 1, 2)

    with pytest.raises(ValueError, match="No eligible transition cells"):
        SyntheticDatasetGenerator._require_eligible_class_coverage(0, 0, 0)
    with pytest.raises(ValueError, match=r"No positive t-to-t\+1"):
        SyntheticDatasetGenerator._require_eligible_class_coverage(3, 0, 3)
    with pytest.raises(ValueError, match="No eligible negative transitions"):
        SyntheticDatasetGenerator._require_eligible_class_coverage(3, 3, 0)


def test_transition_source_provenance_accepts_only_authorized_simulation():
    with pytest.raises(ValueError, match="transition_source_type"):
        SyntheticDatasetGenerator.validate_transition_source(None, None)

    with pytest.raises(ValueError, match="observed temporal fire rasters"):
        SyntheticDatasetGenerator.validate_transition_source("observed", {})

    digest = "a" * 64
    source_type, provenance = SyntheticDatasetGenerator.validate_transition_source(
        "authorized_simulated",
        {
            "simulator": "named-simulator",
            "simulator_version": "version-1",
            "simulator_code_revision": "revision-1",
            "config_sha256": digest,
            "seed": 7,
            "wind": _wind_config(10.0, 315.0),
            "input_hashes": {"state_t": digest},
            "authorization_reference": "approval-record-1",
        },
    )
    assert source_type == "authorized_simulated"
    assert provenance["seed"] == 7
    assert provenance["wind"] == _wind_config(10.0, 315.0)

    with pytest.raises(ValueError, match="incomplete"):
        SyntheticDatasetGenerator.validate_transition_source(
            "authorized_simulated", {"simulator": "named-simulator"}
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda source: source["transition_source_provenance"].pop("wind"),
        lambda source: source["transition_source_provenance"]["wind"].update(
            direction_convention="meteorological_to"
        ),
        lambda source: source["transition_source_provenance"]["wind"].update(
            direction_deg=675.0
        ),
        lambda source: source["transition_source_provenance"]["wind"].update(
            unapproved_field="value"
        ),
    ],
)
def test_transition_source_wind_provenance_fails_closed(mutate):
    source = _simulated_source()
    mutate(source)

    with pytest.raises(ValueError, match="wind|incomplete"):
        SyntheticDatasetGenerator.validate_transition_source(
            source["transition_source_type"],
            source["transition_source_provenance"],
        )


def test_roi_is_full_aligned_domain_until_crop_is_approved():
    assert SyntheticDatasetGenerator.validate_roi(None) is None
    with pytest.raises(ValueError, match="full aligned raster domain"):
        SyntheticDatasetGenerator.validate_roi(
            {"row_start": 0, "row_end": 10, "col_start": 0, "col_end": 10}
        )


def test_scenarios_require_explicit_wind_matched_simulated_state_pairs():
    config = {
        "wind": _wind_config(0.0, 0.0),
        "simulation": {"seed": 1},
        "dataset_generation": {
            "target": {"state_t_file": None, "state_t1_file": None},
            "provenance": {"experiment_id": "experiment-1"},
            "scenario_records": [],
        },
    }
    with pytest.raises(ValueError, match="explicit wind-matched"):
        build_scenarios(config)

    config["dataset_generation"]["scenario_records"] = [
        {
            "event_id": "event-1",
            "scenario_id": "scenario-1",
            "run_id": "run-1",
            "seed": 7,
            "timestep_t": 3,
            "grid_id": "grid-1",
            "speed_kmh": 10.0,
            "direction_deg": 315.0,
            "state_t_file": "state-event-1-run-1-t3.tif",
            "state_t1_file": "state-event-1-run-1-t4.tif",
            **_simulated_source(),
        }
    ]
    scenarios = build_scenarios(config)
    scenario = scenarios[0]
    assert scenario["wind"] == _wind_config(10.0, 315.0)
    assert scenario["dataset_generation"]["target"]["state_t_file"].endswith("t3.tif")
    assert scenario["dataset_generation"]["provenance"]["timestep_t"] == 3


def test_state_pair_cannot_be_rebound_to_another_wind_scenario():
    base_record = {
        "event_id": "event-1",
        "run_id": "run-1",
        "seed": 7,
        "timestep_t": 3,
        "grid_id": "grid-1",
        "speed_kmh": 10.0,
        "direction_deg": 315.0,
        "state_t_file": "state-t3.tif",
        "state_t1_file": "state-t4.tif",
        **_simulated_source(),
    }
    config = {
        "wind": _wind_config(),
        "simulation": {},
        "dataset_generation": {
            "target": {},
            "provenance": {"experiment_id": "experiment-1"},
            "scenario_records": [
                {**base_record, "scenario_id": "scenario-1"},
                {
                    **base_record,
                    "scenario_id": "scenario-2",
                    "direction_deg": 90.0,
                    **_simulated_source(wind=_wind_config(10.0, 90.0)),
                },
            ],
        },
    }
    with pytest.raises(ValueError, match="cannot be rebound"):
        build_scenarios(config)


def test_scenario_wind_must_match_originating_simulated_source():
    record = {
        "event_id": "event-1",
        "scenario_id": "scenario-1",
        "run_id": "run-1",
        "seed": 7,
        "timestep_t": 3,
        "grid_id": "grid-1",
        "speed_kmh": 10.0,
        "direction_deg": 90.0,
        "state_t_file": "state-t3.tif",
        "state_t1_file": "state-t4.tif",
        **_simulated_source(wind=_wind_config(10.0, 315.0)),
    }
    config = {
        "wind": _wind_config(),
        "simulation": {},
        "dataset_generation": {
            "target": {},
            "provenance": {"experiment_id": "experiment-1"},
            "scenario_records": [record],
        },
    }

    with pytest.raises(ValueError, match="does not match its simulated source"):
        build_scenarios(config)


def test_grid_identity_must_be_stable_and_path_free():
    record = {
        "event_id": "event-1",
        "scenario_id": "scenario-1",
        "run_id": "run-1",
        "seed": 7,
        "timestep_t": 3,
        "grid_id": "../unstable-grid",
        "speed_kmh": 10.0,
        "direction_deg": 315.0,
        "state_t_file": "state-t3.tif",
        "state_t1_file": "state-t4.tif",
        **_simulated_source(),
    }
    config = {
        "wind": _wind_config(),
        "simulation": {},
        "dataset_generation": {
            "target": {},
            "provenance": {"experiment_id": "experiment-1"},
            "scenario_records": [record],
        },
    }

    with pytest.raises(ValueError, match="grid identifiers"):
        build_scenarios(config)
