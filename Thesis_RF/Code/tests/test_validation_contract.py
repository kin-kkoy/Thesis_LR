import numpy as np
import pytest

from validation_engine import (
	FINAL_FOOTPRINT_METRIC_SCOPE,
	PRIMARY_FINAL_FOOTPRINT_DOMAIN,
	REFERENCE_BURN_MASK_DESCRIPTION,
	evaluate_binary_spatial_masks,
	evaluate_estimator_probabilities,
	evaluate_final_footprint_arrays,
	run_default_validation,
)


def test_binary_spatial_metrics_exclude_invalid_and_non_building_cells():
    truth = np.array([[True, True, False, True]])
    prediction = np.array([[True, False, True, False]])
    valid_building_domain = np.array([[True, True, False, False]])

    result = evaluate_binary_spatial_masks(truth, prediction, valid_building_domain)

    assert result["confusion_matrix"] == [[0, 0], [1, 1]]
    assert result["evaluation_pixel_count"] == 2
    assert result["excluded_pixel_count"] == 2
    assert "auc_roc" not in result
    assert "roc_auc" not in result


def test_probability_metrics_are_separate_and_use_continuous_scores():
    result = evaluate_estimator_probabilities(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.7, 0.9]),
    )

    assert set(result) == {
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
        "observation_count",
    }
    assert result["observation_count"] == 4


def test_probability_metrics_reject_hard_binary_prediction_masks():
	with pytest.raises(ValueError, match="hard binary prediction mask"):
		evaluate_estimator_probabilities(
			np.array([0, 0, 1, 1]),
			np.array([0, 1, 0, 1]),
		)


def test_final_footprint_reference_language_rejects_temporal_claims():
	assert REFERENCE_BURN_MASK_DESCRIPTION == (
		"satellite-derived approximate reference burned-area mask"
	)
	assert FINAL_FOOTPRINT_METRIC_SCOPE == (
		"final-footprint agreement only; not temporal ignition, arrival-time, "
		"or probability-calibration evidence"
	)


def test_primary_final_footprint_domain_and_reference_exclusions_reconcile():
	final_state = np.array(
		[
			[5, 0, 0, 0, 0],
			[0, 0, 5, 0, 0],
		],
		dtype=np.int8,
	)
	reference_burned = np.array(
		[
			[True, True, True, True, True],
			[True, True, False, False, False],
		]
	)
	environment_valid = np.ones_like(reference_burned)
	environment_valid[0, 1] = False
	environment_valid[1, 0] = False
	prediction_valid = np.ones_like(reference_burned)
	prediction_valid[0, 2] = False
	reference_valid = np.ones_like(reference_burned)
	reference_valid[1, 1] = False
	building_valid = np.ones_like(reference_burned)
	building_valid[0, 3] = False
	mapped_building = np.ones_like(reference_burned)
	mapped_building[0, 4] = False
	mapped_building[1, 0] = False

	result = evaluate_final_footprint_arrays(
		final_state,
		reference_burned,
		environment_valid,
		prediction_valid,
		reference_valid,
		building_valid,
		mapped_building,
	)

	assert result["metric_population"] == "primary_common_valid_mapped_building"
	assert result["primary_domain_definition"] == PRIMARY_FINAL_FOOTPRINT_DOMAIN
	assert result["confusion_matrix"] == [[2, 1], [0, 1]]
	assert result["evaluation_pixel_count"] == 4
	assert "roc_auc" not in result
	assert "auc_roc" not in result

	coverage = result["reference_coverage"]
	assert coverage["reference_valid_burned_total"] == 6
	assert coverage["reference_burned_included_in_primary_domain"] == 1
	assert coverage["reference_burned_excluded_from_primary_domain"] == 5
	assert coverage["reference_invalid_pixel_count"] == 1
	assert coverage["exclusion_marginal_counts"] == {
		"outside_environmental_validity": 2,
		"outside_prediction_validity": 1,
		"outside_building_raster_validity": 1,
		"outside_mapped_buildings": 2,
	}
	assert coverage["exclusion_marginals_are_additive"] is False
	assert coverage["exclusion_priority_partition"] == {
		"outside_environmental_validity": 2,
		"outside_prediction_validity": 1,
		"outside_building_raster_validity": 1,
		"outside_mapped_buildings": 1,
		"included_in_primary_domain": 1,
	}
	assert sum(coverage["exclusion_priority_partition"].values()) == 6
	assert coverage["priority_partition_reconciles"] is True


def test_environment_validity_is_explicit_and_not_inferred_from_prediction_validity():
	state = np.array([[5, 5]], dtype=np.int8)
	reference = np.array([[True, True]])
	prediction_valid = np.array([[True, True]])
	reference_valid = np.array([[True, True]])
	building_valid = np.array([[True, True]])
	mapped_building = np.array([[True, True]])

	result = evaluate_final_footprint_arrays(
		state,
		reference,
		np.array([[True, False]]),
		prediction_valid,
		reference_valid,
		building_valid,
		mapped_building,
	)

	assert result["evaluation_pixel_count"] == 1
	assert result["reference_coverage"]["exclusion_priority_partition"][
		"outside_environmental_validity"
	] == 1


def test_reference_invalid_cells_are_unknown_not_burned_exclusions():
	result = evaluate_final_footprint_arrays(
		np.array([[5, 0]], dtype=np.int8),
		np.array([[True, True]]),
		np.array([[True, True]]),
		np.array([[True, True]]),
		np.array([[True, False]]),
		np.array([[True, True]]),
		np.array([[True, True]]),
	)

	coverage = result["reference_coverage"]
	assert coverage["reference_valid_burned_total"] == 1
	assert coverage["reference_burned_excluded_from_primary_domain"] == 0
	assert coverage["reference_invalid_pixel_count"] == 1
	assert "unknown" in coverage["reference_invalid_semantics"]


def test_timestep_limited_active_states_are_censored_footprint_through_stop():
	result = evaluate_final_footprint_arrays(
		np.array([[3, 4, 5, 0]], dtype=np.int8),
		np.array([[True, True, True, False]]),
		np.ones((1, 4), dtype=bool),
		np.ones((1, 4), dtype=bool),
		np.ones((1, 4), dtype=bool),
		np.ones((1, 4), dtype=bool),
		np.ones((1, 4), dtype=bool),
		termination_reason="timestep_limit",
	)

	assert result["confusion_matrix"] == [[1, 0], [0, 3]]
	assert result["termination"] == {
		"reason": "timestep_limit",
		"active_cell_count": 2,
		"is_complete_final_footprint": False,
		"status": "censored_incomplete_footprint_through_stopping_time",
		"predicted_footprint_state_values": [3, 4, 5],
	}


def test_timestep_limit_without_active_states_remains_censored_and_incomplete():
	result = evaluate_final_footprint_arrays(
		np.array([[5, 0]], dtype=np.int8),
		np.array([[True, False]]),
		np.ones((1, 2), dtype=bool),
		np.ones((1, 2), dtype=bool),
		np.ones((1, 2), dtype=bool),
		np.ones((1, 2), dtype=bool),
		np.ones((1, 2), dtype=bool),
		termination_reason="timestep_limit",
	)

	assert result["confusion_matrix"] == [[1, 0], [0, 1]]
	assert result["termination"] == {
		"reason": "timestep_limit",
		"active_cell_count": 0,
		"is_complete_final_footprint": False,
		"status": "censored_incomplete_footprint_through_stopping_time",
		"predicted_footprint_state_values": [3, 4, 5],
	}


def test_inactive_termination_rejects_active_states():
	with pytest.raises(ValueError, match="inactive termination"):
		evaluate_final_footprint_arrays(
			np.array([[4]], dtype=np.int8),
			np.array([[True]]),
			np.array([[True]]),
			np.array([[True]]),
			np.array([[True]]),
			np.array([[True]]),
			np.array([[True]]),
			termination_reason="inactive",
		)


def test_whole_reference_comparison_is_optional_secondary_and_descriptive():
	result = evaluate_final_footprint_arrays(
		np.array([[5, 0]], dtype=np.int8),
		np.array([[True, False]]),
		np.array([[True, True]]),
		np.array([[True, True]]),
		np.array([[True, True]]),
		np.array([[True, True]]),
		np.array([[True, False]]),
		include_secondary_whole_reference=True,
	)

	secondary = result["secondary_descriptive_whole_reference"]
	assert secondary["role"] == "secondary_descriptive_only"
	assert "building restrictions not applied" in secondary["domain_definition"]
	assert "roc_auc" not in secondary
	assert "auc_roc" not in secondary


def test_final_footprint_masks_must_be_boolean_and_shape_aligned():
	with pytest.raises(ValueError, match="simulation_environment_valid_mask"):
		evaluate_final_footprint_arrays(
			np.array([[5]], dtype=np.int8),
			np.array([[True]]),
			np.array([[1]], dtype=np.uint8),
			np.array([[True]]),
			np.array([[True]]),
			np.array([[True]]),
			np.array([[True]]),
		)


def test_default_validation_requires_explicit_environment_mask_before_io():
	with pytest.raises(ValueError, match="exact simulation-environment-valid mask"):
		run_default_validation()
