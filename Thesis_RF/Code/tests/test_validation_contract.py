import numpy as np

from validation_engine import (
	FINAL_FOOTPRINT_METRIC_SCOPE,
	REFERENCE_BURN_MASK_DESCRIPTION,
    evaluate_binary_spatial_masks,
    evaluate_estimator_probabilities,
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


def test_final_footprint_reference_language_rejects_temporal_claims():
	assert REFERENCE_BURN_MASK_DESCRIPTION == (
		"satellite-derived approximate reference burned-area mask"
	)
	assert FINAL_FOOTPRINT_METRIC_SCOPE == (
		"final-footprint agreement only; not temporal ignition, arrival-time, "
		"or probability-calibration evidence"
	)
