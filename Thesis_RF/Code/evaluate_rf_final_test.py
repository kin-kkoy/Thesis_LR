"""Fail-closed final-test evaluator interface, isolated from model development."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from modules.feature_pipeline import (
    CANONICAL_FEATURE_NAMES,
    FINAL_TEST_ROLE,
    POSITIVE_LABEL,
    TARGET_NAME,
    predict_positive_probability,
)
from validation_engine import evaluate_estimator_probabilities


def evaluate_authorized_final_test(
    model: object,
    protected_rows: pd.DataFrame,
    authorization: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate protected rows; storage and ACL enforcement stay external."""
    if authorization.get("enabled") is not True:
        raise PermissionError("Final-test evaluation is disabled")
    for field in ("authorization_reference", "protected_storage_mechanism"):
        if not isinstance(authorization.get(field), str) or not str(authorization[field]).strip():
            raise PermissionError(f"Final-test evaluation requires approved {field}")
    if "split_role" not in protected_rows:
        raise ValueError("Protected rows must carry evaluator-only role membership")
    if set(protected_rows["split_role"].astype(str)) != {FINAL_TEST_ROLE}:
        raise PermissionError("The final evaluator accepts only protected final-test rows")
    missing = sorted({*CANONICAL_FEATURE_NAMES, TARGET_NAME}.difference(protected_rows.columns))
    if missing:
        raise ValueError(f"Protected final-test rows are missing columns: {missing}")
    target = protected_rows[TARGET_NAME]
    if target.empty or not pd.api.types.is_integer_dtype(target.dtype) or set(target.astype(int)) != {0, POSITIVE_LABEL}:
        raise ValueError("Protected final test must contain both integer target classes")
    probabilities = predict_positive_probability(
        model, protected_rows.loc[:, list(CANONICAL_FEATURE_NAMES)]
    )
    metrics = evaluate_estimator_probabilities(target.to_numpy(dtype=np.int8), probabilities)
    return {
        "evaluation_scope": FINAL_TEST_ROLE,
        "observation_count": int(len(target)),
        "probability_metrics": metrics,
        "threshold_metrics": None,
        "authorization_reference": str(authorization["authorization_reference"]),
    }


def main() -> None:
    raise PermissionError(
        "Final-test CLI remains disabled until protected storage/ACL and execution are approved"
    )


if __name__ == "__main__":
    main()
