"""Validate a simulation final_state raster against a ground truth raster.

Computes confusion matrix and standard binary classification metrics
(Precision, Recall, F1, AUC-ROC, Jaccard) for spatial fire prediction.

The final_state raster contains CA cell states (1-5):
  1 = NON_BURNABLE
  2 = NOT_YET_BURNING
  3 = IGNITED
  4 = BLAZING
  5 = EXTINGUISHED

Cells with state in {3, 4, 5} are treated as "fire". Anything else is "no fire".

The ground_truth raster is expected to be binary (0 = no fire, 1 = fire).

Usage:
    python validate_simulation.py final_state.tif ground_truth.tif
    python validate_simulation.py --final output/final_state.tif --gt processed-data-tif/stack_ground_truth.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# CA states that count as "burned" for the spatial validation
FIRE_STATES = {3, 4, 5}  # IGNITED, BLAZING, EXTINGUISHED

# Recall is the primary metric per the thesis proposal
RECALL_TARGET = 0.80
F1_TARGET = 0.80


def load_binary_fire_raster(
    path: str,
    treat_states_as_fire: bool = True,
) -> tuple[np.ndarray, dict]:
    """Read a raster and return a binary 'fire' mask.

    If `treat_states_as_fire`, integer values in FIRE_STATES become 1.
    Otherwise the raster is treated as already-binary (any non-zero = fire).
    """
    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "crs": str(src.crs) if src.crs else None,
            "transform": src.transform,
        }

    if treat_states_as_fire:
        binary = np.isin(arr, list(FIRE_STATES)).astype(np.int8)
    else:
        binary = (arr != 0).astype(np.int8)

    return binary, meta


def validate(
    final_state_path: str,
    ground_truth_path: str,
    label: str = "Simulation vs ground truth",
) -> dict:
    """Compute spatial validation metrics for a simulation against ground truth."""

    # Final state has CA states; map states 3/4/5 to fire
    pred, pred_meta = load_binary_fire_raster(final_state_path, treat_states_as_fire=True)
    # Ground truth is already binary 0/1
    truth, truth_meta = load_binary_fire_raster(ground_truth_path, treat_states_as_fire=False)

    if pred.shape != truth.shape:
        raise ValueError(
            f"Shape mismatch between final_state {pred.shape} "
            f"and ground_truth {truth.shape}"
        )

    pred_flat = pred.ravel()
    truth_flat = truth.ravel()

    cm = confusion_matrix(truth_flat, pred_flat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(truth_flat, pred_flat, zero_division=0)
    recall = recall_score(truth_flat, pred_flat, zero_division=0)
    f1 = f1_score(truth_flat, pred_flat, zero_division=0)
    jaccard = jaccard_score(truth_flat, pred_flat, zero_division=0)

    # AUC-ROC requires probabilities, not hard predictions. With binary
    # predictions it just scores how well the predicted classes separate
    # the truth — equivalent to balanced accuracy.
    auc_roc = roc_auc_score(truth_flat, pred_flat)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Final state:   {final_state_path}")
    print(f"  Ground truth:  {ground_truth_path}")
    print(f"  Grid shape:    {pred_meta['shape']}")
    print()
    print(f"  Confusion Matrix:")
    print(f"    TN = {tn:>12,}    FP = {fp:>12,}")
    print(f"    FN = {fn:>12,}    TP = {tp:>12,}")
    print()
    print(f"  Precision:  {precision:.6f}")
    recall_marker = " >= 0.80 PASS" if recall >= RECALL_TARGET else " < 0.80 FAIL (target >= 0.80)"
    f1_marker = " >= 0.80 PASS" if f1 >= F1_TARGET else " < 0.80 FAIL (target >= 0.80)"
    print(f"  Recall:     {recall:.6f}{recall_marker}")
    print(f"  F1-Score:   {f1:.6f}{f1_marker}")
    print(f"  AUC-ROC:    {auc_roc:.6f}")
    print(f"  Jaccard:    {jaccard:.6f}")

    # Sanity checks
    truth_fire_count = int(truth_flat.sum())
    pred_fire_count = int(pred_flat.sum())
    print()
    print(f"  Sanity:")
    print(f"    Ground truth fire cells:  {truth_fire_count:,}")
    print(f"    Predicted fire cells:     {pred_fire_count:,}")
    print(f"    Total cells:              {pred_flat.size:,}")
    print(f"    Class balance (truth):    {truth_fire_count / pred_flat.size:.6%}")

    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc_roc),
        "jaccard": float(jaccard),
        "truth_fire_count": truth_fire_count,
        "pred_fire_count": pred_fire_count,
        "total_cells": int(pred_flat.size),
    }


def main():
    parser = argparse.ArgumentParser(description="Spatial validation of CA fire simulation against ground truth")
    parser.add_argument("--final", "-f", type=str, required=True,
                        help="Path to final_state.tif (simulation output, CA states 1-5)")
    parser.add_argument("--gt", "-g", type=str, required=True,
                        help="Path to ground_truth.tif (binary 0/1)")
    parser.add_argument("--label", type=str, default="Simulation vs ground truth",
                        help="Label for the result printout")
    args = parser.parse_args()

    final_path = Path(args.final).resolve()
    gt_path = Path(args.gt).resolve()

    if not final_path.exists():
        raise FileNotFoundError(f"final_state raster not found: {final_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"ground_truth raster not found: {gt_path}")

    validate(str(final_path), str(gt_path), label=args.label)


if __name__ == "__main__":
    main()
