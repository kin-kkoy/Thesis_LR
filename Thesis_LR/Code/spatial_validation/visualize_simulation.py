"""Visual comparison of simulation final_state against ground truth.

Renders three panels using matplotlib:
  1. Ground truth (real fire perimeter)
  2. Simulation final state (states 3, 4, 5 = burned)
  3. Overlay/difference map (TP / FP / FN color-coded)

Outputs PNGs that can be loaded into QGIS as overlays or used directly in
the thesis. Cropped to the bounding box around the fire by default to keep
the visualizations focused.

Usage:
    python visualize_simulation.py --final final_state.tif --gt ground_truth.tif --out out.png
    python visualize_simulation.py --final final_state.tif --gt ground_truth.tif --out out.png --no-crop
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

FIRE_STATES = {3, 4, 5}


def load_binary(path: str, treat_states_as_fire: bool) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1)
    if treat_states_as_fire:
        return np.isin(arr, list(FIRE_STATES)).astype(np.int8)
    return (arr != 0).astype(np.int8)


def find_bbox(mask: np.ndarray, padding: int = 50) -> tuple[int, int, int, int]:
    """Find the bounding box of non-zero pixels in `mask` with optional padding."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return 0, mask.shape[0], 0, mask.shape[1]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, int(rmin) - padding)
    cmin = max(0, int(cmin) - padding)
    rmax = min(mask.shape[0], int(rmax) + padding)
    cmax = min(mask.shape[1], int(cmax) + padding)
    return rmin, rmax, cmin, cmax


def render(
    final_path: str,
    gt_path: str,
    out_path: str,
    crop: bool = True,
    title_prefix: str = "",
) -> None:
    pred = load_binary(final_path, treat_states_as_fire=True)
    truth = load_binary(gt_path, treat_states_as_fire=False)

    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs truth {truth.shape}")

    # Difference map: 0=both negative, 1=TP, 2=FP, 3=FN
    diff = np.zeros_like(pred, dtype=np.int8)
    diff[(pred == 1) & (truth == 1)] = 1   # TP
    diff[(pred == 1) & (truth == 0)] = 2   # FP
    diff[(pred == 0) & (truth == 1)] = 3   # FN

    if crop:
        # Use the union of pred and truth fire areas to set the crop window
        union = (pred | truth).astype(np.int8)
        rmin, rmax, cmin, cmax = find_bbox(union, padding=80)
    else:
        rmin, rmax = 0, pred.shape[0]
        cmin, cmax = 0, pred.shape[1]

    pred_crop = pred[rmin:rmax, cmin:cmax]
    truth_crop = truth[rmin:rmax, cmin:cmax]
    diff_crop = diff[rmin:rmax, cmin:cmax]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(truth_crop, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"{title_prefix}Ground Truth\n(real fire perimeter)")
    axes[0].axis("off")

    axes[1].imshow(pred_crop, cmap="Oranges", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"{title_prefix}Simulation Final State\n(predicted burned cells)")
    axes[1].axis("off")

    # Diff map: black=both negative, green=TP, red=FP, blue=FN
    diff_rgb = np.zeros((*diff_crop.shape, 3), dtype=np.uint8)
    diff_rgb[diff_crop == 1] = (40, 200, 40)    # TP green
    diff_rgb[diff_crop == 2] = (220, 40, 40)    # FP red
    diff_rgb[diff_crop == 3] = (40, 40, 220)    # FN blue

    axes[2].imshow(diff_rgb, interpolation="nearest")
    axes[2].set_title(f"{title_prefix}Difference Map\nGreen=TP  Red=FP  Blue=FN")
    axes[2].axis("off")

    # Compact stats footer
    tp = int(np.sum(diff == 1))
    fp = int(np.sum(diff == 2))
    fn = int(np.sum(diff == 3))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fig.suptitle(
        f"TP={tp:,}  FP={fp:,}  FN={fn:,}    "
        f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}",
        fontsize=11,
        y=0.04,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Render simulation vs ground-truth comparison")
    parser.add_argument("--final", "-f", type=str, required=True)
    parser.add_argument("--gt", "-g", type=str, required=True)
    parser.add_argument("--out", "-o", type=str, required=True)
    parser.add_argument("--no-crop", action="store_true", help="Render full extent (default: crop to fire area)")
    parser.add_argument("--title", type=str, default="", help="Optional prefix for panel titles")
    args = parser.parse_args()

    render(
        final_path=str(Path(args.final).resolve()),
        gt_path=str(Path(args.gt).resolve()),
        out_path=str(Path(args.out).resolve()),
        crop=not args.no_crop,
        title_prefix=(args.title + " " if args.title else ""),
    )


if __name__ == "__main__":
    main()
