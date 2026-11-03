"""Multi-seed simulation runs (Gap 2.1): mean +/- std instead of one seed.

All published numbers so far come from a single run with seed=42. The CA
ignition draw is stochastic, so a different seed gives a slightly different
fire. This script re-runs the full simulation for several seeds, validates
each against the ground truth, and prints per-seed metrics plus mean +/- std
-- the numbers to report in the paper's tables.

Each simulation takes roughly 5-15 minutes; five seeds is an hour-ish of
unattended runtime. The per-seed final states are kept as
output/final_state_seed<NN>.tif.

Usage (from Thesis_1/Code with the venv active):
    python spatial_validation/multi_seed_runs.py \
        --gt processedData/raster/raster/stack_ground_truth.tif \
        --seeds 41 42 43 44 45
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from orchestrator import apply_pipeline_defaults, load_config, run_simulation  # noqa: E402
from validate_simulation import validate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simulation across seeds and aggregate metrics")
    parser.add_argument("--config", default=str(CODE_DIR / "config" / "default_experiment.yaml"))
    parser.add_argument("--gt", required=True, help="Binary ground truth raster")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    args = parser.parse_args()

    final_state = CODE_DIR / "output" / "final_state.tif"
    results = []
    for seed in args.seeds:
        print(f"\n===== seed {seed} =====")
        config = apply_pipeline_defaults(load_config(args.config))
        config["simulation"]["seed"] = seed
        run_simulation(config)

        kept = CODE_DIR / "output" / f"final_state_seed{seed}.tif"
        shutil.move(final_state, kept)
        results.append((seed, validate(str(kept), args.gt, label=f"seed {seed}")))

    print(f"\n{'=' * 60}\n  Aggregate over {len(results)} seeds\n{'=' * 60}")
    for key in ("precision", "recall", "f1", "jaccard"):
        vals = [m[key] for _, m in results]
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {key:<10} mean {statistics.mean(vals):.4f}  +/- {std:.4f}   "
              f"(min {min(vals):.4f}, max {max(vals):.4f})")
    print("\nReport these as mean +/- std in the paper, with a note that all "
          "prior single numbers were seed=42.")


if __name__ == "__main__":
    main()
