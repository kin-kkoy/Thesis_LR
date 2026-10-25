#!/usr/bin/env bash
# Multi-seed runs of the key wind-fix configurations (2026-10-24).
# Seed 42 of each configuration was run by run_overnight_validation.sh; this
# adds seeds 43-46 so each configuration has 5 seeds. 16 full-map simulations,
# one after another (~13 min, ~5.9 GB RAM each, ~3.5 h total). Each run is
# validated and scored whole-map + buildings-only right after it finishes.
# Failed runs are logged and skipped; finished runs are skipped on re-run.
# At the end, aggregate_seeds.py prints mean +/- std per configuration.
#
# Start it detached so it survives closing the terminal (fish or bash):
#   setsid -f bash ~/Documents/Projects/School/THEsis/Thesis_1/Code/run_multiseed.sh
# Progress: cat ~/Documents/Projects/School/THEsis/Thesis_1/Code/output/windfix/multiseed_summary.txt

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$CODE/.venv/bin/python"
GT="$CODE/processedData/raster/raster/stack_ground_truth.tif"
OUT="$CODE/output/windfix"
SUMMARY="$OUT/multiseed_summary.txt"
mkdir -p "$OUT/logs"

# configuration : folder the simulation runs from
CONFIGS=(
	"B_nasa_t040:."
	"B_nasa_t000:."
	"C_nasa_t000:sandbox_kent"
	"C_nasa_t030:sandbox_kent"
	"A_nasa_t000:sandbox_kent"   # added 2026-10-25
)
SEEDS=(43 44 45 46)

echo "=== started $(date '+%F %T')" >> "$SUMMARY"

for seed in "${SEEDS[@]}"; do
	for entry in "${CONFIGS[@]}"; do
		base="${entry%%:*}"
		name="${base}_s${seed}"
		folder="$CODE/${entry#*:}"
		final="$folder/output/windfix/$name/final_state.tif"
		log="$OUT/logs/$name.log"

		if [ -f "$final" ]; then
			echo "$name: already done, skipping" >> "$SUMMARY"
			continue
		fi

		echo "$name: start $(date +%T)" >> "$SUMMARY"
		(cd "$folder" && "$PY" -u -W ignore -c "
import resource
from orchestrator import load_config, apply_pipeline_defaults, run_simulation
try:
    run_simulation(apply_pipeline_defaults(load_config('config/windfix_$name.yaml')))
finally:
    print('PEAK_RSS_MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
") > "$log" 2>&1
		status=$?

		if [ $status -ne 0 ] || [ ! -f "$final" ]; then
			echo "$name: FAILED (exit $status) $(date +%T), see logs/$name.log" >> "$SUMMARY"
			continue
		fi

		"$PY" -W ignore "$CODE/spatial_validation/validate_simulation.py" \
			--final "$final" --gt "$GT" --label "windfix $name" >> "$log" 2>&1
		peak=$(grep -o 'PEAK_RSS_MB [0-9]*' "$log" | awk '{print $2}')
		echo "$name: done $(date +%T)  peak ${peak}MB" >> "$SUMMARY"
		"$PY" -W ignore "$CODE/spatial_validation/score_both_areas.py" \
			--final "$final" --label "$name" >> "$SUMMARY" 2>&1
	done
done

echo "=== finished $(date '+%F %T')" >> "$SUMMARY"
"$PY" -W ignore "$CODE/spatial_validation/aggregate_seeds.py" >> "$SUMMARY" 2>&1
