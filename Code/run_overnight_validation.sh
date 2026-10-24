#!/usr/bin/env bash
# Overnight validation of the wind-fixed LR models (2026-10-24).
# Runs 11 full-map simulations one after another (~14 min, ~5.7 GB RAM each),
# validating each against the Sitio Santa Maria ground truth right after it
# finishes. A failed run is logged and skipped; runs whose final_state.tif
# already exists are skipped, so re-running the script resumes where it stopped.
#
# Start it detached so it survives closing the terminal (fish or bash):
#   setsid -f bash ~/Documents/Projects/School/THEsis/Thesis_1/Code/run_overnight_validation.sh
# Progress: cat ~/Documents/Projects/School/THEsis/Thesis_1/Code/output/windfix/summary.txt

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$CODE/.venv/bin/python"
GT="$CODE/processedData/raster/raster/stack_ground_truth.tif"
OUT="$CODE/output/windfix"
SUMMARY="$OUT/summary.txt"
mkdir -p "$OUT/logs"

# name : folder the simulation runs from
RUNS=(
	"A_nasa_t000:sandbox_kent"
	"B_nasa_t000:."
	"C_nasa_t000:sandbox_kent"
	"B_nasa_t040:."
	"C_nasa_t030:sandbox_kent"
	"B_placeholder_t040:."
	"B_rpvm_t040:."
	# Threshold sensitivity (report all values; do not pick the best on Sta. Maria)
	"B_nasa_t030:."
	"B_nasa_t050:."
	"C_nasa_t040:sandbox_kent"
	"C_nasa_t050:sandbox_kent"
)

echo "=== started $(date '+%F %T')" >> "$SUMMARY"

for entry in "${RUNS[@]}"; do
	name="${entry%%:*}"
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
	metrics=$(grep -E '^\s+(Precision|Recall|F1-Score|Jaccard):' "$log" | awk '{printf "%s %s  ", $1, $2}')
	echo "$name: done $(date +%T)  peak ${peak}MB  $metrics" >> "$SUMMARY"

	# Same output scored two ways: whole map and building cells only
	"$PY" -W ignore "$CODE/spatial_validation/score_both_areas.py" \
		--final "$final" --label "$name" >> "$SUMMARY" 2>&1
done

echo "=== finished $(date '+%F %T')" >> "$SUMMARY"
