#!/usr/bin/env bash
# Second fire (Sitio Airbase / Sanggi Interior, Pajo, 2025-01-24 13:00 LST), 2026-11-02.
# Stage B + t=0.40 (fixed in advance, not re-tuned), seeds 42-46. Ignition 606063.2784, 1140377.2699 (row 1332, col 2501; Kent's point moved 3 m S onto a building);
# NASA POWER wind 10.6 km/h from 61 deg. Scored against stack_ground_truth_airbase.tif.
# 5 full-map simulations, one after another (~8-13 min, ~6 GB RAM each).
#
# Start it detached so it survives closing the terminal (fish or bash):
#   setsid -f bash ~/Documents/Projects/School/THEsis/Thesis_1/Code/run_airbase.sh
# Progress: cat ~/Documents/Projects/School/THEsis/Thesis_1/Code/output/windfix/airbase_summary.txt

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$CODE/.venv/bin/python"
GT="$CODE/processedData/raster/raster/stack_ground_truth_airbase.tif"
OUT="$CODE/output/windfix"
SUMMARY="$OUT/airbase_summary.txt"
mkdir -p "$OUT/logs"

echo "=== started $(date '+%F %T')" >> "$SUMMARY"

for suffix in "" _s43 _s44 _s45 _s46; do
	name="airbase_B_t040${suffix}"
	final="$OUT/$name/final_state.tif"
	log="$OUT/logs/$name.log"

	if [ -f "$final" ]; then
		echo "$name: already done, skipping" >> "$SUMMARY"
		continue
	fi

	echo "$name: start $(date +%T)" >> "$SUMMARY"
	(cd "$CODE" && "$PY" -u -W ignore -c "
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

	peak=$(grep -o 'PEAK_RSS_MB [0-9]*' "$log" | awk '{print $2}')
	echo "$name: done $(date +%T)  peak ${peak}MB" >> "$SUMMARY"
	"$PY" -W ignore "$CODE/spatial_validation/score_both_areas.py" \
		--final "$final" --gt "$GT" --label "$name" >> "$SUMMARY" 2>&1
done

echo "=== finished $(date '+%F %T')" >> "$SUMMARY"
