#!/usr/bin/env bash
# LR trained on Kent's shared Set C dataset (train role, unweighted; train_lr_kent_r2.py)
# run in our sandbox CA (11-feature schema): Sta. Maria, NASA 17.7 km/h from 49 deg,
# no probability cutoff (probabilities used directly, as in Kent's CA), 100 steps,
# seeds 42-46. Configs = windfix_C_nasa_t000* with only the model swapped (2026-11-01).
#
# Start detached: setsid -f bash ~/Documents/Projects/School/THEsis/Thesis_1/Code/run_kent_r2.sh
# Progress: cat ~/Documents/Projects/School/THEsis/Thesis_1/Code/sandbox_kent/output/windfix/kent_r2_summary.txt

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$CODE/.venv/bin/python"
GT="$CODE/processedData/raster/raster/stack_ground_truth.tif"
OUT="$CODE/sandbox_kent/output/windfix"
SUMMARY="$OUT/kent_r2_summary.txt"
mkdir -p "$OUT/logs"

echo "=== started $(date '+%F %T')" >> "$SUMMARY"

for suffix in "" _s43 _s44 _s45 _s46; do
	name="KR2u_nasa_t000${suffix}"
	final="$OUT/$name/final_state.tif"
	log="$OUT/logs/$name.log"

	if [ -f "$final" ]; then
		echo "$name: already done, skipping" >> "$SUMMARY"
		continue
	fi

	echo "$name: start $(date +%T)" >> "$SUMMARY"
	(cd "$CODE/sandbox_kent" && "$PY" -u -W ignore -c "
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
		--final "$final" --label "$name" >> "$SUMMARY" 2>&1
done

echo "=== finished $(date '+%F %T')" >> "$SUMMARY"
