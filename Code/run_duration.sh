#!/usr/bin/env bash
# Duration study (stopping protocol frozen with the RF side, 2026-11-03).
# One trajectory per seed to the longest horizon, snapshots at the shorter ones:
#   Airbase:    43 / 86 / 172 updates  (2 / 1 / 0.5 min per update x 86 min, DSWD fire-out)
#   Sta. Maria: 106 / 212 / 424 updates (2 / 1 / 0.5 min per update x 212 min, DSWD fire-out)
# Step 100 is also saved in both, as a check against the existing 100-step runs.
# Stage B LR, t=0.40, seeds 42-46. 10 full-map runs, one after another (~6-7 h total).
#
# Start detached:  setsid -f bash ~/Documents/Projects/School/THEsis/Thesis_1/Code/run_duration.sh
# Progress:        cat ~/Documents/Projects/School/THEsis/Thesis_1/Code/output/windfix/duration/summary.txt

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$CODE/.venv/bin/python"
OUT="$CODE/output/windfix/duration"
SUMMARY="$OUT/summary.txt"
mkdir -p "$OUT/logs"

echo "=== started $(date '+%F %T')" >> "$SUMMARY"

for fire in airbase stamaria; do
	for suffix in "" _s43 _s44 _s45 _s46; do
		name="${fire}_B_t040${suffix}"
		done_flag="$OUT/$name/termination.json"
		log="$OUT/logs/$name.log"

		if [ -f "$done_flag" ]; then
			echo "$name: already done, skipping" >> "$SUMMARY"
			continue
		fi

		echo "$name: start $(date +%T)" >> "$SUMMARY"
		(cd "$CODE" && "$PY" -u -W ignore -c "
import resource
from orchestrator import load_config, apply_pipeline_defaults, run_simulation
try:
    run_simulation(apply_pipeline_defaults(load_config('config/duration_$name.yaml')))
finally:
    print('PEAK_RSS_MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
") > "$log" 2>&1
		status=$?

		if [ $status -ne 0 ] || [ ! -f "$done_flag" ]; then
			echo "$name: FAILED (exit $status) $(date +%T), see logs/$name.log" >> "$SUMMARY"
			continue
		fi
		peak=$(grep -o 'PEAK_RSS_MB [0-9]*' "$log" | awk '{print $2}')
		stop=$(grep -o '"stop_step": [0-9]*' "$done_flag" | awk '{print $2}')
		echo "$name: done $(date +%T)  stopped at step $stop  peak ${peak}MB" >> "$SUMMARY"
	done
done

echo "=== finished $(date '+%F %T')" >> "$SUMMARY"
