#!/usr/bin/env bash
# CA-only baseline (rule-based ignition, no ML) with the wind-fixed engine and
# NASA POWER wind for Dec 12, 2023 (2026-10-24). One full-map run, ~5.9 GB RAM.
# orchestrator.run_simulation always loads a model, so load_model is stubbed
# out here to keep automata.model None (the engine's rule path).
#   setsid -f bash ~/Documents/Projects/School/THEsis/Thesis_1/Code/run_ca_only_windfix.sh

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$CODE/.venv/bin/python"
GT="$CODE/processedData/raster/raster/stack_ground_truth.tif"
OUT="$CODE/output/windfix"
SUMMARY="$OUT/summary.txt"
name="CA_nasa"
final="$OUT/$name/final_state.tif"
log="$OUT/logs/$name.log"
mkdir -p "$OUT/logs"

if [ -f "$final" ]; then
	echo "$name: already done, skipping" >> "$SUMMARY"
	exit 0
fi

echo "$name: start $(date '+%F %T')" >> "$SUMMARY"
(cd "$CODE" && "$PY" -u -W ignore -c "
import resource
import orchestrator
from orchestrator import load_config, apply_pipeline_defaults, run_simulation
orchestrator.load_model = lambda automata, path: print('CA-only: ML model NOT loaded')
try:
    run_simulation(apply_pipeline_defaults(load_config('config/windfix_$name.yaml')))
finally:
    print('PEAK_RSS_MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
") > "$log" 2>&1
status=$?

if [ $status -ne 0 ] || [ ! -f "$final" ]; then
	echo "$name: FAILED (exit $status) $(date +%T), see logs/$name.log" >> "$SUMMARY"
	exit 1
fi

"$PY" -W ignore "$CODE/spatial_validation/validate_simulation.py" \
	--final "$final" --gt "$GT" --label "windfix $name" >> "$log" 2>&1

peak=$(grep -o 'PEAK_RSS_MB [0-9]*' "$log" | awk '{print $2}')
metrics=$(grep -E '^\s+(Precision|Recall|F1-Score|Jaccard):' "$log" | awk '{printf "%s %s  ", $1, $2}')
echo "$name: done $(date +%T)  peak ${peak}MB  $metrics" >> "$SUMMARY"

"$PY" -W ignore "$CODE/spatial_validation/score_both_areas.py" \
	--final "$final" --label "$name" >> "$SUMMARY" 2>&1
