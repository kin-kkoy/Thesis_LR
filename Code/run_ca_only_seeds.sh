#!/usr/bin/env bash
# CA-only (rule-based ignition, no ML) multi-seed check, 2026-10-24.
# Seed 42 died at step 1 (run_ca_only_windfix.sh), so seeds 43-51 are run one
# after another with the same wind-fixed engine, NASA wind and config; only
# simulation.seed and output_dir change. ~5.9 GB RAM for runs that spread.
#   setsid -f bash ~/Documents/Projects/School/THEsis/Thesis_1/Code/run_ca_only_seeds.sh

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$CODE/.venv/bin/python"
GT="$CODE/processedData/raster/raster/stack_ground_truth.tif"
OUT="$CODE/output/windfix"
SUMMARY="$OUT/summary.txt"
mkdir -p "$OUT/logs"

echo "=== CA-only seeds started $(date '+%F %T')" >> "$SUMMARY"
for seed in 43 44 45 46 47 48 49 50 51; do
	name="CA_nasa_s$seed"
	cfg="config/windfix_$name.yaml"
	final="$OUT/$name/final_state.tif"
	log="$OUT/logs/$name.log"
	if [ -f "$final" ]; then
		echo "$name: already done, skipping" >> "$SUMMARY"
		continue
	fi
	sed -e "s/^  seed: 42$/  seed: $seed/" -e "s#output/windfix/CA_nasa\$#output/windfix/$name#" \
		"$CODE/config/windfix_CA_nasa.yaml" > "$CODE/$cfg"

	echo "$name: start $(date +%T)" >> "$SUMMARY"
	(cd "$CODE" && "$PY" -u -W ignore -c "
import resource
import orchestrator
from orchestrator import load_config, apply_pipeline_defaults, run_simulation
orchestrator.load_model = lambda automata, path: print('CA-only: ML model NOT loaded')
try:
    run_simulation(apply_pipeline_defaults(load_config('$cfg')))
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
	stopped=$(grep -o 'stopped early at step [0-9]*' "$log" | awk '{print $NF}')
	metrics=$(grep -E '^\s+(Precision|Recall|F1-Score|Jaccard):' "$log" | awk '{printf "%s %s  ", $1, $2}')
	echo "$name: done $(date +%T)  peak ${peak}MB  stopped_at ${stopped:-none}  $metrics" >> "$SUMMARY"
	"$PY" -W ignore "$CODE/spatial_validation/score_both_areas.py" \
		--final "$final" --label "$name" >> "$SUMMARY" 2>&1
done
echo "=== CA-only seeds finished $(date '+%F %T')" >> "$SUMMARY"
