#!/usr/bin/env bash
#
# Convert a hand-traced fire perimeter (KML from Google Earth Pro, or any
# OGR-readable vector) into a binary ground-truth GeoTIFF aligned exactly to
# the project raster grid, then verify the alignment.
#
# The grid parameters below are read off the existing reference ground truth
# (stack_ground_truth.tif) and must not be changed -- validate_simulation.py
# compares rasters cell-by-cell and will refuse a mismatched shape.
#
# Usage:
#   ./make_ground_truth.sh <input_trace.kml> <output_ground_truth.tif>
#
set -euo pipefail

# --- Reference grid (EPSG:32651, 6896 x 5489 @ 3 m) --------------------------
TARGET_CRS="EPSG:32651"
XMIN=598558.7784
YMIN=1127907.7699
XMAX=619246.7784
YMAX=1144374.7699
PIXEL=3.0
EXPECTED_W=6896
EXPECTED_H=5489

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFERENCE_TIF="${SCRIPT_DIR}/../processedData/raster/raster/stack_ground_truth.tif"
PYTHON="${SCRIPT_DIR}/../.venv/bin/python"

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <input_trace.kml> <output_ground_truth.tif>" >&2
    exit 1
fi

INPUT="$1"
OUTPUT="$2"

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: input trace not found: $INPUT" >&2
    exit 1
fi

TMPDIR_LOCAL="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT
REPROJECTED="${TMPDIR_LOCAL}/trace_utm.gpkg"

echo "==> Reprojecting trace to ${TARGET_CRS}"
ogr2ogr \
    -f GPKG \
    -t_srs "$TARGET_CRS" \
    -nln trace \
    -nlt POLYGON \
    -explodecollections \
    "$REPROJECTED" \
    "$INPUT"

FEATURES="$(ogrinfo -so "$REPROJECTED" trace | grep -i '^Feature Count' | awk '{print $3}')"
echo "    polygons found: ${FEATURES}"
if [[ "${FEATURES}" == "0" ]]; then
    echo "ERROR: no polygons in the trace. Did you save a path/placemark instead of a polygon?" >&2
    exit 1
fi

echo "==> Rasterizing onto the reference grid"
mkdir -p "$(dirname "$OUTPUT")"
rm -f "$OUTPUT"
gdal_rasterize \
    -l trace \
    -burn 1 \
    -init 0 \
    -ot Int16 \
    -a_srs "$TARGET_CRS" \
    -te "$XMIN" "$YMIN" "$XMAX" "$YMAX" \
    -tr "$PIXEL" "$PIXEL" \
    -co COMPRESS=LZW \
    "$REPROJECTED" \
    "$OUTPUT"

echo "==> Verifying alignment against the reference grid"
"$PYTHON" - "$OUTPUT" "$REFERENCE_TIF" "$EXPECTED_W" "$EXPECTED_H" <<'PY'
import sys
import numpy as np
import rasterio

out_path, ref_path, exp_w, exp_h = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

with rasterio.open(out_path) as new, rasterio.open(ref_path) as ref:
    problems = []
    if (new.width, new.height) != (exp_w, exp_h):
        problems.append(f"size {new.width}x{new.height} != expected {exp_w}x{exp_h}")
    if str(new.crs) != str(ref.crs):
        problems.append(f"CRS {new.crs} != reference {ref.crs}")
    if not np.allclose(np.array(new.transform).astype(float),
                       np.array(ref.transform).astype(float), atol=1e-6):
        problems.append("geotransform differs from the reference raster")

    arr = new.read(1)
    values = np.unique(arr)
    if not set(values.tolist()) <= {0, 1}:
        problems.append(f"raster is not binary; found values {values[:10].tolist()}")

    burned = int((arr == 1).sum())

print()
print(f"  Output:        {out_path}")
print(f"  Size:          {new.width} x {new.height}")
print(f"  CRS:           {new.crs}")
print(f"  Burned cells:  {burned:,}   (~{burned * 9 / 10000:.2f} hectares)")
print(f"  Reference fire (Sitio Sta. Maria) for comparison: 3,312 cells (~2.98 ha)")

if problems:
    print()
    print("  ALIGNMENT CHECK FAILED:")
    for p in problems:
        print(f"    - {p}")
    print("  Do NOT run validation with this raster; the metrics would be meaningless.")
    sys.exit(1)

print()
print("  Alignment check PASSED - grid matches the reference exactly.")

if burned == 0:
    print("  WARNING: zero burned cells. The trace probably falls outside the raster extent.")
elif burned < 300:
    print("  WARNING: very small burn area. Check you traced the scar, not a single roof.")
elif burned > 30000:
    print("  WARNING: very large burn area (>27 ha). Check you didn't trace the whole barangay.")
PY

echo
echo "Done. Next: get the ignition point in UTM (Step 4 of GUIDE_ground_truth_tracing.md)."
