"""Convert a Google Earth KML trace into a ground-truth GeoTIFF on the reference grid.

Uses only rasterio + stdlib (no GDAL CLI, shapely, fiona, pyproj needed).

- Every <Polygon> in the KML is burned as 1 (holes / innerBoundaryIs are cut out).
- Every <Point> placemark is reported in UTM (use one for the ignition spot).

Usage:
  .venv/bin/python spatial_validation/kml_to_gt.py TRACE.kml OUT.tif
  [--ref processedData/raster/raster/stack_ground_truth.tif]
"""
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform as warp_transform

DEFAULT_REF = "processedData/raster/raster/stack_ground_truth.tif"


def local(tag):
    return tag.rsplit("}", 1)[-1]


def find_all(elem, name):
    return [e for e in elem.iter() if local(e.tag) == name]


def parse_coords(text):
    lons, lats = [], []
    for tok in text.split():
        parts = tok.split(",")
        lons.append(float(parts[0]))
        lats.append(float(parts[1]))
    return lons, lats


def to_utm(lons, lats, crs):
    xs, ys = warp_transform("EPSG:4326", crs, lons, lats)
    return list(zip(xs, ys))


def ring_of(boundary_elem, crs):
    coords = find_all(boundary_elem, "coordinates")[0].text
    return to_utm(*parse_coords(coords), crs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kml")
    ap.add_argument("out")
    ap.add_argument("--ref", default=DEFAULT_REF)
    args = ap.parse_args()

    ref = rasterio.open(args.ref)
    crs = ref.crs
    root = ET.parse(args.kml).getroot()

    shapes = []
    for pm in find_all(root, "Placemark"):
        name_el = [e for e in pm if local(e.tag) == "name"]
        name = name_el[0].text if name_el else "(no name)"
        for poly in find_all(pm, "Polygon"):
            outer = [e for e in poly if local(e.tag) == "outerBoundaryIs"]
            inner = [e for e in poly if local(e.tag) == "innerBoundaryIs"]
            rings = [ring_of(outer[0], crs)] + [ring_of(i, crs) for i in inner]
            shapes.append(({"type": "Polygon", "coordinates": rings}, 1))
            print(f"polygon '{name}': {len(rings[0])} vertices, {len(inner)} hole(s)")
        for pt in find_all(pm, "Point"):
            lons, lats = parse_coords(find_all(pt, "coordinates")[0].text)
            (x, y), = to_utm(lons, lats, crs)
            row, col = ref.index(x, y)
            print(f"point '{name}': lon {lons[0]:.6f} lat {lats[0]:.6f} -> "
                  f"UTM {x:.2f}, {y:.2f} (row {row}, col {col})")

    if not shapes:
        raise SystemExit("No polygons found in KML.")

    arr = rasterize(shapes, out_shape=(ref.height, ref.width),
                    transform=ref.transform, fill=0, dtype="int16")

    profile = ref.profile.copy()
    profile.update(dtype="int16", count=1, nodata=None, compress="lzw")
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(arr, 1)

    n = int(arr.sum())
    rows, cols = np.nonzero(arr)
    cx, cy = ref.xy(int(rows.mean()), int(cols.mean()))
    print(f"wrote {args.out}: {n} burned cells ({n * 9 / 1e4:.2f} ha), "
          f"rows {rows.min()}-{rows.max()}, cols {cols.min()}-{cols.max()}, "
          f"centroid UTM {cx:.1f}, {cy:.1f}")

    chk = rasterio.open(args.out)
    assert chk.crs == ref.crs and chk.transform == ref.transform
    assert (chk.width, chk.height) == (ref.width, ref.height)
    print("grid check: CRS, transform, shape match reference")


if __name__ == "__main__":
    main()
