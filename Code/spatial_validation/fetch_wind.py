"""Fetch actual historical wind data from NASA POWER for a fire incident.

Closes Gap 1.2 (wind not matched to actual weather). Queries the free NASA
POWER API (no signup, no key) for hourly 10-metre wind speed and direction
at the fire location on the fire date, then prints the values to put into
the simulation config (wind.speed_kmh / wind.direction_deg).

WD10M follows the meteorological convention -- degrees the wind blows FROM,
clockwise from north -- which is the same convention as the config's
direction_deg (315 = from the northwest). Copy the numbers straight across.

Usage (from Thesis_1/Code with the venv active):
    python spatial_validation/fetch_wind.py --date 20231212 --hour 11
    python spatial_validation/fetch_wind.py --date 20250422 --utm 606748.78 1141695.77

Defaults to the Sitio Santa Maria ignition point. --hour highlights the
fire hour (local standard time); Sta. Maria ignited around 11:31, so 11.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

DEFAULT_UTM = (606748.78, 1141695.77)  # Sitio Santa Maria ignition point (EPSG:32651)

API = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point"
    "?parameters=WS10M,WD10M"
    "&community=RE"
    "&longitude={lon:.5f}&latitude={lat:.5f}"
    "&start={date}&end={date}"
    "&format=JSON&time-standard=LST"
)


def utm_to_latlon(x: float, y: float) -> tuple[float, float]:
    from rasterio.warp import transform as warp_transform

    lons, lats = warp_transform("EPSG:32651", "EPSG:4326", [x], [y])
    return lats[0], lons[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="NASA POWER hourly wind lookup for a fire date")
    parser.add_argument("--date", required=True, help="Fire date as YYYYMMDD, e.g. 20231212")
    parser.add_argument("--hour", type=int, default=11,
                        help="Fire hour in local standard time, 0-23 (default 11 = 1131H)")
    parser.add_argument("--utm", nargs=2, type=float, default=DEFAULT_UTM,
                        metavar=("X", "Y"), help="Ignition point in EPSG:32651 metres")
    parser.add_argument("--lat", type=float, help="Latitude (overrides --utm)")
    parser.add_argument("--lon", type=float, help="Longitude (overrides --utm)")
    args = parser.parse_args()

    if args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
    else:
        lat, lon = utm_to_latlon(*args.utm)

    url = API.format(lat=lat, lon=lon, date=args.date)
    print(f"Location: lat {lat:.5f}, lon {lon:.5f}")
    print(f"Querying: {url}\n")

    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)

    params = data["properties"]["parameter"]
    ws = params["WS10M"]   # m/s, keyed by "YYYYMMDDHH"
    wd = params["WD10M"]   # degrees FROM north

    print(f"  Hour (LST)   Speed (km/h)   Direction (deg from)")
    speeds_kmh = {}
    for key in sorted(ws):
        hour = int(key[-2:])
        s, d = ws[key], wd[key]
        if s is None or s < 0 or d is None or d < 0:
            print(f"  {hour:02d}:00        (missing)")
            continue
        kmh = s * 3.6
        speeds_kmh[hour] = (kmh, d)
        marker = "   <-- fire hour" if hour == args.hour else ""
        print(f"  {hour:02d}:00        {kmh:6.1f}         {d:5.0f}{marker}")

    if not speeds_kmh:
        raise SystemExit("No valid data returned. Check the date (POWER lags ~1 week behind real time).")

    day_speed = sum(v[0] for v in speeds_kmh.values()) / len(speeds_kmh)
    if args.hour in speeds_kmh:
        fire_kmh, fire_dir = speeds_kmh[args.hour]
        print(f"\nAt the fire hour ({args.hour:02d}:00 LST): {fire_kmh:.1f} km/h from {fire_dir:.0f} deg")
        print("\nPut this in the config (wind section):")
        print(f"  speed_kmh: {fire_kmh:.1f}")
        print(f"  direction_deg: {fire_dir:.0f}")
    print(f"\nDaily mean speed for reference: {day_speed:.1f} km/h")
    print("(Current placeholder in default_experiment.yaml: 10.0 km/h from 315 deg.)")


if __name__ == "__main__":
    main()
