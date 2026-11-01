"""Frozen spatial design for the cropped R2 campaign; no input reads or execution."""
import json
from copy import deepcopy

import numpy as np

SCHEMA = "candidate_family_windows.v1"
CAMPAIGN_ID = "candidate-r2-cropped-1024-v1"
PARENT_SHAPE = (5489, 6896)
WINDOW_SIZE = 1024
ALIGNMENT = 128
MAX_UPDATES = 300

# Retained accepted preflight inventory, in its original family order.
FAMILIES = (
    ("p8f01_cluster_322_m1", 894, 2730),
    ("p8f02_cluster_1052_m3", 1198, 2523),
    ("p8f03_cluster_4974_m2", 1723, 1569),
    ("p8f04_cluster_160_m5", 791, 3164),
    ("p8v2f05_cluster_4146_m3", 1603, 2799),
    ("p8v2f06_cluster_6201_m5", 1789, 2602),
    ("p9f_m1_r1_c1462", 1249, 2018),
    ("p9f_m1_r2_c8733", 2028, 2384),
    ("p9f_m1_r3_c1085", 1197, 2270),
    ("p9f_m1_r4_c4476", 1649, 2050),
    ("p9f_m2_r1_c4632", 1657, 2397),
    ("p9f_m2_r2_c145", 785, 2989),
    ("p9f_m2_r3_c428", 971, 2887),
    ("p9f_m2_r4_c728", 1082, 2672),
    ("p9f_m3_r1_c11129", 2144, 2224),
    ("p9f_m3_r2_c1635", 1295, 2848),
    ("p9f_m3_r3_c810", 1080, 2994),
    ("p9f_m3_r4_c14671", 2371, 2255),
    ("p9f_m4_r1_c1224", 1211, 3178),
    ("p9f_m4_r2_c17934", 2603, 2903),
    ("p9f_m4_r3_c16550", 2508, 2824),
    ("p9f_m4_r4_c16769", 2540, 3469),
    ("p9f_m5_r1_c2957", 1504, 2888),
    ("p9f_m5_r2_c1794", 1329, 3062),
    ("p9f_m5_r3_c2560", 1427, 2686),
    ("p9f_m5_r4_c1748", 1306, 2146),
)


def frozen_design():
    """Return fresh primitive data; callers cannot mutate the frozen inventory."""
    windows = []
    for ordinal, (identity, row, col) in enumerate(FAMILIES, 1):
        origin = [ALIGNMENT * ((value - WINDOW_SIZE // 2) // ALIGNMENT)
                  for value in (row, col)]
        windows.append(dict(family_ordinal=ordinal, scenario_family_id=identity,
                            global_origin=origin, shape=[WINDOW_SIZE, WINDOW_SIZE]))
    return dict(schema_version=SCHEMA, campaign_id=CAMPAIGN_ID,
                rng_policy="prescribed_seed_crop_shaped_draw.v1",
                parent_shape=list(PARENT_SHAPE), family_windows=windows)


def validate_design(design, inventory, grid_shape, maximum_updates):
    """No adaptive windows, seed substitutions, type coercion or larger horizon."""
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False)
    if canonical(design) != canonical(frozen_design()):
        raise ValueError("Cropped campaign requires the exact frozen spatial design")
    if (canonical(grid_shape) != canonical(list(PARENT_SHAPE))
            or type(maximum_updates) is not int or maximum_updates != MAX_UPDATES
            or not isinstance(inventory, list) or len(inventory) != len(FAMILIES)):
        raise ValueError("Cropped parent geometry, horizon or inventory mismatch")
    for ordinal, (family, (identity, row, col), window) in enumerate(
            zip(inventory, FAMILIES, design["family_windows"], strict=True), 1):
        if (type(family.get("ordinal")) is not int or family["ordinal"] != ordinal
                or family.get("scenario_family_id") != identity
                or family.get("coordinate_system") != "grid_row_col"
                or canonical(family.get("ignition_points")) != canonical([[row, col]])):
            raise ValueError("Cropped family identity or prescribed ignition changed")
        origin = window["global_origin"]
        if any(o < 0 or o + WINDOW_SIZE > n for o, n in zip(origin, PARENT_SHAPE)):
            raise ValueError("Frozen crop is outside its parent grid")
        margin = min(row - origin[0], origin[0] + WINDOW_SIZE - 1 - row,
                     col - origin[1], origin[1] + WINDOW_SIZE - 1 - col)
        if margin <= maximum_updates:
            raise ValueError("Crop boundary could affect the admitted propagation horizon")


def validate_preparation(design, spec, prepared):
    window = design["family_windows"][spec["family_ordinal"] - 1]
    if (window["scenario_family_id"] != spec["scenario_family_id"]
            or tuple(window["global_origin"]) != prepared.global_origin
            or tuple(window["shape"]) != prepared.environment["grid_shape"]
            or prepared.full_grid_shape != PARENT_SHAPE
            or prepared.interpretation.get("window") != window["global_origin"] + window["shape"]):
        raise ValueError("Preparation does not bind the prescribed family crop")


def inventory_entry(window, prepared):
    """Primitive, outcome-blind inventory; no social labels inferred from materials."""
    e = prepared.environment
    receiving = prepared.receiving_mask
    return dict(window=deepcopy(window), mask_identities=prepared.mask_identities(),
        local_transform=list(e["transform"]), crs=str(e["crs"]),
        valid_cells=int(np.count_nonzero(e["burnable_mask"])),
        building_cells=int(np.count_nonzero(e["building_presence"])),
        receiving_cells=int(np.count_nonzero(receiving)),
        building_density=float(np.count_nonzero(e["building_presence"]) / receiving.size),
        receiving_material_counts={str(m): int(np.count_nonzero(receiving & (e["material_class"] == m)))
                                   for m in range(1, 6)})


def overlapping_windows(design):
    """Geometric overlaps are warnings, not independent leakage components."""
    result = []
    windows = design["family_windows"]
    for i, a in enumerate(windows):
        for b in windows[i + 1:]:
            extents = [max(0, min(x + n, y + m) - max(x, y))
                       for x, n, y, m in zip(a["global_origin"], a["shape"],
                                             b["global_origin"], b["shape"], strict=True)]
            if all(extents):
                result.append(dict(families=[a["scenario_family_id"], b["scenario_family_id"]],
                                   overlap_cells=extents[0] * extents[1]))
    return result
