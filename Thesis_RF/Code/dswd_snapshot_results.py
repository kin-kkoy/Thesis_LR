"""Array-only DSWD snapshot scoring preparation; no artifact reader or launcher."""
from copy import deepcopy
from statistics import mean, stdev
import math

import numpy as np

try:
    from . import orchestrator as runner
    from .validation_engine import evaluate_final_footprint_arrays
    from .modules.transition_observer import mask_sha256
except ImportError:
    import orchestrator as runner
    from validation_engine import evaluate_final_footprint_arrays
    from modules.transition_observer import mask_sha256


METRICS = ("f1_score", "recall", "precision", "accuracy", "jaccard_index")
PARITY_FIELDS = ("incident", "reference_sha256", "buildings_sha256", "crop_origin", "shape",
                 "crs", "transform", "validity_policy", "building_encoding", "native_metadata",
                 "domain_mask_sha256", "domain_cells", "reference_positive_included",
                 "reference_positive_total")


def alignment_table(*, design=None):
    """Recorded expectations; new ignition membership still needs native evidence."""
    if design not in (None, runner.PHASE16_DSWD_SHARED_PROTOCOL["design"]):
        raise ValueError("Unknown scoring design")
    protocol = runner.PHASE16_DSWD_SHARED_PROTOCOL if design else runner.PHASE16_DSWD_PROTOCOL
    return {name: {
        **({"design": design, "ignition_local_row_col": deepcopy(item["ignition"]),
            "ignition_global_row_col": [[r + item["crop_origin"][0], c + item["crop_origin"][1]]
                                       for r, c in item["ignition"]]} if design else {}),
        "incident": name, "reference_path": item["reference_path"],
        "reference_sha256": item["reference_sha256"],
        "reference_metadata": deepcopy(item["reference_metadata"]),
        "buildings_path": "processedData/raster/raster/stack_buildings.tif",
        "buildings_sha256": protocol["input_sha256"]["buildings"],
        "crop_origin": deepcopy(item["crop_origin"]), "shape": deepcopy(item["shape"]),
        "crs": protocol["alignment_policy"]["crs"], "transform": deepcopy(item["crop_transform"]),
        "validity_policy": protocol["alignment_policy"]["validity"],
        "building_encoding": deepcopy(protocol["alignment_policy"]["building_encoding"]),
        "native_metadata": None, "native_metadata_status": "NEEDS EVIDENCE: all four native layers",
        "domain_mask_sha256": item["rf_domain_mask_sha256"],
        "domain_mask_status": item["rf_domain_mask_status"],
        "domain_cells": item["expected_domain_cells"],
        "reference_positive_included": item["expected_reference_positive_included"],
        "reference_positive_total": item["expected_reference_positive_total"],
        "lr_side_confirmation": deepcopy(protocol["alignment_policy"]["lr_side_confirmation"]),
    } for name, item in protocol["incidents"].items()}


def verify_parity(rf, lr):
    """Counts alone never release cross-simulator production acceptance."""
    if (lr is None or not rf.get("domain_mask_sha256") or rf.get("native_metadata") is None
            or not isinstance(lr.get("confirmation_reference"), str)
            or not lr["confirmation_reference"].strip()):
        return False
    if any(key not in lr or lr[key] != rf[key] for key in PARITY_FIELDS):
        raise ValueError("RF/LR reference, geometry, validity or spatial domain parity differs")
    return True


def evaluate_snapshot_arrays(arrays, outcome, incident, *, native_metadata=None, lr_alignment=None,
                             design=None, arm=None):
    """Reuse shared reached-footprint metrics; references never enter runner stopping."""
    if (outcome.get("output_valid") is not True
            or outcome.get("status") not in ("complete", "censored")
            or outcome.get("termination_reason") not in ("inactive", "timestep_limit")):
        raise ValueError("Incomplete/failed snapshot cannot be scored")
    if design is not None:
        basis = runner._phase16_shared_basis(incident, arm)
        expected = {key: basis[key] for key in
                    ("design", "arm", "model_policy", "ignition_local_row_col", "ignition_global_row_col")}
        if (design != basis["design"] or any(outcome.get(key) != value for key, value in expected.items())
                or type(outcome["seed"]) is not int or outcome["seed"] not in range(42, 47)):
            raise ValueError("Shared scoring attribution differs")
    elif arm is not None or "design" in outcome or "arm" in outcome:
        raise ValueError("Explicit shared design and arm required")
    alignment = alignment_table(design=design)[incident]
    state = np.asarray(arrays["prediction"])
    shape = tuple(alignment["shape"])
    if (state.shape != shape or state.dtype != np.int8
            or any(np.shape(value) != shape for value in arrays.values())
            or not np.all(np.isin(state, (1, 2, 3, 4, 5)))):
        raise ValueError("Native full-state/crop contract differs")
    for name in ("prediction", "reference", "buildings", "environment"):
        if arrays[name + "_valid"].dtype != np.bool_:
            raise ValueError("Explicit native boolean validity masks required")
    if (not np.all(arrays["prediction_valid"]) or not np.all(arrays["environment_valid"])
            or not np.all(np.isin(arrays["environment"], (0, 1)))
            or not np.all(np.isin(arrays["reference"][arrays["reference_valid"]], (0, 1)))
            or not np.all(np.isin(arrays["buildings"][arrays["buildings_valid"]], (0, 10)))):
        raise ValueError("Snapshot/reference/building/environment value or validity contract differs")
    environment = arrays["environment"] == 1
    mapped = arrays["buildings"] == 10
    if not np.array_equal(state != 1, environment & arrays["buildings_valid"] & mapped):
        raise ValueError("State eligibility differs from environmental-valid mapped buildings")
    domain = (environment & arrays["prediction_valid"] & arrays["reference_valid"]
              & arrays["buildings_valid"] & mapped)
    reference = arrays["reference"] == 1
    observed = (int(np.count_nonzero(domain)), int(np.count_nonzero(domain & reference)),
                int(np.count_nonzero(arrays["reference_valid"] & reference)))
    if observed != (alignment["domain_cells"], alignment["reference_positive_included"],
                    alignment["reference_positive_total"]):
        raise ValueError("Historical incident domain/positive expectations differ; do not intersect to fit")
    alignment.update(domain_mask_sha256=mask_sha256(domain), domain_mask_status="OBSERVED_ARRAYS",
                     native_metadata=deepcopy(native_metadata))
    if native_metadata is not None:
        if (set(native_metadata) != {"prediction", "reference", "buildings", "environment"}
                or any(set(value) != {"dtype", "nodata", "mask_flags"}
                       for value in native_metadata.values())):
            raise ValueError("All four native dtype/nodata/mask declarations required")
        alignment["native_metadata_status"] = "Supplied by separately reviewed native reader"
    parity = False if design else verify_parity(alignment, lr_alignment)
    metrics = evaluate_final_footprint_arrays(
        state, reference, environment, arrays["prediction_valid"], arrays["reference_valid"],
        arrays["buildings_valid"], mapped, termination_reason=outcome["termination_reason"])
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    metrics["accuracy"] = (tn + tp) / (tn + fp + fn + tp)
    metrics["reference_kind"] = "Owner-supplied approximate reference; provenance/registration qualified"
    reached = int(np.count_nonzero(np.isin(state, (3, 4, 5))))
    active = int(np.count_nonzero(np.isin(state, (3, 4))))
    if "reached" in arrays and (arrays["reached"].dtype != np.uint8
            or not np.array_equal(arrays["reached"], np.isin(state, (3, 4, 5)).astype(np.uint8))):
        raise ValueError("Native-state and cumulative reached export disagree")
    counts = {name: int(np.count_nonzero(state == code)) for code, name in enumerate(
        ("non_burnable", "not_yet_burning", "ignited", "blazing", "extinguished"), 1)}
    if (reached != outcome["cumulative_reached_count"] or active != outcome["active_cell_count"]
            or counts != outcome["state_counts"]
            or (outcome["status"] == "complete") != (outcome["termination_reason"] == "inactive")
            or (outcome["termination_reason"] == "timestep_limit" and active == 0)):
        raise ValueError("Snapshot counts/termination/censoring metadata disagree")
    return {**({"design": design, "arm": arm, "model_policy": deepcopy(outcome["model_policy"])} if design else {}),
            "incident": incident, "seed": outcome["seed"],
            "assumed_minutes_per_update": outcome["assumed_minutes_per_update"],
            "outcome": deepcopy(outcome), "metrics": metrics, "alignment": alignment,
            "parity_status": "MATCH" if parity else "NEEDS EVIDENCE",
            "accepted_for_cross_simulator_comparison": parity,
            "qualification": "F01/Airbase development; assumed clock; no physical/held-out claim"}


def summarize_five_seed_groups(rows, *, design=None):
    """Six legacy or twelve shared-design groups; never pool models/incidents/clocks."""
    if design not in (None, runner.PHASE16_DSWD_SHARED_PROTOCOL["design"]):
        raise ValueError("Unknown reporting design")
    shared = design is not None or any("design" in row or "arm" in row for row in rows)
    protocol = runner.PHASE16_DSWD_SHARED_PROTOCOL if shared else runner.PHASE16_DSWD_PROTOCOL
    groups = {}
    for row in rows:
        if shared:
            basis = runner._phase16_shared_basis(row["incident"], row.get("arm"))
            if (row.get("design") != basis["design"] or row.get("model_policy") != basis["model_policy"]
                    or row["outcome"].get("arm") != row["arm"]
                    or row["outcome"].get("design") != basis["design"]
                    or row["outcome"].get("model_policy") != basis["model_policy"]
                    or row["outcome"].get("seed") != row["seed"]):
                raise ValueError("Mixed design/model/arm reporting attribution")
            horizons = [h for h in basis["horizons"]
                        if h["assumed_minutes_per_update"] == row["assumed_minutes_per_update"]]
            if (len(horizons) != 1 or any(row["outcome"].get(k) != v for k, v in horizons[0].items())
                    or row["outcome"].get("incident") != row["incident"]
                    or any(row["outcome"].get(k) != basis[k] for k in
                           ("ignition_local_row_col", "ignition_global_row_col"))):
                raise ValueError("Shared reporting clock/horizon/incident/ignition differs")
        key = (row["incident"], row.get("arm") if shared else None, row["assumed_minutes_per_update"])
        groups.setdefault(key, []).append(row)
    expected = {(incident, arm, clock["minutes_per_update"])
                for incident in protocol["incidents"]
                for arm in (protocol["model_policies"] if shared else (None,))
                for clock in protocol["clocks"]}
    if set(groups) - expected:
        raise ValueError("Unexpected incident/clock group")
    result = []
    for incident, arm, clock in sorted(expected):
        group = groups.get((incident, arm, clock), [])
        seeds = [row["seed"] for row in group]
        complete = (len(group) == 5 and sorted(seeds) == [42, 43, 44, 45, 46]
                    and all(row["outcome"]["output_valid"] is True
                            and row["outcome"]["status"] in ("complete", "censored") for row in group))
        entry = {**({"design": protocol["design"], "arm": arm} if shared else {}),
                 "incident": incident, "assumed_minutes_per_update": clock,
                 "n": len(group), "status": "complete" if complete else "incomplete",
                 "accepted_for_cross_simulator_comparison": not shared and complete and all(
                     row.get("accepted_for_cross_simulator_comparison") is True for row in group),
                 "mean_sample_sd": None, "trajectories_independent_across_clocks": False}
        if complete:
            values = {key: [row["metrics"][key] for row in group] for key in METRICS}
            if shared:
                for name, r, c in (("tn", 0, 0), ("fp", 0, 1), ("fn", 1, 0), ("tp", 1, 1)):
                    values[name] = [row["metrics"]["confusion_matrix"][r][c] for row in group]
            values.update({key: [row["outcome"][key] for row in group] for key in
                           ("cumulative_reached_count", "active_cell_count", "actual_capture_update")})
            if any(type(x) not in (int, float) or not math.isfinite(x)
                   for data in values.values() for x in data):
                raise ValueError("Finite numeric reporting values required")
            entry["mean_sample_sd"] = {key: {"mean": mean(data), "sample_sd": stdev(data)}
                                       for key, data in values.items()}
        result.append(entry)
    return result
