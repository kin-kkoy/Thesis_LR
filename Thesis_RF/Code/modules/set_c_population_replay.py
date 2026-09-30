"""Disabled, aggregate-only replay of the approved Gate 6 leakage policy."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .set_c_pilot import publish_aggregate_report
from .set_c_population_adequacy import (
    AGGREGATE_SCHEMA_VERSION, APPROVED_CANDIDATE_SHA256,
    APPROVED_INVENTORY_FILE_SHA256, APPROVED_INVENTORY_PAYLOAD_SHA256,
    APPROVED_PRIMARY_SHA256, APPROVED_RESERVE_SHA256, BLOCK_ORIGIN, BLOCK_SIZE,
    EXPECTED_FAMILY_COUNT, EXPECTED_RUN_COUNT, EXPANSION_ID, ROLE_REQUIREMENTS,
    WIND_CONDITIONS, _UnionFind,
    _canonical_json, _require_temp_descendant, _sha256_file,
    _unexposed_allocation_feasible, _validate_aggregate_only,
    _validate_source_context, PopulationAdequacyError,
)
from .transition_observer import canonical_metadata_hash


REPLAY_ID = "phase9_set_c_population_adequacy_policy_replay_v3"
REPLAY_SCHEMA = "phase9_set_c_population_adequacy_policy_replay.v3"
SOURCE_FILE_SHA256 = "a998601d6f4db3ca5536633e17d6fa11ecc7dd99465c345b43bc960ef9e693c8"
SOURCE_PAYLOAD_SHA256 = "b4ff93b232e5ee55ff70d6d0469d858f47382cbbf80fa9eed0b1a4011450c197"
SOURCE_REVISION = "dfc37033526de483e2a35727cd01df597424257e"
LEGACY_V2_DUPLICATE_SEMANTICS = "sha256_canonical_float32_feature_row.v1"
SOURCE_RELATIVE_PATH = Path("output/phase9/phase9_set_c_population_adequacy_edge_diagnostic_v2.aggregate.json")
RESERVED_DESTINATION = Path("output/phase9/phase9_set_c_population_adequacy_policy_replay_v3.aggregate.json")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PopulationAdequacyError(f"V2 replay source invalid: {message}")


def _integer(value: object, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, "invalid nonnegative count")
    return value


def _hash(value: object) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdef" for c in value), "invalid hash")
    return value


def _load_bound_report(path: Path, *, file_hash: str, payload_hash: str) -> dict:
    _require(_sha256_file(path) == file_hash, "whole-file hash mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(report, dict), "report is not an object")
    _require(report.pop("payload_sha256", None) == payload_hash, "payload hash mismatch")
    _require(sha256(_canonical_json(report)).hexdigest() == payload_hash,
             "self-hash mismatch")
    _validate_aggregate_only(report)
    return report


def _components(report: dict, *, revision: str, require_production: bool) -> tuple[list[dict], int]:
    _require(report.get("aggregate_schema_version") == AGGREGATE_SCHEMA_VERSION
             and report.get("expansion_id") == EXPANSION_ID
             and report.get("status") == "complete", "schema or completion drift")
    provenance = report.get("provenance")
    _require(isinstance(provenance, dict), "missing provenance")
    _validate_source_context(provenance, synthetic=not require_production)
    _require(provenance.get("source_revision") == revision, "source revision drift")
    _require(report.get("approved_inventory") == {
        "payload_sha256": APPROVED_INVENTORY_PAYLOAD_SHA256,
        "file_sha256": APPROVED_INVENTORY_FILE_SHA256,
        "ordered_candidate_sha256": APPROVED_CANDIDATE_SHA256,
        "ordered_primary_sha256": APPROVED_PRIMARY_SHA256,
        "ordered_reserve_sha256": APPROVED_RESERVE_SHA256,
    }, "approved inventory drift")
    contract = report.get("population_contract", {})
    _require(contract.get("family_count") == EXPECTED_FAMILY_COUNT
             and contract.get("run_count") == EXPECTED_RUN_COUNT
             and contract.get("existing_family_count") == 6
             and contract.get("approved_primary_family_count") == 20
             and contract.get("wind_condition_count") == 17
             and contract.get("replicates_per_condition") == 2
             and contract.get("run_concurrency") == 1
             and contract.get("block_size") == list(BLOCK_SIZE)
             and contract.get("block_origin") == list(BLOCK_ORIGIN)
             and contract.get("split_roles_assigned") is False,
             "population or spatial contract drift")
    expected_roles = {
        name: {"components": n, "positive_1": positive, "negative_0": negative,
               "material_classes": [1, 2, 3, 4, 5],
               "maximum_single_component_positive_share": 0.5}
        for name, n, positive, negative in ROLE_REQUIREMENTS
    }
    _require(contract.get("prospective_role_minima") == expected_roles,
             "role minima drift")
    aggregate = report.get("aggregate", {})
    families = aggregate.get("scenario_families", ())
    runs = aggregate.get("runs", ())
    _require(isinstance(families, list) and len(families) == EXPECTED_FAMILY_COUNT
             and isinstance(runs, list) and len(runs) == EXPECTED_RUN_COUNT,
             "family or run count drift")
    family_by_id = {}
    family_totals = defaultdict(lambda: [0, 0])
    for order, family in enumerate(families, 1):
        _require(isinstance(family, dict), "family record malformed")
        fid = family.get("scenario_family_id")
        _require(isinstance(fid, str) and fid and fid not in family_by_id,
                 "family order or identity drift")
        _require(family.get("source_tier") == ("existing" if order <= 6 else "primary")
                 and family.get("material_class") in (1, 2, 3, 4, 5)
                 and family.get("run_count") == 34,
                 "family material or tier drift")
        family_by_id[fid] = family
    run_ids = []
    for order, run in enumerate(runs, 1):
        _require(isinstance(run, dict), "run record malformed")
        family = families[(order - 1) // 34]
        wind = ((order - 1) % 34) // 2 + 1
        replicate = ((order - 1) % 2) + 1
        fid = family["scenario_family_id"]
        seed = 9_000_000 + 10_000 * ((order - 1) // 34 + 1) + 100 * wind + replicate
        speed, direction = WIND_CONDITIONS[wind - 1]
        scenario_id = f"{fid}__w{direction:03d}_s{speed:02d}"
        termination = run.get("termination", {})
        counts = run.get("class_counts", {})
        negative = _integer(counts.get("negative_0"))
        positive = _integer(counts.get("positive_1"))
        _require(run.get("order") == order and run.get("scenario_family_id") == fid
                 and run.get("wind_condition_ordinal") == wind
                 and run.get("replicate_ordinal") == replicate
                 and run.get("seed") == seed
                 and run.get("scenario_id") == scenario_id
                 and run.get("run_id") == f"{scenario_id}__rep{replicate}__seed_{seed}"
                 and run.get("rows") == negative + positive
                 and termination.get("reason") == "inactive"
                 and termination.get("completeness_status") == "complete"
                 and termination.get("authoritative") is True
                 and termination.get("expected_transition_count") ==
                     termination.get("captured_transition_count") == run.get("observation_count"),
                 "run order, counts, seed, or termination drift")
        for key in ("configuration_sha256", "run_identity_sha256", "provenance_sha256"):
            _hash(run.get(key))
        family_totals[fid][0] += negative
        family_totals[fid][1] += positive
        run_ids.append(run["run_id"])
    _require(contract.get("run_order_sha256") == canonical_metadata_hash(run_ids),
             "run order hash mismatch")
    for family in families:
        counts = family["class_counts"]
        _require(counts == dict(zip(("negative_0", "positive_1"),
                                    family_totals[family["scenario_family_id"]])),
                 "family totals mismatch")
    total_negative = sum(value[0] for value in family_totals.values())
    total_positive = sum(value[1] for value in family_totals.values())
    _require(aggregate.get("totals") == {
        "rows": total_negative + total_positive,
        "class_counts": {"negative_0": total_negative, "positive_1": total_positive},
        "family_count": EXPECTED_FAMILY_COUNT, "run_count": EXPECTED_RUN_COUNT,
    }, "global totals mismatch")
    integrity = report.get("integrity", {})
    _require(isinstance(integrity, dict) and all(integrity.get(key) is False for key in (
        "row_level_payload_retained", "row_level_payload_published", "split_roles_assigned",
        "estimator_accessed", "protected_spatial_reference_accessed", "training_performed",
        "calibration_performed", "raster_checkpoint_or_model_written"))
        and integrity.get("sole_persistent_output_is_aggregate_json") is True,
        "protected-operation integrity drift")
    attribution = aggregate.get("leakage", {}).get("edge_attribution", {})
    links = attribution.get("family_pair_links", ())
    _require(isinstance(links, list), "edge attribution missing")
    union = _UnionFind()
    for family in families:
        union.find(family["scenario_family_id"])
    seen = set()
    collision_pairs = 0
    for link in links:
        _require(isinstance(link, dict), "edge link malformed")
        pair = link.get("family_ordinals")
        _require(isinstance(pair, list) and len(pair) == 2
                 and all(type(v) is int for v in pair)
                 and 1 <= pair[0] < pair[1] <= EXPECTED_FAMILY_COUNT
                 and tuple(pair) not in seen, "family pair order drift")
        seen.add(tuple(pair))
        duplicate = _integer(link.get("duplicate_identity_count"))
        block = _integer(link.get("spatial_block_count"))
        _require(duplicate + block > 0, "empty edge link")
        collision_pairs += duplicate > 0
        if block:
            union.union(families[pair[0] - 1]["scenario_family_id"],
                        families[pair[1] - 1]["scenario_family_id"])
    _require(links == sorted(links, key=lambda item: item["family_ordinals"]),
             "edge order drift")
    _require(attribution.get("block_only", {}).get("component_count") == len({
        union.find(family["scenario_family_id"]) for family in families
    }), "block component count mismatch")
    groups = defaultdict(list)
    for family in families:
        groups[union.find(family["scenario_family_id"])].append(family)
    sizes = sorted((len(group) for group in groups.values()), reverse=True)
    _require(attribution["block_only"].get("family_counts") == sizes,
             "block component sizes mismatch")
    components = []
    for group in sorted(groups.values(), key=lambda group: sorted(
            family["scenario_family_id"] for family in group)):
        ids = tuple(sorted(family["scenario_family_id"] for family in group))
        negative = sum(family_totals[fid][0] for fid in ids)
        positive = sum(family_totals[fid][1] for fid in ids)
        components.append({
            "component_sha256": canonical_metadata_hash({
                "block_size": BLOCK_SIZE, "scenario_family_ids": ids}),
            "family_count": len(ids), "run_count": 34 * len(ids),
            "rows": negative + positive,
            "class_counts": {"negative_0": negative, "positive_1": positive},
            "material_classes": sorted({family_by_id[fid]["material_class"] for fid in ids}),
            "qualifies": negative >= 5 and positive >= 5,
        })
    return components, collision_pairs


def replay_policy_v3(
    source: Path, destination: Path, *, synthetic_test: bool = False,
    expected_file_sha256: str = SOURCE_FILE_SHA256,
    expected_payload_sha256: str = SOURCE_PAYLOAD_SHA256,
    expected_revision: str = SOURCE_REVISION,
) -> dict[str, object]:
    """Test a v2-shaped temporary report; production execution stays closed."""
    if not synthetic_test:
        raise PermissionError("Production policy replay requires separate one-run approval")
    source = _require_temp_descendant(source, "replay source")
    destination = _require_temp_descendant(destination, "replay destination")
    if source == destination or destination.exists():
        raise FileExistsError("Replay destination exists or aliases source")
    report = _load_bound_report(source, file_hash=_hash(expected_file_sha256),
                                payload_hash=_hash(expected_payload_sha256))
    components, collision_pairs = _components(
        report, revision=expected_revision, require_production=False)
    if expected_revision == SOURCE_REVISION:
        _require(collision_pairs == 97, "approved feature collision count drift")
    qualifying = sum(component["qualifies"] is True for component in components)
    gates = dict(report["adequacy"]["gates"])
    _require(set(gates) == {
        "all_884_runs_complete_inactive_authoritative_and_count_matched",
        "at_least_23_qualifying_components",
        "every_qualifying_component_has_five_of_each_class",
        "unexposed_role_minima_material_coverage_and_concentration_feasible",
        "one_live_run_and_bounded_complete_batch_draining",
        "no_estimator_protected_input_row_split_or_model_output",
    } and all(gates[k] is True for k in (
        "all_884_runs_complete_inactive_authoritative_and_count_matched",
        "one_live_run_and_bounded_complete_batch_draining",
        "no_estimator_protected_input_row_split_or_model_output")),
        "original execution safeguards failed")
    gates["at_least_23_qualifying_components"] = qualifying >= 23
    gates["every_qualifying_component_has_five_of_each_class"] = all(
        component["qualifies"] for component in components)
    gates["unexposed_role_minima_material_coverage_and_concentration_feasible"] = (
        _unexposed_allocation_feasible(components))
    result = {
        "replay_schema_version": REPLAY_SCHEMA,
        "replay_id": REPLAY_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file_sha256": expected_file_sha256,
        "source_payload_sha256": expected_payload_sha256,
        "source_revision": expected_revision,
        "population_run_count": EXPECTED_RUN_COUNT,
        "policy": "scenario_family_stable_cell_spatial_block_provenance_record; predictor_collisions_audit_only",
        "source_duplicate_semantics": LEGACY_V2_DUPLICATE_SEMANTICS,
        "feature_collision_family_pair_count": collision_pairs,
        "leakage": {
            "component_count": len(components),
            "qualifying_component_count": qualifying,
            "component_family_counts": sorted((c["family_count"] for c in components), reverse=True),
            "role_identities_exposed": False,
        },
        "adequacy": {"status": "pass" if all(gates.values()) else "fail", "gates": gates},
        "integrity": {"row_level_payload_published": False,
                      "split_roles_assigned": False, "estimator_accessed": False,
                      "protected_spatial_reference_accessed": False},
    }
    _validate_aggregate_only(result)
    path, payload = publish_aggregate_report(result, destination)
    return {"path": path, "payload_sha256": payload,
            "adequacy_status": result["adequacy"]["status"]}
