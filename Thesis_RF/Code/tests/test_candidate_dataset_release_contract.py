"""Synthetic release checks only; future tmp_path writes require separate approval."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
import hashlib
import json

import numpy as np
import pytest

from modules import candidate_dataset_collection as collection
from modules import candidate_dataset_release as release
import run_candidate_dataset as runner
from test_candidate_dataset_collection_contract import context


def noop(_):
    pass


def rows_fixture():
    _, config, _, source, _ = context()
    config["grid"]["shape"] = [4000, 4000]
    plan = collection.planned_runs(config["inventory"])
    rows = []
    for family in config["inventory"]:
        spec = next(s for s in plan if s["family_ordinal"] == family["ordinal"])
        for i in range(100):
            features = np.array([0, 0, 1, .5, i % 5 + 1, 0, 0, 1, 1, .5, .125],
                                dtype=np.float32)
            label = int(i < 20)
            p = dict(identity_version="sha256_observation_record_provenance.v2",
                scenario_family_id=spec["scenario_family_id"], scenario_id=spec["scenario_id"],
                run_id=spec["run_id"], seed=spec["seed"], timestep_t=0, timestep_t1=1,
                grid_id=source["grid_id"], feature_schema_version="set_c.v1",
                target_version="ignition_t_plus_1.v1", run_identity_sha256="a" * 64,
                provenance_sha256="b" * 64, cell_row=128 * family["ordinal"], cell_col=i)
            rows.append(dict(observation_id=release.digest_bytes(release.json_bytes(p)),
                row_sha256=hashlib.sha256(features.astype("<f4").tobytes() + bytes([label])).hexdigest(),
                q32_bits=0x3F000000, payload=p, csv=release.row_bytes(features, label).decode("ascii"),
                order=[family["ordinal"], 1, 1, 0, p["cell_row"], i],
                event_id=family["event_id"], material_class=int(features[4]),
                grid_id=source["grid_id"], scenario_family_id=spec["scenario_family_id"]))
    eligibility = {f["scenario_family_id"]: dict(eligible=True, prior_use=f["prior_use"],
                   assessment_reference="synthetic custody substitute") for f in config["inventory"]}
    return rows, config, source, eligibility


def publication_fixture(tmp_path):
    rows, config, source, eligibility = rows_fixture()
    package = tmp_path / "candidate-v1"
    authority = dict(source=source, custody=dict(approval_reference="synthetic custody",
        families=eligibility), publication_approval_reference="synthetic publication",
        destinations=dict(evidence=str(tmp_path), package=str(package)), allocation_node_budget=10000)
    # Controlled pure publisher fixture, not evidence of a collection campaign.
    execution_sha = release.digest_bytes(release.json_bytes(
        runner.execution_payload(config, authority)))
    binding = runner.publication_payload(execution_sha, authority)
    h = hashlib.sha256(release.HEADER)
    for row in rows:
        h.update(row["csv"].encode("ascii"))
    provenance = dict(binding=binding, dataset_sha256=h.hexdigest(), configuration=config,
        execution_authority=authority, teacher_id=config["teacher_id"],
        numerical_contract_id=config["numerical_contract_id"])
    # Fabricated completion metadata for this pure publication fixture; no campaign executes.
    provenance["run_records"] = []
    for spec in collection.planned_runs(config["inventory"]):
        selected = [r for r in rows if r["payload"]["run_id"] == spec["run_id"]]
        positives = sum(release.decode_row(r["csv"])[1] for r in selected)
        provenance["run_records"].extend([
            dict(kind="initialization", spec=spec, completed=dict(state_timestep=0,
                committed_rows=0, state_counts=dict(blazing=1))),
            dict(kind="inactive", spec=spec, completed=dict(state_timestep=1,
                committed_rows=len(selected), state_counts=dict(ignited=0, blazing=0,
                extinguished=1 + positives)))])
    provenance["evidence_files"] = {name: dict(bytes=1, sha256="0" * 64)
                                   for name in ("transitions.jsonl", "runs.jsonl")}
    items = release.components(rows, [], eligibility, noop)
    kwargs = dict(payload=tmp_path / "payload", destination=package, authority=provenance,
        check=noop, publication_authorized=True,
        expected_binding_sha256=release.digest_bytes(release.json_bytes(binding)))
    return rows, items, kwargs


def test_independent_golden_serialization_signed_zero_and_exponents():
    values = np.array([0, -0.0, 1, .5, 5, 10, -1, 1, 8, .25, 1e9], dtype=np.float32)
    expected = b"0,-0,1,0.5,5,10,-1,1,8,0.25,1e+09,1\n"
    assert release.row_bytes(values, 1) == expected
    expected_header = (b"slope_risk,proximity_risk,building_presence,material_risk,material_class,"
        b"wind_speed,wind_sin,wind_cos,neighbor_burning_count,composite_flammability,"
        b"wind_weighted_score,Ignited\n")
    assert release.HEADER == expected_header
    for value, text in ((np.float32(1e-5), "9.99999975e-06"),
                        (np.float32(1e-4), "9.99999975e-05"),
                        (np.float32(100000000), "100000000")):
        assert release.predictor_text(value) == text
    parsed, label = release.decode_row(expected.decode("ascii"))
    assert label == 1
    assert np.array_equal(parsed.view(np.uint32), values.view(np.uint32))
    assert parsed[1].view(np.uint32) == 0x80000000


@pytest.mark.parametrize("change", ["dtype", "shape", "nan", "label", "row_format"])
def test_serialization_rejects_noncanonical_inputs(change):
    values = np.zeros(11, dtype=np.float32)
    with pytest.raises((ValueError, TypeError)):
        if change == "dtype":
            release.row_bytes(values.astype(np.float64), 0)
        elif change == "shape":
            release.row_bytes(values[:10], 0)
        elif change == "nan":
            values[0] = np.nan
            release.row_bytes(values, 0)
        elif change == "label":
            release.row_bytes(values, True)
        else:
            release.decode_row("0," * 11 + "0\r\n")


def test_actual_capture_reconciliation_global_coordinates_and_tampering():
    _, _, _, _, construct = context(origin=(128, 256))
    engine = construct()
    batch = engine.step()
    envelope, rows = release.reconcile_batch(batch, engine.binding)
    assert len(rows) == len(batch.labels)
    assert all(r["payload"]["cell_row"] >= 128 and r["payload"]["cell_col"] >= 256 for r in rows)
    assert [r["q32_bits"] for r in rows] == batch.q32_bits.tolist()
    bad = deepcopy(engine.binding)
    bad["spec"]["seed"] += 1
    with pytest.raises(ValueError):
        release.reconcile_batch(batch, bad)
    if rows:
        rows[0]["payload"]["cell_row"] += 1
        with pytest.raises(ValueError):
            release.validate_rows(rows, noop)


def test_unconditional_family_event_cell_block_and_repeated_observation_edges():
    rows, _, _, eligibility = rows_fixture()
    # First two rows share a family even when only one event is complete.
    subset = [deepcopy(rows[i]) for i in (0, 1, 100, 200)]
    subset[0]["event_id"] = "complete"
    subset[1]["event_id"] = "other"
    subset[2]["event_id"] = "complete"
    subset[3]["payload"]["cell_row"] = subset[2]["payload"]["cell_row"]
    subset[3]["payload"]["cell_col"] = subset[2]["payload"]["cell_col"]
    small_eligibility = {r["scenario_family_id"]: eligibility[r["scenario_family_id"]] for r in subset}
    grouped = release.components(subset, ["complete"], small_eligibility, noop)
    assert len(grouped) == 1 and len(grouped[0]["indices"]) == 4
    subset[3]["payload"]["cell_row"] += 1
    grouped = release.components(subset, ["complete"], small_eligibility, noop)
    assert len(grouped) == 1  # shared raster-anchored block still contains it
    subset[3]["payload"]["cell_row"] += 128
    subset[3]["observation_id"] = subset[0]["observation_id"]
    with pytest.raises(ValueError, match="duplication"):
        release.components(subset, ["complete"], small_eligibility, noop)


def test_component_identity_partition_and_no_outcome_deduplication():
    rows, _, _, eligibility = rows_fixture()
    release.validate_rows(rows, noop)
    items = release.components(rows, [], eligibility, noop)
    assert len(items) == 26
    assert sorted(i for c in items for i in c["indices"]) == list(range(len(rows)))
    ids = sorted(rows[i]["observation_id"] for i in items[0]["indices"])
    independent_bytes = ("[" + ",".join('"' + v + '"' for v in ids) + "]").encode("ascii")
    assert items[0]["component_id"] == hashlib.sha256(independent_bytes).hexdigest()
    duplicated = rows + [deepcopy(rows[0])]
    with pytest.raises(ValueError, match="Repeated observation"):
        release.validate_rows(duplicated, noop)


def tiny_items():
    return [dict(positives=5, negatives=5, materials=[1, 2, 3, 4, 5],
                 final_test_eligible=True) for _ in range(8)]


def test_tiny_lexicographic_solution_matches_independent_enumeration():
    items = tiny_items()
    independently_feasible = [v for v in product(range(4), repeat=8)
                              if all(v.count(role) == 2 for role in range(4))]
    expected = min(independently_feasible)
    actual = release.lexicographic_assignment(items, (2, 2, 2, 2), ((10, 10),) * 4,
        node_budget=10000, check=noop)
    assert actual == expected == (0, 0, 1, 1, 2, 2, 3, 3)


def test_infeasible_and_budget_incomplete_are_distinct_and_nonpassing():
    items = tiny_items()
    with pytest.raises(release.AllocationIncomplete):
        release.lexicographic_assignment(items, (2, 2, 2, 2), ((10, 10),) * 4,
            node_budget=1, check=noop)
    for item in items:
        item["final_test_eligible"] = False
    with pytest.raises(release.AllocationInfeasible):
        release.lexicographic_assignment(items, (2, 2, 2, 2), ((10, 10),) * 4,
            node_budget=100000, check=noop)


def test_allocation_adequacy_order_counts_and_final_test_restrictions():
    rows, _, _, eligibility = rows_fixture()
    items = release.components(rows, [], eligibility, noop)
    h = hashlib.sha256(release.HEADER + b"".join(r["csv"].encode("ascii") for r in rows)).hexdigest()
    assignment = release.allocate(items, h, node_budget=10000, check=noop)
    ordered = sorted(items, key=lambda c: (
        hashlib.sha256((h + ":" + c["component_id"]).encode("ascii")).hexdigest(), c["component_id"]))
    assert [assignment[c["component_id"]] for c in ordered] == (
        ["train"] * 11 + ["validation"] * 5 + ["calibration"] * 5 + ["final_test"] * 5)
    items[0]["positives"] = 4
    with pytest.raises(ValueError, match="adequacy"):
        release.allocate(items, h, node_budget=10000, check=noop)


def test_publication_byte_projections_and_final_test_custody(tmp_path, monkeypatch):
    rows, items, kwargs = publication_fixture(tmp_path)
    result = release.publish(rows, items, **kwargs)
    package = tmp_path / "candidate-v1"
    assert result["committed"] and (package / "COMMITTED.json").is_file()
    master = (package / "custodian/master.csv").read_bytes()
    assert master == release.HEADER + b"".join(r["csv"].encode("ascii") for r in rows)
    final_membership = [json.loads(v) for v in
                       (package / "custodian/final_test/membership.jsonl").read_bytes().splitlines()]
    final_ids = {r["observation_id"] for r in final_membership}
    for role in ("train", "validation"):
        folder = package / "development" / role
        members = [json.loads(v) for v in (folder / "membership.jsonl").read_bytes().splitlines()]
        assert not final_ids.intersection(r["observation_id"] for r in members)
        assert [r["row_ordinal"] for r in members] == list(range(len(members)))
        assert (folder / "data.csv").read_bytes() == release.HEADER + b"".join(
            r["csv"].encode("ascii") for r in rows if r["observation_id"] in
            {m["observation_id"] for m in members})
        manifest = json.loads((folder / "manifest.json").read_bytes())
        assert set(manifest["files"]) == {"data.csv", "membership.jsonl", "probabilities.jsonl"}
        assert not (folder / "master.csv").exists()
        assert b"final_test" not in (folder / "manifest.json").read_bytes()

    before = {p.relative_to(package).as_posix(): p.read_bytes() for p in package.rglob("*") if p.is_file()}
    monkeypatch.setattr("builtins.print", lambda *a, **kw: (_ for _ in ()).throw(OSError("postcommit console")))
    report = dict(complete=True, committed=result, original_failure=None, finalization_failures=[])
    with pytest.raises(runner.CandidateFailure):
        runner.print_final_report(report)
    assert before == {p.relative_to(package).as_posix(): p.read_bytes() for p in package.rglob("*") if p.is_file()}


@pytest.mark.parametrize("stage", ["collision", "denied", "short_write", "final_check", "rename"])
def test_publication_failures_never_commit_or_cleanup(tmp_path, monkeypatch, stage):
    rows, items, kwargs = publication_fixture(tmp_path)
    destination = tmp_path / "candidate-v1"
    if stage == "collision":
        destination.mkdir()
        (destination / "keep.txt").write_bytes(b"preserve")
    elif stage == "denied":
        kwargs["publication_authorized"] = False
    elif stage == "short_write":
        original = release.write_new
        def failing(path, chunks, check):
            if path.name == "observations.jsonl":
                with path.open("xb") as stream:
                    stream.write(b"partial")
                raise OSError("injected partial publication")
            return original(path, chunks, check)
        monkeypatch.setattr(release, "write_new", failing)
    elif stage == "final_check":
        def check(name):
            if name == "publication_completion_decision":
                raise RuntimeError("final resource failure")
        kwargs["check"] = check
    else:
        monkeypatch.setattr(release, "rename_new", lambda *_: (_ for _ in ()).throw(OSError("commit failure")))
    with pytest.raises((OSError, ValueError, PermissionError, RuntimeError)):
        release.publish(rows, items, **kwargs)
    assert not (destination / "COMMITTED.json").exists()
    if stage == "collision":
        assert (destination / "keep.txt").read_bytes() == b"preserve"
    if stage in ("short_write", "final_check", "rename"):
        assert (tmp_path / "payload/custodian/master.csv").is_file()


@pytest.mark.parametrize("field", ["features", "labels", "global_rows", "observation_ids", "q32_bits", "envelope"])
def test_probability_row_and_observation_reconciliation_rejects_substitution(field):
    engine = context()[-1]()
    batch = engine.step()
    if field == "envelope":
        envelope = json.loads(batch.envelope_bytes)
        envelope["bindings"].append(envelope["bindings"][0])
        bad = replace(batch, envelope_bytes=release.json_bytes(envelope))
    else:
        values = getattr(batch, field).copy()
        if field == "features":
            values[0, 0] += np.float32(1)
        elif field == "labels":
            values[0] = 1 - values[0]
        elif field == "observation_ids":
            values[0] = b"0" * 64
        else:
            values[0] += 1
        values.setflags(write=False)
        bad = replace(batch, **{field: values})
    with pytest.raises(ValueError):
        release.reconcile_batch(bad, engine.binding)


@pytest.mark.parametrize("count", [23, 24, 25, 26])
def test_component_count_aware_policy_without_historical_exactly_26_rule(count):
    items = [dict(component_id=f"{i:064x}", positives=40, negatives=160,
        materials=[1, 2, 3, 4, 5], final_test_eligible=True) for i in range(count)]
    result = release.allocate(items, "a" * 64, node_budget=10000, check=noop)
    assert [sum(v == role for v in result.values()) for role in release.ROLES] == [count - 15, 5, 5, 5]


def test_package_cap_is_failure_without_row_truncation(monkeypatch):
    rows = rows_fixture()[0]
    before = len(rows)
    monkeypatch.setattr(release, "MAX_PACKAGE_ROWS", before - 1)
    with pytest.raises(ValueError, match="row cap"):
        release.validate_rows(rows, noop)
    assert len(rows) == before


def test_incomplete_run_evidence_blocks_publication_before_payload_creation(tmp_path):
    rows, items, kwargs = publication_fixture(tmp_path)
    kwargs["authority"]["run_records"].pop()
    with pytest.raises(ValueError, match="completed run records"):
        release.publish(rows, items, **kwargs)
    assert not kwargs["payload"].exists()


def test_heterogeneous_allocation_backtracks_to_independent_exhaustive_minimum():
    facts = [(5, 5, (1, 2, 3), False), (7, 6, (1, 4), True),
             (5, 7, (4, 5), False), (7, 8, (2, 3, 5), True),
             (9, 9, (1, 2), True), (11, 10, (1, 3, 5), True),
             (9, 11, (3, 4, 5), True), (11, 12, (2, 4), True)]
    def feasible(vector, raw=facts, minima=((10, 10),) * 4, omitted=None):
        for role in range(4):
            group = [fact for fact, code in zip(raw, vector) if code == role]
            if len(group) != 2:
                return False
            if omitted != "eligibility" and role == 3 and not all(f[3] for f in group):
                return False
            positives, negatives = sum(f[0] for f in group), sum(f[1] for f in group)
            materials = {material for fact in group for material in fact[2]}
            if ((omitted != "positives" and positives < minima[role][0])
                    or (omitted != "negatives" and negatives < minima[role][1])
                    or (omitted != "materials" and materials != {1, 2, 3, 4, 5})
                    or (omitted != "concentration" and max(f[0] for f in group) * 2 > positives)):
                return False
        return True
    # Enumerate raw fixture facts without the production search or adequacy helper.
    feasible_vectors = [v for v in product(range(4), repeat=8) if feasible(v)]
    assert len(feasible_vectors) > 1
    assert not feasible((0, 0, 1, 1, 2, 2, 3, 3))
    items = [dict(positives=p, negatives=n, materials=list(m), final_test_eligible=e)
             for p, n, m, e in facts]
    before, visits = deepcopy(items), []
    actual = release.lexicographic_assignment(items, (2, 2, 2, 2), ((10, 10),) * 4,
        node_budget=100000, check=visits.append)
    assert actual == min(feasible_vectors) == (0, 1, 0, 1, 2, 3, 2, 3)
    assert len(visits) > len(items) + 1 and set(visits) == {"allocation_search"}
    assert items == before

    all_materials = (1, 2, 3, 4, 5)
    base = [(5, 5, all_materials, True)] * 8
    minima = ((10, 10),) * 4
    greedy = (0, 0, 1, 1, 2, 2, 3, 3)
    variants = [
        ("materials", [(5, 5, (1, 2, 3, 4), True)] * 2 + base[2:], minima,
         (0, 1, 0, 1, 2, 2, 3, 3)),
        ("concentration", [(p, 5, all_materials, True) for p in (5, 7, 5, 7, 5, 5, 5, 5)],
         minima, (0, 1, 0, 1, 2, 2, 3, 3)),
        ("eligibility", base[:6] + [(5, 5, all_materials, False)] + base[7:], minima,
         (0, 0, 1, 1, 2, 3, 2, 3)),
        ("positives", [(p, 5, all_materials, True) for p in (5, 5, 6, 6, 5, 5, 5, 5)],
         ((12, 10),) + minima[1:], (1, 1, 0, 0, 2, 2, 3, 3)),
        ("negatives", base, ((10, 11),) + minima[1:], None),
    ]
    for constraint, raw, limits, expected in variants:
        full = min((v for v in product(range(4), repeat=8) if feasible(v, raw, limits)),
                   default=None)
        relaxed = min((v for v in product(range(4), repeat=8)
                       if feasible(v, raw, limits, omitted=constraint)), default=None)
        assert full == expected and relaxed == greedy and full != relaxed, constraint
        assert not feasible(relaxed, raw, limits), constraint
        candidate_items = [dict(positives=p, negatives=n, materials=list(m), final_test_eligible=e)
                           for p, n, m, e in raw]
        before = deepcopy(candidate_items)
        if full is None:
            with pytest.raises(release.AllocationInfeasible):
                release.lexicographic_assignment(candidate_items, (2, 2, 2, 2), limits,
                    node_budget=100000, check=noop)
        else:
            assert release.lexicographic_assignment(candidate_items, (2, 2, 2, 2), limits,
                node_budget=100000, check=noop) == full, constraint
        assert candidate_items == before, constraint


def test_write_new_detects_actual_short_return_and_retains_partial_bytes(tmp_path, monkeypatch):
    path = tmp_path / "short-return.bin"
    path_type, calls, checks = type(path), [], []
    original_open = path_type.open
    class ShortStream:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            self.stream.__enter__()
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def write(self, data):
            returned = self.stream.write(data[:-1])
            calls.append((len(data), returned))
            return returned
    def open_stream(target, mode="r", *args, **kwargs):
        if target == path and mode == "xb":
            return ShortStream(original_open(target, mode, *args, **kwargs))
        if target == path and mode == "rb":
            pytest.fail("Short-return failure must precede publication readback")
        return original_open(target, mode, *args, **kwargs)
    def chunks():
        yield b"abcdef"
        pytest.fail("Short-return failure must stop before another chunk")
    with monkeypatch.context() as guard:
        guard.setattr(path_type, "open", open_stream)
        with pytest.raises(OSError, match="^Short publication write$"):
            release.write_new(path, chunks(), checks.append)
    assert calls == [(6, 5)] and checks == ["publication_write"]
    assert path.read_bytes() == b"abcde"
    assert path_type.open is original_open
