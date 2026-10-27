"""Candidate-specific canonical release; no historical publisher or cleanup."""
from collections import defaultdict
from itertools import chain
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .candidate_dataset_collection import CandidateBatch, MAX_PACKAGE_ROWS, planned_runs, validate_configuration
from .candidate_dataset_inputs import require_digest, require_text
from .feature_pipeline import CANONICAL_FEATURE_NAMES, SET_C_SCHEMA_VERSION, TARGET_VERSION
from .transition_observer import canonical_metadata_hash

RELEASE_SCHEMA = "candidate_teacher_shared_dataset_release.v1"
COMPONENT_SCHEMA = "candidate_component_sorted_observation_ids.v1"
ROLES = ("train", "validation", "calibration", "final_test")
ROLE_MINIMA = ((200, 800), (100, 400), (100, 400), (100, 400))
HEADER = (",".join((*CANONICAL_FEATURE_NAMES, "Ignited")) + "\n").encode("ascii")


def json_bytes(value: object) -> bytes:
    """Existing compact ASCII-escaped JSON, rejecting non-string keys/surrogates."""
    def validate(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError("JSON keys must be strings")
                validate(key)
                validate(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                validate(child)
        elif isinstance(item, str):
            if any(0xD800 <= ord(c) <= 0xDFFF for c in item):
                raise ValueError("Unpaired surrogate")
    validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def uint(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Nonnegative integer required")
    return value


def predictor_text(value: np.float32) -> str:
    if not isinstance(value, np.float32) or not np.isfinite(value):
        raise TypeError("Canonical finite float32 predictor required")
    text = format(float(value), ".9g")
    if np.float32(text).view(np.uint32) != value.view(np.uint32):
        raise ValueError("Predictor did not round trip")
    return text


def row_bytes(features: np.ndarray, label: int) -> bytes:
    if (not isinstance(features, np.ndarray) or features.dtype != np.float32
            or features.shape != (11,) or not np.all(np.isfinite(features))):
        raise ValueError("Exact eleven finite float32 predictors required")
    if type(label) is not int or label not in (0, 1):
        raise ValueError("Exact binary integer label required")
    return (",".join(predictor_text(v) for v in features) + f",{label}\n").encode("ascii")


def decode_row(line: str) -> tuple:
    if not isinstance(line, str) or not line.endswith("\n") or "\r" in line:
        raise ValueError("Noncanonical CSV row")
    parts = line[:-1].split(",")
    if len(parts) != 12 or parts[-1] not in ("0", "1"):
        raise ValueError("Exact twelve-column row required")
    values, label = np.array(parts[:11], dtype=np.float32), int(parts[-1])
    if row_bytes(values, label) != line.encode("ascii"):
        raise ValueError("CSV row is not the canonical original serialization")
    return values, label


def observation_payload(binding: dict, envelope: dict, row: int, col: int) -> dict:
    spec = binding["spec"]
    return dict(identity_version="sha256_observation_record_provenance.v2",
        scenario_family_id=spec["scenario_family_id"], scenario_id=spec["scenario_id"],
        run_id=spec["run_id"], seed=uint(spec["seed"]), timestep_t=uint(envelope["timestep_t"]),
        timestep_t1=uint(envelope["timestep_t1"]), grid_id=binding["source"]["grid_id"],
        feature_schema_version=SET_C_SCHEMA_VERSION, target_version=TARGET_VERSION,
        run_identity_sha256=envelope["run_identity_sha256"],
        provenance_sha256=envelope["provenance_sha256"],
        cell_row=uint(row), cell_col=uint(col))


def reconcile_batch(batch: CandidateBatch, binding: dict) -> tuple:
    """Validate actual-comparison capture; never replace q32 by recomputation."""
    if not isinstance(batch, CandidateBatch):
        raise TypeError("Committed CandidateBatch required")
    n = len(batch.labels)
    arrays = ((batch.features, np.float32, (n, 11)), (batch.labels, np.int8, (n,)),
              (batch.global_rows, np.int64, (n,)), (batch.global_cols, np.int64, (n,)),
              (batch.observation_ids, np.dtype("S64"), (n,)),
              (batch.q32_bits, np.uint32, (n,)))
    if any(a.dtype != dtype or a.shape != shape or a.flags.writeable
           for a, dtype, shape in arrays):
        raise ValueError("Batch shape/dtype/immutability mismatch")
    envelope = json.loads(batch.envelope_bytes)
    if json_bytes(envelope) != batch.envelope_bytes:
        raise ValueError("Noncanonical comparison envelope")
    if (envelope["schema_version"] != "candidate_comparison_capture.v1"
            or envelope["run_identity_sha256"] != digest_bytes(json_bytes(binding))
            or envelope["timestep_t1"] != envelope["timestep_t"] + 1
            or len(envelope["bindings"]) != n):
        raise ValueError("Comparison envelope/run/count mismatch")
    generic = envelope["generic_provenance"]
    for name in ("state_t_sha256", "state_t1_sha256", "eligible_mask_sha256"):
        require_digest(envelope[name])
    if canonical_metadata_hash(generic) != envelope["provenance_sha256"]:
        raise ValueError("Generic capture provenance mismatch")
    for key in ("teacher_id", "numerical_contract_id", "interpretation_policy", "mask_seed_policy"):
        if envelope[key] != binding[key] or generic[key] != binding[key]:
            raise ValueError("Candidate provenance identity mismatch")
    if (generic["candidate_run_identity_sha256"] != envelope["run_identity_sha256"]
            or generic["run_id"] != binding["spec"]["run_id"]
            or generic["seed"] != binding["spec"]["seed"]
            or generic["grid_id"] != binding["source"]["grid_id"]):
        raise ValueError("Generic observation does not bind candidate run")
    result = []
    for i in range(n):
        row, col = int(batch.global_rows[i]), int(batch.global_cols[i])
        origin, shape = binding["global_origin"], binding["full_grid_shape"]
        if not (origin[0] <= row < shape[0] and origin[1] <= col < shape[1]):
            raise ValueError("Global coordinate outside approved grid")
        identity = batch.observation_ids[i].decode("ascii")
        require_digest(identity)
        payload = observation_payload(binding, envelope, row, col)
        if digest_bytes(json_bytes(payload)) != identity:
            raise ValueError("Observation identity mismatch")
        bits = int(batch.q32_bits[i])
        q = batch.q32_bits[i:i + 1].view(np.float32)[0]
        if not np.isfinite(q) or not 0 <= q <= 1 or bits == 0x80000000:
            raise ValueError("Invalid actual comparison q32")
        label = int(batch.labels[i])
        line = row_bytes(batch.features[i], label)
        row_hash = digest_bytes(np.asarray(batch.features[i], dtype="<f4").tobytes()
                                + bytes([label]))
        captured = dict(observation_id=identity, q32_bits=bits, row_sha256=row_hash)
        if envelope["bindings"][i] != captured:
            raise ValueError("Feature/label/probability binding mismatch")
        spec = binding["spec"]
        result.append(dict(**captured, payload=payload, csv=line.decode("ascii"),
            order=[spec["family_ordinal"], spec["wind_ordinal"], spec["replicate_ordinal"],
                   envelope["timestep_t"], row, col],
            event_id=generic["event_id"], material_class=int(batch.features[i, 4]),
            grid_id=generic["grid_id"], scenario_family_id=spec["scenario_family_id"]))
    return envelope, result


def validate_rows(rows: list, check) -> None:
    if len(rows) > MAX_PACKAGE_ROWS:
        raise ValueError("Candidate package row cap exceeded")
    seen, last = set(), None
    for ordinal, row in enumerate(rows):
        if ordinal % 256 == 0:
            check("row_reconciliation")
        require_digest(row["observation_id"])
        if row["observation_id"] in seen:
            raise ValueError("Repeated observation identity; no deduplication")
        seen.add(row["observation_id"])
        key = tuple(uint(v) for v in row["order"])
        if len(key) != 6 or (last is not None and key <= last):
            raise ValueError("Master row order/uniqueness failure")
        last = key
        payload = row["payload"]
        expected_fields = {"identity_version", "scenario_family_id", "scenario_id", "run_id",
            "seed", "timestep_t", "timestep_t1", "grid_id", "feature_schema_version",
            "target_version", "run_identity_sha256", "provenance_sha256", "cell_row", "cell_col"}
        if (set(payload) != expected_fields
                or payload["identity_version"] != "sha256_observation_record_provenance.v2"
                or payload["feature_schema_version"] != SET_C_SCHEMA_VERSION
                or payload["target_version"] != TARGET_VERSION):
            raise ValueError("Observation schema mismatch")
        require_digest(payload["run_identity_sha256"])
        require_digest(payload["provenance_sha256"])
        for name in ("seed", "timestep_t", "timestep_t1", "cell_row", "cell_col"):
            uint(payload[name])
        for name, value in payload.items():
            if isinstance(value, str):
                require_text(value)
        if (payload["timestep_t1"] != payload["timestep_t"] + 1
                or payload["cell_row"] != key[4] or payload["cell_col"] != key[5]
                or payload["timestep_t"] != key[3]
                or payload["scenario_family_id"] != row["scenario_family_id"]
                or payload["grid_id"] != row["grid_id"]
                or digest_bytes(json_bytes(payload)) != row["observation_id"]):
            raise ValueError("Master observation payload mismatch")
        features, label = decode_row(row["csv"])
        if (digest_bytes(np.asarray(features, dtype="<f4").tobytes() + bytes([label]))
                != row["row_sha256"] or features[4] not in (1, 2, 3, 4, 5)
                or int(features[4]) != row["material_class"]):
            raise ValueError("Master row binding/material mismatch")
        bits = uint(row["q32_bits"])
        if bits > 0xFFFFFFFF:
            raise ValueError("q32 encoding overflow")
        q = np.array([bits], dtype=np.uint32).view(np.float32)[0]
        if not np.isfinite(q) or not 0 <= q <= 1 or bits == 0x80000000:
            raise ValueError("Invalid q32 encoding")


def components(rows: list, complete_event_ids: list, eligibility: dict, check) -> list:
    """Family edges are unconditional; other containment edges only add unions."""
    if complete_event_ids != sorted(set(complete_event_ids)):
        raise ValueError("Explicit sorted unique complete-event list required")
    families = sorted({r["scenario_family_id"] for r in rows})
    if not families or not set(families).issubset(eligibility):
        raise ValueError("Every observed family needs a frozen eligibility assessment")
    parent = {f: f for f in families}
    def root(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f
    def join(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)
    edges = {}
    observed_events = set()
    for i, row in enumerate(rows):
        if i % 256 == 0:
            check("grouping")
        f, p = row["scenario_family_id"], row["payload"]
        observed_events.add(row["event_id"])
        keys = [("cell", row["grid_id"], p["cell_row"], p["cell_col"]),
                ("block", row["grid_id"], p["cell_row"] // 128, p["cell_col"] // 128),
                ("repeated_observation", row["observation_id"])]
        if row["event_id"] in complete_event_ids:
            keys.append(("event", row["event_id"]))
        for key in keys:
            if key in edges:
                join(f, edges[key])
            else:
                edges[key] = f
    if set(complete_event_ids) - observed_events:
        raise ValueError("Complete-event inventory has no retained observations")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(row["scenario_family_id"])].append(i)
    result, component_ids = [], set()
    for indices in groups.values():
        ids = sorted(rows[i]["observation_id"] for i in indices)
        if len(ids) != len(set(ids)):
            raise ValueError("Component observation duplication")
        identity = digest_bytes(json_bytes(ids))
        if identity in component_ids:
            raise ValueError("Component identity collision")
        component_ids.add(identity)
        positives = 0
        for ordinal, i in enumerate(indices):
            if ordinal % 256 == 0:
                check("component_adequacy")
            positives += decode_row(rows[i]["csv"])[1]
        fs = sorted({rows[i]["scenario_family_id"] for i in indices})
        result.append(dict(component_id=identity, indices=indices, families=fs,
            positives=positives, negatives=len(indices) - positives,
            materials=sorted({rows[i]["material_class"] for i in indices}),
            final_test_eligible=all(eligibility[f]["eligible"] for f in fs)))
    return result


class AllocationIncomplete(RuntimeError):
    """Budget exhausted; not evidence of infeasibility."""


class AllocationInfeasible(RuntimeError):
    """Exhaustive constrained search established no feasible vector."""


def lexicographic_assignment(items: list, counts: tuple, minima: tuple, *, node_budget: int, check) -> tuple:
    """First feasible DFS leaf is lexicographically smallest; no soft objective."""
    if type(node_budget) is not int or node_budget <= 0:
        raise ValueError("Explicit positive allocation node budget required")
    if len(counts) != 4 or sum(counts) != len(items):
        raise ValueError("Role component counts mismatch")
    used, vector, nodes = [0] * 4, [], 0
    def search(position):
        nonlocal nodes
        nodes += 1
        if nodes > node_budget:
            raise AllocationIncomplete("allocation_incomplete: node budget exhausted")
        check("allocation_search")
        if position == len(items):
            for role in range(4):
                group = [items[i] for i, code in enumerate(vector) if code == role]
                pos, neg = sum(c["positives"] for c in group), sum(c["negatives"] for c in group)
                materials = {m for c in group for m in c["materials"]}
                if (used[role] != counts[role] or pos < minima[role][0]
                        or neg < minima[role][1] or materials != {1, 2, 3, 4, 5}
                        or any(2 * c["positives"] > pos for c in group)):
                    return None
            return tuple(vector)
        for role in range(4):
            if used[role] == counts[role] or (
                    role == 3 and not items[position]["final_test_eligible"]):
                continue
            used[role] += 1
            vector.append(role)
            answer = search(position + 1)
            if answer is not None:
                return answer
            vector.pop()
            used[role] -= 1
        return None
    answer = search(0)
    if answer is None:
        raise AllocationInfeasible("allocation_infeasible: exhaustive search")
    return answer


def allocate(items: list, dataset_sha256: str, *, node_budget: int, check) -> dict:
    require_digest(dataset_sha256)
    if not 23 <= len(items) <= 26 or any(
            c["positives"] < 5 or c["negatives"] < 5 for c in items):
        raise ValueError("Candidate component adequacy failed; no component removal")
    ordered = sorted(items, key=lambda c: (
        digest_bytes((dataset_sha256 + ":" + c["component_id"]).encode("ascii")),
        c["component_id"]))
    vector = lexicographic_assignment(ordered, (len(items) - 15, 5, 5, 5),
        ROLE_MINIMA, node_budget=node_budget, check=check)
    check("allocation_completed")
    return {c["component_id"]: ROLES[v] for c, v in zip(ordered, vector, strict=True)}


def write_new(path: Path, chunks, check) -> dict:
    """Exclusive bytes, fsync, close, then streaming readback of the same identity."""
    h, size = hashlib.sha256(), 0
    with path.open("xb") as stream:
        for chunk in chunks:
            check("publication_write")
            if stream.write(chunk) != len(chunk):
                raise OSError("Short publication write")
            h.update(chunk)
            size += len(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    actual, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            check("publication_readback")
            actual.update(chunk)
            count += len(chunk)
    if count != size or actual.hexdigest() != h.hexdigest():
        raise OSError("Publication readback mismatch")
    return dict(bytes=size, sha256=h.hexdigest())


def rename_new(source: Path, destination: Path) -> None:
    # Windows rename refuses an existing destination, including an empty directory.
    if os.name != "nt":
        raise OSError("Exclusive directory commit currently supports Windows only")
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    os.rename(source, destination)


def _validate_completion(rows, authority, check):
    config = authority["configuration"]
    plan, records = planned_runs(config["inventory"]), authority["run_records"]
    if len(records) != 2 * len(plan):
        raise ValueError("Publication requires all completed run records")
    evidence = authority["evidence_files"]
    if set(evidence) != {"transitions.jsonl", "runs.jsonl"}:
        raise ValueError("Publication requires complete evidence identities")
    for identity in evidence.values():
        require_digest(identity["sha256"])
        if uint(identity["bytes"]) == 0:
            raise ValueError("Completed evidence files cannot be empty")
    counts, positives = defaultdict(int), defaultdict(int)
    for row in rows:
        check("publication_run_membership")
        run_id = row["payload"]["run_id"]
        counts[run_id] += 1
        positives[run_id] += decode_row(row["csv"])[1]
    for i, spec in enumerate(plan):
        check("publication_run_completion")
        initial, final = records[2 * i:2 * i + 2]
        before, after = initial["completed"], final["completed"]
        if (initial["kind"] != "initialization" or final["kind"] != "inactive"
                or initial["spec"] != spec or final["spec"] != spec
                or uint(before["state_timestep"]) != 0 or uint(before["committed_rows"]) != 0
                or not 1 <= uint(after["state_timestep"]) <= 300
                or uint(after["committed_rows"]) != counts[spec["run_id"]]
                or counts[spec["run_id"]] > 1800000
                or after["state_counts"]["ignited"] != 0 or after["state_counts"]["blazing"] != 0
                or after["state_counts"]["extinguished"] != before["state_counts"]["blazing"] + positives[spec["run_id"]]):
            raise ValueError("Publication run completeness/label reconciliation failed")


def publish(rows: list, items: list, *, payload: Path, destination: Path, authority: dict, check,
            publication_authorized: bool = False, expected_binding_sha256: str | None = None) -> dict:
    """Custodian root commit; failures retain payload/evidence without cleanup."""
    if publication_authorized is not True:
        raise PermissionError("Separate publication authority required")
    require_digest(expected_binding_sha256)
    if digest_bytes(json_bytes(authority["binding"])) != expected_binding_sha256:
        raise PermissionError("Independent publication binding mismatch")
    require_text(authority["binding"]["approval_reference"])
    payload, destination = Path(payload), Path(destination)
    if (not payload.is_absolute() or not destination.is_absolute()
            or not payload.parent.is_dir() or not destination.parent.is_dir()
            or os.path.lexists(payload) or os.path.lexists(destination)):
        raise FileExistsError("Exclusive existing parents and absent destinations required")
    if str(payload) != str(Path(authority["binding"]["destinations"]["evidence"]) / "payload"):
        raise PermissionError("Publication staging binding mismatch")
    if str(destination) != authority["binding"]["destinations"]["package"]:
        raise PermissionError("Publication destination binding mismatch")
    if digest_bytes(json_bytes(dict(schema_version="candidate_dataset_execution.v1",
            configuration=authority["configuration"], authority=authority["execution_authority"]))) != authority["binding"]["execution_sha256"]:
        raise PermissionError("Publication execution/configuration binding mismatch")
    validate_configuration(authority["configuration"], active=True)
    validate_rows(rows, check)
    _validate_completion(rows, authority, check)
    plan = {p["run_id"]: p for p in planned_runs(authority["configuration"]["inventory"])}
    for row in rows:
        check("publication_run_membership")
        spec = plan[row["payload"]["run_id"]]
        if (row["order"][:3] != [spec[k] for k in ("family_ordinal", "wind_ordinal", "replicate_ordinal")]
                or row["scenario_family_id"] != spec["scenario_family_id"]
                or row["payload"]["seed"] != spec["seed"]
                or row["payload"]["scenario_id"] != spec["scenario_id"]
                or row["grid_id"] != authority["execution_authority"]["source"]["grid_id"]):
            raise ValueError("Published row is outside the approved run inventory")
    rebuilt = components(rows, authority["configuration"]["complete_event_ids"],
                         authority["execution_authority"]["custody"]["families"], check)
    if json_bytes(rebuilt) != json_bytes(items):
        raise ValueError("Publication grouping/statistics mismatch")
    if not 23 <= len(items) <= 26 or any(c["positives"] < 5 or c["negatives"] < 5 for c in items):
        raise ValueError("Publication adequacy failure")
    payload.mkdir()
    (payload / "custodian").mkdir()
    identities = {}
    def save(relative, chunks):
        identities[relative] = write_new(payload / relative, chunks, check)
    save("custodian/master.csv", chain((HEADER,), (r["csv"].encode("ascii") for r in rows)))
    master = identities["custodian/master.csv"]
    if master["sha256"] != authority["dataset_sha256"]:
        raise ValueError("Frozen master bytes differ from reconciled collection")
    assignments = allocate(items, master["sha256"],
        node_budget=authority["binding"]["allocation_node_budget"], check=check)
    expected = {c["component_id"] for c in items}
    if set(assignments) != expected or any(v not in ROLES for v in assignments.values()):
        raise ValueError("Exhaustive role mapping required")
    index_role = {}
    for c in items:
        for i in c["indices"]:
            if i in index_role or not 0 <= i < len(rows):
                raise ValueError("Component partition is not disjoint")
            index_role[i] = assignments[c["component_id"]]
    if set(index_role) != set(range(len(rows))):
        raise ValueError("Component partition is not exhaustive")
    save("custodian/observations.jsonl", (json_bytes(r) + b"\n" for r in rows))
    save("custodian/probabilities.jsonl", (json_bytes(dict(master_row_ordinal=i,
        csv_sha256=master["sha256"], observation_id=r["observation_id"],
        row_sha256=r["row_sha256"], q32_bits=r["q32_bits"])) + b"\n"
        for i, r in enumerate(rows)))
    save("custodian/components.json", [json_bytes(dict(schema_version=COMPONENT_SCHEMA,
        components=items, assignments=assignments)) + b"\n"])
    for role in ROLES:
        relative = ("custodian/final_test" if role == "final_test" else
                    "calibration" if role == "calibration" else "development/" + role)
        folder = payload / relative
        folder.parent.mkdir(exist_ok=True)
        folder.mkdir()
        selected = [r for i, r in enumerate(rows) if index_role[i] == role]
        save(relative + "/data.csv", chain((HEADER,), (r["csv"].encode("ascii") for r in selected)))
        csv_hash = identities[relative + "/data.csv"]["sha256"]
        save(relative + "/membership.jsonl", (json_bytes(dict(row_ordinal=i,
            csv_sha256=csv_hash, observation_id=r["observation_id"],
            row_sha256=r["row_sha256"], payload=r["payload"])) + b"\n"
            for i, r in enumerate(selected)))
        save(relative + "/probabilities.jsonl", (json_bytes(dict(row_ordinal=i,
            csv_sha256=csv_hash, observation_id=r["observation_id"],
            row_sha256=r["row_sha256"], q32_bits=r["q32_bits"])) + b"\n"
            for i, r in enumerate(selected)))
        scoped = {p[len(relative) + 1:]: value for p, value in identities.items()
                  if p.startswith(relative + "/")}
        save(relative + "/manifest.json", [json_bytes(dict(schema_version=RELEASE_SCHEMA,
            role=role, files=scoped, rows=len(selected), teacher_id=authority["teacher_id"],
            numerical_contract_id=authority["numerical_contract_id"],
            custody="role-scoped; same-account protection is procedural")) + b"\n"])
    save("custodian/manifest.json", [json_bytes(dict(schema_version=RELEASE_SCHEMA,
        authority=authority, master_path="custodian/master.csv", master=master,
        rows=len(rows), files=identities.copy(), complete=True)) + b"\n"])
    check("before_publication_commit")
    write_new(payload / "COMMITTED.json", [json_bytes(dict(schema_version=RELEASE_SCHEMA,
        manifest_sha256=identities["custodian/manifest.json"]["sha256"])) + b"\n"], check)
    check("publication_completion_decision")
    result = dict(package=str(destination), manifest=identities["custodian/manifest.json"],
                  committed=True)
    rename_new(payload, destination)
    return result
