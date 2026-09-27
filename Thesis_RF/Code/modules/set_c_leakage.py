"""Source-only Set C leakage graph; it never allocates split roles."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from typing import Iterable

import pandas as pd


LEAKAGE_GRAPH_VERSION = "set_c_joint_leakage_graph.v1"
ALLOCATION_POLICY_VERSION = "set_c_deterministic_allocation_policy.v1"
LEAKAGE_KEYS = (
    "scenario_family_id",
    "spatial_block_id",
    "stable_cell_id",
    "duplicate_group_id",
)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def build_joint_leakage_components(
    observations: pd.DataFrame,
    *,
    complete_event_ids: Iterable[str] = (),
) -> pd.Series:
    """Return deterministic component IDs without exposing any future role."""
    required = {"event_id", *LEAKAGE_KEYS}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"Leakage graph is missing columns: {missing}")
    if observations.empty:
        raise ValueError("Leakage graph requires observations")
    if observations.loc[:, sorted(required)].isna().any().any():
        raise ValueError("Leakage identities cannot be null")
    values = observations.loc[:, sorted(required)].astype(str)
    if values.apply(lambda column: column.str.strip().eq("").any()).any():
        raise ValueError("Leakage identities cannot be empty")

    complete_events = {str(value) for value in complete_event_ids}
    unknown = complete_events.difference(set(values["event_id"]))
    if unknown:
        raise ValueError(f"Complete-event inventory contains unknown IDs: {sorted(unknown)}")

    graph = _UnionFind(len(values))
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(values.to_dict("records")):
        event_id = row["event_id"]
        if event_id in complete_events:
            groups[("event_id", event_id)].append(index)
        else:
            groups[("scenario_family_id", row["scenario_family_id"])].append(index)
        for key in LEAKAGE_KEYS[1:]:
            groups[(key, row[key])].append(index)
    for members in groups.values():
        for member in members[1:]:
            graph.union(members[0], member)

    roots: dict[int, list[int]] = defaultdict(list)
    for index in range(len(values)):
        roots[graph.find(index)].append(index)
    identifiers: dict[int, str] = {}
    for members in roots.values():
        identities = values.iloc[members].sort_values(list(values.columns)).to_dict("records")
        digest = sha256(
            json.dumps(identities, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for member in members:
            identifiers[member] = digest
    return pd.Series(
        [identifiers[index] for index in range(len(values))],
        index=observations.index,
        name="leakage_component_id",
        dtype="string",
    )


def deterministic_allocation_policy_contract(dataset_sha256: str) -> dict[str, object]:
    """Describe the future deterministic order; numerical targets stay gated."""
    if len(dataset_sha256) != 64 or any(char not in "0123456789abcdef" for char in dataset_sha256):
        raise ValueError("dataset_sha256 must be lowercase SHA-256")
    return {
        "policy_version": ALLOCATION_POLICY_VERSION,
        "dataset_sha256": dataset_sha256,
        "component_order": "sha256(dataset_sha256 + ':' + leakage_component_id)",
        "component_atomicity": True,
        "numerical_role_targets": None,
        "assignments": None,
        "status": "contract_only_not_executable",
    }
