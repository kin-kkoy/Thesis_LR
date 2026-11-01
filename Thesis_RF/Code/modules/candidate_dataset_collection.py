"""Candidate R2 collection foundation; no runner, allocation or file publisher."""
from copy import deepcopy
from dataclasses import dataclass, fields as dataclass_fields
import hashlib
import json
from types import MappingProxyType

import numpy as np
from scipy.ndimage import correlate

from .additive_exposure_teacher import (
    COEFFICIENTS, NUMERICAL_ID, TEACHER_ID,
    compute_additive_exposure_probability, compute_susceptibility32,
)
from .automata_engine import FireAutomata
from . import candidate_dataset_crops as crops
from .ca_state import STATE_CODES, STATE_ENCODING, validate_full_state_grid
from .candidate_dataset_inputs import (
    MASK_POLICY_ID, POLICY_ID, PreparedInputs, require_digest, require_text, validate_seeds,
)
from .feature_pipeline import (
    CANONICAL_FEATURE_NAMES, MOORE_NEIGHBOR_KERNEL, FeatureAssembler, validate_feature_values,
)
from .set_c_collector import observation_record_ids
from .transition_observer import (
    SOURCE_ONLY_AUTHORIZATION, canonical_metadata_hash, mask_sha256,
    TransitionObservation, TRANSITION_OBSERVATION_SCHEMA_VERSION,
    validate_grid_provenance, validate_transition_observation, validate_transition_provenance,
)
from .wind_convention import WindContract

SCHEMA = "candidate_teacher_shared_dataset.v1"
WINDS = ((0.0, 0.0), (10.0, 0.0), (10.0, 90.0), (10.0, 180.0),
         (10.0, 270.0), (10.0, 315.0), (17.7, 315.0))
MAX_UPDATES = 300
MAX_RUN_ROWS = 1800000
MAX_PACKAGE_ROWS = 2000000


def disabled_configuration():
    return {
        "schema_version": SCHEMA, "enabled": False, "authorization_reference": None,
        "teacher_id": TEACHER_ID, "numerical_contract_id": NUMERICAL_ID,
        "interpretation_policy": POLICY_ID, "mask_seed_policy": MASK_POLICY_ID,
        "susceptibility": dict(COEFFICIENTS), "wind_weight": 0.20,
        "transition_timers": {"T_3_to_4": 1, "T_4_to_5": 1},
        "feature_names": list(CANONICAL_FEATURE_NAMES), "target_name": "Ignited",
        "winds": [list(w) for w in WINDS], "family_count": 26,
        "replicate_ordinals": [1, 2], "planned_runs": 364,
        "seed_base": 9000000, "family_seed_stride": 10000, "wind_seed_stride": 100,
        "order": ["family_ordinal", "wind_ordinal", "replicate_ordinal"],
        "maximum_completed_updates": MAX_UPDATES, "maximum_run_rows": MAX_RUN_ROWS,
        "maximum_package_rows": MAX_PACKAGE_ROWS,
        "inventory": None, "complete_event_ids": None,
        "inputs": None, "grid": None, "configured_nodata": None,
        "reader_capability": None,
        "resources": {"max_process_rss_bytes": None,
                      "min_available_physical_memory_bytes": None,
                      "min_free_disk_bytes": None, "max_wall_seconds": None,
                      "observation_interval_seconds": None},
        "publication": {"enabled": False, "authorization_reference": None},
    }


def _families(inventory):
    if not isinstance(inventory, list) or len(inventory) != 26:
        raise ValueError("An explicit frozen 26-family inventory is required")
    ids = set()
    for ordinal, family in enumerate(inventory, 1):
        if type(family.get("ordinal")) is not int or family["ordinal"] != ordinal:
            raise ValueError("Family inventory order/ordinals changed")
        identity = require_text(family.get("scenario_family_id"))
        if identity in ids:
            raise ValueError("Duplicate family identity")
        ids.add(identity)
        for name in ("event_id", "prior_use"):
            require_text(family.get(name))
        if type(family.get("complete_event")) is not bool:
            raise ValueError("Complete-event designation must be explicit")
        if family.get("coordinate_system") not in ("grid_row_col", "projected_xy"):
            raise ValueError("Unresolved seed convention")
        if not isinstance(family.get("ignition_points"), list) or not family["ignition_points"]:
            raise ValueError("Prescribed family ignition list is missing")
    for event in {f["event_id"] for f in inventory}:
        if len({f["complete_event"] for f in inventory if f["event_id"] == event}) != 1:
            raise ValueError("Inconsistent complete-event designations")
    return inventory


def planned_runs(inventory):
    result = []
    for family in _families(inventory):
        f = family["ordinal"]
        for w, wind in enumerate(WINDS, 1):
            for r in (1, 2):
                seed = 9000000 + 10000 * f + 100 * w + r
                stem = f"candidate-r2-v1-f{f:02d}-w{w:02d}"
                result.append(dict(family_ordinal=f, wind_ordinal=w, replicate_ordinal=r,
                    scenario_family_id=family["scenario_family_id"], scenario_id=stem,
                    run_id=f"{stem}-r{r}-s{seed}", seed=seed, wind=list(wind)))
    if len({r["run_id"] for r in result}) != 364 or len({r["seed"] for r in result}) != 364:
        raise ValueError("Run/seed uniqueness failure")
    return result


def validate_configuration(config, *, active=False):
    expected = disabled_configuration()
    compared = deepcopy(config)
    pending = ("inventory", "complete_event_ids", "inputs", "grid",
               "configured_nodata", "reader_capability")
    if "spatial_design" in compared:
        crops.validate_design(compared["spatial_design"], compared["inventory"],
                              compared["grid"]["shape"], compared["maximum_completed_updates"])
        expected["spatial_design"] = None
        pending += ("spatial_design",)
    if set(compared) != set(expected):
        raise ValueError("Unknown or missing candidate configuration fields")
    if active:
        if compared["enabled"] is not True:
            raise PermissionError("Candidate dataset execution is disabled")
        require_text(compared["authorization_reference"])
        families = _families(compared["inventory"])
        events = compared["complete_event_ids"]
        if (not isinstance(events, list) or any(not isinstance(e, str) for e in events)
                or events != sorted(set(events))
                or events != sorted({f["event_id"] for f in families if f["complete_event"]})):
            raise ValueError("Complete-event inventory is unresolved or inconsistent")
        for field in ("inputs", "grid", "reader_capability"):
            if not isinstance(compared[field], dict) or not compared[field]:
                raise PermissionError("Pending activation field: " + field)
        capability = compared["reader_capability"]
        if (capability.get("policy_id") != POLICY_ID or capability.get("controls_verified") is not True
                or not isinstance(capability.get("runtime"), dict) or not capability["runtime"]):
            raise PermissionError("Reader capability remains unresolved")
        require_text(capability.get("verification_reference"))
        limits = compared["resources"]
        if set(limits) != set(expected["resources"]):
            raise ValueError("Unknown resource settings")
        for key, value in limits.items():
            types = (int,) if key.endswith("_bytes") else (int, float)
            if type(value) not in types or not np.isfinite(value) or value <= 0:
                raise PermissionError("Resource setting unresolved: " + key)
        compared["enabled"] = False
        compared["authorization_reference"] = None
        compared["resources"] = expected["resources"]
    for field in pending:
        compared[field] = expected[field]
    if canonical_metadata_hash(compared) != canonical_metadata_hash(expected):
        raise ValueError("Candidate fixed specification changed")
    if active:
        planned_runs(config["inventory"])


def engine_configuration(spec):
    return {
        "simulation": {"seed": spec["seed"], "emergency_checkpoint_enabled": False},
        "wind": WindContract(*spec["wind"]).to_manifest(),
        "placeholder_transition": {**COEFFICIENTS, "wind_weight": 0.20,
                                  "T_3_to_4": 1, "T_4_to_5": 1},
        "ml_model": {"enabled": False, "inference_mode": "stochastic_probability",
                     "threshold": None, "inference_threshold": None, "positive_label": 1},
    }


def _bytes(value):
    # Existing canonical JSON semantics; identity strings are validated separately.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def execution_binding(config, prepared, spec, provenance, *, static_context=None):
    """Pure binding construction; its digest must be delivered independently."""
    validate_configuration(config, active=True)
    if not isinstance(prepared, PreparedInputs):
        raise TypeError("Candidate common-mask preparation is required")
    if spec not in planned_runs(config["inventory"]):
        raise ValueError("Run is outside the adopted inventory")
    for field in ("source_revision", "execution_id", "grid_id"):
        require_text(provenance.get(field))
    if type(provenance.get("source_dirty")) is not bool:
        raise ValueError("Source dirty state must be explicit")
    if not isinstance(provenance.get("source_sha256"), dict) or not provenance["source_sha256"]:
        raise ValueError("Actual source identities required")
    for name, digest in provenance["source_sha256"].items():
        require_text(name)
        require_digest(digest)
    if set(config["inputs"]) != set(prepared.input_hashes):
        raise ValueError("Configured four-input inventory mismatch")
    for name, digest in prepared.input_hashes.items():
        require_text(config["inputs"][name].get("path"))
        if config["inputs"][name].get("sha256") != digest:
            raise ValueError("Configured/prepared input identity mismatch")
    if config["reader_capability"] != prepared.interpretation.get("capability"):
        raise ValueError("Reader capability does not bind preparation")
    grid = config["grid"]
    if (tuple(grid.get("shape", ())) != prepared.full_grid_shape
            or tuple(grid.get("transform", ())) != prepared.full_grid_transform
            or grid.get("crs") != str(prepared.environment["crs"])):
        raise ValueError("Configured global grid does not bind preparation")
    if "spatial_design" in config:
        crops.validate_preparation(config["spatial_design"], spec, prepared)
        if not provenance["execution_id"].startswith(crops.CAMPAIGN_ID + "-"):
            raise ValueError("Cropped execution requires a new cropped campaign identity")
    if static_context is None:
        _validate_common_masks(prepared)
        masks = prepared.mask_identities()
    else:
        static_context.check(prepared, config)
        masks = static_context.mask_identities
    if (prepared.interpretation.get("configured_nodata") != config["configured_nodata"]
            or prepared.interpretation.get("mask_seed_policy") != MASK_POLICY_ID):
        raise ValueError("Configured preprocessing does not bind preparation")
    if prepared.interpretation.get("policy_id") != POLICY_ID:
        raise ValueError("Interpretation policy mismatch")
    return dict(schema_version=SCHEMA, teacher_id=TEACHER_ID, numerical_contract_id=NUMERICAL_ID,
        interpretation_policy=POLICY_ID, mask_seed_policy=MASK_POLICY_ID,
        configuration_sha256=canonical_metadata_hash(config), spec=deepcopy(spec),
        input_hashes=dict(prepared.input_hashes), mask_identities=dict(masks),
        global_origin=list(prepared.global_origin), full_grid_shape=list(prepared.full_grid_shape),
        full_grid_transform=list(prepared.full_grid_transform),
        local_transform=list(prepared.environment["transform"]), grid_crs=str(prepared.environment["crs"]),
        interpretation=deepcopy(prepared.interpretation), source=deepcopy(provenance),
        execution_approval_reference=config["authorization_reference"])


@dataclass(frozen=True)
class CandidateBatch:
    features: np.ndarray
    labels: np.ndarray
    global_rows: np.ndarray
    global_cols: np.ndarray
    observation_ids: np.ndarray
    q32_bits: np.ndarray
    envelope_bytes: bytes


def _frozen_array(value, dtype):
    return _OwnedArray.make(value, dtype).array


def _validate_feature_source_columns(column):
    # Keep the historical matrix validator's predicates on every cell, including excluded cells.
    if any(not np.all(np.isfinite(value)) for value in column.values()):
        raise ValueError("Feature matrix contains non-finite values")
    if np.any((column["slope_risk"] < 0) | (column["slope_risk"] > 1)):
        raise ValueError("slope_risk must be within [0, 1]")
    if not np.all(np.isin(column["building_presence"], (0.0, 1.0))):
        raise ValueError("building_presence must be binary")
    material_class = column["material_class"]
    if not np.all(np.isclose(material_class, np.rint(material_class))):
        raise ValueError("material_class must be integer-like")
    if np.any((material_class < 0) | (material_class > 5)):
        raise ValueError("material_class must be within 0..5")
    allowed_risk = FeatureAssembler.MATERIAL_CLASS_TO_RISK[np.rint(material_class).astype(np.int8)]
    allowed_risk = np.where(column["building_presence"] > 0, allowed_risk, 0.0)
    if not np.allclose(column["material_risk"], allowed_risk, atol=1e-6):
        raise ValueError("material_risk must match material_class")
    if np.any(column["wind_speed"] < 0):
        raise ValueError("wind_speed must be non-negative")
    for name in ("wind_sin", "wind_cos"):
        if np.any((column[name] < -1) | (column[name] > 1)):
            raise ValueError(f"{name} must be within [-1, 1]")
    neighbor_count = column["neighbor_burning_count"]
    if not np.all(np.isclose(neighbor_count, np.rint(neighbor_count))):
        raise ValueError("neighbor_burning_count must be integer-like")
    if np.any((neighbor_count < 0) | (neighbor_count > 8)):
        raise ValueError("neighbor_burning_count must be within 0..8")
    expected_composite = column["building_presence"] * column["material_risk"]
    if not np.allclose(column["composite_flammability"], expected_composite, atol=1e-6):
        raise ValueError("composite_flammability does not match its derivation")
    if np.any((column["wind_weighted_score"] < 0) | (column["wind_weighted_score"] > 1)):
        raise ValueError("wind_weighted_score must be within [0, 1]")


@dataclass(frozen=True)
class _OwnedArray:
    """Fresh C-order bytes, with checked view identities and metadata."""
    array: np.ndarray
    root: np.ndarray
    owner: bytes
    shape: tuple
    dtype: np.dtype

    @classmethod
    def make(cls, value, dtype=None):
        value = np.asarray(value, dtype=dtype)
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot be snapshot evidence")
        owner = value.tobytes(order="C")
        root = np.frombuffer(owner, dtype=value.dtype)
        return cls(root.reshape(value.shape), root, owner, value.shape, value.dtype)

    def check(self, value):
        if (value is not self.array or type(self.owner) is not bytes
                or value.base is not self.root or self.root.base is not self.owner
                or value.shape != self.shape or value.dtype != self.dtype
                or not value.flags.c_contiguous or value.flags.writeable
                or self.root.flags.writeable or self.root.dtype != self.dtype
                or self.root.shape != (int(np.prod(self.shape)),)):
            raise ValueError("Immutable array ownership/metadata changed")


def _validate_common_masks(prepared):
    e = prepared.environment
    shape = e["grid_shape"]
    for name in ("slope_risk", "proximity_risk", "building_presence", "material_class",
                 "burnable_mask", "nodata_mask"):
        if not isinstance(e[name], np.ndarray) or e[name].shape != shape:
            raise ValueError("Candidate static array shape mismatch: " + name)
    if (e["burnable_mask"].dtype != bool or e["nodata_mask"].dtype != bool
            or prepared.receiving_mask.dtype != bool or prepared.receiving_mask.shape != shape
            or not np.array_equal(e["nodata_mask"], ~e["burnable_mask"])
            or not np.all(np.isfinite(e["building_presence"]))
            or not np.all(np.isin(e["building_presence"], (0, 1)))
            or not np.array_equal(e["building_presence"] > 0, e["material_class"] > 0)
            or not np.array_equal(prepared.receiving_mask,
                                  e["burnable_mask"] & (e["building_presence"] > 0))):
        raise ValueError("Candidate common masks were changed")


def _prepared_metadata(prepared):
    e = prepared.environment
    return dict(input_hashes=dict(prepared.input_hashes), interpretation=prepared.interpretation,
        origin=prepared.global_origin, full_shape=prepared.full_grid_shape,
        full_transform=prepared.full_grid_transform, shape=e["grid_shape"],
        transform=tuple(e["transform"]), crs=str(e["crs"]))


def _context_dependencies():
    return canonical_metadata_hash(dict(coefficients=COEFFICIENTS, teacher=TEACHER_ID,
        numerical=NUMERICAL_ID, features=CANONICAL_FEATURE_NAMES, winds=WINDS,
        material_dtype=FeatureAssembler.MATERIAL_CLASS_TO_RISK.dtype.str,
        material_shape=FeatureAssembler.MATERIAL_CLASS_TO_RISK.shape,
        material_bits=FeatureAssembler.MATERIAL_CLASS_TO_RISK.tobytes().hex(),
        moore_dtype=MOORE_NEIGHBOR_KERNEL.dtype.str, moore_shape=MOORE_NEIGHBOR_KERNEL.shape,
        moore_bits=MOORE_NEIGHBOR_KERNEL.tobytes().hex()))


@dataclass(frozen=True)
class CandidateContext:
    """Only its owned preparation is authoritative; mutable input aliases are retired."""
    prepared: PreparedInputs
    configuration_sha256: str
    metadata_sha256: str
    arrays: object
    features: object
    engine_fields: object
    winds: object
    mask_identities: object
    dependencies_sha256: str
    _validated_identity: int = 0

    def check(self, prepared, config):
        if (self._validated_identity != id(self) or prepared is not self.prepared
                or self.dependencies_sha256 != _context_dependencies()
                or canonical_metadata_hash(config) != self.configuration_sha256
                or canonical_metadata_hash(_prepared_metadata(prepared)) != self.metadata_sha256):
            raise ValueError("Supplied static context does not bind preparation/configuration")
        for record in self.arrays.values():
            record.check(record.array)
        for name in ("slope_risk", "proximity_risk", "building_presence", "material_class",
                     "burnable_mask", "nodata_mask"):
            self.arrays["environment:" + name].check(prepared.environment[name])
        self.arrays["receiving"].check(prepared.receiving_mask)
        for name, array in self.features.items():
            self.arrays["feature:" + name].check(array)
        for name, array in self.engine_fields.items():
            self.arrays["engine:" + name].check(array)
        for key, (wind, kernel) in self.winds.items():
            if _bytes(key) != _bytes((wind.speed_kmh, wind.direction_deg)):
                raise ValueError("Cached wind binding changed")
            self.arrays["wind:" + str(key)].check(kernel)


def build_candidate_context(prepared, config):
    """Validate once, before RNG; return a separate immutable authoritative preparation."""
    validate_configuration(config, active=True)
    if not isinstance(prepared, PreparedInputs):
        raise TypeError("Candidate common-mask preparation is required")
    _validate_common_masks(prepared)
    validate_grid_provenance(prepared.environment["crs"], prepared.environment["transform"])
    records = {}
    def own(name, value, dtype=None):
        array = np.asarray(value, dtype=dtype)
        record = next((item for item in records.values() if item.array is array), None)
        if record is None:
            record = _OwnedArray.make(array)
        records[name] = record
        return record.array
    environment = dict(prepared.environment)
    for name in ("slope_risk", "proximity_risk", "building_presence", "material_class",
                 "burnable_mask", "nodata_mask"):
        environment[name] = own("environment:" + name, environment[name])
    owned = PreparedInputs(MappingProxyType(environment), own("receiving", prepared.receiving_mask),
        tuple(prepared.global_origin), tuple(prepared.full_grid_shape), tuple(prepared.full_grid_transform),
        deepcopy(prepared.input_hashes), deepcopy(prepared.interpretation))
    assembler = FeatureAssembler(environment, WindContract(*WINDS[0]).to_manifest(), {})
    names = ("slope_risk", "proximity_risk", "building_presence", "material_risk",
             "material_class", "composite_flammability")
    features = {name: own("feature:" + name, getattr(assembler, name)) for name in names}
    shape = environment["grid_shape"]
    # Retain the reference's dynamic/wind narrowing, stack promotion, and final f32 conversion.
    stacked_dtype = np.result_type(np.float32, *(a.dtype for a in features.values()))
    for start in range(0, shape[0], 32):
        columns = {name: np.asarray(np.asarray(a[start:start + 32], dtype=stacked_dtype),
                                   dtype=np.float32) for name, a in features.items()}
        zero = np.zeros((min(32, shape[0] - start), shape[1]), dtype=np.float32)
        _validate_feature_source_columns(columns | dict(wind_speed=np.float32(0),
            wind_sin=np.float32(0), wind_cos=np.float32(1),
            neighbor_burning_count=zero, wind_weighted_score=zero))
    fields = dict(slope_risk=environment["slope_risk"], proximity_risk=environment["proximity_risk"],
        building_presence=np.asarray(environment["building_presence"], dtype=np.float32),
        material_class=np.asarray(environment["material_class"], dtype=np.int8))
    fields["material_risk"] = FeatureAssembler.MATERIAL_CLASS_TO_RISK[fields["material_class"]]
    fields["mapped_building_mask"] = fields["building_presence"] > 0
    fields["p_base"] = compute_susceptibility32(slope32=fields["slope_risk"],
        building32=fields["building_presence"], proximity32=fields["proximity_risk"],
        material_risk32=fields["material_risk"]).beta32
    fields = {name: own("engine:" + name, array) for name, array in fields.items()}
    winds = {}
    for key in WINDS:
        wind = WindContract(*key)
        winds[key] = (wind, own("wind:" + str(key), wind.directional_kernel(0.20)))
    result = CandidateContext(owned, canonical_metadata_hash(config),
        canonical_metadata_hash(_prepared_metadata(owned)), MappingProxyType(records),
        MappingProxyType(features), MappingProxyType(fields), MappingProxyType(winds),
        MappingProxyType(owned.mask_identities()), _context_dependencies())
    object.__setattr__(result, "_validated_identity", id(result))
    result.check(owned, config)
    return result


def _active_neighbour_fields(state, kernel, tile_size):
    """Disjoint target tiles; each correlation receives its complete one-cell source halo."""
    height, width = state.shape
    rows, cols = np.nonzero(state == 4)
    tiles = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = rows + dr, cols + dc
            valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            tile_columns = (width + tile_size - 1) // tile_size
            ids = (rr[valid] // tile_size) * tile_columns + cc[valid] // tile_size
            tiles.update(map(int, np.unique(ids)))
    count = np.zeros(state.shape, dtype=np.int8)
    score = np.zeros(state.shape, dtype=np.float32)
    denominator = np.float32(float(kernel.sum()))
    for identity in sorted(tiles):
        tr, tc = divmod(identity, tile_columns)
        r0, c0 = tr * tile_size, tc * tile_size
        r1, c1 = min(r0 + tile_size, height), min(c0 + tile_size, width)
        sr0, sc0 = max(0, r0 - 1), max(0, c0 - 1)
        sr1, sc1 = min(height, r1 + 1), min(width, c1 + 1)
        blazing = (state[sr0:sr1, sc0:sc1] == 4).astype(np.int8)
        local_count = correlate(blazing, MOORE_NEIGHBOR_KERNEL, mode="constant", cval=0,
                                output=np.int16).astype(np.int8)
        weighted = correlate(blazing.astype(np.float32), kernel, mode="constant", cval=0.0,
                             output=np.float32)
        if denominator > 0:
            weighted = weighted / denominator
        inside = np.s_[r0 - sr0:r1 - sr0, c0 - sc0:c1 - sc0]
        count[r0:r1, c0:c1] = local_count[inside]
        score[r0:r1, c0:c1] = weighted[inside]
    return count, score


@dataclass(frozen=True)
class CandidateDrawEvidence:
    generator: object
    bit_generator: object
    state_before: bytes
    state_after: bytes
    indices: _OwnedArray
    draw: _OwnedArray


def _observation_metadata(observation):
    return canonical_metadata_hash({field.name: getattr(observation, field.name)
        for field in dataclass_fields(observation)
        if not isinstance(getattr(observation, field.name), np.ndarray)})


class _CompletedRecord(dict):
    """JSON-compatible accounting; callbacks cannot modify a committed record."""
    def __init__(self, value):
        dict.__init__(self, {key: _CompletedRecord(item) if isinstance(item, dict) else item
                            for key, item in value.items()})

    def _reject(self, *args, **kwargs):
        raise ValueError("Completed accounting is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _reject

    def __deepcopy__(self, memo):
        return {key: deepcopy(value, memo) for key, value in self.items()}


class CandidateAutomata(FireAutomata):
    """Explicit candidate selection; capture permission is not execution authority."""
    def __setattr__(self, name, value):
        if name == "_execution_route" and name in self.__dict__:
            object.__setattr__(self, "failed", True)
            raise ValueError("Candidate execution route mutation")
        if name == "failed" and self.__dict__.get("failed", False) and value is not True:
            raise ValueError("Candidate failure latch cannot be cleared")
        if self.__dict__.get("_integrity_sealed", False) and name in (
                "_integrity_sealed", "_static_context", "_context_bound", "_candidate_configuration",
                "_bound_members", "_bound_settings", "_completed_state", "_completed_rng_state",
                "_mutable_state_arrays", "_settings_bytes", "_check_context", "_check_completed_state",
                "_check_capture", "_check_optimized_handoff", "validate_handoff", "latch_handoff_failure"):
            object.__setattr__(self, "failed", True)
            raise ValueError("Candidate validation anchor/helper mutation")
        if self.__dict__.get("_transition_locked", False) and name in (
                "row_count", "last_completed", "_batches", "_context_bound", "_bound_members",
                "_bound_settings", "_candidate_configuration", "_completed_state", "_completed_rng_state",
                "_mutable_state_arrays", "_transition_locked"):
            object.__setattr__(self, "failed", True)
            raise ValueError("Transition callback cannot replace committed accounting/bindings")
        super().__setattr__(name, value)

    def __init__(self, prepared, config, spec, provenance, *,
                 execution_authorized=False, approval_reference=None,
                 expected_binding_sha256=None, observer=None, static_context=None):
        if execution_authorized is not True:
            raise PermissionError("Separate candidate execution authority required")
        validate_configuration(config, active=True)
        if approval_reference != config["authorization_reference"]:
            raise PermissionError("Candidate execution approval mismatch")
        if static_context is not None and type(static_context) is not CandidateContext:
            raise TypeError("An explicit validated candidate context is required")
        binding = execution_binding(config, prepared, spec, provenance, static_context=static_context)
        require_digest(expected_binding_sha256)
        if hashlib.sha256(_bytes(binding)).hexdigest() != expected_binding_sha256:
            raise PermissionError("Independent candidate execution binding mismatch")
        family = config["inventory"][spec["family_ordinal"] - 1]
        seeds = validate_seeds(family["ignition_points"], prepared, family["coordinate_system"])
        # All fields used by constructor-dispatched overrides precede super().
        self.teacher_id, self.numerical_contract_id = TEACHER_ID, NUMERICAL_ID
        self.prepared, self.binding, self.spec = prepared, binding, deepcopy(spec)
        self._capture = self._pending = None
        self._batches, self.row_count = (), 0
        self.failed, self.incomplete_transition = False, None
        self.last_completed = None
        self._observer = observer
        self._static_context = static_context
        self._context_bound = static_context is not None
        self._execution_route = static_context
        self._draw_evidence = self._verified_observation = None
        self._snapshot_records = ()
        self._candidate_configuration = config
        self.run_identity_sha256 = hashlib.sha256(_bytes(binding)).hexdigest()
        engine_config = engine_configuration(spec)
        masks = prepared.mask_identities() if static_context is None else static_context.mask_identities
        generic = {
            **{key: SCHEMA for key in ("set_id", "experiment_id", "simulator_version",
                                     "environment_id", "capture_source_id")},
            "event_id": family["event_id"], "scenario_family_id": spec["scenario_family_id"],
            "scenario_id": spec["scenario_id"], "run_id": spec["run_id"],
            "grid_id": provenance["grid_id"], "simulator_id": TEACHER_ID,
            "source_revision": provenance["source_revision"], "source_dirty": provenance["source_dirty"],
            "input_hashes": dict(prepared.input_hashes),
            "configuration_hash": canonical_metadata_hash(engine_config),
            "execution_id": provenance["execution_id"], "seed": spec["seed"],
            "authorization": SOURCE_ONLY_AUTHORIZATION,
            "authorization_reference": approval_reference,
            "transition_source_type": "authorized_simulated",
            "simulation_valid_mask_id": masks["environment_sha256"],
            "mapped_building_mask_id": masks["building_sha256"],
            "simulation_valid_mask_sha256": masks["environment_sha256"],
            "mapped_building_mask_sha256": masks["building_sha256"],
            "wind_manifest": engine_config["wind"], "wind_weight": 0.20,
            "candidate_run_identity_sha256": self.run_identity_sha256,
            "teacher_id": TEACHER_ID, "numerical_contract_id": NUMERICAL_ID,
            "interpretation_policy": POLICY_ID, "mask_seed_policy": MASK_POLICY_ID,
        }
        self._capture_provenance_bytes = _bytes(generic)
        if static_context is None:
            super().__init__(prepared.environment, engine_config, model_free=True,
                transition_observer=self._stage_observation, transition_provenance=generic)
        else:
            self._initialize_context(engine_config, generic)
        if self.rng.bit_generator.__class__.__name__ != "PCG64":
            raise RuntimeError("Required fresh default_rng backend is PCG64")
        self.set_ignition(family["ignition_points"], family["coordinate_system"])
        self.last_completed = self._record()
        if self._context_bound:
            self._bound_members = MappingProxyType({name: getattr(self, name) for name in (
                "_static_context", "_candidate_configuration", "prepared", "environment", "config", "wind", "wind_kernel", "rng",
                "feature_assembler", "simulation_valid_mask", "mapped_building_mask", "burnable_mask",
                "nodata_mask", "slope_risk", "proximity_risk", "building_presence", "material_class",
                "material_risk", "p_base", "_observer", "_transition_observer", "_transition_provenance")})
            self._bound_settings = self._settings_bytes()
            self._check_context()
            self._completed_state = tuple(_OwnedArray.make(a) for a in
                (self.grid, self.ignition_timers, self.blazing_timers))
            self._mutable_state_arrays = (self.grid, self.ignition_timers, self.blazing_timers)
            self._completed_rng_state = _bytes(self.rng.bit_generator.state)
        self._integrity_sealed = True

    def _initialize_context(self, config, generic):
        context = self._static_context
        self.environment, self.config = self.prepared.environment, config
        self._model_free, self._model, self._ml_enabled = True, None, False
        for name, value in context.engine_fields.items():
            setattr(self, name, value)
        self.grid_shape = self.environment["grid_shape"]
        self.nodata_mask = self.environment["nodata_mask"]
        self.simulation_valid_mask = self.environment["burnable_mask"]
        self.burnable_mask = self.prepared.receiving_mask
        self.transform, self.crs = self.environment["transform"], self.environment["crs"]
        self.simulation_cfg, self.wind_cfg = config["simulation"], config["wind"]
        self.transition_cfg, self.ml_cfg = config["placeholder_transition"], config["ml_model"]
        self.emergency_checkpoint_enabled = False
        self.flammability_weights, self.output_cfg = {}, {}
        self._validate_ml_inference_config()
        self.wind, self.wind_kernel = context.winds[tuple(self.spec["wind"])]
        self.t_3_to_4 = self.t_4_to_5 = 1
        self.grid = np.where(self.burnable_mask, np.int8(2), np.int8(1))
        self.ignition_timers = np.zeros(self.grid_shape, dtype=np.int16)
        self.blazing_timers = np.zeros(self.grid_shape, dtype=np.int16)
        self.timestep = 0
        self._transition_observer = self._stage_observation
        self._transition_provenance = validate_transition_provenance(generic,
            expected_configuration_hash=canonical_metadata_hash(config),
            expected_seed=self.spec["seed"], expected_wind_manifest=self.wind.to_manifest(),
            expected_wind_weight=0.20,
            expected_simulation_valid_mask_sha256=context.mask_identities["environment_sha256"],
            expected_mapped_building_mask_sha256=context.mask_identities["building_sha256"])
        self._transition_source_type = "authorized_simulated"
        assembler = FeatureAssembler.__new__(FeatureAssembler)
        assembler.__dict__.update(context.features)
        assembler.grid_shape, assembler.wind = self.grid_shape, self.wind
        assembler.wind_speed = self.wind.speed_kmh
        assembler.wind_sin, assembler.wind_cos = self.wind.from_sin, self.wind.from_cos
        assembler.wind_dy, assembler.wind_dx = self.wind.source_direction_rc
        assembler.wind_propagation_dr, assembler.wind_propagation_dc = self.wind.propagation_direction_rc
        assembler.burnable_mask = self.burnable_mask
        assembler.feature_names = list(CANONICAL_FEATURE_NAMES)
        self.feature_assembler = assembler
        # Context validation and the independently supplied run binding precede this fresh RNG.
        self.rng = np.random.default_rng(self.spec["seed"])

    def _settings_bytes(self):
        return _bytes(dict(config=self.config, simulation=self.simulation_cfg, wind=self.wind_cfg,
            transition=self.transition_cfg, ml=self.ml_cfg, spec=self.spec, binding=self.binding,
            grid_shape=self.grid_shape, transform=tuple(self.transform), crs=str(self.crs),
            timers=[self.t_3_to_4, self.t_4_to_5], model_free=self._model_free,
            emergency=self.emergency_checkpoint_enabled, capture=self._capture_provenance_bytes.decode(),
            source_type=self._transition_source_type, run_identity=self.run_identity_sha256,
            teacher=self.teacher_id, numerical=self.numerical_contract_id,
            context_bound=self._context_bound,
            dependencies=_context_dependencies()))

    def _check_context(self):
        if any(getattr(self, name) is not value for name, value in self._bound_members.items()):
            raise ValueError("Candidate context/engine object substitution")
        self._static_context.check(self.prepared, self._candidate_configuration)
        if self._settings_bytes() != self._bound_settings or self._model is not None:
            raise ValueError("Candidate configuration/engine settings changed")
        assembler = self.feature_assembler
        if (assembler.grid_shape != self.grid_shape or assembler.wind is not self.wind
                or assembler.burnable_mask is not self.burnable_mask
                or tuple(assembler.feature_names) != CANONICAL_FEATURE_NAMES
                or (assembler.wind_dy, assembler.wind_dx) != self.wind.source_direction_rc
                or (assembler.wind_propagation_dr, assembler.wind_propagation_dc)
                != self.wind.propagation_direction_rc
                or _bytes(tuple(map(float, (assembler.wind_speed, assembler.wind_sin, assembler.wind_cos))))
                != _bytes((self.wind.speed_kmh, self.wind.from_sin, self.wind.from_cos))):
            raise ValueError("Candidate feature/wind binding changed")
        for name, value in self._static_context.features.items():
            if getattr(assembler, name) is not value:
                raise ValueError("Candidate static feature replacement: " + name)

    @property
    def draw_evidence(self):
        """Internal authoritative-draw evidence; never part of persisted row formats."""
        return self._draw_evidence

    def _check_completed_state(self):
        for name, record, owned in zip(("grid", "ignition_timers", "blazing_timers"),
                                       self._completed_state, self._mutable_state_arrays, strict=True):
            record.check(record.array)
            actual = getattr(self, name)
            if (actual is not owned or not actual.flags.owndata or not actual.flags.c_contiguous
                    or not actual.flags.writeable or actual.shape != record.shape or actual.dtype != record.dtype
                    or not np.array_equal(actual, record.array)):
                raise ValueError("Completed candidate state/timer mutation: " + name)

    def latch_handoff_failure(self):
        # Preserve completed evidence and any existing incomplete-transition diagnosis.
        object.__setattr__(self, "failed", True)

    def validate_handoff(self, *, expected_context=None):
        try:
            route = self._execution_route
            if (expected_context is not None and route is not expected_context
                    or self._static_context is not route or self._context_bound is not (route is not None)):
                raise ValueError("Candidate execution route/context mutation")
            if self.failed or self.incomplete_transition is not None:
                raise ValueError("Failed or incomplete candidate handoff")
            if route is not None:
                self._check_context()
                self._check_completed_state()
                if (self.row_count != self.last_completed["committed_rows"]
                        or self.timestep != self.last_completed["state_timestep"]):
                    raise ValueError("Completed candidate accounting mutation")
                if _bytes(self.rng.bit_generator.state) != self._completed_rng_state:
                    raise ValueError("Completed candidate RNG mutation")
        except BaseException:
            object.__setattr__(self, "failed", True)
            raise

    def _verify_candidate_transition(self, before, timestep, rng_before, count, score, evidence):
        """Re-derive decisions from trusted pre-state and the actual draw, never another draw."""
        self._check_context()
        state, ignition, blazing = (record.array for record in before)
        # A different tiling independently derives coverage from trusted pre-step state.
        expected_count, expected_score = _active_neighbour_fields(state, self.wind_kernel, 127)
        if (count.dtype != np.int8 or score.dtype != np.float32
                or count.shape != self.grid_shape or score.shape != self.grid_shape
                or not np.array_equal(count, expected_count)
                or not np.array_equal(score.view(np.uint32), expected_score.view(np.uint32))):
            raise ValueError("Candidate neighbourhood reconstruction mismatch")
        eligible = self.burnable_mask & (state == 2) & (expected_count > 0)
        indices = np.flatnonzero(eligible.ravel())
        q = compute_additive_exposure_probability(beta32=self.p_base.ravel()[indices],
            wind_weighted_score32=expected_score.ravel()[indices]).q32
        expected_state, expected_ignition, expected_blazing = state.copy(), ignition.copy(), blazing.copy()
        ignited, active = state == 3, state == 4
        expected_ignition[ignited] += np.int16(1)
        expected_blazing[active] += np.int16(1)
        promoted = ignited & (expected_ignition >= self.t_3_to_4)
        extinguished = active & (expected_blazing >= self.t_4_to_5)
        expected_state[promoted], expected_state[extinguished] = 4, 5
        expected_ignition[promoted] = 0
        expected_blazing[promoted | extinguished] = 0
        if len(indices):
            if evidence is None or evidence is not self._draw_evidence:
                raise ValueError("Missing authoritative draw evidence")
            evidence.indices.check(evidence.indices.array)
            evidence.draw.check(evidence.draw.array)
            if (evidence.generator is not rng_before[0] or evidence.bit_generator is not rng_before[1]
                    or evidence.state_before != rng_before[2]
                    or evidence.state_after != _bytes(self.rng.bit_generator.state)
                    or self.rng is not evidence.generator or self.rng.bit_generator is not evidence.bit_generator
                    or evidence.draw.dtype != np.float64 or evidence.draw.shape != self.grid_shape
                    or evidence.indices.dtype != indices.dtype or evidence.indices.shape != indices.shape
                    or not np.array_equal(evidence.indices.array, indices)):
                raise ValueError("Authoritative draw/index/generator binding mismatch")
            draw = evidence.draw.array
            if not np.all(np.isfinite(draw)) or np.any((draw < 0) | (draw > 1)):
                raise ValueError("Invalid authoritative uniform draw")
            chosen = indices[draw.ravel()[indices] < q]
            expected_state.ravel()[chosen] = 3
            expected_ignition.ravel()[chosen] = expected_blazing.ravel()[chosen] = 0
            self._check_capture(indices, q, expected_score)
        elif (evidence is not None or self._draw_evidence is not None or self._capture is not None
                or self.rng is not rng_before[0] or self.rng.bit_generator is not rng_before[1]
                or _bytes(self.rng.bit_generator.state) != rng_before[2]):
            raise ValueError("Empty eligibility changed RNG or retained stale evidence")
        if self.timestep != timestep + 1:
            raise ValueError("Candidate timestep mismatch")
        for actual, expected in zip((self.grid, self.ignition_timers, self.blazing_timers),
                (expected_state, expected_ignition, expected_blazing), strict=True):
            if (actual.dtype != expected.dtype or actual.shape != expected.shape
                    or not np.array_equal(actual, expected) or np.any(actual < 0)):
                raise ValueError("Candidate full-domain state/timer mismatch")
        return eligible, q

    def _check_capture(self, indices, q, score):
        if self._capture is None:
            raise ValueError("Missing actual-comparison probability capture")
        captured, actual, bits, selected_score = self._capture
        if (actual.dtype != np.float32 or bits.dtype != np.uint32
                or selected_score.dtype != np.float32 or not np.array_equal(captured, indices)
                or not np.array_equal(actual.view(np.uint32), q.view(np.uint32))
                or not np.array_equal(bits, q.view(np.uint32))
                or not np.array_equal(selected_score.view(np.uint32), score.ravel()[indices].view(np.uint32))):
            raise ValueError("Actual comparison capture/independent decision mismatch")

    def _optimized_step(self):
        self._check_context()
        self._check_completed_state()
        validate_full_state_grid(self.grid)
        if not np.array_equal(self.grid == 1, ~self.burnable_mask):
            raise ValueError("Candidate full-grid domain invariant")
        for timer in (self.ignition_timers, self.blazing_timers):
            if timer.dtype != np.int16 or timer.shape != self.grid_shape or np.any(timer < 0):
                raise ValueError("Candidate timer invariant")
        before = tuple(_OwnedArray.make(a) for a in
            (self.grid, self.ignition_timers, self.blazing_timers))
        retained = []
        for batch in self._batches:
            arrays = []
            for name in ("features", "labels", "global_rows", "global_cols", "observation_ids", "q32_bits"):
                array = getattr(batch, name)
                record = _OwnedArray(array, array.base, array.base.base, array.shape, array.dtype)
                record.check(array)
                arrays.append((name, record))
            retained.append((batch, batch.envelope_bytes, tuple(arrays)))
        accounting = (self._batches, self.row_count, self.last_completed, _bytes(self.last_completed),
                      self._completed_state, self._completed_rng_state, tuple(retained))
        timestep = self.timestep
        rng_before = (self.rng, self.rng.bit_generator, _bytes(self.rng.bit_generator.state))
        state = before[0].array
        next_grid = state.copy()
        ignition_indices = np.flatnonzero((state == 3).ravel())
        blazing_indices = np.flatnonzero((state == 4).ravel())
        self.ignition_timers.ravel()[ignition_indices] += np.int16(1)
        self.blazing_timers.ravel()[blazing_indices] += np.int16(1)
        promoted = ignition_indices[self.ignition_timers.ravel()[ignition_indices] >= self.t_3_to_4]
        extinguished = blazing_indices[self.blazing_timers.ravel()[blazing_indices] >= self.t_4_to_5]
        next_grid.ravel()[promoted], next_grid.ravel()[extinguished] = 4, 5
        self.ignition_timers.ravel()[promoted] = 0
        self.blazing_timers.ravel()[promoted] = self.blazing_timers.ravel()[extinguished] = 0
        count, score = _active_neighbour_fields(state, self.wind_kernel, 64)
        indices = np.flatnonzero(((state == 2) & (count > 0)).ravel())
        self._draw_evidence = None
        if len(indices):
            q = self._model_free_probability(indices, score)
            # The only production draw; its immutable bytes are the checker's decision evidence.
            draw = _OwnedArray.make(self.rng.random(self.grid_shape))
            self._draw_evidence = CandidateDrawEvidence(rng_before[0], rng_before[1], rng_before[2],
                _bytes(self.rng.bit_generator.state), _OwnedArray.make(indices), draw)
            chosen = indices[draw.array.ravel()[indices] < q]
            next_grid.ravel()[chosen] = 3
            self.ignition_timers.ravel()[chosen] = self.blazing_timers.ravel()[chosen] = 0
        self.grid, self.timestep = next_grid, timestep + 1
        evidence = self._draw_evidence
        eligible, checked_q = self._verify_candidate_transition(before, timestep, rng_before, count, score, evidence)
        records = {}
        def snapshot(name, array):
            records[name] = _OwnedArray.make(array)
            return records[name].array
        completed = tuple(_OwnedArray.make(a) for a in
            (self.grid, self.ignition_timers, self.blazing_timers))
        observation = TransitionObservation(
            schema_version=TRANSITION_OBSERVATION_SCHEMA_VERSION,
            transition_source_type=self._transition_source_type, state_encoding=STATE_ENCODING,
            state_codes=STATE_CODES, timestep_t=timestep, timestep_t1=self.timestep,
            state_t=before[0].array, state_t1=completed[0].array,
            blazing_mask_t=snapshot("blazing_mask_t", state == 4),
            blazing_neighbor_count_t=snapshot("blazing_neighbor_count_t", count),
            wind_weighted_score_t=snapshot("wind_weighted_score_t", score),
            simulation_valid_mask=self.simulation_valid_mask, mapped_building_mask=self.mapped_building_mask,
            eligible_mask_t=snapshot("eligible_mask_t", eligible),
            newly_ignited_mask_t1=snapshot("newly_ignited_mask_t1", eligible & (self.grid == 3)),
            wind_manifest=MappingProxyType(self.wind.to_manifest()), wind_weight=0.20,
            seed=self.spec["seed"], grid_shape=self.grid_shape, grid_crs=str(self.crs).strip(),
            grid_transform=tuple(map(float, self.transform)),
            domain_mask_hashes=MappingProxyType(dict(
                simulation_valid_mask_sha256=self._static_context.mask_identities["environment_sha256"],
                mapped_building_mask_sha256=self._static_context.mask_identities["building_sha256"],
                eligible_mask_t_sha256=mask_sha256(eligible))), provenance=self._transition_provenance)
        records["state_t"], records["state_t1"] = before[0], completed[0]
        self._snapshot_records = tuple(records.items())
        self._verified_observation = observation
        capture, members, settings = self._capture, self._bound_members, self._bound_settings
        engine_arrays = (self.grid, self.ignition_timers, self.blazing_timers)
        metadata, owned_records = _observation_metadata(observation), self._snapshot_records
        rng_after = (self.rng, self.rng.bit_generator, _bytes(self.rng.bit_generator.state))
        draw_binding = None if evidence is None else (evidence.generator, evidence.bit_generator,
            evidence.state_before, evidence.state_after, evidence.indices, evidence.draw)
        self._transition_observer(observation)
        handoff = (observation, completed, evidence, capture, members, settings, engine_arrays,
                   indices, checked_q, metadata, owned_records, rng_after, accounting, draw_binding)
        self._check_optimized_handoff(*handoff)
        return handoff

    def _check_optimized_handoff(self, observation, completed, evidence, capture, members, settings,
                                 engine_arrays, indices, checked_q, metadata, owned_records, rng_after, accounting,
                                 draw_binding):
        if (self._bound_members is not members or self._bound_settings is not settings
                or self._pending is not observation or self._verified_observation is not observation
                or self._capture is not capture or self._draw_evidence is not evidence
                or self._snapshot_records is not owned_records or _observation_metadata(observation) != metadata
                or observation.simulation_valid_mask is not self.simulation_valid_mask
                or observation.mapped_building_mask is not self.mapped_building_mask
                or self._batches is not accounting[0] or self.row_count != accounting[1]
                or self.last_completed is not accounting[2] or _bytes(self.last_completed) != accounting[3]
                or self._completed_state is not accounting[4] or self._completed_rng_state is not accounting[5]
                or self.failed or self.incomplete_transition is not None):
            raise ValueError("Candidate callback/evidence substitution")
        self._check_context()
        for batch, envelope, arrays in accounting[6]:
            if batch.envelope_bytes != envelope:
                raise ValueError("Retained batch envelope changed")
            for name, record in arrays:
                record.check(getattr(batch, name))
        for name, record in owned_records:
            record.check(getattr(observation, name))
        if observation.timestep_t1 != self.timestep:
            raise ValueError("Candidate callback timestep mutation")
        for name, actual, expected in zip(("grid", "ignition_timers", "blazing_timers"),
                                          engine_arrays, completed, strict=True):
            expected.check(expected.array)
            if (getattr(self, name) is not actual or not actual.flags.owndata
                    or not actual.flags.c_contiguous or not actual.flags.writeable or actual.dtype != expected.dtype
                    or actual.shape != expected.shape or not np.array_equal(actual, expected.array)):
                raise ValueError("Candidate callback state/timer mutation")
        if (self.rng is not rng_after[0] or self.rng.bit_generator is not rng_after[1]
                or _bytes(self.rng.bit_generator.state) != rng_after[2]):
            raise ValueError("Candidate callback RNG mutation")
        if evidence is not None:
            if (evidence.generator is not draw_binding[0] or evidence.bit_generator is not draw_binding[1]
                    or evidence.state_before != draw_binding[2] or evidence.state_after != draw_binding[3]
                    or evidence.indices is not draw_binding[4] or evidence.draw is not draw_binding[5]):
                raise ValueError("Authoritative draw evidence changed at handoff")
            evidence.draw.check(evidence.draw.array)
            evidence.indices.check(evidence.indices.array)
            if (self.rng is not evidence.generator or self.rng.bit_generator is not evidence.bit_generator
                    or _bytes(self.rng.bit_generator.state) != evidence.state_after):
                raise ValueError("Candidate callback RNG mutation")
            self._check_capture(indices, checked_q, observation.wind_weighted_score_t)

    @property
    def batches(self):
        return self._batches

    def acknowledge_batches(self, snapshot: tuple) -> None:
        """Release engine references after caller persistence; identity is not durability proof."""
        if type(snapshot) is not tuple or not snapshot or snapshot is not self._batches:
            raise ValueError("Exact current nonempty committed-batch snapshot required")
        self._batches = ()

    def _precompute_base_probability(self):
        self.p_base = compute_susceptibility32(slope32=self.slope_risk,
            building32=self.building_presence, proximity32=self.proximity_risk,
            material_risk32=self.material_risk).beta32

    def set_ignition(self, points, coordinate_system="grid_row_col"):
        cells = validate_seeds(points, self.prepared, coordinate_system)
        if self.timestep != 0 or np.any(np.isin(self.grid, (3, 4, 5))):
            raise ValueError("Ignition is permitted only once before updates")
        # Validation is complete before the first mutation.
        super().set_ignition(list(cells))

    def _model_free_probability(self, susceptible_indices, wind_weighted_score):
        q32 = compute_additive_exposure_probability(
            beta32=self.p_base.ravel()[susceptible_indices],
            wind_weighted_score32=wind_weighted_score.ravel()[susceptible_indices]).q32
        if q32.dtype != np.float32 or q32.shape != susceptible_indices.shape:
            raise ValueError("Candidate comparison vector mismatch")
        q32.setflags(write=False)
        self._capture = (susceptible_indices.copy(), q32, q32.view(np.uint32).copy(),
                         wind_weighted_score.ravel()[susceptible_indices].copy())
        return q32

    def _stage_observation(self, observation):
        if self._pending is not None:
            raise ValueError("Duplicate transition observation")
        if self._context_bound:
            if observation is not self._verified_observation:
                raise ValueError("Candidate observation lacks independent transition checking")
            for name, record in self._snapshot_records:
                record.check(getattr(observation, name))
            self._pending = observation
        else:
            self._pending = validate_transition_observation(observation)
        if self._observer is not None:
            self._observer(self._pending)

    def _record(self):
        record = dict(state_timestep=self.timestep, state_counts=self.get_state_counts(),
                      committed_rows=self.row_count)
        return _CompletedRecord(record) if self._context_bound else record

    def _assemble_eligible_features(self, observation, indices):
        assembler = self.feature_assembler
        if assembler.grid_shape != self.grid_shape:
            raise ValueError("Feature assembler grid shape mismatch")
        wind_names = ("wind_speed", "wind_sin", "wind_cos")
        dynamic = {"neighbor_burning_count": observation.blazing_neighbor_count_t,
                   "wind_weighted_score": observation.wind_weighted_score_t}
        sources = {name: np.asarray(dynamic[name] if name in dynamic else getattr(assembler, name))
                   for name in CANONICAL_FEATURE_NAMES if name not in wind_names}
        for name, values in sources.items():
            if values.shape != self.grid_shape:
                raise ValueError(f"{name} shape mismatch: {values.shape} != {self.grid_shape}")
        # The reference converts dynamics/wind first, promotes all stacked columns, then casts to f32.
        stacked_dtype = np.result_type(np.float32,
            *(values.dtype for name, values in sources.items() if name not in dynamic))
        def converted(values, name):
            if name in dynamic or name in wind_names:
                values = np.asarray(values, dtype=np.float32)
            return np.asarray(np.asarray(values, dtype=stacked_dtype), dtype=np.float32)
        winds = {name: converted(getattr(assembler, name), name) for name in wind_names}
        if getattr(self, "_context_bound", False):
            self._check_context()
        else:
            for start in range(0, self.grid_shape[0], 32):
                column = {name: converted(values[start:start + 32], name)
                          for name, values in sources.items()}
                _validate_feature_source_columns(column | winds)
        rows, cols = np.unravel_index(indices, self.grid_shape)
        features = np.empty((len(indices), len(CANONICAL_FEATURE_NAMES)), dtype=np.float32)
        for index, name in enumerate(CANONICAL_FEATURE_NAMES):
            features[:, index] = winds[name] if name in winds else converted(sources[name][rows, cols], name)
        return validate_feature_values(features)

    def _make_batch(self):
        observation = self._pending
        if observation is None:
            raise ValueError("Missing completed transition observation")
        indices = np.flatnonzero(observation.eligible_mask_t.ravel())
        if len(indices):
            if self._capture is None:
                raise ValueError("Missing actual-comparison probability capture")
            captured, actual, bits, score = self._capture
            if (not np.array_equal(indices, captured)
                    or not np.array_equal(actual.view(np.uint32), bits)
                    or not np.array_equal(score.view(np.uint32),
                        observation.wind_weighted_score_t.ravel()[indices].view(np.uint32))):
                raise ValueError("Actual comparison capture/observation alignment failure")
        else:
            if self._capture is not None:
                raise ValueError("Stale probability capture on no-susceptible update")
            bits = np.empty(0, dtype=np.uint32)
        features = self._assemble_eligible_features(observation, indices)
        labels = observation.newly_ignited_mask_t1.ravel()[indices].astype(np.int8)
        rows, cols = np.unravel_index(indices, self.grid_shape)
        rows, cols = rows + self.prepared.global_origin[0], cols + self.prepared.global_origin[1]
        provenance_hash = canonical_metadata_hash(observation.provenance)
        ids = observation_record_ids(self.spec | {"grid_id": self.binding["source"]["grid_id"]},
            run_identity_sha256=self.run_identity_sha256, provenance_sha256=provenance_hash,
            seed=self.spec["seed"], timestep_t=observation.timestep_t,
            timestep_t1=observation.timestep_t1, rows=rows, cols=cols)
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("Duplicate observation binding")
        if np.any((bits & np.uint32(0x7FFFFFFF) == 0) & (bits != 0)):
            raise ValueError("Comparison probability has negative zero")
        bindings = [dict(observation_id=identity.decode("ascii"), q32_bits=int(q),
            row_sha256=hashlib.sha256(np.asarray(row, dtype="<f4").tobytes() +
                                      bytes([int(label)])).hexdigest())
            for identity, q, row, label in zip(ids, bits, features, labels, strict=True)]
        envelope = dict(schema_version="candidate_comparison_capture.v1",
            run_identity_sha256=self.run_identity_sha256, provenance_sha256=provenance_hash,
            teacher_id=TEACHER_ID, numerical_contract_id=NUMERICAL_ID,
            interpretation_policy=POLICY_ID, mask_seed_policy=MASK_POLICY_ID,
            timestep_t=observation.timestep_t, timestep_t1=observation.timestep_t1,
            generic_provenance=json.loads(self._capture_provenance_bytes),
            state_t_sha256=hashlib.sha256(observation.state_t.tobytes()).hexdigest(),
            state_t1_sha256=hashlib.sha256(observation.state_t1.tobytes()).hexdigest(),
            eligible_mask_sha256=mask_sha256(observation.eligible_mask_t),
            row_hash_encoding="eleven_little_endian_float32_then_int8_label.v1",
            bindings=bindings)
        return CandidateBatch(_frozen_array(features, np.float32), _frozen_array(labels, np.int8),
            _frozen_array(rows, np.int64), _frozen_array(cols, np.int64),
            _frozen_array(ids, "S64"), _frozen_array(bits, np.uint32), _bytes(envelope))

    def step(self):
        if self.failed or self.timestep >= MAX_UPDATES:
            raise RuntimeError("Failed or capped candidate run cannot continue")
        self._capture = self._pending = None
        before = self.timestep
        try:
            handoff = None
            self.validate_handoff()
            if self._execution_route is not None:
                self._snapshot_records = ()
                self._verified_observation = None
                self._transition_locked = True
                handoff = self._optimized_step()
            else:
                super().step()
            for timer in (self.ignition_timers, self.blazing_timers):
                if timer.dtype != np.int16 or timer.shape != self.grid_shape or np.any(timer < 0):
                    raise ValueError("Candidate timer invariant")
            batch = self._make_batch()
            if handoff is not None:
                self._check_optimized_handoff(*handoff)
            count = self.row_count + len(batch.labels)
            if count > MAX_RUN_ROWS:
                raise RuntimeError("Candidate run row cap exceeded; no truncation")
            batches = self._batches + (batch,)
            record = dict(state_timestep=self.timestep, state_counts=self.get_state_counts(),
                          committed_rows=count)
            if handoff is not None:
                record = _CompletedRecord(record)
        except BaseException:
            object.__setattr__(self, "_transition_locked", False)
            self.failed = True
            self.incomplete_transition = dict(from_timestep=before, to_timestep=self.timestep,
                                              collection_completed=False)
            self._capture = self._pending = None
            raise
        object.__setattr__(self, "_transition_locked", False)
        self._batches, self.row_count, self.last_completed = batches, count, record
        if handoff is not None:
            object.__setattr__(self, "_completed_state", handoff[1])
            object.__setattr__(self, "_mutable_state_arrays", handoff[6])
            object.__setattr__(self, "_completed_rng_state", handoff[11][2])
        self._capture = self._pending = None
        return batch
