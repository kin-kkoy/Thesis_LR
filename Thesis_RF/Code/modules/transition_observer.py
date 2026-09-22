"""Fail-closed, in-memory observations of synchronous CA transitions.

This module deliberately contains no persistence or artifact-producing APIs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any

import numpy as np

from .ca_state import (
	STATE_BLAZING,
	STATE_CODES,
	STATE_ENCODING,
	STATE_EXTINGUISHED,
	STATE_IGNITED,
	STATE_NON_BURNABLE,
	STATE_NOT_YET_BURNING,
	validate_full_state_grid,
)
from .feature_pipeline import compute_moore_neighbor_count
from .wind_convention import WindContract, compute_wind_weighted_score


TRANSITION_OBSERVATION_SCHEMA_VERSION = "ca_transition_observation.v1"
SOURCE_ONLY_AUTHORIZATION = "d013_source_only_in_memory"
AUTHORIZED_SIMULATED_TRANSITION_SOURCE = "authorized_simulated"

_REQUIRED_TEXT_PROVENANCE = (
	"set_id",
	"experiment_id",
	"event_id",
	"scenario_id",
	"run_id",
	"grid_id",
	"simulation_valid_mask_id",
	"mapped_building_mask_id",
	"simulator_id",
	"simulator_version",
	"source_revision",
	"configuration_hash",
	"environment_id",
	"execution_id",
	"capture_source_id",
	"authorization",
	"authorization_reference",
	"transition_source_type",
)


def _require_sha256(value: object, name: str) -> str:
	if not isinstance(value, str) or len(value) != 64:
		raise ValueError(f"{name} must be a 64-character SHA-256 digest")
	try:
		int(value, 16)
	except ValueError as exc:
		raise ValueError(f"{name} must contain only hexadecimal characters") from exc
	return value.lower()


def _immutable_array(values: object, dtype: np.dtype[Any] | type[np.generic]) -> np.ndarray:
	array = np.array(values, dtype=dtype, copy=True)
	array.setflags(write=False)
	return array


def _freeze_metadata(value: object) -> object:
	"""Deep-copy supported metadata into immutable scalar/container values."""
	if value is None or isinstance(value, (str, bool, int, float)):
		return value
	if isinstance(value, np.generic):
		return _freeze_metadata(value.item())
	if isinstance(value, Mapping):
		return MappingProxyType(
			{str(key): _freeze_metadata(item) for key, item in value.items()}
		)
	if isinstance(value, (tuple, list)):
		return tuple(_freeze_metadata(item) for item in value)
	raise TypeError(
		"Transition provenance must contain only scalars, mappings, and sequences"
	)


def _canonical_hash_value(value: object) -> object:
	if value is None or isinstance(value, (str, bool, int)):
		return value
	if isinstance(value, float):
		if not np.isfinite(value):
			raise ValueError("Configuration hash input contains a non-finite float")
		return value
	if isinstance(value, np.generic):
		return _canonical_hash_value(value.item())
	if isinstance(value, Mapping):
		return {
			str(key): _canonical_hash_value(item)
			for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
		}
	if isinstance(value, (tuple, list)):
		return [_canonical_hash_value(item) for item in value]
	raise TypeError(
		"Configuration hashing supports only scalars, mappings, and sequences"
	)


def canonical_metadata_hash(value: object) -> str:
	"""Return a deterministic SHA-256 digest for scalar configuration metadata."""
	payload = json.dumps(
		_canonical_hash_value(value),
		sort_keys=True,
		separators=(",", ":"),
		ensure_ascii=True,
	).encode("utf-8")
	return hashlib.sha256(payload).hexdigest()


def mask_sha256(mask: object) -> str:
	"""Hash a validated boolean mask together with its shape and dtype contract."""
	array = np.asarray(mask)
	if array.ndim != 2 or array.dtype != np.dtype(bool):
		raise TypeError("Domain masks must be two-dimensional boolean arrays")
	hasher = hashlib.sha256()
	hasher.update(str(array.shape).encode("ascii"))
	hasher.update(b"|bool|")
	hasher.update(np.ascontiguousarray(array).view(np.uint8).tobytes())
	return hasher.hexdigest()


def _validate_mask(mask: object, name: str, shape: tuple[int, int]) -> np.ndarray:
	array = np.asarray(mask)
	if array.ndim != 2 or array.shape != shape:
		raise ValueError(f"{name} must match the full CA grid shape {shape}")
	if array.dtype != np.dtype(bool):
		raise TypeError(f"{name} must be a boolean array without coercion")
	return array


def validate_transition_provenance(
	provenance: object,
	*,
	expected_configuration_hash: str | None = None,
	expected_seed: int | None = None,
	expected_wind_manifest: Mapping[str, object] | None = None,
	expected_wind_weight: float | None = None,
	expected_simulation_valid_mask_sha256: str | None = None,
	expected_mapped_building_mask_sha256: str | None = None,
) -> Mapping[str, object]:
	"""Validate and freeze the required live-capture provenance fields."""
	if not isinstance(provenance, Mapping):
		raise TypeError("transition_provenance must be a mapping")

	missing = [name for name in _REQUIRED_TEXT_PROVENANCE if name not in provenance]
	missing.extend(
		name
		for name in (
			"source_dirty",
			"input_hashes",
			"seed",
			"wind_manifest",
			"wind_weight",
			"simulation_valid_mask_sha256",
			"mapped_building_mask_sha256",
		)
		if name not in provenance
	)
	if missing:
		raise ValueError(
			"Missing required transition provenance: " + ", ".join(sorted(missing))
		)

	for name in _REQUIRED_TEXT_PROVENANCE:
		value = provenance[name]
		if not isinstance(value, str) or not value.strip():
			raise ValueError(f"transition_provenance.{name} must be a non-empty string")
	if provenance["authorization"] != SOURCE_ONLY_AUTHORIZATION:
		raise ValueError(
			"transition_provenance.authorization must authorize only D-013 "
			"source-only in-memory observation"
		)
	if provenance["transition_source_type"] != AUTHORIZED_SIMULATED_TRANSITION_SOURCE:
		raise ValueError(
			"transition_provenance.transition_source_type must be "
			"'authorized_simulated'"
		)
	if not isinstance(provenance["source_dirty"], (bool, np.bool_)):
		raise TypeError("transition_provenance.source_dirty must be boolean")
	_require_sha256(
		provenance["configuration_hash"],
		"transition_provenance.configuration_hash",
	)

	input_hashes = provenance["input_hashes"]
	if not isinstance(input_hashes, Mapping) or not input_hashes:
		raise ValueError("transition_provenance.input_hashes must be a non-empty mapping")
	for name, digest in input_hashes.items():
		if not isinstance(name, str) or not name.strip():
			raise ValueError("Every input hash must have a non-empty string identity")
		_require_sha256(digest, f"transition_provenance.input_hashes[{name!r}]")

	if isinstance(provenance["seed"], (bool, np.bool_)) or not isinstance(
		provenance["seed"], (int, np.integer)
	):
		raise TypeError("transition_provenance.seed must be an integer")
	if not isinstance(provenance["wind_manifest"], Mapping):
		raise TypeError("transition_provenance.wind_manifest must be a mapping")
	provenance_wind = WindContract.from_config(
		dict(provenance["wind_manifest"])
	).to_manifest()
	if dict(provenance["wind_manifest"]) != provenance_wind:
		raise ValueError(
			"transition_provenance.wind_manifest must exactly match the canonical wind contract"
		)
	if isinstance(provenance["wind_weight"], (bool, np.bool_)) or not isinstance(
		provenance["wind_weight"], (int, float, np.integer, np.floating)
	):
		raise TypeError("transition_provenance.wind_weight must be numeric")
	provenance_wind_weight = float(provenance["wind_weight"])
	if not np.isfinite(provenance_wind_weight) or provenance_wind_weight < 0.0:
		raise ValueError(
			"transition_provenance.wind_weight must be finite and non-negative"
		)
	for name in (
		"simulation_valid_mask_sha256",
		"mapped_building_mask_sha256",
	):
		_require_sha256(provenance[name], f"transition_provenance.{name}")

	if (
		expected_configuration_hash is not None
		and provenance["configuration_hash"] != expected_configuration_hash
	):
		raise ValueError(
			"transition_provenance.configuration_hash does not bind the active config"
		)
	if expected_seed is not None and int(provenance["seed"]) != int(expected_seed):
		raise ValueError("transition_provenance.seed does not bind the active seed")
	if (
		expected_wind_manifest is not None
		and dict(provenance["wind_manifest"]) != dict(expected_wind_manifest)
	):
		raise ValueError(
			"transition_provenance.wind_manifest does not bind the active wind"
		)
	if (
		expected_wind_weight is not None
		and provenance_wind_weight != float(expected_wind_weight)
	):
		raise ValueError(
			"transition_provenance.wind_weight does not bind the active wind weight"
		)
	for name, expected in (
		(
			"simulation_valid_mask_sha256",
			expected_simulation_valid_mask_sha256,
		),
		(
			"mapped_building_mask_sha256",
			expected_mapped_building_mask_sha256,
		),
	):
		if expected is not None and provenance[name] != expected:
			raise ValueError(
				f"transition_provenance.{name} does not bind the active domain mask"
			)
	return _freeze_metadata(dict(provenance))  # type: ignore[return-value]


def validate_grid_provenance(
	grid_crs: object, grid_transform: object
) -> tuple[str, tuple[float, ...]]:
	"""Require immutable CRS identity and finite numeric grid-transform metadata."""
	if grid_crs is None or not str(grid_crs).strip():
		raise ValueError("grid_crs must provide a non-empty CRS identity")
	if grid_transform is None:
		raise ValueError("grid_transform must be a finite 6- or 9-value numeric sequence")
	if isinstance(grid_transform, (str, bytes)):
		raise TypeError("grid_transform must be a numeric sequence, not text")
	try:
		values = tuple(float(item) for item in grid_transform)  # type: ignore[arg-type]
	except (TypeError, ValueError) as exc:
		raise TypeError("grid_transform must be a finite numeric sequence") from exc
	if len(values) not in (6, 9) or not np.all(np.isfinite(values)):
		raise ValueError("grid_transform must be a finite 6- or 9-value numeric sequence")
	return str(grid_crs).strip(), values


@dataclass(frozen=True, slots=True)
class TransitionObservation:
	"""Immutable metadata plus mutation-isolated arrays for one CA step."""

	schema_version: str
	transition_source_type: str
	state_encoding: str
	state_codes: Mapping[str, int]
	timestep_t: int
	timestep_t1: int
	state_t: np.ndarray
	state_t1: np.ndarray
	blazing_mask_t: np.ndarray
	blazing_neighbor_count_t: np.ndarray
	wind_weighted_score_t: np.ndarray
	simulation_valid_mask: np.ndarray
	mapped_building_mask: np.ndarray
	eligible_mask_t: np.ndarray
	newly_ignited_mask_t1: np.ndarray
	wind_manifest: Mapping[str, object]
	wind_weight: float
	seed: int
	grid_shape: tuple[int, int]
	grid_crs: str
	grid_transform: tuple[float, ...]
	domain_mask_hashes: Mapping[str, str]
	provenance: Mapping[str, object]


TransitionObserver = Callable[[TransitionObservation], None]


def build_transition_observation(
	*,
	state_t: object,
	state_t1: object,
	transition_source_type: object,
	timestep_t: int,
	timestep_t1: int,
	simulation_valid_mask: object,
	mapped_building_mask: object,
	wind_manifest: object,
	wind_weight: float,
	seed: int,
	provenance: object,
	grid_crs: object = None,
	grid_transform: object = None,
) -> TransitionObservation:
	"""Build one validated observation from full state grids, failing closed."""
	if transition_source_type != AUTHORIZED_SIMULATED_TRANSITION_SOURCE:
		raise ValueError("transition_source_type must be 'authorized_simulated'")
	before = validate_full_state_grid(state_t, "state_t")
	after = validate_full_state_grid(state_t1, "state_t1")
	if after.shape != before.shape:
		raise ValueError("state_t and state_t1 must have equal shapes")
	if (
		isinstance(timestep_t, (bool, np.bool_))
		or not isinstance(timestep_t, (int, np.integer))
		or isinstance(timestep_t1, (bool, np.bool_))
		or not isinstance(timestep_t1, (int, np.integer))
	):
		raise TypeError("Transition timesteps must be integers")
	if int(timestep_t1) != int(timestep_t) + 1:
		raise ValueError("timestep_t1 must equal timestep_t + 1")
	if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
		raise TypeError("seed must be an integer")
	crs_value, transform_value = validate_grid_provenance(
		grid_crs, grid_transform
	)

	allowed_transition = (
		((before == STATE_NON_BURNABLE) & (after == STATE_NON_BURNABLE))
		| (
			(before == STATE_NOT_YET_BURNING)
			& np.isin(after, (STATE_NOT_YET_BURNING, STATE_IGNITED))
		)
		| (
			(before == STATE_IGNITED)
			& np.isin(after, (STATE_IGNITED, STATE_BLAZING))
		)
		| (
			(before == STATE_BLAZING)
			& np.isin(after, (STATE_BLAZING, STATE_EXTINGUISHED))
		)
		| ((before == STATE_EXTINGUISHED) & (after == STATE_EXTINGUISHED))
	)
	if not np.all(allowed_transition):
		invalid_pairs = sorted(
			{
				(int(source), int(destination))
				for source, destination in zip(
					before[~allowed_transition], after[~allowed_transition], strict=True
				)
			}
		)
		formatted = ", ".join(
			f"{source}->{destination}" for source, destination in invalid_pairs
		)
		raise ValueError(f"Impossible synchronous CA state transition(s): {formatted}")

	simulation_valid = _validate_mask(
		simulation_valid_mask, "simulation_valid_mask", before.shape
	)
	mapped_building = _validate_mask(
		mapped_building_mask, "mapped_building_mask", before.shape
	)
	domain = simulation_valid & mapped_building
	for name, state in (("state_t", before), ("state_t1", after)):
		if not np.array_equal(state == STATE_NON_BURNABLE, ~domain):
			raise ValueError(
				f"{name} violates the CA burnable invariant: state 1 must exactly "
				"represent cells outside the valid mapped-building domain"
			)

	if not isinstance(wind_manifest, Mapping):
		raise TypeError("wind_manifest must be the canonical wind provenance mapping")
	wind = WindContract.from_config(dict(wind_manifest))
	canonical_wind_manifest = wind.to_manifest()
	if dict(wind_manifest) != canonical_wind_manifest:
		raise ValueError("wind_manifest must exactly match the canonical wind contract")
	if isinstance(wind_weight, (bool, np.bool_)) or not isinstance(
		wind_weight, (int, float, np.integer, np.floating)
	):
		raise TypeError("wind_weight must be numeric")
	wind_weight_value = float(wind_weight)
	if not np.isfinite(wind_weight_value) or wind_weight_value < 0.0:
		raise ValueError("wind_weight must be finite and non-negative")

	computed_simulation_valid_hash = mask_sha256(simulation_valid)
	computed_mapped_building_hash = mask_sha256(mapped_building)
	frozen_provenance = validate_transition_provenance(
		provenance,
		expected_seed=int(seed),
		expected_wind_manifest=canonical_wind_manifest,
		expected_wind_weight=wind_weight_value,
		expected_simulation_valid_mask_sha256=computed_simulation_valid_hash,
		expected_mapped_building_mask_sha256=computed_mapped_building_hash,
	)
	if frozen_provenance["transition_source_type"] != transition_source_type:
		raise ValueError(
			"transition_source_type must match transition provenance"
		)
	blazing = before == STATE_BLAZING
	blazing_neighbor_count = compute_moore_neighbor_count(blazing.astype(np.int8))
	wind_weighted_score, _ = compute_wind_weighted_score(
		blazing.astype(np.int8), wind, wind_weight_value
	)
	eligible = (
		domain
		& (before == STATE_NOT_YET_BURNING)
		& (blazing_neighbor_count > 0)
	)
	newly_ignited = eligible & (after == STATE_IGNITED)

	if np.any(~np.isin(after[eligible], (STATE_NOT_YET_BURNING, STATE_IGNITED))):
		raise ValueError("Eligible cells at t+1 must be only state 2 or state 3")
	if np.any((before == STATE_NOT_YET_BURNING) & (after == STATE_IGNITED) & ~eligible):
		raise ValueError("A state-2 to state-3 transition occurred outside eligibility")
	if np.any(newly_ignited & ~eligible):
		raise ValueError("Positive ignition labels must be a subset of eligible cells")
	if np.any(eligible & ~domain) or np.any(newly_ignited & ~domain):
		raise ValueError("Eligible and positive cells must be valid mapped buildings")

	mask_hashes = MappingProxyType(
		{
			"simulation_valid_mask_sha256": computed_simulation_valid_hash,
			"mapped_building_mask_sha256": computed_mapped_building_hash,
			"eligible_mask_t_sha256": mask_sha256(eligible),
		}
	)
	return TransitionObservation(
		schema_version=TRANSITION_OBSERVATION_SCHEMA_VERSION,
		transition_source_type=AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
		state_encoding=STATE_ENCODING,
		state_codes=STATE_CODES,
		timestep_t=int(timestep_t),
		timestep_t1=int(timestep_t1),
		state_t=_immutable_array(before, np.int8),
		state_t1=_immutable_array(after, np.int8),
		blazing_mask_t=_immutable_array(blazing, bool),
		blazing_neighbor_count_t=_immutable_array(blazing_neighbor_count, np.int8),
		wind_weighted_score_t=_immutable_array(wind_weighted_score, np.float32),
		simulation_valid_mask=_immutable_array(simulation_valid, bool),
		mapped_building_mask=_immutable_array(mapped_building, bool),
		eligible_mask_t=_immutable_array(eligible, bool),
		newly_ignited_mask_t1=_immutable_array(newly_ignited, bool),
		wind_manifest=_freeze_metadata(canonical_wind_manifest),  # type: ignore[arg-type]
		wind_weight=wind_weight_value,
		seed=int(seed),
		grid_shape=before.shape,
		grid_crs=crs_value,
		grid_transform=transform_value,
		domain_mask_hashes=mask_hashes,
		provenance=frozen_provenance,
	)


def validate_transition_observation(
	observation: object,
) -> TransitionObservation:
	"""Validate an observation and return a canonical immutable copy."""
	if not isinstance(observation, TransitionObservation):
		raise TypeError("observation must be a TransitionObservation")
	if observation.schema_version != TRANSITION_OBSERVATION_SCHEMA_VERSION:
		raise ValueError("Unsupported transition observation schema_version")
	if observation.transition_source_type != AUTHORIZED_SIMULATED_TRANSITION_SOURCE:
		raise ValueError("transition_source_type must be 'authorized_simulated'")
	if observation.state_encoding != STATE_ENCODING:
		raise ValueError("Transition observation must use ca_five_state_int8.v1")
	if dict(observation.state_codes) != dict(STATE_CODES):
		raise ValueError("Transition observation state codes do not match the CA contract")

	array_names = (
		"state_t",
		"state_t1",
		"blazing_mask_t",
		"blazing_neighbor_count_t",
		"wind_weighted_score_t",
		"simulation_valid_mask",
		"mapped_building_mask",
		"eligible_mask_t",
		"newly_ignited_mask_t1",
	)
	for name in array_names:
		if np.asarray(getattr(observation, name)).flags.writeable:
			raise ValueError(f"Transition observation {name} must be read-only")

	rebuilt = build_transition_observation(
		state_t=observation.state_t,
		state_t1=observation.state_t1,
		transition_source_type=observation.transition_source_type,
		timestep_t=observation.timestep_t,
		timestep_t1=observation.timestep_t1,
		simulation_valid_mask=observation.simulation_valid_mask,
		mapped_building_mask=observation.mapped_building_mask,
		wind_manifest=observation.wind_manifest,
		wind_weight=observation.wind_weight,
		seed=observation.seed,
		provenance=observation.provenance,
		grid_crs=observation.grid_crs,
		grid_transform=observation.grid_transform,
	)
	if observation.grid_shape != rebuilt.grid_shape:
		raise ValueError("Transition observation grid_shape does not match its states")
	for name in array_names:
		if not np.array_equal(getattr(observation, name), getattr(rebuilt, name)):
			raise ValueError(f"Transition observation {name} is inconsistent")
	if dict(observation.domain_mask_hashes) != dict(rebuilt.domain_mask_hashes):
		raise ValueError("Transition observation domain mask hashes are inconsistent")
	return rebuilt
