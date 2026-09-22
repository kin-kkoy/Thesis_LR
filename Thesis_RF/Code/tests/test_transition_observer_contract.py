from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import modules.automata_engine as automata_engine_module
from dataset_generator import (
	LEGACY_BINARY_STATE_CONTRACT_STATUS,
	LEGACY_BINARY_STATE_ENCODING,
	TRANSITION_DYNAMIC_FEATURE_NAMES,
	TRANSITION_ROW_BATCH_SCHEMA_VERSION,
	TRANSITION_TARGET_VERSION,
	build_in_memory_transition_rows,
)
from modules.automata_engine import (
	STATE_BLAZING,
	STATE_EXTINGUISHED,
	STATE_IGNITED,
	STATE_NON_BURNABLE,
	STATE_NOT_YET_BURNING,
	FireAutomata,
)
from modules.ca_state import (
	STATE_BLAZING as OWNED_STATE_BLAZING,
	STATE_ENCODING as OWNED_STATE_ENCODING,
	validate_full_state_grid,
)
from modules.transition_observer import (
	AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
	SOURCE_ONLY_AUTHORIZATION,
	STATE_ENCODING,
	TRANSITION_OBSERVATION_SCHEMA_VERSION,
	build_transition_observation,
	canonical_metadata_hash,
	mask_sha256,
)
from modules.wind_convention import (
	CALM_REPRESENTATION,
	DIRECTION_CONVENTION,
	DIRECTION_UNITS,
	NORTH_REFERENCE,
	WIND_SCHEMA_VERSION,
	WindContract,
)


def _config() -> dict:
	return {
		"simulation": {"seed": 19},
		"wind": {
			"speed_kmh": 12.0,
			"direction_deg": 45.0,
			"direction_convention": DIRECTION_CONVENTION,
			"direction_units": DIRECTION_UNITS,
			"north_reference": NORTH_REFERENCE,
			"schema_version": WIND_SCHEMA_VERSION,
			"calm_representation": CALM_REPRESENTATION,
		},
		"placeholder_transition": {
			"base_ignition_prob": 1.0,
			"wind_weight": 2.0,
			"T_3_to_4": 1,
			"T_4_to_5": 1,
		},
		"ml_model": {
			"inference_mode": "stochastic_probability",
			"threshold": None,
		},
	}


def _environment(shape: tuple[int, int] = (3, 5)) -> dict:
	return {
		"slope_risk": np.ones(shape, dtype=np.float32),
		"proximity_risk": np.zeros(shape, dtype=np.float32),
		"building_presence": np.ones(shape, dtype=np.float32),
		"material_class": np.ones(shape, dtype=np.int8),
		"burnable_mask": np.ones(shape, dtype=bool),
		"nodata_mask": np.zeros(shape, dtype=bool),
		"grid_shape": shape,
		"transform": (1.0, 0.0, 100.0, 0.0, -1.0, 200.0),
		"crs": "EPSG:32651",
	}


def _provenance(
	config: dict,
	*,
	shape: tuple[int, int] = (3, 5),
	simulation_valid_mask: np.ndarray | None = None,
	mapped_building_mask: np.ndarray | None = None,
) -> dict:
	if simulation_valid_mask is None:
		simulation_valid_mask = np.ones(shape, dtype=bool)
	if mapped_building_mask is None:
		mapped_building_mask = np.ones(shape, dtype=bool)
	return {
		"set_id": "synthetic-set-c",
		"experiment_id": "synthetic-experiment",
		"event_id": "synthetic-event",
		"scenario_id": "synthetic-scenario",
		"run_id": "synthetic-run",
		"grid_id": "synthetic-grid",
		"simulation_valid_mask_id": "synthetic-environment-valid",
		"mapped_building_mask_id": "synthetic-buildings",
		"simulator_id": "fire-automata",
		"simulator_version": "synthetic-test-v1",
		"source_revision": "synthetic-revision",
		"source_dirty": False,
		"configuration_hash": canonical_metadata_hash(config),
		"input_hashes": {"synthetic-input": "a" * 64},
		"authorization": SOURCE_ONLY_AUTHORIZATION,
		"authorization_reference": "D-012/D-013 source-only approval",
		"transition_source_type": AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
		"seed": config["simulation"]["seed"],
		"wind_manifest": WindContract.from_config(config["wind"]).to_manifest(),
		"wind_weight": config["placeholder_transition"]["wind_weight"],
		"simulation_valid_mask_sha256": mask_sha256(simulation_valid_mask),
		"mapped_building_mask_sha256": mask_sha256(mapped_building_mask),
		"environment_id": "synthetic-environment",
		"execution_id": "synthetic-execution",
		"capture_source_id": "synthetic-live-ca-step",
	}


def _direct_observation_inputs() -> dict:
	config = _config()
	simulation_valid = np.ones((3, 5), dtype=bool)
	simulation_valid[0, 0] = False
	mapped_building = np.ones((3, 5), dtype=bool)
	mapped_building[0, 1] = False

	state_t = np.full((3, 5), STATE_NOT_YET_BURNING, dtype=np.int8)
	state_t[~(simulation_valid & mapped_building)] = STATE_NON_BURNABLE
	state_t[1, 0] = STATE_IGNITED
	state_t[1, 1] = STATE_BLAZING
	state_t[1, 3] = STATE_EXTINGUISHED
	state_t[2, 2] = STATE_IGNITED
	state_t[2, 3] = STATE_BLAZING
	state_t1 = state_t.copy()
	state_t1[1, 0] = STATE_BLAZING
	state_t1[1, 1] = STATE_EXTINGUISHED
	state_t1[1, 2] = STATE_IGNITED

	return {
		"state_t": state_t,
		"state_t1": state_t1,
		"transition_source_type": AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
		"timestep_t": 4,
		"timestep_t1": 5,
		"simulation_valid_mask": simulation_valid,
		"mapped_building_mask": mapped_building,
		"wind_manifest": WindContract.from_config(config["wind"]).to_manifest(),
		"wind_weight": config["placeholder_transition"]["wind_weight"],
		"seed": config["simulation"]["seed"],
		"provenance": _provenance(
			config,
			simulation_valid_mask=simulation_valid,
			mapped_building_mask=mapped_building,
		),
		"grid_crs": "EPSG:32651",
		"grid_transform": (1.0, 0.0, 100.0, 0.0, -1.0, 200.0),
	}


def test_record_preserves_five_states_and_exact_transition_semantics():
	record = build_transition_observation(**_direct_observation_inputs())

	assert record.schema_version == TRANSITION_OBSERVATION_SCHEMA_VERSION
	assert record.transition_source_type == AUTHORIZED_SIMULATED_TRANSITION_SOURCE
	assert record.state_encoding == STATE_ENCODING
	assert record.state_t.dtype == np.int8
	assert record.state_t1.dtype == np.int8
	assert set(np.unique(record.state_t)) == {1, 2, 3, 4, 5}
	assert record.state_codes == {
		"non_burnable": 1,
		"not_yet_burning": 2,
		"ignited": 3,
		"blazing": 4,
		"extinguished": 5,
	}
	assert record.blazing_mask_t[1, 1]
	assert not record.blazing_mask_t[1, 0]
	assert not record.blazing_mask_t[1, 3]
	assert record.newly_ignited_mask_t1[1, 2]
	assert np.count_nonzero(record.newly_ignited_mask_t1) == 1
	assert not record.newly_ignited_mask_t1[1, 0]  # 3 -> 4
	assert not record.newly_ignited_mask_t1[1, 1]  # 4 -> 5
	assert not record.newly_ignited_mask_t1[1, 3]  # 5 -> 5


def test_neutral_state_module_owns_constants_with_engine_compatibility_exports():
	assert STATE_BLAZING is OWNED_STATE_BLAZING
	assert STATE_ENCODING == OWNED_STATE_ENCODING
	grid = np.full((1, 1), STATE_BLAZING, dtype=np.int8)
	assert validate_full_state_grid(grid) is grid


def test_transition_source_type_is_required_and_fails_closed():
	inputs = _direct_observation_inputs()
	inputs["transition_source_type"] = "legacy_binary_burning_0_1.v1"
	with pytest.raises(ValueError, match="authorized_simulated"):
		build_transition_observation(**inputs)

	inputs = _direct_observation_inputs()
	del inputs["provenance"]["transition_source_type"]
	with pytest.raises(ValueError, match="transition_source_type"):
		build_transition_observation(**inputs)


def test_only_blazing_drives_neighbor_and_wind_features_and_domains_exclude_cells():
	inputs = _direct_observation_inputs()
	# Remove the sole blazing source. Nearby ignited/extinguished states must not drive.
	inputs["state_t"][1, 1] = STATE_NOT_YET_BURNING
	inputs["state_t1"][1, 1] = STATE_NOT_YET_BURNING
	inputs["state_t1"][1, 2] = STATE_NOT_YET_BURNING
	inputs["state_t"][2, 3] = STATE_NOT_YET_BURNING
	inputs["state_t1"][2, 3] = STATE_NOT_YET_BURNING
	record = build_transition_observation(**inputs)

	assert not np.any(record.blazing_mask_t)
	assert not np.any(record.blazing_neighbor_count_t)
	assert not np.any(record.wind_weighted_score_t)
	assert not np.any(record.eligible_mask_t)
	assert not record.eligible_mask_t[0, 0]  # environmentally invalid/non-burnable
	assert not record.eligible_mask_t[0, 1]  # off-building/non-burnable
	assert not record.eligible_mask_t[2, 4]  # valid building with no blazing neighbor


class _ZeroRng:
	def random(self, shape):
		return np.zeros(shape, dtype=np.float64)


def test_blazing_cell_that_extinguishes_still_drives_the_t_to_t1_transition():
	config = _config()
	records = []
	automata = FireAutomata(
		_environment((1, 2)),
		config,
		transition_observer=records.append,
		transition_provenance=_provenance(config, shape=(1, 2)),
	)
	automata.grid[0, 0] = STATE_BLAZING
	automata.rng = _ZeroRng()

	automata.step()

	record = records[0]
	assert record.state_t[0, 0] == STATE_BLAZING
	assert record.state_t1[0, 0] == STATE_EXTINGUISHED
	assert not record.newly_ignited_mask_t1[0, 0]  # initial seed is not a label
	assert record.blazing_neighbor_count_t[0, 1] == 1
	assert record.eligible_mask_t[0, 1]
	assert record.newly_ignited_mask_t1[0, 1]
	assert record.state_t1[0, 1] == STATE_IGNITED


@pytest.mark.parametrize(
	("mutation", "error_type", "message"),
	[
		(lambda values: values.update(state_t=values["state_t"].astype(np.float32)), TypeError, "np.int8"),
		(
			lambda values: values.update(state_t=values["state_t"].ravel()),
			ValueError,
			"two-dimensional",
		),
		(lambda values: values["state_t"].__setitem__((0, 2), np.int8(0)), ValueError, "codes 1 through 5"),
		(
			lambda values: values.update(state_t1=values["state_t1"][:, :-1]),
			ValueError,
			"equal shapes",
		),
		(lambda values: values.update(timestep_t1=8), ValueError, r"timestep_t \+ 1"),
	],
)
def test_binary_float_bad_shape_and_nonadjacent_records_fail_closed(
	mutation, error_type, message
):
	inputs = _direct_observation_inputs()
	if "codes" in message:
		inputs["state_t"] = np.where(
			inputs["state_t"] == STATE_NON_BURNABLE, 1, 0
		).astype(np.int8)
	mutation(inputs)
	with pytest.raises(error_type, match=message):
		build_transition_observation(**inputs)


def test_positive_and_eligible_transition_violations_fail_closed():
	outside_eligibility = _direct_observation_inputs()
	outside_eligibility["state_t1"][0, 4] = STATE_IGNITED
	with pytest.raises(ValueError, match="outside eligibility"):
		build_transition_observation(**outside_eligibility)

	bad_eligible_destination = _direct_observation_inputs()
	bad_eligible_destination["state_t1"][1, 2] = STATE_BLAZING
	with pytest.raises(ValueError, match="Impossible synchronous CA state transition"):
		build_transition_observation(**bad_eligible_destination)


@pytest.mark.parametrize(
	("row", "col", "source", "destination", "pair"),
	[
		(0, 0, STATE_NON_BURNABLE, STATE_NOT_YET_BURNING, "1->2"),
		(1, 0, STATE_IGNITED, STATE_EXTINGUISHED, "3->5"),
		(1, 1, STATE_BLAZING, STATE_IGNITED, "4->3"),
		(1, 3, STATE_EXTINGUISHED, STATE_NOT_YET_BURNING, "5->2"),
	],
)
def test_impossible_full_state_transitions_fail_closed(
	row, col, source, destination, pair
):
	inputs = _direct_observation_inputs()
	inputs["state_t"][row, col] = source
	inputs["state_t1"][row, col] = destination
	with pytest.raises(ValueError, match=pair):
		build_transition_observation(**inputs)


def test_valid_ignited_and_blazing_timer_persistence_is_accepted():
	record = build_transition_observation(**_direct_observation_inputs())
	assert record.state_t[2, 2] == STATE_IGNITED
	assert record.state_t1[2, 2] == STATE_IGNITED
	assert record.state_t[2, 3] == STATE_BLAZING
	assert record.state_t1[2, 3] == STATE_BLAZING


def test_wrong_mask_dtype_and_burnable_invariant_fail_closed():
	wrong_dtype = _direct_observation_inputs()
	wrong_dtype["simulation_valid_mask"] = wrong_dtype[
		"simulation_valid_mask"
	].astype(np.int8)
	with pytest.raises(TypeError, match="boolean array"):
		build_transition_observation(**wrong_dtype)

	bad_invariant = _direct_observation_inputs()
	bad_invariant["state_t"][0, 0] = STATE_NOT_YET_BURNING
	bad_invariant["state_t1"][0, 0] = STATE_NOT_YET_BURNING
	with pytest.raises(ValueError, match="burnable invariant"):
		build_transition_observation(**bad_invariant)


def test_wind_and_provenance_are_canonical_and_bound_to_active_configuration():
	inputs = _direct_observation_inputs()
	inputs["wind_manifest"] = dict(inputs["wind_manifest"])
	del inputs["wind_manifest"]["north_reference"]
	with pytest.raises(ValueError, match="north_reference"):
		build_transition_observation(**inputs)

	config = _config()
	missing = _provenance(config)
	del missing["run_id"]
	with pytest.raises(ValueError, match="run_id"):
		FireAutomata(
			_environment(),
			config,
			transition_observer=lambda record: None,
			transition_provenance=missing,
		)

	stale = _provenance(config)
	stale["configuration_hash"] = "b" * 64
	with pytest.raises(ValueError, match="active config"):
		FireAutomata(
			_environment(),
			config,
			transition_observer=lambda record: None,
			transition_provenance=stale,
		)


@pytest.mark.parametrize(
	("field", "replacement", "message"),
	[
		("seed", 999, "active seed"),
		("wind_weight", 999.0, "active wind weight"),
		("simulation_valid_mask_sha256", "b" * 64, "domain mask"),
		("mapped_building_mask_sha256", "b" * 64, "domain mask"),
	],
)
def test_enabled_observer_rejects_mismatched_capture_provenance_before_step(
	field, replacement, message
):
	config = _config()
	provenance = _provenance(config)
	provenance[field] = replacement
	with pytest.raises(ValueError, match=message):
		FireAutomata(
			_environment(),
			config,
			transition_observer=lambda record: None,
			transition_provenance=provenance,
		)


def test_in_memory_row_adapter_selects_only_eligible_cells_without_mutation():
	record = build_transition_observation(**_direct_observation_inputs())
	state_t_before = record.state_t.copy()
	rows = build_in_memory_transition_rows(record)
	selected = np.flatnonzero(record.eligible_mask_t.ravel())
	expected_rows, expected_cols = np.unravel_index(selected, record.grid_shape)
	expected_features = np.column_stack(
		(
			record.blazing_neighbor_count_t.ravel()[selected],
			record.wind_weighted_score_t.ravel()[selected],
		)
	).astype(np.float32)

	assert rows.schema_version == TRANSITION_ROW_BATCH_SCHEMA_VERSION
	assert rows.feature_names == TRANSITION_DYNAMIC_FEATURE_NAMES
	assert rows.target_version == TRANSITION_TARGET_VERSION
	assert rows.transition_source_type == AUTHORIZED_SIMULATED_TRANSITION_SOURCE
	assert rows.features == pytest.approx(expected_features)
	assert rows.labels.tolist() == record.newly_ignited_mask_t1.ravel()[
		selected
	].astype(np.int8).tolist()
	assert rows.cell_rows.tolist() == expected_rows.tolist()
	assert rows.cell_cols.tolist() == expected_cols.tolist()
	assert rows.timestep_t1 == rows.timestep_t + 1
	assert rows.run_id == "synthetic-run"
	assert rows.provenance["simulator_id"] == "fire-automata"
	assert rows.provenance["configuration_hash"] == canonical_metadata_hash(_config())
	assert rows.provenance["authorization_reference"] == (
		"D-012/D-013 source-only approval"
	)
	assert not hasattr(rows, "state_t")
	assert not hasattr(rows, "state_t1")
	assert not hasattr(rows, "save")
	assert not rows.features.flags.writeable
	assert not rows.labels.flags.writeable
	assert np.array_equal(record.state_t, state_t_before)


def test_legacy_binary_state_contract_is_explicitly_non_authoritative():
	assert LEGACY_BINARY_STATE_ENCODING == "binary_burning_0_1"
	assert LEGACY_BINARY_STATE_CONTRACT_STATUS == "deprecated_non_authoritative"
	with pytest.raises(TypeError, match="TransitionObservation"):
		build_in_memory_transition_rows(object())


@pytest.mark.parametrize("missing_name", ["crs", "transform"])
def test_enabled_observer_requires_complete_grid_provenance(missing_name):
	config = _config()
	environment = _environment()
	environment[missing_name] = None
	with pytest.raises(ValueError, match=f"grid_{missing_name}"):
		FireAutomata(
			environment,
			config,
			transition_observer=lambda record: None,
			transition_provenance=_provenance(config),
		)
	# Missing grid provenance remains backward compatible when observation is disabled.
	FireAutomata(environment, config)


@pytest.mark.parametrize(
	("field", "message"),
	[("grid_crs", "grid_crs"), ("grid_transform", "grid_transform")],
)
def test_direct_record_requires_complete_grid_provenance(field, message):
	inputs = _direct_observation_inputs()
	inputs[field] = None
	with pytest.raises(ValueError, match=message):
		build_transition_observation(**inputs)


@pytest.mark.parametrize("bad_transform", ["123456", (1.0,), (1.0,) * 7])
def test_grid_transform_rejects_text_and_incomplete_coefficient_sequences(bad_transform):
	inputs = _direct_observation_inputs()
	inputs["grid_transform"] = bad_transform
	with pytest.raises((TypeError, ValueError), match="grid_transform"):
		build_transition_observation(**inputs)


def test_disabled_observer_preserves_legacy_mask_coercion_but_enabled_fails_closed():
	config = _config()
	legacy_environment = _environment()
	legacy_environment["burnable_mask"] = legacy_environment[
		"burnable_mask"
	].astype(np.int8)
	legacy_environment["nodata_mask"] = legacy_environment["nodata_mask"].astype(
		np.int8
	)

	disabled = FireAutomata(legacy_environment, config)
	assert disabled.simulation_valid_mask.dtype == bool
	assert disabled.nodata_mask.dtype == bool
	with pytest.raises(TypeError, match="boolean array without coercion"):
		FireAutomata(
			legacy_environment,
			config,
			transition_observer=lambda record: None,
			transition_provenance=_provenance(config),
		)

	nonbinary_buildings = _environment()
	nonbinary_buildings["building_presence"][0, 0] = 2.0
	FireAutomata(nonbinary_buildings, config)
	with pytest.raises(ValueError, match="mapped values 0 and 1"):
		FireAutomata(
			nonbinary_buildings,
			config,
			transition_observer=lambda record: None,
			transition_provenance=_provenance(config),
		)


def test_records_are_read_only_and_metadata_is_immutable_and_deep_copied():
	inputs = _direct_observation_inputs()
	caller_provenance = inputs["provenance"]
	record = build_transition_observation(**inputs)

	caller_provenance["run_id"] = "mutated-after-capture"
	inputs["state_t"][1, 1] = STATE_EXTINGUISHED
	assert record.provenance["run_id"] == "synthetic-run"
	assert record.state_t[1, 1] == STATE_BLAZING
	with pytest.raises(ValueError):
		record.state_t[0, 0] = STATE_EXTINGUISHED
	with pytest.raises(TypeError):
		record.provenance["run_id"] = "observer-mutation"
	assert not record.state_t.flags.writeable
	assert not record.newly_ignited_mask_t1.flags.writeable


def _assert_same_automata_state(left: FireAutomata, right: FireAutomata) -> None:
	assert np.array_equal(left.grid, right.grid)
	assert np.array_equal(left.ignition_timers, right.ignition_timers)
	assert np.array_equal(left.blazing_timers, right.blazing_timers)
	assert left.rng.bit_generator.state == right.rng.bit_generator.state
	assert left.timestep == right.timestep


def test_disabled_noop_and_collecting_observers_do_not_change_ca_or_rng():
	config = _config()
	disabled = FireAutomata(_environment(), deepcopy(config))
	noop = FireAutomata(
		_environment(),
		deepcopy(config),
		transition_observer=lambda record: None,
		transition_provenance=_provenance(config),
	)
	records = []
	collecting = FireAutomata(
		_environment(),
		deepcopy(config),
		transition_observer=records.append,
		transition_provenance=_provenance(config),
	)
	for automata in (disabled, noop, collecting):
		automata.set_ignition([(1, 1)])

	for _ in range(2):
		disabled.step()
		noop.step()
		collecting.step()

	_assert_same_automata_state(disabled, noop)
	_assert_same_automata_state(disabled, collecting)
	assert len(records) == 2


def test_engine_callback_mutation_attempt_is_rejected_without_affecting_ca_or_rng():
	config = _config()
	control = FireAutomata(_environment(), deepcopy(config))
	attempts = []

	def mutation_attempting_observer(record):
		with pytest.raises(ValueError):
			record.state_t1[0, 0] = STATE_NOT_YET_BURNING
		attempts.append("state_rejected")
		with pytest.raises(ValueError):
			record.eligible_mask_t[0, 0] = False
		attempts.append("mask_rejected")
		with pytest.raises(TypeError):
			record.provenance["run_id"] = "mutated"
		attempts.append("metadata_rejected")

	observed = FireAutomata(
		_environment(),
		deepcopy(config),
		transition_observer=mutation_attempting_observer,
		transition_provenance=_provenance(config),
	)
	for automata in (control, observed):
		automata.set_ignition([(1, 1)])

	control.step()
	observed.step()

	assert attempts == ["state_rejected", "mask_rejected", "metadata_rejected"]
	_assert_same_automata_state(control, observed)


def test_observer_failure_occurs_after_commit_without_emergency_checkpoint(monkeypatch):
	config = _config()
	control = FireAutomata(_environment(), deepcopy(config))

	class ObserverFailure(RuntimeError):
		pass

	def fail_observer(record):
		raise ObserverFailure("synthetic observer failure")

	observed = FireAutomata(
		_environment(),
		deepcopy(config),
		transition_observer=fail_observer,
		transition_provenance=_provenance(config),
	)
	for automata in (control, observed):
		automata.set_ignition([(1, 1)])

	checkpoint_attempted = False

	def forbidden_checkpoint():
		nonlocal checkpoint_attempted
		checkpoint_attempted = True
		raise AssertionError("observer failures must not create checkpoints")

	monkeypatch.setattr(observed, "_save_emergency_checkpoint", forbidden_checkpoint)
	control.step()
	with pytest.raises(ObserverFailure, match="synthetic observer failure"):
		observed.step()

	assert not checkpoint_attempted
	_assert_same_automata_state(control, observed)


def test_observed_internal_step_failure_never_attempts_emergency_checkpoint(
	monkeypatch,
):
	config = _config()
	automata = FireAutomata(
		_environment(),
		config,
		transition_observer=lambda record: None,
		transition_provenance=_provenance(config),
	)
	automata.set_ignition([(1, 1)])
	checkpoint_attempted = False

	def fail_inside_transition(_state):
		raise ArithmeticError("synthetic internal transition failure")

	def forbidden_checkpoint():
		nonlocal checkpoint_attempted
		checkpoint_attempted = True
		raise AssertionError("source-only mode must not write checkpoints")

	monkeypatch.setattr(
		automata_engine_module,
		"compute_moore_neighbor_count",
		fail_inside_transition,
	)
	monkeypatch.setattr(automata, "_save_emergency_checkpoint", forbidden_checkpoint)

	with pytest.raises(RuntimeError, match="no emergency checkpoint was written"):
		automata.step()
	assert not checkpoint_attempted


def test_unobserved_internal_step_failure_retains_emergency_checkpoint_hook(
	monkeypatch,
):
	config = _config()
	automata = FireAutomata(_environment(), config)
	automata.set_ignition([(1, 1)])
	checkpoint_attempts = []

	def fail_inside_transition(_state):
		raise ArithmeticError("synthetic internal transition failure")

	def fake_checkpoint():
		checkpoint_attempts.append("synthetic-emergency-checkpoint.npz")
		return checkpoint_attempts[-1]

	monkeypatch.setattr(
		automata_engine_module,
		"compute_moore_neighbor_count",
		fail_inside_transition,
	)
	monkeypatch.setattr(automata, "_save_emergency_checkpoint", fake_checkpoint)

	with pytest.raises(RuntimeError, match="emergency checkpoint was saved"):
		automata.step()
	assert checkpoint_attempts == ["synthetic-emergency-checkpoint.npz"]
