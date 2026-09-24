from __future__ import annotations

from pathlib import Path

import pytest

import orchestrator


def _minimal_config() -> dict:
	return {
		"environment": {"raster_dir": "unused"},
		"simulation": {},
	}


def _isolate_before_model_or_output(monkeypatch):
	constructed = []

	class FakeEnvironmentManager:
		def __init__(self, config):
			self.config = config

		def load_rasters(self):
			return None

		def build_masks(self):
			return None

		def normalize_layers(self):
			return None

		def summary(self):
			return None

		def get_environment(self):
			return {"synthetic": "environment"}

	class FakeAutomata:
		def __init__(self, *args, **kwargs):
			constructed.append((args, kwargs))

	def no_model(_automata, _model_path):
		raise FileNotFoundError("synthetic stop before stepping or output")

	monkeypatch.setattr(orchestrator, "EnvironmentManager", FakeEnvironmentManager)
	monkeypatch.setattr(orchestrator, "FireAutomata", FakeAutomata)
	monkeypatch.setattr(orchestrator, "load_model", no_model)
	return constructed


def test_orchestrator_observer_boundary_is_disabled_by_default(monkeypatch):
	constructed = _isolate_before_model_or_output(monkeypatch)

	orchestrator.run_simulation(_minimal_config())

	assert len(constructed) == 1
	assert constructed[0][1] == {}


def test_orchestrator_passes_enabled_observer_and_provenance_without_output(
	monkeypatch,
):
	constructed = _isolate_before_model_or_output(monkeypatch)
	records = []
	provenance = {"synthetic": "complete context validated by FireAutomata"}

	orchestrator.run_simulation(
		_minimal_config(),
		transition_observer=records.append,
		transition_provenance=provenance,
	)

	assert len(constructed) == 1
	assert constructed[0][1] == {
		"transition_observer": records.append,
		"transition_provenance": provenance,
	}


def test_orchestrator_rejects_incomplete_observer_pair_before_environment_access(
	monkeypatch,
):
	accessed_environment = False

	class ForbiddenEnvironmentManager:
		def __init__(self, _config):
			nonlocal accessed_environment
			accessed_environment = True

	monkeypatch.setattr(
		orchestrator, "EnvironmentManager", ForbiddenEnvironmentManager
	)
	with pytest.raises(ValueError, match="required when an observer"):
		orchestrator.run_simulation(
			_minimal_config(), transition_observer=lambda record: None
		)
	assert not accessed_environment


def test_enabled_observation_path_never_writes_or_checkpoints(monkeypatch):
	writes = []

	class FakeEnvironmentManager:
		def __init__(self, _config):
			pass

		def load_rasters(self):
			return None

		def build_masks(self):
			return None

		def normalize_layers(self):
			return None

		def summary(self):
			return None

		def get_environment(self):
			return {}

	class FakeAutomata:
		timestep = 0

		def __init__(self, *_args, **_kwargs):
			pass

		def set_ignition(self, _points):
			return None

		def step(self):
			self.timestep += 1

		def is_active(self):
			return False

		def save_checkpoint(self, _path):
			writes.append("checkpoint")

	monkeypatch.setattr(orchestrator, "EnvironmentManager", FakeEnvironmentManager)
	monkeypatch.setattr(orchestrator, "FireAutomata", FakeAutomata)
	monkeypatch.setattr(orchestrator, "load_model", lambda *_args: None)
	monkeypatch.setattr(
		orchestrator, "_pick_random_ignition_points", lambda *_args, **_kwargs: [(0, 0)]
	)
	monkeypatch.setattr(Path, "mkdir", lambda *_args, **_kwargs: writes.append("mkdir"))
	monkeypatch.setattr(
		orchestrator.rasterio,
		"open",
		lambda *_args, **_kwargs: writes.append("raster") or None,
	)
	config = _minimal_config()
	config["simulation"] = {"max_timesteps": 1, "checkpoint_interval": 1}

	orchestrator.run_simulation(
		config,
		transition_observer=lambda record: None,
		transition_provenance={"synthetic": "validated by the engine in production"},
	)

	assert writes == []


def test_generic_joblib_loader_delegates_to_estimator_boundary(monkeypatch):
	loaded = []

	class GenericEstimatorHost:
		def load_model(self, path):
			loaded.append(path)

	monkeypatch.setattr(Path, "exists", lambda self: True)
	orchestrator.load_model(GenericEstimatorHost(), "models/synthetic.joblib")

	assert len(loaded) == 1
	assert loaded[0].endswith("synthetic.joblib")
	with pytest.raises(ValueError, match="extensions"):
		orchestrator.load_model(GenericEstimatorHost(), "models/synthetic.txt")


def test_dedicated_teacher_path_never_loads_model_or_writes(monkeypatch):
	writes = []
	engine_calls = []
	observer = lambda observation: None
	provenance = {"synthetic": "validated by the engine"}
	termination_record = object()

	class FakeEnvironmentManager:
		def __init__(self, config):
			self.config = config

		def load_rasters(self):
			return None

		def build_masks(self):
			return None

		def normalize_layers(self):
			return None

		def get_environment(self):
			return {
				"grid_shape": (2, 2),
				"transform": orchestrator.rasterio.transform.from_origin(
					100.0, 200.0, 1.0, 1.0
				),
			}

	def fake_teacher_engine(environment, config, **kwargs):
		engine_calls.append((environment, config, kwargs))
		return termination_record

	monkeypatch.setattr(orchestrator, "EnvironmentManager", FakeEnvironmentManager)
	monkeypatch.setattr(
		orchestrator, "run_model_free_teacher_engine", fake_teacher_engine
	)
	monkeypatch.setattr(
		orchestrator,
		"load_model",
		lambda *_args, **_kwargs: pytest.fail("teacher path attempted model loading"),
	)
	monkeypatch.setattr(Path, "mkdir", lambda *_args, **_kwargs: writes.append("mkdir"))
	monkeypatch.setattr(
		orchestrator.rasterio,
		"open",
		lambda *_args, **_kwargs: writes.append("raster") or None,
	)
	config = {
		"environment": {"raster_dir": "synthetic"},
		"simulation": {"ignition_points": [[100.5, 199.5]]},
		"model_free_teacher": {
			"enabled": True,
			"runner_schema_version": "model_free_teacher_run.v1",
			"safety_limit_timesteps": 1,
			"persistent_output_enabled": False,
		},
	}

	result = orchestrator.run_model_free_teacher(
		config,
		transition_observer=observer,
		transition_provenance=provenance,
	)

	assert result is termination_record
	assert writes == []
	assert len(engine_calls) == 1
	assert engine_calls[0][2] == {
		"ignition_points": [(0, 0)],
		"transition_observer": observer,
		"transition_provenance": provenance,
	}


def test_dedicated_teacher_is_disabled_before_environment_access(monkeypatch):
	accessed_environment = False

	class ForbiddenEnvironmentManager:
		def __init__(self, _config):
			nonlocal accessed_environment
			accessed_environment = True

	monkeypatch.setattr(
		orchestrator, "EnvironmentManager", ForbiddenEnvironmentManager
	)
	with pytest.raises(ValueError, match="enabled must be true"):
		orchestrator.run_model_free_teacher(
			{
				"environment": {"raster_dir": "unused"},
				"simulation": {"ignition_points": [[0.0, 0.0]]},
				"model_free_teacher": {"enabled": False},
			},
			transition_observer=lambda observation: None,
			transition_provenance={"synthetic": "not reached"},
		)
	assert not accessed_environment
