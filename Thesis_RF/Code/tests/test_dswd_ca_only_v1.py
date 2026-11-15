"""Synthetic Phase 17 contracts; intentionally does not read project rasters."""
import tempfile
from pathlib import Path
import unittest
from contextlib import ExitStack
from unittest import mock

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

try:
    from .. import dswd_ca_only_v1 as phase17
    from .. import orchestrator
    from ..modules.automata_engine import FireAutomata
except ImportError:
    import dswd_ca_only_v1 as phase17
    import orchestrator
    from modules.automata_engine import FireAutomata


class _ClockedAutomata:
    def __init__(self, *, stop=False, fail=False):
        self.grid = np.full((3, 3), 2, dtype=np.int8)
        self.grid[1, 1] = 4
        self.timestep = 0
        self.stop = stop
        self.fail = fail

    def get_grid(self):
        return self.grid.copy()

    def step(self):
        if self.fail:
            raise RuntimeError("synthetic step failure")
        self.timestep += 1
        if self.stop:
            self.grid[1, 1] = 5


def _basis(horizons):
    return {"incident": "airbase", "seed": 42, "duration_basis": None,
            "horizons": [{"requested_update": h, "assumed_minutes_per_update": 1,
                          "clock_role": "primary" if h == 2 else "sensitivity"}
                         for h in horizons]}


def _fire_engine(seed=42):
    shape = (5, 5)
    environment = {
        "slope_risk": np.full(shape, 0.2, dtype=np.float32),
        "proximity_risk": np.full(shape, 0.1, dtype=np.float32),
        "building_presence": np.ones(shape, dtype=np.int8),
        "material_class": np.ones(shape, dtype=np.int8),
        "burnable_mask": np.ones(shape, dtype=bool),
        "nodata_mask": np.zeros(shape, dtype=bool),
        "grid_shape": shape,
        "transform": Affine(3, 0, 600000, 0, -3, 1140000),
        "crs": CRS.from_epsg(32651),
    }
    config = {
        "simulation": {"seed": seed, "emergency_checkpoint_enabled": False},
        "wind": {**phase17._WIND_CONFIG, "speed_kmh": 10.6, "direction_deg": 60.826537},
        "placeholder_transition": {"T_3_to_4": 1, "T_4_to_5": 1,
                                   "base_ignition_prob": 0.12, "slope_weight": 0.20,
                                   "building_weight": 0.10, "proximity_weight": 0.20,
                                   "material_weight": 0.35, "wind_weight": 0.20},
        "ml_model": {"inference_mode": "stochastic_probability", "positive_label": 1,
                     "threshold": None, "inference_threshold": None,
                     "classification_threshold": None},
        "flammability_weights": {"base_weight": 0.5, "slope_weight": 0.3, "proximity_weight": 0.2},
        "output": {},
    }
    return FireAutomata(environment, config, model_free=True)


class TestPhase17RuleAndGuards(unittest.TestCase):
    def test_d014_rule_coefficients_multiplication_and_seeded_rng(self):
        first, second = _fire_engine(42), _fire_engine(42)
        self.assertTrue(first.model_free)
        self.assertIsNone(first.model)
        self.assertEqual(type(first.rng.bit_generator).__name__, "PCG64")
        self.assertAlmostEqual(float(first.p_base[0, 0]), 0.6125, places=6)
        first_score = np.full(first.grid_shape, 0.25, dtype=np.float32)
        indices = np.array([0, 1], dtype=np.int64)
        np.testing.assert_allclose(first._model_free_probability(indices, first_score), [0.153125] * 2)
        first.set_ignition([(2, 2)])
        second.set_ignition([(2, 2)])
        first.step()
        second.step()
        np.testing.assert_array_equal(first.get_grid(), second.get_grid())
        np.testing.assert_array_equal(first.rng.random(5), second.rng.random(5))

    def test_model_attachment_loading_inference_and_checkpoint_guards(self):
        engine = _fire_engine()
        with self.assertRaises(RuntimeError):
            engine.model = object()
        with self.assertRaises(RuntimeError):
            engine.load_model("missing-estimator.joblib")
        with self.assertRaises(RuntimeError):
            engine.load_checkpoint("missing-checkpoint.npz")
        with self.assertRaises(RuntimeError):
            engine.save_checkpoint("forbidden.npz")
        with self.assertRaises(RuntimeError):
            engine._predict_with_model(None, None)
        engine._model = object()
        with self.assertRaises(RuntimeError):
            engine.step()

    def test_default_manifest_is_disabled_and_references_are_not_engine_inputs(self):
        manifest, _, _ = phase17.load_manifest()
        self.assertFalse(manifest["phase17"]["enabled"])
        self.assertFalse(manifest["phase17"]["execution_authorized"])
        fire_config = phase17._fire_config(manifest, "airbase", 42)
        self.assertNotIn("reference", fire_config)
        self.assertEqual(fire_config["placeholder_transition"]["wind_weight"], 0.2)

    def test_manifest_rejects_scoring_contract_drift(self):
        manifest, _, _ = phase17.load_manifest()
        manifest["scoring"]["contrasts"] = ["ca_minus_rf", "ca_minus_lr"]
        with self.assertRaises(ValueError):
            phase17.validate_manifest(manifest)


class TestPhase17CaptureSemantics(unittest.TestCase):
    def _run(self, automata, horizons):
        captured = []
        snapshots = orchestrator._Phase16Snapshots(_basis(horizons))

        def capture(row, state, reached):
            captured.append((row["requested_update"], automata.timestep,
                             state.copy(), reached.copy()))
            return [{"filename": f"h{row['requested_update']}_{kind}.tif"}
                    for kind in ("state", "reached")]

        try:
            result = snapshots.run(automata, lambda *_args, **_kwargs: None, capture)
        except RuntimeError:
            return snapshots, captured, None
        return snapshots, captured, result

    def test_capture_is_after_completed_update_and_inactive_carries_forward(self):
        automata = _ClockedAutomata(stop=True)
        snapshots, captured, result = self._run(automata, [1, 2, 3])
        self.assertEqual(result[0], "inactive")
        self.assertEqual([(horizon, update) for horizon, update, *_ in captured],
                         [(1, 1), (2, 1), (3, 1)])
        self.assertTrue(all(row["status"] == "complete" for row in snapshots.rows))
        self.assertEqual([row["absorbing_carry_forward"] for row in snapshots.rows],
                         [False, True, True])
        self.assertTrue(all(np.count_nonzero(np.isin(state, (3, 4))) == 0
                            for _, _, state, _ in captured))

    def test_active_horizons_are_censored(self):
        automata = _ClockedAutomata(stop=False)
        snapshots, captured, result = self._run(automata, [1, 2])
        self.assertEqual(result[0], "timestep_limit")
        self.assertEqual([(horizon, update) for horizon, update, *_ in captured], [(1, 1), (2, 2)])
        self.assertTrue(all(row["status"] == "censored" and row["censoring"]
                            for row in snapshots.rows))

    def test_trajectory_failure_invalidates_every_horizon(self):
        snapshots, _, result = self._run(_ClockedAutomata(fail=True), [1, 2, 3])
        self.assertIsNone(result)
        self.assertTrue(all(row["status"] == "incomplete" and row["output_valid"] is False
                            for row in snapshots.rows))

    def test_resource_failure_after_capture_invalidates_every_horizon(self):
        automata = _ClockedAutomata(stop=True)
        snapshots = orchestrator._Phase16Snapshots(_basis([1, 2, 3]))
        captures = []

        def monitor(boundary, force=True):
            if boundary == "after_snapshot":
                raise RuntimeError("synthetic resource limit")

        def capture(row, state, reached):
            captures.append(row["requested_update"])
            return []

        with self.assertRaisesRegex(RuntimeError, "resource limit"):
            snapshots.run(automata, monitor, capture)
        self.assertTrue(captures)
        self.assertTrue(all(row["status"] == "incomplete" and row["output_valid"] is False
                            for row in snapshots.rows))


class TestPhase17Contracts(unittest.TestCase):
    def test_exclusive_destination_collision_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "one-run"
            phase17._exclusive_destination(destination, allowed_root=Path(directory))
            with self.assertRaises(FileExistsError):
                phase17._exclusive_destination(destination, allowed_root=Path(directory))

    def test_digest_and_binding_mismatches_fail_closed(self):
        with self.assertRaises(ValueError):
            phase17._digest("not-a-digest", "synthetic")
        with self.assertRaises(ValueError):
            phase17._same({"binding_sha256": "a" * 64}, {"binding_sha256": "b" * 64},
                          "binding mismatch")
        with self.assertRaises(ValueError):
            phase17._same("a" * 64, "b" * 64, "digest mismatch")
        with self.assertRaises(ValueError):
            phase17._check_binding_sha("a" * 64, "b" * 64)

    def test_common_source_and_runtime_mismatches_fail_before_input_preparation(self):
        benchmark = {"source_pins": {"shared.py": "a" * 64}, "runtime": {"python": "pinned"}}
        source = {"repository_root": str(Path.cwd()), "files": {"shared.py": "b" * 64}}
        with self.assertRaisesRegex(ValueError, "source pin mismatch"):
            phase17._check_route_a_common_identity(
                benchmark, source, {"python": "pinned"})
        source["files"]["shared.py"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "runtime mismatch for python"):
            phase17._check_route_a_common_identity(
                benchmark, source, {"python": "observed"})

    def test_equal_counts_at_different_cells_fail_exact_carry_forward(self):
        captured = np.array([[1, 2], [3, 4]], dtype=np.int8)
        terminal = np.array([[2, 1], [3, 4]], dtype=np.int8)
        self.assertEqual(np.unique(captured, return_counts=True)[1].tolist(),
                         np.unique(terminal, return_counts=True)[1].tolist())
        with self.assertRaisesRegex(ValueError, "carry-forward"):
            phase17._assert_exact_carry_forward(captured, terminal)

    def test_runner_integration_reaches_trajectory_with_synthetic_inputs_and_publishers(self):
        class SyntheticMonitor:
            def __init__(self, configuration, fail_boundary=None):
                self.configuration = configuration
                self.fail_boundary = fail_boundary

            def check(self, boundary, force=True):
                if boundary == self.fail_boundary:
                    raise RuntimeError("synthetic resource limit")

        class SyntheticAutomata:
            def __init__(self, environment, config, model_free=False):
                self.grid = np.full((5, 5), 2, dtype=np.int8)
                self.grid[2, 2] = 4
                self.timestep = 0
                self.model = None
                self.model_free = model_free
                self.simulation_valid_mask = environment["burnable_mask"] & ~environment["nodata_mask"]
                self.burnable_mask = self.simulation_valid_mask & environment["building_presence"].astype(bool)
                self.grid[~self.burnable_mask] = 1
                self.grid[2, 2] = 4
                self.rng = np.random.default_rng(42)
                self.t_3_to_4, self.t_4_to_5 = 1, 1

            def set_ignition(self, _points):
                return None

            def get_grid(self):
                return self.grid.copy()

            def step(self):
                self.timestep += 1
                self.grid[2, 2] = 5

        config, _, _ = phase17.load_manifest()
        config["phase17"]["enabled"] = True
        config["phase17"]["execution_authorized"] = True
        config["phase17"]["resource_monitor"] = {
            "schema_version": "phase17_resource_monitor.v1", "authorization_reference": "synthetic-card",
            "limits": {"max_wall_seconds": 1, "max_process_rss_bytes": 1,
                       "min_system_available_bytes": 1, "min_free_output_bytes": 1,
                       "sample_interval_seconds": 1}}
        environmental_valid = np.ones((5, 5), dtype=bool)
        environmental_valid[0, 0] = False
        building_presence = np.ones((5, 5), dtype=np.int8)
        building_presence[0, 1] = 0
        receiving = environmental_valid & building_presence.astype(bool)
        domain_sha = phase17.mask_sha256(receiving)
        benchmark = {"source_pins": {"synthetic.py": "a" * 64}, "runtime": {"python": "pinned"}}
        repo = Path.cwd()
        source_files = {
            Path(phase17.__file__).resolve().relative_to(repo).as_posix(): phase17._sha256_file(Path(phase17.__file__)),
            Path(phase17.DEFAULT_MANIFEST).resolve().relative_to(repo).as_posix(): "b" * 64,
        }
        source = {"repository_root": str(repo), "files": source_files, "head": "synthetic",
                  "dirty": False, "absent_paths": []}
        observed = {"mask_identities": {"receiving_sha256": domain_sha,
                                          "environment_sha256": phase17.mask_sha256(environmental_valid)},
                    "converted_ignitions": [[2, 2]]}
        environment = {"burnable_mask": environmental_valid, "nodata_mask": np.zeros((5, 5), dtype=bool),
                       "building_presence": building_presence, "grid_shape": environmental_valid.shape,
                       "transform": Affine(3, 0, 600000, 0, -3, 1140000),
                       "crs": CRS.from_epsg(32651)}
        policy = config["phase17"]["resource_monitor"]
        original_trajectory_contract = phase17._trajectory_contract
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "phase17-output"
            output_root.mkdir(parents=True)
            config["phase17"]["output_root"] = "output/phase17_dswd_ca_only_v1"
            records = {
                ("airbase", seed): {
                    "rf": {"run_id": f"rf-{seed}", "binding_sha256": "c" * 64},
                    "lr": {"run_id": f"lr-{seed}", "binding_sha256": "d" * 64},
                } for seed in (42, 43)
            }
            published = {}
            published_calls = []

            def trajectory_contract(_config, incident, seed):
                trajectory = original_trajectory_contract(_config, incident, seed)
                trajectory.update(shape=[5, 5], domain_mask_sha256=domain_sha,
                                  ignition_local_row_col=[[2, 2]])
                return trajectory

            def publish(_dest, name, array, *_args, **_kwargs):
                published[name] = np.array(array, copy=True)
                published_calls.append(name)
                return {"filename": name, "sha256": "e" * 64, "verified": True}

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(phase17, "load_manifest",
                                                      return_value=(config, b"synthetic", "b" * 64)))
                stack.enter_context(mock.patch.object(phase17, "_benchmark_identity",
                                                      return_value=(benchmark, "a" * 64)))
                stack.enter_context(mock.patch.object(phase17, "_route_a_binding_pairs",
                                                      side_effect=lambda _cfg, _benchmark: records))
                stack.enter_context(mock.patch.object(
                    phase17.orchestrator, "_phase15_source",
                    return_value={"repository_root": str(repo), "files": {"synthetic.py": "a" * 64}}))
                stack.enter_context(mock.patch.object(
                    phase17.orchestrator, "_phase15_runtime", return_value={"python": "pinned"}))
                stack.enter_context(mock.patch.object(
                    phase17.rasterio, "__gdal_version__",
                    config["candidate_inputs"]["reader_capability"]["runtime"]["gdal"]))
                stack.enter_context(mock.patch.object(phase17, "_phase17_source_identity", return_value=source))
                stack.enter_context(mock.patch.object(phase17, "_trajectory_contract", side_effect=trajectory_contract))
                stack.enter_context(mock.patch.object(phase17, "_code_path", return_value=output_root))
                stack.enter_context(mock.patch.object(phase17, "_exclusive_destination",
                                                      side_effect=lambda path, **_: Path(path).mkdir()))
                candidate_reader = stack.enter_context(mock.patch.object(
                    phase17.orchestrator, "_phase15_candidate_environment",
                    return_value=(environment, observed)))
                stack.enter_context(mock.patch.object(phase17.orchestrator, "_phase15_raster", side_effect=publish))
                stack.enter_context(mock.patch.object(phase17, "FireAutomata", SyntheticAutomata))
                stack.enter_context(mock.patch.object(phase17, "_read_snapshot",
                                                      side_effect=lambda path, **_kwargs: published[Path(path).name]))

                with self.assertRaisesRegex(ValueError, "configured resource monitor is required"):
                    phase17.run_trajectory(
                        phase17.DEFAULT_MANIFEST, incident="airbase", seed=42,
                        explicitly_authorized=True, authority_reference="synthetic-run",
                        expected_manifest_sha256="b" * 64, expected_binding_sha256="f" * 64)
                candidate_reader.assert_not_called()

                benchmark["source_pins"] = {"synthetic.py": "b" * 64}
                with self.assertRaisesRegex(ValueError, "source pin mismatch"):
                    phase17.run_trajectory(
                        phase17.DEFAULT_MANIFEST, incident="airbase", seed=42,
                        explicitly_authorized=True, authority_reference="synthetic-run",
                        expected_manifest_sha256="b" * 64, expected_binding_sha256="f" * 64,
                        resource_monitor=SyntheticMonitor(policy), resource_monitor_config=policy)
                candidate_reader.assert_not_called()
                benchmark["source_pins"] = {"synthetic.py": "a" * 64}
                benchmark["runtime"] = {"python": "expected"}
                with self.assertRaisesRegex(ValueError, "runtime mismatch"):
                    phase17.run_trajectory(
                        phase17.DEFAULT_MANIFEST, incident="airbase", seed=42,
                        explicitly_authorized=True, authority_reference="synthetic-run",
                        expected_manifest_sha256="b" * 64, expected_binding_sha256="f" * 64,
                        resource_monitor=SyntheticMonitor(policy), resource_monitor_config=policy)
                candidate_reader.assert_not_called()

                benchmark["runtime"] = {"python": "pinned"}
                binding = phase17.build_execution_binding(config, phase17.DEFAULT_MANIFEST, "b" * 64,
                                                          "airbase", 42, "synthetic-run", policy)
                result = phase17.run_trajectory(
                    phase17.DEFAULT_MANIFEST, incident="airbase", seed=42, explicitly_authorized=True,
                    authority_reference="synthetic-run", expected_manifest_sha256="b" * 64,
                    expected_binding_sha256=phase17.canonical_metadata_hash(binding),
                    resource_monitor=SyntheticMonitor(policy), resource_monitor_config=policy)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(len(result["snapshot_series"]["outcomes"]), 3)
                self.assertTrue(np.any(published["simulation_environment_valid.tif"] != receiving))
                validated = phase17.validate_trajectory_output(
                    Path(binding["destination"]), manifest_path=phase17.DEFAULT_MANIFEST,
                    explicitly_authorized=True, review_reference="synthetic-review",
                    expected_binding_sha256=result["binding_sha256"],
                    expected_start_sha256=phase17._sha256_file(Path(binding["destination"]) / "start.json"),
                    expected_termination_sha256=result["termination_record_sha256"])
                self.assertEqual(validated["snapshots"][172]["outcome"]["absorbing_carry_forward"], True)
                np.testing.assert_array_equal(validated["snapshots"][172]["state"], published["final_state.tif"])

                failed_binding = phase17.build_execution_binding(config, phase17.DEFAULT_MANIFEST, "b" * 64,
                                                                  "airbase", 43, "synthetic-run-failure", policy)
                calls_before_failure = len(published_calls)
                with self.assertRaisesRegex(RuntimeError, "resource limit"):
                    phase17.run_trajectory(
                        phase17.DEFAULT_MANIFEST, incident="airbase", seed=43,
                        explicitly_authorized=True, authority_reference="synthetic-run-failure",
                        expected_manifest_sha256="b" * 64,
                        expected_binding_sha256=phase17.canonical_metadata_hash(failed_binding),
                        resource_monitor=SyntheticMonitor(policy, fail_boundary="after_snapshot"),
                        resource_monitor_config=policy)
                failure_record, _ = phase17._record(Path(failed_binding["destination"]) / "termination.json")
                self.assertGreaterEqual(len(published_calls) - calls_before_failure, 2)
                self.assertTrue(all(row["status"] == "incomplete" and not row["output_valid"]
                                    for row in failure_record["snapshot_series"]["outcomes"]))

    def test_signed_rf_and_lr_deltas_are_model_minus_ca(self):
        ca = {"f1_score": 0.4, "recall": 0.5, "precision": 0.3, "accuracy": 0.8,
              "jaccard_index": 0.25, "confusion_matrix": [[10, 2], [3, 5]]}
        rf = {"f1_score": 0.6, "recall": 0.7, "precision": 0.5, "accuracy": 0.9,
              "jaccard_index": 0.4, "confusion_matrix": [[8, 4], [2, 6]]}
        lr = {"f1_score": 0.2, "recall": 0.3, "precision": 0.15, "accuracy": 0.7,
              "jaccard_index": 0.1, "confusion_matrix": [[12, 0], [4, 4]]}
        rf_delta, lr_delta = phase17._signed_delta(rf, ca), phase17._signed_delta(lr, ca)
        self.assertAlmostEqual(rf_delta["f1_score"], 0.2)
        self.assertEqual(rf_delta["FP"], 2)
        self.assertAlmostEqual(lr_delta["f1_score"], -0.2)
        self.assertEqual(lr_delta["FP"], -2)

    def test_scoring_rejects_an_empty_validated_run_set(self):
        with self.assertRaises(ValueError):
            phase17.score_validated_outputs([], {}, explicitly_authorized=True,
                                            review_reference="synthetic-review")

    def test_only_exact_five_distinct_valid_seeds_are_summarized(self):
        rows = [{"incident": "airbase", "requested_update": 86, "seed": seed, "valid": True,
                 "value": {"f1_score": seed / 100}} for seed in (42, 43, 44, 45, 46)]
        summary = phase17._summarize_exact_five(
            rows, grouping=("incident", "requested_update"), value_key="value")
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["n"], 5)
        duplicate = rows[:-1] + [{**rows[0], "seed": 45}]
        self.assertEqual(phase17._summarize_exact_five(
            duplicate, grouping=("incident", "requested_update"), value_key="value"), [])
        failed = [{**row, "valid": False} if row["seed"] == 46 else row for row in rows]
        self.assertEqual(phase17._summarize_exact_five(
            failed, grouping=("incident", "requested_update"), value_key="value"), [])


if __name__ == "__main__":
    unittest.main()
