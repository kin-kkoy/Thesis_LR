"""Synthetic crop contracts only; no real rasters, models or campaign."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
from rasterio.transform import Affine

import run_candidate_dataset as runner
from modules import candidate_dataset_collection as collection
from modules import candidate_dataset_crops as crops
from modules import candidate_dataset_inputs as inputs
from modules import candidate_dataset_release as release
from test_candidate_dataset_inputs_contract import arrays, reader
from test_candidate_dataset_collection_contract import (
    context, FullGridReferenceCandidate, TracedCandidateGenerator, assert_candidate_pair,
)
from test_candidate_dataset_runner_contract import (
    authority_fixture, _constructed_validation_batch, retirement_runner,
)


def crop_config(config):
    config = deepcopy(config)
    config["inventory"] = [dict(ordinal=i, scenario_family_id=name,
        coordinate_system="grid_row_col", ignition_points=[[r, c]],
        event_id="santa_maria_development" if i == 1 else "candidate_r2_synthetic_scenarios",
        complete_event=i == 1, prior_use=f"P15-CDV1-ODP-20261025-R1-F{i:02d}")
        for i, (name, r, c) in enumerate(crops.FAMILIES, 1)]
    config["complete_event_ids"] = ["santa_maria_development"]
    config["grid"] = dict(shape=list(crops.PARENT_SHAPE), crs="EPSG:32651",
                          transform=list(Affine(3, 0, 598558.7784, 0, -3, 1144374.7699)))
    config["spatial_design"] = crops.frozen_design()
    config["reader_capability"].update(windowed_reads_verified=True,
        window_verification_reference="explicit synthetic window-reader substitute")
    return config


def test_frozen_windows_preserve_26_ignitions_alignment_and_horizon():
    config = crop_config(context()[1])
    collection.validate_configuration(config, active=True)
    windows = config["spatial_design"]["family_windows"]
    margins = []
    for (name, r, c), window in zip(crops.FAMILIES, windows, strict=True):
        a, b = window["global_origin"]
        assert name == window["scenario_family_id"]
        assert (a, b) == (128 * ((r - 512) // 128), 128 * ((c - 512) // 128))
        assert a >= 0 and b >= 0 and a + 1024 <= 5489 and b + 1024 <= 6896
        margins.append(min(r - a, a + 1023 - r, c - b, b + 1023 - c))
    assert len(windows) == 26 and min(margins) == 385 > collection.MAX_UPDATES
    assert windows[0]["global_origin"] == [256, 2176]
    assert windows[19]["global_origin"] == [2048, 2304]
    assert len(collection.planned_runs(config["inventory"])) == 364
    assert all(len([p for p in collection.planned_runs(config["inventory"])
                    if p["family_ordinal"] == n]) == 14 for n in range(1, 27))
    assert collection.MAX_PACKAGE_ROWS == 2000000  # Ceiling, not a target.


@pytest.mark.parametrize("fault", ["origin", "shape", "rng", "seed", "family", "parent", "horizon", "bool"])
def test_crop_design_rejects_adaptive_or_stale_substitutions(fault):
    config = crop_config(context()[1])
    design = config["spatial_design"]
    if fault == "origin":
        design["family_windows"][0]["global_origin"][0] += 128
    elif fault == "shape":
        design["family_windows"][0]["shape"] = [800, 800]
    elif fault == "rng":
        design["rng_policy"] = "full-map replay"
    elif fault == "seed":
        config["inventory"][0]["ignition_points"][0][0] += 1
    elif fault == "family":
        config["inventory"][0]["scenario_family_id"] = "replacement"
    elif fault == "parent":
        config["grid"]["shape"] = [1024, 1024]
    elif fault == "horizon":
        config["maximum_completed_updates"] = 400
    else:
        design["family_windows"][0]["family_ordinal"] = True
    with pytest.raises(ValueError):
        collection.validate_configuration(config, active=True)


def test_native_window_reads_and_masks_share_geometry_and_normalization(reader):
    seen = []
    def change(dataset):
        original_read, original_masks = dataset.read, dataset.read_masks
        def read(band, **options):
            seen.append(options)
            w = options["window"]
            assert (w.row_off, w.col_off, w.height, w.width) == (1, 1, 2, 2)
            return original_read(band)[1:3, 1:3]
        def masks(band, **options):
            assert options == seen[-1]
            return original_masks(band)[1:3, 1:3]
        dataset.read, dataset.read_masks = read, masks
    reader.change = change
    reader.capability.update(windowed_reads_verified=True, window_verification_reference="synthetic")
    result = inputs.read_inputs(reader.paths, dict.fromkeys(inputs.LAYERS, "a" * 64),
        expected_grid=dict(shape=[3, 3], crs="EPSG:32651", transform=list(Affine(3, 0, 0, 0, -3, 9))),
        configured_nodata=None, capability=reader.capability, window=[1, 1, 2, 2])
    assert len(seen) == 4 and result.environment["grid_shape"] == (2, 2)
    assert result.global_origin == (1, 1) and result.full_grid_shape == (3, 3)
    assert result.environment["transform"] == Affine(3, 0, 3, 0, -3, 6)
    assert result.full_grid_transform == tuple(Affine(3, 0, 0, 0, -3, 9))
    assert inputs.validate_seeds([[1, 1]], result, "grid_row_col") == ((0, 0),)
    assert inputs.validate_seeds([[4.5, 4.5]], result, "projected_xy") == ((0, 0),)


@pytest.mark.parametrize("window", [[-1, 0, 2, 2], [0, 0, 4, 4], [True, 0, 2, 2],
                                   [0, 0, 0, 2], [0, 0, 2.0, 2], [0, 0]])
def test_invalid_window_rejects_before_file_access(reader, monkeypatch, window):
    monkeypatch.setattr(inputs, "_file_hash", lambda *_: pytest.fail("Unapproved artifact access"))
    with pytest.raises(ValueError):
        inputs.read_inputs(reader.paths, dict.fromkeys(inputs.LAYERS, "a" * 64),
            expected_grid=dict(shape=[3, 3]), configured_nodata=None,
            capability=reader.capability, window=window)


def test_full_reader_capability_does_not_silently_authorize_windows(reader, monkeypatch):
    monkeypatch.setattr(inputs, "_file_hash", lambda *_: pytest.fail("Unapproved artifact access"))
    with pytest.raises(PermissionError, match="Windowed-reader"):
        inputs.read_inputs(reader.paths, dict.fromkeys(inputs.LAYERS, "a" * 64),
            expected_grid=dict(shape=[3, 3]), configured_nodata=None,
            capability=reader.capability, window=[0, 0, 2, 2])


def test_installed_native_reader_window_matches_identified_synthetic_tiffs(tmp_path):
    """Exercise installed GDAL/Rasterio on four newly written tiny test-owned TIFFs."""
    import hashlib
    import rasterio
    shape, transform = (4, 5), Affine(3, 0, 600000, 0, -3, 1140000)
    raw = dict(slope=np.arange(20, dtype=np.float32).reshape(shape) - 2,
               proximity=np.arange(20, dtype=np.float32).reshape(shape) - 10,
               buildings=np.full(shape, 10, dtype=np.float32),
               materials=(np.arange(20, dtype=np.int16).reshape(shape) % 5 + 1))
    raw["slope"][2, 2] = -9999
    raw["buildings"][1, 1] = raw["materials"][1, 1] = 0
    paths, pins = {}, {}
    for name, array in raw.items():
        path = tmp_path / (name + ".tif")
        with rasterio.Env(**inputs.ENV_OPTIONS):
            with rasterio.open(path, "w", driver="GTiff", width=5, height=4, count=1,
                               crs="EPSG:32651", transform=transform, dtype=array.dtype,
                               nodata=-9999) as dataset:
                dataset.write(array, 1)
        paths[name], pins[name] = str(path), hashlib.sha256(path.read_bytes()).hexdigest()
    capability = dict(policy_id=inputs.POLICY_ID, controls_verified=True,
        runtime=dict(rasterio=rasterio.__version__, gdal=rasterio.__gdal_version__),
        environment_options=dict(inputs.ENV_OPTIONS), open_options=dict(inputs.OPEN_OPTIONS),
        verification_reference="synthetic native TIFF test", windowed_reads_verified=True,
        window_verification_reference="synthetic native window test")
    options = dict(expected_grid=dict(shape=list(shape), crs="EPSG:32651", transform=list(transform)),
                   configured_nodata=-9999, capability=capability)
    full = inputs.read_inputs(paths, pins, **options)
    cropped = inputs.read_inputs(paths, pins, window=[1, 1, 2, 3], **options)
    for name, value in full.environment.items():
        if isinstance(value, np.ndarray):
            assert value[1:3, 1:4].tobytes() == cropped.environment[name].tobytes()
    assert full.receiving_mask[1:3, 1:4].tobytes() == cropped.receiving_mask.tobytes()
    assert cropped.full_grid_transform == tuple(transform)
    assert cropped.environment["transform"] == transform * Affine.translation(1, 1)
    assert cropped.interpretation["window"] == [1, 1, 2, 3]
    assert len(list(tmp_path.iterdir())) == 4


def test_crop_normalization_matches_full_preparation_bits_including_invalid_cells():
    raw, masks = arrays(9)
    raw["materials"][:] = np.arange(9)[None, :] % 5 + 1
    raw["slope"][:] = np.arange(9)[:, None] - 1
    raw["proximity"][:] = np.arange(9)[None, :] - 3
    raw["slope"][3, 4] = -9999
    raw["materials"][4, 4] = raw["buildings"][4, 4] = 0
    transform = Affine(3, 0, 0, 0, -3, 27)
    options = dict(configured_nodata=-9999, crs="EPSG:32651",
                   input_hashes=dict.fromkeys(inputs.LAYERS, "a" * 64), interpretation={})
    full = inputs.prepare_arrays(raw, masks, dict.fromkeys(inputs.LAYERS), transform=transform, **options)
    cropped = inputs.prepare_arrays({k: v[2:7, 2:7] for k, v in raw.items()},
        {k: v[2:7, 2:7] for k, v in masks.items()}, dict.fromkeys(inputs.LAYERS),
        transform=transform * Affine.translation(2, 2), global_origin=(2, 2), full_grid_shape=(9, 9), **options)
    for name, value in full.environment.items():
        if isinstance(value, np.ndarray):
            assert value[2:7, 2:7].tobytes() == cropped.environment[name].tobytes()
    assert full.receiving_mask[2:7, 2:7].tobytes() == cropped.receiving_mask.tobytes()


def cropped_preparation(config, ordinal=1):
    window = config["spatial_design"]["family_windows"][ordinal - 1]
    raw, masks = arrays(crops.WINDOW_SIZE)
    raw["materials"][:] = np.arange(crops.WINDOW_SIZE)[None, :] % 5 + 1
    r, c = window["global_origin"]
    return inputs.prepare_arrays(raw, masks, dict.fromkeys(inputs.LAYERS),
        configured_nodata=config["configured_nodata"], crs=config["grid"]["crs"],
        transform=Affine(*config["grid"]["transform"][:6]) * Affine.translation(c, r),
        global_origin=(r, c), full_grid_shape=crops.PARENT_SHAPE,
        input_hashes={k: v["sha256"] for k, v in config["inputs"].items()},
        interpretation=dict(policy_id=inputs.POLICY_ID, capability=config["reader_capability"],
                            window=[r, c, 1024, 1024]))


def test_crop_engine_matches_independent_r2_reference_and_persists_global_rows(monkeypatch, tmp_path):
    _, base, _, source, _ = context()
    config = crop_config(base)
    source["execution_id"] = crops.CAMPAIGN_ID + "-synthetic-verification"
    prepared = cropped_preparation(config)
    spec = collection.planned_runs(config["inventory"])[12]
    binding = collection.execution_binding(config, prepared, spec, source)
    static = collection.build_candidate_context(prepared, config)
    assert collection.execution_binding(config, static.prepared, spec, source, static_context=static) == binding
    factory = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda seed:
                        TracedCandidateGenerator(factory(seed), force_zero=True))
    options = dict(execution_authorized=True, approval_reference=config["authorization_reference"],
                   expected_binding_sha256=release.digest_bytes(release.json_bytes(binding)))
    actual = collection.CandidateAutomata(static.prepared, config, spec, source,
                                          static_context=static, **options)
    reference = FullGridReferenceCandidate(prepared, config, spec, source, **options)
    initial = np.array(crops.FAMILIES[0][1:]) - prepared.global_origin
    store = runner.EvidenceStore(tmp_path / "crop-evidence")
    try:
        for step in range(1, 4):
            left, right = actual.step(), reference.step()
            assert_candidate_pair(actual, reference, left, right)
            rr, cc = np.nonzero(actual.grid >= 3)
            assert np.all(np.maximum(abs(rr - initial[0]), abs(cc - initial[1])) <= step)
            assert np.all(left.global_rows >= prepared.global_origin[0])
            assert np.all(left.global_cols >= prepared.global_origin[1])
            runner.persist_and_acknowledge(actual, store)
            reference.acknowledge_batches(reference.batches)
        assert actual.rng.calls == [(1024, 1024), (1024, 1024)]
        assert store.transitions[spec["run_id"]] == 3 and not actual.batches
    finally:
        store.close()
    assert len((tmp_path / "crop-evidence/transitions.jsonl").read_bytes().splitlines()) == 3
    with pytest.raises(ValueError, match="family crop"):
        collection.execution_binding(config, replace(prepared, global_origin=(0, 0)), spec, source)
    stale = deepcopy(source)
    stale["execution_id"] = "old-full-map-attempt"
    with pytest.raises(ValueError, match="new cropped campaign"):
        collection.execution_binding(config, prepared, spec, stale)


def test_geometric_overlap_does_not_claim_independent_families():
    design = crops.frozen_design()
    pairs = crops.overlapping_windows(design)
    assert pairs and all(0 < p["overlap_cells"] <= 1024 ** 2 for p in pairs)
    assert len({tuple(p["families"]) for p in pairs}) == len(pairs)


def test_shared_global_block_unions_crop_families_and_propagates_f01_exclusion():
    features = np.array([.1, .2, 1, .5, 1, 10, 0, 1, 1, .5, .125], dtype=np.float32)
    rows = [dict(scenario_family_id=crops.FAMILIES[i][0], grid_id="same-parent-grid",
        payload=dict(cell_row=129 + i, cell_col=257 + i), event_id="synthetic-shared-block",
        observation_id=f"{i + 1:064x}", csv=release.row_bytes(features, i).decode("ascii"),
        material_class=1) for i in range(2)]
    eligibility = {crops.FAMILIES[i][0]: dict(eligible=i != 0) for i in range(2)}
    components = release.components(rows, [], eligibility, lambda _: None)
    assert len(components) == 1 and len(components[0]["families"]) == 2
    assert components[0]["final_test_eligible"] is False


def test_cropped_f01_exclusion_and_campaign_identity_precede_any_inputs(tmp_path):
    *_, authority, _ = authority_fixture(tmp_path)
    config = crop_config(context()[1])
    authority["source"]["execution_id"] = crops.CAMPAIGN_ID + "-synthetic"
    plan = collection.planned_runs(config["inventory"])
    authority["run_bindings"] = {p["run_id"]: "a" * 64 for p in plan}
    authority["custody"]["families"] = {f["scenario_family_id"]: dict(eligible=f["ordinal"] != 1,
        prior_use=f["prior_use"], assessment_reference=f["prior_use"]) for f in config["inventory"]}
    def invoke():
        execution = release.digest_bytes(release.json_bytes(runner.execution_payload(config, authority)))
        publication = release.digest_bytes(release.json_bytes(runner.publication_payload(execution, authority)))
        return runner.validate_authority(config, authority, execution_authorized=True,
            publication_authorized=True, expected_execution_sha256=execution,
            expected_publication_sha256=publication)
    assert invoke()[0] == plan
    authority["custody"]["families"][crops.FAMILIES[0][0]]["eligible"] = True
    with pytest.raises(PermissionError, match="F01"):
        invoke()


def test_crop_state_conservation_uses_local_cells_in_real_chunked_loader(monkeypatch, tmp_path):
    _, base, _, source, _ = context()
    config = crop_config(base)
    source["execution_id"] = crops.CAMPAIGN_ID + "-synthetic-inactive"
    prepared = cropped_preparation(config)
    environment = dict(prepared.environment)
    building = np.zeros((1024, 1024), dtype=np.int8)
    r, c = np.array(crops.FAMILIES[0][1:]) - prepared.global_origin
    building[r, c] = 1
    material = building.copy()
    for a in (building, material):
        a.setflags(write=False)
    environment.update(building_presence=building, material_class=material)
    receiving = building.astype(bool)
    receiving.setflags(write=False)
    prepared = replace(prepared, environment=environment, receiving_mask=receiving)
    spec = collection.planned_runs(config["inventory"])[0]
    binding = collection.execution_binding(config, prepared, spec, source)
    digest = release.digest_bytes(release.json_bytes(binding))
    static = collection.build_candidate_context(prepared, config)
    factory = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda seed: SimpleNamespace(
        bit_generator=factory(seed).bit_generator,
        random=lambda *_: pytest.fail("Empty eligibility consumed a draw")))
    engine = collection.CandidateAutomata(static.prepared, config, spec, source,
        static_context=static, execution_authorized=True,
        approval_reference=config["authorization_reference"], expected_binding_sha256=digest)
    store = runner.EvidenceStore(tmp_path / "inactive-crop")
    store.append("runs", dict(kind="initialization", spec=spec, completed=deepcopy(engine.last_completed)))
    assert len(engine.step().labels) == 0 and not engine.is_active()
    runner.persist_and_acknowledge(engine, store)
    store.append("runs", dict(kind="inactive", spec=spec, completed=deepcopy(engine.last_completed)))
    store.close()
    monkeypatch.setattr(collection, "planned_runs", lambda _: [spec])
    rows, records = runner.load_completed_evidence(tmp_path / "inactive-crop", config,
        dict(run_bindings={spec["run_id"]: digest}), lambda _: None)
    assert rows == [] and len(records) == 2
    assert sum(records[0]["completed"]["state_counts"].values()) == 1024 ** 2


@pytest.mark.parametrize("fault", [None, "last_preflight", "execution_recheck"])
def test_runner_preflights_all_families_and_reuses_only_one_context(monkeypatch, tmp_path, fault):
    config = crop_config(context()[1])
    plan = collection.planned_runs(config["inventory"])[:15]
    events, refs, owners, checks = [], [], [], []
    class Context:
        def __init__(self, ordinal):
            self.ordinal = ordinal
            self.prepared = SimpleNamespace(family=ordinal)
    def family_context(config, family, check):
        assert all(ref() is None for ref in refs), "Previous family context remains owned"
        result = Context(family["ordinal"])
        refs.append(weakref.ref(result))
        events.append(("context", family["ordinal"]))
        return result
    def commitments(config, authority, family_plan, static):
        checks.append(static.ordinal)
        if (fault == "last_preflight" and len(checks) == 26
                or fault == "execution_recheck" and len(checks) == 27):
            raise PermissionError("synthetic stale crop commitment")
    class Guard:
        def __init__(self, *args):
            self.samples = []
        def check(self, boundary):
            pass
    class Store:
        rows = 0
        def __init__(self, *args, **kwargs):
            assert len(checks) == 26
            events.append(("store",))
            self.run_rows, self.transitions = {}, {}
        def append(self, *args):
            pass
        def close(self):
            pass
    class Engine:
        batches, row_count, timestep, incomplete_transition = (), 0, 0, None
        last_completed = {"synthetic": "already inactive lifecycle substitute"}
        def __init__(self, prepared, config, spec, source, *, static_context, **kwargs):
            assert len(checks) >= 27 and prepared is static_context.prepared
            assert prepared.family == spec["family_ordinal"]
            self.owner = static_context
            owners.append((spec["family_ordinal"], id(static_context)))
        def is_active(self):
            return False
        def validate_handoff(self, *, expected_context):
            assert self.owner is expected_context
        def latch_handoff_failure(self):
            pass
    authority = dict(source={}, run_bindings={p["run_id"]: "a" * 64 for p in plan},
        custody=dict(families={}), destinations=dict(evidence=str(tmp_path / "unused-evidence"),
                                                    package=str(tmp_path / "unused-package")))
    monkeypatch.setattr(runner, "validate_authority", lambda *a, **kw: (plan, {}))
    monkeypatch.setattr(runner, "verify_sources_runtime", lambda *a: None)
    monkeypatch.setattr(runner, "_family_context", family_context)
    monkeypatch.setattr(runner, "_check_family_bindings", commitments)
    monkeypatch.setattr(runner, "Resources", Guard)
    monkeypatch.setattr(runner, "EvidenceStore", Store)
    monkeypatch.setattr(collection, "CandidateAutomata", Engine)
    monkeypatch.setattr(runner, "load_completed_evidence", lambda *a: ([], []))
    monkeypatch.setattr(runner, "file_identity", lambda *a: dict(bytes=0, sha256="a" * 64))
    monkeypatch.setattr(release, "components", lambda *a: [])
    monkeypatch.setattr(release, "publish", lambda *a, **kw: {"committed": True})
    if fault:
        with pytest.raises(runner.CandidateFailure) as caught:
            runner.run(config, authority)
        assert "stale crop" in caught.value.report["original_failure"]["message"]
        assert not owners
        if fault == "last_preflight":
            assert ("store",) not in events
    else:
        assert runner.run(config, authority)["complete"]
        assert checks == list(range(1, 27)) + [1, 2]
        assert len(owners) == 15
        assert len({owner for ordinal, owner in owners if ordinal == 1}) == 1
        assert all(ref() is None for ref in refs)


def test_inventory_has_no_context_or_rng_and_retains_material_counts(monkeypatch):
    config = crop_config(context()[1])
    monkeypatch.setattr(collection, "build_candidate_context", lambda *a: pytest.fail("Context construction"))
    monkeypatch.setattr(np.random, "default_rng", lambda *a: pytest.fail("RNG construction"))
    def read(config, family):
        # Small array substitute; the frozen geometry is independently checked above.
        raw, masks = arrays(5)
        raw["materials"][:] = np.arange(5)[None, :] + 1
        window = config["spatial_design"]["family_windows"][family["ordinal"] - 1]
        prepared = inputs.prepare_arrays(raw, masks, dict.fromkeys(inputs.LAYERS),
            configured_nodata=None, crs="EPSG:32651", transform=Affine(3, 0, 0, 0, -3, 15),
            input_hashes=dict.fromkeys(inputs.LAYERS, "a" * 64), interpretation={})
        return prepared
    monkeypatch.setattr(runner, "_read_family_crop", read)
    monkeypatch.setattr(crops, "validate_preparation", lambda *a: None)
    monkeypatch.setattr(runner, "validate_seeds", lambda *a: ((2, 2),))
    report = runner.inspect_crop_inputs(config, lambda _: None)
    assert len(report["windows"]) == 26 and report["geometric_overlaps"]
    assert report["windows"][0]["receiving_material_counts"] == dict.fromkeys("12345", 5)
    assert report["windows"][0]["building_density"] == 1.0
    json.dumps(report, allow_nan=False)


def _crop_extent_fixture():
    """Real synthetic crop preparation; observations below are fabricated, not CA history."""
    _, base, _, source, _ = context()
    config = crop_config(base)
    source["execution_id"] = crops.CAMPAIGN_ID + "-synthetic-extent"
    prepared = cropped_preparation(config)
    spec = collection.planned_runs(config["inventory"])[0]
    return collection.execution_binding(config, prepared, spec, source), config


def _crop_extent_record(binding, config, coordinate, timestep):
    batch = _constructed_validation_batch(binding, config["inventory"][0], 1, timestep)
    envelope = json.loads(batch.envelope_bytes)
    row, col = coordinate
    payload = release.observation_payload(binding, envelope, row, col)
    identity = release.digest_bytes(release.json_bytes(payload))
    envelope["bindings"][0]["observation_id"] = identity
    rr, cc = np.array([row], dtype=np.int64), np.array([col], dtype=np.int64)
    ids = np.array([identity.encode("ascii")], dtype="S64")
    for array in (rr, cc, ids):
        array.setflags(write=False)
    batch = replace(batch, global_rows=rr, global_cols=cc, observation_ids=ids,
                    envelope_bytes=release.json_bytes(envelope))
    spec, captured = binding["spec"], envelope["bindings"][0]
    retained = dict(**captured, payload=payload,
        csv=release.row_bytes(batch.features[0], int(batch.labels[0])).decode("ascii"),
        order=[spec["family_ordinal"], spec["wind_ordinal"], spec["replicate_ordinal"], timestep, row, col],
        event_id=envelope["generic_provenance"]["event_id"], material_class=int(batch.features[0, 4]),
        grid_id=binding["source"]["grid_id"], scenario_family_id=spec["scenario_family_id"])
    assert release.digest_bytes(np.asarray(batch.features[0], dtype="<f4").tobytes()
                                + bytes([int(batch.labels[0])])) == captured["row_sha256"]
    return batch, dict(binding=binding, envelope=envelope, rows=[retained])


def _crop_extent_journals(tmp_path, binding, config, coordinate):
    spec = binding["spec"]
    first, _ = _crop_extent_record(binding, config, (256, 2176), 0)
    _, offending = _crop_extent_record(binding, config, coordinate, 1)
    store = runner.EvidenceStore(tmp_path / "extent-evidence")
    before = dict(non_burnable=0, not_yet_burning=1024 ** 2 - 1,
                  ignited=0, blazing=1, extinguished=0)
    try:
        store.append("runs", dict(kind="initialization", spec=spec,
            completed=dict(state_timestep=0, committed_rows=0, state_counts=before)))
        store.persist(binding, (first,))
        # Construct a test-owned retained record directly; persistence must reject it independently.
        store.append("transitions", offending)
        store.append("runs", dict(kind="inactive", spec=spec, completed=dict(
            state_timestep=2, committed_rows=2,
            state_counts=dict(before, not_yet_burning=1024 ** 2 - 3, blazing=0, extinguished=3))))
    finally:
        store.close()
    return store.root, {"run_bindings": {spec["run_id"]: release.digest_bytes(release.json_bytes(binding))}}


@pytest.mark.parametrize("coordinate,accepted", [((1280, 2176), False), ((256, 3200), False),
    ((1279, 3199), True), ((255, 2176), False), ((256, 2175), False)])
def test_crop_extent_persistence_is_atomic_and_never_acknowledges_rejection(tmp_path, coordinate, accepted):
    binding, config = _crop_extent_fixture()
    first, _ = _crop_extent_record(binding, config, (256, 2176), 0)
    batch, record = _crop_extent_record(binding, config, coordinate, 1)
    if coordinate[0] >= 256 and coordinate[1] >= 2176:
        assert release.reconcile_batch(batch, binding) == (record["envelope"], record["rows"])
    store = runner.EvidenceStore(tmp_path / "extent-evidence")
    acknowledged = []
    engine = SimpleNamespace(binding=binding, batches=(batch,))
    def acknowledge(snapshot):
        assert snapshot is engine.batches
        acknowledged.append(snapshot)
        engine.batches = ()
    engine.acknowledge_batches = acknowledge
    try:
        store.persist(binding, (first,))
        prior_bytes = {name: (store.root / name).read_bytes() for name in ("transitions.jsonl", "runs.jsonl")}
        prior_accounting = deepcopy((store.rows, store.run_rows, store.transitions, store.last_order))
        if accepted:
            runner.persist_and_acknowledge(engine, store)
            assert len(acknowledged) == 1 and not engine.batches
            assert store.rows == store.run_rows[binding["spec"]["run_id"]] == 2
            assert store.transitions[binding["spec"]["run_id"]] == 2
        else:
            with pytest.raises(ValueError, match="bound crop extent"):
                runner.persist_and_acknowledge(engine, store)
            assert not acknowledged and engine.batches[0] is batch
            assert (store.rows, store.run_rows, store.transitions, store.last_order) == prior_accounting
            assert {name: (store.root / name).read_bytes() for name in prior_bytes} == prior_bytes
    finally:
        store.close()


@pytest.mark.parametrize("coordinate,accepted", [((1280, 2176), False), ((256, 3200), False),
    ((1279, 3199), True), ((255, 2176), False), ((256, 2175), False)])
def test_crop_extent_retained_reconstruction_commits_only_valid_transitions(monkeypatch, tmp_path, coordinate, accepted):
    binding, config = _crop_extent_fixture()
    root, authority = _crop_extent_journals(tmp_path, binding, config, coordinate)
    prior_bytes = {name: (root / name).read_bytes() for name in ("transitions.jsonl", "runs.jsonl")}
    monkeypatch.setattr(collection, "planned_runs", lambda _: [binding["spec"]])
    if accepted:
        rows, records = runner.load_completed_evidence(root, config, authority, lambda _: None)
        assert len(rows) == 2 and len(records) == 2
        assert (rows[-1]["payload"]["cell_row"], rows[-1]["payload"]["cell_col"]) == coordinate
    else:
        with pytest.raises(ValueError, match="bound crop extent") as caught:
            runner.load_completed_evidence(root, config, authority, lambda _: None)
        trace = caught.value.__traceback__
        while trace is not None and trace.tb_frame.f_code is not runner.load_completed_evidence.__code__:
            trace = trace.tb_next
        assert trace is not None
        local, run_id = trace.tb_frame.f_locals, binding["spec"]["run_id"]
        assert len(local["rows"]) == local["row_counts"][run_id] == 1
        assert local["transitions"][run_id] == local["positive_counts"][run_id] == 1
        assert local["prior_states"][run_id] == f"{2:064x}"
    assert {name: (root / name).read_bytes() for name in prior_bytes} == prior_bytes


@pytest.mark.parametrize("route", ["persist", "retained"])
@pytest.mark.parametrize("fault", ["missing_window", "null_window", "bool_height", "origin", "parent", "execution", "ordinal"])
def test_identified_crop_requires_consistent_metadata_even_for_empty_batches(monkeypatch, tmp_path, route, fault):
    binding, config = _crop_extent_fixture()
    if fault == "missing_window":
        del binding["interpretation"]["window"]
    elif fault == "null_window":
        binding["interpretation"]["window"] = None
    elif fault == "bool_height":
        binding["interpretation"]["window"][2] = True
    elif fault == "origin":
        binding["global_origin"][0] += 1
    elif fault == "parent":
        binding["full_grid_shape"][0] += 1
    elif fault == "execution":
        binding["source"]["execution_id"] = "synthetic-legacy-id-with-window"
    else:
        binding["spec"]["family_ordinal"] = True
    empty = _constructed_validation_batch(binding, config["inventory"][0], 0)
    store = runner.EvidenceStore(tmp_path / "metadata-evidence")
    try:
        if route == "persist":
            with pytest.raises(ValueError, match="crop extent metadata"):
                store.persist(binding, (empty,))
            assert store.rows == 0 and store.run_rows == store.transitions == {} and store.last_order is None
            assert all((store.root / name).read_bytes() == b"" for name in ("transitions.jsonl", "runs.jsonl"))
        else:
            store.append("transitions", dict(binding=binding, envelope=json.loads(empty.envelope_bytes), rows=[]))
    finally:
        store.close()
    if route == "retained":
        authority = {"run_bindings": {binding["spec"]["run_id"]: release.digest_bytes(release.json_bytes(binding))}}
        retained = (store.root / "transitions.jsonl").read_bytes()
        with pytest.raises(ValueError, match="crop extent metadata"):
            runner.load_completed_evidence(store.root, config, authority, lambda _: None)
        assert (store.root / "transitions.jsonl").read_bytes() == retained


@pytest.mark.parametrize("route", ["persist", "retained"])
def test_legacy_full_map_binding_keeps_existing_parent_extent(tmp_path, route):
    prepared, config, spec, source, _ = context()
    binding = collection.execution_binding(config, prepared, spec, source)
    batch, record = _crop_extent_record(binding, config, (2, 2), 0)
    if route == "persist":
        store = runner.EvidenceStore(tmp_path / "legacy-evidence")
        try:
            store.persist(binding, (batch,))
            assert store.rows == 1 and store.transitions[spec["run_id"]] == 1
        finally:
            store.close()
    else:
        assert runner._reconcile_retained_chunk(record["rows"], record["envelope"], binding, 0, 1) == 1


def test_crop_extent_reconstruction_failure_blocks_runner_grouping_and_publication(tmp_path, monkeypatch, request):
    binding, config = _crop_extent_fixture()
    root, authority = _crop_extent_journals(tmp_path, binding, config, (1280, 2176))
    retained = {name: (root / name).read_bytes() for name in ("transitions.jsonl", "runs.jsonl")}
    real_load = runner.load_completed_evidence
    lifecycle = request.getfixturevalue("retirement_runner")
    with monkeypatch.context() as patch:
        patch.setattr(runner, "load_completed_evidence", lambda *a: real_load(root, config, authority, lambda _: None))
        patch.setattr(release, "components", lambda *a: pytest.fail("Grouped after crop extent rejection"))
        patch.setattr(release, "publish", lambda *a, **kw: pytest.fail("Published after crop extent rejection"))
        with pytest.raises(runner.CandidateFailure) as caught:
            runner.run(lifecycle.config, lifecycle.authority, emit=lifecycle.emit, **lifecycle.kwargs)
        assert caught.value.report["complete"] is False and caught.value.report["committed"] is None
        assert caught.value.report["original_failure"]["type"] == "ValueError"
        assert "bound crop extent" in caught.value.report["original_failure"]["message"]
        assert isinstance(caught.value.__cause__, ValueError)
        assert {name: (root / name).read_bytes() for name in retained} == retained
        assert lifecycle.authority["destinations"]["package"] == str(tmp_path / "package")
        assert not (tmp_path / "package").exists()
