"""Candidate-only embedded-TIFF interpretation, common masks and atomic seeds.

No historical loader is changed. Calling read_inputs requires separately
authorized artifact access; importing this module reads no candidate artifacts.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import warnings

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.env import get_gdal_config
from rasterio.transform import Affine, rowcol

from .transition_observer import mask_sha256

POLICY_ID = "embedded_tiff_only.v1"
MASK_POLICY_ID = "candidate_common_environment_receiving_atomic_seeds.v1"
LAYERS = ("slope", "proximity", "buildings", "materials")
ENV_OPTIONS = {"GDAL_PAM_ENABLED": "NO", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
OPEN_OPTIONS = {"GEOREF_SOURCES": "INTERNAL"}


def require_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty exact text identity is required")
    if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise ValueError("Unpaired surrogate in identity")
    return value


def require_digest(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("A lowercase SHA-256 identity is required")
    return value


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _geometry(shape, crs, transform):
    if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("Positive two-dimensional grid shape required")
    crs = CRS.from_user_input(crs)
    transform = Affine(*tuple(transform)[:6])
    if (not crs.is_projected or crs.linear_units_factor[1] != 1.0
            or tuple(transform)[:6] != (3.0, 0.0, transform.c, 0.0, -3.0, transform.f)
            or not np.all(np.isfinite(tuple(transform)))):
        raise ValueError("Expected finite north-up projected 3 m grid")
    return crs, transform


@dataclass(frozen=True)
class PreparedInputs:
    environment: dict
    receiving_mask: np.ndarray
    global_origin: tuple
    full_grid_shape: tuple
    full_grid_transform: tuple
    input_hashes: dict
    interpretation: dict

    def mask_identities(self):
        return {
            "environment_sha256": mask_sha256(self.environment["burnable_mask"]),
            "building_sha256": mask_sha256(self.environment["building_presence"] > 0),
            "receiving_sha256": mask_sha256(self.receiving_mask),
        }


def prepare_arrays(raw, native_masks, declared_nodata, *, configured_nodata,
                   crs, transform, input_hashes, interpretation,
                   global_origin=(0, 0), full_grid_shape=None):
    """Validate raw categorical precision before narrowing; never infer ocean."""
    if any(set(items) != set(LAYERS) for items in (raw, native_masks, declared_nodata, input_hashes)):
        raise ValueError("Exactly four identified layers required")
    for digest in input_hashes.values():
        require_digest(digest)
    shape = raw["slope"].shape
    crs, transform = _geometry(shape, crs, transform)
    if (len(global_origin) != 2 or
            any(type(i) is not int or i < 0 for i in global_origin)):
        raise ValueError("Global origin must contain nonnegative integers")
    full_grid_shape = shape if full_grid_shape is None else tuple(full_grid_shape)
    if (len(full_grid_shape) != 2 or
            any(type(n) is not int or n <= 0 for n in full_grid_shape) or
            any(o + n > f for o, n, f in zip(global_origin, shape, full_grid_shape))):
        raise ValueError("Local extent is outside the declared global grid")
    if configured_nodata is not None and (
            type(configured_nodata) not in (int, float) or not np.isfinite(configured_nodata)):
        raise ValueError("Configured sentinel must be finite numeric or explicit null")
    validity = {}
    for name in LAYERS:
        values, mask = raw[name], native_masks[name]
        if (not isinstance(values, np.ndarray) or values.shape != shape or
                values.dtype.kind not in "iuf" or
                not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.shape != shape):
            raise ValueError("Native numeric arrays and aligned uint8 masks required")
        nodata = declared_nodata[name]
        if nodata is not None and (isinstance(nodata, (str, bool)) or not np.isscalar(nodata)):
            raise ValueError("Unsupported embedded nodata")
        valid = (mask != 0) & np.isfinite(values)
        for sentinel in (nodata, configured_nodata, -9999):
            if sentinel is not None:
                if not isinstance(sentinel, (int, float, np.integer, np.floating)):
                    raise ValueError("Unsupported nodata value")
                if np.isfinite(sentinel):
                    valid &= values != sentinel
        validity[name] = valid
    if not np.all(np.isin(raw["buildings"][validity["buildings"]], (0, 10))):
        raise ValueError("Unsupported raw building code before conversion")
    if not np.all(np.isin(raw["materials"][validity["materials"]], range(6))):
        raise ValueError("Unsupported raw material code before conversion")
    building = validity["buildings"] & (raw["buildings"] == 10)
    material = np.where(validity["materials"], raw["materials"], 0).astype(np.int8)
    if not np.array_equal(building, material > 0):
        raise ValueError("Full-grid building/material disagreement")
    environmental = np.logical_and.reduce([validity[name] for name in LAYERS])
    with np.errstate(over="raise", invalid="raise"):
        slope = np.where(environmental, raw["slope"], 0).astype(np.float32)
        slope[(slope < 0) | (slope > 10)] = 0
        slope = np.divide(slope, np.float32(10), dtype=np.float32)
        proximity = np.divide(np.where(environmental, raw["proximity"], 0).astype(np.float32),
                              np.float32(10), dtype=np.float32)
    if not np.all(np.isfinite(slope)) or not np.all(np.isfinite(proximity)):
        raise ValueError("Nonfinite canonical predictors")
    environment = dict(slope_risk=_readonly(slope), proximity_risk=_readonly(proximity),
        building_presence=_readonly(building.astype(np.int8)), material_class=_readonly(material),
        burnable_mask=_readonly(environmental), nodata_mask=_readonly(~environmental),
        grid_shape=shape, crs=crs, transform=transform)
    full_transform = transform * Affine.translation(-global_origin[1], -global_origin[0])
    return PreparedInputs(environment, _readonly(environmental & building & (material > 0)),
        tuple(global_origin), full_grid_shape, tuple(full_transform), dict(input_hashes),
        dict(interpretation, configured_nodata=configured_nodata, mask_seed_policy=MASK_POLICY_ID))


def validate_seeds(points, prepared, coordinate_system):
    """Resolve the complete list without mutating states, timers or the RNG."""
    if not isinstance(points, (list, tuple)) or not points:
        raise ValueError("A nonempty prescribed seed list is required")
    resolved = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each seed must have two coordinates")
        if coordinate_system == "grid_row_col":
            if any(type(n) is not int or n < 0 for n in point):
                raise ValueError("Global row/column seeds must be nonnegative integers")
            row, col = (point[i] - prepared.global_origin[i] for i in range(2))
        elif coordinate_system == "projected_xy":
            if any(type(n) not in (int, float) or not np.isfinite(n) for n in point):
                raise ValueError("Projected coordinates must be finite numbers")
            transform = prepared.environment["transform"]
            row, col = map(int, rowcol(transform, *point, op=np.floor))
            ca, ra = ~transform * tuple(point)
            if (row, col) != (int(np.floor(ra)), int(np.floor(ca))):
                raise ValueError("Coordinate conversion disagreement")
        else:
            raise ValueError("Unsupported seed coordinate convention")
        rows, cols = prepared.receiving_mask.shape
        if not (0 <= row < rows and 0 <= col < cols) or not prepared.receiving_mask[row, col]:
            raise ValueError("Seed is outside the receiving population")
        if (row, col) in resolved:
            raise ValueError("Duplicate resolved seed")
        resolved.append((row, col))
    return tuple(resolved)


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read_inputs(paths, pins, *, expected_grid, configured_nodata, capability):
    """Full native-resolution reads only; no companion inventory or fallback.

    capability is a later independently reviewed installed-reader verification
    record, not execution authority. Echoed open options alone are not proof.
    """
    if set(paths) != set(LAYERS) or set(pins) != set(LAYERS):
        raise ValueError("Exactly four TIFF paths/pins required")
    runtime = {"rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__}
    if (not isinstance(capability, dict) or capability.get("policy_id") != POLICY_ID
            or capability.get("runtime") != runtime or capability.get("controls_verified") is not True
            or capability.get("environment_options") != ENV_OPTIONS
            or capability.get("open_options") != OPEN_OPTIONS):
        raise PermissionError("Installed-reader interpretation verification is unresolved")
    require_text(capability.get("verification_reference"))
    if any(str(path).replace("/", "\\").startswith("\\\\") for path in paths.values()):
        raise ValueError("UNC/network TIFF paths are not permitted")
    raw, masks, nodata, metadata = {}, {}, {}, {}
    normalized = {}
    for name in LAYERS:
        path = Path(paths[name])
        if (not path.is_absolute() or path.suffix.lower() not in (".tif", ".tiff")
                or path.is_symlink() or path.is_junction() or not path.is_file()):
            raise ValueError("Explicit ordinary local TIFF required")
        require_digest(pins[name])
        if _file_hash(path) != pins[name]:
            raise ValueError("TIFF identity mismatch")
        normalized[name] = path
    with rasterio.Env(**ENV_OPTIONS), warnings.catch_warnings():
        warnings.simplefilter("error")
        for name, path in normalized.items():
            if any(get_gdal_config(k, normalize=False) != v for k, v in ENV_OPTIONS.items()):
                raise RuntimeError("Restricted GDAL controls not effective")
            with rasterio.open(str(path), mode="r", driver="GTiff", sharing=False,
                               **OPEN_OPTIONS) as dataset:
                if (dataset.driver != "GTiff" or dataset.mode != "r" or dataset.count != 1
                        or dataset.subdatasets or dataset.options.get("GEOREF_SOURCES") != "INTERNAL"):
                    raise ValueError("Unsupported or ignored TIFF opening contract")
                if [Path(p).absolute() for p in dataset.files] != [path.absolute()]:
                    raise ValueError("Associated files contradict embedded-only interpretation")
                flags = {f.name for f in dataset.mask_flag_enums[0]}
                if flags not in ({"all_valid"}, {"nodata"}, {"per_dataset"}):
                    raise ValueError("Unsupported band mask interpretation")
                if (tuple(dataset.shape) != tuple(expected_grid["shape"])
                        or dataset.crs != CRS.from_user_input(expected_grid["crs"])
                        or tuple(dataset.transform) != tuple(expected_grid["transform"])):
                    raise ValueError("Raster alignment or approved grid mismatch")
                _geometry(tuple(dataset.shape), dataset.crs, dataset.transform)
                raw[name] = dataset.read(1)
                masks[name] = dataset.read_masks(1)
                if "all_valid" in flags and not np.all(masks[name] != 0):
                    raise ValueError("Contradictory all-valid mask")
                nodata[name] = dataset.nodatavals[0]
                metadata[name] = dict(dtype=dataset.dtypes[0], mask_flags=sorted(flags),
                    nodata=(None if nodata[name] is None else
                            float(nodata[name]) if np.isfinite(nodata[name]) else str(nodata[name])))
            if any(get_gdal_config(k, normalize=False) != v for k, v in ENV_OPTIONS.items()):
                raise RuntimeError("GDAL controls changed during interpretation")
    for name, path in normalized.items():
        if _file_hash(path) != pins[name]:
            raise ValueError("TIFF identity changed during read")
    interpretation = dict(policy_id=POLICY_ID, runtime=runtime,
        environment_options=dict(ENV_OPTIONS), open_options=dict(OPEN_OPTIONS),
        capability_reference=capability["verification_reference"], capability=dict(capability), layers=metadata)
    return prepare_arrays(raw, masks, nodata, configured_nodata=configured_nodata,
        crs=expected_grid["crs"], transform=expected_grid["transform"],
        input_hashes=pins, interpretation=interpretation)
