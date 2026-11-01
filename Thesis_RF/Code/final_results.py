"""Read-only viewer for the accepted F01 development footprint; no figure export."""
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import sys
import warnings

CODE_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_PATH = CODE_DIR / "processedData/raster/raster/stack_ground_truth.tif"
FINAL_STATE_PATH = CODE_DIR / "output/phase15/p15-candidate-r2-f01-s42-r1/final_state.tif"
BUILDINGS_PATH = CODE_DIR / "processedData/raster/raster/stack_buildings.tif"
PROJECT_PYTHON = CODE_DIR.parents[1] / ".venv/Scripts/python.exe"


def require_project_environment() -> None:
    if (Path(sys.executable).resolve() != PROJECT_PYTHON.resolve()
            or sys.version_info[:3] != (3, 14, 3)):
        raise RuntimeError(f"Use the reviewed project environment: {PROJECT_PYTHON} final_results.py")
    for package, expected in (("numpy", "2.4.4"), ("rasterio", "1.5.0"),
                              ("scikit-learn", "1.8.0"), ("matplotlib", "3.10.9")):
        if version(package) != expected:
            raise RuntimeError(f"{package} must be {expected}; no environment changes were made")


if __name__ == "__main__":
    require_project_environment()

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.env import get_gdal_config
from rasterio.transform import Affine
from rasterio.windows import Window
from validation_engine import evaluate_final_footprint_arrays

# Paths may point to identical copies. Different cases require an explicit reviewed binding.
INPUT_SHA256 = {
    "reference": "fcf83513ab223072299e33696cd4ec2265d83ee330e1957ba1eb42feeb3a3880",
    "prediction": "a1d51034bde2dfeadd5caedcab6dfe28c9587540a98c6ae128b9ecdc65f1282a",
    "buildings": "da53c3e63b661471b3366fc68249793c6a5a84e0d63bcc96492f60baaf4b83c7",
    "environment": "6537f9bd15dc63acfd743027d706cc70d69e759c44207924ca5323d75ebcddff",
}
PARENT_SHAPE = (5489, 6896)
CROP_SHAPE = (1024, 1024)
PARENT_TRANSFORM = Affine(3, 0, 598558.7784, 0, -3, 1144374.7699)
CROP_TRANSFORM = Affine(3, 0, 605086.7784, 0, -3, 1143606.7699)
CROP_WINDOW = Window(2176, 256, 1024, 1024)
CRS_CODE = "EPSG:32651"
ENV_OPTIONS = {"GDAL_PAM_ENABLED": "NO", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_geometry(shape, crs, transform, *, parent: bool) -> None:
    if (tuple(shape) != (PARENT_SHAPE if parent else CROP_SHAPE)
            or crs != CRS.from_user_input(CRS_CODE)
            or transform != (PARENT_TRANSFORM if parent else CROP_TRANSFORM)):
        raise ValueError("Geometry differs from the bound F01 grid; no resampling is permitted")


def check_controls() -> None:
    if any(get_gdal_config(key, normalize=False) != value for key, value in ENV_OPTIONS.items()):
        raise RuntimeError("Restricted embedded-TIFF controls are not effective")


def load_inputs() -> tuple[dict, dict]:
    if rasterio.__gdal_version__ != "3.12.1":
        raise RuntimeError("Reviewed GDAL 3.12.1 required; no reader fallback is permitted")
    paths = dict(reference=Path(GROUND_TRUTH_PATH), prediction=Path(FINAL_STATE_PATH),
                 buildings=Path(BUILDINGS_PATH),
                 environment=Path(FINAL_STATE_PATH).parent / "simulation_environment_valid.tif")
    identities = {}
    for name, path in paths.items():
        if path.is_symlink() or path.is_junction() or not path.is_file():
            raise ValueError(f"Ordinary TIFF required: {path}")
        observed = file_sha256(path)
        if observed != INPUT_SHA256[name]:
            raise ValueError(f"F01 identity mismatch for {name}: {observed}")
        identities[name] = dict(path=str(path.resolve()), sha256=observed)
    arrays = {}
    with rasterio.Env(**ENV_OPTIONS), warnings.catch_warnings():
        warnings.simplefilter("error")
        for name, path in paths.items():
            check_controls()
            with rasterio.open(path, mode="r", driver="GTiff", sharing=False,
                               GEOREF_SOURCES="INTERNAL") as src:
                if (src.driver != "GTiff" or src.mode != "r" or src.count != 1 or src.subdatasets
                        or src.options.get("GEOREF_SOURCES") != "INTERNAL"
                        or [Path(p).absolute() for p in src.files] != [path.absolute()]):
                    raise ValueError("Unsupported layout or associated-file interpretation")
                flags = {flag.name for flag in src.mask_flag_enums[0]}
                if flags not in ({"all_valid"}, {"nodata"}, {"per_dataset"}):
                    raise ValueError("Unsupported mask flags")
                parent = name in ("reference", "buildings")
                validate_geometry(src.shape, src.crs, src.transform, parent=parent)
                options = {"window": CROP_WINDOW} if parent else {}
                if parent and src.window_transform(CROP_WINDOW) != CROP_TRANSFORM:
                    raise ValueError("Window does not align with prediction")
                raw = src.read(1, **options)
                native = src.read_masks(1, **options)
                if raw.shape != CROP_SHAPE or native.shape != CROP_SHAPE:
                    raise ValueError("Unexpected read shape")
                if raw.dtype.kind not in "biuf":
                    raise ValueError("Unsupported categorical dtype")
                if flags == {"all_valid"} and not np.all(native != 0):
                    raise ValueError("Contradictory all-valid mask")
                valid = (native != 0) & np.isfinite(raw) & (raw != -9999)
                if src.nodata is not None:
                    valid &= raw != src.nodata
                arrays[name] = raw
                arrays[name + "_valid"] = valid
                identities[name].update(dtype=str(raw.dtype), mask_flags=sorted(flags), nodata=src.nodata)
            check_controls()
    for name, path in paths.items():
        if file_sha256(path) != identities[name]["sha256"]:
            raise ValueError(f"Input changed during reading: {name}")
    return arrays, identities


def evaluate_arrays(arrays: dict) -> dict:
    state, reference, buildings, environment = (arrays[name] for name in
                                               ("prediction", "reference", "buildings", "environment"))
    shape = state.shape
    if state.ndim != 2 or any(np.shape(value) != shape for value in arrays.values()):
        raise ValueError("All arrays and validity masks must share a two-dimensional shape")
    for name in ("prediction", "reference", "buildings", "environment"):
        if arrays[name].dtype.kind not in "biuf" or arrays[name + "_valid"].dtype != np.bool_:
            raise ValueError("Numeric arrays and boolean validity masks required")
    # Full state/environment grids cannot hide active, nonfinite or invalid codes under a mask.
    if not np.all(np.isin(state, (1, 2, 5))):
        raise ValueError("Inactive final state must contain only codes 1, 2 and 5")
    if not np.all(arrays["environment_valid"]) or not np.all(np.isin(environment, (0, 1))):
        raise ValueError("Environment mask must be completely valid and binary")
    for name, codes in (("reference", (0, 1)), ("buildings", (0, 10))):
        if not np.all(np.isin(arrays[name][arrays[name + "_valid"]], codes)):
            raise ValueError(f"Unexpected {name} category before conversion")
    env = environment == 1
    domain = (env & arrays["prediction_valid"] & arrays["reference_valid"]
              & arrays["buildings_valid"] & (buildings == 10))
    metrics = evaluate_final_footprint_arrays(
        state, reference == 1, env, arrays["prediction_valid"], arrays["reference_valid"],
        arrays["buildings_valid"], buildings == 10, termination_reason="inactive")
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    metrics["accuracy"] = (tn + tp) / (tn + fp + fn + tp)
    metrics["predicted_burned_cells"] = int(np.count_nonzero((state == 5) & domain))
    metrics["predicted_burned_full_crop"] = int(np.count_nonzero(state == 5))
    coverage = metrics["reference_coverage"]
    total = coverage["reference_valid_burned_total"]
    metrics["excluded_percentage"] = (100 * coverage["reference_burned_excluded_from_primary_domain"] / total
                                      if total else None)
    # 0 excluded, 1 TN, 2 TP, 3 FP, 4 FN; display never selects the metric domain.
    difference = np.zeros(shape, dtype=np.uint8)
    truth, prediction = reference == 1, state == 5
    for code, cells in ((1, ~truth & ~prediction), (2, truth & prediction),
                        (3, ~truth & prediction), (4, truth & ~prediction)):
        difference[domain & cells] = code
    return dict(metrics=metrics, domain=domain, difference=difference,
                reference=truth & arrays["reference_valid"], prediction=prediction)


def accepted_disagreements(result: dict) -> list[str]:
    m = result["metrics"]
    c = m["reference_coverage"]
    expected = {"confusion_matrix": [[133219, 863], [115, 1795]],
                "evaluation_pixel_count": 135992, "predicted_burned_cells": 2658,
                "predicted_burned_full_crop": 2658}
    differences = [f"{key}: computed {m[key]!r}, expected {value!r}"
                   for key, value in expected.items() if m[key] != value]
    for key, value in (("reference_valid_burned_total", 3312),
                       ("reference_burned_included_in_primary_domain", 1910),
                       ("reference_burned_excluded_from_primary_domain", 1402)):
        if c[key] != value:
            differences.append(f"{key}: computed {c[key]}, expected {value}")
    partition = c["exclusion_priority_partition"]
    for key, value in (("outside_environmental_validity", 20), ("outside_prediction_validity", 0),
                       ("outside_building_raster_validity", 0), ("outside_mapped_buildings", 1382)):
        if partition[key] != value:
            differences.append(f"{key}: computed {partition[key]}, expected {value}")
    for key, expected_value in (("accuracy", "0.992808"), ("precision", "0.675320"),
                                ("recall", "0.939791"), ("f1_score", "0.785902"),
                                ("jaccard_index", "0.647313")):
        if f"{m[key]:.6f}" != expected_value:
            differences.append(f"{key}: computed {m[key]!r}, expected {expected_value} rounded")
    if m["excluded_percentage"] is None or f"{m['excluded_percentage']:.2f}" != "42.33":
        differences.append("Excluded percentage differs from accepted 42.33%")
    return differences


def print_results(result: dict, identities: dict) -> None:
    for name, identity in identities.items():
        print(f"{name}: {identity}")
    print("F01: parent 5489x6896; rows [256,1280), columns [2176,3200); EPSG:32651")
    print(f"Output transform: {tuple(CROP_TRANSFORM)[:6]}")
    m = result["metrics"]
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {m['confusion_matrix']}")
    for label, key in (("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                       ("F1", "f1_score"), ("Jaccard", "jaccard_index")):
        print(f"{label}: {m[key]:.6f}")
    print(f"Evaluated mapped-building cells: {m['evaluation_pixel_count']:,}")
    print(f"Predicted burned cells: {m['predicted_burned_cells']:,} (full crop {m['predicted_burned_full_crop']:,})")
    c = m["reference_coverage"]
    print(f"Reference burned within crop: {c['reference_valid_burned_total']:,}")
    print(f"Included: {c['reference_burned_included_in_primary_domain']:,}; "
          f"excluded: {c['reference_burned_excluded_from_primary_domain']:,}")
    percentage = m["excluded_percentage"]
    print("Excluded percentage: " + (f"{percentage:.2f}%" if percentage is not None else "undefined (no reference burns)"))
    print(f"Disjoint exclusions: {c['exclusion_priority_partition']}")
    print("Spatial ROC/AUC is not reported for categorical final-footprint comparison.")
    print("F01 is a development/calibration case; no held-out or physical-validity claim.")


def draw_figure(figure, result: dict, extent: tuple) -> None:
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    axes = figure.subplots(1, 3, sharex=True, sharey=True)
    colors = ["#d0d0d0", "white", "#303030", "#e7ad48"]
    for ax, key, title in zip(axes[:2], ("reference", "prediction"),
                              ("Provided approximate burned-area reference",
                               "RF + CA final extinguished footprint")):
        panel = np.where(result["domain"], np.where(result[key], 2, 1),
                         np.where(result[key], 3, 0))
        ax.imshow(panel, cmap=ListedColormap(colors), vmin=0, vmax=3, origin="upper",
                  extent=extent, interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=10)
    diff_colors = ["#d0d0d0", "white", "green", "red", "blue"]
    axes[2].imshow(result["difference"], cmap=ListedColormap(diff_colors), vmin=0, vmax=4,
                   origin="upper", extent=extent, interpolation="nearest", aspect="equal")
    axes[2].set_title("Difference within evaluation domain", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Easting (m)")
        ax.ticklabel_format(style="plain", useOffset=False)
    axes[0].set_ylabel("Northing (m)")
    figure.legend(handles=[Patch(facecolor=c, edgecolor="black", label=label)
                           for c, label in zip(colors, ("Excluded", "Evaluated unburned",
                                                       "Burned", "Burned outside domain"))],
                  loc="upper center", bbox_to_anchor=(0.34, 0.88), ncol=2, fontsize=8)
    axes[2].legend(handles=[Patch(facecolor=c, edgecolor="black", label=label)
                           for c, label in zip(diff_colors, ("Excluded", "TN", "TP", "FP", "FN"))],
                   loc="upper right", fontsize=8)
    m = result["metrics"]
    excluded = m["excluded_percentage"]
    percent = f"{excluded:.2f}%" if excluded is not None else "undefined"
    figure.text(0.5, 0.035, f"F01 development case | common-valid mapped buildings: {m['evaluation_pixel_count']:,} | "
                f"F1 {m['f1_score']:.6f}, Jaccard {m['jaccard_index']:.6f}\n"
                f"Crop reference-burn cells excluded: {percent}. Excluded cells are not TN. "
                "Approximate reference; no temporal/physical validation.", ha="center", fontsize=9)
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.76, wspace=0.15)


def main() -> None:
    arrays, identities = load_inputs()
    result = evaluate_arrays(arrays)
    print_results(result, identities)
    differences = accepted_disagreements(result)
    if differences:
        raise ValueError("DISAGREEMENT with accepted F01 evaluation:\n" + "\n".join(differences))
    print("Accepted F01 expectation check: PASS", flush=True)
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 7))
    left, top = CROP_TRANSFORM * (0, 0)
    right, bottom = CROP_TRANSFORM * (CROP_SHAPE[1], CROP_SHAPE[0])
    draw_figure(figure, result, (left, right, bottom, top))
    plt.show()


if __name__ == "__main__":
    main()
