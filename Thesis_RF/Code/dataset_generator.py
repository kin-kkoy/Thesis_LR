"""Generate ML training rows from aligned raster stacks with balanced label sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml

from modules.feature_pipeline import FeatureAssembler


class SyntheticDatasetGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.base_dir = Path(__file__).resolve().parent
        self.dataset_cfg = config["dataset_generation"]
        self.output_csv = self.dataset_cfg.get("output_csv", "dataFiles/revised_dataset_5tier.csv")

        self.seed = int(config["simulation"].get("seed", 42))
        self.rng = np.random.default_rng(self.seed)

        self.roi = self.dataset_cfg.get("roi")
        self.min_rows = int(self.dataset_cfg.get("min_rows", 50_000))
        self.negative_to_positive_ratio = int(self.dataset_cfg.get("negative_to_positive_ratio", 10))
        self.fallback_to_full_extent = bool(
            self.dataset_cfg.get("fallback_to_full_extent_if_no_positive", True)
        )

        self.wind_cfg = config["wind"]

        environment_cfg = dict(config["environment"])
        raster_dir = Path(environment_cfg["raster_dir"])
        if not raster_dir.is_absolute():
            raster_dir = (self.base_dir / raster_dir).resolve()

        self.raster_dir = raster_dir
        self.slope_file = environment_cfg["slope_file"]
        self.proximity_file = environment_cfg["proximity_file"]
        self.buildings_file = environment_cfg["buildings_file"]
        self.materials_file = environment_cfg.get("materials_file", "stack_materials.tif")
        if Path(self.materials_file).name != "stack_materials.tif":
            raise ValueError(
                "5-tier dataset generation requires stack_materials.tif as the materials raster"
            )
        self.ground_truth_file = self.dataset_cfg.get("ground_truth_file", "stack_ground_truth.tif")
        self.nodata_value = float(environment_cfg.get("nodata_value", -9999))

    def _read_raster(
        self,
        name: str,
        filename: str,
        reference: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        raster_path = self.raster_dir / filename
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster file not found for {name}: {raster_path}")

        with rasterio.open(raster_path) as src:
            arr = src.read(1)
            valid_mask = src.read_masks(1) > 0
            if src.nodata is not None:
                valid_mask &= arr != src.nodata

            current = {
                "shape": arr.shape,
                "crs": src.crs,
                "transform": src.transform,
            }

        if reference is None:
            reference = current
        else:
            if current["shape"] != reference["shape"]:
                raise ValueError(
                    f"Alignment mismatch for {name}: shape {current['shape']} != {reference['shape']}"
                )
            if current["crs"] != reference["crs"]:
                raise ValueError(
                    f"Alignment mismatch for {name}: CRS {current['crs']} != {reference['crs']}"
                )
            if current["transform"] != reference["transform"]:
                raise ValueError(
                    "Alignment mismatch for "
                    f"{name}: transform {current['transform']} != {reference['transform']}"
                )

        return np.asarray(arr, dtype=np.float32), valid_mask, reference

    def _load_aligned_stack(self) -> dict[str, Any]:
        reference: dict[str, Any] | None = None

        slope, slope_valid, reference = self._read_raster("slope", self.slope_file, reference)
        proximity, proximity_valid, reference = self._read_raster(
            "proximity",
            self.proximity_file,
            reference,
        )
        buildings, buildings_valid, reference = self._read_raster(
            "buildings",
            self.buildings_file,
            reference,
        )
        materials, materials_valid, reference = self._read_raster(
            "materials",
            self.materials_file,
            reference,
        )
        ground_truth, gt_valid, reference = self._read_raster(
            "ground_truth",
            self.ground_truth_file,
            reference,
        )

        valid_mask = slope_valid & proximity_valid & buildings_valid & materials_valid & gt_valid

        # Defend against rasters where nodata is encoded in values but nodata metadata is unset.
        valid_mask &= np.isfinite(slope)
        valid_mask &= np.isfinite(proximity)
        valid_mask &= np.isfinite(buildings)
        valid_mask &= np.isfinite(materials)
        valid_mask &= np.isfinite(ground_truth)
        valid_mask &= slope != self.nodata_value
        valid_mask &= slope != -9999.0

        rounded_materials = np.rint(materials)
        material_class = rounded_materials.astype(np.int16)
        material_is_integer_like = np.isclose(materials, rounded_materials, atol=1e-6)
        gt_class = np.rint(ground_truth).astype(np.int16)
        valid_mask &= material_is_integer_like
        valid_mask &= np.isin(material_class, np.arange(6, dtype=np.int16))
        valid_mask &= np.isin(gt_class, np.array([0, 1], dtype=np.int16))

        return {
            "slope": slope,
            "proximity": proximity,
            "buildings": buildings,
            "materials": materials,
            "ground_truth": ground_truth,
            "valid_mask": valid_mask,
            "shape": reference["shape"],
            "crs": reference["crs"],
            "transform": reference["transform"],
        }

    def _crop_stack(self, stack: dict[str, Any], roi: dict | None) -> dict[str, Any]:
        if roi is None:
            return stack

        row_start = int(roi["row_start"])
        row_end = int(roi["row_end"])
        col_start = int(roi["col_start"])
        col_end = int(roi["col_end"])

        row_slice = slice(row_start, row_end)
        col_slice = slice(col_start, col_end)

        cropped = {
            "slope": stack["slope"][row_slice, col_slice],
            "proximity": stack["proximity"][row_slice, col_slice],
            "buildings": stack["buildings"][row_slice, col_slice],
            "materials": stack["materials"][row_slice, col_slice],
            "ground_truth": stack["ground_truth"][row_slice, col_slice],
            "valid_mask": stack["valid_mask"][row_slice, col_slice],
            "shape": stack["slope"][row_slice, col_slice].shape,
            "crs": stack["crs"],
            "transform": stack["transform"],
        }
        return cropped

    def _build_feature_layers(self, stack: dict[str, Any]) -> dict[str, np.ndarray]:
        slope_risk = np.clip(stack["slope"], 0.0, 10.0).astype(np.float32) / np.float32(10.0)
        proximity_risk = stack["proximity"].astype(np.float32) / np.float32(10.0)

        building_presence_raw = np.where(stack["buildings"] == 10.0, 1.0, 0.0).astype(np.float32)
        material_class_raw = np.where(stack["valid_mask"], stack["materials"], 0.0).astype(np.float32)
        assembler_env = {
            "slope_risk": slope_risk,
            "proximity_risk": proximity_risk,
            "building_presence": building_presence_raw,
            "material_class": material_class_raw,
            "burnable_mask": stack["valid_mask"].astype(bool),
            "grid_shape": stack["shape"],
        }
        assembler = FeatureAssembler(
            assembler_env,
            self.wind_cfg,
            self.config.get("flammability_weights"),
        )

        labels = (np.rint(stack["ground_truth"]).astype(np.int8) == 1)

        return {
            "slope_risk": assembler.slope_risk,
            "proximity_risk": assembler.proximity_risk,
            "building_presence": assembler.building_presence,
            "material_risk": assembler.material_risk,
            "material_class": assembler.material_class,
            "composite_flammability": assembler.composite_flammability,
            "labels": labels,
        }

    def _sample_balanced_indices(self, valid_mask: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
        flat_valid = valid_mask.ravel()
        flat_labels = labels.ravel()

        positive_indices = np.flatnonzero(flat_valid & flat_labels)
        negative_indices = np.flatnonzero(flat_valid & (~flat_labels))

        positive_count = int(positive_indices.size)
        total_negative_count = int(negative_indices.size)

        if positive_count == 0:
            return np.empty(0, dtype=np.int64), 0, 0, total_negative_count, int(np.count_nonzero(flat_valid))

        ratio_target = positive_count * self.negative_to_positive_ratio
        min_rows_target = max(self.min_rows - positive_count, 0)
        target_negative = max(ratio_target, min_rows_target)
        sampled_negative_count = int(min(total_negative_count, target_negative))

        if sampled_negative_count > 0:
            sampled_negative = self.rng.choice(
                negative_indices,
                size=sampled_negative_count,
                replace=False,
            )
            selected_indices = np.concatenate([positive_indices, sampled_negative]).astype(np.int64)
        else:
            selected_indices = positive_indices.astype(np.int64)

        self.rng.shuffle(selected_indices)
        valid_count = int(np.count_nonzero(flat_valid))
        return selected_indices, positive_count, sampled_negative_count, total_negative_count, valid_count

    def generate(self) -> str:
        full_stack = self._load_aligned_stack()
        working_stack = self._crop_stack(full_stack, self.roi)

        features = self._build_feature_layers(working_stack)
        selected, positives, sampled_negatives, all_negatives, valid_count = self._sample_balanced_indices(
            working_stack["valid_mask"],
            features["labels"],
        )

        if positives == 0 and self.roi is not None and self.fallback_to_full_extent:
            print("ROI has no Ignited=1 pixels in stack_ground_truth.tif. Falling back to full raster extent.")
            working_stack = full_stack
            features = self._build_feature_layers(working_stack)
            selected, positives, sampled_negatives, all_negatives, valid_count = self._sample_balanced_indices(
                working_stack["valid_mask"],
                features["labels"],
            )

        if selected.size == 0:
            raise ValueError(
                "No positive pixels found in the selected extent after masking. "
                "Check ROI and stack_ground_truth.tif coverage."
            )

        n_samples = int(selected.size)
        direction_deg = float(self.wind_cfg["direction_deg"])
        radians = np.deg2rad(direction_deg)

        wind_speed_col = np.full(n_samples, float(self.wind_cfg["speed_kmh"]), dtype=np.float32)
        wind_sin_col = np.full(n_samples, float(np.sin(radians)), dtype=np.float32)
        wind_cos_col = np.full(n_samples, float(np.cos(radians)), dtype=np.float32)
        neighbor_burning_count_col = np.zeros(n_samples, dtype=np.float32)

        dataframe = pd.DataFrame(
            {
                "slope_risk": features["slope_risk"].ravel()[selected].astype(np.float32),
                "proximity_risk": features["proximity_risk"].ravel()[selected].astype(np.float32),
                "building_presence": features["building_presence"].ravel()[selected].astype(np.float32),
                "material_risk": features["material_risk"].ravel()[selected].astype(np.float32),
                "material_class": features["material_class"].ravel()[selected].astype(np.int8),
                "wind_speed": wind_speed_col,
                "wind_sin": wind_sin_col,
                "wind_cos": wind_cos_col,
                "neighbor_burning_count": neighbor_burning_count_col,
                "composite_flammability": features["composite_flammability"].ravel()[selected].astype(np.float32),
                "Ignited": features["labels"].ravel()[selected].astype(np.float32),
            }
        )

        output_path = Path(self.output_csv)
        if not output_path.is_absolute():
            output_path = self.base_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)

        row_count = int(len(dataframe))
        ignited_fraction = float(dataframe["Ignited"].mean()) if row_count > 0 else 0.0

        print(f"Saved synthetic dataset: {output_path}")
        print(f"Rows recorded: {row_count}")
        print(f"Valid stacked pixels: {valid_count}")
        print(f"All positives kept: {positives}")
        print(f"Negatives sampled: {sampled_negatives} (available={all_negatives})")
        print(f"Ignited=1 percentage: {ignited_fraction * 100.0:.2f}%")

        return str(output_path)


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "config" / "default_experiment.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    generator = SyntheticDatasetGenerator(config)
    generator.generate()