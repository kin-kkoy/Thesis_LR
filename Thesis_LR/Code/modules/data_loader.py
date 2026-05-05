"""Load and prepare aligned environmental rasters for fire spread simulation."""
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


class EnvironmentManager:
    def __init__(self, config: dict):
        self.config = config
        self.raster_dir = Path(config["raster_dir"])
        self.slope_file = config["slope_file"]
        self.proximity_file = config["proximity_file"]
        self.buildings_file = config["buildings_file"]
        self.materials_file = config.get("materials_file", "")
        self.nodata_value = config.get("nodata_value", -9999)

        self.transform = None
        self.crs = None
        self.grid_shape = None

        self.slope_raw = None
        self.proximity_raw = None
        self.buildings_raw = None
        self.materials_raw = None

        self.nodata_mask = None
        self.burnable_mask = None

        self.slope_risk = None
        self.proximity_risk = None
        self.building_presence = None
        self.material_risk = None

    def load_rasters(self) -> None:
        raster_specs = [
            ("slope", self.slope_file),
            ("proximity", self.proximity_file),
            ("buildings", self.buildings_file),
            ("materials", self.materials_file),
        ]

        for index, (name, filename) in enumerate(raster_specs):
            raster_path = self.raster_dir / filename
            with rasterio.open(raster_path) as src:
                arr = src.read(1).astype(np.float32)

                if index == 0:
                    self.transform = src.transform
                    self.crs = src.crs
                    self.grid_shape = arr.shape
                else:
                    if arr.shape != self.grid_shape:
                        raise ValueError(
                            f"Alignment mismatch for {name}: shape {arr.shape} != {self.grid_shape}"
                        )
                    if src.crs != self.crs:
                        raise ValueError(
                            f"Alignment mismatch for {name}: CRS {src.crs} != {self.crs}"
                        )
                    if src.transform != self.transform:
                        raise ValueError(
                            f"Alignment mismatch for {name}: transform {src.transform} != {self.transform}"
                        )

                if name == "slope":
                    self.slope_raw = arr
                elif name == "proximity":
                    self.proximity_raw = arr
                elif name == "buildings":
                    self.buildings_raw = arr
                else:
                    self.materials_raw = arr

    def build_masks(self) -> None:
        if (
            self.slope_raw is None
            or self.proximity_raw is None
            or self.buildings_raw is None
            or self.materials_raw is None
        ):
            raise ValueError("Rasters are not loaded. Call load_rasters() first.")

        nodata_mask = (
            (self.slope_raw == self.nodata_value)
            | (self.proximity_raw == self.nodata_value)
            | (self.buildings_raw == self.nodata_value)
            | (self.materials_raw == self.nodata_value)
            | (self.slope_raw == -9999)
        )
        ocean_mask = (
            (self.slope_raw == 0)
            & (self.proximity_raw == 0)
            & (self.buildings_raw == 0)
            & (self.materials_raw == 0)
        )

        self.nodata_mask = nodata_mask | ocean_mask
        self.burnable_mask = ~self.nodata_mask

    def normalize_layers(self) -> None:
        if self.nodata_mask is None:
            raise ValueError("Masks are not built. Call build_masks() first.")

        slope = self.slope_raw.copy()
        slope[(slope < 0) | (slope > 10)] = 0
        self.slope_risk = slope / 10.0
        self.slope_risk[self.nodata_mask] = 0

        self.proximity_risk = self.proximity_raw / 10.0
        self.proximity_risk[self.nodata_mask] = 0

        self.building_presence = np.where(self.buildings_raw == 10, 1, 0).astype(np.int8)

        self.material_risk = self.materials_raw / 10.0
        self.material_risk[self.nodata_mask] = 0

    def get_environment(self) -> dict[str, Any]:
        return {
            "slope_risk": self.slope_risk,
            "proximity_risk": self.proximity_risk,
            "building_presence": self.building_presence,
            "material_risk": self.material_risk,
            "burnable_mask": self.burnable_mask,
            "nodata_mask": self.nodata_mask,
            "grid_shape": self.grid_shape,
            "transform": self.transform,
            "crs": self.crs,
        }

    def summary(self) -> None:
        if (
            self.slope_risk is None
            or self.proximity_risk is None
            or self.building_presence is None
            or self.material_risk is None
        ):
            raise ValueError("Layers are not normalized. Call normalize_layers() first.")

        burnable_count = int(np.count_nonzero(self.burnable_mask))
        building_count = int(np.count_nonzero(self.building_presence == 1))

        slope_min = float(np.min(self.slope_risk))
        slope_max = float(np.max(self.slope_risk))
        proximity_min = float(np.min(self.proximity_risk))
        proximity_max = float(np.max(self.proximity_risk))
        material_min = float(np.min(self.material_risk))
        material_max = float(np.max(self.material_risk))

        print(f"Grid shape: {self.grid_shape}")
        print(f"CRS: {self.crs}")
        print(f"Burnable cells: {burnable_count}")
        print(f"Building cells: {building_count}")
        print(f"Slope risk range: [{slope_min:.4f}, {slope_max:.4f}]")
        print(f"Proximity risk range: [{proximity_min:.4f}, {proximity_max:.4f}]")
        print(f"Material risk range: [{material_min:.4f}, {material_max:.4f}]")