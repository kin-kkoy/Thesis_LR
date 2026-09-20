"""Authoritative simulated-wind convention for RF/CA source code.

Directions are meteorological from-bearings measured clockwise from projected
raster grid north. Raster rows increase southward and columns increase
eastward. Fire propagation therefore follows the opposite bearing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.ndimage import correlate


WIND_SCHEMA_VERSION = "simulated_wind.v1"
DIRECTION_CONVENTION = "meteorological_from"
DIRECTION_UNITS = "degrees_clockwise"
NORTH_REFERENCE = "projected_grid_north"
CALM_DIRECTION_DEG = 0.0
CALM_REPRESENTATION = "speed_zero_direction_zero"


def normalize_direction_deg(direction_deg: object) -> float:
    """Normalize a finite bearing to the half-open interval ``[0, 360)``."""
    direction = float(direction_deg)
    if not np.isfinite(direction):
        raise ValueError("wind.direction_deg must be finite")
    normalized = direction % 360.0
    if normalized == 0.0:
        return 0.0
    return float(normalized)


@dataclass(frozen=True)
class WindContract:
    """Validated wind values and their fixed coordinate-system semantics."""

    speed_kmh: float
    direction_deg: float

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "WindContract":
        """Validate configuration metadata and return canonical wind values."""
        expected_metadata = {
            "direction_convention": DIRECTION_CONVENTION,
            "direction_units": DIRECTION_UNITS,
            "north_reference": NORTH_REFERENCE,
            "schema_version": WIND_SCHEMA_VERSION,
            "calm_representation": CALM_REPRESENTATION,
        }
        for key, expected in expected_metadata.items():
            if config.get(key) != expected:
                raise ValueError(f"wind.{key} must be {expected!r}")

        speed = float(config["speed_kmh"])
        if not np.isfinite(speed) or speed < 0.0:
            raise ValueError("wind.speed_kmh must be finite and non-negative")
        direction = normalize_direction_deg(config["direction_deg"])
        if speed == 0.0 and direction != CALM_DIRECTION_DEG:
            raise ValueError(
                "Calm wind must use speed_kmh=0 and direction_deg=0 "
                f"({CALM_REPRESENTATION})"
            )
        return cls(speed_kmh=speed, direction_deg=direction)

    @property
    def is_calm(self) -> bool:
        return self.speed_kmh == 0.0

    @property
    def from_sin(self) -> float:
        """Sine of the normalized original meteorological from-bearing."""
        return float(np.sin(np.deg2rad(self.direction_deg)))

    @property
    def from_cos(self) -> float:
        """Cosine of the normalized original meteorological from-bearing."""
        return float(np.cos(np.deg2rad(self.direction_deg)))

    @property
    def source_direction_rc(self) -> tuple[float, float]:
        """Unit vector from a cell toward the wind's source, as (row, column)."""
        return (-self.from_cos, self.from_sin)

    @property
    def propagation_direction_rc(self) -> tuple[float, float]:
        """Unit propagation vector opposite the from-bearing, as (row, column)."""
        source_row, source_col = self.source_direction_rc
        return (-source_row, -source_col)

    def directional_kernel(
        self,
        wind_weight: float,
        *,
        minimum_weight: float = 0.05,
    ) -> np.ndarray:
        """Return Moore-neighbor weights indexed by source offset from candidate.

        The kernel is intended for ``scipy.ndimage.correlate``. At output cell
        ``(row, col)``, entry ``kernel[dr + 1, dc + 1]`` weights the source cell
        ``(row + dr, col + dc)``. Calm wind gives every neighbor equal weight.
        """
        influence = float(wind_weight)
        floor = float(minimum_weight)
        if not np.isfinite(influence) or influence < 0.0:
            raise ValueError("wind_weight must be finite and non-negative")
        if not np.isfinite(floor) or floor < 0.0:
            raise ValueError("minimum_weight must be finite and non-negative")

        kernel = np.ones((3, 3), dtype=np.float32)
        kernel[1, 1] = np.float32(0.0)
        if self.is_calm:
            return kernel

        propagation_row, propagation_col = self.propagation_direction_rc
        speed_factor = self.speed_kmh / 10.0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                distance = float(np.hypot(dr, dc))
                travel_row = -dr / distance
                travel_col = -dc / distance
                alignment = (
                    travel_row * propagation_row
                    + travel_col * propagation_col
                )
                kernel[dr + 1, dc + 1] = np.float32(
                    max(1.0 + influence * speed_factor * alignment, floor)
                )
        return kernel

    def to_config(self) -> dict[str, object]:
        """Return the canonical configuration representation."""
        return {
            "speed_kmh": self.speed_kmh,
            "direction_deg": self.direction_deg,
            "direction_convention": DIRECTION_CONVENTION,
            "direction_units": DIRECTION_UNITS,
            "north_reference": NORTH_REFERENCE,
            "schema_version": WIND_SCHEMA_VERSION,
            "calm_representation": CALM_REPRESENTATION,
        }

    def to_manifest(self) -> dict[str, object]:
        """Return provenance fields that bind values to this convention."""
        return self.to_config()


def compute_wind_weighted_score(
    blazing_mask: object,
    wind: WindContract,
    wind_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a constant-boundary directional score and its authoritative kernel."""
    blazing = np.asarray(blazing_mask)
    if blazing.ndim != 2:
        raise ValueError("blazing_mask must be two-dimensional")
    if not np.all(np.isin(blazing, (False, True, 0, 1))):
        raise ValueError("blazing_mask must contain only binary values")

    kernel = wind.directional_kernel(wind_weight)
    weighted = correlate(
        blazing.astype(np.float32, copy=False),
        kernel,
        mode="constant",
        cval=0.0,
        output=np.float32,
    )
    kernel_sum = float(kernel.sum())
    if kernel_sum > 0.0:
        weighted = weighted / np.float32(kernel_sum)
    return weighted.astype(np.float32, copy=False), kernel
