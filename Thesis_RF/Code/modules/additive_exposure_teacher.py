"""Exploratory R2 arithmetic only; never selects or runs a CA."""
from dataclasses import dataclass

import numpy as np

TEACHER_ID = "exploratory_additive_exposure_teacher.r2"
NUMERICAL_ID = "additive_exposure_quantized_f32_f64.r2"
COEFFICIENTS = {
    "base_ignition_prob": 0.12, "slope_weight": 0.20,
    "building_weight": 0.10, "proximity_weight": 0.20,
    "material_weight": 0.35,
}


def _arrays32(**values):
    arrays = tuple(values.values())
    if not arrays or any(not isinstance(a, np.ndarray) or a.dtype != np.float32
                         for a in arrays):
        raise TypeError("Inputs must be float32 arrays without coercion")
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Input shapes must match")
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("Inputs must be finite")
    return arrays


@dataclass(frozen=True)
class Susceptibility:
    products: tuple
    additions: tuple
    beta32: np.ndarray


@dataclass(frozen=True)
class Probability:
    exposure64: np.ndarray
    q64: np.ndarray
    q32: np.ndarray


def compute_susceptibility32(*, slope32, building32, proximity32, material_risk32):
    """Separate float32 operations; negative proximity is arithmetic-only."""
    values = _arrays32(s=slope32, B=building32, p=proximity32, m=material_risk32)
    products, additions = [], []
    total = np.full(values[0].shape, np.float32(0.12), dtype=np.float32)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        for coefficient, value in zip((0.20, 0.10, 0.20, 0.35), values):
            product = np.multiply(np.float32(coefficient), value, dtype=np.float32)
            if not np.all(np.isfinite(product)):
                raise ValueError("Nonfinite susceptibility product")
            products.append(product)
        for product in products:
            total = np.add(total, product, dtype=np.float32)
            if not np.all(np.isfinite(total)):
                raise ValueError("Nonfinite susceptibility addition")
            additions.append(total)
    beta = np.clip(total, np.float32(0), np.float32(1))
    return Susceptibility(tuple(products), tuple(additions), beta)


def compute_additive_exposure_probability(*, beta32, wind_weighted_score32):
    beta, score = _arrays32(beta=beta32, score=wind_weighted_score32)
    if np.any((beta < 0) | (beta > 1)) or np.any((score < 0) | (score > 1)):
        raise ValueError("Susceptibility and score must be in [0, 1]")
    exposure = np.multiply(score.astype(np.float64), np.float64(8), dtype=np.float64)
    exposure = np.clip(exposure, np.float64(0), np.float64(8))
    exposure[exposure == 0] = np.float64(0)
    widened = beta.astype(np.float64)
    q64 = np.zeros(beta.shape, dtype=np.float64)
    q64[(exposure > 0) & (widened == 1)] = np.float64(1)
    interior = (exposure > 0) & (widened > 0) & (widened < 1)
    result = np.empty(0, dtype=np.float64)
    if np.any(interior):
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            logarithm = np.log1p(-widened[interior], dtype=np.float64)
            argument = np.multiply(exposure[interior], logarithm, dtype=np.float64)
            result = np.negative(np.expm1(argument, dtype=np.float64))
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ValueError("Interior probability violates [0, 1]")
    q64[interior] = result
    q64 = np.clip(q64, np.float64(0), np.float64(1))
    q64[q64 == 0] = np.float64(0)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        q32 = q64.astype(np.float32)
    q32[q32 == 0] = np.float32(0)
    return Probability(exposure, q64, q32)
