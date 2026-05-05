"""Physics-informed interaction features for Logistic Regression.

LR can only learn additive combinations of features. Fire ignition depends on
nonlinear interactions (e.g., steep slope WITH a burning upwind neighbor is far
more dangerous than either alone). These engineered features expose key
interactions as explicit columns so LR can weight them directly.

Grounding:
  - Alexandridis et al. (2011): slope and wind directionally modulate spread
    probability from burning neighbors in CA fire models.
  - Gao et al. (2008): wind velocity compounds with neighbor fire presence.
  - Wildland-urban interface literature: building proximity to fire-prone
    boundaries is a key risk factor.

Usage:
    from modules.feature_engineering import add_interaction_features
    df = add_interaction_features(df)  # adds 7 new columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd


INTERACTION_FEATURES = [
    "slope_x_neighbors",
    "wind_x_neighbors",
    "slope_x_building",
    "slope_x_wind",
    "proximity_x_building",
    "proximity_x_neighbors",
    "neighbors_squared",
]


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physics-informed interaction features to a fire dataset.

    Expects the original 9 feature columns to be present.
    Returns a copy with 7 new columns appended (does not modify the input).
    """
    out = df.copy()

    # 1. Slope x burning neighbors: fire travels uphill faster, compounding
    #    with neighbor fire presence (Alexandridis 2011)
    out["slope_x_neighbors"] = out["slope_risk"] * out["neighbor_burning_count"]

    # 2. Wind speed x burning neighbors: wind fans flames from neighbors,
    #    stronger wind + more neighbors = faster spread (Gao 2008)
    out["wind_x_neighbors"] = out["wind_speed"] * out["neighbor_burning_count"]

    # 3. Slope x building presence: buildings on steep slopes are more
    #    vulnerable — uphill fire channels heat into structures
    out["slope_x_building"] = out["slope_risk"] * out["building_presence"]

    # 4. Slope x wind speed: steep terrain + wind creates channeling effects,
    #    both factors compound each other (Alexandridis 2011, Gao 2008)
    out["slope_x_wind"] = out["slope_risk"] * out["wind_speed"]

    # 5. Proximity x building: wildland-urban interface risk — buildings
    #    near roads/boundaries are more exposed to approaching fire
    out["proximity_x_building"] = out["proximity_risk"] * out["building_presence"]

    # 6. Proximity x burning neighbors: cells near edges with burning
    #    neighbors are more accessible to fire spread
    out["proximity_x_neighbors"] = out["proximity_risk"] * out["neighbor_burning_count"]

    # 7. Neighbor count squared: the relationship between burning neighbor
    #    count and ignition is likely nonlinear — surrounded cells face
    #    converging heat from multiple angles
    out["neighbors_squared"] = out["neighbor_burning_count"] ** 2

    return out
