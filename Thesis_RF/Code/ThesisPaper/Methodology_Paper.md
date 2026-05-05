# Methodology Review

This document summarizes the verification of the implementation against the thesis methodology.

## May 5, 2026

### Feature Engineering and Pipeline

- **Feature Order**: Verified that the 11 features in `feature_pipeline.py` are correctly ordered as per the `FEATURE_NAMES` list. The order is: `slope_risk`, `proximity_risk`, `building_presence`, `material_risk`, `material_class`, `wind_speed`, `wind_sin`, `wind_cos`, `neighbor_burning_count`, `composite_flammability`, `wind_weighted_score`.

### Automata Engine

- **Wind Convolution**: Verified that the `wind_weighted_score` is calculated using a dynamic spatial convolution in `automata_engine.py`. The `_compute_wind_kernel` method correctly generates a directional kernel based on wind speed and direction, and this kernel is used to weight the influence of burning neighbors when predicting ignition. This matches the mathematical methodology described in the thesis.