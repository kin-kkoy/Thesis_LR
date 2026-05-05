# Methodology Review

This document summarizes the verification of the implementation against the thesis methodology.

## May 5, 2026

### Feature Engineering and Pipeline

- **Feature Order**: Verified that the 11 features in `feature_pipeline.py` are correctly ordered as per the `FEATURE_NAMES` list. The order is: `slope_risk`, `proximity_risk`, `building_presence`, `material_risk`, `material_class`, `wind_speed`, `wind_sin`, `wind_cos`, `neighbor_burning_count`, `composite_flammability`, `wind_weighted_score`.

### Automata Engine

- **Wind Convolution**: Verified that the `wind_weighted_score` is calculated using a dynamic spatial convolution in `automata_engine.py`. The `_compute_wind_kernel` method correctly generates a directional kernel based on wind speed and direction, and this kernel is used to weight the influence of burning neighbors when predicting ignition. This matches the mathematical methodology described in the thesis.

## Model Architecture Impact on Emergent Spatial Behavior

A critical finding emerged from comparing the spatial simulation behavior of the Logistic Regression (LR) and Random Forest (RF) models. While both were trained on identical synthetic data using the same per-cell methodology, their distinct architectures produced dramatically different emergent fire spread patterns within the Cellular Automata (CA) framework.

### Random Forest Integration Findings

1.  **Trial 1 (Aggressive Balancing):** The initial RF model was trained using `balanced_subsample`. This aggressive balancing technique resulted in a model that assigned an ignition probability greater than 50% to 46% of all susceptible cells. When integrated into the CA, this led to a cascading, runaway fire spread that quickly hit the simulation's 300-step ceiling, causing massive over-prediction of the burned area.

2.  **Trial 2 (Conservative Tuning):** To counteract the explosive spread, aggressive balancing was removed (`class_weight=None`). This produced a mathematically precise but overly strict model, where only 2.1% of susceptible cells were assigned an ignition probability above 50%. In the spatial simulation, this model caused the fire to starve and self-extinguish by Step 7, resulting in near-zero predictive power (Recall=0.009, Precision=1.0).

### Core Thesis Conclusion

The per-cell probability calibration methodology yields structurally different emergent spatial behaviors that are highly dependent on the underlying model architecture. RF's non-linear tree-splits create sharp probability cliffs that either explosively cascade or immediately starve in a Moore-neighborhood CA, unlike Logistic Regression's smooth linear boundaries. CA threshold scaling (an inference-time multiplier) is required to bridge the gap between tree-based per-cell probabilities and realistic continuous spatial spread.