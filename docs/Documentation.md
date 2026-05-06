# Documentation

Date: 2026-05-06

## Context7 references
- scikit-learn RandomForestClassifier uses predict_proba() to return per-class probabilities.
- feature_names_in_ can be used to align feature columns when the model was fitted with named features.

## Execution notes
- Stage A RF simulation previously failed with a memory error when predicting on the full grid (Unable to allocate 578 MiB for predict_proba output).
- After chunked, susceptible-only inference, the Stage A run completed but still stopped at step 1 (no active fire spread).
- Added explicit transition timers for Stage A: T_3_to_4=1, T_4_to_5=4.
- Grid-specific p_ignite stats at t=0 (susceptible cells around ignition):
  susceptible_cells=8, min=0.0368, max=0.0827, mean=0.0769, median=0.0827,
  %>0.10=0.00%, %>0.30=0.00%, %>0.50=0.00%.
- Validation (t=0.40): precision 1.0, recall 0.000302, F1 0.000604, AUC-ROC 0.500151, Jaccard 0.000302.
- Visualization saved to: Thesis_LR/Code/sandbox_kent/output/final_state_stage_a.png

## Implementation updates
- sandbox_kent/modules/automata_engine.py now predicts only on susceptible cells and uses chunked inference to reduce memory usage.
- Chunk size is configurable via ml_model.inference_chunk_size (default 200000).
- Feature-name alignment is applied when model.feature_names_in_ is available.
- ML thresholding now respects ml_model.proba_threshold (default 0.5). If > 0, susceptible cells with p_ignite >= threshold ignite deterministically, while the remaining susceptible cells still use stochastic ignition.

## Formulas and feature ordering
- Susceptible mask: (grid == STATE_NOT_YET_BURNING) & (blazing_neighbor_count > 0)
- Transition timers:
  ignite_to_blazing when ignition_timers >= T_3_to_4
  blazing_to_extinguished when blazing_timers >= T_4_to_5
- Stage A feature vector order:
  slope_risk, proximity_risk, building_presence, material_risk, material_class,
  wind_speed, wind_sin, wind_cos, neighbor_burning_count, composite_flammability
- ML ignition probability: p_ignite = model.predict_proba(features)[:, 1]
