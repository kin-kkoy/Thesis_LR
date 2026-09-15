# Coder memory

Date: 2026-05-08

## Context7 notes
- scikit-learn estimators store feature names in `feature_names_in_` when fit with a pandas DataFrame (all-string columns), which can be used to align columns during predict.
- Classifiers like `RandomForestClassifier` expose `predict_proba(X)` for per-class probabilities.

## Execution notes
- Stage C RF (sandbox_kent multi_scenario_dataset.csv): rows=24922, threshold=0.10,
  precision=0.081272, recall=0.640669, F1=0.144246, AUC-ROC=0.556544, Jaccard=0.077729.
- Model saved to Thesis_LR/Code/sandbox_kent/models/fire_rf_stage_c.joblib.
