# Code — CA + ML Fire Spread Prediction (Lapu-Lapu City)

This directory contains the codebase for the thesis investigating Cellular Automata + Machine Learning hybrid fire spread prediction. Logistic Regression is owned by Kristian; Random Forest is owned by Kent (his work is on the same git branch lineage but in a separate train/validate flow).

## Directory layout

```
Code/
├── README.md                       this file
├── main.py                         entry point — runs orchestrator with config/default_experiment.yaml
├── orchestrator.py                 loads rasters, builds CA, loads ML model, runs simulation, saves final_state.tif
├── dataset_generator.py            single-scenario synthetic data generation
├── generate_multi_scenario.py      batch data generation (200 runs across wind speed × direction × seed)
├── train_lr.py                     baseline LR training
├── train_lr_optuna.py              LR with Bayesian hyperparameter tuning (5-fold CV)
├── train_lr_pathb.py               LR with feature engineering + Optuna
│
├── config/
│   └── default_experiment.yaml     simulation parameters (rasters, ignition points, wind, etc.)
│
├── modules/                        core engine + ML pipeline
│   ├── automata_engine.py          5-state CA with directional wind kernel
│   ├── feature_pipeline.py         FeatureAssembler — builds per-cell feature vectors
│   ├── feature_engineering.py      physics-informed interaction features (Path B)
│   ├── data_loader.py              EnvironmentManager — loads & normalizes rasters
│   └── model_trainer.py            ModelTrainer — fits/evaluates LR or RF with threshold optimization
│
├── spatial_validation/
│   └── validate_simulation.py      compares final_state.tif against ground_truth.tif
│
├── dataFiles/                      training datasets (multi_scenario_dataset.csv is the live one)
├── models/                         trained model artifacts (.joblib)
├── output/                         simulation outputs (gitignored)
├── processedData/                  raster data (gitignored)
│
├── sandbox_kent/                   sandbox for running Kent's pipeline (Stage A baseline)
└── backups/                        pre-modification snapshots (option2_information_gap_fix/)
```

## Common tasks

**Train baseline LR on the current dataset:**
```
python train_lr.py
```

**Run Optuna hyperparameter search:**
```
python train_lr_optuna.py
```

**Run feature-engineering + Optuna:**
```
python train_lr_pathb.py
```

**Generate fresh multi-scenario training data (200 simulation runs, ~30 min):**
```
python generate_multi_scenario.py
```

**Run a full CA + ML simulation:**
```
python main.py
```
Edit `config/default_experiment.yaml` first to set ignition points, wind, max_timesteps, and the model path.

**Validate a simulation result spatially:**
```
python spatial_validation/validate_simulation.py \
    --final output/final_state.tif \
    --gt processedData/raster/raster/stack_ground_truth.tif
```

## Feature pipeline notes

The current `modules/feature_pipeline.py` exposes 10 features ending in `wind_weighted_score` (added during the Information Gap investigation — see `docs/findings.md`). Kent's RF was originally trained on a 10-feature schema that includes `material_class` instead. **A unified 11-feature set is planned** but requires both LR and RF to be retrained; until that's done, comparisons should note which schema is in use.

## Branches

- `rf-training-8params-test` — Kent's baseline (9 features, original)
- `[new branch — set during cleanup commit]` — current work with `wind_weighted_score`, Optuna, feature engineering, spatial validation script
