# Repository Guidelines

## Project Structure & Module Organization

All implementation code lives under `Code/`. `main.py` and `orchestrator.py` run the Random Forest-assisted fire simulation; `pure_ca_baseline.py` provides the non-ML comparison. Reusable components are in `Code/modules/`: raster loading, feature assembly, automata transitions, and model training. Experiment settings belong in `Code/config/default_experiment.yaml`.

Dataset generation and model tuning are handled by `dataset_generator.py`, `generate_multi_scenario.py`, and `train_rf_optuna.py`. Validation logic is in `validation_engine.py`. Thesis notes are kept in `Code/ThesisPaper/`. Large rasters, generated CSVs, trained models, and simulation results live in `Code/processedData/`, `Code/dataFiles/`, `Code/models/`, and `Code/output/`; these paths are intentionally ignored by Git.

## Build, Test, and Development Commands

This project has no build step or committed dependency file. Use Python 3.10+ in a virtual environment and install the imports required by the scripts (notably NumPy, pandas, Rasterio, SciPy, scikit-learn, Joblib, PyYAML, Optuna, and Matplotlib).

- `python Code/main.py` runs the configured RF-assisted simulation.
- `python Code/pure_ca_baseline.py --config Code/config/default_experiment.yaml` runs the deterministic baseline.
- `python Code/dataset_generator.py` creates the configured synthetic dataset.
- `python Code/train_rf_optuna.py` tunes and saves the Random Forest model.
- `python Code/validation_engine.py` reports raster-level precision, recall, F1, AUC-ROC, and Jaccard metrics.
- `python -m compileall Code` performs a quick syntax check.

Run commands from the repository root. Confirm the required ignored raster/model inputs exist before starting long simulations.

## Coding Style & Naming Conventions

Follow PEP 8, use four spaces, and add type hints to new public functions. Prefer `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for module constants. Use `pathlib.Path` for paths and resolve project-relative inputs from `Path(__file__)`. Keep stochastic behavior reproducible through the YAML `simulation.seed`; do not introduce unseeded global randomness.

## Testing Guidelines

There is currently no automated test directory or coverage threshold. For changes, run `compileall`, the affected entry point, and validation when outputs change. New tests should use `pytest`, live under `tests/`, and follow `test_<module>.py` and `test_<behavior>()` naming. Use small synthetic arrays instead of committing large raster fixtures.

## Commit & Pull Request Guidelines

Recent history uses short, descriptive, sentence-style commit subjects; no formal prefix convention is established. Keep each commit focused and describe the outcome (for example, `Add checkpoint resume validation`). Pull requests should summarize the method changed, list commands run, identify config/data assumptions, and link related issues. Include key metric changes or small screenshots for altered simulation outputs, but do not commit generated rasters, CSVs, models, checkpoints, or credentials.
