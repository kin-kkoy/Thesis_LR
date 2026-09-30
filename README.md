# Predictive Fire Spread Modelling Using Cellular Automata with Machine Learning

Undergraduate thesis codebase that asks whether a machine-learning model can replace the static transition rules of a cellular automaton (CA) to better predict how an urban fire spreads, using Lapu-Lapu City (Philippines) as the case study. The CA runs over a raster grid of the city built from slope, proximity, building and building-material layers. Because no spatial records of real burns were available, a dataset generator runs the CA under many wind and ignition scenarios to produce synthetic per-cell training data. A classifier trained on that data predicts each cell's ignition probability, and the model is then plugged back into the CA to drive the simulation. The result is checked against a hand-traced perimeter of a real fire (Sitio Santa Maria, 12 December 2023).

"LR" is short for **Logistic Regression**. This repository is the Logistic Regression line of the thesis. The team's Random Forest (RF) work follows the same pipeline in a separate train/validate flow. The default branch, `lr-information-gap-fix`, holds the current LR work; the other branches (`10-Params-SetA`, `10-Params-SetB-WeightedClass`, `RF-11-Params`) hold other feature-set variants.

This is a team thesis project with three contributors.

## Key results

Taken from the master table in [docs/findings.md](docs/findings.md) (2026-05-03). These are spatial-validation metrics comparing each simulated burn area with the hand-traced ground-truth raster of the Sitio Santa Maria fire. The proposal set targets of Recall ≥ 0.80 and F1 ≥ 0.80.

| Variant | F1 | Recall | Precision | AUC-ROC |
|---|:-:|:-:|:-:|:-:|
| CA only (baseline) | 0.684 | 0.544 | 0.921 | 0.772 |
| Stage B: LR + `wind_weighted_score` | 0.607 | 0.925 | 0.452 | 0.963 |
| **Stage B + probability threshold 0.40** | **0.691** | **0.867** | 0.574 | 0.933 |
| Stage C: unified 11-feature LR | 0.622 | 0.910 | 0.473 | 0.955 |

- Every LR variant in the master table meets the recall target. None reaches the F1 target: the best LR result that still meets the recall target is Stage B with a threshold of 0.40, at F1 = 0.691.
- A Random Forest run scored F1 0.785 and Recall 0.963. The findings say this is not a fair comparison, because that RF was trained on data generated before the CA's wind handling was fixed.
- Validation uses a single fire, and the findings list this as a major limitation.

## Repository layout

| Path | Contents |
|---|---|
| `Code/` | The CA engine, dataset generation, LR training scripts, simulation entry point and spatial validation. See [Code/README.md](Code/README.md). |
| `Code/config/default_experiment.yaml` | Simulation settings: raster files, ignition points, wind, CA weights, ML model path and threshold, dataset-generation region. |
| `Code/models/` | Trained model artifacts. Only `fire_rf_model.joblib` (Random Forest) is committed. |
| `Code/sandbox_kent/` | A separate copy of the pipeline used for the Stage A and Stage C experiments. |
| `docs/findings.md` | The full log of experiments and results, including every LR variant, spatial validation and threshold tuning. |
| `docs/loggingActivity.md` | A chronological activity log that also records methodology limitations. |
| `RRLS/` | PDFs of related literature (stored with Git LFS). |
| `BuildingLayout/` | NYC PLUTO 16v2 borough tables and documentation (stored with Git LFS). The current code does not use them. |

## How to run

There is no `requirements.txt`. The code imports `numpy`, `pandas`, `scipy`, `scikit-learn`, `rasterio`, `pyyaml`, `joblib`, `optuna` and `matplotlib`. Run the commands below from inside `Code/`.

**The data is not in the repository.** The input rasters (`processedData/raster/raster/stack_*.tif`) and the training CSVs (`dataFiles/*.csv`) are excluded by `.gitignore`, so you need to supply them separately. The PDF and CSV files that are committed are Git LFS pointers, so you need `git lfs` to fetch them.

```bash
cd Code
python generate_multi_scenario.py   # 200 CA runs -> dataFiles/multi_scenario_dataset.csv
python train_lr.py                  # baseline LR -> models/fire_lr_model.joblib
python train_lr_optuna.py           # LR + Optuna tuning (5-fold CV)
python train_lr_pathb.py            # LR + engineered features + Optuna
python main.py                      # full CA + ML simulation -> output/final_state.tif
python spatial_validation/validate_simulation.py \
    --final output/final_state.tif \
    --gt processedData/raster/raster/stack_ground_truth.tif
```

Before running `main.py`, edit `config/default_experiment.yaml`:

- `ml_model.model_path` defaults to `models/fire_lr_windweighted.joblib`. That file is not committed and none of the scripts writes it, so point this setting at a model you have.
- `ml_model.proba_threshold` sets the inference-time probability cutoff. Setting it to `0.0` turns the cutoff off.

`spatial_validation/visualize_simulation.py --final ... --gt ... --out ...` draws a map comparing the simulation with the ground truth.

For the per-file layout, the feature-schema notes and more tasks, see [Code/README.md](Code/README.md).

## Data sources

As described in the repository:

- **Study-area rasters** for Lapu-Lapu City (slope, proximity, buildings, materials). The grid is 5489 × 6896 cells in EPSG:32651, and the rasters were copied in from the team's Google Drive (see `docs/loggingActivity.md`). The thesis manuscript names OpenStreetMap and NAMRIA as the underlying sources.
- **Ground truth** is `stack_ground_truth.tif`, a burn area hand-traced from Google satellite imagery of the 12 December 2023 Sitio Santa Maria fire.
- **Bureau of Fire Protection (BFP)** supplied historical incident records: 28 text addresses with no spatial data. Because these could not be used as spatial training data, the training data is synthetic and generated by the CA.
- **Wind** in the simulations is a fixed placeholder of 10 km/h from the NW. It was not matched to the actual weather on the day of the fire.
