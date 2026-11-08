# Running Kristian's LR in your phase16 F01 setup

## Quick start (plain English)

**Do I need to install anything?** No. The script only uses scikit-learn, numpy, pandas and joblib, which your Thesis venv already has (the same ones your RF uses).

**What you're doing:** your F01 run, again, with my LR in place of your RF. Same map square, same fire start, same wind, same seed. Only the "brain" deciding which cells catch fire changes.

1. **Unzip** this folder anywhere, e.g. `C:\Users\gibgib\LR_for_Kent`.
2. **Open PowerShell in that folder** and turn on your Thesis venv:
   ```
   cd C:\Users\gibgib\LR_for_Kent
   C:\Users\gibgib\Thesis-Cellular-Automata\.venv\Scripts\Activate.ps1
   ```
3. **Rebuild the model** (one command, takes a few seconds). Point it at your validation CSV:
   ```
   python rebuild_lr.py --check C:\Users\gibgib\Thesis-Cellular-Automata\Thesis_RF\Code\output\candidate-r2-cropped-1024-v1-g364-r1-package\development\validation\data.csv
   ```
   It should print `AP 0.8331  ROC AUC 0.7439 ...` and create `fire_lr_kent_r2_unweighted_sk1.8.0.joblib`. That's the model to use. (Why: I'm on scikit-learn 1.9, you're on 1.8, so this builds the same model in your version.)
4. **Get its fingerprint** for your config:
   ```
   Get-FileHash fire_lr_kent_r2_unweighted_sk1.8.0.joblib -Algorithm SHA256
   ```
5. **Run your 49° F01 again with this model.** Copy your 49° config, change only the model path, the model hash and the run name, and run it the same way you ran the RF. Your orchestrator currently only allows your RF's hash, so it needs a small allowance for this one (details in Step 2 below; your Codex can do it).
6. **Score it with `final_results.py`** like the RF, and send me the confusion matrix, F1, precision, recall and the number of burned cells.

The sections below have the details for each step.

Goal: the same run as your 49° F01 (`phase16_candidate_r2_f01_nasa_hr11_exec_r1.yaml`), with only the model swapped from your RF to my LR. Then LR and RF are compared in the same setup.

## Files

| File | What it is |
|---|---|
| `fire_lr_kent_r2_unweighted_params.json` | The model's numbers: scaler means/scales, coefficients, feature order |
| `rebuild_lr.py` | Rebuilds the identical model inside your scikit-learn version |
| `fire_lr_kent_r2_unweighted.joblib` | The original file, saved with scikit-learn **1.9.0**. Use it only if it loads cleanly in your 1.8.0; otherwise use the rebuilt one |

The model: trained on your **train role only** (227,576 rows), validation role used only for scoring, calibration and final_test never touched. `StandardScaler` + `LogisticRegression(C=1, l2, lbfgs, max_iter=2000, random_state=42)`, `class_weight=None`, **not calibrated**. All 11 features in your canonical order; it exposes `feature_names_in_`, `classes_ = [0, 1]` and `predict_proba`, which your `validate_model_feature_schema` / `positive_class_index` checks need.

## Step 1: rebuild the model in your environment

From this folder, with your Thesis venv:

```
python rebuild_lr.py --check <path>\development\validation\data.csv
```

It should print `AP 0.8331  ROC AUC 0.7439  Brier 0.1921  log loss 0.5643` and write `fire_lr_kent_r2_unweighted_sk1.8.0.joblib`. Use that file for the run, and record its SHA-256.

## Step 2: make an LR copy of your 49° execution config

Copy `Thesis_RF/Code/config/phase16_candidate_r2_f01_nasa_hr11_exec_r1.yaml` to a new file (e.g. `..._lr_exec_r1.yaml`). **Change only:**

- `ml_model.model_path` → the rebuilt LR joblib
- `ml_model.model_sha256` → its SHA-256
- `calibration_manifest_path` / `calibration_manifest_sha256` → not applicable (LR is uncalibrated)
- `run_id`, `scenario_id`, output folder → new names (e.g. `p16-f01-nasa-hr11-lr-s42-r1`) so the RF outputs are not overwritten

**Keep identical:** crop origin `[256, 2176]`, local ignition `[638, 554]`, seed 42, wind 17.7 km/h from `48.825208`, wind_weight 0.20, timers 1/1, max 300 updates, early stop, `inference_mode: stochastic_probability`, all thresholds `null`, same rasters/masks.

**Code note:** `orchestrator.py` (around lines 705–730) only accepts the candidate RF: it checks `model_sha256` against `PHASE15_CANDIDATE_MODEL_SHA256` and requires the calibration manifest next to the model. You'll need a small allowance for the LR hash with no manifest. Nothing else in the CA or scoring should change. This is your code, so make that change however your review process prefers.

## Step 3: score it exactly like the RF

Use `final_results.py` on the same domain (135,992 common-valid building cells, 1,910 reference positives). Please send back:

- confusion matrix `[[TN, FP], [FN, TP]]`, precision, recall, F1, Jaccard
- predicted burned cells, and the update count at which it went inactive

## Step 4 (if you can): five seeds for both

Your 0.852 is one seed. Seeds 42–46 for **both** RF and LR in this setup would give mean ± SD, the same way my results are reported. Even just the LR at seed 42 is useful first.

## For reference: my numbers so far (Sta. Maria, 17.7 km/h from 49°)

| Model | Dataset | CA | Runs | Buildings F1 |
|---|---|---|---|---|
| Your RF | yours | yours | 1 | 0.852 |
| My LR | yours | mine | 5 | 0.650 ± 0.005 |
| My LR | mine | mine | 5 | 0.672 ± 0.017 |
| Circle, same size as your prediction | — | — | — | 0.733 |
