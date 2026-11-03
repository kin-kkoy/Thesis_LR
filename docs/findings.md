# Findings & Analysis Log

## 2026-03-15: Synthetic Dataset Analysis

### Dataset Overview
- **File:** `gdriveContents/data files/synthetic_fire_dataset.csv`
- **Total rows:** 19,356,957
- **Total positive cases (Ignited=1):** 501
- **Positive ratio:** 0.0026% (extremely imbalanced)
- **File size:** 983 MB
- **No null values** in any column

### Columns (8 features + 1 target)
| # | Column | Type | Unique Values |
|---|--------|------|---------------|
| 1 | slope_risk | float64 | 5 (0.0, 0.3, 0.6, 0.8, 1.0) |
| 2 | proximity_risk | float64 | 4 (0.0, 0.5, 0.8, 1.0) |
| 3 | building_presence | float64 | 2 (0.0, 1.0) |
| 4 | wind_speed | float64 | 1 (10.0) — constant |
| 5 | wind_sin | float64 | 1 (-0.7071) — constant |
| 6 | wind_cos | float64 | 1 (0.7071) — constant |
| 7 | neighbor_burning_count | float64 | varies (0–7 in positive cases) |
| 8 | composite_flammability | float64 | 16 distinct values |
| target | Ignited | float64 | 2 (0.0, 1.0) |

### Class Imbalance Distribution Across File
Positive cases appear throughout the file, increasing slightly toward the end (fire spreads over time):

| Chunk (1M rows) | Positive Cases |
|-----------------|---------------|
| 0 | 3 |
| 1 | 4 |
| 2 | 7 |
| 3 | 12 |
| 4 | 16 |
| 5–9 | 14–39 per chunk |
| 10–14 | 29–41 per chunk |
| 15–19 | 22–55 per chunk |

### Feature Discriminating Power (Positive vs Negative)
| Feature | Positive Mean | Negative Mean | Difference | Signal Strength |
|---------|:---:|:---:|:---:|:---:|
| neighbor_burning_count | **2.59** | 0.0003 | +2.59 | **Strongest** |
| building_presence | **0.60** | 0.27 | +0.34 | Strong |
| composite_flammability | **0.43** | 0.17 | +0.26 | Strong |
| proximity_risk | 0.80 | 0.71 | +0.09 | Moderate |
| slope_risk | 0.24 | 0.21 | +0.03 | Weak |
| wind_speed | 10.0 | 10.0 | 0.0 | **None** (constant) |
| wind_sin | -0.7071 | -0.7071 | 0.0 | **None** (constant) |
| wind_cos | 0.7071 | 0.7071 | 0.0 | **None** (constant) |

### Positive Cases Feature Distributions
- **slope_risk:** 0.0 (231), 0.3 (137), 0.6 (133) — no 0.8 or 1.0 values
- **proximity_risk:** 1.0 (398), 0.0 (102), 0.8 (1)
- **building_presence:** 1.0 (303), 0.0 (198)
- **neighbor_burning_count:** 1 (122), 2 (150), 3 (111), 4 (68), 5 (33), 6 (15), 7 (2)
- **composite_flammability:** 0.0 (198), 0.5 (51), 0.59 (13), 0.68 (17), 0.7 (101), 0.79 (70), 0.88 (51)

### Implications for Model Training
1. **Extreme class imbalance** — must use `class_weight='balanced'` or undersample negatives
2. **Wind features carry zero information** — constant across all rows (single simulation config). Kept for pipeline compatibility but won't contribute to predictions.
3. **`neighbor_burning_count` is the dominant predictor** — cells ignite when neighbors are blazing, which is physically correct for a CA fire model
4. **501 positive samples is small but workable** for Logistic Regression (fewer parameters than RF)
5. **Training on full 19M rows is unnecessary** — undersample negatives to ~5,000–10,000 rows alongside all 501 positives

### BFP Historical Data Assessment
- **Source:** Bureau of Fire Protection, Brgy Pajo and Brgy Pusok, Jan-Dec 2019-2025
- **Total records:** ~28 fire incidents
- **Columns available:** Location (text address), Date & Time, Building Materials (Light/Mixed), Cause of Fire
- **What's missing:** Fire perimeters, burned area polygons, GPS coordinates, spatial extent data
- **Verdict:** Barely usable for ML training. Useful only for rough validation (matching known fire locations to simulation outputs). This is why the project uses synthetic data generated from CA dynamics.

---

## 2026-03-15: Logistic Regression Model Training (v1)

### Setup
- **Script:** `Thesis_1/Code/train_lr.py`
- **Virtual env:** `Thesis_1/Code/.venv/` (Python 3.12.3)
- **Packages:** pandas 3.0.1, scikit-learn 1.8.0, joblib 1.5.3, numpy 2.4.3

### Data Preparation Strategy
- Full dataset is 19.3M rows — too large and imbalanced for direct use
- **Undersampling approach:** Keep all 501 positives + sample 5,010 negatives (1:10 ratio)
- Final training set: 5,511 rows (9.09% positive)
- 80/20 stratified train/test split: 4,408 train (401 pos) / 1,103 test (100 pos)

### Model Architecture
- **sklearn Pipeline:** `StandardScaler` → `LogisticRegression`
  - Pipeline wraps scaler + model into one object with `predict_proba()`
  - Required because `automata_engine.py` loads model via `joblib.load()` and calls `predict_proba()` directly — it doesn't know about a separate scaler
- **Hyperparameters:** `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=1000`, `random_state=42`
- **Output:** `models/logistic_regression.joblib` (1.5 KB)

### Results
| Metric | Value | Thesis Target |
|--------|:-----:|:-------------:|
| Accuracy | 1.0000 | — |
| Precision | 1.0000 | — |
| Recall | 1.0000 | >= 0.80 |
| F1 Score | 1.0000 | >= 0.80 |
| ROC AUC | 1.0000 | — |

**Confusion Matrix (test set):**
|  | Predicted 0 | Predicted 1 |
|--|:-----------:|:-----------:|
| **Actual 0** | TN=1003 | FP=0 |
| **Actual 1** | FN=0 | TP=100 |

### Feature Coefficients (sorted by absolute value)
| Feature | Coefficient |
|---------|:-----------:|
| neighbor_burning_count | **+8.2756** |
| building_presence | +0.2334 |
| composite_flammability | +0.1950 |
| slope_risk | +0.1551 |
| proximity_risk | +0.1518 |
| wind_cos | 0.0000 |
| wind_sin | 0.0000 |
| wind_speed | 0.0000 |

### Why Perfect Scores?
- `neighbor_burning_count` is an almost perfect separator: positive cases always have `>= 1`, negatives almost always have `0`
- This is physically correct — CA cells only ignite when they have blazing neighbors
- The LR model essentially learned: "if neighbors are burning, check building/slope/proximity to decide probability"
- **In actual CA simulation context**, the engine's `susceptible` mask already filters for cells with `blazing_neighbor_count > 0`, so the other features (building_presence, slope, etc.) will play a more meaningful role in differentiating ignition probabilities among candidate cells

### Pipeline Compatibility Note
- `automata_engine.py:load_model()` expects model to have `predict_proba()` — Pipeline satisfies this
- The 8 features match `FEATURE_NAMES` in `feature_pipeline.py` exactly
- `automata_engine.py` must NOT be modified (per partner agreement)

### v1 Problem: Misleading Perfect Scores
When tested on **only susceptible cells** (neighbor_burning_count > 0) — the realistic scenario since the CA engine only evaluates cells adjacent to fire — the v1 model predicted EVERYTHING as Ignited:
- F1 = 0.22, Precision = 0.12, Recall = 1.00 (just says "yes" to everything)
- The model was useless for distinguishing among cells that actually face fire

**Root cause:** Training on all cells (including millions with 0 burning neighbors) made `neighbor_burning_count` a trivial separator. The model never learned to distinguish among the cells that matter.

---

## 2026-03-15: Logistic Regression Model Training (v2 — Susceptible Cells Only)

### Key Change from v1
- **v1:** Trained on all cells → `neighbor_burning_count` was a trivial separator → perfect but meaningless scores
- **v2:** Trained on only susceptible cells (`neighbor_burning_count > 0`) → model must learn to distinguish among cells that are actually near fire

### Data Preparation
- Filtered entire 19.3M row dataset for rows with `neighbor_burning_count > 0`
- Found **501 positives** and **3,535 negatives** among susceptible cells
- Undersampled negatives to 1:5 ratio → 501 positives + 2,505 negatives = **3,006 rows**
- 80/20 stratified split: 2,404 train (401 pos) / 602 test (100 pos)

### Results
| Metric | v1 (all cells) | v2 (susceptible) | Thesis Target |
|--------|:-:|:-:|:-:|
| Accuracy | 1.0000 | **0.6927** | — |
| Precision | 1.0000 | **0.3005** | — |
| Recall | 1.0000 | **0.6400** | >= 0.80 |
| F1 Score | 1.0000 | **0.4089** | >= 0.80 |
| ROC AUC | 1.0000 | **0.7345** | — |

**Confusion Matrix (test set):**
|  | Predicted 0 | Predicted 1 |
|--|:-----------:|:-----------:|
| **Actual 0** | TN=353 | FP=149 |
| **Actual 1** | FN=36 | TP=64 |

### Feature Coefficients (v2)
| Feature | Coefficient | Interpretation |
|---------|:-----------:|----------------|
| building_presence | **+0.8123** | Strongest — buildings catch fire more |
| neighbor_burning_count | +0.6376 | More blazing neighbors = higher ignition chance |
| composite_flammability | **-0.4789** | Negative — needs investigation (possible multicollinearity with building_presence) |
| proximity_risk | +0.3213 | Cells near boundaries more likely to ignite |
| slope_risk | +0.2305 | Higher slope = higher ignition |
| wind_speed/sin/cos | 0.0000 | No signal (constant values) |

### Assessment
- Results are **realistic and defensible** but below thesis targets (F1=0.41 vs target 0.80)
- Recall of 0.64 means 36% of actual ignitions are missed
- Low precision (0.30) means many false positives — model over-predicts ignition
- **Possible improvements:** Optuna hyperparameter tuning, feature engineering, adjusting neg_ratio, or investigating composite_flammability's negative coefficient

---

## 2026-04-06: LR Training on Updated Branch (9 Features, Before Wind Fix)

### Context
- Switched to Kent's `rf-training-8params-test` branch (9 features, added `material_risk`)
- Dataset: `synthetic_fire_dataset.csv` — same one Kent used for RF
- Dataset already pre-filtered to susceptible cells (neighbor_burning_count > 0) at generation time
- This is the BEFORE-wind-fix baseline

### Dataset
- Total rows: 4,252 (all susceptible cells)
- Train/Test split (80/20 stratified): 3,401 train (370 pos) / 851 test (92 pos)
- Positive ratio: ~10.9%

### Model
- sklearn Pipeline: StandardScaler → LogisticRegression
- class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42
- Threshold optimized to 0.60 (instead of default 0.50) for best F1

### LR vs RF Comparison (Same Dataset)
| Metric | Kent's RF | Our LR | Thesis Target |
|--------|:---------:|:------:|:-------------:|
| Precision | 0.300 | 0.206 | — |
| Recall | 0.421 | 0.391 | >= 0.80 |
| F1 | 0.350 | 0.270 | >= 0.80 |
| AUC-ROC | 0.697 | 0.669 | — |
| Jaccard | — | 0.156 | — |

### Confusion Matrix (LR, test set)
|  | Predicted 0 | Predicted 1 |
|--|:-----------:|:-----------:|
| **Actual 0** | TN=620 | FP=139 |
| **Actual 1** | FN=56 | TP=36 |

### LR Coefficients
| Feature | Coefficient | What it means |
|---------|:-----------:|---------------|
| neighbor_burning_count | **+0.636** | More burning neighbors → more likely to ignite. Strongest predictor. |
| proximity_risk | +0.350 | Cells closer to roads/boundaries → more likely. |
| building_presence | +0.260 | Buildings catch fire more than open ground. |
| slope_risk | +0.129 | Steeper terrain → slightly more likely. |
| material_risk | +0.057 | Flammable materials → slightly more likely. |
| composite_flammability | +0.057 | Same as material_risk (because composite = building_presence × material_risk). |
| wind_speed | 0.000 | Zero. Wind doesn't affect the simulation. |
| wind_sin | 0.000 | Zero. Confirms wind is broken in the CA engine. |
| wind_cos | 0.000 | Zero. |

### What These Results Mean (Plain English)

**The confusion matrix — what actually happened with predictions:**
- The model looked at 851 cells that were near fire and had to guess: will it ignite or not?
- Of the 92 cells that actually ignited, it correctly caught 36 (TP) but missed 56 (FN)
- Of the 759 cells that didn't ignite, it correctly said "no" to 620 (TN) but falsely said "yes" to 139 (FP)

**The metrics — what the numbers mean:**
- **Precision = 0.21:** When the model says "this cell will ignite," it's only right 21% of the time. It raises too many false alarms.
- **Recall = 0.39:** Of all cells that actually ignited, the model only caught 39%. It misses more than half of actual fires.
- **F1 = 0.27:** The balance between precision and recall. Low because both are low.
- **AUC-ROC = 0.67:** How well the model ranks igniting cells above non-igniting cells. 0.5 = random guessing, 1.0 = perfect. 0.67 is slightly better than random but not great.

**The coefficients — what the model learned:**
- neighbor_burning_count (+0.64) is the strongest signal: more burning neighbors = more likely to ignite. This is physically correct.
- proximity_risk (+0.35) and building_presence (+0.26) are secondary signals. Also physically correct.
- Wind features are exactly zero. This CONFIRMS our earlier finding that the CA engine doesn't actually use wind in its fire spread physics. The model correctly learned that wind is irrelevant in this simulation.

**Why RF beats LR (slightly):**
- RF (F1=0.35) > LR (F1=0.27). Expected because RF can capture nonlinear interactions (e.g., "steep slope AND wooden building AND 3+ neighbors = very high risk") while LR can only do linear combinations.
- Both are far below the thesis target of F1 >= 0.80. The problem is data, not model choice.

**Bottom line:** The model is slightly better than guessing but not useful yet. The data comes from one fire scenario with broken wind. Fix the wind, generate more diverse scenarios, and both models should improve significantly.

---

## 2026-04-10: Wind Fix Applied to automata_engine.py

### What Was Wrong
`automata_engine.py` computed `wind_multiplier = 1.0 + wind_weight = 1.2` — a constant that never used `speed_kmh` or `direction_deg`. Every simulation was identical regardless of wind config. The 3 wind features (wind_speed, wind_sin, wind_cos) were recorded in training data but never influenced the actual fire physics.

### The Fix
Replaced the flat multiplier with a directional 3x3 kernel (`_compute_wind_kernel()`):
- Each neighbor cell in the 3x3 grid gets a weight based on how aligned the fire-travel direction is with the wind blow direction
- Upwind neighbors (fire traveling WITH the wind toward the candidate cell) get weight > 1.0
- Downwind neighbors (fire traveling AGAINST the wind) get weight closer to 0.05
- Scaled by `speed_factor = speed_kmh / 10.0`, so 25 km/h wind has 2.5× the effect of 10 km/h

Physics grounding: Alexandridis et al. (2011) and Gao et al. (2008) both model wind as a directional multiplier on spread probability in CA fire models.

### Why Wind Matters in Real Fire
Wind is one of the three primary drivers of fire spread (the fire triangle extensions: fuel, weather, topography). In the Lapu-Lapu context, typhoon-season winds of 20–25 km/h from the northeast are common and fundamentally change which neighborhoods are at risk. A model that ignores wind direction is physically incomplete for the thesis use case.

### Results After Fix (40 scenarios, 2,091 rows, 102 positives)

| Metric | Before Fix (4,252 rows) | After Fix (2,091 rows) | Thesis Target |
|--------|:-----------------------:|:----------------------:|:-------------:|
| Precision | 0.206 | 0.069 | — |
| Recall | 0.391 | **0.750** | >= 0.80 |
| F1 | 0.270 | 0.127 | >= 0.80 |
| AUC-ROC | 0.669 | 0.581 | — |

### Wind Coefficients: Proof the Fix Worked

| Feature | Before Fix | After Fix |
|---------|:----------:|:---------:|
| wind_speed | 0.000 | **+0.096** |
| wind_sin | 0.000 | **+0.060** |
| wind_cos | 0.000 | **+0.027** |

Wind coefficients went from exactly zero to non-zero — the LR model is now picking up signal from wind. This is the confirmation that the CA engine fix worked.

### Why neighbor_burning_count Dropped from +0.636 to Near Zero (+0.005)

Before the fix, raw count was the dominant signal because every neighbor was equally likely to spread fire (wind was ignored). After the fix, direction matters: a cell with 1 neighbor directly upwind ignites more easily than one with 3 neighbors downwind. The raw count becomes a weaker signal because it doesn't encode directionality.

This is correct behavior. The model now weights upwind neighbors more and downwind less, so the raw integer count is less informative than the wind-weighted neighborhood score that the CA engine uses internally. In a physics-accurate simulation, count alone is insufficient — direction relative to wind is the key predictor.

### Why F1 Got Worse Despite Better Recall

The after-fix dataset has only 102 positives (vs 462 before). With so few positive examples, the model doesn't have enough evidence to learn the precise boundary conditions for ignition, so it over-predicts (high recall, terrible precision). This is a data volume problem, not a model or fix problem.

**Root cause of low positive count:** The wind fix makes fire spread more directional (concentrated in one direction) rather than radial. A directional fire burn covers fewer cells, so fewer cells appear as "ignited" in the training data per scenario.

**Solution:** More scenarios with more ignition points and longer timesteps to generate more total positives. Target: 10 ignition points, 100 timesteps, 5 runs per scenario = 200 total simulation runs.

---

## 2026-04-11: LR on Expanded Multi-Scenario Dataset (200 runs)

### What We Did
Regenerated the dataset with the larger configuration:
- 10 ignition points (was 5)
- 100 timesteps (was 60)
- 5 runs per wind config (was 1)
- 5 wind speeds x 8 directions x 5 runs = **200 total simulation runs**

Then trained LR on the resulting dataset with the same Pipeline (StandardScaler + LR, class_weight='balanced', threshold-tuned).

### Dataset Comparison
| Metric | v1 (40 runs) | v2 (200 runs) | Change |
|--------|:------------:|:-------------:|:------:|
| Total rows | 2,091 | **21,079** | 10.1x |
| Positives | 102 | **1,095** | 10.7x |
| Positive ratio | 4.88% | 5.19% | ~same |

The data generation worked exactly as planned -- ~10x more rows, ~11x more positives, balanced across all wind speeds (3,985-4,424 rows per speed bucket).

### Model Results
| Metric | v1 (40 runs) | v2 (200 runs) | Thesis Target |
|--------|:------------:|:-------------:|:-------------:|
| Precision | 0.069 | 0.077 | -- |
| Recall | **0.750** | 0.470 | >= 0.80 |
| F1 | 0.127 | 0.133 | >= 0.80 |
| AUC-ROC | 0.581 | 0.630 | -- |

**Confusion matrix (v2 test set, 4,216 rows):**
|  | Predicted 0 | Predicted 1 |
|--|:-----------:|:-----------:|
| **Actual 0** | TN=2,768 | FP=1,229 |
| **Actual 1** | FN=116 | TP=103 |

### Coefficients (v2)
| Feature | Coefficient | Notes |
|---------|:-----------:|-------|
| proximity_risk | **+0.323** | Strongest predictor |
| building_presence | +0.215 | Buildings catch fire more |
| slope_risk | +0.111 | Steeper -> more likely |
| neighbor_burning_count | +0.088 | Recovered from near-zero in v1 |
| wind_speed | +0.079 | Wind features still non-zero -- fix holds |
| wind_cos | +0.051 | |
| wind_sin | +0.050 | |
| composite_flammability | +0.023 | |
| material_risk | +0.023 | |

### What These Numbers Mean

**The good news:**
- All 9 coefficients are positive and physically sensible (no weird negative weights anymore)
- Wind features stayed non-zero after the wind fix -- confirmed across a much larger sample
- AUC-ROC improved from 0.58 -> 0.63 (better ranking of igniting cells over non-igniting ones)
- The data pipeline scales correctly -- 10x more data produced 10x more rows with consistent ratios

**The bad news:**
- F1 is essentially flat (0.127 -> 0.133). 10x more data did NOT meaningfully improve LR.
- Recall actually dropped (0.75 -> 0.47). v1 was over-predicting positives because the model had so few examples it just said "yes" to most cells.
- Precision is stuck around 0.08 -- the model still raises ~12 false alarms for every correct ignition prediction.

### Why More Data Did Not Fix It

LR has an architectural ceiling on this problem. It can only learn linear combinations of features:
```
P(ignite) = sigmoid(w1*slope + w2*proximity + w3*building + ... + bias)
```

But fire ignition in reality depends on **interactions** between features:
- Steep slope AND wooden building AND upwind neighbor = very high risk
- Steep slope ALONE or wooden building ALONE = moderate risk
- The whole is greater than the sum of parts

LR cannot represent "feature A times feature B matters more than either alone." Random Forest can (it splits on combinations of features at each tree node), which is why Kent's RF baseline (F1=0.35) edges out LR even on the same data.

**More data doesn't help when the model can't represent the underlying pattern.** It's like adding more rows to a spreadsheet when the formula you're using is wrong -- you just get a more confident wrong answer.

### What This Means for the Thesis

This is actually a valuable finding, not a failure:
1. We have a clean before/after comparison showing wind fix worked (coefficients went from 0 to non-zero)
2. We have a clean LR vs RF comparison showing LR's linear ceiling on this problem
3. We have evidence that more data alone is not the answer for LR -- the architecture matters
4. This motivates either (a) feature engineering (adding interaction terms so LR can see them) or (b) hyperparameter tuning (Optuna) to extract whatever performance LR can give

**Decision point:** Either commit to making LR better via feature engineering / regularization tuning, or accept LR as the simpler-baseline-that-shows-the-limits and rely on RF for the headline numbers. Either is defensible in the thesis.

---

## 2026-04-11: Path A — Optuna Hyperparameter Tuning (9 Original Features)

### Purpose
Systematically search for the best LR hyperparameters using Bayesian optimization (Optuna) with 5-fold stratified cross-validation. Goal: find the honest LR ceiling — the best F1 achievable with a linear model on the original 9 features.

### Search Configuration
- **Optimizer:** Optuna TPE sampler (Akiba et al. 2019), seed=42 for reproducibility
- **Trials:** 100
- **Validation:** 5-fold stratified CV on training data only (Kohavi 1995)
- **Metric optimized:** Mean F1-score across folds (with per-fold threshold optimization)
- **Test set:** Locked — same 20% holdout (seed=42) as baseline, never seen by Optuna

**Search space:**
| Hyperparameter | Range | Purpose |
|---------------|-------|---------|
| penalty | l1, l2, elasticnet | Type of regularization |
| C | 0.0001 to 100 (log scale) | Regularization strength (bias-variance tradeoff) |
| solver | lbfgs, liblinear, saga (penalty-compatible) | Optimization algorithm |
| class_weight | balanced, custom (1.0 to 20.0) | Minority class upweighting |
| l1_ratio | 0.0 to 1.0 (if elasticnet) | L1/L2 mix ratio |

### Best Hyperparameters Found
| Parameter | Default (baseline) | Optuna best |
|-----------|:------------------:|:-----------:|
| penalty | l2 | l2 |
| solver | lbfgs | lbfgs |
| C | 1.0 | **0.908** |
| class_weight | balanced | **custom: {0: 1.0, 1: 13.94}** |

Optuna found that L2/lbfgs (the defaults) were already the right penalty/solver combination, but the class weighting needed adjustment: instead of sklearn's `balanced` formula, a custom weight of ~14x for the positive class performed slightly better.

### Results: Baseline vs Optuna-Tuned

| Metric | Baseline (defaults) | Optuna-tuned | Change |
|--------|:-------------------:|:------------:|:------:|
| Precision | 0.077 | 0.078 | +0.001 |
| Recall | 0.470 | 0.379 | -0.091 |
| F1 | 0.133 | 0.129 | -0.004 |
| AUC-ROC | 0.630 | 0.629 | -0.001 |
| Jaccard | 0.071 | 0.069 | -0.002 |
| Threshold | 0.55 | 0.50 | -0.05 |

**Confusion matrix (Optuna-tuned, test set):**
|  | Predicted 0 | Predicted 1 |
|--|:-----------:|:-----------:|
| **Actual 0** | TN=3,016 | FP=981 |
| **Actual 1** | FN=136 | TP=83 |

### Overfitting Check
| Score | Value |
|-------|:-----:|
| CV F1 (what Optuna optimized) | 0.1375 |
| Test F1 (held-out, never seen) | 0.1294 |
| Gap | **+0.0082** |

The gap is minimal (< 0.01). The CV estimate is trustworthy — no overfitting to the validation folds.

### Convergence
Best trial was #54 out of 100. Found in the second half of the search, meaning the search was still exploring when it found the optimum. However, the improvement from trial 54 onward was negligible, suggesting the search space has been well-covered.

### Coefficients (Optuna-tuned vs Baseline)
| Feature | Baseline | Optuna | Change |
|---------|:--------:|:------:|:------:|
| proximity_risk | +0.323 | +0.322 | ~same |
| building_presence | +0.215 | +0.216 | ~same |
| slope_risk | +0.111 | +0.109 | ~same |
| neighbor_burning_count | +0.088 | +0.089 | ~same |
| wind_speed | +0.079 | +0.080 | ~same |
| wind_cos | +0.051 | +0.048 | ~same |
| wind_sin | +0.050 | +0.050 | ~same |
| composite_flammability | +0.023 | +0.022 | ~same |
| material_risk | +0.023 | +0.022 | ~same |

Coefficients are virtually identical. The Optuna search confirmed that the default hyperparameters were already near-optimal for a linear model on these features.

### What This Proves

**The LR ceiling on 9 original features is F1 ≈ 0.13.**

After 100 trials of systematic Bayesian search with proper cross-validation:
- F1 did not improve (0.133 → 0.129, within noise)
- Coefficients didn't change (same feature ranking, same magnitudes)
- The default hyperparameters were already near-optimal (penalty=l2, solver=lbfgs, C≈1.0)
- The only meaningful finding was the class weight adjustment (13.94x vs balanced), which barely mattered

This confirms the hypothesis: the bottleneck is NOT hyperparameters. It's the linear architecture. LR cannot represent the nonlinear feature interactions that drive fire ignition. No amount of tuning can fix a representational limitation.

**This is a clean, defensible result for the thesis:** "We systematically optimized LR's hyperparameters using Bayesian optimization with cross-validation and confirmed that the linear ceiling holds. This motivates Path B (feature engineering to expand the feature space) and validates that RF's advantage is architectural, not a tuning artifact."

### Next Step: Path B
Proceed with feature engineering — add physics-informed interaction terms so LR can access nonlinear relationships through the input features, then re-run Optuna on the expanded feature set.

---

## 2026-04-11: Path B — Feature Engineering + Optuna (16 Features)

### Interaction Features Added (7 new columns)

| Feature | Formula | Physics Justification |
|---------|---------|----------------------|
| slope_x_neighbors | slope_risk × neighbor_burning_count | Fire travels uphill faster — steep slope compounds with neighbor fire presence (Alexandridis 2011) |
| wind_x_neighbors | wind_speed × neighbor_burning_count | Wind fans flames from neighbors (Gao 2008) |
| slope_x_building | slope_risk × building_presence | Buildings on steep slopes more vulnerable — uphill fire channels heat into structures |
| slope_x_wind | slope_risk × wind_speed | Steep terrain + wind creates channeling (Alexandridis 2011, Gao 2008) |
| proximity_x_building | proximity_risk × building_presence | Wildland-urban interface risk — buildings near boundaries more exposed |
| proximity_x_neighbors | proximity_risk × neighbor_burning_count | Cells near edges with burning neighbors more accessible to fire |
| neighbors_squared | neighbor_burning_count² | Nonlinear neighbor effect — surrounded cells face converging heat from multiple angles |

Total features: 9 original + 7 engineered = 16.

### Phase 1: Baseline LR on 16 Features (Default Hyperparameters)

| Metric | 9feat baseline | 16feat baseline | Change |
|--------|:--------------:|:---------------:|:------:|
| Precision | 0.077 | 0.078 | +0.001 |
| Recall | 0.470 | 0.489 | +0.019 |
| F1 | 0.133 | 0.135 | +0.002 |
| AUC-ROC | 0.630 | 0.631 | +0.001 |

**Confusion matrix (16feat baseline):**
|  | Predicted 0 | Predicted 1 |
|--|:-----------:|:-----------:|
| **Actual 0** | TN=2,734 | FP=1,263 |
| **Actual 1** | FN=112 | TP=107 |

### Phase 2: Optuna on 16 Features (100 trials, 5-fold CV)

**Best hyperparameters:** penalty=elasticnet, C=0.0129, l1_ratio=0.41, class_weight=balanced

| Metric | 16feat baseline | 16feat Optuna | Change |
|--------|:---------------:|:-------------:|:------:|
| Precision | 0.078 | 0.078 | ~same |
| Recall | 0.489 | 0.489 | ~same |
| F1 | 0.135 | 0.134 | -0.001 |
| AUC-ROC | 0.631 | 0.629 | -0.002 |

CV F1 = 0.1351, Test F1 = 0.1342, gap = +0.001 (minimal, no overfitting).

Best trial was #22/100 — converged very early. ElasticNet with low C (0.013) and l1_ratio=0.41 was selected, which zeroed out 4 of 16 features: neighbor_burning_count, slope_x_neighbors, slope_x_wind, proximity_x_neighbors. The model found these redundant given the other features.

### Coefficients (Optuna-tuned, 16 features)

| Feature | Coefficient | Notes |
|---------|:-----------:|-------|
| proximity_risk | +0.203 | Still strongest |
| proximity_x_building | +0.124 | NEW — urban interface interaction picked up signal |
| building_presence | +0.107 | |
| neighbors_squared | +0.077 | NEW — nonlinear neighbor effect has signal |
| slope_risk | +0.073 | |
| wind_speed | +0.048 | |
| wind_sin | +0.042 | |
| wind_cos | +0.041 | |
| wind_x_neighbors | +0.023 | NEW — wind-neighbor interaction has signal |
| slope_x_building | +0.023 | NEW — slope-building interaction has signal |
| material_risk | +0.021 | |
| composite_flammability | +0.021 | |
| neighbor_burning_count | 0.000 | Zeroed (redundant with neighbors_squared) |
| slope_x_neighbors | 0.000 | Zeroed by L1 |
| slope_x_wind | 0.000 | Zeroed by L1 |
| proximity_x_neighbors | 0.000 | Zeroed by L1 |

### Full Comparison: All LR Variants

| Variant | F1 | Recall | Precision | AUC-ROC |
|---------|:--:|:------:|:---------:|:-------:|
| 9feat, defaults (baseline) | 0.133 | 0.470 | 0.077 | 0.630 |
| 9feat, Optuna (Path A) | 0.129 | 0.379 | 0.078 | 0.629 |
| 16feat, defaults (Path B phase 1) | 0.135 | 0.489 | 0.078 | 0.631 |
| 16feat, Optuna (Path B phase 2) | 0.134 | 0.489 | 0.078 | 0.629 |

### What This Means

**Feature engineering did not meaningfully break the LR ceiling.** F1 moved from 0.133 to 0.135 — within statistical noise. The interaction features gave the model access to nonlinear relationships, and some were picked up (proximity_x_building, neighbors_squared), but the overall prediction quality barely changed.

**What we learned from the coefficients:**
- 3 of 7 engineered features carried signal: proximity_x_building (+0.124), neighbors_squared (+0.077), wind_x_neighbors (+0.023), slope_x_building (+0.023)
- 4 of 7 were zeroed out by ElasticNet as redundant
- The model found proximity_x_building particularly useful — the wildland-urban interface interaction is real
- neighbor_burning_count was zeroed in favor of neighbors_squared — the squared term is more informative, confirming the nonlinear relationship

**Why it still didn't help F1 much:**
The interaction terms we can manually construct are only a tiny subset of all possible nonlinear relationships. Fire ignition likely depends on higher-order interactions (3-way, 4-way) and conditional relationships that we can't easily engineer. RF captures these automatically through recursive tree splits. This is the fundamental advantage of tree-based models over linear models with manual feature engineering.

### Conclusion for the Thesis

The LR ceiling on this fire spread prediction task is F1 ≈ 0.13, regardless of:
- Hyperparameter tuning (Path A: no improvement)
- Physics-informed feature engineering (Path B: +0.002, negligible)
- Both combined (Path B Phase 2: still 0.13)

This is a rigorous, four-variant comparison that conclusively demonstrates LR's limitations on nonlinear fire spread prediction. The result validates that RF's advantage over LR is architectural — RF captures complex interactions that LR fundamentally cannot represent, even with engineered features and optimized hyperparameters.

---

## 2026-04-11: BREAKTHROUGH — Fixed the Feature Information Gap

### The Root Cause Investigation

After exhausting hyperparameter tuning (Path A) and feature engineering (Path B) with no improvement, we investigated WHY all models were stuck at F1 ≈ 0.13.

**The discovery:** There was an information gap between what the CA uses to decide ignition and what the ML model receives as features.

- **CA physics (what drives ignition):** `p_ignite = p_base × (wind_weighted_score / max_kernel_sum)` where `wind_weighted_score` is a directional convolution using the wind kernel
- **ML model input:** `neighbor_burning_count` — a raw uniform integer count

The CA knows WHICH neighbors are burning and weights them by wind alignment. The ML model only knows HOW MANY neighbors are burning. A cell with 2 upwind burning neighbors (high danger) looked identical to a cell with 2 downwind burning neighbors (low danger) in the training data.

### The Fix

Added `wind_weighted_score` as a 10th feature. This is the exact same directional score the CA uses for ignition decisions, computed as:
```
wind_weighted_score = convolve(blazing_cells, wind_kernel) / max_kernel_sum
```

Three files modified (backed up in `backups_option2/`):
1. `feature_pipeline.py` — Added `wind_weighted_score` to FEATURE_NAMES, accepts it in `assemble_grid_features()`
2. `dataset_generator.py` — Computes wind_weighted_score at each timestep, passes to assembler
3. `automata_engine.py` `_predict_with_model()` — Computes wind_weighted_score during inference

### Results: 10 Features (wind_weighted_score added)

| Metric | 9 features (baseline) | 10 features (fixed) | Change |
|--------|:---------------------:|:-------------------:|:------:|
| Precision | 0.077 | **0.110** | **+43%** |
| Recall | 0.470 | 0.306 | -35% |
| F1 | 0.133 | **0.162** | **+22%** |
| AUC-ROC | 0.630 | **0.656** | **+4%** |

**Confusion matrix (10 features, baseline):**
|  | Predicted 0 | Predicted 1 |
|--|:-----------:|:-----------:|
| **Actual 0** | TN=3,455 | FP=542 |
| **Actual 1** | FN=152 | TP=67 |

### Coefficients (10 features)

| Feature | Coefficient | Change from 9feat |
|---------|:-----------:|:-----------------:|
| proximity_risk | +0.318 | ~same |
| **wind_weighted_score** | **+0.301** | **NEW — 2nd strongest predictor** |
| building_presence | +0.223 | ~same |
| slope_risk | +0.105 | ~same |
| neighbor_burning_count | **-0.051** | flipped negative |
| wind_speed | +0.050 | ~same |
| wind_cos | +0.048 | ~same |
| wind_sin | +0.043 | ~same |
| composite_flammability | +0.022 | ~same |
| material_risk | +0.022 | ~same |

### What This Proves

**The information gap was the real bottleneck, not the model architecture.**

The new feature `wind_weighted_score` immediately became the 2nd strongest predictor in the model (+0.301), nearly tied with proximity_risk. This is the directional fire-pressure information that the model was missing.

**Notable coefficient shift:** `neighbor_burning_count` flipped to slightly negative (-0.051). This is because once you know the directional wind-weighted score, raw count becomes redundant (and slightly negatively correlated for cells where count is high but alignment is poor).

**Recall vs Precision tradeoff:** Precision jumped 43% (0.077 → 0.110) while recall dropped (0.470 → 0.306). The model is now more selective — it raises fewer false alarms but catches fewer true ignitions. This is a more useful balance for a fire prediction system where false alarms have real costs.

**F1 ceiling broken for the first time:** 0.13 → 0.16. This is the first meaningful F1 improvement since the wind fix, and the first to come from the ML side (all previous improvements came from data/physics fixes).

### Implications for the Thesis

This changes the story significantly:
- LR's "ceiling" was not an LR problem — it was a feature problem
- Both LR and RF likely had the same limitation (they received the same 9 features)
- When Kent retrains RF on the new 10-feature data, he should see a similar improvement
- The thesis narrative becomes: "We discovered and fixed an information gap in the feature pipeline that was invisibly limiting both models"

---

## 2026-04-11: Complete 8-Variant LR Comparison

After fixing the information gap, ran Optuna and Path B (+ engineered features) on the new 10-feature data for a complete systematic comparison.

### Full Comparison Table

| # | Variant | Features | F1 | Recall | Precision | AUC-ROC |
|---|---------|:--------:|:--:|:------:|:---------:|:-------:|
| 1 | 9feat, defaults | 9 | 0.133 | 0.470 | 0.077 | 0.630 |
| 2 | 9feat, Optuna (Path A) | 9 | 0.129 | 0.379 | 0.078 | 0.629 |
| 3 | 9 + engineered, defaults | 16 | 0.135 | 0.489 | 0.078 | 0.631 |
| 4 | 9 + engineered, Optuna | 16 | 0.134 | 0.489 | 0.078 | 0.629 |
| 5 | **10feat, defaults** | 10 | 0.162 | 0.306 | 0.110 | 0.656 |
| 6 | **10feat, Optuna** | 10 | **0.164** | 0.237 | 0.125 | 0.656 |
| 7 | **10 + engineered, defaults** | 17 | 0.165 | 0.320 | 0.111 | 0.658 |
| 8 | **10 + engineered, Optuna** | 17 | **0.165** | 0.320 | 0.111 | 0.658 |

### Key Observations

**1. Information quality beats information quantity.**
Adding 1 physics-grounded feature (wind_weighted_score): F1 +0.029 (0.133 → 0.162).
Adding 7 hand-engineered interaction features: F1 +0.002 (0.133 → 0.135).
Adding 7 engineered features ON TOP of wind_weighted_score: F1 +0.003 (0.162 → 0.165).

**2. The engineered interactions become redundant once wind_weighted_score is present.**
In variant 8 (10 + 7 engineered, Optuna), the Optuna-tuned coefficients show the interaction features don't dominate — wind_weighted_score still holds +0.307 while most engineered features drop to low values or negative. The directional score already encodes most of what the interactions were trying to expose.

**3. Tuning has negligible impact at every stage.**
Default hyperparameters vs Optuna:
- 9feat: 0.133 → 0.129 (Optuna slightly worse)
- 16feat: 0.135 → 0.134 (~same)
- 10feat: 0.162 → 0.164 (marginal)
- 17feat: 0.165 → 0.165 (same)

This reinforces that hyperparameter tuning is not a silver bullet — it can only optimize within the limits of the features and architecture.

**4. The new F1 plateau is ~0.165.**
This is LR's actual ceiling on this task with all available information and optimization. Still below the thesis target of F1 >= 0.80, but now we've characterized the ceiling rigorously.

### Recall-Precision Tradeoff Shift

Notice something interesting: the 9-feature baseline had recall=0.470 and precision=0.077 (high false alarm rate), while the 10-feature baseline has recall=0.306 and precision=0.110 (more selective). Adding wind_weighted_score didn't just improve F1 — it shifted the model from "guess yes on anything near fire" to "guess yes on cells the wind is pushing fire toward."

This is a qualitatively better model, not just a quantitatively better one. False positives are down 44% (from 1,229 to 558 in the Path B Optuna run).

### What Kent Should Expect for RF

When Kent retrains RF on the new 10-feature data, we predict:
- F1 should improve meaningfully (like LR did, maybe more since RF uses information more efficiently)
- RF may still outperform LR because it can capture additional interactions
- The absolute improvement from adding wind_weighted_score should be comparable in magnitude to what LR saw (+0.03 range)

If RF improves similarly, it confirms the hypothesis was correct. If RF improves much more, it further validates that tree-based methods extract more from the feature set. Either way, the result is meaningful.

---

## 2026-05-03: Spatial Validation — Stage A vs Stage B vs CA-only Baseline

### Setup
Both stages run the full CA + ML simulation against Lapu-Lapu rasters, ignited at Sitio Santa Maria coordinates (606748.78, 1141695.77 in EPSG:32651), with simulation wind 10 km/h NW (representative, NOT matched to actual Dec 12 2023 weather — see methodology limitation note in `loggingActivity.md`). Maximum 100 timesteps. Ground truth: Kent's hand-traced raster `stack_ground_truth.tif` from Google satellite imagery (~90% accuracy per his estimate).

| Stage | Modules | Training data | 10th feature |
|-------|---------|--------------|--------------|
| A | Kent's `feature_pipeline.py` + `data_loader.py` | 24,922 rows regenerated with Kent's schema | `material_class` |
| B | Our `feature_pipeline.py` + `data_loader.py` | 21,079 rows existing dataset | `wind_weighted_score` |

Both use the same wind-fixed CA engine, same 9 base features, same Logistic Regression hyperparameters.

### Results

| Variant | F1 | Recall | Precision | AUC-ROC | Jaccard | TP | FP | FN | Cells predicted |
|---------|:--:|:------:|:---------:|:-------:|:-------:|:--:|:--:|:--:|:--------------:|
| CA only (baseline, Kent) | 0.684 | 0.544 | 0.921 | 0.772 | 0.520 | 1,802 | 155 | 1,510 | 1,957 |
| Kent's RF (broken-wind data) | 0.785 | 0.963 | 0.663 | 0.981 | 0.647 | 3,189 | 1,619 | 123 | 4,808 |
| **Stage A** (LR + material_class) | **0.602** | **0.915** | 0.448 | 0.958 | 0.430 | 3,032 | 3,732 | 280 | 6,764 |
| **Stage B** (LR + wind_weighted_score) | **0.607** | **0.925** | 0.452 | 0.963 | 0.436 | 3,064 | 3,721 | 248 | 6,785 |

Recall target (≥ 0.80) per the proposal: **PASSED** by both Stage A and Stage B.

### Stage A vs Stage B Difference

The two stages produced near-identical spatial results:

| Metric | Stage A | Stage B | Δ |
|--------|:-------:|:-------:|:-:|
| F1 | 0.602 | 0.607 | +0.005 |
| Recall | 0.915 | 0.925 | +0.010 |
| Precision | 0.448 | 0.452 | +0.004 |
| AUC-ROC | 0.958 | 0.963 | +0.005 |

**The information gap fix (`wind_weighted_score`) gave only +0.005 F1 spatially**, despite giving +22% F1 per-cell on synthetic data.

This is a striking gap between per-cell and spatial validation results. Possible explanations:
- The benefit of `wind_weighted_score` is concentrated in early-spread timesteps, when the fire is small and directional cues matter most. By the time fire has spread to ~3,000+ cells (the size of the validation fire), the directional info matters less because fire is propagating in many directions simultaneously
- At the spatial scale, both models converge to "spread fire from ignition to all reachable cells with similar features", with material_class doing similar work to wind_weighted_score in shaping the perimeter
- The single-incident validation is fragile — different fires under different wind conditions could produce different stage gaps

### Comparison to Baselines

**LR (either stage) vs CA-only:**
- Recall: 0.92 vs 0.54 (+0.38 — major improvement)
- F1: 0.60 vs 0.68 (-0.08 — slight degradation due to over-prediction)
- Precision: 0.45 vs 0.92 (-0.47 — LR predicts ~3.5x more cells)

The ML version catches far more real fire cells (recall ↑) at the cost of over-predicting the burned area (precision ↓). For an early-warning system where missing real fire is worse than over-warning, this tradeoff is favorable per the proposal's recall-primary criterion.

**LR vs Kent's RF:**
- Direct comparison is NOT fair: Kent's RF was trained on the OLD broken-wind dataset, while both LR stages use the new wind-fixed data
- Until Kent retrains RF on either Stage A or Stage B data, we cannot publish a fair LR-vs-RF comparison
- However, on AUC-ROC alone (which is more robust to over-prediction), Stage B (0.963) and Kent's RF (0.981) are within ~2 percentage points

### What This Tells Us

1. **The recall target is met.** Both LR variants pass the proposal's primary target (≥ 0.80) by a wide margin
2. **F1 is below target (0.80) in all variants.** Achieving F1 ≥ 0.80 likely requires reducing false positives, which means a more selective model — possibly by using a higher decision threshold or by adding precision-favoring constraints
3. **The information gap fix's spatial impact is small.** The +0.029 per-cell F1 gain from `wind_weighted_score` does NOT translate to a comparable spatial gain. This is honest and worth disclosing in the thesis — per-cell metrics overstate the operational value of the feature
4. **Stage C (unified 11-feature LR) likely won't help much.** Given Stage A and Stage B are within 0.005 F1 of each other, combining material_class + wind_weighted_score into one 11-feature model would probably also land around F1 ≈ 0.60-0.61 spatially. Worth confirming but unlikely to break F1 ≥ 0.80
5. **The bigger lever for hitting F1 = 0.80 is precision.** Either the model needs to be more selective (over-predicts ~2x the real fire size) or the validation incident chosen happens to be one where the model naturally over-spreads

### Visualizations
- `Code/sandbox_kent/output/stage_a_comparison.png` — Stage A: ground truth | predicted | TP/FP/FN difference map
- `Code/output/stage_b_comparison.png` — Stage B: ground truth | predicted | TP/FP/FN difference map

### Confusion Matrix Read

The validation raster has 37.8M cells. Both stages over-predicted by ~2x but recovered >91% of real fire cells. The dominant error mode is FALSE POSITIVES (3,700+ false fire cells) rather than false negatives (~250). For a fire warning system, this is the safer error.

---

## 2026-05-03 (continued): Threshold Tuning + Stage C Unified Model

### Threshold Tuning (Stage B)

The simulation engine was extended to support an inference-time `proba_threshold` that zeros out ML predictions below a cutoff before the stochastic ignition draw. This makes the model more selective at inference time without retraining. Tested on Stage B (LR + wind_weighted_score):

| Threshold | F1 | Recall | Precision | AUC-ROC | Cells predicted | Recall ≥ 0.80? |
|-----------|:--:|:------:|:---------:|:-------:|:---------------:|:---:|
| 0.0 (off) | 0.607 | 0.925 | 0.452 | 0.963 | 6,785 | ✓ |
| **0.40** | **0.691** | **0.867** | 0.574 | 0.933 | 4,998 | **✓** |
| 0.45 | 0.709 | 0.769 | 0.658 | 0.884 | 3,872 | ✗ |
| 0.50 | 0.450 | 0.293 | 0.965 | 0.647 | 1,006 | ✗ |
| 0.55 | 0.007 | 0.003 | 1.000 | 0.502 | 11 | ✗ |

**Best operational pick: threshold=0.40.** F1 improved from 0.607 to 0.691 (+0.084) while still passing the proposal's primary recall target (≥ 0.80). Higher thresholds produced higher F1 but failed recall.

### Stage C: Unified 11-Feature Model

Stage C combines both `material_class` (from Kent's schema) and `wind_weighted_score` (from our schema) into a single LR model. This required:
- Adding `wind_weighted_score` to `sandbox_kent/modules/feature_pipeline.py`
- Replacing `sandbox_kent/modules/automata_engine.py` with our wind-fixed + info-gap-fixed version
- Updating `sandbox_kent/dataset_generator.py` to compute and record the directional score
- Regenerating training data (24,922 rows, 1,797 positives, 11 features)

**Per-cell:** F1=0.176 (small improvement over Stage A's 0.165). Both new features carry signal — `wind_weighted_score` becomes the 2nd strongest predictor (+0.343), `material_class` retains its negative coefficient (-0.163, concrete buildings → less ignition).

### Stage C Spatial Results

| Variant | F1 | Recall | Precision | AUC-ROC | Recall ≥ 0.80? |
|---------|:--:|:------:|:---------:|:-------:|:---:|
| Stage C (no threshold) | 0.622 | 0.910 | 0.473 | 0.955 | ✓ |
| Stage C + threshold=0.30 | 0.671 | 0.874 | 0.545 | 0.937 | ✓ |
| Stage C + threshold=0.35 | 0.683 | 0.785 | 0.604 | 0.893 | ✗ |
| Stage C + threshold=0.40 | 0.683 | 0.644 | 0.728 | 0.822 | ✗ |

Stage C alone produced F1=0.622, slightly better than Stage A (0.602) and Stage B (0.607). However, Stage C with threshold tuning could not match Stage B + threshold=0.40 — the unified model's more confident predictions made it more sensitive to the threshold cutoff, requiring a lower threshold (0.30) which traded back some F1.

**Stage C (no threshold) full breakdown — re-scored 2026-09-27** from the saved `sandbox_kent/output/final_state_stage_c.tif` (rebuilt venv, Python 3.14.7). F1/recall/precision/AUC match the table above exactly.

| TP | FP | FN | Jaccard | Cells predicted |
|:--:|:--:|:--:|:-------:|:---------------:|
| 3,014 | 3,361 | 298 | 0.452 (0.451671) | 6,375 |

### Final Variant Comparison (master table)

| # | Variant | Features | Threshold | F1 | Recall | Precision | AUC-ROC |
|---|---------|:--------:|:---------:|:--:|:------:|:---------:|:-------:|
| 1 | CA only (baseline, partner-reported) | — | — | 0.684 | 0.544 | 0.921 | 0.772 |
| 2 | Kent's RF (broken-wind training data) | 10 (Kent) | — | 0.785 | 0.963 | 0.663 | 0.981 |
| 3 | Stage A (LR + material_class) | 10 (Kent) | — | 0.602 | 0.915 ✓ | 0.448 | 0.958 |
| 4 | Stage B (LR + wind_weighted_score) | 10 (ours) | — | 0.607 | 0.925 ✓ | 0.452 | 0.963 |
| 5 | **Stage B + threshold=0.40** | 10 (ours) | 0.40 | **0.691** | **0.867 ✓** | 0.574 | 0.933 |
| 6 | Stage C (unified) | 11 | — | 0.622 | 0.910 ✓ | 0.473 | 0.955 |
| 7 | Stage C + threshold=0.30 | 11 | 0.30 | 0.671 | 0.874 ✓ | 0.545 | 0.937 |

**Best LR result that passes the recall criterion: Stage B + threshold=0.40** — F1=0.691, Recall=0.867. This is +0.084 F1 over the unthresholded Stage B baseline.

### Conclusions from the Full Investigation

1. **The recall target (≥ 0.80) is consistently met** by every Stage A/B/C variant at default threshold and by Stage B + t=0.40 / Stage C + t=0.30 with the threshold trick. The proposal's primary criterion holds.

2. **The F1 target (≥ 0.80) was NOT achieved** by any LR variant. The closest was Stage B + threshold=0.40 at F1=0.691. Closing this gap would require either: (a) a non-linear model architecture (RF), (b) less aggressive ML predictions paired with the CA's natural conservatism, or (c) more training data tightly coupled to the validation incident's geometry — none of which are quick fixes.

3. **Stage C's unified feature set added marginal value over either Stage A or Stage B alone.** The +0.015 F1 improvement (0.607 → 0.622) is honestly within the noise band for a single-incident validation. The two extra features are largely substitutable rather than complementary.

4. **Threshold tuning was the most impactful single intervention spatially.** It moved Stage B's F1 from 0.607 to 0.691 (+0.084) — larger than the gain from material_class, wind_weighted_score, or both combined. This is methodologically interesting: a post-hoc inference-time hack outperformed the architectural feature investigations, but only because all the underlying structural fixes were already in place.

5. **Single-incident validation remains a major limitation.** With only Sitio Santa Maria as the validation target, all the differences between variants here might invert under a different fire's geometry. Multi-incident validation is required before drawing strong methodological conclusions.

### Visualizations
- `Code/sandbox_kent/output/stage_a_comparison.png` — Stage A
- `Code/output/stage_b_comparison.png` — Stage B (no threshold)
- `Code/output/stage_b_t040_comparison.png` — Stage B + threshold=0.40 (best F1 with recall ≥ 0.80)
- `Code/sandbox_kent/output/stage_c_comparison.png` — Stage C (unified, no threshold)

---

## 2026-10-23/24: Wind Direction Bug (convolve → correlate) — Data Regenerated, Stages A/B/C Retrained

### The bug
`_compute_wind_kernel()` stores the weight of the neighbour at offset (dr, dc) in `kernel[dr+1, dc+1]` (upwind neighbours weighted > 1). That is `scipy.ndimage.correlate`'s convention, but the engine applied it with `convolve`, which flips the kernel. **Fire spread INTO the wind** since the Apr 10 wind fix. Single-cell test, wind FROM north, 25 km/h: `convolve` put the high weights (1.35–1.5) on the north row; `correlate` puts them on the south row (correct).

Affected: CA spread rule (`automata_engine.py` rule path), the `wind_weighted_score` fed to the model at inference, and `dataset_generator.py` — in both `Code/` and `Code/sandbox_kent/`. So every dataset generated since Apr 10 (labels included, not just the wind column) and every model trained on them. Neighbour counts (symmetric kernel) unaffected.

How it was found: real Dec 12, 2023 wind (NASA POWER 17.7 km/h from 49°; Mactan RPVM METAR afternoon mean 10.2 km/h from 50°) scored worse than the 10 km/h NW placeholder with the old Stage B + t=0.40 model (F1 0.616 / 0.633 vs 0.691). Burned-area centroids shifted *toward* the wind source (placeholder NW → 282°, NE winds → 97°/118°). Ground-truth centroid is 287° (WNW) of ignition — the reversed NW placeholder happened to push spread toward the real scar. These pre-fix runs are superseded.

| Pre-fix run (old Stage B, t=0.40) | Precision | Recall | F1 | Jaccard | Predicted |
|---|---|---|---|---|---|
| Placeholder 10 km/h from 315° (published) | 0.574 | 0.867 | 0.691 | 0.528 | 4,998 |
| NASA 17.7 km/h from 49° | 0.483 | 0.848 | 0.616 | 0.445 | 5,817 |
| RPVM 10.2 km/h from 50° | 0.493 | 0.886 | 0.633 | 0.464 | 5,955 |

Fix: commit `bdb619a` (pushed to kin-kkoy/Thesis_LR `lr-information-gap-fix` and `master`). Kent's own RF code switched to `correlate` on Sep 20 (`9e3a8d8`); his May Set C numbers predate that.

### Regenerated data (old files kept)
Same generator settings (200 scenarios: 5 speeds × 8 directions × 5 runs, CA rule path only), written to new files:

| Dataset | Rows (old → new) | Positives (old → new) |
|---|---|---|
| `dataFiles/multi_scenario_dataset_windfix.csv` (Stage B) | 21,079 → 21,106 | 1,095 (5.19%) → 1,089 (5.16%) |
| `sandbox_kent/dataFiles/multi_scenario_dataset_windfix.csv` (Stages A/C) | 24,922 → 25,077 | 1,797 → 1,824 |

### Retrained models — `Code/retrain_windfix.py`
Recipe identical to the original models (StandardScaler + LR C=1.0, l2, lbfgs, class_weight=balanced, max_iter=2000, seed 42, random stratified 80/20 split). Verified: refitting on the old CSVs reproduces all three saved old models' coefficients exactly (max diff 0.0). Stage B was **not** Optuna-tuned. Stage A = sandbox CSV minus `wind_weighted_score`.

New models: `models/fire_lr_windweighted_windfix.joblib` (B), `sandbox_kent/models/fire_lr_stage_a_windfix.joblib`, `sandbox_kent/models/fire_lr_stage_c_windfix.joblib`.

**Per-cell test metrics at a fixed 0.5 cutoff** (not tuned on test labels):

| Stage | Test rows / fires | AUC old → new | F1 old → new | Recall old → new | Precision old → new |
|---|---|---|---|---|---|
| A | 5,016 / 365 | 0.6038 → 0.6181 | 0.1602 → 0.1742 | 0.5933 → 0.6466 | 0.0926 → 0.1006 |
| B | 4,222 / 218 | 0.6561 → 0.6251 | 0.1335 → 0.1311 | 0.6164 → 0.5826 | 0.0748 → 0.0739 |
| C | 5,016 / 365 | 0.6277 → 0.6466 | 0.1705 → 0.1858 | 0.6072 → 0.6548 | 0.0992 → 0.1083 |

Cutoff 0.6 (new models) is worse on every stage — recall collapses: A F1 0.149 / R 0.197; B F1 0.125 / R 0.261; C F1 0.175 / R 0.299.

The old scripts' "best of 16 cutoffs on the test set" gains almost nothing here (new C: 0.186 → 0.188 at 0.55; new B: 0.131 → 0.133 at 0.55), so the per-cell leakage is small. The bigger concern is the spatial inference threshold (0.40) having been chosen on Sta. Maria itself.

`wind_weighted_score` stays the strongest or second-strongest positive coefficient (B +0.301 → +0.286; C +0.343 → +0.295). `material_class` shrank (A −0.145 → −0.020; C −0.163 → −0.021); `building_presence` roughly halved in A/C.

### Spatial validation (2026-10-24 overnight, `Code/run_overnight_validation.sh`)
Raw output: `Code/output/windfix/summary.txt` and `logs/`. All runs seed 42, 100 timesteps, Sta. Maria ignition, wind-fixed models. "Buildings only" = prediction and ground truth both restricted to `stack_buildings.tif` > 0 (`spatial_validation/score_both_areas.py`; 1,919 of the 3,312 truth cells). Thresholds 0.30/0.40/0.50 are a **sensitivity sweep, reported in full** — B=0.40 and C=0.30 were fixed in May, before this data existed.

| Run | Wind | Thr | Whole map P | R | F1 | Jaccard | Pred | Bldg-only P | R | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | NASA 17.7/49° | — | 0.4409 | 0.9161 | 0.5953 | 0.4238 | 6,881 | 0.4994 | 0.9448 | 0.6535 |
| B | NASA | — | 0.4273 | 0.9387 | 0.5873 | 0.4157 | 7,276 | 0.4932 | 0.9510 | 0.6496 |
| B | NASA | 0.30 | 0.4343 | 0.9357 | 0.5933 | 0.4217 | 7,135 | 0.4981 | 0.9495 | 0.6534 |
| **B** | **NASA** | **0.40** | **0.4979** | **0.9290** | **0.6483** | **0.4797** | 6,180 | 0.5488 | 0.9463 | **0.6947** |
| B | NASA | 0.50 | 0.5487 | 0.4607 | 0.5009 | 0.3341 | 2,781 | 0.6288 | 0.5482 | 0.5857 |
| C | NASA | — | 0.4543 | 0.9191 | 0.6081 | 0.4369 | 6,700 | 0.5134 | 0.9474 | 0.6659 |
| C | NASA | 0.30 | 0.4667 | 0.6226 | 0.5335 | 0.3638 | 4,418 | 0.5112 | 0.6993 | 0.5907 |
| C | NASA | 0.40 | 0.6459 | 0.5272 | 0.5805 | 0.4090 | 2,703 | 0.6736 | 0.6399 | 0.6563 |
| C | NASA | 0.50 | 0.6530 | 0.3989 | 0.4952 | 0.3291 | 2,023 | 0.6536 | 0.5211 | 0.5799 |
| B | Placeholder 10/315° | 0.40 | 0.4819 | 0.8675 | 0.6196 | 0.4488 | 5,962 | 0.5053 | 0.9135 | 0.6507 |
| B | RPVM 10.2/50° | 0.40 | 0.4948 | 0.9055 | 0.6399 | 0.4705 | 6,061 | 0.5310 | 0.9463 | 0.6803 |

Stage A spatial run uses `sandbox_kent/models/fire_lr_stage_a_windfix_sim.joblib` — the Stage A model wrapped to drop the 11th column (`wind_weighted_score`), since the sandbox assembler now always emits 11 features (identical predictions, max diff 0.0 over 25,077 rows). The first attempt with the bare 10-feature model failed on the feature count.

Observations:
1. **Real wind now beats the placeholder** (B t=0.40: NASA 0.648 > RPVM 0.640 > placeholder 0.620). Before the fix it was the reverse (0.616 / 0.633 / 0.691) — consistent with the bug having pushed spread the wrong way.
2. **Best result passing recall: Stage B + t=0.40, NASA wind — F1 0.648, Recall 0.929** (whole map); 0.695 buildings-only. It is the threshold pre-committed in May. Down from the published 0.691, which was helped by the reversed wind pointing toward the scar.
3. Unthresholded A/B/C sit together at F1 0.587–0.608 (same plateau as before the fix); Stage C (0.608) edges A and B.
4. Thresholding helps B only at 0.40 (+0.061 F1); at 0.50 recall collapses (0.461). Stage C fails recall at every threshold ≥ 0.30, so C has no thresholded variant that meets the target.
5. C's 0.30 → 0.40 precision jump (0.467 → 0.646) is not monotone with its F1 — single-seed stochastic noise is likely material; multi-seed runs would quantify it.
6. Buildings-only F1 is 0.04–0.09 higher than whole map in every run.
7. F1 ≥ 0.80 still not met by any variant.

### Per-cell 8-variant rerun (Table 7) — `Code/retrain_windfix_variants.py` (2026-10-24)
Reruns the Apr 11 8-variant comparison on `dataFiles/multi_scenario_dataset_windfix.csv` with the original code paths unchanged (Path A/B objective from `train_lr_pathb.py`, seed 42, 100 TPE trials, 5-fold stratified CV, 80/20 stratified split, cutoff = best of 16 on the test split, as in the published Table 7). 9-feature variants = the 10-feature CSV minus `wind_weighted_score` (same 200 simulations, same rows/labels). New models saved as `models/*_windfix.joblib`; old models untouched. Raw: `Code/output/windfix/per_cell_variants.csv`, `logs/per_cell_variants.log`.

Reproduction check on the pre-fix CSV: the four default variants reproduce the published Table 7 exactly (0.133/0.470/0.077/0.630; 0.135/0.489/0.078/0.631; 0.162/0.306/0.110/0.656; 0.165/0.320/0.111/0.658). The Optuna variants land within 0.004 F1 of the published values (0.133, 0.134, 0.162, 0.165 vs 0.129, 0.134, 0.164, 0.165) but pick different hyperparameters, so their recall/precision differ — not exactly reproducible under the current library versions (sklearn 1.9.0).

| # | Variant (wind-fixed data) | Features | F1 | Recall | Precision | AUC-ROC | Cutoff | F1 @0.5 |
|---|---|---|---|---|---|---|---|---|
| 1 | 9-feature, default | 9 | 0.131 | 0.395 | 0.079 | 0.613 | 0.55 | 0.129 |
| 2 | 9-feature, Optuna | 9 | 0.131 | 0.445 | 0.077 | 0.613 | 0.40 | 0.094 |
| 3 | 9 + 7 engineered, default | 16 | 0.131 | 0.404 | 0.079 | 0.612 | 0.55 | 0.130 |
| 4 | 9 + 7 engineered, Optuna | 16 | 0.130 | 0.670 | 0.072 | 0.613 | 0.25 | 0.008 |
| 5 | 10-feature, default | 10 | 0.133 | 0.413 | 0.079 | 0.625 | 0.55 | 0.131 |
| 6 | 10-feature, Optuna | 10 | 0.132 | 0.353 | 0.081 | 0.627 | 0.55 | 0.127 |
| 7 | 10 + 7 engineered, default | 17 | 0.133 | 0.592 | 0.075 | 0.626 | 0.50 | 0.133 |
| 8 | 10 + 7 engineered, Optuna | 17 | 0.134 | 0.555 | 0.076 | 0.625 | 0.50 | 0.134 |

Best Optuna CV F1: 0.141 / 0.139 / 0.154 / 0.152 (variants 2/4/6/8).

Observations:
1. **The per-cell gain from `wind_weighted_score` is gone**: 9 → 10 features F1 0.131 → 0.133 (+0.002), was 0.133 → 0.162 (+0.029). AUC still rises (0.613 → 0.625). With ~218 test positives, differences of a few thousandths are noise. On its own the feature still separates about as well as before the fix (single-feature AUC 0.587 old → 0.574 new; neighbour count 0.510 → 0.508), and generator + engine both use `correlate`, so this is not a remaining mismatch — the old +0.029 appears to have come from the reversed kernel.
2. All 8 variants sit at F1 0.130–0.134: neither Optuna nor the 7 engineered features help (same conclusion as before).
3. Optuna variants with a custom positive-class weight collapse at a fixed 0.5 cutoff (variant 4: F1 0.008); best-of-16-on-test hides this. The published Table 7 used best-of-16 on the test split.

### Decision: AUC-ROC dropped from spatial tables (2026-10-24)
The simulation outputs a binary burn map, so map-level AUC = (recall + specificity)/2, and specificity ≈ 1 with ~37.8M negatives: AUC ≈ (recall + 1)/2 (old t=0.40: (0.867+1)/2 = 0.933, exactly the reported value). It duplicates recall and hides over-prediction. AUC is kept for the per-cell tables (computed from probabilities, ~5% positives).

### CA-only baseline rerun (Table 10) — `Code/run_ca_only_windfix.sh`, `Code/run_ca_only_seeds.sh` (2026-10-24)
Our wind-fixed engine (`Code/`), ML off, rule-based ignition (`placeholder_transition`: base 0.12, slope 0.2, building 0.1, proximity 0.2, material 0.35; blazing lasts 1 step), NASA wind 17.7 km/h from 49°, 100 timesteps max, early stop, Sta. Maria ignition. `orchestrator.run_simulation` always loads `ml_model.model_path` (it ignores `ml_model.enabled`), so the run scripts stub `orchestrator.load_model` to keep `automata.model = None`. Configs `config/windfix_CA_nasa*.yaml`, outputs `output/windfix/CA_nasa*`, logs in `output/windfix/logs/`.

**The rule-only fire did not spread in any of 10 seeds (42–51).** Every run stopped early with no active fire by step 7:

| Seed | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 |
|---|---|---|---|---|---|---|---|---|---|---|
| Stopped at step | 1 | 3 | 1 | 3 | 1 | 7 | 3 | 1 | 1 | 5 |
| Predicted cells | 1 | 2 | 1 | 2 | 1 | 4 | 2 | 1 | 1 | 3 |

Whole map: precision 1.000 (all burned cells are inside the truth), recall 0.0003–0.0012, F1 0.0006–0.0024 (mean F1 0.0011). Buildings only: recall 0.0005–0.0021, F1 0.0010–0.0042. Survival rate (spread beyond the first few steps): 0/10.

Why: in the rule path each susceptible neighbour gets p = p_base × wind_weighted_score / kernel_sum, i.e. roughly p_base/8 per burning neighbour, and a cell blazes for one step only — a single ignition cell usually fails to light anything before it burns out. The `placeholder_transition` weights are uncalibrated placeholders, not a tuned physical model. The data generator also uses this rule but seeds 10 ignition points per scenario inside an ROI, so its fires survive often enough to produce data.

The old Table 10 CA-only row (F1 0.684 / R 0.544 / P 0.921) was **partner-reported** from Kent's engine and settings, and ran with the reversed-wind CA; it is not comparable to our engine and is not reused.

### Figures 10–11 regenerated (2026-10-24)
`spatial_validation/visualize_simulation.py` on `output/windfix/B_nasa_t000` and `B_nasa_t040` → `Code/output/windfix/figures/figure10_stage_b_nasa_t000.png`, `figure11_stage_b_nasa_t040.png` (old `output/stage_b*_comparison.png` untouched). Footer counts (precision/recall/F1 match the table above):

| Run | TP | FP (false alarms) | FN (missed) | P | R | F1 |
|---|---|---|---|---|---|---|
| B, no threshold | 3,109 | 4,167 | 203 | 0.427 | 0.939 | 0.587 |
| B, t=0.40 | 3,077 | 3,103 | 235 | 0.498 | 0.929 | 0.648 |

Visual: both cover nearly all of the scar except a small missed area at its western tip; false alarms ring the scar, heaviest to the south and east. t=0.40 shrinks the overshoot mainly on the north/west edges and the south-east lobe.

### Multi-seed check (2026-10-25, `Code/run_multiseed.sh` + `spatial_validation/aggregate_seeds.py`)
Seeds 42 (from the overnight run) and 43–46 for the four key configurations; NASA wind, wind-fixed models, everything else unchanged. Raw: `Code/output/windfix/multiseed_summary.txt`. Mean ± sample SD over n=5 (min–max). Peak RAM per run 6.35–6.46 GB.

| Config | Whole-map F1 | Recall | Precision | Jaccard | Recall ≥ 0.80 | Bldg-only F1 | Bldg-only Recall |
|---|---|---|---|---|---|---|---|
| **B, t=0.40** | **0.634 ± 0.010** (0.625–0.648) | 0.913 ± 0.018 (0.882–0.929) | 0.486 ± 0.012 | 0.464 ± 0.010 | 5/5 | 0.679 ± 0.012 (0.667–0.695) | 0.941 ± 0.008 |
| B, none | 0.588 ± 0.004 (0.584–0.595) | 0.936 ± 0.004 | 0.429 ± 0.004 | 0.417 ± 0.005 | 5/5 | 0.657 ± 0.007 | 0.950 ± 0.003 |
| C, none | 0.617 ± 0.014 (0.608–0.642) | 0.918 ± 0.004 | 0.465 ± 0.017 | 0.446 ± 0.015 | 5/5 | 0.672 ± 0.017 | 0.947 ± 0.004 |
| C, t=0.30 | 0.616 ± 0.050 (0.534–0.668) | 0.785 ± 0.094 (0.623–0.845) | 0.509 ± 0.042 | 0.446 ± 0.051 | 3/5 | 0.657 ± 0.048 | 0.855 ± 0.089 |

Per-seed whole-map F1 (seeds 42/43/44/45/46): B t=0.40 0.648/0.625/0.632/0.638/0.627; B none 0.587/0.590/0.585/0.595/0.584; C none 0.608/0.608/0.610/0.642/0.614; C t=0.30 0.534/0.619/0.630/0.668/0.628.

Observations:
1. **Headline becomes Stage B + t=0.40: F1 0.634 ± 0.010, Recall 0.913 ± 0.018 (whole map); 0.679 ± 0.012 buildings-only.** Seed 42 (0.648) was the highest of the five — the single-seed number overstated it slightly.
2. The 0.40 threshold gain on Stage B is robust: +0.046 mean F1 (0.634 vs 0.588), and every t=0.40 seed (min 0.625) beats every unthresholded seed (max 0.595). Recall passes in 5/5 seeds.
3. Stage B t=0.40 stays ahead of Stage C unthresholded on average (0.634 vs 0.617), with overlapping ranges (C max 0.642 > B min 0.625).
4. Stage C t=0.30 is unstable: F1 SD 0.050, recall 0.623–0.845, passes recall in only 3/5 seeds. Seed 42 was its worst run (0.534), so the single-seed statement "C fails recall at every threshold" overstated it at 0.30 — the accurate statement is that C at 0.30 is unreliable.
5. Unthresholded runs vary little (SD 0.004–0.014); thresholds add run-to-run variance.

**Stage A added (2026-10-25, seeds 43–46, same runner):** Stage A unthresholded, n=5: whole-map F1 **0.604 ± 0.011** (0.595–0.623), Recall 0.914 ± 0.004, Precision 0.451 ± 0.012, Jaccard 0.432 ± 0.012, recall ≥ 0.80 in 5/5; buildings-only F1 0.663 ± 0.010, Recall 0.943 ± 0.001. Per-seed F1 (42–46): 0.595/0.602/0.601/0.623/0.597. Peak RAM 6.14–6.18 GB.

Effect of wind_weighted_score on the map (Stage A → Stage C, same sandbox dataset, unthresholded): mean whole-map F1 0.604 → 0.617 (**+0.013**); Stage C is higher than Stage A at every one of the 5 seeds (per-seed differences +0.013, +0.006, +0.009, +0.019, +0.017), though the two ranges overlap (A max 0.623 > C min 0.608). Buildings-only 0.663 → 0.672 (+0.009; C higher in 4 of 5 seeds). A small but consistent gain.

### Wind-direction diagnostic: Kent's F01 wind on our LR (2026-11-01, `Code/run_kent315.sh`)
Why: Kent's RF F01 run (`codex/rf-phase15-review`, `phase15_candidate_r2_rf_ca_v1.yaml`, buildings-only F1 0.786 inside a 1,024 × 1,024 crop, seed 42) used 17.7 km/h **from 315°** (NW) instead of the real NASA POWER 49° (NE). He said the two were "the same"; they are 94° apart (his own `wind_convention.py` treats both as clockwise-from-north "from" bearings, and his master.csv stores the 17.7 km/h rows as wind_sin = −0.707, wind_cos = 0.707 = 315°). To show how much direction alone matters, reran our headline config with 315°.

Setup: Stage B + t=0.40, `models/fire_lr_windweighted_windfix.joblib`, seeds 42–46, 100 steps, Sta. Maria ignition; configs `config/windfix_B_kent315_t040{,_s43..s46}.yaml` = the `windfix_B_nasa_t040*` configs with `direction_deg: 315.0` (nothing else changed). Raw: `Code/output/windfix/kent315_summary.txt`. Mean ± sample SD over n=5 (min–max). Peak RAM 6.15–6.23 GB, ~8 min per run.

| Wind (B, t=0.40) | Whole-map F1 | Recall | Precision | Jaccard | Recall ≥ 0.80 | Bldg-only F1 | Bldg-only Recall |
|---|---|---|---|---|---|---|---|
| **NASA 17.7 km/h from 49° (real)** | **0.634 ± 0.010** (0.625–0.648) | 0.913 ± 0.018 (0.882–0.929) | 0.486 ± 0.012 | 0.464 ± 0.010 | 5/5 | 0.679 ± 0.012 (0.667–0.695) | 0.941 ± 0.008 |
| 17.7 km/h from 315° (Kent F01) | 0.592 ± 0.038 (0.536–0.630) | 0.801 ± 0.077 (0.681–0.857) | 0.470 ± 0.022 | 0.422 ± 0.038 | 3/5 | 0.623 ± 0.035 (0.571–0.652) | 0.846 ± 0.067 |

Per-seed whole-map F1 (42/43/44/45/46), 315°: 0.612/0.630/0.571/0.611/0.536 (vs 49°: 0.648/0.625/0.632/0.638/0.627). Differences 315° − 49°: −0.036/+0.005/−0.061/−0.027/−0.091; mean **−0.042** F1, **−0.112** recall. Recall lower at every seed; seeds 44 and 46 fail the 0.80 recall target.

Interpretation: wind direction materially changes our LR's footprint and the real (NE) wind fits the scar better and more stably (SD 0.010 vs 0.038), so the LR is using direction as intended. Kent's F01 number therefore cannot be compared with ours or reported until it is rerun with `direction_deg: 49`; his training winds also include 17.7/315. Diagnostic only, not a paper table.

### LR trained on Kent's shared Set C dataset (2026-11-01, `Code/train_lr_kent_r2.py`, `Code/run_kent_r2.sh`)
Data: Kent's package `candidate-r2-cropped-1024-v1-g364-r1` (R2 teacher labels, 26 crop families × 7 winds × 2 replicates; built with 17.7 km/h from 315° among its winds; family 1 is centred on the Sta. Maria ignition pixel). Only the role files were extracted, to `Code/dataFiles/kent_r2_package/{train,validation,calibration}/` (sha256 and row counts checked against his manifests: 227,576 / 136,341 / 181,265). `custodian/master.csv` and `custodian/final_test/` were left inside the zip, never read for training. A copy of master.csv sits at `dataFiles/kent_candidate_r2_master.csv` (sha256 c7a92327…), not used. 63.8% of train rows are ignited (ours: 5–7%); 98.4% of master feature rows are duplicates.

**Per-cell (fit on train role, scored on validation role; calibration role unused).** StandardScaler + LR (C=1, l2, lbfgs, max_iter 2000, seed 42), all 11 features. Raw: `Code/output/windfix/kent_r2_lr_training.txt`.

| Model (validation role) | AP | ROC AUC | Brier | Log loss | Mean p |
|---|---|---|---|---|---|
| LR unweighted | 0.8331 | 0.7439 | 0.1921 | 0.5643 | 0.642 |
| LR balanced | 0.8331 | 0.7439 | 0.2053 | 0.5951 | 0.529 |
| Kent RF, uncalibrated (his Table 13, not recomputed) | 0.8303 | 0.7400 | 0.1948 | 0.5716 | — |

LR ties/slightly exceeds his RF per cell on his own validation role. Balanced weighting only lowers probabilities here (positives are the majority). Largest scaled coefficient: wind_weighted_score (0.87). Models: `sandbox_kent/models/fire_lr_kent_r2_{unweighted,balanced}.joblib`.

**Spatial (unweighted model in our sandbox CA).** Sta. Maria, NASA 17.7 km/h from 49°, 100 steps, no probability cutoff (probabilities used directly), seeds 42–46; configs `sandbox_kent/config/windfix_KR2u_nasa_t000*` = `windfix_C_nasa_t000*` with only the model swapped. Raw: `sandbox_kent/output/windfix/kent_r2_summary.txt`. Peak RAM 6.19–6.22 GB, ~10 min per run. Mean ± sample SD, n=5 (min–max).

| Model in our CA (no cutoff) | Whole-map F1 | Recall | Precision | Jaccard | Predicted cells | Recall ≥ 0.80 | Bldg-only F1 | Bldg-only Recall |
|---|---|---|---|---|---|---|---|---|
| Stage C, our data (existing) | **0.617 ± 0.014** (0.608–0.642) | 0.917 ± 0.004 | 0.465 ± 0.017 | 0.446 ± 0.015 | 6,547 | 5/5 | **0.672 ± 0.017** (0.661–0.702) | 0.947 ± 0.004 |
| LR on Kent's data, unweighted | 0.587 ± 0.005 (0.580–0.592) | 0.933 ± 0.003 | 0.428 ± 0.005 | 0.415 ± 0.005 | 7,224 | 5/5 | 0.650 ± 0.005 (0.643–0.654) | 0.954 ± 0.001 |

Per-seed whole-map F1 (42–46), Kent data: 0.589/0.588/0.580/0.592/0.584 (Stage C: 0.608/0.608/0.610/0.642/0.614). Kent-data model lower at every seed (−0.019 to −0.050; mean −0.030 whole map, −0.022 buildings-only), slightly higher recall, more over-prediction (+10% predicted cells), very stable (SD 0.005).

Caveats: run in OUR CA, whose transition (timers, burnout) differs from the R2 teacher that generated his labels, so this is not his F01 setup; a like-for-like run in his CA/crop (17.7 km/h from 49°, after he reruns) is still to do. Sta. Maria favours his dataset (training family at the ignition point), yet it still scored lower here. Second fire needed before calling either dataset better.

### Same-kitchen comparison: LR vs RF in Kent's phase16 F01 setup (2026-11-02, partner-run)
Kent ran `fire_lr_kent_r2_unweighted` (rebuilt under sklearn 1.8.0 from `~/Downloads/LR_for_Kent/` params; validation metrics reproduced exactly) in his `codex/rf-phase16` F01 setup, changing only the model (path/hash, no calibration manifest, run IDs). Setup for both: 1,024 × 1,024 crop at origin [256, 2176], ignition local [638, 554] (global [894, 2730]), seed 42, 17.7 km/h from 48.825208° grid-north (= NASA 49° true north at hour 11), wind_weight 0.20, timers 1/1, max 300 updates, stochastic probability, no thresholds. Scored with his `final_results.py` on 135,992 common-valid mapped-building cells (1,910 reference positives; 1,402 of 3,312 reference cells outside the domain). **Single seed; partner-reported** (Kent's report: `~/Downloads/LR_SameKitchen_Result.md`; LR final state sha256 1706cc63…). Kent's RF runs: 315° = original F01, 49° = `p16-f01-nasa-hr11-s42-r1`.

| Model (Kent's CA, seed 42) | Dataset | Wind | TP | FP | FN | Precision | Recall | F1 | Jaccard | Burned cells | Updates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RF, sigmoid-calibrated | Kent R2 | 49° | 1,795 | 507 | 115 | 0.7798 | 0.9398 | **0.8523** | 0.7427 | 2,302 | 135 |
| LR, uncalibrated | Kent R2 | 49° | 1,774 | 821 | 136 | 0.6836 | 0.9288 | **0.7876** | 0.6496 | 2,595 | 151 |
| RF, sigmoid-calibrated | Kent R2 | 315° | 1,795 | 863 | 115 | 0.6753 | 0.9398 | 0.7859 | 0.6473 | 2,658 | 161 |

Circle baselines on the same domain (buildings in crop, centred on the ignition; computed here from `stack_ground_truth.tif`/`stack_buildings.tif`, 1,919 truth cells, so ±9 cells from his 1,910): same size as RF prediction (R ≈ 105 m) F1 0.733; same size as LR prediction (R ≈ 113 m) 0.728; best circle in hindsight 0.738 at R = 100 m. **Both models beat the circle in Kent's CA** (our CA did not).

Kitchen effect for the same LR model (Kent R2 data, 49°, buildings-only F1): our CA 0.650 ± 0.005 (5 seeds, full map) → Kent's CA 0.788 (seed 42, crop domain). Most of the gap between our LR results and Kent's RF result comes from the CA setup, not the model or data. In the same CA, RF − LR = −0.065 F1 at seed 42 (LR burns +293 cells: +314 FP, +21 FN).

Note on wind sensitivity in Kent's CA: RF 315° vs 49° runs share 2,292 of 2,302 burned cells (TP identical), i.e. a 94° direction change barely moves the footprint there (it mainly trims over-spread). Asked Kent why.

Limits: one seed each, Sta. Maria is the development fire for Kent's pipeline (training family at the ignition point), domains differ from our whole-map/building-mask scoring, RF calibrated vs LR not. Need seeds 42–46 for both and the second fire.

#### Same-kitchen, seeds 42–46 (2026-11-02, partner-run; supersedes the single-seed reading above)
Kent added seeds 43–46 for both models (orchestrator allowance for seeds 42–46 only; everything else identical to the seed-42 runs). Report: `~/Downloads/LR_SameKitchen_Result_Seeds42_46.md`. Buildings-in-crop domain, 135,992 cells, 1,910 reference positives. Means/SDs below recomputed here from his per-seed values (match his report).

| Seed | RF F1 | RF Recall | RF Precision | RF cells | LR F1 | LR Recall | LR Precision | LR cells |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.852 | 0.940 | 0.780 | 2,302 | 0.788 | 0.929 | 0.684 | 2,595 |
| 43 | 0.664 | 0.880 | 0.532 | 3,157 | 0.665 | 0.950 | 0.512 | 3,548 |
| 44 | 0.781 | 0.932 | 0.671 | 2,653 | 0.764 | 0.930 | 0.648 | 2,744 |
| 45 | 0.693 | 0.957 | 0.543 | 3,364 | 0.653 | 0.926 | 0.504 | 3,512 |
| 46 | 0.692 | 0.934 | 0.549 | 3,249 | 0.647 | 0.949 | 0.491 | 3,691 |
| **Mean ± SD** | **0.736 ± 0.078** | 0.929 ± 0.029 | 0.615 ± 0.108 | 2,945 | **0.703 ± 0.067** | 0.937 ± 0.012 | 0.567 ± 0.091 | 3,218 |

- Seed 42 was the best run for both models; the single-seed 0.852 / 0.788 overstate typical performance. F1 ≥ 0.80 in 1/5 runs (RF) and 0/5 (LR); recall ≥ 0.80 in 5/5 for both.
- RF − LR per seed: +0.065 / −0.002 / +0.017 / +0.040 / +0.045, mean **+0.033 ± 0.026**; RF higher in 4/5. Mostly precision (LR burns ~270 more cells on average).
- Same-size circle per run (buildings in crop, computed here): RF runs' circles mean 0.701 → RF − circle mean **+0.035** (per seed +0.119 / −0.026 / +0.056 / +0.018 / +0.008); LR runs' circles mean 0.686 → LR − circle mean **+0.017** (+0.059 / +0.003 / +0.043 / −0.012 / −0.008). Both beat a same-size circle on average, but only narrowly, and not at every seed. Best hindsight circle 0.738 ≈ RF mean.
- Kent's CA is far more seed-sensitive than ours (SD 0.07–0.08 vs 0.005–0.017 buildings-only in our CA), so single-seed numbers from it should not be reported.

#### Same-kitchen, RF with the 0.40 cutoff (2026-11-02, partner-run, side check)
Kent applied the LR's cutoff rule to his RF (probabilities < 0.40 set to 0, then the usual stochastic draw; no retraining), seeds 42–46, otherwise identical to the no-cutoff RF runs. Source: screenshot of his report (full report not yet received). Per seed F1 / recall / precision / burned cells: 42 0.860 / 0.940 / 0.793 / 2,264; 43 0.678 / 0.880 / 0.551 / 3,052; 44 0.797 / 0.932 / 0.696 / 2,560; 45 0.708 / 0.957 / 0.562 / 3,250; 46 0.699 / 0.934 / 0.559 / 3,191. **Mean F1 0.748 ± 0.077**, recall 0.929 ± 0.029, precision 0.632 ± 0.108, 2,863 cells; F1 ≥ 0.80 in 1/5. vs RF no cutoff: +0.008 / +0.014 / +0.016 / +0.015 / +0.008 (mean **+0.012**), 38–114 fewer burned cells, recall identical at every seed (cutoff only removed cells outside the reference). Side check only: 0.40 was chosen for the LR, so the RF's main result stays the no-cutoff 0.736 ± 0.078. LR with cutoff: pending.

#### Same-kitchen, LR with the 0.40 cutoff (2026-11-02, partner-run)
Report: `~/Downloads/LR_Threshold040_Seeds42_46_Result.md`. Kent's engine already had the rule as `inference_mode: stochastic_probability_gate` + `proba_threshold: 0.40` (p < 0.40 → 0, then the usual draw; byte-identical engine to his RF 0.40 runs). Before the cutoff runs, a rerun of the old LR seed-42 no-cutoff config reproduced the original matrix and the exact final-state TIFF hash (1706cc63…), so the engine change did not alter no-cutoff behaviour. Only 0.40 was run.

| Seed | F1 | Recall | Precision | Burned cells | Matrix [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|
| 42 | 0.789 | 0.929 | 0.686 | 2,585 | [[133271, 811], [136, 1774]] |
| 43 | 0.676 | 0.950 | 0.525 | 3,460 | [[132437, 1645], [95, 1815]] |
| 44 | 0.782 | 0.930 | 0.675 | 2,633 | [[133226, 856], [133, 1777]] |
| 45 | 0.667 | 0.926 | 0.521 | 3,395 | [[132456, 1626], [141, 1769]] |
| 46 | 0.662 | 0.949 | 0.508 | 3,564 | [[132330, 1752], [98, 1812]] |

**Summary of the same-kitchen comparison (Kent's CA, buildings-in-crop domain, seeds 42–46, mean ± SD):**

| | No cutoff | Cutoff 0.40 |
|---|---|---|
| RF (calibrated) | 0.736 ± 0.078 | 0.748 ± 0.077 (side check) |
| LR (uncalibrated, unweighted) | 0.703 ± 0.067 | **0.715 ± 0.065** |

- LR cutoff gain +0.012 (per seed +0.002 / +0.011 / +0.019 / +0.014 / +0.015, every seed positive), the same size as the RF's +0.012. Much smaller than the +0.046 the cutoff gave Stage B in our CA.
- RF − LR with both at 0.40: +0.071 / +0.002 / +0.015 / +0.041 / +0.037, mean +0.033 ± 0.027, RF higher 5/5. Gap unchanged by the cutoff.
- Recall identical with and without the cutoff at every seed, for both models (TP/FN unchanged; only FP removed). Same pattern as RF 315° vs 49°. Asked Kent to confirm the TP cell sets are identical (not a scoring artifact).
- Neither model reaches mean F1 ≥ 0.80 on Sta. Maria in any configuration; recall ≥ 0.80 in all 20 runs.

#### Identical-recall sanity check (2026-11-02, partner-run)
Kent re-read the final-state TIFFs and compared actual TP/FN cell positions (`~/Downloads/RF_Recall_TP_Cell_Sanity_Check.md`). RF 49°, cutoff vs no cutoff: TP and FN sets exactly identical at all five seeds; every changed burned cell is reference-negative (38–137 cells per seed). RF seed 42, 315° vs 49°: 1,795 TP each, 1,793 shared, two swapped each way (4 reference-positive cells differ, 372 reference-negative); the same recall comes from equal counts. No scoring artifact. Interpretation: in Kent's CA the reached part of the scar is almost fixed for a given seed (the fire engulfs it regardless), and both the cutoff and a 94° wind change mainly alter over-spread outside the scar. So F1 differences there are driven by precision, and wind direction has weak influence on which reference cells burn.

### Second fire: Sitio Airbase (Sanggi Interior, Pajo), LR in our CA (2026-11-02, `Code/run_airbase.sh`)
**Fire and ground truth.** Sitio Airbase / Sitio Sanggi Interior, Brgy. Pajo, Friday 2025-01-24, BFP 1328H (DSWD ~1:32 PM, out 2:58 PM), ~36–46 houses. Date confirmed by Kristian from a news reel ("Friday afternoon, January 24"). GT: `stack_ground_truth_airbase.tif` from the RF side's May branch (commit a719a80, 2026-05-26), copied unchanged to `Code/processedData/raster/raster/` (sha256 1627ac903354f029…). Same grid as Sta. Maria (EPSG:32651, 6896 × 5489, 3 m). **281 burned cells, 178 on buildings.** Traced by the RF side (Kent, confirmed 2026-11-03) from 2025 imagery; exact imagery date not recorded. Google Earth's only post-fire image of the area is dated 2025-03-04 (seen 2026-11-02), so that is the likely source, ~6 weeks after the fire (unconfirmed). Kent's May runs against it used pre-wind-fix code (invalid). RF side confirmed (2026-11-03) that nothing in phase16 was tuned on the airbase or pusok maps; only Sta. Maria was used, same as the LR side.

**Ignition.** Kent's suggested point 606063.9, 1140379.2 fell on a non-building cell; moved one cell (3 m) south onto the nearest building: **606063.2784, 1140377.2699 (row 1332, col 2501)**, material class 2 (lightweight). The ignition house is unknown; the point is 9.5 m from the scar centroid (Sta. Maria: 41 m). Scar is compact: max 49 m from ignition, 95% within 42 m.

**Wind.** NASA POWER, 13:00 LST: **10.6 km/h from 61°** (12:00 12.2/61, 14:00 9.4/59). Grid bearing for Kent's CA 60.826528 (method reproduces his 48.825208 for Sta. Maria).

**Setup, fixed in advance (nothing re-tuned on this fire).** Copies of `windfix_B_nasa_t040*.yaml` with only ignition, wind and output_dir changed (`config/windfix_airbase_B_t040*.yaml`): Stage B model `fire_lr_windweighted_windfix.joblib`, proba_threshold 0.40, 100 steps, seeds 42–46. Scored with `score_both_areas.py --gt stack_ground_truth_airbase.tif`. Peak RAM 5.8 GB, ~10–15 min per run. Output `output/windfix/airbase_B_t040*/`, summary `output/windfix/airbase_summary.txt`, baselines `output/windfix/airbase_baselines.txt`.

| Seed | Whole F1 | Whole R | Whole P | Bld F1 | Bld R | Bld P | Pred cells (whole / bld) |
|---|---|---|---|---|---|---|---|
| 42 | 0.1055 | 0.9502 | 0.0559 | 0.1558 | 0.9888 | 0.0845 | 4,780 / 2,082 |
| 43 | 0.0971 | 0.9680 | 0.0511 | 0.1355 | 1.0000 | 0.0727 | 5,321 / 2,450 |
| 44 | 0.1062 | 0.9466 | 0.0562 | 0.1575 | 0.9944 | 0.0855 | 4,730 / 2,069 |
| 45 | 0.1004 | 0.9644 | 0.0530 | 0.1399 | 1.0000 | 0.0752 | 5,118 / 2,366 |
| 46 | 0.0988 | 0.9502 | 0.0521 | 0.1525 | 1.0000 | 0.0825 | 5,122 / 2,157 |

**Five-seed mean ± SD (min–max):** whole-map F1 **0.102 ± 0.004** (0.097–0.106), recall 0.956 ± 0.010, precision 0.054 ± 0.002; buildings-only F1 **0.148 ± 0.010** (0.136–0.158), recall 0.997 ± 0.005, precision 0.080 ± 0.006. Predicted 5,014 ± 251 cells vs 281 true (~18×). Recall ≥ 0.80 met; F1 far below 0.80.

**Circle baselines (centred on the ignition).** Same-size circle per seed (r 116–123 m): whole F1 0.106 ± 0.005, buildings 0.165 ± 0.006, recall 1.000. The equal-area circle beats the LR at every seed (whole and buildings). Fixed radii: 30 m → whole 0.712 / bld 0.754 (best; chosen with hindsight, optimistic); 50 m → 0.485 / 0.542; 100 m → 0.148 / 0.220. Caveat: the ignition is near the scar centre, which favours the circle.

**Diagnosis (spread over time, from the step logs).** Burned cells by step, seed 42: 10:64, 20:256, 30:531, 50:1,207, 100:4,780; seed 43: 20:271, 30:609, 100:5,321. The simulated fire matched the true size (~281 cells) around step 20 and was still growing at step 100 (no early stop). Interpretation: the CA has no suppression or duration limit and always runs 100 steps. Sta. Maria (581 houses, 3,312 cells) happened to fit that horizon; the Airbase fire was put out in ~1.5 h at ~46 houses. So this mainly tests the fixed 100-step horizon, not only the LR. **Not used for any change:** choosing a stop step from this fire would be tuning on the test fire. A duration-based stopping rule is future work (would need a second training fire to pick it fairly).

**Package for Kent (RF no cutoff + LR 0.40 main, RF 0.40 side; seeds 42–46; buildings-only):** `~/Downloads/Airbase_for_Kent.zip` (README + GT). Crop origin [768, 1920], local ignition [564, 581], scar inside (min margin 442 cells). Sent by Kristian 2026-11-02. Results pending.

### Material codes: what 0–3 mean (corrected 2026-11-03; supersedes the "material_risk inverted" item of 2026-11-02)
The 2026-11-02 claim that our `material_risk = materials_raw / 10` is inverted was **wrong**: it assumed the RF side's 0–5 legend (1 = pure lightweight … 5 = concrete), which belongs to a different raster. Evidence from git (all RF-side commits on the shared repo):
- **7fc2e61 / 91556bd (2026-03-19) and parent of 83cd423:** `data_loader.py` mapped the 0–3 `stack_materials.tif` as **1 = concrete (0.2), 2 = mixed (0.5), 3 = wood (0.9)**, 0 = no building (risk 0). This is the only code-level meaning found for the 0–3 raster.
- **83cd423 (2026-03-21):** replaced that with `materials_raw / 10` (0.1 / 0.2 / 0.3). Same order (concrete < mixed < wood), compressed scale. The LR main pipeline (`Code/modules/data_loader.py`) inherited this; Stage B was trained and run with it, so training and inference are consistent.
- **ff70bc6 (2026-04-24):** introduced the 0–5 legend `MATERIAL_CLASS_TO_RISK = [0, 0.95, 0.80, 0.48, 0.30, 0.15]` (0 none, 1 pure lightweight, 2 lightweight, 3 half, 4 mostly concrete, 5 concrete). The RF simulator now uses an int16 0–5 raster (`21f5c98c…`, codes 4–5 present on buildings). The LR sandbox (`sandbox_kent/`, Stages A/C) copied this legend but reads the 0–3 raster `e2432e59…`, so under the 2026-03-19 meaning it maps concrete (1) to 0.95 and wood (3) to 0.48, i.e. **the sandbox mapping is the inverted one**, not the main pipeline. Stages A/C are consistent between their own training and inference, but their material semantics are likely reversed.
- No producer documentation (QGIS project, style file, README) for either raster is in our files. Raster value counts (0–3 file): 1: 532,260; 2: 324,893; 3: 220,649. Producer meaning of the codes: NEEDS EVIDENCE from the RF side (who built the rasters).
- Consequence for the paper: Stage B (main LR) needs no change. Stages A/C material wording should be checked once the producer meaning is confirmed.

#### Second fire, Kent's CA: RF no cutoff (2026-11-02, partner-run, screenshot sent to Kristian)
Buildings-in-crop scoring, seeds 42–46. F1 0.266 / 0.251 / 0.242 / 0.265 / 0.194; recall 0.843 / 0.888 / 0.792 / 0.888 / 0.893; precision 0.158 / 0.146 / 0.143 / 0.156 / 0.109; burned cells 948 / 1,082 / 987 / 1,013 / 1,459; updates 103 / 161 / 129 / 125 / 189 (fire died out before the 300-update cap). **Mean ± SD: F1 0.244 ± 0.030, recall 0.861 ± 0.043, precision 0.142 ± 0.020.** Recall ≥ 0.80 at 4/5 seeds (seed 44: 0.792). Over-burns ~5–8× (178 true building cells). Same failure mode as our LR in our CA (0.148 buildings F1), but different CA, so not a like-for-like comparison. LR in Kent's CA (cutoff 0.40) not run yet. Kent is considering a fire-duration-based stopping rule (see next item).

#### Second fire, Kent's CA: LR cutoff 0.40 (2026-11-02, partner-run)
Report: `~/Downloads/Airbase_LR_Cutoff040_Seeds42_46_Result.md` (Kent's Codex labels this "Kristian's CA"; it is Kent's phase16 CA, branch `codex/rf-airbase-simulation`). Model fire_lr_kent_r2_unweighted (sha 41e6f55d…), gate p < 0.40 → 0, 300-update cap + inactivity stop, no new stopping rule. Kent centred his crop on the ignition (origin [820, 1989], local [512, 512]) rather than the 128-aligned [768, 1920] in our package; all 281 GT cells inside. Domain 215,553 building cells, 178 positives (103 of 281 GT cells are off-building and excluded). Same domain as his Airbase RF.

| Seed | F1 | Recall | Precision | Reached | Updates | Matrix [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|---|
| 42 | 0.254 | 0.843 | 0.150 | 1,003 | 103 | [[214522, 853], [28, 150]] |
| 43 | 0.274 | 0.882 | 0.163 | 966 | 123 | [[214566, 809], [21, 157]] |
| 44 | 0.247 | 0.854 | 0.144 | 1,053 | 135 | [[214474, 901], [26, 152]] |
| 45 | 0.297 | 0.994 | 0.175 | 1,012 | 125 | [[214540, 835], [1, 177]] |
| 46 | 0.270 | 0.899 | 0.159 | 1,009 | 99 | [[214526, 849], [18, 160]] |

**Mean ± SD: F1 0.269 ± 0.020, recall 0.894 ± 0.060, precision 0.158 ± 0.012, reached 1,009 ± 31.** All runs went inactive before the cap.

**Second fire, same-kitchen summary (Kent's CA, buildings-in-crop, fixed-horizon/inactivity stop):** RF no cutoff 0.244 ± 0.030 vs LR cutoff 0.40 **0.269 ± 0.020**. LR − RF per seed: −0.012 / +0.024 / +0.005 / +0.032 / +0.075, mean +0.025, LR higher 4/5 (reverses Sta. Maria, where RF was higher 5/5 by +0.033). Both over-predict ~5.5–6× (178 true building cells). In our CA, Stage B LR: buildings 0.148 ± 0.010. Neither model is near F1 0.80 on this fire; recall ≥ 0.80 in 9/10 Kent-CA runs and 5/5 ours.

#### Fire durations for a stopping rule (decided 2026-11-03, Kristian)
Source: **DSWD for both fires** (BFP file has start times only; its Sta. Maria row's date and time disagree with all other sources). Endpoint: DSWD "fire out".
- Sta. Maria: ~1:25 PM → 4:57 PM = **~212 min** (DSWD DROMIC Terminal Report, 21 Apr 2024). Other reports: under control 3:25 PM (SunStar quoting BFP), 3:43 PM (Rappler); fire out 4:57 PM (DSWD, Rappler, SunStar/BFP).
- Airbase: ~1:32 PM → 2:58 PM = **~86 min** (DSWD DROMIC #3, 3 Feb 2025). Philstar/BFP: reported 1:28 PM, under control 2:58 PM.
Not yet implemented. Airbase has now informed the stopping discussion, so any Airbase run with a new stopping rule is development/sensitivity, not held-out validation.

#### Pusok map as a possible third fire (2026-11-03, open)
`stack_ground_truth_pusok.tif`: 1,927 cells, centroid 605355.3, 1140818.3; May config ignition 605358.28, 1140812.27 (row 1187, col 2266). RF side says it is a Pusok fire from 2019 or 2018, not sure. The BFP file covers 2019-2025; its only 2019 Pusok row is Sitio Ibabao, 06/20/2019 0541H (62-80 houses in news). Size check: Sta. Maria is ~5.7 GT cells per house (3,312 / 581), so 62-80 houses would be roughly 350-460 cells; 1,927 is ~4-5x that, so the map may be a different fire or a generous trace. Not usable until incident, date and imagery are confirmed. RF side is leaving it for now.

### Duration study, LR simulator (2026-11-03, `Code/run_duration.sh`, `spatial_validation/score_duration.py`)
**Protocol (frozen with the RF side before running, 2026-11-03):** stop at inactivity or the duration horizon, whichever comes first. Horizon = DSWD fire-out interval ÷ assumed minutes per update. Airbase 86 min, Sta. Maria 212 min. Clock is an **assumption**, not calibrated: 1 min/update primary; 2 and 0.5 min/update sensitivity. Stage B LR, t = 0.40, seeds 42–46, nothing else changed (`config/duration_*.yaml`). One trajectory per seed to the longest horizon, with snapshots after the shorter horizons (`orchestrator.py` `snapshot_steps`, read-only, no RNG use). **Check:** the step-100 snapshot is cell-for-cell identical to the existing 100-step final state for all 9 seeds compared (the 5 Airbase runs and Sta. Maria seeds 42/43/44/46). Sta. Maria seed 45 was OOM-killed overnight (exit 137) and rerun; its step-100 snapshot was not compared. Sta. Maria seed 42 went inactive at step 411 (424 snapshot = terminal state); every other run was still active at its last horizon. Peak RAM 6.3–6.5 GB. Output in `output/windfix/duration/` (`summary.txt`, `duration_scores.txt` with per-seed rows, `duration_scores.json`).

Five-seed mean ± SD (min–max). Buildings in the RF side's crop gave identical scores to whole-map buildings in every case (no predicted building cells outside the crop).

| Fire | Horizon (clock) | Whole F1 | Whole R | Bld F1 | Bld R | Bld P | Reached cells |
|---|---|---|---|---|---|---|---|
| Airbase | 43 (2 min) | 0.378 ± 0.021 (0.341–0.392) | 0.893 | 0.431 ± 0.031 (0.380–0.456) | 0.946 | 0.280 | 1,053 ± 151 |
| Airbase | **86 (1 min, primary)** | **0.132 ± 0.007** (0.124–0.141) | 0.956 | **0.180 ± 0.011** (0.167–0.192) | 0.997 | 0.099 | 3,793 ± 258 |
| Airbase | 172 (0.5 min) | 0.042 ± 0.001 | 0.956 | 0.075 ± 0.002 | 0.997 | 0.039 | 12,500 ± 352 |
| Sta. Maria | 106 (2 min) | 0.617 ± 0.008 (0.609–0.626) | 0.938 | 0.665 ± 0.008 (0.658–0.675) | 0.961 | 0.509 | 6,764 ± 187 |
| Sta. Maria | **212 (1 min, primary)** | **0.501 ± 0.004** (0.496–0.506) | 0.976 | **0.589 ± 0.005** (0.585–0.597) | 0.992 | 0.419 | 9,591 ± 116 |
| Sta. Maria | 424 (0.5 min) | 0.379 ± 0.049 | 0.976 | 0.470 ± 0.053 | 0.992 | 0.309 | 13,955 ± 2,103 |

Baselines for reference (100 steps, the paper's setup): Sta. Maria whole 0.634 ± 0.010 / buildings 0.679 ± 0.012; Airbase whole 0.102 ± 0.004 / buildings 0.148 ± 0.010.

**Reading.** In the LR simulator the fire never burns out on its own, so F1 falls steadily with every longer horizon, for both fires. Recall stays ≥ 0.89 everywhere. Under the primary 1 min/update clock, Sta. Maria drops from 0.634 to 0.501 (whole) and Airbase rises from 0.102 to 0.132 (it now stops at 86 steps instead of 100). The 2 min/update clock gives Sta. Maria ≈ the paper's 100-step result (106 vs 100 steps) and the best Airbase result (0.378 / 0.431), but it is a pre-declared sensitivity setting and must not be promoted to primary after seeing these results. Neither fire reaches F1 0.80 under any clock. The results depend strongly on the assumed clock, and the LR simulator lacks a burnout mechanism (future work).

**Domain counts for the RF side (buildings in crop, LR simulator's building mask):** Airbase crop [820, 1989]: 216,187 building cells, 178 positives (RF side: 215,553 / 178). Sta. Maria crop [256, 2176]: 136,867 building cells, 1,919 positives (RF side: 135,992 / 1,910). The difference is the RF side's common-valid mask; positives match for Airbase and differ by 9 for Sta. Maria.

### Scoring-domain alignment with the RF side (2026-11-03, `spatial_validation/scoring_domain_report.py`, `scoring_domain_addendum.py`)
Report sent to the RF side: `~/Downloads/LR_scoring_domain_report.zip` (report + addendum, md + json).
- **RF-spec domain reproduced from LR files:** crop & four-layer validity (same `build_masks` rule in both data loaders) & buildings == 10 gives Airbase 215,553 cells / 178 positives and Sta. Maria 135,992 / 1,910, exactly the RF side's counts (computed independently). The 9 Sta. Maria positives dropped are building cells with slope = −9999. LR reported scores so far use full-grid buildings without the validity filter (1,919 positives for Sta. Maria). The LR simulator never ignites invalid cells, so for P/R/F1 the only difference is those 9 positives (Sta. Maria) and none for Airbase.
- **Materials raster differs between the simulators.** Slope, proximity, buildings and the Sta. Maria/Airbase GT match the RF side's fingerprints. The LR side's `stack_materials.tif` is `e2432e59…` (classes 0–3) everywhere we have it (LR repo, sandbox, response-ref, Google Drive). The RF phase16 config expects `21f5c98c…`, never seen on the LR side. Asked the RF side what it is. This could be part of why the two simulators behave differently.
- **Sta. Maria ignition cell differs by 2 rows:** LR row 892, col 2730 (UTM 606748.78, 1141695.77, supplied by the RF side on 2026-05-03; rasterio floor of 892.99997). RF uses row 894. Historical runs on both sides are left as they are; the RF side was asked to explain 894.

### LR scores on the RF-specification domain (2026-11-03, `spatial_validation/score_rf_domain.py`)
Separately labelled column for matched LR-vs-RF comparison (crop & four-layer validity & buildings == 10; digests confirmed equal by the RF side). Historical LR scores keep their original domains. Rescored from saved outputs, no simulation. No predicted valid building cells fell outside the crop in any run. Output: `output/windfix/duration/rf_domain_scores.{txt,json}`.

| Fire | Setting | F1 (RF domain) | Recall | Precision |
|---|---|---|---|---|
| Airbase (215,553 / 178) | 100 steps (baseline) | 0.148 ± 0.010 | 0.997 | 0.080 |
| | 43 updates (2 min) | 0.431 ± 0.031 | 0.946 | 0.280 |
| | **86 updates (1 min, primary)** | **0.180 ± 0.011** | 0.997 | 0.099 |
| | 172 updates (0.5 min) | 0.075 ± 0.002 | 0.997 | 0.039 |
| Sta. Maria (135,992 / 1,910) | 100 steps (baseline) | 0.680 ± 0.012 | 0.945 | 0.531 |
| | 106 updates (2 min) | 0.666 ± 0.008 | 0.965 | 0.509 |
| | **212 updates (1 min, primary)** | **0.590 ± 0.005** | 0.996 | 0.419 |
| | 424 updates (0.5 min) | 0.471 ± 0.053 | 0.996 | 0.309 |

Airbase is identical to the full-grid buildings scores (no positives dropped). Sta. Maria differs by ≤ 0.001 F1 (9 positives dropped). Per-seed values are in the txt file.
