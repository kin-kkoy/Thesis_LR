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
