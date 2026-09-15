# Model Findings and Validation - Set B

## Set B (10 features: with wind_weighted_score, no material_class)

**Date:** May 7, 2026

This document summarizes the validation results for the Random Forest model trained on Set B, which incorporates the `wind_weighted_score` feature and removes `material_class`.

### Summary of Results

The threshold sweep reveals a model that is highly conservative and requires aggressive thresholding to achieve meaningful recall.

| Threshold | Precision | Recall | F1-Score | AUC-ROC | Jaccard | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.30** | 1.000 | 0.009 | 0.019 | 0.505 | 0.009 | Extremely conservative, fire dies immediately. |
| **0.10** | 0.475 | 0.241 | 0.319 | 0.620 | 0.190 | Fails Recall target (< 0.80). |
| **0.05** | 0.515 | 0.792 | **0.624** | 0.896 | 0.453 | **Peak F1-Score**, narrowly misses Recall target. |
| **0.03** | 0.411 | **0.936** | 0.571 | 0.968 | 0.400 | **PASSES Recall target (>= 0.80)**. |

### Findings

The introduction of the `wind_weighted_score` feature in Set B fundamentally shifted the model's probability distributions compared to the Set A baseline. The model became significantly more conservative, requiring a very aggressive probability threshold to produce a spatially relevant fire spread.

Key observations include:
-   **High Sensitivity to Wind Feature:** The model's predictions are heavily influenced by the wind-weighting, but it scales the resulting probabilities very conservatively. At a high threshold of `t=0.30`, the model predicted almost no fire spread (Recall: 0.009).
-   **Aggressive Thresholding Required:** The target `Recall >= 0.80` was only achieved after lowering the threshold to an aggressive **0.03**, at which point the Recall jumped to **0.936**.
-   **Peak F1-Score:** The best balance between Precision and Recall was found at `t=0.05`, which yielded a peak F1-Score of **0.624**, though it narrowly missed the recall target with a value of 0.792.

This behavior demonstrates that while the `wind_weighted_score` is a powerful predictor, its non-linear interaction within the Random Forest architecture leads to a strict model that requires significant calibration at inference time to translate per-cell probabilities into a realistic, continuous spatial simulation.