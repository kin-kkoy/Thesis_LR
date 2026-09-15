# Model Findings and Validation - Set C

## Set C (11 features: with wind_weighted_score and material_class)

**Date:** May 8, 2026

This document summarizes the validation results for the Random Forest model trained on Set C, which incorporates the full 11-feature schema, including both wind_weighted_score and material_class.

### Summary of Results

The threshold sweep for Set C demonstrates the most balanced performance of all three sets, successfully leveraging both the dynamic wind feature and the static material risk feature.

| Threshold | Precision | Recall | F1-Score | AUC-ROC | Jaccard | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.30** | 0.650 | **0.827** | **0.728** | 0.914 | 0.572 | **PASSES Recall & Best F1-Score.** |
| **0.10** | 0.447 | 0.943 | 0.607 | 0.971 | 0.435 | High Recall, but lower Precision. |
| **0.05** | 0.381 | 0.947 | 0.544 | 0.974 | 0.373 | Very high Recall, but Precision suffers. |
| **0.00** | 0.741 | 0.495 | 0.594 | 0.748 | 0.422 | Default (no threshold), fails Recall target. |

### Findings and Comparative Analysis

The inclusion of both wind_weighted_score and material_class in Set C produced a model that is both powerful and well-calibrated, outperforming both Set A and Set B.

1.  **Synergistic Feature Impact:**
    *   Compared to **Set A** (which lacked wind weighting), Set C's model is far more effective at capturing fire spread, achieving a peak F1-score of **0.728** vs. Set A's 0.603.
    *   Compared to **Set B** (which lacked material class), Set C's model is much less conservative and better calibrated. Set B required an extremely aggressive threshold of 0.03 to meet the recall target, resulting in a low F1-score (0.571). In contrast, Set C achieved the recall target at a much more moderate threshold of **0.30**, while also delivering the highest F1-score.

2.  **Optimal Threshold and Performance:**
    *   The threshold sweep shows that `t=0.30` is the optimal setting for this model. It not only **passes the `Recall >= 0.80` target** (achieving 0.827) but also yields the highest F1-Score (**0.728**) across all experiments conducted.
    *   This indicates that the 11-feature model provides the best representation of fire dynamics, allowing the CA simulation to produce spatially accurate results without requiring extreme, performance-damaging thresholds.

### Conclusion for Set C

The results from Set C validate the thesis that a richer feature set, combining dynamic environmental factors (wind_weighted_score) with detailed static attributes (material_class), produces a superior predictive model. The 11-feature RF model is the most effective configuration, providing a strong balance of Precision and Recall that leads to the most accurate simulation of fire spread.