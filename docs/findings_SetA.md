# Findings and Validation

## RF Model - Set A Validation Results (Threshold Sweep)
**Date:** May 7, 2026

The threshold sweep for the Random Forest model on Set A (baseline 10-feature schema without wind-weighting) was completed successfully. The fix to the threshold enforcement mechanism in the Cellular Automata engine allowed for a direct trade-off between Precision and Recall by adjusting the proba_threshold parameter.

### Summary of Results

| Threshold | Precision | Recall | F1-Score | AUC-ROC | Jaccard | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.50 | 0.840 | 0.463 | 0.597 | 0.732 | 0.426 | High Precision, but fails Recall target (< 0.80) |
| 0.40 | 0.831 | 0.463 | 0.595 | 0.732 | 0.423 | Similar to t=0.50, no significant Recall gain |
| 0.30 | 0.604 | 0.435 | 0.506 | 0.718 | 0.338 | Precision and Recall both drop |
| 0.10 | 0.510 | 0.738 | 0.603 | 0.869 | 0.431 | Peak F1-Score |
| 0.05 | 0.447 | 0.767 | 0.564 | 0.883 | 0.393 | Peak Recall |

### Analysis

The probability threshold sweep performed as expected. By systematically lowering the threshold from a strict 0.50 down to a lenient 0.05, we successfully traded a high Precision for a much-needed higher Recall. This adjustment was critical for counteracting the massive class imbalance of the dataset, where true fire events represent only 0.0087% of the data.

The key takeaways are:

* Recall was boosted significantly, from 0.463 (at t=0.50) up to a peak of 0.767 (at t=0.05). While this still narrowly misses the target of 0.80, it demonstrates the effectiveness of the thresholding mechanism.
* The F1-Score, which balances Precision and Recall, peaked at 0.603 at a threshold of 0.10. This represents the most balanced performance for this model configuration.

### Conclusion for Set A

Set A is now fully explored. These results establish a clear empirical baseline for the RF model's performance before the introduction of the wind_weighted_score feature. The subsequent sets (B and C) will be compared against this reference to quantify the impact of the enhanced feature engineering.

### Extra Notes

I have noticed that the higher time steps and higher value for the proba_threshold, the more likely the fire will stop spreading at some point. This is because the susceptible cells with probabilities below the threshold will never ignite, which can lead to a situation where there are no more burning neighbors to ignite the remaining susceptible cells. This is a critical insight for understanding the dynamics of the CA simulation and will be important to consider when analyzing the results of Sets B and C.

### TLDR FOR THE NOTES

- Higher proba_threshold values = fire spread stop early
- Higher proba_threshold = requires more timesteps to reach the same final state, if it is reached at all
- Lower proba_threshold values = more aggressive fire spread, but also more false positives (lower Precision)
- Lower proba_threshold values = higher Recall, but also more noise in the predictions