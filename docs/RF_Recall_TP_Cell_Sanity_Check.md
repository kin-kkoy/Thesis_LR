# RF Recall Sanity Check — Exact True-Positive Cell Positions

Date: **2 November 2026 (Asia/Manila)**.

**PASS — direct raster/set counts reproduce the reported recall values.**

**WARNING — equal recall does not mean identical TP locations in the 315° versus 49° comparison.**

- At 49°, **RF cutoff 0.40 versus no cutoff has exactly identical TP and FN cell sets at every seed 42–46**.
- For **315° versus 49° at seed 42**, both have 1,795 TP cells, but only 1,793 are shared. There are **two TP cells unique to each run**: four differing reference-positive cell positions in total.

The equal recall values are reproduced from the actual masks, independently of stored metrics. They are explained by the TP/FN counts; no recall-calculation discrepancy was found. Identical output reuse is ruled out for these comparisons: all 11 prediction TIFF hashes differ, and their burned/FP footprints differ.

## Exact position comparison

Domain and truth arrays are byte-for-byte equal across all 11 loaded cases: **135,992 common-valid mapped-building cells** and **1,910 included reference positives**. A/B denotes no cutoff/cutoff for the first five rows and 315°/49° for the last row. Counts below are based on exact zero-based cell indices, not rounded scores.

| Comparison | TP A | TP B | Shared TP | TP only A | TP only B | Identical TP set? | Recall, both runs |
|---|---:|---:|---:|---:|---:|---|---:|
| 49° seed 42: no cutoff vs 0.40 | 1,795 | 1,795 | 1,795 | 0 | 0 | Yes | 0.939791 |
| 49° seed 43: no cutoff vs 0.40 | 1681 | 1681 | 1681 | 0 | 0 | Yes | 0.880105 |
| 49° seed 44: no cutoff vs 0.40 | 1781 | 1781 | 1781 | 0 | 0 | Yes | 0.932461 |
| 49° seed 45: no cutoff vs 0.40 | 1827 | 1827 | 1827 | 0 | 0 | Yes | 0.956545 |
| 49° seed 46: no cutoff vs 0.40 | 1784 | 1784 | 1784 | 0 | 0 | Yes | 0.934031 |
| Seed 42: 315° vs 49°, no cutoff | 1,795 | 1,795 | 1,793 | 2 | 2 | No | 0.939791 |

For the wind comparison, each run also has 115 FN cells; 113 are shared and two differ on each side. Both recalls are exactly `1795 / 1910 = 0.9397905759162304` before rounding. The same TP-count result also holds over the reference-valid crop outside the primary-domain restriction; no run has additional reference-positive burned cells outside the primary domain.

## The four wind-case positions, independently checked

All coordinates below use **zero-based [row,col]**. Global origin of the 1024×1024 crop is [256,2176]. State 5 means extinguished/burned; state 2 means not yet burning. Direct 1×1 native TIFF reads confirmed reference=1, mapped-building value=10 and environmental-validity=1 at all four positions.

| TP membership | Local row,col | Global row,col | 315° state | 49° state |
|---|---|---|---:|---:|
| 315° only | `[632, 548]` | `[888, 2724]` | 5 | 2 |
| 315° only | `[646, 518]` | `[902, 2694]` | 5 | 2 |
| 49° only | `[620, 551]` | `[876, 2727]` | 2 | 5 |
| 49° only | `[645, 510]` | `[901, 2686]` | 2 | 5 |

## The overall predictions are different

| Comparison | Burned A | Burned B | Burned cells only A | Burned cells only B | Changed reference-positive cells | Changed reference-negative cells |
|---|---:|---:|---:|---:|---:|---:|
| 49° seed 42: no cutoff vs 0.40 | 2302 | 2264 | 38 | 0 | 0 | 38 |
| 49° seed 43: no cutoff vs 0.40 | 3157 | 3052 | 121 | 16 | 0 | 137 |
| 49° seed 44: no cutoff vs 0.40 | 2653 | 2560 | 93 | 0 | 0 | 93 |
| 49° seed 45: no cutoff vs 0.40 | 3364 | 3250 | 114 | 0 | 0 | 114 |
| 49° seed 46: no cutoff vs 0.40 | 3249 | 3191 | 61 | 3 | 0 | 64 |
| Seed 42: 315° vs 49° | 2658 | 2302 | 366 | 10 | 4 | 372 |

At each cutoff/no-cutoff seed, every changed burned cell is reference-negative within this domain. For 315°/49°, four changed cells are reference-positive and 372 are reference-negative. Some cutoff footprints exchange false-positive locations rather than merely form strict subsets (seeds 43 and 46); the audit reports those changes rather than assuming monotonic spread.

## Verification and evidence

The audit reread **11 retained RF final-state TIFFs**, their environmental masks and the same hash-bound reference/building rasters through the unchanged restricted loader. It checked hashes before/after reading, matching grid/crop/CRS/transform/native validity, inactive state codes and identical domain/truth arrays. No models were loaded and no simulations were run.

It first derived the domain and the actual positions of `burned=(state==5)` and `reference_positive=(reference==1)`, then formed TP/FN/FP/TN using independent Python set intersections/differences. Every confusion matrix matched the previously reported values. Only after this calculation were counts/recall cross-checked with `final_results.evaluate_arrays`. An assert-based self-check verifies that equal-sized sets with exchanged positions fail equality. The four differing wind-case positions were independently reread using native 1×1 TIFF windows, without any metric helper.

Evidence directory: `Thesis_RF/Code/output/rf_tp_sanity/20261102-r1/`:

- `results.json`: input/source hashes, direct counts, coordinate-set hashes and all six pairwise comparisons.
- `*.positions.json`: sorted TP/FN/FP and included-reference positions for each of 11 cases, with crop/global-origin metadata.
- `four_cell_native_check.json`: direct native values at the four wind-case positions.
- `audit.py`, `check_four_cells.py`, stdout and final verification records: reproducible audit and preservation evidence.

The five cutoff/no-cutoff pairs use the existing RF49 run IDs and RFgate040 counterparts for seeds42–46. The wind pair is `p15-candidate-r2-f01-s42-r1` versus `p16-f01-nasa-hr11-s42-r1`. Earlier score reports and source/run artifacts are preserved. This audit does not reassess the historical315° run-record qualifications or repeat its simulation.

## Limits

These findings establish position identity or difference for the identified retained RF footprints. They do not explain all underlying spread dynamics or independently validate training lineage, reference construction, physical wind/spread or held-out performance. F01 remains one calibration/development case with an approximate reference and 1,402/3,312 reference-positive crop cells excluded (42.33%). NASA true 49° remains inferred/owner-adopted with constant ignition-centre conversion to grid-from 48.825208°. Existing incident/time/year/location qualifications remain unchanged. LR positions were outside this RF-only check.
