# Phase 11 Independent Set C Dataset Validation

**Tasks:** 11.1 immutable-package validation and 11.2 grouped-evaluation feasibility  
**Review date:** 2026-10-01 (Asia/Manila)  
**Reviewed package:** `Thesis_RF/Code/output/datasets/authoritative_set_c/v1`  
**Review mode:** read-only, bounded, no persistent analysis output or cache  
**Overall result:** **PASS WITH WARNINGS — formally accepted by D-034 for Phase 12 policy planning, but not for split execution or model work**

## Executive finding

The authoritative Set C v1 package passes the checked integrity, row-quality, provenance, run-completeness, leakage-containment, and prospective grouped-evaluation feasibility gates. The package contains 10,733 eligible-cell observations from 884 complete model-free teacher runs: 797 positive and 9,936 negative rows. The actual approved metadata graph contains 26 qualifying singleton-family components, matching the Gate 6 v3 prediction. A role-blind feasibility solve found that the D-015 four-role minima, material-coverage rule, component atomicity, and 50% positive-concentration limit can be satisfied simultaneously.

No component-to-role mapping, split seed, split manifest, or final-test membership was produced or exposed. Formal acceptance should preserve the limitations on synthetic-teacher validity, conservative spatial grouping, absent observed resource extrema, and behavior that cannot be proven from published rows without replaying the teacher.

## Artifact identity

| Gate | Status | Evidence |
|---|---|---|
| Manifest whole-file SHA-256 | **PASS** | `b700f1b9f196893f23e7190daf1bc3c68b0f569f99a977ec9e919f0de73e9e3c`, matching D-033 |
| Canonical CSV SHA-256 | **PASS** | `7f0f1867445141f365687d127e449117f234348cec571990d70bc2b226ba40f6`, matching D-033 |
| Gate 6 v3 replay whole-file SHA-256 | **PASS** | `a6cd757ff78da2174625a773ba5a5df34b5dd077da41de4bf17bfe88b7bf8bbc` |
| Manifest and replay canonical payload hashes | **PASS** | Task 11.2 independently recomputed both; package binds the approved replay file and payload hashes |
| Full package validation | **PASS** | `validate_published_package` completed without exception across the manifest, canonical CSV, 884 run manifests, and 1,508 batch layers |

## Task 11.1 — Immutable-package validation

### Gate findings

| Gate | Status | Finding |
|---|---|---|
| Canonical schema | **PASS** | Header exactly matches the manifest; the 11 D-010 predictors are ordered correctly; `Ignited` is the integer 0/1 label; `split_role` is absent. |
| Missingness and parsing | **PASS** | No empty fields, nonfinite or unparseable predictors, invalid labels, or nonadjacent timestep pairs in 10,733 rows. Text values round-trip through float32 parsing; this does not independently prove collection-time NumPy dtypes. |
| Feature domains and derivations | **PASS** | `validate_feature_values` accepted all rows, including material-risk mapping, composite-flammability derivation, integer-like material and neighbor counts, and approved bounds. |
| Observation uniqueness | **PASS** | No duplicate `(run_id, timestep_t, cell_row, cell_col)` observation keys. |
| Stable-cell identity | **PASS** | Independent recomputation produced 0 mismatches across 10,733 rows. |
| Spatial and duplicate containment | **PASS** | Independent source-formula recomputation found zero spatial-block and observation-record identity mismatches across all 10,733 rows. Task 11.2 found 29 blocks, 817 stable cells, 10,733 unique observation-record IDs, and no cross-family sharing of these identities. |
| Batch/run/package totals | **PASS** | All 1,508 referenced batch CSVs exist and match their hashes, headers, row counts, and label counts. All 884 run totals reconcile with the CSV and package totals. |
| Population contract | **PASS** | 26 families × 17 wind conditions × 2 replicates = 884 distinct ordered runs. D-015 seed, scenario, wind, and run-order identities match. |
| Termination | **PASS** | All 884 runs are inactive, complete, authoritative, transition-count matched, and end with zero IGNITED/BLAZING cells. No failed or censored run was published. |
| Recorded provenance | **PASS** | All 17 recorded source-file hashes and the configuration-file hash match current files. Per-run configuration hashes, teacher ID/version, PCG64 identity, source revision/dirty state, input-hash records, and mask-hash records have zero run-level disagreements. Raw raster bytes were not opened; this proves agreement among published records and approved hashes. |
| Normalized configuration payload | **PASS** | Read-only reconstruction of the generation runner's loaded and activated configuration reproduced manifest `configuration_sha256` `f18953d5854286d29e9082caf8e5cd35f1719038627840a2c227ba42227dddfe`; the file hash and all 884 per-run hashes also pass. |
| `all_eligible` and feature timing | **NEEDS EVIDENCE** | The manifest, source contract, complete transition records, and row structure support the intended behavior. Published rows alone cannot prove that every eligible cell was captured or that every feature came strictly from state at `t` without teacher replay. |
| Runtime telemetry | **WARNING** | Approved limits are bound, but observed peak RSS, minimum available memory, minimum free disk, and elapsed-time telemetry are absent. |

The package prevalence is 7.4257%. `building_presence` is constant at 1 by the eligible mapped-building estimand. Observed `neighbor_burning_count` spans only 1–3; future claims should stay within this observed support.

### Per-family counts

| Family | Rows | Positive | Negative | Prevalence |
|---|---:|---:|---:|---:|
| `p8f01_cluster_322_m1` | 326 | 27 | 299 | 8.282% |
| `p8f02_cluster_1052_m3` | 211 | 10 | 201 | 4.739% |
| `p8f03_cluster_4974_m2` | 200 | 12 | 188 | 6.000% |
| `p8f04_cluster_160_m5` | 491 | 33 | 458 | 6.721% |
| `p8v2f05_cluster_4146_m3` | 407 | 27 | 380 | 6.634% |
| `p8v2f06_cluster_6201_m5` | 357 | 15 | 342 | 4.202% |
| `p9f_m1_r1_c1462` | 404 | 35 | 369 | 8.663% |
| `p9f_m1_r2_c8733` | 395 | 27 | 368 | 6.835% |
| `p9f_m1_r3_c1085` | 506 | 58 | 448 | 11.462% |
| `p9f_m1_r4_c4476` | 502 | 47 | 455 | 9.363% |
| `p9f_m2_r1_c4632` | 460 | 39 | 421 | 8.478% |
| `p9f_m2_r2_c145` | 443 | 35 | 408 | 7.901% |
| `p9f_m2_r3_c428` | 412 | 27 | 385 | 6.553% |
| `p9f_m2_r4_c728` | 493 | 48 | 445 | 9.736% |
| `p9f_m3_r1_c11129` | 239 | 9 | 230 | 3.766% |
| `p9f_m3_r2_c1635` | 434 | 28 | 406 | 6.452% |
| `p9f_m3_r3_c810` | 479 | 39 | 440 | 8.142% |
| `p9f_m3_r4_c14671` | 388 | 37 | 351 | 9.536% |
| `p9f_m4_r1_c1224` | 444 | 28 | 416 | 6.306% |
| `p9f_m4_r2_c17934` | 393 | 28 | 365 | 7.125% |
| `p9f_m4_r3_c16550` | 345 | 13 | 332 | 3.768% |
| `p9f_m4_r4_c16769` | 522 | 49 | 473 | 9.387% |
| `p9f_m5_r1_c2957` | 541 | 45 | 496 | 8.318% |
| `p9f_m5_r2_c1794` | 444 | 27 | 417 | 6.081% |
| `p9f_m5_r3_c2560` | 453 | 28 | 425 | 6.181% |
| `p9f_m5_r4_c1748` | 444 | 26 | 418 | 5.856% |

### Per-material counts

| Material | Rows | Positive | Negative | Prevalence |
|---:|---:|---:|---:|---:|
| 1 | 2,089 | 191 | 1,898 | 9.143% |
| 2 | 1,309 | 116 | 1,193 | 8.862% |
| 3 | 1,826 | 125 | 1,701 | 6.846% |
| 4 | 1,916 | 133 | 1,783 | 6.942% |
| 5 | 3,593 | 232 | 3,361 | 6.457% |

Material 5 supplies 33.48% of rows; material 2 supplies 12.20%. This is an observed controlled-population imbalance, not a defect, and makes material coverage an essential future allocation constraint.

## Task 11.2 — Grouped-evaluation feasibility

### Gate findings

| Gate | Status | Finding |
|---|---|---|
| Approved containment graph | **PASS** | With no independently established complete-event inventory, rows were kept within scenario family and unioned by 128×128 origin-(0,0) block, stable cell, and observation-record identity. The shared synthetic event placeholder was not treated as a complete event. |
| Actual component graph | **PASS** | 26 components, each containing exactly one complete scenario family. Cross-family block, stable-cell, and observation-record links are all zero. |
| Component adequacy | **PASS** | All 26 components qualify: 200–541 rows, 9–58 positives, and 188–496 negatives per component. This exceeds the D-015 minimum of 23 components and five observations of each class per component. |
| Gate 6 v3 comparison | **PASS** | Actual component count and qualification status exactly match the replay prediction: 26 components and 26 qualifying components. Discrepancy count is zero. |
| Run nesting | **PASS** | Runs, seeds, wind conditions, replicates, scenarios, and timesteps remain nested within their family; zero identity/order mismatches. |
| Prospective four-role feasibility | **PASS** | A role-blind in-memory binary constraint solve satisfied train ≥8/200+/800− and each other prospective role ≥5/100+/400−, material classes 1–5 in every role, component atomicity, and ≤50% of role positives from one component. The solution was independently checked and neither printed nor saved. |
| Predictor-vector collisions | **WARNING** | 2,211 unique predictor vectors; 1,795 repeated groups; 914 cross-family vector groups; 97 cross-family pair links; 543 conflicting-label groups containing 4,942 retained rows. Under D-029, these are audit collisions rather than containment edges. |
| Spatial independence | **WARNING** | The 128×128 blocks are a conservative containment convention, not evidence of measured decorrelation or physical independence. |
| Replay detail | **WARNING** | The v3 replay publishes aggregate component qualification but no per-component class/material table, so its individual component totals cannot be compared row-for-row. The actual CSV supplies those totals. |

Material occurrence across the 26 actual components is: class 1 in 5 components, class 2 in 10, class 3 in 9, class 4 in 8, and class 5 in 17. The constrained feasibility result shows simultaneous prospective coverage is possible; it does not select or authorize an allocation.

Anonymous component summaries are recorded as `(rows, positive, negative; material 1–5 row counts)`:

```text
(200,12,188; 0,78,122,0,0)     (211,10,201; 0,1,210,0,0)
(239,9,230; 0,0,151,44,44)     (326,27,299; 323,3,0,0,0)
(345,13,332; 0,0,0,345,0)      (357,15,342; 0,0,0,0,357)
(388,37,351; 0,0,157,0,231)    (393,28,365; 0,0,0,393,0)
(395,27,368; 383,0,0,0,12)     (404,35,369; 400,4,0,0,0)
(407,27,380; 0,0,293,0,114)    (412,27,385; 0,397,0,2,13)
(434,28,406; 0,0,288,0,146)    (443,35,408; 0,340,2,92,9)
(444,26,418; 0,0,0,0,444)      (444,27,417; 0,0,0,0,444)
(444,28,416; 0,0,0,444,0)      (453,28,425; 0,0,0,0,453)
(460,39,421; 0,273,0,0,187)    (479,39,440; 0,0,301,158,20)
(491,33,458; 0,0,0,0,491)      (493,48,445; 0,188,302,0,3)
(502,47,455; 485,17,0,0,0)     (506,58,448; 498,8,0,0,0)
(522,49,473; 0,0,0,438,84)     (541,45,496; 0,0,0,0,541)
```

## Evidence limits and claim boundary

- The package supports validation of published identities, row values, counts, run completeness, and grouping feasibility. It does not independently replay the teacher's state arrays.
- `all_eligible` and state-at-`t` feature timing remain source-and-provenance-supported behavior rather than row-only proof.
- Performance against these labels would measure fidelity to the approved synthetic model-free teacher within the tested case-study domain and scenario population. It would not establish observed real-fire accuracy, physical validity, arrival-time accuracy, unseen-event generalization, or superiority to the teacher.
- The controlled 26-family × 17-condition × 2-replicate population is not a local-weather prevalence sample or 884 independent fire events.
- No observed peak RSS, minimum available memory, minimum free disk, or complete elapsed-runtime telemetry exists in the package.

## Inspection and write record

The validators inspected repository instructions and state, D-010/D-014/D-015/D-017/D-029/D-033, the Phase 11 handoff, `feature_pipeline.py`, `set_c_publication.py`, `set_c_leakage.py`, `set_c_collector.py`, the package manifest and canonical CSV, all referenced batch/run layers, and the Gate 6 v3 replay. Commands used were `Get-Content`, `rg`, `Get-ChildItem`, `Get-FileHash`, and `.venv/Scripts/python.exe -B -` stdin/in-memory audits. Protected generated paths initially returned access denied in the sandbox; authorized read-only host access succeeded. No validator wrote a file, created a cache, changed an ACL, regenerated data, assigned roles, accessed protected final-test membership or `stack_ground_truth.tif`, or performed model work.

## Recommendation and next approval

**Recorded formal decision:** D-034 accepts Tasks 11.1 and 11.2 and closes Phase 11 dataset validation, subject to the warnings and claim boundaries above.

The next exact approval should authorize preparation and review of a concrete grouped-allocation policy bound to CSV SHA-256 `7f0f1867445141f365687d127e449117f234348cec571990d70bc2b226ba40f6`. That approval must fix numerical role targets and constraints and specify the protected final-test storage, permissions, and ACL mechanism. It must not itself expose role membership or execute a split. Split execution, training, tuning, class-weight selection, calibration, threshold selection, and final-test access remain separately gated.
