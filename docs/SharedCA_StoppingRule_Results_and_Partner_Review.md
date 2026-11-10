# RF and LR stopping-rule study — results and partner review

**Evidence cutoff:** 10 November 2026, Asia/Manila. **Audience:** Thesis partner and Claude. This supersedes the status in the [pre-pilot consultation](Claude_SharedCA_StoppingRule_Review_Request_v1.md). The review and scoring request below concerns retained results. Run any proposed scorer only after the owners approve the exact read and output scope.

## 1. Where we are now

- **Verified execution:** All **20 selected trajectories** completed in the shared RF-side CA. There are five RF and five LR runs per incident and three dependent horizon outcomes per trajectory, for **60 horizon records**. This includes all five Santa Maria LR runs. Both arms use the same CA. Results from the partner’s separate LR simulator are a different experiment.
- **Independent output acceptance:** Airbase RF42-r3 and its three TIFF snapshots passed independent native review and were accepted into the primary set. The other 19 trajectories have verified launcher/run records and are awaiting independent native-output review and owner acceptance. Paired acceptance and the full v4 score table are pending. The older, separately qualified Airbase RF42-r2 pilot has descriptive scores.
- **Question for this review:** The owner’s QGIS comparison suggested that the new Santa Maria RF and LR final footprints looked more spread out than earlier finals. We have recorded cell counts, but the exact QGIS layer pairs, pixel differences and new Santa Maria reference-agreement scores still need verification. Please help separate differences due to horizon selection, changed ignition, the scoring domain, and model/CA behavior.

Evidence: [D-385 acceptance](ai/DECISIONS.md#d-385--owner-accepts-airbase-rf42-r3-into-the-route-a-primary-set), [D-391 execution closeout](ai/DECISIONS.md#d-391--rf45-completed-all-twenty-route-a-executions-accounted-for).

## 2. What we planned and implemented

The earlier v3 plan considered outputs from separate CA implementations. Partner-side differences in inputs, masks, provenance and execution made that comparison a separate alignment problem. In v4, we put both frozen releases in the **same RF-side CA** to hold the simulation machinery and incident inputs common within each pair. The comparison still includes differences between calibrated RF without a cutoff and reconstructed uncalibrated LR with a cutoff. The result describes those deployed pipelines on these development incidents.

| Control | Airbase | Santa Maria |
|---|---|---|
| Incident | Sitio Sangi Interior, Brgy. Pajo, 24 Jan 2025 | Sitio Sta. Maria, Brgy. Pusok, 12 Dec 2023 |
| Approximate DSWD interval | 13:32–14:58, 86 min | 13:25–16:57, 212 min |
| 2 min/update sensitivity | 43 updates | 106 updates |
| **1 min/update primary** | **86 updates** | **212 updates** |
| 0.5 min/update sensitivity and maximum | 172 updates | 424 updates |
| 1,024 × 1,024 crop origin, row/column | [820, 1989] | [256, 2176] |
| Global / crop-local ignition | [1332, 2501] / [512, 512] | [892, 2730] / [636, 554] |
| Frozen wind speed and grid-from bearing | 10.6 km/h, 60.826537° | 17.7 km/h, 48.825208° |

- Both arms use seeds 42–46, the same 11 Set C features in canonical order, environmental rasters, eligibility and validity rules, neighborhood/transition machinery and 1/1 timers. RF uses the frozen sigmoid-calibrated probabilities with no cutoff. LR uses the frozen reconstructed release and maps float32 probabilities below 0.40 to zero. The other probabilities are retained for the same stochastic draw. No fitting, recalibration or cutoff search was part of v4.
- The clock maps an approximate reported incident interval to an update horizon. It does not rescale per-step probabilities or timers. The simulation has no suppression intervention tied to reported fire-out. Historical Airbase BFP 13:28/H90 and DSWD 13:32/H86 are distinct. Santa Maria’s historical HR11 wind basis and time-source qualifications remain in the protocol.
- Each trajectory runs to natural inactivity or the largest horizon. Natural inactivity means zero ignited and zero blazing cells. Capture is immediately after the requested completed update. An active snapshot is **censored** and retains its active cells. A horizon after natural inactivity carries forward the absorbing state from the actual stop. Snapshots consume no extra random draws and do not alter spread. A failed trajectory supplies no valid horizon outcomes.
- **Historical Santa Maria used global [894, 2730], local [638, 554].** Both v4 arms use global [892, 2730]. The old row-894 pixel center remains the inherited anchor for the frozen wind conversion. The baseline comparison therefore changes ignition as well as the observation/stop schedule. The approximate timing and the adopted ignition location retain their provenance qualifications.

Sources: [v3](StoppingRule_DSWD_Plan_v3.md), [v4 protocol and frozen identities](StoppingRule_DSWD_Plan_v4.md), [D-200 ignition decision](ai/DECISIONS.md#d-200--shared-ca-six-path-preparation-scope-reviewed-owner-adopts-new-santa-maria-row892-while-preserving-historical894).

### Implementation and validation milestones

| Milestone | Evidence and consequence |
|---|---|
| Snapshot contract and native readiness | The orchestrator uses existing CA updates and native state/reached exports. D-212/D-213 recorded bounded native readiness, including receiving eligibility at the new seed. Readiness and later paired acceptance are separate. |
| Earlier campaign and guard | Five historical Airbase LR captures were retained. An RF run stopped when the execution guard blocked temporary label encoding during calibrated prediction. The canonical guard permits that narrow operation and retains the ban on estimator/calibrator fitting. D-335 qualified one fixed-input frozen-RF compatibility result. |
| Partner’s first consultation | The partner found no stopping-logic defect in the reviewed revision and recommended avoiding more wrapper work. The Airbase LR carry-forward hypothesis can now be checked against the retained runs. The focused RF warning-name and restricted Git PATH fixes were applied during pilot preparation. |
| Pilot and final launcher | Airbase RF42-r2 produced valid TIFFs and was separately qualified after the parent rejected a JSON property-order difference. The repaired caller passed the focused verification of **339 synthetic tests** (D-379). D-381 verified fresh bindings for all 20 on the finalized launcher/source/runtime. Airbase RF42-r3 then completed cleanly and passed independent TIFF review. All eight r3 TIFF hashes matched the qualified r2 hashes. The r2 pilot adds no independent seed. |
| Remaining executions | D-386 completed 12 runs and stopped before Santa Maria RF45 dispatched at a resource-journal identity check. A concurrent append is plausible and the mismatch fields were not retained. D-388 completed the six unattempted runs. D-390 then completed one separately approved RF45 attempt using the same bound launcher and source. Preserve the earlier failed evidence. |

Sources: [decision ledger](ai/DECISIONS.md), especially D-213, D-245, D-335, D-341, D-364, D-379, D-381, D-384–D-385, D-387, D-389 and D-391. The earlier partner reply was supplied separately as `LR_side_review_SharedCA_StoppingRule_v1.md`. D-381 records HEAD `e949a5c91dbf153a21247efd58eb45d79ac00bc9` with a dirty worktree, so a Git checkout alone does not identify the executed bytes. The final launcher SHA-256 is `212b85cfc5583a2552f603eed8921f420c88f41e53c3f27c38c62ba93c377432`. The authoritative D-381 JSONL is 176,301 bytes with SHA-256 `aa97e663acdd96a6ee486bf1041f36e28404d11e7d90b0271b40628cd02f8310`.

## 3. Recorded results for every selected trajectory

The tables below come from the selected D-381 runs’ retained `termination.json` records. Each horizon entry is **reached / active @ actual capture update**. Reached means states 3/4/5, active means 3/4. Counts are full-crop raster cells. The stop is natural inactivity unless explicitly shown as cap. A zero-active capture earlier than its requested horizon is a carry-forward. These tables do not contain reference-agreement scores.

Run directories are under `Thesis_RF/Code/output/phase15/`. Prefix each suffix below with `p16-dswd-shared-v1-`. The bound revisions in the table are the primary set.

### Santa Maria

| Run suffix | Stop | 106 | **212 primary** | 424 |
|---|---:|---|---|---|
| santa-maria-rf-s42-r1 | 195 | 2255 / 27 @ 106 | 3235 / 0 @ 195 | 3235 / 0 @ 195 |
| santa-maria-rf-s43-r1 | 147 | 2797 / 50 @ 106 | 3270 / 0 @ 147 | 3270 / 0 @ 147 |
| santa-maria-rf-s44-r1 | 163 | 2299 / 72 @ 106 | 2994 / 0 @ 163 | 2994 / 0 @ 163 |
| santa-maria-rf-s45-r1 | 167 | 2889 / 65 @ 106 | 3658 / 0 @ 167 | 3658 / 0 @ 167 |
| santa-maria-rf-s46-r1 | 173 | 2846 / 42 @ 106 | 3306 / 0 @ 173 | 3306 / 0 @ 173 |
| santa-maria-lr-s42-r1 | 201 | 2176 / 27 @ 106 | 3129 / 0 @ 201 | 3129 / 0 @ 201 |
| santa-maria-lr-s43-r1 | 137 | 2263 / 36 @ 106 | 2524 / 0 @ 137 | 2524 / 0 @ 137 |
| santa-maria-lr-s44-r1 | 217 | 2301 / 62 @ 106 | 3613 / 5 @ 212 | 3615 / 0 @ 217 |
| santa-maria-lr-s45-r1 | 135 | 3119 / 71 @ 106 | 3401 / 0 @ 135 | 3401 / 0 @ 135 |
| santa-maria-lr-s46-r1 | 151 | 3013 / 51 @ 106 | 3543 / 0 @ 151 | 3543 / 0 @ 151 |

**Nine of ten runs were already inactive by 212.** All ten were inactive before 424. For LR44, the primary snapshot at 212 has five active cells and 3,613 reached. Its ordinary final at 217 has 3,615 reached. At 106 every run was still active. This is why the earlier sensitivity and primary horizon need to be inspected separately.

### Airbase

| Run suffix | Stop | 43 | **86 primary** | 172 |
|---|---:|---|---|---|
| airbase-rf-s42-r3 | 103 | 307 / 20 @ 43 | 863 / 29 @ 86 | 948 / 0 @ 103 |
| airbase-rf-s43-r1 | 161 | 227 / 14 @ 43 | 802 / 19 @ 86 | 1082 / 0 @ 161 |
| airbase-rf-s44-r1 | 129 | 254 / 16 @ 43 | 696 / 26 @ 86 | 987 / 0 @ 129 |
| airbase-rf-s45-r1 | 125 | 275 / 12 @ 43 | 682 / 26 @ 86 | 1013 / 0 @ 125 |
| airbase-rf-s46-r1 | 172 cap | 339 / 26 @ 43 | 855 / 17 @ 86 | 1431 / 9 @ 172 |
| airbase-lr-s42-r2 | 103 | 347 / 24 @ 43 | 895 / 29 @ 86 | 1003 / 0 @ 103 |
| airbase-lr-s43-r2 | 123 | 251 / 17 @ 43 | 815 / 26 @ 86 | 966 / 0 @ 123 |
| airbase-lr-s44-r2 | 135 | 257 / 18 @ 43 | 716 / 23 @ 86 | 1053 / 0 @ 135 |
| airbase-lr-s45-r2 | 125 | 254 / 13 @ 43 | 752 / 19 @ 86 | 1012 / 0 @ 125 |
| airbase-lr-s46-r2 | 99 | 349 / 21 @ 43 | 956 / 22 @ 86 | 1009 / 0 @ 99 |

All ten Airbase runs were active at 86. All five LR runs stopped at the same updates and reached the same final cell counts recorded in the earlier D-169 LR runs. Pixel equality is a separate check. RF46 hit the 172 cap with nine cells active.

### QGIS layer selection

- For Santa Maria, load `snapshot_h0212_state.tif` or the binary `snapshot_h0212_reached.tif` from each run folder. For Airbase use `snapshot_h0086_state.tif` or `snapshot_h0086_reached.tif`. Use the same crop extent and symbology across models and revisions. In the binary layer display value 1 as reached. In a state layer treat values 3/4/5 collectively as reached.
- Use `final_state.tif` for the actual last state of a trajectory. It may be later than the primary horizon or an absorbing state captured before it. Keep each earlier run’s exact ID and map type, and record the seed, historical ignition, requested/captured horizon, stop update and active counts for each layer you compared.
- Example Santa Maria RF45 primary path: `Thesis_RF/Code/output/phase15/p16-dswd-shared-v1-santa-maria-rf-s45-r1/snapshot_h0212_state.tif`. Example LR44 actual final: `Thesis_RF/Code/output/phase15/p16-dswd-shared-v1-santa-maria-lr-s44-r1/final_state.tif`.

## 4. Metrics we have and what could explain the maps

### Earlier shared-engine benchmarks

These are **historical five-seed means ± sample SD**, before the v4 duration comparison. Historical Santa Maria used row 894. Airbase used the same historical ignition as v4. Earlier runs used natural inactivity with a 300-update cap. The LR rows here use the fixed 0.40 gate. The historical ungated LR Santa Maria mean F1 was 0.703169 ± 0.066985 and is a separate baseline. The historical RF 0.40 experiment is also separate from v4 RF.

| Incident | RF sigmoid, no cutoff F1 | LR 0.40 F1 |
|---|---:|---:|
| Santa Maria | 0.736180 ± 0.078442 | 0.715312 ± 0.064599 |
| Airbase | 0.243773 ± 0.029503 | 0.268503 ± 0.019688 |

Sources: [Santa Maria same-CA history](LR_SameKitchen_Result_Seeds42_46.md), [D-165 LR 0.40](ai/DECISIONS.md#d-165--owner-requested-lr-cutoff040-matched-engine-runs-and-regression-completed), [Airbase RF](Airbase_RF_NoCutoff_Seeds42_46_Result.md), [Airbase LR](Airbase_LR_Cutoff040_Seeds42_46_Result.md).

### Airbase RF42 pilot scores already measured

These are **descriptive, one-seed r2 pilot** values on the 215,553-cell mask with 178 included reference positives. The historical final and the pilot’s carried-forward h172 footprint had zero differing scored cells. RF42-r3 reproduced all eight r2 TIFF byte hashes. Keep the r2 scores labeled pilot-only and score the selected r3 row in the formal v4 table. H43 remains a prespecified sensitivity even though it scored higher in this pilot.

| Capture | TP | FP | FN | TN | F1 | Precision | Recall | Accuracy | IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 43, sensitivity | 135 | 172 | 43 | 215203 | 0.556701 | 0.439739 | 0.758427 | 0.999003 | 0.385714 |
| **86, primary** | **140** | **723** | **38** | **214652** | **0.268972** | **0.162225** | **0.786517** | **0.996470** | **0.155383** |
| Historical final / 172 carried from 103 | 150 | 798 | 28 | 214577 | 0.266430 | 0.158228 | 0.842697 | 0.996168 | 0.153689 |

The primary pilot snapshot removed 75 false positives and lost 10 true positives relative to the historical final. Its F1 rose by only 0.002542. The high accuracy alongside low precision reflects the large number of reference-negative cells. The separate historical Airbase RF H90 experiment reported mean F1 0.294022 and mean reached cells 822.6, versus 0.243773 and 1,097.8 for its historical full runs. Keep those results outside the v4 horizon table. Sources: [D-367 pilot score](ai/DECISIONS.md#d-367--airbase-rf-seed42-score-comparison-small-primary-change-identical-final-footprint), [Airbase H90 study](Airbase_RF_Duration90_Seeds42_46_Result.md).

### Santa Maria’s larger-looking footprints

There is some support for the visual observation in the recorded final counts. Below, “old” is historical row 894, using RF no-cutoff from D-163 and LR 0.40 from D-165. “New” is v4 row 892, using the actual final (also the 424 carry-forward). These are descriptive count differences between experiments. Verify spatial domain and pixel identity separately before interpreting the deltas.

| Seed | RF old → new (difference) | LR old → new (difference) |
|---:|---:|---:|
| 42 | 2302 → 3235 (+933) | 2585 → 3129 (+544) |
| 43 | 3157 → 3270 (+113) | 3460 → 2524 (−936) |
| 44 | 2653 → 2994 (+341) | 2633 → 3615 (+982) |
| 45 | 3364 → 3658 (+294) | 3395 → 3401 (+6) |
| 46 | 3249 → 3306 (+57) | 3564 → 3543 (−21) |
| Mean | 2945.0 → 3292.6 (+347.6) | 3127.4 → 3242.4 (+115.0) |

**Inferred interpretation:** The new RF finals are larger in all five recorded seed comparisons. LR changes direction across seeds. For nine Santa Maria runs, the primary horizon simply observes the fire after it has already become inactive. Within a single trajectory, capturing an earlier snapshot cannot add cells to its cumulative reached footprint. Comparing historical row-894 runs against new row-892 runs cannot isolate the effect of time. More reached cells may add true positives, false positives or both. The new Santa Maria score direction is still unknown.

Please check these hypotheses in order:

1. **Layer selection and baseline identity.** Compare the actual primary snapshot and final separately. Verify whether the historical LR layer used 0.40, no gate, or the partner’s separate CA, and whether it was a natural final or fixed-step output.
2. **Spatial spread and masks.** On a verified common domain, count cells reached only by the new map, only by the historical map, and by both. Split the difference by reference positive/negative. Determine whether extra spread reaches new reference cells or mainly adds false positives. Map the difference by location and direction if the rasters are authorized.
3. **Ignition, clock and CA effects.** Investigate whether the two-row ignition change, eligible-building connectivity, frozen probabilities, transition assumptions or lack of explicit suppression could plausibly explain particular spatial patterns. A footprint comparison alone cannot validate the physical clock or establish a sole cause. Separate exact evidence from hypotheses.
4. **Reference and model evaluation.** Check reference registration, building coverage/exclusions and prior-use limitations. Santa Maria is a calibration/development incident. Airbase is a development comparison with independence and overlap qualified. No held-out or general model-superiority claim follows. Review model improvements after we report the frozen comparison, with separate evaluation and no threshold or horizon selection from these reference scores.

**Partner-side update requested:** Your earlier separate-LR-CA review reported Airbase reached counts of 1,053 / 3,793 / 12,500 at 43 / 86 / 172 and continued activity there. Please confirm the run IDs, input/domain, and whether those findings apply to the exact files you examined. Report any separate-LR-CA Santa Maria evidence in its own table. The shared-CA LR results above are the comparison arm for v4.

## 5. Task for the thesis partner — run and check the retained-output scores

**Start with both Santa Maria arms at the primary horizon of 212.** Once native alignment, pairing and score checks pass, report the other horizons and repeat for Airbase. The purpose is to score retained snapshots, not to rerun either model. Arrange a separate reviewed scoring authorization and use the authorized incident references only.

1. **Assemble the inputs.** Use the 20 exact run IDs in §3. For each run obtain `start.json`, `termination.json`, the parent/caller receipts that pin them, the independent D-381 binding, and the eight final TIFFs (final state, environmental validity, and three state/reached pairs). Use the published final files, not partial TIFFs. Confirm raw output hashes, lengths, grid and band semantics, exact incident crop, state categories and observed counts. Use the embedded-only native settings from the reviewed reader. Preserve the D-385 RF42-r3 result and exclude earlier attempts. D-387, D-389 and D-391 list the local evidence roots. GitHub may not contain outputs or OS-temp evidence. Ask me for the exact approved files if any are missing. Do not infer metrics from the report’s full-crop counts.
2. **Confirm the scoring layers.** Under `Thesis_RF/Code/`, the Santa Maria reference is `processedData/raster/raster/stack_ground_truth.tif`; the Airbase reference is `dataFiles/stack_ground_truth_airbase.tif`. The common building raster is `processedData/raster/raster/stack_buildings.tif`. Use the protocol pins and native crop windows. Reference-positive means value 1. Mapped building means value 10. The common domain is environmental validity **and** prediction, reference and building validity **and** mapped building. Verify the actual common mask and included-positive digests across RF and LR. Expected historical counts to reconcile are 135,992 domain cells and 1,910 of 3,312 reference positives for Santa Maria, and 215,553 cells and 178 of 281 positives for Airbase. Report included and excluded positives. Treat any count, digest, geometry, provenance or input mismatch as a finding; do not resample or change the mask to make it pass.
3. **Run the existing verified reader and scorer in the approved environment.** Use [`dswd_snapshot_readiness.py`](../Thesis_RF/Code/dswd_snapshot_readiness.py) and [`dswd_snapshot_results.py`](../Thesis_RF/Code/dswd_snapshot_results.py). These are library entry points. There is no reviewed one-line v4 scoring CLI in this handoff. Prepare a small read-only driver for review using this call order:
   - Use the shared design from `orchestrator.PHASE16_DSWD_SHARED_PROTOCOL["design"]`. Obtain separate approval for `collect_alignment(alignment_read_whitelist(), capability=..., authorized=..., authorization_reference=..., design=...)` unless an authenticated alignment package already provides the matching report **and native arrays**. The readiness summary alone does not provide those arrays. That function opens six fixed alignment TIFFs. Reuse only documented reader capability evidence for the actual Rasterio/GDAL runtime, including windowed reads. Supply real reviewed evidence and authority for both arms.
   - For each run supply independently reviewed `expected` values to `read_snapshot_run`: `start_sha256`, `termination_sha256`, full D-381 `binding`, reviewed `effective_configuration`, `alignment_sha256`, and `review_reference`. Verify effective configuration against its bound hash. Use each reader’s returned `outcomes` for the three scores. The native reader validates artifacts and calls `evaluate_snapshot_arrays`, which uses the existing shared final-footprint metric arithmetic.
   - For formal shared-CA pair acceptance, use `read_shared_runs` on each incident/seed RF–LR pair with both explicit roots, reviewed expectations, alignment/capability evidence and separate authorization arguments. The function verifies both runs through native reads and checks common inputs, geometry, actual ignition, wind, masks, timing, features and strict source/runtime equality. Review a driver that accounts for the pair reader’s additional reads if both score and pair records are produced. A dictionary-only match from `compare_shared_runs` is diagnostic. Neither reader requires deserializing a model for scoring.
   - The canonical reader checks the original bound absolute run destination and the retained termination publication hard-link. A copied folder on a different machine may fail those checks. If so, arrange the authorized canonical run on the original workspace and review its pinned output. An independently approved check on portable copies can be reported separately. Do not rewrite the binding or bypass path, hard-link or hash checks to claim canonical acceptance.
4. **Independently verify the metric arithmetic.** On the common domain, predict positive for states 3, 4 or 5. Cross-tabulate prediction and reference to reproduce `[[TN, FP], [FN, TP]]`. Check that the sum equals the domain size and TP + FN equals the included reference positives. Report F1 = 2TP/(2TP+FP+FN), precision = TP/(TP+FP), recall = TP/(TP+FN), accuracy = (TP+TN)/N, and IoU = TP/(TP+FP+FN), using the existing evaluator’s zero-denominator convention. For Santa Maria RF and LR, prioritize per-seed differences in TP and FP at h212 before interpreting F1.
5. **Return reproducible outputs.** Send the exact command or driver, code/runtime identity, receipt and input/output hashes, and one row per incident × arm × seed × horizon. Include run ID, clock role, requested/capture/stop updates, censoring/carry-forward, full-crop reached/active and in-domain predicted positives, confusion counts and all five metrics. Use `summarize_five_seed_groups` for means and sample SD across exactly seeds 42–46 in each complete incident/arm/clock group. Return primary comparisons first and sensitivities separately. Label missing, unmatched or rejected cases and withhold their group summaries. Horizons share one seed trajectory. Do not pool them as independent replicates. Any score exports should be fresh files in the approved evidence location.

Keep diagnostic historical map differences separate from same-incident v4 RF/LR pairing. Preserve original records, outputs, models, configs and the synthetic final-test restriction.

## 6. Questions for your review and Claude’s independent assessment

Please return separate RF and LR findings, followed by a joint conclusion. Use **VERIFIED**, **INFERRED** and **NEEDS VERIFICATION**. For each substantive claim, cite the run ID, file/record or source location that supports it. If you lack the data, name the smallest evidence needed.

1. Does the retained timeline and selected QGIS layer explain the larger-looking Santa Maria finals for each arm and seed? Identify any concrete discrepancy in capture, carry-forward, inactivity, or output selection.
2. How much of the primary result is determined before 212? Compare early and primary footprints and the final separately. Which added cells are true positives versus false positives? Does the conclusion change across seeds or between RF and LR?
3. Which historical comparisons are genuinely like-for-like? Which differences are confounded by row 894 versus 892, model policy, crop/mask, or the partner’s separate CA? Review the RF and LR pipelines individually as well as their shared CA behavior.
4. What best explains the low Airbase precision/F1 we actually measured? After scoring Santa Maria, does the same pattern occur there? Distinguish evidence about false positives, false negatives and reference exclusions from possible effects of ignition, winds, CA transitions, data provenance or model transfer.
5. What is the smallest defensible next step for the thesis? Separate conclusions we can write now from findings requiring scores or new evidence. If you recommend a future model, clock or CA experiment, state the single question, required controls, and independent evaluation needed. Keep the current frozen comparison and prespecified primary clocks intact.

Please send back the score tables, a short list of verified findings and remaining uncertainties, and your recommendation for Chapters 5 and 6. We will review your results together before changing the experiment or interpreting the new metrics in the thesis.
