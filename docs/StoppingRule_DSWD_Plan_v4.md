# Stopping Protocol v4 — Frozen RF/LR Models in the Shared RF-Side CA

Date: 3 November 2026 (Asia/Manila). D-200/D-201 authorize source preparation and test authoring only. This document grants no execution, protected read or test authority.

## Design and preserved evidence

Both frozen releases use the RF-side CA, environmental inputs, feature construction/order, validity and receiving masks, crops, wind, ignition, timers, transitions and stochastic sampling. This is a development comparison of these releases and policies, not an isolated model-family effect. V3, its own-simulator comparison, RF-only snapshot v1, full runs, original315/RF49, Airbase H90 and every historical configuration/result remain unchanged.

RF uses calibrated rf_sigmoid.joblib, SHA-256 78828dd4b2fcc66e9cdfd8ecb4a3ccd18c268478f3b5e6a4374c3286e3903e54, with calibration_manifest.json SHA-256 3141b8554662d703691c06a327f9a6c96c79f20840ce3738c358d4bdb6632afb. Its policy is stochastic_probability without cutoff.

LR uses the existing reconstructed release output/lr_same_kitchen/20261102-r1/fire_lr_kent_r2_unweighted_sk1.8.0.joblib, SHA-256 41e6f55d3e07fa27a57b42d7d0be0828c72cf87ff6cc92c3834b0c0dfd8d217b; no calibration manifest. Its stochastic_probability_gate applies float32 p<0.40 -> 0, p>=0.40 unchanged, followed by the existing draw. Legacy threshold, inference_threshold and classification_threshold remain null. No fit, rebuild, recalibration, partner Stage B or sandbox substitution is included. D-162/D-165/D-169 interface evidence is reused; original LR training-feature lineage and original-pickle equivalence remain qualified.

## Incident and ignition controls

| Incident | Crop origin / shape | Local ignition | Global ignition | Wind km/h / grid-from |
|---|---|---|---|---|
| Airbase: Sitio Sangi Interior, Brgy. Pajo, 2025-01-24 | [820,1989] / [1024,1024] | [512,512] | [1332,2501] | 10.6 / 60.826537 degrees |
| Santa Maria: Sitio Sta. Maria, Brgy. Pusok, 2023-12-12 | [256,2176] / [1024,1024] | [636,554] | [892,2730] | 17.7 / 48.825208 degrees |

D-200 owner adopts Santa Maria source point EPSG:32651 [606748.78,1141695.77] and reported affine-floor rationale for the new shared cases. Geographic truth, source precision and near-boundary rationale remain qualified. Historical global [894,2730] / local [638,554], frozen family definitions and calibration inventory are preserved. No row893 sensitivity or extra row894 duration trajectories are included. Both arms use892. Future native readiness must verify receiving eligibility before engine construction, ignition or inference; invalid seeds stop, never move. That membership is presently NEEDS VERIFICATION.

Santa Maria inherits the existing six-decimal constant grid-from bearing48.825208 and the original conversion anchor EPSG:32651 [606750.2784,1141691.2699], the historical894 ignition pixel centre. This anchor is distinct from the new892 ignition; no conversion is recomputed. NASA true north remains INFERRED; HR11 selection, adopted BFP11:31, researcher screenshot12/12/2024-1131H versus owner incident year2023 and DSWD approximately13:25 discrepancy are retained. Airbase retains its existing HR13/LST/north/anchor qualifications. No additional180-degree reversal is applied.

Comparisons with historical894 outcomes change ignition as well as duration and cannot isolate stopping-rule effects.

## Duration assumptions and trajectory matrix

DSWD Airbase approximately13:32–14:58 gives86minutes; BFP13:28 remains separate and historical H90 is unchanged. Santa Maria DSWD approximately13:25–16:57 gives212minutes. Fire out is the approximate reported endpoint, not simulated suppression. Sources remain DSWD DROMIC Report3 (2025-02-03), p1 and Pusok Terminal Report (2024-04-21), p1, as recorded in v3; report date2024 is not incident year.

| Assumed minutes/update | Role | Airbase horizon | Santa Maria horizon |
|---|---|---:|---:|
| 0.5 | sensitivity | 172 | 424 |
| 1 | primary | 86 | 212 |
| 2 | sensitivity | 43 | 106 |

Every clock is ASSUMPTION, not calibrated physical time or confidence bounds. It labels updates without rescaling probabilities or timers. Timers remain1/1, seeds42–46, canonical11-feature Set C order, existing weights/masks, no resume/checkpoint. One trajectory per incident/arm/seed runs to inactivity or largest horizon:20trajectories,60dependent outcomes,12incident/arm/clock groups. Horizons from the same trajectory are not independent replicates.

## Observation, termination and evidence

Capture immediately after each requested completed update and before the next, with independent native int8 state and uint8 reached(states3/4/5) copies. No extra random draws, reseeding, inference or mutation. Record initial ignition separately at update0, per-update new ignitions/reached/active/five-state counts, requested and actual capture/stop updates, clocks/roles/duration basis, censoring, absorbing carry-forward, output identities and configuration/execution/model/arm/ignition bindings.

Natural inactivity requires IGNITED and BLAZING both zero. Later horizons may reference that earlier unchanged absorbing state, explicitly without claiming updates were executed. Active horizon outcomes are censored; never force extinction. Resource, step, export or termination-publication failure invalidates every outcome of that trajectory. Retain all partials and final receipts; orphan success-looking files cannot supply outcomes. Existing exclusive creation, no overwrite, hard-link publication, resource/source/runtime/configuration/approval bindings and profile-scoped output allowlists apply. No destination is allocated by this protocol or its disabled manifest.

## Alignment, scoring and reporting

Use existing final reached-footprint evaluator and shared metric arithmetic; references/scores never affect stopping. Both verified arms must match exact common input hashes, native geometry, actual ignition, wind, validity/eligible/evaluation mask digests, reference identity/validity/positive coverage and scoring domain. Counts alone do not prove equality. Missing counterpart means comparison pending; shared pairing does not accept cross-simulator parity or fabricate an LR packet. Old894 alignment evidence must not be relabelled892.

Historical expectations remain Airbase215553 domain cells/178 included positives of281 and Santa Maria135992/1910 of3312. Native metadata/actual new seed eligibility and independently reviewed paired-run evidence are future gates. Reference provenance/registration, exclusions and prior-use qualifications remain. Separate-CA materials/validity/provenance differences remain documentary evidence, not replacement shared inputs.

Retain each seed's confusion matrix [[TN,FP],[FN,TP]], F1, recall, precision, accuracy, IoU, reached/active counts, clocks and stopping metadata. Report mean and sample SD(n-1) only for exactly seeds42–46 in each of12groups. Missing/duplicate/failed/mixed-arm outcomes cannot yield a complete group or partial mean. Never pool incidents, models or clocks, select best seeds/steps, stop at reference area, tune or retrain.

Interpretation is retrospective footprint agreement conditional on an approximate reported fire-out interval. It predicts neither fire-out nor suppression. RF calibration, LR cutoff/reconstruction/lineage, F01 development/calibration use and Airbase development/prior-use/overlap/independence limitations remain explicit. No held-out claim is restored. Synthetic final-test access remains restricted.

## Release gates

Source preparation and authored synthetic tests require parent review and a separately approved owner-manual verification card. Future native readiness, per-run disabled configurations/resources, independently reviewed binding and fresh owner execution approval are separate gates. Historical approvals are consumed; no automatic next stage, retries, overwrites or cleanup.
