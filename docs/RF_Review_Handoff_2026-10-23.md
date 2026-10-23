# RF review handoff - 2026-10-23

This branch shares the current RF source for the thesis partner's independent review. It does not authorize execution or accept new methodology. Phase 14 is closed and Phase 15 remains open according to the owner's supplied review reports.

## What is in this snapshot

The Phase 15 diff from `8962fbba4c924b183f091edcc3744c256f1e6357` contains simulation entrypoint/readiness safeguards, resource and output handling, probability-interface checks, an isolated exploratory additive-exposure R2 teacher and synthetic assessment adapter, disabled configurations, and focused tests. Earlier local commits also contain the Phase 9-14 work; the branch preserves that history.

`Thesis_LR/**` is partner-owned and read-only. No LR changes, datasets, models, rasters or generated run outputs are added by this snapshot commit. Both Phase 15 configurations are disabled on disk. Absolute input/model paths in the pilot configuration refer to the owner's machine, not portable bundled artifacts.

## Evidence and limits

The following milestones are owner-reported from the independent RF Validator reports relayed in chat. Raw evidence files are not bundled here, and no tests or experiments were rerun while preparing this branch:

- Santa Maria pilot engineering completion, output audit and spatial-result integrity were accepted. The pilot became inactive at timestep 7 with four extinguished cells. Primary mapped-building recall was about 0.209424% and F1 about 0.417973%; 100% precision rested on only four predicted positives. The target recall/F1 of 0.8 was not achieved.
- Local topology and four-row S0 diagnostics were accepted within their bounded scopes. Original paired random draws and trajectory were not retained; the complete historical extinction cause remains unresolved.
- R2 deterministic verification passed 157 selected cases and 22,464 unique numerical pattern records. This is distinct from the historical 195-test milestone; neither count means every current test was run together.
- The prespecified 1,024-run synthetic assessment passed its bounded integrity review. It establishes synthetic behavior under the reviewed fixtures, not physical realism, RF fidelity or improved real-site scores.

Relevant evidence anchors supplied by the owner:

| Evidence | Retained identity |
| --- | --- |
| Deterministic verification stdout | 17,543,144 bytes; SHA-256 `0f81496fea5aa401231361d4ecaaff8f61035e69977264a645ba5c6462eb94cb` |
| Synthetic assessment stdout | 10,044,394 bytes; SHA-256 `2f3fe1869e23bc949ac409c58622878e08f8c766eae6bcb71d293120656ffe4b` |

These anchors identify external evidence; they do not independently prove its contents. Earlier failures, consumed approvals, sampling limitations and reference provenance/date/coverage qualifications remain in force.

## Current task: shared RF/LR dataset contract

A versioned dataset using the exploratory candidate teacher is being specified. The owner has chosen to continue preparation; this snapshot does not establish that a new dataset has been generated or either model retrained. The historical Set C package and frozen RF release remain preserved.

The proposed shared CSV has eleven predictors in canonical order plus `Ignited`. Metadata and frozen split assignments belong outside the predictor columns. Grouped train/validation/calibration/test handling, final-test isolation, calibration and reporting rules still require contract review/adoption and later scoped implementation.

Read-only source inspection found an existing eleven-feature LR path at `Thesis_LR/Code/sandbox_kent/train_lr_stage_c.py`, paired with that sandbox's `main.py` and configuration. The main LR Code directory has a different ten-feature assembler. Stage C naming alone does not prove identity with the RF Set C dataset or the partner's actual command history.

The sandbox trainer currently uses a random 80/20 row split and chooses a reporting threshold using test labels. Its CA configuration zeros probabilities below 0.30 before stochastic comparison, whereas the reviewed RF path uses the predicted probability directly. CSV shape compatibility therefore does not establish shared training methodology or identical CA behavior. No LR policy or source change is adopted here.

## Suggested review scope

Review the source diff and proposed compatibility contract. Identify concrete correctness, leakage, schema or reproducibility issues with file/line evidence. Separate source findings from execution claims. Do not treat old run cards, historical authorization references, enabled legacy configurations or this branch as permission to run anything.

No simulation, data generation, fitting, tuning, dependency changes, artifact replacement or LR edits are authorized by this handoff. A revised teacher does not guarantee the desired spatial recall or F1.
