# Thesis RF/CA Handoff — Phase 9

**Prepared:** 2026-09-27 (Asia/Manila)
**Repository:** `C:\Users\gibgib\Thesis-Cellular-Automata`
**Branch:** `RF-11-Params`
**Reviewed HEAD:** `df65217b91c1624bbfcff82e95cca15a45a6df2c`
**Current phase:** Phase 9 — Production Dataset and Grouping Contract
**Immediate task:** Establish the reviewed Phase 9/10 revision, then seek separate authorization for one site-inventory execution

## 1. Purpose of this handoff

This document transfers the verified project state, accepted methodology, completed work, immutable evidence, open issues, authorization boundaries, and exact next actions for the Random Forest-assisted Cellular Automata thesis track.

It is a navigation and continuity document, not primary scientific evidence. Source code, approved decisions, immutable aggregate reports, Git revisions, configurations, and validator reports remain authoritative for their respective claims.

### Post-handoff status update — 2026-09-27

- Task 9.2's disabled aggregate-only site inventory is implemented and passed focused, complete-suite, and read-only RF Validator review.
- D-015 and D-016 record the population, adequacy, and aggregate-diagnostic boundaries.
- D-017 approves only disabled source implementation of the Task 9.3 publication, leakage, and protected-test architecture; 61 focused synthetic tests and the final read-only RF Validator review passed.
- D-018 approves deterministic class-1-to-5 ownership of mixed-material components and global reservation of their complete 128×128-block leakage footprints.
- No production inventory, family approval, teacher expansion, row publication, split assignment, model operation, final-test access, or ground-truth access is authorized by these source changes.

Sections describing Task 9.2 as future work preserve the original handoff-time record. For current authority and next actions, use D-015 through D-018 and `docs/ai/PROJECT_STATE.md`.

## 2. Repository ownership and safety boundaries

- `Thesis_RF/Code/**` is the active Random Forest implementation area, excluding datasets, trained models, generated results, and manuscripts unless separately authorized.
- `Thesis_LR/**` belongs to the partner and is read-only. Do not inspect or modify it unless the thesis owner explicitly authorizes that work.
- `docs/ai/PROJECT_STATE.md`, `docs/ai/DECISIONS.md`, and this handoff are shared coordination documents.
- Datasets, rasters, models, checkpoints, split manifests, generated reports, and experiment outputs are immutable or generated artifacts. Never overwrite them.
- `Thesis-try.qgz` is the owner's QGIS visualization project for inspecting ground truth and RF+CA output rasters. It is intentionally untracked and must not be deleted, cleaned, reverted, or added to provenance accidentally.
- `stack_ground_truth.tif` is forbidden during the current Phase 9 site-inventory and aggregate-population work.
- Manuscript changes are unauthorized.

## 3. Current Git and worktree state

At handoff creation, the visible pre-existing worktree state was:

```text
 M docs/ai/PROJECT_STATE.md
?? Thesis-try.qgz
```

The modified `PROJECT_STATE.md` records accepted Phase 8 evidence. `Thesis-try.qgz` is the preserved visualization project. Neither item should be cleaned or reverted.

This handoff adds only:

```text
docs/HandoffPhase9.md
```

No Phase 9 implementation has begun.

## 4. Thesis objective and current scientific scope

The thesis investigates predictive fire-spread modelling using Cellular Automata with a Random Forest component.

The active Set C task is:

> Predict whether an eligible mapped-building cell in state `STATE_NOT_YET_BURNING` at timestep `t` enters `STATE_IGNITED` at timestep `t+1`, using the canonical 11 predictors captured from the same state-at-`t` transition context.

The current authoritative label source is a versioned, model-free stochastic CA teacher. RF-generated transitions are not independent truth and cannot be used to retrain Set C except under an explicitly separate self-distillation claim.

Claims remain limited to the verified case-study domain, approved simulated scenarios, and the approved teacher. No observed temporal-ignition accuracy, real-world arrival-time accuracy, broad geographic generalization, or real-fire predictive validity has been established.

## 5. Approved methodology decisions

`docs/ai/DECISIONS.md` D-009 through D-014 are the durable approved basis:

- **D-009:** next-timestep ignition estimand; group-first evaluation; separate train, hyperparameter-validation, calibration, and untouched final-test roles; duplicate containment; immutable provenance.
- **D-010:** canonical Set C feature order and positive label; metadata outside predictors; reporting thresholds separated from stochastic CA probabilities.
- **D-011:** preserve the `all_eligible` population; prohibit uncorrected case-control sampling.
- **D-012:** accept only authorized simulated adjacent-state transition evidence; constrain claims to tested simulations and the case-study area.
- **D-013:** grid-north from-bearing wind; mapped-building final-footprint domain; BLAZING-at-`t` dynamic-neighbour timing; mutation-isolated transition observation.
- **D-014:** use only the model-free stochastic teacher for authoritative Set C training labels; require complete bounded whole-group collection and simulator-limited claims.

The owner has now approved additional Phase 9 policies, but they have not yet been appended as a new durable decision record:

- final 128×128-cell spatial containment blocks anchored at `(row 0, column 0)`;
- a controlled 26-family production-population design;
- aggregate-only population expansion before row publication;
- exact statistical adequacy gates;
- preservation and containment of conflicting-label duplicate groups.

Do not add a new decision entry casually. When the Task 9.2 scope includes shared-document maintenance, record these approved policies as the next durable decision without altering D-009–D-014.

## 6. Completed phases

### Phase 6 — Authoritative Set C training-data contract

Completed and accepted.

Key result:

- The authoritative Set C label generator is the explicitly versioned model-free stochastic CA teacher.
- Incumbent RF-assisted transitions are not independent predictive truth.
- Labels are integer `1` only for eligible `2→3` transitions.
- Dynamic neighbour features use only BLAZING neighbours at `t`.
- Full CA grids preserve five-state `np.int8` semantics.
- All 11 canonical predictors must be bound to the same `t→t+1` transition.
- Inactive completion is authoritative; safety-limit completion is censored; failure is non-authoritative.
- Publication must be immutable, versioned, non-overwriting, bounded, and disabled by default.

### Phase 7 — Model-free teacher and bounded collector

Completed and formally accepted.

Implemented source behaviour includes:

- a dedicated model-free teacher that cannot load, attach, or invoke an estimator;
- the D-014 ignition law using environmental base probability, directional-wind score, and one seeded Bernoulli draw;
- five-state `np.int8` transitions;
- vectorized non-wrapping SciPy neighbourhood operations;
- ignition coordinates and ignition-set hashes;
- immutable termination records;
- exact canonical 11-feature collection with `float32` predictors and integer targets;
- `all_eligible` retention;
- bounded batches, backpressure, and exact-observation retry without silent row loss;
- deterministic duplicate and spatial identities;
- a disabled-by-default synthetic-test-only publisher with atomic, hash-bound, non-overwriting behaviour;
- rejection of legacy binary-state publication for authoritative Set C.

Phase 7 verification:

- 160 synthetic RF tests passed with Python bytecode and pytest caching disabled.
- RF Validator Tasks 7.2 and 7.4 passed the acceptance gates.
- No production simulation, model deserialization, dataset publication, training, calibration, checkpoint, raster, result, or manuscript operation occurred.

### Phase 8 — Bounded Set C feasibility pilot

Completed and formally accepted, including the v2 extension.

Important source revisions:

- `3d8439c1271f0038342a0c525b12121d4b4f7a09`: accepted v1 pilot baseline and building/material normalization correction.
- `df65217b91c1624bbfcff82e95cca15a45a6df2c`: validated v2 pilot harness and execution source.

Focused v2 source verification:

```text
110 passed
Python bytecode disabled
pytest cache disabled
```

Accepted v2 report:

```text
Path:
Thesis_RF/Code/output/phase8/phase8_set_c_feasibility_pilot_v2.aggregate.json

Canonical payload SHA-256:
9a4f48ea6b2b7f882985d63ff3cad7ef3c785736e223860f82637fdd9e45cbf3

Whole-file SHA-256:
b62a7bf542d0ab74a3ac71399e85048b3169d0a04b33c95625aec3ee39339d17
```

Execution:

```text
Start: 2026-09-25T10:55:50.9684119Z
End:   2026-09-25T11:04:07.7176086Z
Exit:  0
Source revision: df65217b91c1624bbfcff82e95cca15a45a6df2c
source_dirty: false
```

Verified v2 results:

- 32/32 runs ended inactive, complete, authoritative, and count-matched.
- Final IGNITED and BLAZING counts were zero for every run.
- 74 expected transitions equalled 74 captured transitions.
- 384 eligible observations were counted: 26 positive and 358 negative.
- The eight v1 replay runs reconciled exactly.
- No backpressure drain, retry-limit failure, overflow, provenance mismatch, resource violation, censoring, partial run, or row loss occurred.
- Peak process RSS: `3,538,198,528` bytes.
- Minimum available physical memory: `4,298,653,696` bytes.
- Total elapsed runtime: `494.16115249996074` seconds.
- The v2 JSON was the sole persistent output.
- No estimator was loaded.
- No row-level data, split, model, checkpoint, CA raster, calibration record, result raster, or manuscript output was created.
- `stack_ground_truth.tif` was not opened.

Family totals, expressed as rows/positives:

| Family | Rows | Positives | Negatives |
|---|---:|---:|---:|
| `p8f01_cluster_322_m1` | 43 | 7 | 36 |
| `p8f02_cluster_1052_m3` | 15 | 1 | 14 |
| `p8f03_cluster_4974_m2` | 14 | 1 | 13 |
| `p8f04_cluster_160_m5` | 109 | 5 | 104 |
| `p8v2f05_cluster_4146_m3` | 78 | 2 | 76 |
| `p8v2f06_cluster_6201_m5` | 125 | 10 | 115 |

Leakage connectivity was identical for 32×32, 64×64, and 128×128 diagnostic blocks:

1. `p8f01`
2. `p8f02 + p8f03`
3. `p8f04`
4. `p8v2f05`
5. `p8v2f06`

All five components contained both labels. Phase 8 therefore proved mathematical four-role feasibility, but not statistical adequacy, spatial decorrelation, final block selection, or split assignments.

Duplicate summary:

```text
Unique groups: 62
Repeated groups: 46
Conflicting-label groups: 20
Maximum rows in one group: 27
```

## 7. Phase 9 Task 9.1 outcome

Task 9.1 was a read-only RF Validator review. No write, simulation, model, dataset, split, or artifact operation occurred.

Final decision:

```text
AGGREGATE-ONLY EXPANSION REQUIRED
```

Why:

- Only six scenario families and five components exist.
- The complete pilot has only 26 positives.
- Component positive counts are `7, 2, 5, 2, 10`.
- Wind coverage is limited to 10 km/h and directions 45°, 135°, and 315°.
- Material class 4 is absent even by current family identifier.
- No empirical spatial-dependence or decorrelation distance was measured.
- The evidence is insufficient for robust train, hyperparameter-validation, calibration, and untouched final-test roles.

## 8. Approved Phase 9 policy

The owner approved Task 9.1 without revisions.

### 8.1 Final block policy

Use 128×128-cell blocks anchored at row 0 and column 0:

```text
block_row = floor(cell_row / 128)
block_col = floor(cell_col / 128)
```

On the verified 3-m grid this is 384×384 metres.

This is an owner-approved conservative containment policy because it is the largest tested candidate. It is not an empirically measured independence distance. Future claims must not describe it as proof of decorrelation or geographic independence.

### 8.2 Intended controlled production population

After exact site discovery and approval:

- retain the six existing approved ignition designs;
- add four independent ignition families for each material class 1–5;
- add 20 families, for 26 total;
- use 17 controlled wind conditions per family;
- use two deterministic replicates per condition;
- run 34 simulations per family, 884 total.

Wind conditions:

```text
0 km/h: 0° calm
10 km/h: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
20 km/h: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
```

This factorial population is an approved controlled design, not a measured local-weather distribution.

Seed formula:

```text
seed = 9_000_000
       + 10_000 * family_ordinal
       + 100 * wind_condition_ordinal
       + replicate_ordinal
```

Where:

```text
family_ordinal = 1..26
wind_condition_ordinal = 0..16
replicate_ordinal = 1,2
```

Scenario ID:

```text
{family_id}__w{direction:03d}_s{speed:02d}
```

Run order:

1. Existing six families in their current order.
2. New families ordered by material class, rank, then component ordinal.
3. Calm, then eight 10-km/h directions, then eight 20-km/h directions.
4. Replicate 1, then replicate 2.

Represented population is limited to the verified case-study mapped-building domain, the 26 approved ignition designs, material classes 1–5, the approved teacher, and controlled calm/10/20-km/h winds. It does not represent actual weather frequency, arbitrary ignitions, observed fire transitions, or other geographies.

### 8.3 Approved future adequacy gates

Before row publication, an aggregate-only analysis must prove an allocation exists with:

| Future role | Minimum components | Per-component gate | Positives | Negatives |
|---|---:|---|---:|---:|
| Train | 8 | ≥5 positive and ≥5 negative | 200 | 800 |
| Hyperparameter validation | 5 | ≥5 positive and ≥5 negative | 100 | 400 |
| Calibration | 5 | ≥5 positive and ≥5 negative | 100 | 400 |
| Untouched final test | 5 | ≥5 positive and ≥5 negative | 100 | 400 |

Also required:

- at least 23 authoritative leakage components overall;
- every qualifying component independently contains both labels;
- each prospective role can cover material classes 1–5;
- no component is divided;
- no component supplies more than 50% of one role's positives;
- all family winds, seeds, timesteps, blocks, and duplicate identities remain with their leakage-connected component;
- role feasibility may be assessed, but role identities must not yet be assigned or exposed;
- final-test access remains prohibited until the estimator, class weighting, tuning result, calibration method, and reporting threshold are frozen.

If any gate fails: stop, publish no rows or split manifest, and add separately approved independent ignition families. Do not repair inadequacy merely by adding seeds to existing families.

### 8.4 Conflicting-label duplicates

Conflicting outcomes from repeated seeded Bernoulli trials are expected evidence, not automatically data errors.

Required treatment:

- retain every `all_eligible` observation;
- preserve both labels;
- report unique, repeated, and conflicting groups;
- do not majority-vote, deduplicate, relabel, delete, or downsample;
- connect complete scenario-family, 128×128 block, and duplicate-group identities;
- place the complete connected component into one future role;
- investigate only provenance, feature-identity, timing, or schema inconsistencies.

## 9. Task 9.2 — exact authorized scope

Task 9.2 is authorized only for source implementation and synthetic testing of a disabled-by-default, aggregate-only site-inventory preflight.

It is not yet authorized to run against production rasters.

Identifiers:

```text
inventory_id:
phase9_set_c_site_inventory_v1

future immutable destination:
Thesis_RF/Code/output/phase9/
phase9_set_c_site_inventory_v1.aggregate.json
```

The implementation must bind:

```text
Grid shape: 5489 × 6896
CRS: EPSG:32651
Resolution: 3 m

Grid identity:
26e9da1104da2caf15039b5f9fbcf719fea506adcee95f9c2da247e8a9e2bc46

Authoritative-domain identity:
f9bb9dbfe00c61a5b08d228bcbec4b4313853a47f4ca2ea82b7a4359d81814a4
```

It must also bind all approved environmental raster identities from the Phase 8 v2 report.

### Site-discovery algorithm

1. Revalidate the exact material/building context of all six existing ignition cells.
2. Deterministically label authoritative-domain 8-connected components with constant non-wrapping boundaries.
3. Exclude components containing existing ignition points.
4. Admit a component only when:
   - its size is 256–6,000 cells;
   - its complete footprint uses no 128×128 block occupied by an existing or already selected component;
   - its ignition cell is simulation-valid, mapped-building, and material class 1–5.
5. Produce eight candidates for each material class 1–5:
   - four primary;
   - four reserve;
   - 40 total.
6. Rank components by descending size, then ascending deterministic component ordinal.
7. Choose the ignition cell by:
   - greatest count of authoritative Moore neighbours;
   - smallest squared distance to the component centroid;
   - smallest row;
   - smallest column.
8. Use proposed family IDs:

```text
p9f_m{material_class}_r{rank}_c{component_ordinal}
```

### Permitted aggregate fields

- proposed family ID;
- deterministic component ordinal and size;
- ignition row/column;
- ignition-set SHA-256;
- verified material class;
- mapped-building status;
- simulation-valid status;
- 128×128 block-footprint SHA-256;
- occupied-block count;
- bound grid, domain, configuration, source, and raster provenance;
- completeness and contract status.

### Prohibited output

The inventory must not contain or create:

- feature rows or predictor matrices;
- transition labels;
- cell inventories;
- masks or grids;
- rasters or checkpoints;
- models;
- dataset files;
- split roles or manifests;
- calibration records;
- manuscript output;
- teacher transitions.

It must never access `stack_ground_truth.tif`.

It must fail without a report if:

- any bound identity differs;
- any existing ignition fails revalidation;
- any material class has fewer than eight valid candidates;
- a candidate violates the component, block, domain, or material rules;
- an overwrite would occur;
- any unexpected artifact would be created.

### Minimal implementation principle

Reuse the existing Phase 8 grid-identity, provenance, atomic JSON, no-overwrite, resource-probe, and fail-closed helpers wherever possible. Do not duplicate a second framework or add dependencies. Task 9.2 must receive an explicit owned-file list before editing.

Likely minimum surface, subject to the Task 9.2 ownership prompt:

- one site-inventory module;
- one small disabled CLI entry point only if the existing runner cannot safely host it;
- one focused synthetic contract test file;
- the smallest required configuration addition;
- no unrelated refactor.

## 10. Future aggregate population expansion — approved design, execution unauthorized

After the site inventory is implemented, validated, separately executed, and the exact family manifest is approved, a later aggregate-only expansion may use:

```text
pilot_id:
phase9_set_c_population_adequacy_expansion_v1

future destination:
Thesis_RF/Code/output/phase9/
phase9_set_c_population_adequacy_expansion_v1.aggregate.json
```

Approved future bounds:

| Bound | Value |
|---|---:|
| Safety limit | 300 transitions |
| New-family component ceiling | 6,000 cells |
| Per-run row ceiling | 1,800,000 |
| Total row ceiling | 1,418,667,000 |
| Batch size | 16,384 rows |
| Maximum retained rows | 932,864 |
| Maximum retained NumPy bytes | 176,311,296 |
| Retry limit per observation | 58 |
| Configured concurrency | 0 |
| Maximum active runs | 1 |
| Launch available-memory floor | 6,442,450,944 bytes |
| Runtime available-memory floor | 2,147,483,648 bytes |
| Process RSS ceiling | 6,442,450,944 bytes |
| Free-disk floor | 2,147,483,648 bytes |
| Per-run runtime limit | 900 seconds |
| Total runtime limit | 21,600 seconds |

Worst-case row bound:

```text
Known component cells = 19,085

34 × 300 × (19,085 + 20 × 6,000)
= 1,418,667,000 rows
```

This is a hard stop ceiling, not an expected dataset size.

Any memory, disk, runtime, provenance, count, backpressure, estimator-isolation, input-identity, unexpected-output, or termination failure must stop the operation.

Only inactive, complete, count-matched runs with zero final IGNITED and BLAZING cells can be authoritative. Censored, failed, partial, duplicated, missing, sampled, or row-losing runs invalidate the fixed matrix.

## 11. What remains unauthorized

The Phase 9 approval does not authorize:

- running the site inventory against production inputs;
- automatically approving discovered sites;
- implementing or running the 884-run expansion before the manifest gate;
- row-level Set C dataset publication;
- split-role assignment;
- RF training or hyperparameter tuning;
- calibration;
- final-test access;
- model publication;
- CA checkpoint or raster generation;
- final result rasters;
- manuscript changes;
- `stack_ground_truth.tif` access;
- any write under `Thesis_LR/**`.

## 12. Required next actions

### Immediate: Task 9.2 implementation

1. Define a narrow RF Engineer ownership list.
2. Follow the required Context7 gate before implementation if the RF Engineer profile is used.
3. Implement only the disabled aggregate site-inventory source path.
4. Add focused synthetic tests for:
   - default-disabled execution;
   - exact grid/domain/raster binding;
   - non-wrapping 8-connected components;
   - component-size filtering;
   - deterministic ordering and ignition selection;
   - existing-site exclusion;
   - 128×128 block disjointness;
   - four-primary/four-reserve coverage for each material class;
   - fail-closed insufficient-candidate behaviour;
   - atomic no-overwrite aggregate publication in temporary test storage;
   - absence of prohibited row-level fields and outputs;
   - rejection of `stack_ground_truth.tif`, teacher, model, split, and training paths.
5. Run only the focused authorized synthetic tests with bytecode and pytest cache disabled.
6. Review the diff and confirm writes stayed within the owned files.

### Then: read-only validation

The RF Validator must return PASS for:

- default-disabled behaviour;
- exact identity and provenance binding;
- deterministic component and site selection;
- correct material-class coverage;
- existing-site and block-footprint exclusion;
- aggregate-only output;
- atomic non-overwrite publication;
- no teacher/model/ground-truth access;
- ownership compliance.

### Then: separately authorized inventory execution

Only after validation:

1. Establish a clean identifiable Git revision.
2. Confirm the destination does not exist.
3. Perform memory and disk preflight.
4. Run the inventory once.
5. Validate the sole immutable aggregate report read-only.
6. Preserve report payload and whole-file SHA-256 values.

### Then: exact manifest approval

The owner must separately approve:

- the 20 primary families;
- the reserve families;
- every coordinate and ignition hash;
- verified material classes;
- component ordinals and sizes;
- block-footprint identities;
- final family ordering.

No teacher execution may occur before this approval.

### Later: population-adequacy expansion

Implement and validate the aggregate-only 884-run expansion, then obtain separate execution authorization. Proceed toward dataset publication only if every adequacy gate passes.

## 13. Known issues and evidence gaps

- Phase 8 proved mathematical role feasibility but not statistical adequacy.
- The exact material raster values at the six existing ignition points must be revalidated by the site inventory; family suffixes are not sufficient evidence.
- Material class 4 lacks an existing family even by identifier.
- Exact new ignition coordinates and hashes do not exist yet.
- Spatial decorrelation was not measured; 128×128 is a conservative policy.
- The 884-run expansion may still fail the positive-count or component gates; it must remain aggregate-only until adequacy is proven.
- Calibration method, RF search space, trial budget, tuning objective, class-weight choice, reporting-threshold objective, and final evaluation procedure remain unset.
- Historical Set A/B/C metrics still require complete primary provenance before reuse.
- `docs/ai/PROJECT_STATE.md` predates the Task 9.1 approval and should be updated when shared-document maintenance is next explicitly in scope.
- The approved Phase 9 methodology has not yet been appended as a durable decision after D-014.

## 14. Evidence labels and confidence

- **VERIFIED:** Phase 7 source behaviour and acceptance.
- **VERIFIED:** Phase 8 v1/v2 artifact identities, execution facts, counts, resources, and validation outcomes.
- **VERIFIED:** Phase 9 Task 9.1 concluded that aggregate-only expansion is required.
- **APPROVED POLICY:** 128×128 block containment, controlled wind population, seed formula, adequacy gates, duplicate policy, and future bounds.
- **NEEDS EVIDENCE:** exact new-family coordinates, hashes, material classes, and block footprints.
- **BLOCKED:** production site-inventory execution, teacher expansion, row publication, splits, training, calibration, final-test access, model/raster publication, and manuscript claims.

Confidence is high for recorded Phase 7/8 evidence and current insufficiency. Confidence in 128×128 as an independence distance is intentionally not claimed. The adequacy minima are owner-approved conservative policy rather than empirical conclusions from Phase 8.

## 15. Safe restart checklist

Before continuing in a new session:

1. Read `AGENTS.md` and `Thesis_RF/AGENTS.md`.
2. Read `docs/ai/PROJECT_STATE.md`, this handoff, and D-009–D-014.
3. Check `git status` and preserve all existing work.
4. Confirm branch and HEAD rather than assuming `df65217` remains current.
5. Verify the Phase 8 v2 whole-file and payload hashes before relying on the artifact.
6. Do not touch `Thesis-try.qgz`.
7. Do not inspect or modify `Thesis_LR/**`.
8. Do not access `stack_ground_truth.tif`.
9. Obtain an explicit Task 9.2 owned-file list before editing.
10. Keep Task 9.2 source-only and synthetic until the validator passes.

## 16. Recommended next prompt shape

The next user request should assign the RF Engineer a narrow Task 9.2 implementation scope, explicitly list owned files, require the Context7 gate, preserve current worktree changes, restrict tests to synthetic temporary paths, and prohibit production inventory execution. After implementation, a separate RF Validator task should inspect only those files and run only the directly affected tests.

Do not skip directly to the 884-run expansion. The exact site manifest is the next evidence gate.
