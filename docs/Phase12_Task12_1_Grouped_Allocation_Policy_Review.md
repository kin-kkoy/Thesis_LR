# Phase 12 Task 12.1 — Grouped-Allocation Policy Review

**Review date:** 2026-10-01 (Asia/Manila)  
**Dataset binding:** Set C CSV SHA-256 `7f0f1867445141f365687d127e449117f234348cec571990d70bc2b226ba40f6`  
**Mode:** read-only policy and bounded feasibility review  
**Recommendation:** **approve the numerical policy; revise and approve the protection architecture before implementation**

## Decision summary

The proposed exact component allocation of **11 train / 5 validation / 5 calibration / 5 final-test components** is feasible under every D-015 hard gate. It uses all 26 components exactly once and requires no random seed. The numerical policy is ready for owner approval.

The current split/training source is not ready for protected execution. `ModelTrainer` and `train_rf_optuna.py` load the complete observation CSV and require a complete row assignment before rejecting final-test roles. Releasing a three-role development manifest beside the readable full dataset would expose final-test membership by set difference. Copying final-test rows into a protected directory while leaving the full package and linkable development identities readable would not solve that problem.

Phase 12 should therefore approve the numbers now but require a source-only protected-export design and verified Windows identities/ACLs before authorizing allocation or split publication.

## Proposed numerical policy

### Hard constraints

| Role | Exact components | Minimum positives | Minimum negatives |
|---|---:|---:|---:|
| Train | 11 | 200 | 800 |
| Hyperparameter validation | 5 | 100 | 400 |
| Probability calibration | 5 | 100 | 400 |
| Untouched final test | 5 | 100 | 400 |

Every allocation must also satisfy all of the following:

1. Revalidate the exact CSV and package manifest hashes before constructing the graph.
2. Rebuild the D-029 graph from the package: complete event only where independently established; otherwise complete scenario family, plus 128×128 origin-(0,0) block, stable-cell, and observation-record edges.
3. Use all 26 components exactly once; no component, family, run, condition, replicate, timestep, stable cell, block, or repeated observation record may cross roles.
4. Cover material classes 1–5 in every role.
5. No component may supply more than 50% of a role's positives.
6. Preserve every conflicting-label observation. Predictor-vector equality remains an audit statistic and never becomes an allocation edge.
7. Fail closed on hash, schema, graph, count, class, material, concentration, provenance, ACL, destination, or no-overwrite drift.
8. Produce no split artifact unless every gate passes atomically. A failed attempt must publish neither partial assignments nor membership-bearing logs.
9. Keep final-test membership absent from development outputs and logs. Training/tuning/calibration code must not read the full assignment or protected rows.
10. Bind every future policy and artifact to this dataset hash, a versioned policy identifier, source revision, implementation hash, exact commands, and authorization reference.

### Deterministic selection

Use the existing component order contract `sha256(dataset_sha256 + ':' + leakage_component_id)`. Do not select or record a random split seed.

Among hard-feasible allocations, use this fixed lexicographic objective:

1. Minimize total absolute positive-count deviation from `[338, 153, 153, 153]` for train, validation, calibration, and final test.
2. Then minimize total absolute row-count deviation from `[4541, 2064, 2064, 2064]`.
3. Then break ties by canonical component-digest order and fixed role order `train`, `validation`, `calibration`, `final_test`.

These are soft balance targets derived from the package totals. They do not replace the hard adequacy gates.

### Feasibility result

An in-memory role-blind binary constraint solve over the 26 anonymous Phase 11 component summaries found the exact 11/5/5/5 cardinalities jointly feasible with every class, material, atomicity, and concentration constraint. The solver result was independently checked. No component-to-role decision, split seed, final-test membership, or solver decision vector was printed or persisted.

## Source-readiness finding

| Gate | Status | Finding |
|---|---|---|
| Numerical feasibility | **PASS** | Exact 11/5/5/5 allocation is feasible under all proposed hard gates. |
| Determinism | **PASS** | Existing component digest ordering supports a seedless deterministic policy. |
| Development loader isolation | **FAIL** | Current loaders consume the full CSV and complete assignment. They cannot consume a protected three-role projection without either failing membership checks or allowing complement inference. |
| Final evaluator isolation | **WARNING** | `evaluate_rf_final_test.py` rejects unauthorized execution but delegates storage and ACL enforcement to external controls; it does not verify a protected artifact or DACL itself. |
| Existing repository ACL | **FAIL for protected use** | The repository tree inherits access for `CodexSandboxUsers`; ordinary placement under `Thesis_RF/Code/output` is insufficient for final-test protection. |
| Selected vault filesystem | **NEEDS EVIDENCE** | The chosen volume must be verified as NTFS with stable owner/DACL behavior and no reparse-point path components before creation. |
| Custodian identity | **PASS, pending owner confirmation** | Observed repository owner is `LAPTOP-HDOU1IC7\gibgib`, SID `S-1-5-21-2674249978-1424284920-1115605237-1001`. |
| Evaluation identity | **NEEDS OWNER DECISION** | No separate evaluation-only Windows account/SID has been approved. |

No split currently exists, so actual final-test exposure or metric inflation has not occurred.

## Recommended protection architecture

### Owner-only vault

Use `C:\SetC_Protected\set_c_v1\` as the proposed non-repository vault, subject to owner confirmation and NTFS verification. Do not create it during policy review.

At creation time, the trusted owner-run setup must:

- reject non-NTFS volumes and all reparse-point/symlink path components;
- set the owner to `LAPTOP-HDOU1IC7\gibgib` / SID `...-1001`;
- disable inheritance on the vault and every membership-bearing child;
- grant explicit Full Control only to the owner SID and `NT AUTHORITY\SYSTEM`;
- remove inherited or explicit access for `CodexSandboxUsers`, `codexsandboxoffline`, `Users`, `Authenticated Users`, `Everyone`, and development principals;
- verify the effective DACL after every atomic creation and before every protected read;
- prohibit plaintext temporary files outside the vault and refuse overwrite;
- store cryptographic hashes and an ACL/owner attestation beside each protected artifact.

Do not grant a development account access to the vault. A separate evaluator SID may later receive read-only access only to the frozen final-test input subtree, after a separate final-evaluation authorization. Until that identity exists, final evaluation remains owner-run and disabled.

The vault should contain:

```text
C:\SetC_Protected\set_c_v1\
  custody\                 # protected canonical package or verified protected copy
  allocation\              # complete four-role mapping and policy evidence
  final_test\              # final-test rows and evaluator-only manifest
  attestations\            # hashes, owner/DACL checks, source/policy identities
```

### Development release

Before allocation execution, implement and independently validate a trusted exporter that runs under the custodian identity. It should read the protected canonical package and full mapping, then release only train/validation/calibration material to a separate development location.

The development release must:

- exclude all final-test rows and every full four-role mapping;
- contain three role-specific feature/label partitions or one three-role projection;
- replace linkable canonical family/scenario/run/cell/block/duplicate/provenance identities with vault-keyed pseudonymous identifiers, or expose no row identities where training does not need them;
- bind its manifest to the canonical CSV hash, policy version, protected full-mapping hash, source revision, role counts, feature/target schema, and exporter attestation without disclosing final membership;
- avoid releasing a row-level assignment whose complement can be derived from a readable canonical dataset;
- adapt development loaders to validate the projection and attestation without opening the canonical package, full mapping, or final-test files;
- run development from a sanitized checkout or execution context that cannot read the protected package, allocation, detailed membership evidence, or Git objects containing any future membership artifact.

Once membership exists, the development principal must not retain read access to the canonical package path. The immutable package bytes may remain unchanged while their ACL is protected under a separately approved operation.

## Owner choices required

The owner should decide these before approving protection implementation:

1. Confirm or revise vault path `C:\SetC_Protected\set_c_v1\`.
2. Confirm custodian `LAPTOP-HDOU1IC7\gibgib` / SID `S-1-5-21-2674249978-1424284920-1115605237-1001`.
3. Choose either owner-only final evaluation initially, or name/create a distinct evaluation-only Windows account and supply its SID.
4. Confirm that the development principal will use a separate sanitized execution context with no canonical-package or protected-report access after allocation.

## Inspection and execution record

The validator inspected the Phase 11 report; `PROJECT_STATE.md`; D-010, D-015, D-017, D-029, D-033, and D-034; `set_c_leakage.py`; `feature_pipeline.py`; `model_trainer.py`; `train_rf_optuna.py`; `evaluate_rf_final_test.py`; relevant publication/configuration source; repository and parent dataset ACL information; and the active Windows identity. Commands included `Get-Content`, `rg`, `Get-ChildItem`, `Get-FileHash`, `Get-Acl`, `whoami`, `Get-Volume`, Git status, and one `python -B` in-memory feasibility solve. Protected package and volume queries were denied in the validator sandbox where noted. No file, cache, split, role mapping, seed, ACL, model, or dataset was written or changed by the validator.

## Approval recommendation

Approve the 11/5/5/5 numerical policy and hard constraints. Revise the original protection proposal to require the owner-only vault, trusted unlinkable development export, loader adaptation, and loss of development-principal access to the canonical package. Do not authorize allocation implementation or ACL changes until the four owner choices above are resolved and the source-only protection design passes independent review.

## Ready-to-copy decision prompt

> I approve the Phase 12 Task 12.1 numerical grouped-allocation policy for Set C CSV SHA-256 `7f0f1867445141f365687d127e449117f234348cec571990d70bc2b226ba40f6`: exactly 11 train, 5 validation, 5 calibration, and 5 untouched final-test components; the D-015 class minima; material classes 1–5 in every role; component atomicity; and no component above 50% of a role's positives. Use the seedless deterministic digest order and the approved lexicographic balance targets. I do not yet authorize allocation execution, split publication, ACL changes, final-test membership access, training, tuning, calibration, threshold selection, or model evaluation. Revise the protection architecture for source-only implementation and independent validation: use an owner-only NTFS vault outside the repository, keep the complete mapping and final-test rows there, release only an unlinkable three-role development projection through a trusted exporter, adapt development loaders so they never read the canonical package or infer the protected complement, and keep final evaluation disabled. Before implementation, confirm the vault path, custodian SID, evaluation identity, sanitized development context, exact DACL verifier, atomic/no-overwrite behavior, and artifact/attestation schemas. Preserve all D-034 limitations and prohibited ground-truth, manuscript, LR, model, and final-test operations.
