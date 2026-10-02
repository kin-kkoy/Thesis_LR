# Durable Decisions — Secondary Brain

## Record contract

- **Purpose:** Append-only record of material project decisions, alternatives, rationale, approval, and supporting evidence.
- **Authority:** This records approved decisions; it does not turn an unsupported implementation or experiment claim into evidence.
- **Scope:** Shared repository coordination and the Random Forest thesis track. Partner-owned Logistic Regression decisions require the partner's explicit participation and are not recorded here by inference.
- **Last updated:** 2026-09-20T17:42:17+08:00.
- **Evidence rule:** Link each decision to repository evidence or current official documentation. Use `NEEDS VERIFICATION` when required evidence is absent.
- **Stale-information rule:** Decisions remain historical records. Do not rewrite an accepted entry; append a new entry that supersedes it and identify the triggering evidence.
- **Size policy:** Keep the active file below 32 KiB. When it approaches the limit, move closed entries without alteration to a dated file under `docs/ai/archive/` and leave an index link here.

## Decision template

```text
## D-### — Short title
- Date:
- Status: PROPOSED | APPROVED | SUPERSEDED | REJECTED
- Scope:
- Owner/approver:
- Decision:
- Rationale:
- Alternatives considered:
- Evidence:
- Consequences and follow-up:
- Supersedes / superseded by:
```

## D-001 — Repository-local Markdown is the canonical project memory

- **Date:** 2026-09-10
- **Status:** APPROVED
- **Scope:** Shared repository and Random Forest thesis track
- **Owner/approver:** Thesis owner, through Phase 4 approval
- **Decision:** Use `docs/ai/PROJECT_STATE.md` as the compact current-state index and this file as the append-only decision log. Codex local memory or external note tools may supplement them but do not replace them.
- **Rationale:** Repository-local Markdown is readable by Codex, reviewable with implementation changes, portable, and aligned with Git history.
- **Alternatives considered:** Codex-only memory, Obsidian-only memory, Notion-only memory, and duplicated canonical copies.
- **Evidence:** `AGENTS.md`; `docs/ai/PROJECT_STATE.md`; approved Secondary Brain specification.
- **Consequences and follow-up:** Keep summaries concise and link to primary evidence. Evaluate a human-facing viewer separately without creating another authoritative copy.
- **Supersedes / superseded by:** None.

## D-002 — Automation validates memory maintenance but does not author thesis claims

- **Date:** 2026-09-10
- **Status:** APPROVED
- **Scope:** Shared Codex workflow
- **Owner/approver:** Thesis owner, through approval of the automated update mechanism
- **Decision:** A repository-local Codex completion hook detects material files changed during the session and requests a memory update when neither memory file changed. The agent remains responsible for writing and verifying semantic summaries.
- **Rationale:** Automatic summarization after every edit creates token cost, churn, and a risk of recording intermediate or unsupported claims. A completion validator enforces consideration once per coherent implementation.
- **Alternatives considered:** AI summarization after every write, unconditional memory rewriting, and a manual-only reminder.
- **Evidence:** `.codex/hooks.json`; `.codex/hooks/secondary_brain.py`; [official Codex hooks documentation](https://learn.chatgpt.com/docs/hooks).
- **Consequences and follow-up:** The hook permits an explicit `Memory impact: unchanged — <specific reason>.` exception for genuinely non-material changes. Human review remains required for evidence-sensitive statements.
- **Supersedes / superseded by:** None.

## D-003 — Session baselines and a bounded startup digest limit redundant work

- **Date:** 2026-09-10
- **Status:** APPROVED
- **Scope:** Shared Codex workflow
- **Owner/approver:** Thesis owner, through approval of the Secondary Brain
- **Decision:** On session start, inject only the marked compact digest and record fingerprints of relevant pre-existing working changes in the operating-system temporary directory. On completion, compare against that baseline instead of treating every dirty file as work from the current task.
- **Rationale:** The repository already contains user-owned uncommitted work. A per-session delta avoids false reminders, while a bounded digest reduces full-repository rereading.
- **Alternatives considered:** Injecting the entire memory file, comparing only final `git status`, or storing baseline files inside the repository.
- **Evidence:** `docs/ai/PROJECT_STATE.md`; `.codex/hooks/secondary_brain.py`; [official Codex hooks documentation](https://learn.chatgpt.com/docs/hooks).
- **Consequences and follow-up:** The digest is navigation context, not proof. Changed or stale sources still require focused inspection and targeted verification.
- **Supersedes / superseded by:** None.

## D-004 — Defer a separate experiment index until its need is demonstrated

- **Date:** 2026-09-10
- **Status:** APPROVED
- **Scope:** Random Forest thesis memory
- **Owner/approver:** Thesis owner, through approval of the minimal Phase 4 surface
- **Decision:** Do not create `docs/ai/EXPERIMENT_INDEX.md` yet. Keep current experiment leads in `PROJECT_STATE.md` and existing findings files until verified experiment manifests become numerous enough to justify a separate index.
- **Rationale:** The current provenance problem is missing reliable metadata, not a lack of another summary file. Creating an index now would duplicate uncertain claims.
- **Alternatives considered:** Create a complete experiment index immediately or duplicate every findings table into project memory.
- **Evidence:** `docs/ai/PROJECT_STATE.md`; `docs/findings_SetA.md`; `docs/findings_SetB.md`; `docs/findings_SetC.md`; `Thesis_RF/THESIS_RF_CODEX_HANDOFF.md`.
- **Consequences and follow-up:** Revisit after clean, identified experiment manifests exist.
- **Supersedes / superseded by:** None.

## D-005 — Use project-scoped Context7 MCP with an MCP-only RF Engineer gate

- **Date:** 2026-09-13
- **Status:** APPROVED
- **Scope:** Shared Codex agent tooling
- **Owner/approver:** Thesis owner, through approval of the post-dry-run refinements
- **Decision:** Configure Context7 as a project-scoped MCP server in `.codex/config.toml`. The `rf_engineer` must use the MCP tools before implementation and may produce only a source-only plan when they are unavailable. Other agents prefer MCP but may use the repository-approved `ctx7` CLI fallback.
- **Rationale:** The dry run showed that the Engineer's strict MCP requirement could not be satisfied in the active session, while the Researcher bypassed Context7 for a current library-behavior claim. Project-scoped configuration preserves the Engineer's original boundary and makes the documentation workflow consistent.
- **Alternatives considered:** Permit an automatic Engineer CLI fallback; remove the Context7 requirement; configure Context7 only in a user's global Codex settings.
- **Evidence:** `.codex/config.toml`; `.codex/agents/rf-engineer.toml`; `.codex/agents/thesis-researcher.toml`; [official Codex MCP documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).
- **Consequences and follow-up:** A new or reloaded trusted Codex session must verify that `resolve-library-id` and `query-docs` are callable. Configuration does not make the server callable retroactively in the current session.
- **Supersedes / superseded by:** None.

## D-006 — Keep specialist delegation bounded and auditable

- **Date:** 2026-09-13
- **Status:** APPROVED
- **Scope:** Shared multi-agent orchestration
- **Owner/approver:** Thesis owner, through approval of the post-dry-run refinements
- **Decision:** Every specialist assignment must define a bounded file/source surface, command or test budget, output expectation, and write boundary. Agents must disclose inspected evidence, commands, scope expansion, and whether they attempted a write. Validators must separate source-level risk, observed exposure, and measured impact.
- **Rationale:** The controlled dry run produced strong evidence while avoiding broad repository scans, long tests, and file changes. Persisting those constraints reduces token and coordination waste and prevents overstatement.
- **Alternatives considered:** Unbounded role prompts; running every specialist for every milestone; relying only on sandbox settings to prevent scope drift.
- **Evidence:** `AGENTS.md`; `.codex/agents/*.toml`; controlled four-agent dry-run report from 2026-09-13.
- **Consequences and follow-up:** The primary agent should invoke only roles needed for a bounded milestone and review their outputs before acceptance.
- **Supersedes / superseded by:** None.

## D-007 — Wait for Context7 during MCP tool-catalog construction

- **Date:** 2026-09-13
- **Status:** APPROVED
- **Scope:** Shared Codex MCP startup configuration
- **Owner/approver:** Thesis owner, through the request to resolve Context7 before the next task
- **Decision:** Set `mcp_optional_startup_grace_ms = 0` in the trusted project configuration so Codex waits for Context7's configured 30-second startup timeout instead of omitting the server after the default one-second optional grace. Set the Windows user `CODEX_HOME` to the existing `C:\Users\gibgib\.codex` directory so host-side Codex CLI commands resolve the intended state directory.
- **Rationale:** `codex mcp list` showed Context7 enabled, but an ephemeral agent received no Context7 tools. Current Codex configuration documentation identifies a one-second default grace for optional MCP servers; `npx` cold startup can exceed that interval. The managed verification shell also lacked a resolvable home for Codex CLI state.
- **Alternatives considered:** Mark Context7 required for every project session; move the MCP server to user-global configuration; retain per-command environment overrides; permit the Engineer to use only the CLI.
- **Evidence:** `.codex/config.toml`; host-side `codex mcp list`; failed pre-fix ephemeral tool-call check; [official Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
- **Consequences and follow-up:** New sessions may spend several additional seconds building the MCP tool catalog. Verify one successful `resolve-library-id` and `query-docs` call before RF implementation.
- **Supersedes / superseded by:** None.

## D-008 — Accept fresh-session Context7 catalog verification

- **Date:** 2026-09-13
- **Status:** APPROVED
- **Scope:** Shared Codex MCP readiness
- **Owner/approver:** Thesis owner, through the request to resolve Context7 before the next task
- **Decision:** Treat the Context7 exposure blocker as resolved for new trusted tasks after a fresh Codex session reports the server connected and exposes both required tools. Do not expect an already-running task to acquire a changed tool catalog retroactively.
- **Rationale:** After setting the user-scoped `CODEX_HOME` and disabling the optional MCP startup grace, `codex doctor --json` resolved `CODEX_HOME` to `C:\Users\gibgib\.codex` and reported the MCP configuration locally consistent. A separately opened trusted Codex CLI session then reported `context7: connected (2 tools)`; `/mcp verbose` identified those tools as `resolve-library-id` and `query-docs`.
- **Alternatives considered:** Continue treating all Context7 use as blocked; perform another model-backed health check; rely only on `codex mcp list`, which verifies configuration but not a connected runtime catalog.
- **Evidence:** `.codex/config.toml`; user-scoped Windows `CODEX_HOME`; `codex doctor --json`; fresh-session `/mcp`; fresh-session `/mcp verbose`; [official Codex MCP documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli); [official Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
- **Consequences and follow-up:** Start the next thesis task in a new or reloaded trusted Codex task. The `rf_engineer` should still stop and report a blocker if the two MCP tools are absent from that particular task.
- **Supersedes / superseded by:** Operationally closes the verification follow-up in D-005 and D-007; it does not supersede their configuration decisions.

## D-009 — Adopt RF methodology and provenance implementation contracts

- **Date:** 2026-09-15
- **Status:** APPROVED
- **Scope:** Random Forest methodology, provenance, evaluation, and future implementation under `Thesis_RF/Code/**`
- **Owner/approver:** Thesis owner, by explicit approval of the Phase 1–3 review decisions
- **Decision:** Adopt the following contracts:
  1. Set C is the active development candidate. Sets A and B remain independent historical provenance chains. Their artifacts and metrics must not be merged or directly compared unless target, feature, grouping, split, evaluation, and provenance contracts are equivalent.
  2. The primary RF estimand is next-timestep ignition of an eligible cell. A final burned/unburned raster is reserved for CA-level outcome validation. Any final-burn classifier must be identified and evaluated separately as an outcome or susceptibility model.
  3. Evaluation uses group-first separation by complete events when available, otherwise by complete scenarios plus spatial blocks. Random row splits cannot support spatial- or event-generalization claims.
  4. Model fitting, hyperparameter validation, calibration and threshold selection, and final testing use separate roles. The final test remains untouched until model, weighting, calibration, and threshold choices are frozen.
  5. Class weighting is selected during training/validation; thresholding is selected on calibration data. A fixed recall, F1, or other operating target requires a separately approved cost, safety, or policy rationale.
  6. Estimator probability discrimination/calibration metrics and CA raster/temporal metrics are reported separately. A binary hard mask is not a valid input for probability ROC-AUC.
  7. Invalid/nodata cells are excluded before metric calculation and cannot enter as true negatives. Raster validity and physical burnability use separate documented masks. For the approved physical domain, non-building cells are excluded because the currently masked ground truth defines them as unburned; therefore an eligible cell must be both raster-valid and a building cell.
  8. Duplicate feature groups remain within one split. Conflicting-label groups are investigated rather than automatically deleted.
  9. Each Set A/B/C chain requires an independent immutable provenance manifest covering dataset identity/hash, ordered schema, generator revision, configuration, command, seed, event/scenario/spatial identifiers, split identity, weighting and threshold policy, model/output identity, and metric record. Existing datasets, models, rasters, and results are never overwritten; corrections create versioned replacements only after approval.
  10. Missing lineage remains `NEEDS VERIFICATION`; owner approval of a contract does not validate unsupported historical claims.
- **Rationale:** The controlled provenance, dataset/leakage, and methodology reviews found incompatible 10/10/11-feature histories, incomplete lineage, source-permitted row-level dependence and evaluation reuse, a binary-mask AUC mismatch, and invalid-pixel inclusion. These contracts define the evidentiary and evaluation boundaries required before results can support thesis claims.
- **Alternatives considered:** Merge historical sets by filename; retain random row splitting; optimize thresholds on the reported test; treat class weighting and thresholding as interchangeable; combine estimator and CA metrics; retain invalid cells as negatives; allow non-building cells in the burnable domain; overwrite historical artifacts; or accept missing lineage by assumption.
- **Evidence:** Thesis-owner approval dated 2026-09-15; `docs/ai/PROJECT_STATE.md`; `Thesis_RF/Code/dataset_generator.py`; `Thesis_RF/Code/generate_multi_scenario.py`; `Thesis_RF/Code/modules/model_trainer.py`; `Thesis_RF/Code/train_rf_optuna.py`; `Thesis_RF/Code/validation_engine.py`; the controlled Phase 1 provenance, Phase 2 dataset/leakage, and Phase 3 methodology reviews supplied in the approving task.
- **Consequences and follow-up:** Future implementation contracts must trace each code change and validation check to these clauses. Set C remains a candidate rather than a verified end-to-end experiment until lineage is established. This decision authorizes contract formalization and RF source planning only; it does not authorize training, dataset regeneration, simulation, model deserialization or replacement, raster generation, manuscript modification, immutable-artifact mutation, or changes under `Thesis_LR/**`.
- **Supersedes / superseded by:** Narrows the current-integration statement in `PROJECT_STATE.md` by making Set C explicitly a candidate and establishes mandatory methodology boundaries for future RF work. It does not supersede D-001 through D-008.

## D-010 — Refine Set C source contracts and separate reporting thresholds from stochastic CA inference

- **Date:** 2026-09-15
- **Status:** APPROVED
- **Scope:** Source-only Phase 4 implementation under `Thesis_RF/Code/**`
- **Owner/approver:** Thesis owner, by explicit pre-Phase-4 approval
- **Decision:** Use the exact ordered Set C predictors `slope_risk`, `proximity_risk`, `building_presence`, `material_risk`, `material_class`, `wind_speed`, `wind_sin`, `wind_cos`, `neighbor_burning_count`, `composite_flammability`, and `wind_weighted_score`. The positive label is integer `1`, meaning an eligible building cell ignites at `t+1` from state/features at `t`; probability-column selection must match this label through estimator `classes_` and fail closed. Preserve `set_id`, `experiment_id`, `event_id`, `scenario_id`, `run_id`, `seed`, `timestep_t`, `cell_row`, `cell_col`, stable `grid_id`, `duplicate_group_id`, and `spatial_block_id` as non-predictor observation metadata, while recording `split_role` in a separate immutable split manifest. Use deterministic raster-anchored spatial blocks, but leave block scale unset until spatial-correlation, raster-resolution, extent, group-feasibility, and class-coverage evidence exists. Keep train, validation, calibration, and final-test roles separate; determine allocation only after group inventory, using nested grouped CV if independent groups are insufficient. Select any probability-calibration method only with grouped non-test evidence. A frozen hard threshold applies only to classification reporting and requires a separately approved objective; stochastic CA ignition uses calibrated probabilities unchanged with a recorded seed and has no threshold. Retain the D-009 building-only domain, metric separation, immutable artifacts, and per-set provenance requirements.
- **Rationale:** The pre-Phase-4 review verified the Set C order and the need for explicit identifiers, group-first isolation, positive-class lookup, separate spatial masks, continuous-score probability metrics, and immutable manifests. It also found that applying a classification threshold inside the stochastic CA would conflate hard reporting with Bernoulli ignition and that numeric block, split, calibration, cost, and threshold choices remain unsupported.
- **Alternatives considered:** Embed split roles by rewriting the observation dataset; treat metadata as predictors; assume probability column index 1; choose an unevidenced block size or split percentage; preselect a calibration method; retain a test-optimized threshold; apply a hard threshold to stochastic CA; or multiply calibrated model probability by an additional post-model wind factor.
- **Evidence:** Thesis-owner approval dated 2026-09-15; D-009; `docs/ai/PROJECT_STATE.md`; approved Phase 1–3 reports; pre-Phase-4 methodology validation; current official scikit-learn documentation for `classes_`, `predict_proba`, grouped splitting, and disjoint calibration; and the authorized RF source files named in the Phase 4 contract.
- **Consequences and follow-up:** Source-only implementation and small synthetic/unit/static verification are authorized in the eight approved RF source/config files plus `Thesis_RF/Code/tests/**`. Numeric block scale, split allocation, calibration method, cost/safety objective, and classification threshold remain unset. Training, dataset generation, simulation, model deserialization/replacement, raster/result creation, manuscript changes, immutable-artifact mutation, and `Thesis_LR/**` writes remain separately gated.
- **Supersedes / superseded by:** Clarifies D-009 clauses 4–6 for separate split manifests, positive-class semantics, and threshold-free stochastic CA; it does not rewrite or supersede D-009.

## D-011 — Preserve the Set C eligible population and classify transition evidence

- **Date:** 2026-09-15
- **Status:** APPROVED
- **Scope:** Set C observation sampling, class-weight candidates, transition-source provenance, and evidence-gated methodology validation
- **Owner/approver:** Thesis owner, by explicit post-Phase-4 contract approval
- **Decision:** Retain every eligible transition cell in the observation dataset using the explicit `all_eligible` sampling policy; for groups assigned to calibration, retain every eligible cell from those groups so calibration represents that eligible-cell population. Uncorrected case-control sampling is not approved. Limit validation-selected Random Forest class-weight candidates to `none`, `balanced`, and `balanced_subsample`; select among them using grouped train/validation evidence, never a fixed numeric class-cost ratio. Require every state-at-`t`/state-at-`t+1` pair to declare `observed` or `authorized_simulated` with source-specific provenance, and keep observed and simulated evidence separately identifiable rather than silently combining them. Accept the evidence-gated wind, ROI, spatial-block, evaluation-protocol, tuning, calibration, and reporting resolutions only as a basis for further read-only validation.
- **Rationale:** Retaining the defined eligible-cell population preserves its observed prevalence for probability calibration. Treating class weighting as a grouped development-data selection separates statistical imbalance handling from an owner-approved cost policy. Explicit transition-source classification prevents observed and simulated outcomes from being interpreted as equivalent evidence.
- **Alternatives considered:** Uncorrected case-control sampling; fixed numeric class weights; a default selected weight without grouped evidence; unlabeled transition sources; silent pooling of observed and simulated transitions; or prematurely freezing evidence-dependent methodology values.
- **Evidence:** Thesis-owner approval dated 2026-09-15; D-009 and D-010; `Thesis_RF/Code/config/default_experiment.yaml`; `Thesis_RF/Code/dataset_generator.py`; `Thesis_RF/Code/generate_multi_scenario.py`; `Thesis_RF/Code/modules/model_trainer.py`; `Thesis_RF/Code/train_rf_optuna.py`; synthetic contract tests; current scikit-learn documentation for `RandomForestClassifier.class_weight` and disjoint calibration using a frozen fitted estimator.
- **Consequences and follow-up:** The RF defaults and source contracts must require `all_eligible`, expose only the three approved class-weight tokens, require explicit grouped selection, and fail closed on missing or mixed transition provenance. Wind convention, event/ROI extent, spatial block size, split allocation, nested-CV structure, tuning ranges/objective/budget, calibration method, classification threshold, cost ratio, confidence interval, and stochastic replicate count remain unapproved. Dataset generation, training, model operations, simulation, raster/result creation, manuscript changes, immutable-artifact mutation, and `Thesis_LR/**` writes remain separately gated. Read-only provider/CRS, event/AOI, and group/class evidence collection is authorized.
- **Supersedes / superseded by:** Adds approved sampling, weighting-candidate, and transition-source details without modifying or superseding D-009 or D-010.

## D-012 — Narrow active Set C to authorized simulated transitions and approximate final-footprint validation

- **Date:** 2026-09-17
- **Status:** APPROVED
- **Scope:** Active Set C transition provenance, evidentiary claims, study population, and CA-level final-footprint validation
- **Owner/approver:** Thesis owner, by explicit approval of the narrowed Set C methodology
- **Decision:** Active Set C accepts only `authorized_simulated` adjacent state-at-`t`/state-at-`t+1` pairs; observed temporal fire rasters are not required because suitable observations are unavailable. The RF target remains eligible-building-cell ignition at `t+1` from features and state at `t`, using `all_eligible`. Every simulated pair must bind simulator identity/version and code revision, configuration and input hashes, seed, timestep, scenario, run, grid, controlled wind parameters, and an authorization reference. The manually delineated `stack_ground_truth.tif` is a satellite-derived approximate reference burned-area mask used only for CA final-footprint agreement; it cannot support observed next-timestep calibration, temporal ignition, or arrival-time claims. The intended study population is the common-valid aligned raster footprint restricted to mapped building cells, subject to a read-only check that slope-invalid exclusions do not remove reference-burned cells. Claims are limited to the case-study area and tested simulated scenarios; unseen-event and broad real-world generalization claims are not approved. Wind remains a controlled scenario parameter, but one direction convention must be centralized and approved before simulation.
- **Rationale:** The thesis lacks sufficiently precise temporal fire observations to justify an observed-transition workflow. Explicit simulated provenance preserves the next-step learning contract without misrepresenting synthetic transitions as observations, while the satellite-derived manual mask remains useful as an approximate final-outcome reference.
- **Alternatives considered:** Requiring unavailable observed transition rasters; treating the manual final-burn mask as temporal ground truth; abandoning transition provenance; or using the final reference footprint as an RF row-level label.
- **Evidence:** Thesis-owner approval dated 2026-09-17; D-009 through D-011; current source-only Set C contract; repository validation entry point for `stack_ground_truth.tif`.
- **Consequences and follow-up:** Remove the active observed-source branch and observed-source tests, retain fail-closed simulated provenance, and label validation outputs as final-footprint agreement only. A simulator transition-snapshot/capture interface is not yet implemented. Before any simulation or dataset generation, approve and centralize the wind convention and complete the read-only overlap/domain preflight. Simulation production, dataset generation, training, splitting, calibration, threshold selection, model creation, raster/result creation, and manuscript modification remain unauthorized.
- **Supersedes / superseded by:** Supersedes only D-011's allowance of `observed` sources for active Set C. D-011's `all_eligible`, class-weight, separation, and no-silent-pooling requirements remain in force; D-009 and D-010 remain unchanged.

## D-013 — Fix wind, final-footprint-domain, and in-memory transition-observation contracts

- **Date:** 2026-09-20
- **Status:** APPROVED
- **Scope:** Random Forest-assisted Cellular Automata source contracts under `Thesis_RF/Code/**`
- **Owner/approver:** Thesis owner, by explicit approval
- **Decision:**
  1. `wind.direction_deg` is a meteorological from-bearing clockwise from raster grid north: 0/90/180/270 degrees mean from north/east/south/west. Propagation is opposite the bearing; raster rows increase southward and columns eastward. `wind_sin` and `wind_cos` encode that bearing as `sin(theta)` and `cos(theta)`. One versioned conversion must serve all CA/feature calculations and pass cardinal and wraparound tests before simulation. Provenance records convention, units, north reference, and schema version. True/geodetic-north input requires a recorded grid-north conversion when convergence is material.
  2. The primary CA outcome is `common-valid mapped-building final-footprint agreement`, masked by `simulation_environment_valid ∩ prediction_valid ∩ approximate_reference_valid ∩ building_raster_valid ∩ mapped_building`. The simulation-valid mask is the run mask or a hash-bound reconstruction. Invalid/non-building cells never enter the confusion matrix. The building raster controls eligibility/evaluation; material disagreement fails preflight absent a separately approved rule. Reports give total/included reference-valid burned counts and auditable, non-double-counted exclusions; reference-invalid cells are unknown. Complete-final-footprint claims require inactive termination. A timestep-limited run with active IGNITED/BLAZING cells is censored; the union of IGNITED, BLAZING, and EXTINGUISHED describes only the footprint through stopping time. Broader domains require separate approval.
  3. The CA may expose an optional, synchronous, source-only in-memory transition observer. For transition `t` to `t+1`, `blazing_mask_t = (state_t == STATE_BLAZING)`; `eligible_mask_t = simulation_valid_mask ∩ mapped_building_mask ∩ (state_t == STATE_NOT_YET_BURNING) ∩ (blazing_neighbor_count_t > 0)`; and `newly_ignited_mask_t1 = eligible_mask_t ∩ (state_t1 == STATE_IGNITED)`. State-at-`t` dynamic features use only `blazing_mask_t`; the positive label is an eligible cell entering `STATE_IGNITED` at `t+1`. A generic binary burning pair cannot substitute unless an approved adapter proves equivalence. Positives must be a subset of eligible cells, no eligible/positive cell may be invalid or off-building, and `timestep_t1` must equal `timestep_t + 1`.
  4. Observer arrays are copies or read-only views; metadata is immutable or deep-copied. Observer presence, no-op use, or attempted mutation must not alter CA state, RNG, or outcomes. Observer failure cannot trigger checkpoints or artifact writes. Live-run, hash-bound provenance includes capture schema/version; set/experiment/event/scenario/run/grid IDs; both timesteps; seed; wind values/convention/north reference; simulator/version/revision/dirty state; configuration/input hashes; authorization; grid shape/CRS/transform; domain-mask identity/hash; environment/execution record; and capture-source ID.
  5. This decision authorizes contract implementation and focused synthetic/static verification only. The observer performs no file I/O, serialization, dataset generation, raster creation, model operation, or result publication. Simulation production, dataset generation, training, splitting, calibration, threshold selection, model creation or replacement, raster/result creation, manuscript changes, immutable-artifact mutation, and `Thesis_LR/**` writes remain unauthorized.
- **Rationale:** Source duplicates wind conversion, omits the simulator-valid domain from final-footprint evaluation, and lacks transition observation. Live CA uses BLAZING cells for features but newly IGNITED cells for labels; D-009–D-012 require this distinction and complete provenance.
- **Alternatives considered:** A to-bearing; independent wind formulas; unproved domain proxies; all-reference-pixel evaluation; silent material/building reconciliation; one binary fire-state mask; in-CA transition-raster writes; observer-triggered checkpoints; or observer publication.
- **Evidence:** Thesis-owner approval dated 2026-09-20; D-009–D-012; `docs/ai/PROJECT_STATE.md`; relevant RF automata, feature, loading, dataset, scenario, validation, orchestration, and configuration source; read-only technical-lead and RF-validator reviews.
- **Consequences and follow-up:** Centralize wind conversion and test cardinal orientation; establish and bind authoritative simulation-valid and mapped-building masks; make final-footprint exclusions and run completeness auditable; and implement a non-writing, provenance-bound transition observer with temporal, mutation-isolation, and noninterference tests. Dataset adapters and all artifact-producing activity require separate approval.
- **Supersedes / superseded by:** Clarifies and operationalizes the unresolved wind, evaluation-domain, and transition-capture follow-ups in D-009 through D-012 without superseding their estimand, provenance, artifact, or authorization boundaries.

## D-014 — Adopt the authoritative Set C model-free training-source and collection contract

- **Date:** 2026-09-22
- **Status:** APPROVED
- **Scope:** Authoritative Set C simulator, transition-label source, grouping, collection, provenance, and claim boundaries
- **Owner/approver:** Thesis owner, by explicit approval of the Phase 6 Task 6.1 decision-ready contract
- **Decision:**
  1. **Authoritative source.** Active Set C transition labels shall be generated only by a versioned model-free stochastic rule teacher implementing the no-model transition branch of `FireAutomata`. The teacher must fail closed if any estimator is attached, loaded, or invoked. The current `pure_ca_baseline.RuleBasedAutomata` is not authoritative because its ignition is deterministic and its wind effect is nondirectional.
  2. **Other simulator sources.** Incumbent RF-assisted transitions may be retained only in a separate provenance chain explicitly labeled `teacher_imitation` or `self_distillation`; they shall not be described as independent truth and shall not enter authoritative Set C. No external simulator is approved until its identity, version, executable or source revision, behavior, state adapter, input lineage, and provenance are separately established and approved.
  3. **Wind and ignition law.** The teacher shall use `simulated_wind.v1`, meteorological from-bearing relative to projected grid north. For each eligible cell, `p_base` shall be the versioned clipped rule-based environmental probability, `p_effective = clip(p_base * wind_weighted_score_t, 0, 1)`, and ignition shall be the seeded Bernoulli event `U < p_effective`. The RNG algorithm and software environment shall be recorded. Wind shall not be reapplied after this probability.
  4. **State and domain.** Full states shall use `ca_five_state_int8.v1`: 1 non-burnable, 2 not yet burning, 3 ignited, 4 blazing, and 5 extinguished. Legal transitions are `1->1`, `2->2|3`, `3->3|4`, `4->4|5`, and `5->5`. The authoritative domain is `simulation_valid_mask & mapped_building_mask`; state 1 shall exactly cover its complement. Building/material disagreement shall fail preflight unless a later approved reconciliation rule exists.
  5. **Observation and feature timing.** Each row shall represent one eligible cell for exactly one synchronous adjacent transition. All 11 ordered `set_c.v1` predictors shall be bound to state and environment at `t`; dynamic predictors shall use only BLAZING cells at `t`; positive label 1 shall mean an eligible state-2 cell at `t` enters state 3 at `t+1`; and `timestep_t1` shall equal `timestep_t + 1`. Rebinding features, wind, masks, or states after capture is prohibited.
  6. **Population.** Every eligible row shall be retained under `all_eligible`. Case-control sampling, class balancing by row deletion, silent row dropping, and loss caused by collector pressure are prohibited. Class weighting remains a later grouped model-selection decision.
  7. **Group identities.** Complete `event_id` groups shall be separated when available. Otherwise, split assignment shall use leakage-connected components covering complete scenario families, raster-anchored spatial blocks or stable cells, and duplicate groups. `run_id`, seed, ignition identity, and timestep are nested replicate or repeated-measure identities and do not independently justify cross-split separation. All timesteps of a run and all runs sharing a scenario family shall remain in one split role. Ignition coordinates and a canonical ignition-set hash shall be explicit provenance.
  8. **Run completeness.** Authoritative runs shall be captured from the post-seeding state through inactive termination, with no IGNITED or BLAZING cells remaining. A safety-limit termination shall be recorded as censored and excluded from authoritative training; it may be retained only as separately labeled diagnostic evidence. Termination reason, final timestep, final active-state counts, expected and captured transition counts, and completeness status shall be recorded.
  9. **Bounded memory.** The repository-owned collector shall enforce explicit row and byte limits, retain at most one full-grid transition observation at a time, assemble complete 11-feature eligible rows immediately, use bounded batches and backpressure, and release full-grid arrays after assembly. On a separately approved publication path, batches shall use versioned per-run staging partitions and be published only after count, hash, schema, and provenance validation through an atomic no-overwrite commit. Failure or censoring shall not publish a partial authoritative dataset.
  10. **Provenance.** Each run and row batch shall bind the set, experiment, event, scenario family, run, seed, ignition identity, both timesteps, grid, simulator ID/version, source revision and dirty state, configuration and input hashes, transition parameters, wind manifest, domain-mask IDs and hashes, environment/execution records, RNG identity, capture/collector/feature/target schema versions, termination record, authorization reference, and immutable artifact identity.
  11. **Claims.** A resulting RF may support only fidelity and held-out grouped predictive performance relative to the approved model-free teacher within the case-study mapped-building domain and tested simulator scenarios, winds, ignition designs, and parameter ranges. It may not support claims of observed real-fire accuracy, causal physical validity, arrival-time accuracy, unseen-event or geographic generalization, superiority to the teacher, external-simulator equivalence, or independent validation of RF-assisted transitions. Row-level RF metrics and final spatial CA metrics shall remain separate.
  12. **Authorization boundary.** Approval of this decision authorizes contract implementation and synthetic/static verification only. Production simulation, artifact generation, dataset publication, split creation, training, calibration, model publication, final-raster evaluation, and manuscript claims require later explicit approvals.
- **Rationale:** A read-only Task 6.1 review found that the current deterministic pure-CA baseline lacks directional wind and Bernoulli ignition, normal orchestration always attempts to load the incumbent RF, the D-013 observer-to-row adapter contains only two dynamic fields rather than the complete 11 predictors, and no bounded-memory repository-owned collector or auditable termination record exists. The no-model branch of `FireAutomata` is the only reviewed path that supplies a non-circular, seeded stochastic transition mechanism consistent with D-013 wind, state, and observer timing.
- **Alternatives considered:** Adopt the deterministic `pure_ca_baseline.py`; treat incumbent RF-assisted transitions as independent truth; approve an unidentified external simulator; permit row sampling or silent loss under memory pressure; split runs or timesteps across roles; or include active timestep-censored runs in authoritative training.
- **Evidence:** Thesis-owner approval dated 2026-09-22; D-009–D-013; `docs/ai/PROJECT_STATE.md`; `Thesis_RF/Code/config/default_experiment.yaml`; `Thesis_RF/Code/orchestrator.py`; `Thesis_RF/Code/pure_ca_baseline.py`; `Thesis_RF/Code/modules/automata_engine.py`; `Thesis_RF/Code/modules/feature_pipeline.py`; `Thesis_RF/Code/modules/transition_observer.py`; `Thesis_RF/Code/dataset_generator.py`; `Thesis_RF/Code/generate_multi_scenario.py`; Phase 6 Task 6.1 read-only RF Validator review.
- **Consequences and follow-up:** Implement a dedicated fail-closed no-model teacher runner, repository-owned bounded-memory 11-feature collector, ignition identity, and run-completeness record under a separately reviewed RF-owned source plan. Keep publication disabled by default. Group/block feasibility, production simulation, artifact publication, splitting, training, calibration, model publication, final evaluation, and manuscript claims remain separately gated.
- **Supersedes / superseded by:** Narrows D-012's generic `authorized_simulated` source to the approved model-free teacher for authoritative Set C and adds collection, grouping, completeness, and claim boundaries. It does not supersede D-009–D-013.

## D-015 — Adopt the Phase 9 production-population, spatial-grouping, and adequacy contract

- **Date:** 2026-09-27
- **Status:** APPROVED
- **Scope:** Final Set C spatial containment policy, controlled production population, aggregate-only site discovery and adequacy gates, duplicate handling, and authorization boundaries
- **Owner/approver:** Thesis owner, by explicit approval of the Phase 9 Task 9.1 contract without revisions
- **Decision:**
  1. **Spatial containment.** Future Set C grouping shall use raster-anchored 128×128-cell blocks with origin row 0 and column 0. On the verified 3-m grid this is 384×384 m. This is the largest tested conservative containment policy, not a measured spatial-decorrelation or independence distance, and claims shall state that limitation.
  2. **Current evidence is inadequate.** Phase 8's six families, five leakage-connected components, 384 rows, and 26 positives establish mathematical four-role feasibility only. They are insufficient for production row publication or statistically adequate train, hyperparameter-validation, calibration, and untouched final-test roles.
  3. **Site inventory.** Before teacher expansion, a disabled aggregate-only inventory shall discover four primary and four reserve candidate families for each material class 1–5. Candidates must come from deterministic non-wrapping 8-connected authoritative-domain components of 256–6,000 cells, exclude existing ignition components, and occupy no 128×128 block used by an existing or earlier selected component. Component ordering and ignition selection shall follow the approved deterministic size, ordinal, Moore-neighbour, centroid-distance, row, and column rules. Exact coordinates, hashes, material contexts, and the final 20-family manifest require later human approval.
  4. **Controlled population.** After manifest approval, the intended population shall contain the six existing ignition designs plus four new independent families per material class, for 26 families. Each family shall use calm at 0 km/h plus eight cardinal/intercardinal directions at both 10 and 20 km/h, with two deterministic replicates per condition: 17 conditions, 34 runs per family, and 884 runs total. This is a controlled factorial population, not a measured local-weather distribution.
  5. **Run identities.** Seeds shall follow `9_000_000 + 10_000*family_ordinal + 100*wind_condition_ordinal + replicate_ordinal`. Scenario IDs shall follow `{family_id}__w{direction:03d}_s{speed:02d}`. Existing families precede new families; new families sort by material class, rank, and component ordinal; winds sort calm, 10-km/h directions, then 20-km/h directions; replicate 1 precedes replicate 2.
  6. **Adequacy gates.** Before row publication, an aggregate-only feasibility analysis must establish at least 23 authoritative leakage components. Every qualifying component must contain at least five positives and five negatives. A feasible unexposed allocation must exist with: train ≥8 components/200 positives/800 negatives; hyperparameter validation ≥5/100/400; calibration ≥5/100/400; untouched final test ≥5/100/400. Every prospective role must cover material classes 1–5, no component may supply more than 50% of a role's positives, no component may be divided, and no role identities may yet be assigned or exposed.
  7. **Duplicate conflicts.** All `all_eligible` observations and conflicting seeded-Bernoulli outcomes shall be retained and reported. Majority voting, deduplication, relabeling, deletion, and downsampling are prohibited. Complete scenario-family, 128×128 block, and duplicate-group edges form leakage-connected components that must remain wholly within one future role.
  8. **Failure and bounds.** An inadequate role, identity drift, resource breach, row/count loss, backpressure failure, unexpected output, estimator access, forbidden ground-truth access, censoring, or partial/failed run shall stop the applicable operation and prohibit row or split publication. The approved future aggregate expansion bounds are 300 transitions; 6,000 new-family cells; 1,800,000 rows per run; 1,418,667,000 total rows; 16,384-row batches; 932,864 retained rows; 176,311,296 retained NumPy bytes; retry limit 58; one active run; 6,442,450,944-byte launch-memory floor and RSS ceiling; 2,147,483,648-byte runtime-memory and free-disk floors; 900 seconds per run; and 21,600 seconds total.
  9. **Authorization boundary.** This approval authorizes only source implementation and synthetic testing of the disabled aggregate-only site inventory. Production inventory execution, automatic manifest approval, teacher expansion, row publication, split assignment, training, tuning, calibration, final-test access, model/raster publication, manuscripts, `stack_ground_truth.tif` access, and `Thesis_LR/**` writes remain separately gated.
- **Rationale:** Phase 8 v2 showed five both-label leakage components under all three diagnostic block sizes, but only 26 positives, component-positive counts of 7/2/5/2/10, one wind speed, three directions, and no measured spatial-dependence range. A production population requires broader independent ignition and controlled-wind coverage plus explicit role-level adequacy before publishing rows.
- **Alternatives considered:** Promote the Phase 8 pilot directly; choose a smaller block while claiming independence; multiply seeds within existing families; delete conflicting duplicates; expose split roles during feasibility; or publish rows before proving adequacy.
- **Evidence:** Phase 9 Task 9.1 read-only RF Validator report; accepted Phase 8 v2 aggregate with payload SHA-256 `9a4f48ea6b2b7f882985d63ff3cad7ef3c785736e223860f82637fdd9e45cbf3` and whole-file SHA-256 `b62a7bf542d0ab74a3ac71399e85048b3169d0a04b33c95625aec3ee39339d17`; D-009–D-014; thesis-owner approval dated 2026-09-27.
- **Consequences and follow-up:** Implement and validate the disabled aggregate-only site inventory, then separately authorize one inventory execution. Validate and approve the exact candidate manifest before implementing or running the aggregate population-adequacy expansion. Dataset publication and all model-development operations remain blocked until the adequacy gates pass and receive separate approval.
- **Supersedes / superseded by:** Extends D-009 and D-014 with the approved Phase 9 population, grouping, adequacy, and staged-authorization contract. It does not supersede D-009–D-014.

## D-016 — Authorize only the bounded aggregate population-adequacy diagnostic

- **Date:** 2026-09-27
- **Status:** APPROVED
- **Scope:** Phase 9 aggregate-only population-adequacy evidence and the boundary before any row-level publication architecture
- **Owner/approver:** Thesis owner, after review of Phase 9 Tasks 9.1, 9.2, and 9.3
- **Decision:**
  1. **Phase 9 synthesis.** Task 9.1's `AGGREGATE-ONLY EXPANSION REQUIRED` result remains controlling. Task 9.2 supplies a disabled aggregate-only site-inventory preflight that passed source and synthetic validation, subject to the unresolved mixed-material ordering policy and production prerequisites. Task 9.3 found the model-free teacher and bounded collector suitable foundations but found production publication, joint leakage-graph construction, split allocation, and protected-test isolation unimplemented.
  2. **Only authorized expansion.** Approve `phase9_set_c_population_adequacy_expansion_v1` solely as the bounded aggregate diagnostic defined by D-015, with the sole future persistent output `Thesis_RF/Code/output/phase9/phase9_set_c_population_adequacy_expansion_v1.aggregate.json`. Source implementation and synthetic temporary-path tests are authorized. One later diagnostic execution is authorized only after the checked-in source is clean, the Task 9.2 inventory is separately authorized and validated, the mixed-material rule is resolved, and the exact family manifest is human-approved.
  3. **Aggregate-only boundary.** The diagnostic may report only aggregate run completeness, class/material/component totals, leakage connectivity, duplicate summaries, resource/provenance checks, and whether the D-015 adequacy gates admit an unexposed allocation. It shall not retain or publish predictor rows, labels, cell inventories, split identities, role assignments, models, rasters, checkpoints, or calibration/threshold records. Failed, censored, partial, count-mismatched, provenance-invalid, resource-breaching, or unexpected-output runs make the diagnostic non-authoritative and prohibit promotion.
  4. **No production publisher.** This decision does not authorize implementation or execution of a production row publisher, an authoritative row-level dataset package, a split manifest, training, tuning, class-weight selection, calibration, reporting-threshold selection, protected final-test access, model/raster publication, manuscript changes, `stack_ground_truth.tif` access, or `Thesis_LR/**` writes.
  5. **Task 9.3 remains proposed.** The Task 9.3 package, leakage, allocation, and protected-test architecture is decision-ready but not adopted by this decision. Any later source-only approval must keep final numerical allocation targets and the concrete protected-storage/ACL mechanism separately gated. D-015's diagnostic adequacy minima remain approved evidence gates; they do not by themselves select final allocation targets or expose role membership.
- **Rationale:** Phase 8 and Task 9.1 establish that five leakage components and 26 positives are too sparse for defensible four-role row publication. Building a production publisher before the bounded aggregate expansion proves adequacy would create unused complexity and risk exposing rows or final-test membership prematurely.
- **Alternatives considered:** Implement the production publisher now; publish rows while measuring adequacy; assign provisional split roles; treat Phase 8 mathematical feasibility as statistical adequacy; or delay all source work. These were rejected in favor of the smallest evidence-producing step.
- **Evidence:** Phase 9 Task 9.1 RF Validator result; D-015; Task 9.2 source implementation and final read-only RF Validator PASS with one policy warning; Task 9.3 read-only architecture audit; `docs/HandoffPhase9.md`; `docs/ai/PROJECT_STATE.md`; thesis-owner approval dated 2026-09-27.
- **Consequences and follow-up:** Resolve the Task 9.2 mixed-material policy, establish a clean reviewed revision, separately authorize and validate the inventory execution, and approve the exact family manifest. Then implement and synthetically validate the aggregate adequacy harness before its single bounded execution. Production publisher and row-level architecture remain deferred until aggregate adequacy passes and receive a separate approval.
- **Supersedes / superseded by:** Narrows D-015's next action by authorizing the bounded aggregate diagnostic while preserving every publication, split, model, final-test, and protected-storage gate. It does not supersede D-009–D-015.

## D-017 — Approve the Phase 9 Task 9.3 architecture for source-only implementation

- **Date:** 2026-09-27
- **Status:** APPROVED
- **Scope:** Authoritative Set C package, joint leakage graph, development/final-test isolation, and source-only Phase 10 boundaries
- **Owner/approver:** Thesis owner, by explicit approval of the Task 9.3 architecture contract
- **Decision:**
  1. **Source-only implementation.** Authorize RF-owned source changes and synthetic temporary-path tests for the authoritative Set C publication, leakage, and protected-test architecture. Every checked-in publisher, collection, training-write, and protected-final-test gate remains disabled by default.
  2. **Package contract.** Use `set_c_authoritative_package.v1`, package version `1`, and destination pattern `output/datasets/{package_id}/v{package_version}`. The immutable package contains canonical `set_c.v1.csv`, per-run/per-batch staging and manifests, a hash-bound package manifest, exact row/class totals, termination records, and source/configuration/environment/RNG/input/mask/wind/teacher/schema/authorization provenance. Promotion is same-parent atomic and no-overwrite; only inactive, complete, authoritative, count-matched, fully drained runs may be promoted.
  3. **Row contract.** Preserve the exact ordered 11 `set_c.v1` predictors. Collection and publication staging require `np.float32` predictors and `np.int8` labels; the CA state contract remains five-state `np.int8`. Canonical row metadata records set, experiment, event, scenario family, scenario, run, seed, both timesteps, cell row/column, grid, stable cell, duplicate group, spatial block, ignition-set, run-identity, and provenance hashes. `split_role` is forbidden in the observation dataset.
  4. **Leakage contract.** Construct joint connected components by complete event when explicitly established and otherwise complete scenario family, plus spatial-block, stable-cell, and duplicate-group edges. Runs, seeds, and timesteps remain nested within scenario family. Null or empty identities fail closed. The deterministic allocation-policy record binds the dataset hash but contains no assignments or numerical role targets.
  5. **Protected-test isolation.** Development loaders accept only train, validation, and calibration assignments and cannot load final-test membership. Conflicting-label duplicate observations are retained when their complete duplicate group stays within one role. Final-test probability evaluation lives in a separate fail-closed interface; its CLI remains disabled and it requires a later approved protected-storage mechanism and authorization.
  6. **Separate gates.** Final numerical role adequacy/allocation-target constraints and the concrete protected-storage/permissions/ACL mechanism remain independently unapproved. No value or mechanism may be inferred from this decision.
  7. **Execution boundary.** This decision does not authorize production dataset publication, teacher expansion, row-dataset creation, production leakage/split execution, any role assignment, training, tuning, class-weight or calibration selection, threshold selection, final-test membership/data access, model/raster/checkpoint/manuscript publication, `stack_ground_truth.tif` access, `Thesis_LR/**` writes, or changes to `Thesis-try.qgz`.
- **Rationale:** Task 9.3 supplied a decision-ready contract, and the owner subsequently authorized only its fail-closed source implementation. Separating package mechanics, leakage containment, development loading, and final evaluation now prevents accidental final-test exposure while leaving data adequacy and real access control as explicit later decisions.
- **Evidence:** Owner approval dated 2026-09-27; D-009–D-016; `modules/set_c_publication.py`; `modules/set_c_leakage.py`; `modules/feature_pipeline.py`; `modules/model_trainer.py`; `train_rf_optuna.py`; `evaluate_rf_final_test.py`; 61 focused synthetic tests; final read-only RF Validator PASS after provenance and null-identity repairs.
- **Consequences and follow-up:** Keep every execution gate closed. Complete D-016's aggregate-evidence prerequisites before seeking publication execution. Separately approve exact package identity/destination, numerical allocation targets, protected storage/ACL, and each later split/model/final-test operation.
- **Supersedes / superseded by:** Supersedes only D-016's deferral of Task 9.3 source implementation. It does not relax D-015/D-016 aggregate prerequisites or authorize any production, split, model, or final-test operation.

## D-018 — Approve deterministic ownership of mixed-material inventory components

- **Date:** 2026-09-27
- **Status:** APPROVED
- **Scope:** Phase 9 Set C site-inventory candidate ownership and global leakage-footprint reservation
- **Owner/approver:** Thesis owner, by explicit approval of the mixed-material policy gate
- **Decision:** Process material classes in order 1, 2, 3, 4, then 5. A connected component containing multiple eligible material classes belongs to the first eligible class in that order. Once assigned, its complete 128×128-block leakage footprint is reserved globally; neither the component nor any overlapping reserved block may be reused by another material class, primary family, or reserve family. The inventory must record complete material composition, assigned class, component identity, ignition identity, and reserved block footprint.
- **Claim boundary:** This is only a deterministic inventory-selection rule. It does not establish material prevalence, statistical representativeness, physical independence, or real-fire accuracy.
- **Authorization boundary:** This approval does not authorize inventory execution, candidate/family approval, teacher or aggregate expansion, row publication, split assignment, model operations, final-test access, `stack_ground_truth.tif`, manuscripts, `Thesis_LR/**`, or changes to `Thesis-try.qgz`.
- **Evidence:** Owner approval dated 2026-09-27; D-015–D-017; `modules/set_c_site_inventory.py`; mixed-material synthetic contract; Phase 9 Task 9.2 RF Validator warning.
- **Consequences and follow-up:** The mixed-material policy prerequisite is closed. Establish one reviewed revision and pass the full synthetic suite before seeking separate authorization for one inventory execution.
- **Supersedes / superseded by:** Resolves the mixed-material policy gate left open by D-015 and D-016 without changing any execution or artifact boundary.

## D-019 — Record Gate 3 site-inventory execution and validation block

- **Date:** 2026-09-27
- **Status:** BLOCKED
- **Scope:** One authorized Phase 9 Set C aggregate-only site-inventory execution
- **Owner/approver:** Thesis owner, by explicit one-run production authorization
- **Decision:** Execute the approved inventory once from the clean reviewed revision `d7e455de15a953f826e101ab77f810a1f156b86b`; retain only its aggregate report; and do not approve the candidate/family manifest because independent validation found 21 candidates whose connected components contain a lower eligible material class than the reported assignment. This violates D-018's deterministic class-1-to-5 ownership rule.
- **Evidence:** Preflight passed with 7,711,215,616 available bytes of memory, 91,008,909,312 free bytes of disk, matching approved raster hashes, clean source provenance, absent destination, and preserved `Thesis-try.qgz`. Aggregate report `Thesis_RF/Code/output/phase9/phase9_set_c_site_inventory_v1.aggregate.json`; payload SHA-256 `56521134204b04fb464a12529e6d8f91eec62aec1e689f610731f5bd7d57e53a`; whole-file SHA-256 `804fc0626be7577735b78e5c71802cd0f4a8bbe0d373842e31ada37f99be9859`; 40 candidates (8 per material), 6 existing sites, 105 unique reserved blocks, zero block overlaps, 23,901 components, sole output, and no row-level payload. Independent recomputation from the four approved rasters identified the 21 ownership violations.
- **Authorization boundary:** This record does not authorize regeneration, candidate reassignment, family-manifest approval, aggregate expansion, teacher execution, row publication, split assignment, model operations, final-test access, `stack_ground_truth.tif`, manuscripts, `Thesis_LR/**`, or changes to `Thesis-try.qgz`.
- **Consequences and follow-up:** Gate 4 is blocked. Repair the source selection rule and rerun the full synthetic suite in a new clean revision, then obtain a new explicit one-run authorization before any replacement inventory execution. The existing report is evidence only and must not be promoted or used to approve families.
- **Supersedes / superseded by:** Adds execution evidence to D-018 without changing the approved mixed-material policy or any later-phase authorization boundary.

## D-020 — Record validated Gate 4 v2 site inventory without manifest approval

- **Date:** 2026-09-28
- **Status:** VALIDATED; AWAITING MANIFEST APPROVAL
- **Scope:** One authorized Phase 9 Set C aggregate-only site-inventory v2 execution
- **Owner/approver:** Thesis owner authorized exactly one v2 execution from revision `60740644a305182079eced47d9684a001d31b2a7`; candidate-manifest and Gate 5 approval were explicitly withheld
- **Decision:** Preserve the immutable v2 report as validated Gate 4 evidence. It corrects D-019's ownership failure: all 40 candidates follow deterministic first-present class ownership in order 1–5, their aggregate material compositions match the approved rasters, and all existing/candidate block footprints are disjoint. This record does not approve the 20 primary families, 20 reserves, validator-computed list hashes, or Gate 5.
- **Evidence:** `Thesis_RF/Code/output/phase9/phase9_set_c_site_inventory_v2.aggregate.json`; payload SHA-256 `25a0a27ac81db42a38eaf339b101fbc6c41c44cd60c1525cc5686112332158cc`; whole-file SHA-256 `26fb06f324b7197a6b6cb5242d7b09012400451e5467d3bdb6fdb30e5dc1de75`; 20 primaries and 20 reserves; four of each per material class 1–5; 90 unique reserved blocks with zero overlap; sole-output and aggregate-only checks passed; final read-only RF Validator PASS. Validator audit hashes are all candidates `7336d5f124ce8164275855f10f54eeae4f13c3af81f20ce81f2014a3c56012de`, primaries `d97e3cb7598dbc9c4b4cc53398c0d0daad0a5236f2d59e357db3cd8cf86a7d33`, and reserves `7459ba860d583c6790d033bd3eaf48c8fce7b54ad200a0d6e1c8560581fa79e0`.
- **Authorization note:** The immutable report retains the source-era `authorization_reference` text. The controlling one-run authorization is the owner's 2026-09-28 message and this durable record; no rerun is authorized to alter that immutable field.
- **Authorization boundary:** Family-manifest approval, Gate 5 implementation or execution, teacher expansion, row publication, split assignment, model operations, final-test access, `stack_ground_truth.tif`, manuscripts, `Thesis_LR/**`, and changes to `Thesis-try.qgz` remain unauthorized.
- **Consequences and follow-up:** Seek explicit human approval binding the exact 20 primaries and 20 reserves to the v2 report and hashes. Only then may Gate 5 source implementation and synthetic temporary-path tests begin under a separate approval.
- **Supersedes / superseded by:** Supplies the validated replacement evidence required by D-019 without approving the family manifest or relaxing D-015–D-018.

## D-021 — Approve Gate 5 aggregate-only adequacy source implementation

- **Date:** 2026-09-28
- **Status:** APPROVED; SOURCE-ONLY IMPLEMENTATION VALIDATED
- **Scope:** Phase 9 Gate 5 disabled aggregate-only population-adequacy harness and synthetic contract tests
- **Owner/approver:** Thesis owner, by explicit approval of the exact v2 family manifest and Gate 5 source/synthetic scope
- **Decision:** Bind the Gate 5 harness to the validated v2 inventory report (whole-file SHA-256 `26fb06f324b7197a6b6cb5242d7b09012400451e5467d3bdb6fdb30e5dc1de75`; payload SHA-256 `25a0a27ac81db42a38eaf339b101fbc6c41c44cd60c1525cc5686112332158cc`; ordered candidates `7336d5f124ce8164275855f10f54eeae4f13c3af81f20ce81f2014a3c56012de`; primaries `d97e3cb7598dbc9c4b4cc53398c0d0daad0a5236f2d59e357db3cd8cf86a7d33`; reserves `7459ba860d583c6790d033bd3eaf48c8fce7b54ad200a0d6e1c8560581fa79e0`). The implementation enumerates the deterministic 26-family × 17-wind × 2-replicate population (884 runs), executes sequentially through the model-free all-eligible teacher, drains bounded complete batches, retains only aggregate identity/counter state, constructs unexposed leakage components and D-015 adequacy feasibility, and can publish only one atomic no-overwrite aggregate JSON after a later execution authorization.
- **Validation:** Checked-in execution remains disabled. Synthetic destinations and temporary aggregate stores are confined to the system temporary directory; production config, environmental inputs, source revision/dirty state, raster hashes, mask identities, and behavior-affecting source dependencies are revalidated fail-closed. Focused Gate 5 tests passed 17/17 and the complete RF synthetic suite passed 272/272. Final read-only RF Validator result: PASS.
- **Authorization boundary:** This decision authorizes source implementation and synthetic temporary-path tests only. It does not authorize Gate 5 production expansion, teacher execution, row publication, dataset or split generation, role assignments, training, tuning, class-weight or calibration selection, threshold selection, final-test access, model/raster/checkpoint/manuscript publication, `stack_ground_truth.tif` access, `Thesis_LR/**` writes, or modification of `Thesis-try.qgz`.
- **Consequences and follow-up:** Commit the reviewed RF source, configuration, tests, and memory update as one clean revision. Gate 6 may then seek a separate one-run authorization bound to the exact revision, destination, manifest, and resource contract. No production command is authorized by D-021.
- **Supersedes / superseded by:** Supersedes D-020's pending Gate 5/source approval only; it preserves all D-015–D-018 execution, publication, split, model, and final-test gates.

## D-022 — Pin non-Astra models for the four project agents

- **Date:** 2026-09-28
- **Status:** APPROVED
- **Scope:** Project-scoped Codex agent model and reasoning defaults
- **Owner/approver:** Thesis owner, through direct edits to all four `.codex/agents/*.toml` profiles
- **Decision:** Pin `tech_lead` to `gpt-6-sol` with `model_reasoning_effort = "high"`. `rf_validator` to `gpt-5.6-sol` with `model_reasoning_effort = "high"`. Pin `thesis_researcher` and `rf_engineer` to `gpt-6-sol` with `model_reasoning_effort = "medium"`. Do not assign GPT-6 Astra to these project agents.
- **Rationale:** Architecture and skeptical validation require deeper assumption checking and edge-case review. Bounded research and implementation need balanced judgment and completeness without Astra's higher token cost. Explicit profile settings prevent accidental inheritance from a more expensive parent configuration.
- **Alternatives considered:** Let every profile inherit the parent model; use GPT-6 Astra; use GPT-6 Luna for the lighter roles; or set all four roles to high reasoning. These were rejected in favor of consistent Sol capability with role-specific effort.
- **Evidence:** `.codex/agents/tech-lead.toml`; `.codex/agents/rf-validator.toml`; `.codex/agents/thesis-researcher.toml`; `.codex/agents/rf-engineer.toml`; successful TOML parse; official Codex subagent configuration guidance reviewed on 2026-09-28.
- **Consequences and follow-up:** Recheck availability and cost behavior if model access changes. Separately repair the remaining contradiction in `tech-lead.toml`: the parent owns delegation and no human prompt copying, but the output section still requests “Prompt to Copy,” direct researcher invocation, and Markdown updating. D-022 approves model settings only and does not approve those conflicting instructions or any manuscript write.
- **Supersedes / superseded by:** None.

## D-023 — Record failed Gate 6 aggregate-adequacy execution attempt

- **Date:** 2026-09-28
- **Status:** FAILED BEFORE TEACHER EXECUTION; NO ARTIFACT
- **Scope:** One explicitly authorized Phase 9 Gate 6 aggregate-only execution attempt
- **Owner/approver:** Thesis owner, by explicit instruction to execute and validate Gate 6 from the committed Gate 5 source
- **Decision:** Treat the one-shot authorization as consumed. The exact production command was invoked once from clean revision `8b45b10de9f64cd7a9927733155e828690abeb15` after the manifest, destination, memory, and disk preflight passed. It exited with code 1 after approximately 2.78 seconds while obtaining Git provenance in `run_set_c_population_adequacy.py::_source_context`; the per-command `safe.directory` value used a Windows backslash path and Git rejected the worktree as dubiously owned. The model-free teacher, 884-run matrix, aggregate accumulator, and D-015 adequacy analysis did not start.
- **Evidence:** Launch memory `7,095,463,936` bytes; approximately 95 GB free disk; destination absent before and after; approved v2 inventory hashes matched; worktree remained clean; `output/phase9` retained only inventory v1/v2; no `phase9_set_c_population_adequacy_expansion_v1.aggregate.json` or `set-c-adequacy-*` temporary store existed. A read-only RF Validator independently reproduced failure with the Windows-style per-command override and success with `repository_root.as_posix()`.
- **Authorization boundary:** This record does not authorize a retry, source repair, production teacher execution, dataset generation/publication, split assignment, model operations, final-test access, ground-truth access, manuscripts, `Thesis_LR/**`, or QGIS modification.
- **Consequences and follow-up:** Make the minimal source repair by using `repository_root.as_posix()` for both runner Git overrides, add a focused regression test, rerun synthetic validation, commit a new clean revision, obtain a fresh read-only RF Validator PASS, and then seek a new explicit one-run Gate 6 authorization. D-015 numerical adequacy remains `NEEDS EVIDENCE`; Gate 7 and Phase 11 are blocked.
- **Supersedes / superseded by:** Adds failed execution evidence to D-021 without changing the approved Gate 5 architecture or relaxing any later gate.

## D-024 — Record user-stopped Gate 6 retry at 45 minutes

- **Date:** 2026-09-28
- **Status:** STOPPED; PARTIAL TEMPORARY EVIDENCE ONLY
- **Scope:** Freshly authorized Gate 6 aggregate-only retry from repaired revision `560ea9b70651432e20d3b8e4017da002534c06e1`
- **Owner/approver:** Thesis owner authorized the retry after the D-023 repair, then explicitly required termination if runtime exceeded 45 minutes
- **Decision:** Treat the retry authorization as consumed. Preflight passed and the command started once from the exact clean revision. At 2,711 seconds the same CPU-bound worker remained active within the approved memory ceiling, so it was terminated once under the owner's 45-minute cutoff. The command exited 1 without publishing the aggregate JSON.
- **Evidence:** Launch memory `7,548,125,184` bytes; free disk `94,198,104,064` bytes; process RSS remained below the 6-GiB ceiling. Independent read-only validation found no final report, a clean execution worktree, and one structurally valid 69,632-byte temporary SQLite store at `C:\Users\gibgib\AppData\Local\Temp\set-c-adequacy-a17r5of9.sqlite3`. The store contains 1,834 aggregate row counts for both identity kinds and family masks involving the first six ordered existing families. Sequential ordering and per-run commits imply at least 171 completed runs, but the exact count is not stored and must not be claimed.
- **Interpretation:** The run showed substantial progress and no evidence of a hang. At the observed early pace, 884 sequential runs plausibly require several hours; D-015 permits up to 21,600 seconds. Later-family runtime may differ, so linear completion estimates are diagnostic only.
- **Authorization boundary:** The partial SQLite store is diagnostic, non-authoritative, and must not be promoted. This decision does not authorize cleanup, retry, dataset generation/publication, Gate 7, split assignment, model operations, final-test access, ground-truth access, manuscripts, `Thesis_LR/**`, or QGIS modification.
- **Consequences and follow-up:** Decide whether a future Gate 6 attempt may use the full D-015 six-hour limit. Before retry, explicitly authorize removal of the residual temporary store, recheck the clean revision and resources, and grant a fresh one-run authorization. D-015 adequacy, Gate 7, and Phase 11 remain blocked.
- **Supersedes / superseded by:** Adds retry/cutoff evidence to D-023 without changing D-015 or D-021.

## D-025 — Authorize one manual full-window Gate 6 execution

- **Date:** 2026-09-28
- **Status:** AUTHORIZED; AWAITING MANUAL EXECUTION
- **Scope:** One manual Gate 6 aggregate-only execution from clean revision `560ea9b70651432e20d3b8e4017da002534c06e1`
- **Owner/approver:** Thesis owner, by explicit manual-execution authorization
- **Decision:** Permit exactly one manual invocation of the approved Gate 6 command with the complete D-015 runtime ceiling of 21,600 seconds. The agent must not launch the command. The sole permitted persistent output remains `Thesis_RF/Code/output/phase9/phase9_set_c_population_adequacy_expansion_v1.aggregate.json`; all dataset, model, raster, ground-truth, split, manuscript, `Thesis_LR/**`, and QGIS operations remain prohibited.
- **Cleanup evidence:** The owner authorized deletion of `C:\Users\gibgib\AppData\Local\Temp\set-c-adequacy-a17r5of9.sqlite3`. The exact 69,632-byte residual was verified inside the expected temp directory, the prior worker was absent, and the file was deleted. The aggregate destination remains absent and `Thesis-try.qgz` remains present.
- **Consequences and follow-up:** After the owner completes the manual command, independently validate the immutable aggregate report, hashes, sole-output condition, all 884 run records, and every D-015 adequacy gate before considering Gate 7 or Phase 11.
- **Supersedes / superseded by:** Supersedes only D-024's lack of retry/cleanup authorization; D-015, D-021, and all later publication/model gates remain unchanged.

## D-026 — Record failed Gate 6 population-adequacy result

- **Date:** 2026-09-29
- **Status:** FAIL; IMMUTABLE DIAGNOSTIC ONLY
- **Scope:** Completed manual aggregate-only Gate 6 report and independent D-015 validation
- **Decision:** Preserve `Thesis_RF/Code/output/phase9/phase9_set_c_population_adequacy_expansion_v1.aggregate.json` as failed diagnostic evidence. Do not promote it, authorize Gate 7, or begin Phase 11. The report completed all 884 ordered runs safely but found only three qualifying joint leakage components, not the required 23, and no feasible unexposed four-role allocation.
- **Evidence:** Payload SHA-256 `fe0322be2d11e7204bbb246bc82a2a744a056709f41601eb00f9d0b90ab08cdb`; whole-file SHA-256 `6caf02cdd3735e0e708a79a77dd27481f198bcbecad3bfd6fec0c566863764bb`; 10,733 aggregate rows (797 positive, 9,936 negative); components of 23, 2, and 1 families with class counts 727/8,798, 42/722, and 28/416 positive/negative; elapsed time 14,238.62 seconds. All runs were complete, inactive, authoritative, and count-matched. No row dataset, split, model, raster, protected input, or residual SQLite store was produced, and `Thesis-try.qgz` remains present.
- **Authorization discrepancy:** D-025 bound execution to revision `560ea9b70651432e20d3b8e4017da002534c06e1`, but the immutable report records clean revision `2fec62f0f37ae00972f6fa878f2c49175bdfc3ab`. The later revision is a descendant and has no `Thesis_RF/Code/**` delta from `560ea9b`, but it is not the exact authorized revision. This independently prevents treating the report as compliant Gate 6 acceptance evidence.
- **Interpretation:** The report proves the approved population does not satisfy D-015 under the current joint graph. It does not identify whether duplicate-feature edges, spatial-block edges, or both produced the 23-family component, so changing the grouping rule or population without further aggregate-only evidence would be speculative.
- **Consequences and follow-up:** First perform a source-only/read-only design review for minimal aggregate edge-attribution diagnostics and obtain an explicit human methodology decision on whether D-015 duplicate containment and/or the population design may change. Any successor diagnostic requires a versioned destination, clean exact revision, synthetic validation, independent RF Validator review, and fresh one-run authorization. All row publication, split, model, final-test, ground-truth, manuscript, and `Thesis_LR/**` gates remain closed.
- **Supersedes / superseded by:** Consumes D-025's one-run authorization and replaces its awaiting-execution status with failed diagnostic evidence; it does not supersede D-015's adequacy requirements.

## D-027 — Approve the disabled Gate 6 v2 edge-attribution diagnostic source

- **Date:** 2026-09-29
- **Status:** APPROVED; SOURCE-ONLY IMPLEMENTATION VALIDATED
- **Scope:** Minimal aggregate-only attribution of duplicate-identity and spatial-block leakage edges
- **Owner/approver:** Thesis owner, by directing the Tech Lead to coordinate RF Engineer implementation and RF Validator review of the D-026 recovery action
- **Decision:** Add a separately versioned `phase9_set_c_population_adequacy_edge_diagnostic_v2` contract that reuses the existing temporary aggregate identity store and reports only per-family-pair shared-identity counts plus duplicate-only, block-only, and joint component-size summaries. Preserve the approved D-015 joint graph and adequacy gates unchanged. Keep checked-in execution disabled, require the distinct `--execute-approved-edge-diagnostic` flag, and reserve `output/phase9/phase9_set_c_population_adequacy_edge_diagnostic_v2.aggregate.json` as the no-overwrite successor destination.
- **Validation:** RF Engineer focused tests passed 22/22 and the complete RF synthetic suite passed 277/277. The read-only RF Validator independently passed 22/22 focused tests and found no row identity disclosure, D-015 relaxation, legacy flag acceptance, or production output. Parent review passed the same 22 focused tests and `git diff --check`. The immutable v1 report remains byte-identical with whole-file SHA-256 `6caf02cdd3735e0e708a79a77dd27481f198bcbecad3bfd6fec0c566863764bb`; the v2 destination is absent and `Thesis-try.qgz` remains present.
- **Authorization boundary:** This decision authorizes only source changes and synthetic temporary-path tests. It does not authorize a v2 production execution, alteration of D-015, family replacement, row publication, split assignment, training, calibration, final-test or ground-truth access, manuscripts, `Thesis_LR/**`, or QGIS modification.
- **Consequences and follow-up:** Review and commit the accepted source and memory as one clean identifiable revision. Then seek explicit authorization for exactly one v2 aggregate diagnostic execution bound to that revision, destination, manifest, resource ceiling, and command. Gate 7 remains blocked until the production diagnostic attributes the merge and a reviewed human decision resolves the population/grouping failure.
- **Supersedes / superseded by:** Implements D-026's minimal diagnostic recommendation without superseding D-015 or authorizing execution.

## D-028 — Approve Gate 6 progress and synthetic-only resumability source

- **Date:** 2026-09-29
- **Status:** APPROVED; SOURCE AND SYNTHETIC TESTS ONLY
- **Scope:** Runtime visibility and recoverable aggregate-only test execution without weakening Gate 6 production gates
- **Owner/approver:** Thesis owner, by explicitly directing the Tech Lead to add flushed progress and checkpoints while reserving any later 884-run execution for manual authorization
- **Decision:** Emit `Gate 6 progress: N/884 runs complete` with immediate flushing only after committed runs 100, 200, …, 800, and 884. Add a synthetic temporary-path SQLite journal bound to the aggregate schema, expansion identity, configuration, source context, approved families, and complete run order. Resume only from the first uncommitted run. Commit validated run records, identity aggregates, and resource/time state together; roll back failed-run identity mutations; charge handled failure time and conservatively charge downtime after an interrupted attempt.
- **Production boundary:** Production checkpoint and resume arguments must fail closed, and the production runner exposes no resume CLI. This source-only approval does not authorize a persistent production checkpoint, another Gate 6 execution, a changed D-015 graph, row publication, role assignment, training, calibration, final-test or ground-truth access, manuscripts, `Thesis_LR/**`, or QGIS changes.
- **Validation:** RF Engineer and independent RF Validator focused suites passed 26/26. Parent verification passed the same 26 focused tests and the complete 281-test RF synthetic suite with bytecode and pytest cache disabled. No production execution or artifact occurred.
- **Known production blockers:** Before production enablement, a separately approved D-016 replacement must define the checkpoint destination, retention and cleanup rules, trusted integrity mechanism, recovery after a crash between report publication and journal cleanup, and complete crash-attempt resource telemetry. Current same-database hashes are drift/corruption checks, not adversarial authentication.
- **Adequacy consequence:** D-015 remains failed and unchanged. The v2 report is sufficient for aggregate replay only after an approved duplicate-edge policy expressible from its family-pair link counts and per-family totals. Do not repeat the unchanged 884 runs.
- **Supersedes / superseded by:** Narrows D-016 only for disabled source and synthetic temporary-path resumability. It does not authorize production checkpoint retention or supersede D-015's adequacy contract.

## D-029 — Revise Gate 6 duplicate containment and harden source-only recovery contracts

- **Date:** 2026-09-29
- **Status:** APPROVED; SOURCE AND SYNTHETIC TESTS ONLY
- **Scope:** Set C leakage identity semantics, aggregate-only policy replay, and future checkpoint validation
- **Owner/approver:** Thesis owner, by approving the narrow D-015 item 7 revision and v3 source/test work
- **Decision:** Exact equality of ordered 11-value `np.float32` predictors is an aggregate audit statistic only. Authoritative `duplicate_group_id` values must identify the sampled observation record from scenario family, scenario, run, seed, timestep pair, grid, cell coordinates, schema, and provenance identity. Complete event/scenario family, stable cell, 128×128 origin block, and actual repeated observation records remain containment edges. No deduplication, relabeling, deletion, voting, or downsampling is allowed.
- **Replay boundary:** The disabled v3 policy replay may read only the immutable Gate 6 v2 aggregate report, verify its bound hashes/provenance/order/counts, union reported spatial-block links, retain the 97 feature-collision links as audit counts, and publish one aggregate-only temporary-path report. It must not publish rows, labels, predictors, coordinates, split roles, models, rasters, or ground truth. No replay has executed.
- **Checkpoint boundary:** The source-only checkpoint contract now binds explicit package/expansion identity, recomputes aggregate/identity/resource state digests from checkpoint state, validates protected-storage evidence, and requires an injected independent final-report verifier. Production checkpoint creation/resume remains unreachable and separately gated.
- **Validation:** Focused policy/checkpoint/training tests passed 47/47; complete RF synthetic suite passed 286/286 with cache and bytecode disabled. No production replay, teacher expansion, dataset, split, model, calibration, final-test, or protected-ground-truth operation occurred.
- **Consequences:** The existing v2 adequacy failure remains immutable evidence, not a result to be overwritten. Under the approved block/provenance policy, aggregate replay is expected to yield 26 singleton components and constructive feasibility, but execution requires a separate exact one-run authorization. Gate 7 and Phase 11 remain blocked.
- **Supersedes / superseded by:** Supersedes only the duplicate-edge portion of D-015 for future source implementation; it does not claim statistical independence or real-fire validity and does not authorize replay or production checkpoint use.

## D-030 — Revise RF Engineer and RF Validator model defaults

- **Date:** 2026-09-30
- **Status:** APPROVED
- **Scope:** Project-scoped RF agent model and reasoning defaults
- **Owner/approver:** Thesis owner, by explicit instruction before Gate 7 review
- **Decision:** Pin `rf_engineer` to `gpt-6-luna` with `model_reasoning_effort = "medium"`, and `rf_validator` to `gpt-6-sol` with `model_reasoning_effort = "medium"`. Leave the `tech_lead` and `thesis_researcher` settings unchanged.
- **Evidence:** `.codex/agents/rf-engineer.toml`; `.codex/agents/rf-validator.toml`; successful TOML parse.
- **Consequences and follow-up:** Future project-scoped RF Engineer and RF Validator tasks use these defaults. This changes tooling only and authorizes no production execution, dataset publication, split, training, final-test access, or protected-path write.
- **Supersedes / superseded by:** Supersedes D-022 only for the RF Engineer and RF Validator model/effort settings; all other D-022 boundaries remain in force.

## D-031 — Authorize source-only authoritative Set C generation wiring

- **Date:** 2026-09-30
- **Status:** APPROVED; SOURCE IMPLEMENTATION INDEPENDENTLY VALIDATED
- **Scope:** Disabled authoritative Set C production-runner source and synthetic temporary-path tests
- **Owner/approver:** Thesis owner, by explicit Gate 7 prerequisite authorization
- **Decision:** Add the minimum disabled runner that reuses the D-015 26-family/884-run plan, model-free teacher, bounded collector, and D-017 publication session. Bind the Gate 4 and Gate 6 artifact hashes, ordered family and run identities, package/source/configuration/input identities, per-run configuration hashes, resource limits, and atomic no-overwrite destination. Require an explicit execution flag, exact authorization reference, and separately approved clean source revision.
- **Implementation note:** The publication identity now carries an exact per-run configuration-hash map because approved wind and seed values vary by run. It also carries and enforces the approved run sequence rather than serializing by run-identity hash. These are stricter bindings, not compatibility relaxations.
- **Validation:** Initial RF Engineer focused/full suites passed 15/15 and 295/295; parent focused rerun passed 15/15 and reproduced 26 families, 884 runs, family hash `6101a4729bfc6c4b4c49f48b6ef19b728620532014fb471adf4bb0e1cefd4046`, and run-order hash `97706c938953e44c08539b758535aa5ebd7731f10abdbfe780f248afaea4c250`. The RF Validator found that final CSV assembly was outside the disk reserve and total deadline. Revision `df63ef5affdba97be483bffcd15f77bff93f6154` repaired this by reserving remaining staged CSV bytes, exact manifest size, and 64 MiB above the 2 GiB floor while checking the shared deadline through atomic promotion. Final focused/full suites passed 18/18 and 298/298; independent RF Validator result: PASS.
- **Authorization boundary:** This decision authorizes source and synthetic temporary-path tests only. It does not authorize production simulation, row-level dataset generation or publication, checkpoint/resume, split assignment, training, calibration, final-test or ground-truth access, models, rasters, manuscripts, `Thesis_LR/**`, or `Thesis-try.qgz` modification.
- **Consequences and follow-up:** Source readiness is complete. Production generation remains prohibited until the owner separately authorizes the exact package destination, final clean revision, family/run hashes, resource envelope, and one-time command.
- **Supersedes / superseded by:** Implements the source prerequisite reserved by D-017 without authorizing its production execution.

## D-032 — Record failed Gate 7 generation attempt and batch-mask repair

- **Date:** 2026-09-30
- **Status:** FAILED INVOCATION; NO IMMUTABLE PACKAGE
- **Scope:** One owner-authorized manual authoritative Set C generation invocation from clean revision `45733b16bf9cd64b897800f8441433804f8c95a2`
- **Decision:** Treat the one-run authorization as consumed. The command failed during batch staging because the publisher required equality with only two stable domain-mask hashes, while real transition batches also carry the required timestep-specific `eligible_mask_t_sha256`. The later collector-finalization exception followed the staging failure. Preserve the fail-closed behavior and do not treat this attempt as dataset evidence.
- **Evidence:** Owner traceback through `SetCPublicationSession.stage_batch`; `transition_observer.py` emits all three hashes and `set_c_collector.py` passes them through; the authorized immutable package destination is absent. Repair revision `da25565531204b88a0095574914f1f546532ea94` validates the two stable hashes against package identity and the eligible-mask hash as a 64-character lowercase SHA-256. Focused synthetic tests passed 19/19 and the full RF suite passed 299/299; independent read-only RF Validator review returned PASS. No staging residue remains.
- **Authorization boundary:** No retry, dataset publication, split, training, calibration, final-test or ground-truth access, checkpoint/resume, manuscript, `Thesis_LR/**`, or `Thesis-try.qgz` change is authorized by this record.
- **Consequences and follow-up:** Request a fresh exact one-run authorization bound to the final clean revision before any manual retry.
- **Supersedes / superseded by:** Consumes the prior one-run Gate 7 generation approval without changing D-015, D-017, or D-029 methodology.

## D-033 — Record authoritative Set C v1 package generation and integrity review

- **Date:** 2026-10-01
- **Status:** GENERATED; INDEPENDENT PACKAGE INTEGRITY PASS
- **Scope:** One owner-authorized manual Gate 7 execution from clean revision `161b8d0ce5ee616a96cc94a7da3f994cd8ec8775`
- **Decision:** Preserve `Thesis_RF/Code/output/datasets/authoritative_set_c/v1` as the immutable authoritative Set C observation package. Treat the fresh one-run generation authorization as consumed. Package integrity and recorded completeness passed independent read-only validation; later data-quality and methodological review remains separate.
- **Evidence:** Manifest whole-file SHA-256 `b700f1b9f196893f23e7190daf1bc3c68b0f569f99a977ec9e919f0de73e9e3c`, payload SHA-256 `7e993176acbb57e46d7d229d838351d488b4be063b4f11b37009a57c863e8cbd`; 26 families, 884 ordered complete/inactive/authoritative/count-matched runs, 10,733 rows (797 positive, 9,936 negative), 1,508 batches, and exactly 3,902 expected files. CSV header, predictor values, labels, source/configuration/input and Gate 4/Gate 6 hashes, and no-overwrite/sole-output boundaries passed independent checks; `split_status` remains `unassigned`.
- **Limitation:** The package binds approved resource limits but contains no independent telemetry establishing observed peak RSS, minimum available memory/disk, or elapsed runtime. This integrity result does not establish real-fire accuracy or authorize grouped splitting, training, calibration, or final-test access.
- **Authorization boundary:** Do not rerun generation, overwrite the package, create roles or splits, train/tune/calibrate, access final test or ground truth, publish models/rasters, change manuscripts or `Thesis_LR/**`, or modify `Thesis-try.qgz` without separate approval.
- **Consequences and follow-up:** Continue Phase 11 with independent data-quality and methodology validation of this exact package before deciding whether it is suitable for later grouped allocation.
- **Supersedes / superseded by:** Records the successful fresh execution after D-032; it does not revise the approved Set C methodology.

## D-034 — Accept Phase 11 independent Set C validation

- **Date:** 2026-10-01
- **Status:** APPROVED; PHASE 11 ACCEPTED
- **Scope:** Immutable Set C v1 package acceptance for subsequent grouped-allocation policy planning
- **Owner/approver:** Thesis owner, by explicit acceptance after Tasks 11.1 and 11.2
- **Decision:** Accept `Thesis_RF/Code/output/datasets/authoritative_set_c/v1` for subsequent grouped-allocation planning. Tasks 11.1 and 11.2 passed the checked package-integrity, row-quality, provenance, run-completeness, leakage-containment, and prospective four-role feasibility gates. The accepted package contains 10,733 observations from 884 complete model-free teacher runs, and its actual D-029 metadata graph has 26 qualifying singleton-family components.
- **Evidence:** `docs/Phase11_Tasks_11_1_11_2_Report.md`; manifest whole-file SHA-256 `b700f1b9f196893f23e7190daf1bc3c68b0f569f99a977ec9e919f0de73e9e3c`; canonical CSV SHA-256 `7f0f1867445141f365687d127e449117f234348cec571990d70bc2b226ba40f6`; Gate 6 v3 payload/file SHA-256 `3891f54b041f98a0f8cdc0af32dd01485a269b54546c98e5cd5bd7b559d9ca96` / `a6cd757ff78da2174625a773ba5a5df34b5dd077da41de4bf17bfe88b7bf8bbc`; independent RF Validator Tasks 11.1 and 11.2.
- **Limitations:** This acceptance does not assert observed real-fire accuracy, independently prove every `all_eligible` capture or state-at-`t` feature without teacher replay, establish measured spatial independence, or supply missing observed peak RSS, available-memory/disk minima, or elapsed-time telemetry.
- **Authorization boundary:** This decision authorizes Phase 12 grouped-allocation and protected-final-test policy review only. It does not authorize component-to-role assignment, split execution or publication, split-seed selection, final-test membership access, RF training or tuning, class-weight selection, probability calibration, reporting-threshold selection, model/raster publication, `stack_ground_truth.tif` access or comparison, manuscript changes, `Thesis_LR/**` writes, or modification of `Thesis-try.qgz`.
- **Consequences and follow-up:** Review and obtain owner approval for exact numerical allocation constraints and a concrete protected final-test storage, permissions, and ACL mechanism. Split execution remains a later, separately authorized operation.
- **Supersedes / superseded by:** Completes the Phase 11 validation gate after D-033 without changing D-010, D-014, D-015, D-017, or D-029 methodology.

## D-035 — Approve Phase 12 numerical policy; require protected-source design gate

- **Date:** 2026-10-01
- **Status:** NUMERICAL POLICY APPROVED; PROTECTION CONTRACT PENDING
- **Scope:** Set C CSV SHA-256 `7f0f1867445141f365687d127e449117f234348cec571990d70bc2b226ba40f6`
- **Decision:** Require exactly 11 train, 5 validation, 5 calibration, and 5 untouched final-test leakage components. Retain D-015 class minima, material classes 1–5 in every role, component atomicity, and a maximum 50% contribution to any role's positives from one component. Use seedless `sha256(dataset_sha256 + ':' + leakage_component_id)` order; lexicographically minimize absolute positive-count deviations from `[338,153,153,153]`, then row-count deviations from `[4541,2064,2064,2064]`, then break ties by component-digest order and role order `train`, `validation`, `calibration`, `final_test`. These balance targets are soft; all adequacy gates remain hard.
- **Protection direction:** Before source implementation, confirm the owner-only out-of-repository NTFS vault path, custodian SID, evaluation identity, sanitized development context, exact DACL verifier, atomic/no-overwrite behavior, and protected/development artifact and attestation schemas. The complete mapping and final-test rows remain in the vault. Only an unlinkable three-role development projection may be released by a trusted exporter; development loaders must neither read the canonical package nor infer the protected complement. Final evaluation stays disabled.
- **Evidence:** Owner approval dated 2026-10-01; read-only `docs/Phase12_Task12_1_Grouped_Allocation_Policy_Review.md` reports feasible exact 11/5/5/5 cardinalities and identifies complete-package development-loader and repository-ACL failures for protected use.
- **Authorization boundary:** No allocation or split execution/publication, ACL change, final-test membership access, training, tuning, calibration, threshold selection, model evaluation, ground-truth access, manuscript change, or `Thesis_LR/**` write is authorized. Protection implementation requires the preceding exact-contract confirmation and a separately scoped source-only task; synthetic temporary-path tests may not create a real vault or role mapping.
- **Consequences and follow-up:** Complete read-only architecture and security-contract review, obtain explicit owner confirmation of the unresolved choices, then separately plan source-only implementation and independent validation. D-034 limitations remain in force.

## D-036 — Conditionally confirm Phase 12B protection design

- **Date:** 2026-10-01
- **Status:** OWNER DESIGN CONFIRMED; PRE-IMPLEMENTATION GATE BLOCKED
- **Decision:** Propose `C:\SetC_Protected\set_c_v1\` as the owner-only vault and `C:\SetC_Development\set_c_v1\` as the separate three-role release. Confirm custodian `LAPTOP-HDOU1IC7\gibgib` / SID `S-1-5-21-2674249978-1424284920-1115605237-1001`. Require a separate nonadministrator `LAPTOP-HDOU1IC7\SetCDev` identity with its actual SID verified before source implementation. Ordinary development must not run as custodian. Final evaluation remains owner-only and disabled; any distinct evaluator or partner development-data identity requires a later decision. Partner sharing may use only a separately approved access-controlled development release or non-sensitive reports, never vault access or the complete mapping.
- **Read-only prerequisite check:** C: is fixed, healthy NTFS; its root is not a reparse point. The proposed vault and release directories do not exist, so their child-path boundaries, owner and DACL cannot yet be attested. The custodian name resolves to the confirmed SID. `SetCDev` does not exist as a local user and its name does not resolve to a SID; its nonadministrator status and denied access cannot be verified.
- **Authorization boundary:** This confirmation does not authorize account or directory creation, vault/dataset ACL changes, allocation or export execution, role assignment or split publication, final-test membership access, training, tuning, calibration, threshold selection, model evaluation, ground-truth access, manuscript changes, or `Thesis_LR/**` writes. Source-only implementation approval must be requested separately after the missing identity and exact DACL, no-overwrite, artifact/attestation, and sanitized-development contracts pass review.
- **Consequences and follow-up:** Obtain the `SetCDev` account/SID through an owner-authorized setup, verify it is nonadministrator, then review the exact protection contract. Phase 12 is not complete; Phase 13 is not authorized. D-034 limitations remain in force.

## D-037 — Replace separate-account vault design with procedural two-folder holdout

- **Date:** 2026-10-01
- **Status:** OWNER-APPROVED PROTECTION REVISION; SOURCE AND EXECUTION NOT AUTHORIZED
- **Decision:** Keep D-035's exact 11/5/5/5 numerical and deterministic allocation policy unchanged. Use `C:\SetC_Protected\set_c_v1\` as an out-of-repository protected holding folder for the complete mapping and final-test rows, and `C:\SetC_Development\set_c_v1\` for only a three-role development release. These ordinary folders serve the former “vault” purpose; no special vault product or new `SetCDev` Windows account is required. The owner may remain custodian and run development under the same Windows identity. Strip canonical membership/linking identifiers from the development release, and require development loaders to consume only that release, never the full canonical dataset or complete mapping.
- **Claim boundary:** This is procedural holdout and pipeline-level isolation, not an OS-enforced blind test. The owner's Windows identity may access both folders; folder placement, pseudonymization, and loader guards cannot prove the owner never inspected final-test membership or prevent deliberate same-account access to the canonical package. Future reporting may say the final test was held out by the approved workflow only if execution records support that claim; it must not claim verified inaccessibility to the owner. Keep partner/other-account access to the protected folder disallowed; any partner development release needs separate approval and access control.
- **Remaining source contract:** Before production allocation, independently review and source-test the bounded exporter, development-only loader adaptation, versioned artifact/attestation schemas, hash/provenance binding, atomic no-overwrite publication, path and access checks, and failure cleanup using synthetic temporary paths. Actual directory creation, ACL change, role assignment, export, split publication, final-test access, model work, and ground-truth comparison remain separately gated. Final evaluation stays disabled.
- **Evidence:** Owner request dated 2026-10-01 to remove the account/vault complexity; read-only architecture review of D-017/D-029/D-034–D-036, `docs/Phase12_Task12_1_Grouped_Allocation_Policy_Review.md`, `modules/model_trainer.py`, and `train_rf_optuna.py` found current loaders still consume the full CSV and assignment.
- **Supersedes / superseded by:** Supersedes D-035's stronger protection-direction requirements for an inaccessible development principal and D-036's `SetCDev` account/SID and separate-identity prerequisite. Preserves D-035's numerical policy, D-034's claim limits, and all ungranted execution gates.

## D-038 — Place the procedural Set C holdout folders inside the repository

- **Date:** 2026-10-01
- **Status:** OWNER-APPROVED PATH REVISION; SOURCE AND EXECUTION NOT YET AUTHORIZED
- **Decision:** Use `C:\Users\gibgib\Thesis-Cellular-Automata\SetC_Protected\set_c_v1\` for the complete split mapping and final-test rows, and `C:\Users\gibgib\Thesis-Cellular-Automata\SetC_Development\set_c_v1\` for only the train/validation/calibration release. These are ordinary folders inside the repository checkout. No `SetCDev` account or specialized vault is required. Before any real artifacts are created, exclude both directory trees explicitly from Git and verify their path boundaries and the absence of tracked/committed protected artifacts. A Git ignore rule is a publication guard, not access control; source-only work must use synthetic temporary paths.
- **Claim boundary:** The same Windows identity and workspace tools may access both folders and the accepted canonical CSV. Given the approved deterministic allocation, anyone with the canonical CSV and allocator could recompute role membership. Therefore, the development release must not itself disclose the complete mapping or linkable canonical identities, but the workflow cannot claim cryptographic unlinkability, enforced blindness, or inaccessible final-test membership. Final evaluation stays disabled, and any partner release needs separate approval and access control.
- **Authorization boundary:** This approves the locations and protection design only. It does not authorize directory or ACL changes, allocation/source implementation, role assignment, split/export execution, final-test access, training, tuning, calibration, threshold selection, model evaluation, ground-truth access, manuscript changes, or `Thesis_LR/**` writes. Source-only exporter/allocator/loader work and synthetic checks require a separately scoped approval; production split execution requires a further approval after independent validation.
- **Evidence:** Explicit thesis-owner path approval dated 2026-10-01; D-035–D-037; current `.gitignore` lacks directory-level exclusions; source inspection of `modules/model_trainer.py`, `train_rf_optuna.py`, and `modules/set_c_leakage.py`.
- **Supersedes / superseded by:** Supersedes D-037's out-of-repository path choice and any stronger unlinkability implication. Preserves D-035's numerical allocation policy, D-034's limitations, D-037's procedural-holdout warning, and all ungranted execution gates.

## D-039 — Accept the Phase 12 synthetic source gate only

- **Date:** 2026-10-01
- **Status:** SOURCE-ONLY SYNTHETIC GATE ACCEPTED; PRODUCTION SPLIT NOT APPROVED
- **Decision:** Accept the combined 12B.2–12E.1 source changes for the synthetic temporary-path contract: directory-level Git exclusions, a disabled production exporter with synthetic D-035 allocation/export checks, and development loaders that consume a pinned three-role projection. This is not acceptance of a production allocator/exporter, real Set C provenance binding, two-directory crash-safe publication, or a production release. The training CLI and final evaluation remain disabled.
- **Evidence:** Owner-provided engineer and independent RF Validator reports; current `.gitignore`, `modules/set_c_split_release.py`, `modules/model_trainer.py`, `train_rf_optuna.py`, and focused tests; independent primary-agent rerun of the three focused test files from `Thesis_RF/Code` with `python -B -m pytest -q -p no:cacheprovider --basetemp <OS-temp-path>`: 38 passed, 1 skipped; Git ignore/status/diff checks; neither real folder exists and no tracked file is under either tree.
- **Limits and next gate:** The skipped Windows symlink-creation test leaves that variant unexercised; the simulated reparse test passed in the independent report. Legacy full-assignment helpers remain callable but outside active loaders. Synthetic exception rollback does not provide crash recovery between two directory publications. Production provenance and loader validation remain unimplemented/unverified. Review a separately scoped production-source completion and crash-safe protocol before requesting real split execution. D-038's procedural-only claim applies.
- **Authorization boundary:** No real folder creation, Set C role assignment/export, final-test membership access, training/tuning/calibration, model/final evaluation, ground-truth comparison, manuscript or `Thesis_LR/**` write is authorized by this source-gate acceptance.

## D-040 — Accept repaired Phase 12F/12G source-only contract

- **Date:** 2026-10-02
- **Status:** SOURCE-ONLY CONTRACT ACCEPTED; PRODUCTION SPLIT NOT AUTHORIZED
- **Decision:** Accept the repaired production-shaped allocation/export and three-role development-loading source contract after the engineer and independent RF Validator reports were reviewed together. The exporter derives grouped roles internally, protects the row-level split manifest and final-test rows, binds production-shaped attestations to the approved Set C CSV and manifest hashes and clean source revision, and treats a protected-only publication as incomplete. Public development loading and paired verification reject unpublished `.set-c-development-*` staging paths, including a stage that already contains a completion receipt; private exporter preflight retains staged validation. The real runner and production export switch remain disabled.
- **Evidence:** Owner-provided Phase 12F and 12G reports, including independent reproduction and repair of the staging-path defect; current `modules/set_c_split_release.py`, `run_set_c_split.py`, and focused tests; primary synthetic temporary-path reruns from `Thesis_RF/Code`: four focused files 48 passed/1 skipped, full RF suite 329 passed/1 skipped, with `python -B -m pytest -q -p no:cacheprovider --basetemp <OS-temp-path>`; `git diff --check` passed. Neither real D-038 folder exists or has tracked files.
- **Limitations:** The skipped Windows symlink-privilege variant remains unexercised; simulated reparse checks passed. Real package provenance, allocation runtime, and publication have not been exercised. The current checkout is not clean, and the disabled runner has no approved one-run command. D-038 remains a procedural holdout, not enforced blindness. An interrupted protected-only bundle requires explicit recovery review, not automatic retry or overwrite.
- **Authorization boundary:** This accepts source and synthetic tests only. It does not authorize source enablement for a run, real folder creation, accepted Set C CSV access, role assignment/export, final-test membership access, training/tuning/calibration, model or final evaluation, ACL changes, ground-truth comparison, manuscript changes, or `Thesis_LR/**` writes. Before a one-run execution request, review a disabled-by-default runnable command and source gate, commit an exact clean revision, and specify the approved preflight, destinations, operator, failure response, and independent protected-scope validation.

## D-041 — Approve a procedural Phase 12H.1 command design

- **Date:** 2026-10-02
- **Status:** SOURCE-ONLY DESIGN APPROVED; IMPLEMENTATION AND EXECUTION PENDING
- **Decision:** The future Set C split command shall deny by default and require an explicit operator opt-in plus a reference to a separate owner approval. The reference is audit evidence supplied by the operator, not machine-verifiable proof of owner authorization. Preserve the clean pinned-revision, accepted-package provenance, D-035 allocation, D-038 path, protected-first, no-overwrite, and incomplete-publication checks. The reviewed command should not need a further source edit for a later separately approved invocation. Source testing is limited to synthetic OS-temporary paths.
- **Evidence and limitation:** Explicit thesis-owner design approval after Phase 12H.1 stopped without a lasting change; D-038's same-account procedural holdout; the current runner still denies and production export remains disabled. A flag or free-text reference cannot independently authenticate the owner, and this decision makes no such claim.
- **Authorization boundary:** This approves source-only design and synthetic testing, not real folder creation, accepted Set C access, role assignment/export, final-test membership access, ACL changes, training, calibration, evaluation, ground-truth access, manuscript changes, or `Thesis_LR/**` writes. A later one-run execution requires independent source validation, a clean reviewed commit, an exact run card, and separate owner approval.
- **Supersedes / superseded by:** Clarifies D-040's command-design gate without changing D-035, D-038, or any ungranted execution boundary.
