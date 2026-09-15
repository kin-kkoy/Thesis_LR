# Durable Decisions — Secondary Brain

## Record contract

- **Purpose:** Append-only record of material project decisions, alternatives, rationale, approval, and supporting evidence.
- **Authority:** This records approved decisions; it does not turn an unsupported implementation or experiment claim into evidence.
- **Scope:** Shared repository coordination and the Random Forest thesis track. Partner-owned Logistic Regression decisions require the partner's explicit participation and are not recorded here by inference.
- **Last updated:** 2026-09-15T15:11:17+08:00.
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
