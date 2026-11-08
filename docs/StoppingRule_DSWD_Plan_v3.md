# Stopping Protocol v3 — Confirmed Incident and Shared Clock Assumption

Date: 3 November 2026 (Asia/Manila). **Protocol confirmations recorded; documentation only. No implementation or experiment release.** The RF side and LR side agree on the primary clock assumption; the LR side will run the same matrix in its own simulator.

## 1. Confirmed incident identity and time sources

The DSWD reports use the same endpoint, **fire out**, and give approximate start/end times:

| Incident | Incident date | DSWD start | Fire out | Nominal interval |
|---|---|---|---|---:|
| Airbase: Sitio Sangi Interior, Brgy. Pajo | 24 January 2025 | about 13:32 | about 14:58 | **86 minutes** |
| Sitio Sta. Maria, Brgy. Pusok | 12 December 2023 | about 13:25 | about 16:57 | **212 minutes** |

Sources: [DSWD DROMIC Report #3, 3 February 2025, p.1](https://dromic.dswd.gov.ph/wp-content/uploads/2025/02/DSWD-DROMIC-Report-3-on-the-Fire-Incident-in-Brgy.-Pajo-Lapu-Lapu-City-as-of-03-February-2025-6PM.pdf) and [DSWD DROMIC Terminal Report, 21 April 2024, p.1](https://dromic.dswd.gov.ph/wp-content/uploads/2023/12/DSWD-DROMIC-Terminal-Report-on-the-Fire-Incident-in-Brgy.-Pusok-Lapu-Lapu-City-Cebu-21-April-2024-6AM.pdf). April2024 is the terminal report date, not the Santa Maria incident date.

**OWNER-CONFIRMED linkage:** The owner supplied the BFP row **“Sitio Sangi Interior, Brgy. Pajo, 01/24/2025 1328H”** and confirmed that it identifies the mapped Airbase incident and the DSWD Pajo report. The canonical incident name is **Sitio Sangi Interior, Brgy. Pajo**. The linkage is resolved for this protocol on the owner's evidence; the original BFP sheet was not independently opened.

The BFP row records13:28; DSWD records approximately13:32. Preserve both source times. The agreed duration basis remains **DSWD13:32–14:58 =86minutes**. Using BFP13:28 with the same fire-out time would produce90minutes and is not the duration basis of this matrix.

Use these DSWD intervals for the duration study. Preserve the previous90-update experiment and its actual13:28-based assumption as historical evidence. Do not relabel it an86-minute run. Similarly, do not rewrite the original Santa Maria configurations or their adopted11:31/NASA HR11 basis. Any wind-hour change must be separately declared, rather than silently bundled into a stopping-rule experiment.

## 2. Citation clarification

The requested author replacement is not supported by the original papers: these are two distinct2019 ISPRS articles.

- **Tan & Vicente**, *Fire Spread Prediction Using Probabilistic Cellular Automata: The Case of Urban Settlements in the Philippines*, pp.415–420, DOI ending **415-2019**. Section3.7 specifies one minute per iteration and a predefined end time. [Original paper](https://isprs-archives.copernicus.org/articles/XLII-4-W19/415/2019/isprs-archives-XLII-4-W19-415-2019.pdf).
- **Patac & Vicente**, *Urban Fire Spread Modelling and Simulation Using Cellular Automaton with Extreme Learning Machine*, pp.319–326, DOI ending **319-2019**. Sections5.1.6–5.2 discuss minute-valued transition times and simulations within reported incident durations. [Original paper](https://isprs-archives.copernicus.org/articles/XLII-4-W19/319/2019/isprs-archives-XLII-4-W19-319-2019.pdf).

Cite each work for the claim it supports. Keep correct Tan citations attached to the415 paper; cite Patac separately where appropriate. Neither paper calibrates the clock of our RF/LR implementations. Existing reports and their immutable evidence/builders will not be globally renamed or overwritten.

## 3. Freeze the assumed clock and sensitivity matrix before running

Neither incident currently has identified timed spread observations available to this project. Accordingly, every minutes-per-update mapping is an **ASSUMPTION**, not empirical calibration.

**Agreed primary setting:1minute/update.** The same matrix retains0.5 and2minutes/update as prespecified sensitivity settings. The LR side confirms it will run this matrix in its own simulator; the RF side will use its RF CA. These factor-of-two variants are a deliberately chosen scenario range, not confidence bounds or evidence-derived values. Apply the same set to both simulators and both incidents. Do not choose a winner from Airbase's F1 or the LR side's step20 observation.

| Assumed minutes/update | Airbase updates, nominal86min | Santa Maria updates, nominal212min | Role |
|---:|---:|---:|---|
| 0.5 | 172 | 424 | Sensitivity |
| 1 | 86 | 212 | Primary assumption |
| 2 | 43 | 106 | Sensitivity |

All proposed horizons are exact integer multiples of these clock values. The424-update scenario exceeds the existing300-update contract: a new authorized profile must accommodate it, or explicitly report an earlier cap as computational censoring. Do not truncate at300 and label it a complete212-minute observation.

The designated comparison is **RF in the RF side simulator** and **LR in the LR side simulator**. This matrix is60 planned cases in total:2incidents ×2sides ×3clock assumptions ×5seeds, or30cases per side. Recording these confirmations does not authorize execution. Extra LR runs in the RF side simulator would be a separately identified comparison, rather than silently added to that count.

## 4. Common stopping and experiment contract

For each case, stop at natural inactivity or its prespecified duration horizon, whichever occurs first. Inactivity means **both ignited and blazing counts are zero**; a short period without new ignitions is insufficient. Resource limits remain separate and their interruptions must not be reported as modeled fire-out.

Use seeds42–46 for every case. Retain **RF no cutoff** and **LR cutoff0.40**, with the LR rule p<0.40→0 and p≥0.40 unchanged before the usual random draw. Keep each simulator's model, transition rules, feature order and probability sampling fixed during the clock study. The clock assumption labels existing updates; changing a physical integration interval/probabilities or cell timers would be a different model experiment.

Within each incident, match crop, ignition, environmental inputs, duration source, assumed clock, seed list, reference and scoring domain across the intended comparison. Record any remaining CA implementation differences; shared seeds do not guarantee identical random events across different simulators. RF calibration and the different cutoff policies also remain explicit limitations of a model-family comparison.

## 5. Implementation after separate approval

1. Freeze a shared protocol/config manifest containing incident/source identity, start/end/endpoint, nominal duration, assumed clock, update horizon, seed, cutoff policy and input/model hashes. The owner-confirmed incident linkage and DSWD duration basis are recorded above. Retain the existing wind inputs for the duration-only study; any wind-hour revision must be a separately declared experiment.
2. Add a new versioned runner profile and explicit termination metadata. Preserve every historical configuration and full-run baseline, including the earlier H90 sensitivity. The RF side implementation stays under RF ownership; the LR side applies the same contract in its own simulator. This protocol does not authorize edits to LR side files.
3. Add per-update new-ignition/cumulative/active counts without additional RNG draws. Keep ground truth and scoring metrics outside the stopping logic.
4. Export native final-state and cumulative-footprint TIFFs. Count states3/4/5 as reached; report active cells. An active stop is time-limited/censored. Do not force active cells to state5.
5. Test exact horizon boundaries, inactivity, resource-cap separation, native output/scoring counts, matching protocol values and unchanged seeded trajectories at the same update count.
6. After approval, report every prespecified case, mean±sampleSD, and the sensitivity results alongside retained full runs. No best-seed/best-step selection, circle comparison, ground-truth area stop, fitting or hidden parameter search.

## Interpretation and approval point

This is retrospective footprint agreement **conditional on a reported fire-out interval**. It does not predict fire-out or simulate suppression. A time limit may still leave substantial overprediction, and some sensitivity settings may equal full-run results because inactivity occurs first. Both outcomes must be reported.

Airbase has already informed development; no held-out or spatial-independence claim is restored by prespecifying the next analysis. Reference exclusions and construction/registration/prior-use qualifications remain attached to each incident's scores. Do not pool the two different scoring domains as one accuracy estimate.

**Next action when implementation is requested:** prepare the RF side versioned runner/configuration changes and targeted checks against this shared protocol. The LR side implements the equivalent contract in its own simulator. Preserve all baselines and keep simulation execution as a separately instructed step. The agreed1minute/update remains an assumption; it is not established as physically best.
