# RF Training Across Stages A, B, C (PowerShell, Windows)

**For:** Kent
**From:** Kristian
**Branch:** `lr-information-gap-fix` (already on the shared repo)
**Goal:** Retrain RF on the wind-fixed dataset and produce three comparison points (Stage A, B, C) so the joint thesis paper can present a fair LR-vs-RF comparison.

This document is a complete walkthrough. Stage A → B → C, each with regeneration (where needed), training, simulation, and validation. **Run all commands in PowerShell.** Forward slashes work fine in PowerShell paths; you can mix `/` and `\\\\` if you prefer.

\---

```

### Activate your Python virtual environment

Whatever you normally do, e.g.:

```powershell
.\\\\.venv\\\\Scripts\\\\Activate.ps1
```

(If PowerShell blocks the activation script with a security message, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in that session.)

### Locate your RF training script

This document does NOT prescribe your training script — you have your own. We will only specify which CSV to point it at and where to save the resulting `.joblib`. Whatever script you've used before is fine, as long as you can target a specific CSV and output path.

### Locate your ground truth raster

You'll need the path to `stack\\\_ground\\\_truth.tif` for validation. The shared copy in this repo is at `response-ref\\\\processed-data-tif\\\\stack\\\_ground\\\_truth.tif` (relative to `Thesis\\\_1\\\\`). Use whichever copy is closer to you.

\---

## STAGE A — Original 10-feature schema (with `material\\\_class`, no `wind\\\_weighted\\\_score`)

### A.1 — Regenerate the Stage A training data

The Stage A dataset was overwritten when we generated Stage C, so it has to be rebuilt. The pieces are in the repo; we just swap three files temporarily, run the regen, then put them back.

#### A.1.1 — Open PowerShell at `Thesis\\\_1\\\\Code\\\\` and create a temp swap folder

```powershell
cd Thesis\\\_1\\\\Code
New-Item -ItemType Directory -Force -Path \\\_temp\\\_swap | Out-Null
```

#### A.1.2 — Back up the current Stage C files

```powershell
Copy-Item sandbox\\\_kent\\\\modules\\\\feature\\\_pipeline.py \\\_temp\\\_swap\\\\feature\\\_pipeline\\\_stage\\\_c.py
Copy-Item sandbox\\\_kent\\\\modules\\\\automata\\\_engine.py \\\_temp\\\_swap\\\\automata\\\_engine\\\_stage\\\_c.py
Copy-Item sandbox\\\_kent\\\\dataset\\\_generator.py \\\_temp\\\_swap\\\\dataset\\\_generator\\\_stage\\\_c.py
```

#### A.1.3 — Swap in the Stage A versions

```powershell
Copy-Item ..\\\\..\\\\response-ref\\\\feature\\\_pipeline.py sandbox\\\_kent\\\\modules\\\\feature\\\_pipeline.py
Copy-Item backups\\\\option2\\\_information\\\_gap\\\_fix\\\\automata\\\_engine.py.bak sandbox\\\_kent\\\\modules\\\\automata\\\_engine.py
Copy-Item backups\\\\option2\\\_information\\\_gap\\\_fix\\\\dataset\\\_generator.py.bak sandbox\\\_kent\\\\dataset\\\_generator.py
```

#### A.1.4 — One small manual fix to the dataset generator

Open `sandbox\\\_kent\\\\dataset\\\_generator.py` in any editor. Find the `\\\_crop\\\_environment` method (around line 42). The `cropped` dict has lines like `"slope\\\_risk": env\\\["slope\\\_risk"]\\\[row\\\_slice, col\\\_slice],`. Add this line **immediately after** the `material\\\_risk` line:

```python
            "material\\\_class": env\\\["material\\\_class"]\\\[row\\\_slice, col\\\_slice],
```

Save. Without this, regeneration errors out with "Missing material\_class in environment".

#### A.1.5 — Run regeneration

```powershell
cd sandbox\\\_kent
python generate\\\_multi\\\_scenario.py
```

**Wall time:** \~30 minutes. Output: `dataFiles\\\\multi\\\_scenario\\\_dataset.csv`.

#### A.1.6 — Rename so Stage C isn't overwritten later

```powershell
Move-Item dataFiles\\\\multi\\\_scenario\\\_dataset.csv dataFiles\\\\multi\\\_scenario\\\_dataset\\\_stage\\\_a.csv
cd ..
```

#### A.1.7 — Verify

```powershell
Get-Content sandbox\\\_kent\\\\dataFiles\\\\multi\\\_scenario\\\_dataset\\\_stage\\\_a.csv -TotalCount 1
(Get-Content sandbox\\\_kent\\\\dataFiles\\\\multi\\\_scenario\\\_dataset\\\_stage\\\_a.csv | Measure-Object -Line).Lines
```

The first command should print the header — 10 feature columns ending in `composite\\\_flammability,Ignited`, **no** `wind\\\_weighted\\\_score`. The second command should print roughly 24,000.

If the header includes `wind\\\_weighted\\\_score`, something went wrong with the file swap. Don't proceed — message me.

\---

### A.2 — Train RF on the Stage A dataset

Train your RF on `Thesis\\\_1\\\\Code\\\\sandbox\\\_kent\\\\dataFiles\\\\multi\\\_scenario\\\_dataset\\\_stage\\\_a.csv`.

The 10 columns the RF will be trained on, in this order:

```
slope\\\_risk, proximity\\\_risk, building\\\_presence, material\\\_risk, material\\\_class,
wind\\\_speed, wind\\\_sin, wind\\\_cos, neighbor\\\_burning\\\_count, composite\\\_flammability
```

(`Ignited` is the target column.)

Save the trained model to:

```
Thesis\\\_1\\\\Code\\\\sandbox\\\_kent\\\\models\\\\fire\\\_rf\\\_stage\\\_a.joblib
```

(or any name you prefer — adjust the YAML in A.3 accordingly.)

**Wall time:** depends on your training script. Probably under a minute for RF on \~24k rows.

\---

### A.3 — Configure the YAML to point at the Stage A model

Open `Thesis\\\_1\\\\Code\\\\sandbox\\\_kent\\\\config\\\\default\\\_experiment.yaml`. Find the `ml\\\_model` block and change `model\\\_path`:

```yaml
ml\\\_model:
  enabled: true
  model\\\_path: "models/fire\\\_rf\\\_stage\\\_a.joblib"
  proba\\\_threshold: 0.0
```

`proba\\\_threshold: 0.0` means no thresholding (default behavior). You'll do a threshold sweep later as a separate step.

The ignition point and wind config should already be set for Sitio Santa Maria. Don't change them.

\---

### A.4 — Run the simulation

```powershell
cd sandbox\\\_kent
python main.py
cd ..
```

**Wall time:** \~5–15 minutes depending on how the fire spreads. Output: `sandbox\\\_kent\\\\output\\\\final\\\_state.tif`.

\---

### A.5 — Validate against ground truth

```powershell
python spatial\\\_validation\\\\validate\\\_simulation.py `
    --final sandbox\\\_kent\\\\output\\\\final\\\_state.tif `
    --gt ..\\\\..\\\\response-ref\\\\processed-data-tif\\\\stack\\\_ground\\\_truth.tif `
    --label "RF Stage A"
```

(Replace `..\\\\..\\\\response-ref\\\\processed-data-tif\\\\stack\\\_ground\\\_truth.tif` with your local path if different. The backtick at the end of each line is PowerShell's line-continuation character.)

The script prints F1, Recall, Precision, AUC-ROC, Jaccard, and the confusion matrix.

\---

### A.6 — Save Stage A's `final\\\_state.tif` so it isn't overwritten by later stages

```powershell
Copy-Item sandbox\\\_kent\\\\output\\\\final\\\_state.tif sandbox\\\_kent\\\\output\\\\final\\\_state\\\_rf\\\_stage\\\_a.tif
```

\---

### A.7 — Send results

Send Kristian:

* The output of A.5 (the metric panel)
* Confirmation that the saved file exists: `final\\\_state\\\_rf\\\_stage\\\_a.tif`

\---

### A.8 (optional but recommended) — Threshold sweep on Stage A

Once Stage A's default-threshold result is recorded, try the threshold trick. For each value of `proba\\\_threshold` in {0.30, 0.40, 0.50}:

1. Edit `sandbox\\\_kent\\\\config\\\\default\\\_experiment.yaml`, change `proba\\\_threshold: 0.0` → `proba\\\_threshold: 0.40` (and so on for the other values)
2. Run the simulation: `cd sandbox\\\_kent \\\&\\\& python main.py \\\&\\\& cd ..`
3. Save the output with a unique name: `Copy-Item sandbox\\\_kent\\\\output\\\\final\\\_state.tif sandbox\\\_kent\\\\output\\\\final\\\_state\\\_rf\\\_stage\\\_a\\\_t040.tif`
4. Validate: re-run A.5 with the corresponding `--final` path and a label like `"RF Stage A t=0.40"`
5. Send the metrics

Pick the threshold value that gives the highest F1 *while keeping recall ≥ 0.80*. That's the operational pick for Stage A.

\---

## STAGE B — Our 10-feature schema (with `wind\\\_weighted\\\_score`, no `material\\\_class`)

Stage B uses the modules in `Thesis\\\_1\\\\Code\\\\` directly (NOT `sandbox\\\_kent\\\\`). The dataset already exists, so no regeneration needed.

**Important:** before starting Stage B, leave `sandbox\\\_kent\\\\` alone. Don't restore the Stage C files yet — we'll need to do that when we get to Stage C, and your `\\\_temp\\\_swap\\\\` folder still has the backups.

### B.1 — Verify the Stage B dataset

```powershell
cd Thesis\\\_1\\\\Code
Get-Content dataFiles\\\\multi\\\_scenario\\\_dataset.csv -TotalCount 1
(Get-Content dataFiles\\\\multi\\\_scenario\\\_dataset.csv | Measure-Object -Line).Lines
```

Header should list 10 feature columns ending in `wind\\\_weighted\\\_score,Ignited` (no `material\\\_class`). Row count \~21,000.

### B.2 — Verify the Code\\modules\\ are in Stage B configuration

```powershell
Get-Content modules\\\\feature\\\_pipeline.py -TotalCount 30 | Select-String "wind\\\_weighted\\\_score|material\\\_class"
```

You should see `"wind\\\_weighted\\\_score"` in the output and **not** see `"material\\\_class"`. That confirms the Stage B schema.

If for some reason `Code\\\\modules\\\\` got contaminated with Stage A files, restore from git:

```powershell
git checkout origin/lr-information-gap-fix -- modules/feature\\\_pipeline.py modules/automata\\\_engine.py
```

### B.3 — Train RF on the Stage B dataset

Train on `Thesis\\\_1\\\\Code\\\\dataFiles\\\\multi\\\_scenario\\\_dataset.csv`. The 10 columns, in order:

```
slope\\\_risk, proximity\\\_risk, building\\\_presence, material\\\_risk,
wind\\\_speed, wind\\\_sin, wind\\\_cos, neighbor\\\_burning\\\_count,
composite\\\_flammability, wind\\\_weighted\\\_score
```

Save the model to:

```
Thesis\\\_1\\\\Code\\\\models\\\\fire\\\_rf\\\_stage\\\_b.joblib
```

### B.4 — Configure the YAML

Edit `Thesis\\\_1\\\\Code\\\\config\\\\default\\\_experiment.yaml`:

```yaml
ml\\\_model:
  enabled: true
  model\\\_path: "models/fire\\\_rf\\\_stage\\\_b.joblib"
  proba\\\_threshold: 0.0
```

### B.5 — Run the simulation

```powershell
python main.py
```

(You're already in `Code\\\\`.) Output: `output\\\\final\\\_state.tif`.

### B.6 — Validate

```powershell
python spatial\\\_validation\\\\validate\\\_simulation.py `
    --final output\\\\final\\\_state.tif `
    --gt ..\\\\..\\\\response-ref\\\\processed-data-tif\\\\stack\\\_ground\\\_truth.tif `
    --label "RF Stage B"
```

### B.7 — Save Stage B's output

```powershell
Copy-Item output\\\\final\\\_state.tif output\\\\final\\\_state\\\_rf\\\_stage\\\_b.tif
```

### B.8 — Send results

Same as A.7 — metric panel and saved-file confirmation.

### B.9 (optional but recommended) — Threshold sweep on Stage B

Same procedure as A.8, but with `Code\\\\config\\\\default\\\_experiment.yaml`, simulation run from `Code\\\\`, and outputs saved with names like `final\\\_state\\\_rf\\\_stage\\\_b\\\_t040.tif`.

\---

## STAGE C — Unified 11-feature schema (both `material\\\_class` AND `wind\\\_weighted\\\_score`)

Stage C uses the `sandbox\\\_kent\\\\` folder again, but with the modules restored to their Stage C versions (which we backed up to `\\\_temp\\\_swap\\\\` at the start of Stage A).

### C.1 — Restore the Stage C versions of the three files

```powershell
cd Thesis\\\_1\\\\Code
Copy-Item \\\_temp\\\_swap\\\\feature\\\_pipeline\\\_stage\\\_c.py sandbox\\\_kent\\\\modules\\\\feature\\\_pipeline.py
Copy-Item \\\_temp\\\_swap\\\\automata\\\_engine\\\_stage\\\_c.py sandbox\\\_kent\\\\modules\\\\automata\\\_engine.py
Copy-Item \\\_temp\\\_swap\\\\dataset\\\_generator\\\_stage\\\_c.py sandbox\\\_kent\\\\dataset\\\_generator.py
```

### C.2 — Verify the Stage C dataset is intact

```powershell
Get-Content sandbox\\\_kent\\\\dataFiles\\\\multi\\\_scenario\\\_dataset.csv -TotalCount 1
(Get-Content sandbox\\\_kent\\\\dataFiles\\\\multi\\\_scenario\\\_dataset.csv | Measure-Object -Line).Lines
```

Header should list 11 feature columns including BOTH `material\\\_class` and `wind\\\_weighted\\\_score`. Row count \~25,000.

### C.3 — Verify modules are in Stage C configuration

```powershell
Get-Content sandbox\\\_kent\\\\modules\\\\feature\\\_pipeline.py -TotalCount 30 | Select-String "wind\\\_weighted\\\_score|material\\\_class"
```

You should see BOTH `"material\\\_class"` and `"wind\\\_weighted\\\_score"` in the output. That confirms the unified schema.

### C.4 — Train RF on the Stage C dataset

Train on `Thesis\\\_1\\\\Code\\\\sandbox\\\_kent\\\\dataFiles\\\\multi\\\_scenario\\\_dataset.csv`. The 11 columns, in order:

```
slope\\\_risk, proximity\\\_risk, building\\\_presence, material\\\_risk, material\\\_class,
wind\\\_speed, wind\\\_sin, wind\\\_cos, neighbor\\\_burning\\\_count,
composite\\\_flammability, wind\\\_weighted\\\_score
```

Save the model to:

```
Thesis\\\_1\\\\Code\\\\sandbox\\\_kent\\\\models\\\\fire\\\_rf\\\_stage\\\_c.joblib
```

### C.5 — Configure the YAML

Edit `Thesis\\\_1\\\\Code\\\\sandbox\\\_kent\\\\config\\\\default\\\_experiment.yaml`:

```yaml
ml\\\_model:
  enabled: true
  model\\\_path: "models/fire\\\_rf\\\_stage\\\_c.joblib"
  proba\\\_threshold: 0.0
```

### C.6 — Run the simulation

```powershell
cd sandbox\\\_kent
python main.py
cd ..
```

### C.7 — Validate

```powershell
python spatial\\\_validation\\\\validate\\\_simulation.py `
    --final sandbox\\\_kent\\\\output\\\\final\\\_state.tif `
    --gt ..\\\\..\\\\response-ref\\\\processed-data-tif\\\\stack\\\_ground\\\_truth.tif `
    --label "RF Stage C"
```

### C.8 — Save Stage C's output

```powershell
Copy-Item sandbox\\\_kent\\\\output\\\\final\\\_state.tif sandbox\\\_kent\\\\output\\\\final\\\_state\\\_rf\\\_stage\\\_c.tif
```

### C.9 — Send results

Metric panel + saved-file confirmation.

### C.10 (optional but recommended) — Threshold sweep on Stage C

Same procedure as A.8, but with `sandbox\\\_kent\\\\config\\\\default\\\_experiment.yaml` and outputs named like `final\\\_state\\\_rf\\\_stage\\\_c\\\_t040.tif`.

\---

## Cleanup (after all stages are done)

Once Stage C is validated, the `\\\_temp\\\_swap` folder isn't needed anymore:

```powershell
Remove-Item -Recurse -Force \\\_temp\\\_swap
```

\---

## What to send back at the end

For each stage you complete, send:

1. The full output of the validation script (F1, Recall, Precision, AUC-ROC, Jaccard, confusion matrix)
2. If you did a threshold sweep, the same metrics for the best threshold value
3. Confirmation of which `.tif` files exist in `sandbox\\\_kent\\\\output\\\\` and `Code\\\\output\\\\`

Optimal final state — three model files saved, three to six final-state TIFs (one per stage, plus optional threshold variants), and a metrics summary I can paste into the joint paper Chapter 5.

\---

## Quick reference — folder structure during each stage

|Stage|Working folder|Schema|Dataset path|
|-|-|-|-|
|A|`Code\\\\sandbox\\\_kent\\\\`|10 features with `material\\\_class`|`sandbox\\\_kent\\\\dataFiles\\\\multi\\\_scenario\\\_dataset\\\_stage\\\_a.csv`|
|B|`Code\\\\`|10 features with `wind\\\_weighted\\\_score`|`Code\\\\dataFiles\\\\multi\\\_scenario\\\_dataset.csv`|
|C|`Code\\\\sandbox\\\_kent\\\\`|11 features with both|`sandbox\\\_kent\\\\dataFiles\\\\multi\\\_scenario\\\_dataset.csv`|

If anything errors, screenshot it and message me. The most likely failure modes are: (1) feature schema mismatch between trained model and feature pipeline, (2) ground-truth raster path wrong in A.5/B.6/C.7, (3) PowerShell execution policy blocking venv activation.

