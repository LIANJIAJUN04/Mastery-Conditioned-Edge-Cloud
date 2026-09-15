# Mastery-Conditioned Edge-Cloud Routing — Simulation

This project supports the core argument of `mastery-conditioned-edge-cloud-proposal.pdf`:

> Offloading decisions in wearable skill-training systems should depend on how
> much the learner has mastered, not only on how uncertain the model is.

It combines a **from-scratch policy simulation** (no dataset needed to test
the routing *logic*) with **four real, independently sourced datasets** that
each ground one specific assumption the simulation used to make synthetically:

| # | Dataset | Grounds |
|---|---|---|
| 1 | **UCI HAR** (real wearable IMU, 30 subjects) | The "sensing → tiny edge model → uncertainty" pipeline (§9.1/§9.4) + the cross-subject-split methodology warning (§13.2) |
| 2 | **PAMAP2** (real wearable IMU, 9 subjects, 3-sensor) | An independent replication of dataset #1's two findings, on a different sensor configuration |
| 3 | **ASSISTments 2009-2010** (real student learning logs) | The Bayesian Knowledge Tracing model itself (§6.4/§9.3) — real, EM-fitted p_init/p_learn/p_slip/p_guess instead of hand-picked ranges |
| 4 | **ROSMA** (real surgical-robot kinematics, 12 users, psychomotor skill) | RQ1: whether mastery is recoverable from real sensor data at all, in a real (if not identical) psychomotor-skill domain — see Section 10 |

Datasets #1-#3 each ground a specific *component* of the pipeline (real
sensing+uncertainty, real knowledge-tracing math) with data from adjacent
domains. Dataset #4 (ROSMA) is different in kind: it's the first real test of
whether mastery itself is recoverable from sensor data on an actual
psychomotor-skill task — JIGSAWS (the closest domain match) was **rejected**
on access request, ROSMA is the documented backup (see Section 10). Its
result is an honest, genuinely mixed finding — read Section 10 before citing
it, don't just take the headline number.

**All four real-data training pipelines use a proper, group-disjoint
70% train / 20% test / 10% validation split** — see Section 3.5.

---

## 0. What you need installed (one-time)

You already have these on this machine — check by opening a terminal
(PowerShell) and running:

```powershell
python --version
pip --version
```

If both print a version number, you're ready. If not, install Python from
python.org (check "Add Python to PATH" during install) and restart your
terminal.

## 1. Set up the project (one-time)

Open a terminal **in this `simulation/` folder** (in VS Code: right-click the
folder → "Open in Integrated Terminal") and run:

```powershell
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```

This creates an isolated Python environment (`venv/`) just for this project,
and installs 5 libraries: `numpy`, `pandas`, `matplotlib`, `scipy`,
`scikit-learn`. This step only needs to be done once.

## 2. Get the raw data (one-time, ~2.4 GB total)

Four datasets, all public, no login/account needed. Run these from the
`simulation/` folder:

```powershell
# Dataset #1: UCI HAR (already present in data/UCI HAR Dataset/ if you did Phase 2a before)

# Dataset #2: PAMAP2 (~688 MB zip, contains a nested zip)
mkdir data\pamap2_raw
curl.exe -o data\pamap2_raw\pamap2.zip "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
tar -xf data\pamap2_raw\pamap2.zip -C data\pamap2_raw
tar -xf data\pamap2_raw\PAMAP2_Dataset.zip PAMAP2_Dataset/Protocol -C data
# result: data\PAMAP2_Dataset\Protocol\subject101.dat ... subject109.dat

# Dataset #3: ASSISTments 2009-2010 skill-builder data (~83 MB), via the
# CAHLR pyBKT-examples mirror (a standard, citable source for this exact
# dataset formatted for BKT work)
mkdir data\assistments
curl.exe -o data\assistments\as.csv "https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/as.csv"

# Dataset #4: ROSMA (~1.6 GB, 207 kinematic CSVs, hosted on Zenodo —
# zenodo.org had a multi-day outage in Sept 2026, retry later if it 504s).
# Only the kinematics + scores.csv are needed, not the (much larger) videos.
# See simulation/data/ROSMA/ for the exact files; download by iterating the
# Zenodo API record https://zenodo.org/api/records/3932964 for every file
# whose key ends in .csv (skip "questionnaire" and keep scores.csv), saving
# kinematic trial CSVs into data\ROSMA\kinematics\ and scores.csv into
# data\ROSMA\. A single-threaded curl loop is very slow against Zenodo's
# per-connection throttling (~85 KB/s observed) — download several files
# concurrently (e.g. a small thread pool) to finish in reasonable time.
```

You only need to do this once. `run_experiment.py` (next section) will tell
you clearly if a raw file is missing.

## 3. Run the simulation

Every time you open a **new** terminal window, first `cd` into this
`simulation/` folder and activate the virtual environment — this lets you
type the short `python ...` form instead of the full `.\venv\Scripts\python.exe ...`
path for the rest of that terminal session:

```powershell
cd "c:\Users\Hp\Documents\GitHub\Internship Project\Medical 2\simulation"
.\venv\Scripts\Activate.ps1
python run_experiment.py
```

You'll know activation worked when the prompt gets a `(venv)` prefix. If
`Activate.ps1` refuses to run ("running scripts is disabled on this system"),
that's Windows' PowerShell execution policy blocking it — run this once, in
an admin PowerShell, to allow locally-created scripts to run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

If you'd rather skip activation, the long form always works too, from any
terminal, without activating anything first:

```powershell
.\venv\Scripts\python.exe run_experiment.py
```

**One command runs the whole Phase-1-3 pipeline (datasets #1-3).**
`run_experiment.py` automatically checks for each dataset's processed output
and, if missing, runs that dataset's script first:

1. `train_activity_classifier.py` (dataset #1, UCI HAR) → `results/real_uncertainty_samples.csv`
2. `train_pamap2_classifier.py` (dataset #2, PAMAP2) → `results/real_uncertainty_samples_pamap2.csv`
3. `fit_bkt_assistments.py` (dataset #3, ASSISTments) → `results/bkt_real_fit_report.json`
4. the Phase 1 policy simulation, now grounded in all three

You never need to run these three scripts separately unless you specifically
want to inspect one dataset's own output first (each prints its own numbers
and writes its own report — see Sections 7-9). Re-run any time with the same
one command — the three data-processing steps are skipped once their output
files exist, so a repeat run only re-does the simulation itself (a few
seconds); delete the relevant `results/*.json`/`*.csv` if you want to force
a dataset's script to re-run.

**Dataset #4 (ROSMA) is a separate, standalone RQ1 check — not part of
`run_experiment.py`'s chain**, since it tests a different question (is
mastery recoverable from sensor data at all) rather than feeding a parameter
into the Phase 1 routing simulation. Run it on its own:
`.\venv\Scripts\python.exe train_rosma_classifier.py` — see Section 10.

This will:
1. Train a RandomForest edge-model surrogate on real IMU data (twice — HAR
   and PAMAP2), each with a proper 70/20/10 train/validation/test split
   (Section 3.5), and measure real predictive uncertainty from each.
2. Fit real Bayesian Knowledge Tracing parameters to 25 real ASSISTments
   skills via EM (Baum-Welch), also with a 70/20/10 student-disjoint split
   per skill.
3. Simulate 80 learners each practising 3 skill steps, 60 trials each
   (~14,400 trials), sampling each trial's starting-mastery parameters from
   the real BKT fit and its uncertainty from the real pooled HAR+PAMAP2
   distribution — see `src/mastery.py`, `src/simulate.py`.
4. Replay that *exact same* sequence of trials through 4 routing policies
   (the 4 baselines the proposal requires, §9.6) — see `src/policies.py`.
5. Compute whether the results match the proposal's own falsifiable
   predictions (§12: E1, E2, E3) — see `src/metrics.py`.
6. Save everything to `results/` (raw logs, summaries, JSON reports, 8 PNG charts).

## 3.5. Train / validation / test methodology (all 3 datasets)

Every real-data training pipeline in this project uses a **70% train / 20%
test / 10% validation split**, implemented once in `src/data_splits.py::split_groups()`
and reused by all three datasets. Two rules apply everywhere:

- **The split is by GROUP, never by row.** Group = subject id for UCI
  HAR/PAMAP2, student id for ASSISTments. This is the same principle Section
  13.2's own check validates (a row-level/random split leaks information
  when several rows come from the same person) — it would be inconsistent to
  add a validation set that quietly violates the exact rule this project
  spent Phase 2a proving matters. All three splits (train/test/validation)
  are mutually group-disjoint.
- **The validation split picks hyperparameters or the stopping point; the
  test split is touched exactly once**, for the final reported number.
  Concretely:
  - **UCI HAR / PAMAP2** (`src/model_selection.py::fit_with_validation`): a
    small grid search over RandomForest `n_estimators`/`max_depth`, scored
    on the validation subjects only. The winning model is then evaluated
    once on the held-out test subjects — that test accuracy, and that
    model's uncertainty output, is what's reported and fed into the
    simulation.
  - **ASSISTments BKT** (`src/bkt_fit.py::fit_bkt_em(..., val_sequences=...)`):
    EM is run for 15 iterations on the train students only; after every
    iteration, the current parameters are scored by mean per-observation
    log-likelihood on the validation students, and the **best-scoring
    iteration's parameters are kept** (early stopping) — not necessarily the
    last one. The held-out test students are then used once, to report
    predictive accuracy and log-likelihood, and compared against a trivial
    majority-class baseline.

Group counts are small for two of the three datasets (9 PAMAP2 subjects, 30
HAR subjects) — see Sections 7-9 for the resulting split sizes and an honest
note about how much that limits the validation step for PAMAP2 specifically.

## 4. How to read the results (current run, all 3 datasets grounded)

Open `results/falsifiable_predictions_report.json` for the numeric verdicts:

| Prediction | Result | Verdict |
|---|---|---|
| **E1** offload reduction (target 35-60%, ≤2pp accuracy drop) | 76.5% reduction, **6.6pp accuracy drop** | ❌ overshoots — reduces offload too aggressively, at real accuracy cost |
| **E2** inverted-U by mastery level (mechanistic test) | B4 concave, peak at mastery≈0.42 (close to Phase-1-synthetic's 0.47); B3 not concave | ✅ confirmed |
| **E2** inverted-U by trial index ("live classroom" view) | Neither B4 nor B3 shows a fitted inverted-U | ❌ not visible this way anymore (see explanation below) |
| **E3** latency/energy reduction (target ≥30%/≥25%) | 45.0% / 47.6% | ✅ confirmed |

**Why E1 now overshoots harder, and why E2 disappears in the trial-index view
— read this before reporting these numbers**: the real ASSISTments-fitted
`p_init` (probability a learner already "knows" a step before their first
attempt) has a median around **0.58** across the 25 fitted skills, far above
the hand-picked synthetic range this simulation used before (0.08–0.18).
Real learners in a real dataset start much further along than the original
synthetic assumption. Two consequences, both mechanical, not code bugs:

- B4 (mastery-conditioned) sees "already fairly masterful" almost
  immediately for most simulated learners, so it escalates to cloud far less
  often overall → the offload-rate *reduction* vs B3 balloons past the
  target range.
- Because real `p_slip` is also higher (median ≈0.17 vs the old 0.08–0.12)
  and B4 by design ignores the real, now-pooled, uncertainty signal (see
  Section 8 below — this is the same open question flagged before, now
  compounded), the accuracy it gives up for that aggressiveness is larger too.
- With most learners starting already close to "known", the trial-index view
  has very little runway left to show a rise before the fall — the "hump"
  Section 9's rise-then-fall story predicts is still there **mechanistically**
  (the mastery-level-binned test still confirms it, peaking at almost the
  same mastery value as the original synthetic run), but it is compressed
  into the very first few trials and does not survive a quadratic fit across
  the full trial-index range.

**This is a genuinely useful, reportable finding for your supervisor**: E2's
underlying mechanism (offload rate really does peak at an intermediate
mastery level, not at the extremes) looks robust across very different
starting-mastery assumptions. But whether that mechanism is *visible live, in
calendar/trial-index terms, in a real classroom* depends heavily on how much
prior knowledge real learners already bring — which is an empirical
population-level fact the field trial in Section 13 would need to measure,
not something this simulation can assume either way.

Open the PNGs for the visual picture:
- `results/offload_rate_vs_trial.png` — B1 flat at 1.0, B2 flat at 0.0, B3
  decays smoothly, B4 now starts low and decays further (the "rise" is too
  compressed at this real starting-mastery level to see clearly by eye either).
- `results/mastery_trajectories.png` — sample learner mastery curves; you can
  see most learners are already >0.5 from trial 1, which is the real BKT fit
  showing up directly.

## 5. How the code maps to the proposal

| File | Proposal section | What it does |
|---|---|---|
| `src/mastery.py` | §6.4, §9.3 (Knowledge tracing / mastery estimator) | Simulates each learner's hidden mastery level growing over practice via BKT; loads real BKT parameter ranges from dataset #3 when available |
| `src/policies.py` | §9.1, §9.5, §9.6 (Architecture zoo, mastery-conditioned selection policy, baselines) | Implements the 4 model classes (T/S/M/L) and the 4 routing policies (B1-B4), including the gates (latency, safety) described in §9.5 |
| `src/simulate.py` | §13.1 (system verification methodology) | Runs the same trial sequence through all 4 policies; auto-detects and uses real uncertainty (datasets #1+#2) and real BKT parameters (dataset #3) when present |
| `src/metrics.py` | §12, §14 (Expected results, results analysis) | Computes the E1/E2/E3 falsifiable checks and the inverted-U curve fit |
| `src/plot_results.py` | — | Generates the Phase-1-style charts |
| `src/data_splits.py` | §13.2 (methodology) | Shared group-disjoint 70/20/10 splitter used by all 3 datasets (Section 3.5) |
| `src/model_selection.py` | §13.2 (methodology) | Shared validation-based hyperparameter search for the two RandomForest classifiers |
| `src/real_data.py`, `train_activity_classifier.py` | §9.4, §13.2 | Dataset #1 (UCI HAR): edge classifier, calibration check, split-inflation check |
| `src/pamap2_data.py`, `train_pamap2_classifier.py` | §9.4, §13.2 (replication) | Dataset #2 (PAMAP2): independent replication of the same two checks |
| `src/assistments_data.py`, `src/bkt_fit.py`, `fit_bkt_assistments.py` | §6.4, §9.3 | Dataset #3 (ASSISTments): real EM/Baum-Welch fit of the BKT model, with validation-based early stopping |
| `src/rosma_data.py`, `train_rosma_classifier.py` | RQ1 (§4, §12) | Dataset #4 (ROSMA): first real test of whether mastery is recoverable from sensor data on a psychomotor-skill task — standalone, does not feed into the Phase 1 simulation |

## 6. Honest limitations of this simulation

- **Real psychomotor-skill data (ROSMA) gives a genuinely mixed, not
  confirmatory, answer to RQ1.** Datasets #1-#3 ground a real *component* of
  the pipeline (perception, or the knowledge-tracing math); ROSMA (Section
  10) is the first attempt at the actual question — is mastery recoverable
  from real sensor data on a real psychomotor-skill task — and the honest
  result is: a naive whole-trial-kinematics classifier does **not** clearly
  beat chance on users it has never seen (47.6%, n=21 held-out trials from
  only 2 users), though a per-user-baseline-normalized version does better
  (61.9%) — suggestive that the signal exists but is confounded by
  between-person differences at this small sample size, not proof either
  way. RJ45/fibre/patch-panel skills are also a different (if related)
  motor-skill domain from robotic-surgery teleoperation. Treat RQ1 as
  **opened, not closed**, by this dataset.
- **ASSISTments is a different domain** (academic/cognitive skills — algebra,
  fractions — not psychomotor motor skills). Its fitted BKT parameters
  validate the *machinery*, not literal numbers for cable-termination tasks.
  Treat the shift in E1/E2 above as "what happens when you plug in real
  human learning-curve parameters", not as a claim about network-engineering
  learners specifically.
- **PAMAP2's validation split is very small (1 subject, 48 windows)** — with
  only 9 subjects total, 10% is a single held-out person. The hyperparameter
  grid search in Section 8 in fact came back tied across every setting
  because 48 samples isn't enough to discriminate between them; the
  particular choice picked (whichever tied option numpy's `argmax`-equivalent
  landed on first) shouldn't be over-interpreted as a real preference. This
  is an honest small-N limitation of PAMAP2 itself, not something the split
  code can fix — flagged here rather than hidden.
- **The exact percentages (E1, E3) still depend on hand-tuned cost constants**
  in `src/policies.py` (`LAMBDA_COST`, `ESCALATION_BONUS`, per-class
  latency/energy). These were chosen to be *directionally* realistic, not
  measured from real hardware.
- **E1 and E3 still trade off against each other** — and grounding more of
  the simulation in real data has, if anything, made this trade-off sharper
  rather than resolving it (E1 now overshoots by more than in the
  partially-synthetic Phase 2b run). A more realistic mastery-conditioned
  utility function (Section 8's open question) is the natural next lever.

## 7. Dataset #1 — UCI HAR (real wearable IMU)

`data/UCI HAR Dataset/` contains the **UCI Human Activity Recognition Using
Smartphones** dataset: 30 subjects, waist-worn smartphone accelerometer +
gyroscope, 6 activities. https://archive.ics.uci.edu/dataset/240

Run it standalone with `.\venv\Scripts\python.exe train_activity_classifier.py`.
It runs two separate, clearly-labelled experiments:

**(a) The §13.2 methodology check**, using the dataset's own original
21-train/9-test subject split (kept as-is, since that's the standard
published benchmark split this comparison is illustrating): proper
cross-subject split = 92.9% accuracy; improper random split (subject-leaked)
= 98.0%. **+5.12 percentage points of inflation** from the invalid split alone.

**(b) The actual 70/20/10 training pipeline** (Section 3.5) — all 30
subjects pooled then re-split into 21 train / 6 test / 3 validation
subjects: a small RandomForest grid search picked `n_estimators=200,
max_depth=None` on the 3 validation subjects (86.5% validation accuracy),
then that model was evaluated once on the 6 held-out test subjects:
**94.8% test accuracy**. This is the model whose uncertainty output is
actually used below and fed into the simulation.

**§9.4's claim that raw confidence needs calibration**, checked on this
held-out test set: mean uncertainty 0.216 on correct predictions vs 0.380 on
incorrect (clearly usable/separating) but Expected Calibration Error = 0.095,
and the model is **underconfident** (mean predicted confidence 0.853 <
actual accuracy 0.948) — the opposite direction from the neural-net
overconfidence the proposal specifically warns about, a model-family-dependent
nuance worth raising.

Outputs: `results/real_data_report.json` (both experiments, plus the full
grid-search trace), `results/real_uncertainty_samples.csv`,
`results/real_uncertainty_distribution.png`, `results/reliability_diagram.png`.

## 8. Dataset #2 — PAMAP2 (independent real wearable IMU replication)

`data/PAMAP2_Dataset/Protocol/subject101.dat` … `subject109.dat`: 9 subjects,
**3 body-worn IMUs** (hand/wrist, chest, ankle) + heart rate, 100 Hz, 12+
daily activities. https://archive.ics.uci.edu/dataset/231 — closer to the
proposal's "wearable" framing than HAR's single waist-mounted phone, precisely
because it is multi-sensor.

**Cleaning/transformation** (see `src/pamap2_data.py` docstring for full
detail): drop `activityID == 0` (transient, the dataset's own convention);
keep only acc±16g and gyro channels per IMU (drop the redundant ±6g range,
the dataset's own documented-invalid orientation columns, and the sparse
heart-rate channel) — 18 raw channels; linearly interpolate short
wireless-dropout NaN gaps per subject; slide a 256-sample (2.56s) 50%-overlap
window, dropping any window that straddles an activity change or still has a
NaN; extract mean/std/min/max per channel (72 features/window).

Run standalone with `.\venv\Scripts\python.exe train_pamap2_classifier.py`.
Like HAR, it runs two experiments:

**(a) The §13.2 replication check**, using a fixed held-out pair (subjects
108, 109) so the number is stable across re-runs: proper cross-subject
accuracy 82.1%; improper random-split accuracy 94.1%; **+12.0 percentage
points of inflation** — bigger than HAR's, same direction.

**(b) The 70/20/10 training pipeline** — 9 subjects split into 6 train
({101,103,104,105,107,108}) / 2 test ({102,106}) / 1 validation ({109}):
grid search tied across every hyperparameter combination on the single
validation subject (see Section 6's honest note about why), defaulting to
`n_estimators=100, max_depth=None`; held-out test accuracy **88.6%**.

**Both dataset #1 findings replicate, independently, on this second real
dataset** (using this dataset's own held-out test set from (b)):

| Check | UCI HAR (dataset #1) | PAMAP2 (dataset #2) |
|---|---|---|
| Held-out test accuracy | 94.8% | 88.6% |
| Split-inflation (§13.2 check, separate fixed split) | **+5.1pp** | **+12.0pp** |
| Uncertainty: correct vs incorrect | 0.216 vs 0.380 | 0.367 vs 0.605 (still separates) |
| ECE | 0.095 | 0.197 (worse, but still usable) |
| Calibration direction | underconfident | **underconfident (same direction)** |

The split-inflation warning (§13.2) and the "raw confidence separates but
isn't calibrated" finding (§9.4) both hold up on an independently-collected
dataset with a different sensor rig — this is real, if modest, evidence that
these are general properties of "real wearable IMU + off-the-shelf
classifier", not artefacts of HAR's specific setup. The *underconfident*
direction also replicating on a second dataset strengthens (rather than
merely repeats) the earlier observation that this is likely a RandomForest-family
trait, not a one-off.

Outputs: `results/pamap2_report.json` (both experiments + grid-search trace),
`results/real_uncertainty_samples_pamap2.csv` (pooled into the simulation's
empirical uncertainty distribution alongside HAR's — see
`src/real_data.py::load_empirical_uncertainty`),
`results/pamap2_uncertainty_distribution.png`, `results/pamap2_reliability_diagram.png`.

## 9. Dataset #3 — ASSISTments 2009-2010 (real BKT parameter fit)

`data/assistments/as.csv`: ~525k logged problem-solving attempts from a
real middle/high-school math tutoring system, each tagged with a skill and
correct/incorrect. Downloaded via the CAHLR `pyBKT-examples` GitHub mirror
(a standard source for this exact, well-studied dataset). This is a
different kind of dataset from #1/#2: real *learning* data, not real
*sensing* data — it validates the BKT model itself (§6.4/§9.3), which is
what `src/mastery.py::BKTLearner` simulates for every trial.

**Cleaning** (see `src/assistments_data.py` docstring): drop rows with no
tagged skill (~15% of rows); keep only `original == 1` (non-scaffolding)
problem presentations; drop exact-duplicate logging rows (a known issue in
this dataset); restrict to the 25 best-populated skills (≥100 distinct
students each, capped at 500 students per skill for runtime). 525,534 raw
rows → 251,405 after cleaning.

**70/20/10 split + fitting**: for each of the 25 skills, its students are
split 70/20/10 into train/test/validation (`src/data_splits.py::split_groups`,
same helper as HAR/PAMAP2 — a student's attempts never cross a split).
Standard Baum-Welch EM (`src/bkt_fit.py`) fits the classic 4-parameter BKT
model on the TRAIN students only, 15 iterations; after every iteration the
current parameters are scored on the VALIDATION students by mean
per-observation log-likelihood, and the best-scoring iteration is kept
(early stopping). The TEST students are then used once, to report
generalisation. Typical per-skill split size: ~300 train / ~45 validation /
~85 test students. Full per-skill results:
`results/bkt_fitted_params_per_skill.csv`.

Run standalone with `.\venv\Scripts\python.exe fit_bkt_assistments.py`.
**How the real fitted parameters compare to Phase 1's hand-picked synthetic ranges:**

| Parameter | Synthetic range (Phase 1) | Real fit — median (IQR across skills) |
|---|---|---|
| p_init (starts already "known") | 0.08 – 0.18 | **0.58** (0.46 – 0.69) — much higher |
| p_learn (learns per attempt) | 0.10 – 0.14 | 0.086 (0.065 – 0.139) — close match |
| p_slip (knows it, gets it wrong) | 0.08 – 0.12 | **0.17** (0.13 – 0.21) — notably higher |
| p_guess (doesn't know, gets it right) | 0.18 – 0.22 | 0.34 (0.24 – 0.40) — higher and much more variable across skills |

**Held-out TEST set generalisation** (the number that actually matters for
"did the fit generalise, or did EM just memorise the training students"):
mean test predictive accuracy across the 25 skills = **72.6%**, vs a trivial
majority-class baseline of **70.9%** — only a modest edge, and the fitted
model actually beat the majority baseline on just **13 of the 25 skills**.
**This is a genuinely honest, worth-reporting finding, not a hidden failure**:
it's a well-documented property of the classic 4-parameter BKT model in the
knowledge-tracing literature (part of the motivation for later models like
DKT/DKVMN) — its next-attempt predictive accuracy is often barely better
than base-rate for many skills, even when its fitted parameters (the table
above) are individually sensible. Treat the *parameter values* as the
useful output of this dataset, not the predictive accuracy.

The interquartile range (p25–p75, across the 25 fitted skills) for each
parameter is what `src/mastery.py::real_bkt_param_ranges()` feeds into
`simulate.py` in place of the old hand-picked ranges. `p_learn` turned out to
be a reasonable guess; `p_init` and `p_slip` were both meaningfully
underestimated in the original synthetic design, which is exactly the kind
of thing this dataset was fetched to check, and exactly why Section 4's
results shifted the way they did.

## 10. Dataset #4 — ROSMA (real psychomotor-skill kinematics, RQ1 test)

**Background**: JIGSAWS (surgical robot kinematics with actual
novice/intermediate/expert labels, the closest public domain match to the
proposal) was **rejected** on access request (submitted 2026-09-09, rejected
by 2026-09-15). ROSMA — da Vinci Research Kit (dVRK) kinematic data, 12
users, 3 training maneuvers (Post-and-Sleeve, Pea-on-a-Peg, Wire-Chaser-I), 6
repetitions each — was the documented backup. Its host, Zenodo, had a
multi-day outage through 2026-09-09; it was back up by 2026-09-15 and the
dataset was downloaded then. https://zenodo.org/records/3932964

**Why ROSMA is the right test for RQ1 even though it isn't network-cabling
skills**: it is the first dataset in this project with a genuine, repeated,
real psychomotor-skill *practice trajectory* — each user attempts the same
maneuver 6 times, and completion time/error count visibly improve across
repetitions (a real learning curve, not a simulated one). No other dataset
here has that structure.

**Mastery proxy and features** (full detail in `src/rosma_data.py`
docstring): no per-trial novice/expert label exists (unlike JIGSAWS), so
repetitions 1-2 of each (user, task) sequence are labelled "early"
(lower-mastery) and repetitions 5-6 "late" (higher-mastery); the middle two
repetitions are held out of binary-classifier training and used only for a
monotonic-trend check. Features are whole-trial mean/std/min/max of all 154
numeric kinematic channels (position/orientation/velocity/wrench/joint data
for the two master consoles and two patient-side robot arms) — 616
features/trial. One corrupted row (`X11_Pea_on_a_Peg_01`, a ~1000x
decimal-point logging error in the source `scores.csv`) is dropped.

Run standalone with `.\venv\Scripts\python.exe train_rosma_classifier.py`.
**Four checks, on a 12-user 70/20/10 split (8 train / 2 test / 2
validation users — `X05`,`X11` held out as test, seed 42)**:

| Check | Result | Verdict |
|---|---|---|
| 1. Cross-user early-vs-late accuracy (raw features) | **47.6%** (n=21 test trials, chance=50%) | ❌ does not beat chance |
| — calibration on check 1 | mean confidence 82.0% vs 47.6% accuracy | overconfident, ECE=0.343 (poor) |
| 2. Predicted P(late) vs repetition number, Spearman | rho=0.16, p=0.37 | ❌ not significant |
| 3. §13.2 replicate: proper vs improper split | 47.6% proper vs 58.5% improper | ✅ inflation direction replicates (+10.9pp) a 4th time, on a very different dataset — even though the "inflated" number itself is barely above chance |
| 4. (exploratory) per-(user,task)-normalized features, same split | **61.9%** | directionally supports the "person-identity confound" hypothesis below, but n=21 — not conclusive |

**Honest interpretation — read this before citing any of the above**: Check 1
is the primary, pre-registered-in-spirit test of RQ1, and it came back **at
or below chance**. This is a genuine negative result, not hidden. Check 4
was added *after* seeing Check 1 fail, specifically to test one concrete
hypothesis for *why*: raw whole-trial kinematics mix between-person baseline
differences (body size, motion style, overall skill level) with the
within-person practice-related change that should actually transfer to a new
person. Normalizing each trial against its own (user, task) group's mean/std
removes the between-person component; accuracy rising from 47.6% to 61.9%
under that normalization is *consistent* with the confound hypothesis, but
with only 2 held-out users / 21 trials, a single number swinging 14pp is well
within what small-sample noise alone could produce — this is a lead worth
following with more users, not a demonstrated fix.

**Bottom line for RQ1**: this project's first real test of "can mastery be
estimated from real sensor data, in a real psychomotor-skill domain" did
**not** produce a clean confirmation. It surfaced a plausible, specific
reason (person-identity confound) and one piece of suggestive evidence for
it, on a sample too small to settle the question. RQ1 stays genuinely open —
report it that way to your supervisor, not as either "confirmed" or "ROSMA
doesn't work."

Outputs: `results/rosma_report.json` (all four checks), `results/rosma_mastery_trend.png`
(predicted P(late) by repetition, held-out users), `results/rosma_reliability_diagram.png`.

## 11. Remaining next steps

- **RQ1 is still open** (Section 10) — the natural next step is more ROSMA
  users' worth of signal, which doesn't exist in this dataset (only 12
  users total), so this really means: (a) discuss with supervisor whether
  the per-user-normalization lead (Check 4) is worth a small dedicated
  write-up as a documented open question, or (b) fall back to
  self-collected pilot data (below) for a domain-matched, if n=1, check.
- **Self-collected pilot data** (own RJ45/cable-termination attempts via a
  phone IMU logging app) is now the most direct remaining path to a
  domain-matched motor-skill check, since both JIGSAWS (rejected) and ROSMA
  (tested, genuinely mixed result) are exhausted as *public* options. Zero
  external dependency (no approval, no outage risk).
- **Discuss with supervisor**: Section 4's finding sharpens the open question
  already on record — B4 ignoring the real, pooled uncertainty signal now
  costs even more accuracy once real BKT parameters are also plugged in.
  Should the mastery-conditioned utility function (§9.5) incorporate
  calibrated uncertainty as an input rather than a gate/ignored signal? The
  reverted "hard override" experiment (destroys E2's shape) still stands as
  a documented dead end — a *soft*, weighted incorporation (rather than a
  hard override) is the natural thing to try next, but is a design decision
  worth making with your supervisor rather than silently re-attempting.
- **Riiid! Answer Correctness** (Kaggle) remains an option for a second,
  much larger, real learning-log dataset to cross-check the ASSISTments BKT
  fit against, if you want a robustness check on Section 9's numbers.
