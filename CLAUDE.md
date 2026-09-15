# Project: Mastery-Conditioned Edge–Cloud Model Selection (Internship Support Simulation)

## What this project is

`mastery-conditioned-edge-cloud-proposal.pdf` (root folder) is the supervisor
(Ajay)'s research proposal: an edge–cloud AI system for wearable skill
training in network-engineering education (RJ45 termination, fibre prep,
patch-panel punch-down). Its core argument: routing decisions (which model
size to run — tiny/small/medium/cloud) should depend on the learner's
**mastery level** (estimated via Bayesian Knowledge Tracing), not just on
model uncertainty and network conditions like current state-of-the-art
systems (EdgeFM/DeepEdgeFM) do.

**My task** (assigned by the supervisor): read the proposal, then independently
build a code simulation/training pipeline that tests/supports its core
argument — without the real hardware, ethics approval, or student cohort the
full 12-month field study requires.

Full proposal structure worth knowing: RQ1–RQ4 (research questions), G1–G3
(knowledge gaps), and — most important — **E1–E6 in Section 12**, its
falsifiable predictions. **E2 (an inverted-U shaped offload-rate curve across
the acquisition curve) is explicitly called "the paper"** — the single most
important thing to reproduce/test.

## Current status: Phase 1, 2a/2b, 2c (PAMAP2), Phase 3 (ASSISTments BKT fit), and Phase 4 (ROSMA, RQ1 test) DONE — 4 real datasets now used

All work lives in `simulation/`. Full details, how to reproduce every number,
and file-to-proposal-section mapping are in **`simulation/README.md`** — read
that file for anything beyond this summary.

### Phase 1 — synthetic policy simulation (no dataset needed)
`simulation/run_experiment.py` (+ `src/mastery.py`, `src/policies.py`,
`src/simulate.py`, `src/metrics.py`, `src/plot_results.py`)

Simulates learner mastery via BKT, implements all 4 routing baselines the
proposal requires (B1 cloud-only, B2 edge-only, B3 uncertainty+network, B4
mastery-conditioned = the proposal's contribution), and checks E1/E2/E3.

**Result: E1 met (56.8% offload reduction, +0.5pp accuracy), E2 confirmed
(clean inverted-U by mastery level, peak ≈0.47), E3 below target in this
parameterisation (17.4%/14.4% vs 30%/25% target) — documented as a real
trade-off (E1 and E3 pull against each other in this simplified cost model),
not silently hidden.**

### Data cleanup (already resolved, don't reopen)
A file `medisync_collected_readings.csv` (vital-signs data from an unrelated
prior FYP) was found in `simulation/data/`. Confirmed with the user it did
NOT belong to this project. Moved to
`Medical 2/medisync_collected_readings_FYP_backup.csv` (root folder, kept as
backup, not deleted). **Do not re-use this file for this project.**

### Phase 2a — real wearable IMU data (UCI HAR)
`simulation/train_activity_classifier.py` (+ `src/real_data.py`)

Downloaded UCI "Human Activity Recognition Using Smartphones" dataset
directly from UCI ML Repository (public, no login) into
`simulation/data/UCI HAR Dataset/`. Trained a RandomForest classifier
(stand-in for a T-class tiny edge model) and empirically validated two of the
proposal's own methodology claims using REAL data:

1. **Section 13.2 (random splits inflate accuracy):** confirmed —
   cross-subject split = 92.9% accuracy vs improper random split = 98.0%
   (a measured 5.12pp inflation).
2. **Section 9.4 (raw confidence needs calibration):** confirmed usable but
   imperfect — uncertainty separates correct (mean 0.227) from incorrect
   (mean 0.456) predictions, but ECE = 0.098. Notable nuance: this
   RandomForest came out *underconfident*, whereas the proposal specifically
   warns neural nets are *overconfident* — direction of miscalibration is
   model-family dependent (worth mentioning to supervisor).

### Phase 2b — real uncertainty fed back into the Phase 1 simulator
`src/simulate.py` auto-detects `results/real_uncertainty_samples.csv`
(written by Phase 2a) and uses the REAL empirical uncertainty distribution
instead of the synthetic Gaussian assumption, if present (falls back to
synthetic automatically otherwise — always run Phase 2a's script first if
you want the real-grounded version).

**Result: E3 now clearly met (32.7%/36.8%), E1 overshoots (75.5%, accuracy
now −5pp WORSE for B4 vs B3).** Key honest finding: once uncertainty is real
and turns out to be a strong direct signal of misclassification, B4 (which by
design ignores that signal, deciding only from mastery/engagement/network)
gives up accuracy it could have had for free. **Tried fix:** added a hard
"escalate if real uncertainty is very high" safety rule to B4 — **reverted**,
because it destroyed the E2 inverted-U shape (a learner can "slip" at any
mastery level in BKT, not just near the boundary, so the rule fired
uniformly and swamped the mastery signal). This is written up as a genuine
open question in the README, not a bug — do not silently "fix" this again
without discussing the trade-off explicitly.

### Phase 2c — PAMAP2 (dataset #2: independent real-wearable replication)
`simulation/train_pamap2_classifier.py` (+ `src/pamap2_data.py`)

Supervisor asked for at least 3 datasets. Added PAMAP2 (UCI ML Repository,
public, no login; 9 subjects, 3 body-worn IMUs — hand/chest/ankle — + heart
rate, 100Hz) as a SECOND, independently collected real wearable-IMU dataset,
to check whether Phase 2a's two findings are general or an artefact of HAR's
one-phone setup. Full raw data lives in
`simulation/data/PAMAP2_Dataset/Protocol/subject101.dat`...`subject109.dat`
(~1.3GB extracted; the downloaded zips were deleted after extraction to save
space). Cleaning: drop activityID==0 (transient), drop the dataset's own
documented-invalid orientation columns + redundant ±6g accel range + sparse
heart rate, interpolate short wireless-dropout NaN gaps, window
(256 samples/2.56s, 50% overlap, same convention as HAR), extract
mean/std/min/max per channel (72 features/window). Subjects 108-109 held out
for the proper cross-subject split.

**Result: BOTH Phase 2a findings replicated independently** — split
inflation +12.0pp (vs HAR's +5.1pp, same direction, bigger effect); raw
uncertainty still separates correct (0.452) from incorrect (0.639) but ECE
worse (0.209 vs 0.098); model still **underconfident** (same direction as
HAR) — this last point is now a replicated finding, not a one-off. PAMAP2's
own uncertainty samples get pooled with HAR's into one combined empirical
distribution that `simulate.py` draws from (see `src/real_data.py`).

### Phase 3 — ASSISTments 2009-2010 (dataset #3: real BKT parameter fit)
`simulation/fit_bkt_assistments.py` (+ `src/assistments_data.py`, `src/bkt_fit.py`)

Real student learning-interaction data (~525k rows, downloaded via the CAHLR
`pyBKT-examples` GitHub mirror — a standard source for this well-studied
dataset — into `simulation/data/assistments/as.csv`), used to fit the actual
4-parameter BKT model (p_init/p_learn/p_slip/p_guess) via a from-scratch
Baum-Welch EM implementation (`src/bkt_fit.py`), independently per skill, for
the 25 best-populated skills. Cleaning: drop untagged rows, keep
`original==1` only (non-scaffolding), drop exact-duplicate logging rows (a
known issue in this dataset), sort per (student, skill) by `order_id` to
rebuild each learner's attempt sequence. 525,534 raw rows → 251,405 cleaned.

**Result — real fitted parameters vs the original hand-picked synthetic
ranges** (using the 70/20/10 student-disjoint split, train-only EM fit):
p_learn matched closely (median 0.086 vs synthetic 0.10-0.14), but
**p_init was far higher than assumed** (median ≈0.58 vs synthetic 0.08-0.18 —
real students start much more "already knowing" the skill than the original
synthetic design guessed) and **p_slip was also notably higher** (median
≈0.17 vs synthetic 0.08-0.12). The fitted per-skill interquartile ranges now
feed `simulate.py::build_trial_contexts` directly (`src/mastery.py::real_bkt_param_ranges()`),
replacing the old hand-picked ranges. Held-out TEST-set predictive accuracy
(72.6% vs a 70.9% majority-class baseline, beating it on only 13/25 skills)
is a known, documented weakness of classic BKT, not a bug — see the
train/validation/test note above.

**Downstream effect on Phase 1's E1/E2/E3 (all 3 datasets combined)**: E1
overshoots even harder now (76.5% offload reduction vs 35-60% target, 6.6pp
accuracy drop — bigger than Phase 2b's 5pp) because B4 now sees most
learners as already fairly masterful almost immediately (high real p_init)
AND still ignores the real, now-pooled uncertainty signal. E3 is comfortably
met (45.0%/47.6% vs 30%/25% target). **E2 by mastery-level still holds**
(B4 concave, peak≈0.42 — close to Phase 1's original 0.47, a robust
mechanistic result), **but E2 by trial-index no longer fits an inverted-U for
either B3 or B4** — because most simulated learners now start so close to
already-mastered that there's very little runway left to show a rise before
the fall in calendar/trial terms. This is a genuinely useful nuance, not a
regression: it suggests E2's mechanism is robust, but its real-time
visibility in an actual classroom depends heavily on real learners' prior
knowledge, which only the field trial (Section 13) can actually measure. Do
not "fix" this by re-tuning p_init back down — that would defeat the point
of grounding it in real data.

### Phase 4 — ROSMA (dataset #4: first real test of RQ1, psychomotor-skill mastery)
`simulation/train_rosma_classifier.py` (+ `src/rosma_data.py`)

JIGSAWS (surgical robot kinematics with real novice/intermediate/expert
labels — the closest public domain match to the proposal) was submitted for
access 2026-09-09 and **rejected** by 2026-09-15. ROSMA (da Vinci Research
Kit kinematic data, Zenodo, no access request needed) was the documented
backup (see the prior "Next steps / Phase 4" candidates list this section
replaces). Zenodo had a multi-day outage through
2026-09-09; retried 2026-09-15 and was back up, so the dataset (207 trial
CSVs + scores.csv, ~1.6GB, videos skipped as unneeded) was downloaded into
`simulation/data/ROSMA/`.

ROSMA: 12 users, 3 maneuvers (Post-and-Sleeve, Pea-on-a-Peg, Wire-Chaser-I),
6 repetitions each — a real, repeated psychomotor practice trajectory (time
and error count visibly improve across repetitions), unlike any other
dataset in this project. No per-trial novice/expert label exists, so
repetitions 1-2 of each (user,task) sequence are labelled "early"
(lower-mastery) and 5-6 "late" (higher-mastery); middle reps 3-4 are held
out of binary training and used only for a monotonic-trend check. Features:
whole-trial mean/std/min/max of all 154 numeric kinematic channels (4
manipulators' position/orientation/velocity/wrench/joint data) = 616
features/trial. One corrupted scores.csv row (`X11_Pea_on_a_Peg_01`, a
~1000x decimal-point logging error) dropped.

**Result — a genuine, documented negative/mixed finding, not a clean
confirmation of RQ1**: on a 12-user 70/20/10 split (8 train/2 test/2
validation users, `X05`/`X11` held out as test), the primary cross-user
early-vs-late classifier scored **47.6% accuracy (n=21 test trials, chance =
50%)** — does not beat chance — and was notably overconfident (82.0% mean
confidence vs 47.6% accuracy, ECE=0.343). The predicted-mastery-probability
vs repetition-number trend was not significant (Spearman rho=0.16, p=0.37).
The §13.2 split-inflation check DID replicate a 4th time on this very
different dataset (proper 47.6% vs improper random-split 58.5%, +10.9pp
inflation) — though even the "inflated" number is barely above chance. An
exploratory follow-up — normalizing each trial's features against its own
(user, task) group's mean/std, to isolate within-person practice change from
between-person baseline differences (body size, motion style, overall skill)
— raised accuracy to 61.9%, consistent with (but not proof of, n=21) a
person-identity-confound explanation for Check 1's failure. **Bottom line:
RQ1 stays genuinely open** — this is the first real attempt at it, and it
surfaced a specific, plausible reason for the null result plus one piece of
suggestive counter-evidence, on a sample too small to settle the question
either way. Do not report this as either "RQ1 confirmed" or "ROSMA doesn't
work" — report it as open, with the person-identity-confound lead flagged
for a supervisor discussion or a larger-N follow-up.

Full writeup, results table, and honest interpretation:
`simulation/README.md` Section 10. Outputs: `simulation/results/rosma_report.json`,
`rosma_mastery_trend.png`, `rosma_reliability_diagram.png`. This track is
standalone — it does NOT feed into `simulate.py`'s Phase 1 E1/E2/E3
simulation (unlike datasets #1-3), since it answers a different question
(is mastery recoverable from sensor data at all) rather than grounding a
simulation parameter.

### Train/validation/test split (added after supervisor asked for ≥3 datasets)
All 3 real-data pipelines now use a proper **70% train / 20% test / 10%
validation** split, GROUP-disjoint (subject id for HAR/PAMAP2, student id for
ASSISTments) — shared code in `src/data_splits.py::split_groups()`. Model
selection (RandomForest hyperparams for HAR/PAMAP2 via
`src/model_selection.py`; EM early-stopping iteration for the BKT fit) uses
ONLY the validation group; the test group is touched once for the reported
number. HAR: 21/6/3 subjects, 94.8% held-out test accuracy. PAMAP2: 6/2/1
subjects (**honest caveat: only 1 validation subject/48 windows — grid
search came back tied across all hyperparams, don't over-read the "winning"
choice**), 88.6% test accuracy. ASSISTments BKT: ~300/85/45 students per
skill, early-stopped by validation log-likelihood; held-out test predictive
accuracy 72.6% vs 70.9% majority-class baseline (only beats baseline on
13/25 skills — a known, documented weakness of classic 4-param BKT, not a
bug; report the fitted parameters as the useful output, not this accuracy
number). This changed the downstream E1/E2/E3 numbers slightly from the
first pass (see updated numbers above) but not the qualitative story.

### Reproducing everything
```powershell
cd simulation
.\venv\Scripts\pip install -r requirements.txt        # if venv not already set up
.\venv\Scripts\python.exe train_activity_classifier.py # dataset #1 (HAR) -> real uncertainty CSV
.\venv\Scripts\python.exe train_pamap2_classifier.py    # dataset #2 (PAMAP2) -> replication + pooled uncertainty CSV
.\venv\Scripts\python.exe fit_bkt_assistments.py        # dataset #3 (ASSISTments) -> real BKT param ranges
.\venv\Scripts\python.exe run_experiment.py             # Phase 1 simulation, grounded in all 3 datasets
.\venv\Scripts\python.exe train_rosma_classifier.py     # dataset #4 (ROSMA) -> standalone RQ1 test, not part of run_experiment.py
```
`run_experiment.py` alone actually runs the whole datasets #1-3 chain
automatically if any of those three outputs are missing — the first three
lines above are only needed if you want to inspect one dataset's own
printed numbers first. `train_rosma_classifier.py` (dataset #4) is separate
and standalone (see Phase 4 above) — it is never auto-run by
`run_experiment.py`, since it tests RQ1 directly rather than feeding a
parameter into the Phase 1 simulation.
Outputs go to `simulation/results/` (CSVs, JSON prediction reports, PNG charts).
Raw dataset #2/#3/#4 downloads are NOT small (~688MB PAMAP2 zip, ~83MB
ASSISTments CSV, ~1.6GB ROSMA kinematic CSVs) — see `simulation/README.md`
Section 2 for the exact download commands if `simulation/data/` needs to be
repopulated from scratch.

### Also done
- `Weekly Report/Week6_Report.md`, `Weekly Report/Week7_Report.md`,
  `Weekly Report/Week8_Report.md` — weekly reports for the supervisor,
  written in English, covering the work above in report form.
- `Weekly Report/Supervisor_Discussion_Notes_Phase4.md` — NOT a status
  report; talking points for a live discussion with Ajay on the two open
  design questions Phase 4 raised (ROSMA's person-identity-confound lead,
  and whether the mastery-conditioned utility function should incorporate
  calibrated uncertainty as a weighted input rather than only a gate). Read
  this before making a unilateral call on either question.

## Next steps / Phase 5 (not started yet)

Phase 4 (see above) ran the first real test of RQ1 on ROSMA and got a
genuinely mixed, open result — not a confirmation, not a refutation. JIGSAWS
was rejected. Both of the two top-priority real-data candidates for RQ1 are
now exhausted; what's left, in priority order:

1. **Self-collected pilot data** — record own RJ45/cable-termination
   practice attempts using a phone IMU logging app (e.g. Sensor Logger),
   label each attempt with completion time / error count as a mastery proxy
   (same style as ROSMA's Time/Errors/Score). Zero external dependency (no
   approval, no outage risk), and literally the proposal's own domain — now
   the most direct remaining path to a domain-matched check, since it's an
   n=1 pilot but at least in the *right* domain (unlike ROSMA's surgical
   teleoperation). Still need to (a) recommend a specific free phone
   IMU-logging app, (b) explain how to export the recorded data, and (c)
   decide where it lands under `simulation/data/` and what
   filename/label convention to use before writing a loader for it — this
   was paused mid-decision before Phase 4 and is still open.
2. **Follow up on ROSMA's person-identity-confound lead** (Phase 4's Check
   4: per-user-normalized features raised held-out accuracy from 47.6% to
   61.9%, but on only 21 test trials from 2 users) — worth a supervisor
   discussion on whether this is worth writing up as a documented open
   question rather than further chasing on a dataset this small (only 12
   users total, no more signal to extract by re-tuning).
3. **EPIC-Skills2018** / **BEST** (`hazeld/EPIC-Skills2018`,
   `hazeld/rank-aware-attention-network` on GitHub) — public, no request
   form, pairwise novice/expert video skill judgments across everyday manual
   tasks (dough rolling, chopsticks, tie-tying, egg-scrambling, etc.).
   Video-based, not kinematic/IMU — would need pose/motion feature
   extraction first, so more engineering effort than ROSMA was.
4. **Assembly101** — industrial toy-vehicle assembly/disassembly, closer
   task flavour to patch-panel/RJ45 wiring than surgery, has mistake
   annotations usable as a skill proxy. Now on Hugging Face
   (`cvml-nus/assembly101`) as well as Google Drive; Google Drive access is
   time-limited (14 days per request). Larger and heavier than the other
   options — treat as last resort.
5. **Riiid! Answer Correctness** (Kaggle) — real student interaction logs,
   much larger than ASSISTments; optional robustness cross-check on Phase 3's
   BKT fit (ASSISTments 2009-2010 is already done — see Phase 3 above) rather
   than a required next step now.
6. Discuss with supervisor: how should calibrated uncertainty factor into the
   mastery-conditioned utility function (see Phase 2b/Phase 3's open question
   above)? The proposal's own formula (Section 9.5) treats it as a gate input
   (`r_k`), not a utility term — worth resolving before going further. Now
   more pressing than before: Phase 3 showed real BKT parameters make B4's
   accuracy cost for ignoring uncertainty even larger.

## Working conventions established this session

- Keep results **honest, including negative/partial findings** — the user
  and this project explicitly value documenting what *didn't* work and why,
  not just successes (mirrors the proposal's own "Limitations" section style).
- This IS a git repository (`origin` = `github.com/LIANJIAJUN04/Mastery-Conditioned-Edge-Cloud`).
  `.gitignore` deliberately excludes `Weekly Report/` (kept local only —
  don't push it without asking first), `simulation/venv/`, and
  `simulation/data/` (raw datasets, large — see README.md Section 2 to
  re-download). Still treat file deletions as risky regardless; prefer
  move-and-backup (as was done with the FYP CSV) over deleting.
- The user is a beginner in software engineering — explain setup/run steps
  in full, don't assume familiarity with Python tooling.
- Primary shell is PowerShell on Windows; venv is at `simulation/venv/`.
