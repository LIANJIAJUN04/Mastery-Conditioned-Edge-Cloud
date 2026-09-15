"""
Loader for ROSMA (Robotic Surgical Maneuvers dataset) -- dataset #4, and the
first dataset in this project with a real PSYCHOMOTOR-SKILL learning signal.

UCI HAR (#1) and PAMAP2 (#2) validated the "sensing -> tiny edge model ->
uncertainty" pipeline on daily-activity recognition. ASSISTments (#3) fit
real BKT parameters on cognitive (maths) learning data. None of the three
has a skill/mastery label on a motor-skill task, so RQ1 -- can mastery be
estimated from real sensor data, in this project's actual domain of
physical/psychomotor skill training -- was still untested on real data
(see CLAUDE.md "Next steps / Phase 4"). JIGSAWS (the closest public match,
surgical robot kinematics with novice/intermediate/expert labels) was
access-request-only and was REJECTED. ROSMA is the backup: da Vinci
Research Kit (dVRK) kinematic data, 12 users, 3 training maneuvers, 6
repetitions each, performance-scored per trial.

Source: https://zenodo.org/records/3932964 (public, no access request).

Dataset structure actually used here (see simulation/data/ROSMA/):
  - kinematics/<user>_<task>_<rep>.csv : one file per trial, 50 Hz, 155
    columns = Date (string timestamp, dropped) + 154 numeric kinematic
    channels: position/orientation/velocity(linear+angular)/wrench(force+
    torque)/joint(position+velocity+effort) for 4 manipulators -- MTML/MTMR
    (master consoles, 7 joints each) and PSM1/PSM2 (patient-side robot
    arms, 6 joints each).
  - scores.csv : File_name,Time,Errors,Score per trial. Score = Time +
    15*Errors (a fixed error penalty baked into the dataset's own scoring;
    verified empirically against every row, not assumed).

Task naming note: the top-level ROSMA README calls the third maneuver
"Wire Chaser II" in one place but the actual files (and this project's
data) are named "Wire_Chaser_I" -- used as-is from the real filenames.

Mastery proxy used here (there is no explicit novice/expert label per
trial, only a continuous Time/Errors/Score -- unlike JIGSAWS): within each
(user, task) sequence of 6 repetitions, the EARLY attempts (rep 1-2) are
treated as lower-mastery and the LATE attempts (rep 5-6) as higher-mastery;
the middle two repetitions (3-4) are left out of the binary training label
(ambiguous transition zone) but still used at evaluation time to check
whether predicted mastery probability trends monotonically across all 6
reps -- a stronger test than the binary label alone. This mirrors the
proposal's own idea of discrete mastery bands better than trying to invent
a synthetic BKT fit on top of only 6 attempts per sequence.

Known data issue (documented, not silently fixed): X11_Pea_on_a_Peg_01 has
Time=113472 / Score=128472 in the source scores.csv -- roughly 1000x every
other trial in the same task (which range ~77-290s), an obvious decimal-
point logging error in the original dataset. This single corrupted row is
dropped rather than guessed-and-corrected.

Feature extraction: per trial, mean/std/min/max of each of the 154 numeric
kinematic channels over the whole trial recording (616 features/trial) --
whole-trial summary statistics, not sub-windowed, because the mastery
label here is inherently per-attempt (like a BKT "opportunity"), not
per-instant, so windowing would just replicate one label across many rows
for no benefit. Honest caveat (documented up front, not discovered after
the fact): ~140 labelled training trials against 616 features is a real
small-sample/high-dimension ratio -- treat results as a first directional
signal, not a tightly powered estimate.
"""

from __future__ import annotations
import glob
import os
import re

import numpy as np
import pandas as pd

TASKS = ["Post_and_Sleeve", "Pea_on_a_Peg", "Wire_Chaser_I"]

_TRIAL_RE = re.compile(r"^(X\d{2})_(Post_and_Sleeve|Pea_on_a_Peg|Wire_Chaser_I)_(\d{2})$")

# Sanity bound on trial Time (seconds); every legitimate trial in scores.csv
# is well under this. Anything above is treated as the known corrupted-row
# issue (see module docstring) and dropped.
_MAX_SANE_TIME_S = 600.0

EARLY_REPS = {1, 2}
LATE_REPS = {5, 6}


def _load_scores(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "ROSMA", "scores.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df[["File_name", "Time", "Errors", "Score"]].copy()
    df["File_name"] = df["File_name"].str.strip()

    n_before = len(df)
    df = df[df["Time"] <= _MAX_SANE_TIME_S].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"Dropped {n_dropped} trial(s) with Time > {_MAX_SANE_TIME_S}s "
              "(known corrupted-row issue in scores.csv, see src/rosma_data.py docstring).")

    parsed = df["File_name"].str.extract(_TRIAL_RE)
    parsed.columns = ["user_id", "task", "rep_str"]
    df["user_id"] = parsed["user_id"]
    df["task"] = parsed["task"]
    df["repetition"] = parsed["rep_str"].astype(int)
    df = df.dropna(subset=["user_id", "task"]).reset_index(drop=True)
    return df


def _extract_trial_features(csv_path: str) -> dict:
    """Whole-trial mean/std/min/max per numeric kinematic channel."""
    raw = pd.read_csv(csv_path)
    channel_cols = [c for c in raw.columns if c != "Date"]
    values = raw[channel_cols].to_numpy(dtype=float)

    feats = {}
    for i, col in enumerate(channel_cols):
        v = values[:, i]
        feats[f"{col}_mean"] = v.mean()
        feats[f"{col}_std"] = v.std()
        feats[f"{col}_min"] = v.min()
        feats[f"{col}_max"] = v.max()
    feats["n_samples"] = len(raw)
    return feats


def load_rosma(data_dir: str) -> pd.DataFrame:
    """Loads every available ROSMA trial: whole-trial kinematic features
    (616 cols) + metadata (user_id, task, repetition, time_s, errors,
    score, mastery_label).

    mastery_label is "early" for repetition in {1,2}, "late" for repetition
    in {5,6}, and None for repetition in {3,4} (middle reps are still
    present in the returned DataFrame with mastery_label=None, for the
    monotonic-trend check -- filter them out for binary-classifier training,
    see feature_label_arrays).
    """
    kin_dir = os.path.join(data_dir, "ROSMA", "kinematics")
    if not os.path.isdir(kin_dir) or not glob.glob(os.path.join(kin_dir, "*.csv")):
        raise FileNotFoundError(
            f"No ROSMA kinematic CSVs found under {kin_dir}. "
            "Download the ROSMA dataset first (see README.md)."
        )

    scores = _load_scores(data_dir)

    rows = []
    missing_kin = []
    for _, meta in scores.iterrows():
        trial_name = meta["File_name"]
        csv_path = os.path.join(kin_dir, f"{trial_name}.csv")
        if not os.path.exists(csv_path):
            missing_kin.append(trial_name)
            continue
        feats = _extract_trial_features(csv_path)
        feats.update({
            "user_id": meta["user_id"],
            "task": meta["task"],
            "repetition": int(meta["repetition"]),
            "time_s": float(meta["Time"]),
            "errors": int(meta["Errors"]),
            "score": float(meta["Score"]),
        })
        rep = feats["repetition"]
        feats["mastery_label"] = "early" if rep in EARLY_REPS else ("late" if rep in LATE_REPS else None)
        rows.append(feats)

    if missing_kin:
        print(f"{len(missing_kin)} trial(s) listed in scores.csv have no kinematic CSV on disk "
              f"(download incomplete?), e.g. {missing_kin[:5]}")

    return pd.DataFrame(rows)


def feature_label_arrays(df: pd.DataFrame, labelled_only: bool = True):
    """Splits the loaded DataFrame into (X features, y binary label, user_ids).

    If labelled_only, keeps only rows with mastery_label in {"early","late"}
    (drops the middle repetitions) and returns y as 0/1 (late=1). Otherwise
    returns every row with y = repetition number (for the monotonic-trend
    check), including middle repetitions.
    """
    meta_cols = {"user_id", "task", "repetition", "time_s", "errors", "score", "mastery_label"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    if labelled_only:
        sub = df[df["mastery_label"].isin(["early", "late"])].reset_index(drop=True)
        X = sub[feature_cols].to_numpy()
        y = (sub["mastery_label"] == "late").astype(int).to_numpy()
        users = sub["user_id"].to_numpy()
        return X, y, users, sub

    X = df[feature_cols].to_numpy()
    y = df["repetition"].to_numpy()
    users = df["user_id"].to_numpy()
    return X, y, users, df
