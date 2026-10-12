"""
Loader for the self-collected pilot data (Phase 5, item 1) -- the only
dataset in this project that is actually IN the proposal's own domain
(network-cabling psychomotor skill), unlike ROSMA (dataset #4, surgical
teleoperation). See CLAUDE.md "Next steps / Phase 5" for the full
background and the decisions this loader implements.

Status (see CLAUDE.md): decisions on app/export/storage are resolved and
`simulation/data/self_collected/sessions.csv` exists as a header-only
template, but NO real recordings have been collected yet -- that is
blocked on acquiring a tool-less/reusable keystone jack and cable, a
physical step this code cannot do. This module's parsing logic was
verified ad hoc against a clearly-labelled SYNTHETIC fixture built in a
temp directory outside this repo (not committed, not placed under
simulation/data/) so that once real recordings exist, this loader should
need no further changes -- only real `sessions.csv` rows and session
folders need to be added. Do NOT point this loader at fabricated data and
report its output as a real RQ1 result; the synthetic fixture is for
verifying the PARSING CODE only, exactly as flagged as a research-
integrity concern in Week 9 (see `Weekly Report/Week9_Report.md`).

Expected directory layout (see CLAUDE.md "Storage convention"):
  simulation/data/self_collected/
    sessions.csv   -- columns: session_id, task, rep_number, date,
                       completion_time_sec, error_count, notes
    <task>_rep<NN>_<YYYYMMDD_HHMMSS>/   -- one subfolder per session_id,
                       holding that session's raw Sensor Logger export:
                       Accelerometer.csv, Gyroscope.csv, Metadata.csv

Sensor Logger (Kelvin Choi) CSV export format assumed here: each sensor
CSV has a `seconds_elapsed` column (float, seconds since recording start)
plus per-axis columns `x`, `y`, `z` (case-insensitive match used below to
tolerate either naming). HONEST CAVEAT: this column layout is Sensor
Logger's documented default export and matches the app's publicly shown
sample exports, but it has NOT been verified against a real recording from
this project yet (none exist) -- if the real export differs (e.g. extra
columns, different axis names), `_read_sensor_csv` below is the only place
that needs adjusting.

Feature extraction: per session, mean/std/min/max of each of the 6 raw
channels (accelerometer x/y/z + gyroscope x/y/z) over the whole recording
= 24 features/session. This deliberately mirrors ROSMA's whole-trial
mean/std/min/max convention (src/rosma_data.py) rather than HAR/PAMAP2's
windowed convention, for the same reason: the mastery label here is
per-attempt, not per-instant. The feature COUNT is cut down hard relative
to ROSMA's 616 (which already had a documented small-sample/high-dimension
caveat at ~140 trials) because this pilot will have on the order of 6-8
trials total -- 616 features over 6-8 trials would be unusably degenerate,
so only the 2 raw channel groups this task actually has (no 4-manipulator
kinematic rig here) are used, kept as 24 features.

Mastery proxy: generalises ROSMA's fixed early={1,2}/late={5,6} convention
to whatever repetition numbers actually exist per task in sessions.csv
(Week 9's plan was "at least 6-8 punch-down repetitions", not necessarily
exactly 6) -- the lowest 2 repetition numbers per task are "early", the
highest 2 are "late", everything in between is held out of binary
training and used only for the monotonic-trend check, exactly as in
src/rosma_data.py. With only one task (punch-down) and n in the 6-8
range, this will still be a small-sample first signal, not a powered
test -- same honest framing as Phase 4's ROSMA result.
"""

from __future__ import annotations
import os

import numpy as np
import pandas as pd

ACCEL_CHANNELS = ["accel_x", "accel_y", "accel_z"]
GYRO_CHANNELS = ["gyro_x", "gyro_y", "gyro_z"]
ALL_CHANNELS = ACCEL_CHANNELS + GYRO_CHANNELS


def _read_sensor_csv(session_dir: str, filename: str, prefix: str) -> pd.DataFrame:
    """Reads one Sensor Logger export CSV (Accelerometer.csv or
    Gyroscope.csv) and returns a DataFrame with columns
    [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"], tolerant of x/y/z being
    upper- or lower-case in the real export.
    """
    path = os.path.join(session_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected Sensor Logger export '{filename}' not found in {session_dir}. "
            "Check the session folder was unzipped correctly (see CLAUDE.md "
            "'Storage convention')."
        )
    raw = pd.read_csv(path)
    col_map = {c.lower(): c for c in raw.columns}
    out = pd.DataFrame()
    for axis in ("x", "y", "z"):
        if axis not in col_map:
            raise ValueError(
                f"{filename} in {session_dir} has no '{axis}' column (columns found: "
                f"{list(raw.columns)}). Sensor Logger's export format may differ from "
                "what this loader assumes -- see the module docstring's HONEST CAVEAT."
            )
        out[f"{prefix}_{axis}"] = raw[col_map[axis]].to_numpy(dtype=float)
    return out


def _extract_session_features(session_dir: str) -> dict:
    """Whole-session mean/std/min/max per channel (24 features)."""
    accel = _read_sensor_csv(session_dir, "Accelerometer.csv", "accel")
    gyro = _read_sensor_csv(session_dir, "Gyroscope.csv", "gyro")

    feats = {}
    for prefix, df in (("accel", accel), ("gyro", gyro)):
        for axis in ("x", "y", "z"):
            col = f"{prefix}_{axis}"
            v = df[col].to_numpy(dtype=float)
            feats[f"{col}_mean"] = v.mean()
            feats[f"{col}_std"] = v.std()
            feats[f"{col}_min"] = v.min()
            feats[f"{col}_max"] = v.max()
    feats["n_accel_samples"] = len(accel)
    feats["n_gyro_samples"] = len(gyro)
    return feats


def _load_sessions_log(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "self_collected", "sessions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No sessions.csv found at {path}. See CLAUDE.md 'Storage convention'."
        )
    df = pd.read_csv(path)
    required = {"session_id", "task", "rep_number", "date",
                "completion_time_sec", "error_count", "notes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sessions.csv is missing expected column(s): {sorted(missing)}")
    return df


def load_self_collected(data_dir: str) -> pd.DataFrame:
    """Loads every logged self-collected session: per-session kinematic
    features (24 cols) + metadata (task, repetition, completion_time_sec,
    error_count, mastery_label).

    mastery_label is "early" for the 2 lowest repetition numbers per task,
    "late" for the 2 highest, and None for everything in between (same
    convention as src/rosma_data.py; see module docstring).

    Raises a clear, actionable error (rather than returning an empty/odd
    result) if sessions.csv has no rows yet -- the expected state until
    real recordings are collected (see CLAUDE.md "Next steps / Phase 5").
    """
    sessions = _load_sessions_log(data_dir)
    if len(sessions) == 0:
        raise ValueError(
            "sessions.csv exists but has no recorded sessions yet. "
            "Record at least a few punch-down repetitions with Sensor Logger "
            "and add one row per session before calling load_self_collected() "
            "(see CLAUDE.md 'Next steps / Phase 5', item 1)."
        )

    base_dir = os.path.join(data_dir, "self_collected")
    rows = []
    missing_dirs = []
    for _, meta in sessions.iterrows():
        session_dir = os.path.join(base_dir, str(meta["session_id"]))
        if not os.path.isdir(session_dir):
            missing_dirs.append(meta["session_id"])
            continue
        feats = _extract_session_features(session_dir)
        feats.update({
            "session_id": meta["session_id"],
            "task": meta["task"],
            "repetition": int(meta["rep_number"]),
            "completion_time_sec": float(meta["completion_time_sec"]),
            "error_count": int(meta["error_count"]),
        })
        rows.append(feats)

    if missing_dirs:
        print(f"{len(missing_dirs)} session(s) listed in sessions.csv have no matching "
              f"folder under {base_dir}, e.g. {missing_dirs[:5]}")

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError(
            "sessions.csv had rows but none of their session folders were found on disk "
            f"under {base_dir}. Check the folder-naming convention in CLAUDE.md."
        )

    df["mastery_label"] = None
    for task, group in df.groupby("task"):
        reps_sorted = sorted(group["repetition"].unique())
        early_reps = set(reps_sorted[:2])
        late_reps = set(reps_sorted[-2:])
        idx = group.index
        df.loc[idx, "mastery_label"] = df.loc[idx, "repetition"].apply(
            lambda r: "early" if r in early_reps else ("late" if r in late_reps else None)
        )
    return df


def feature_label_arrays(df: pd.DataFrame, labelled_only: bool = True):
    """Same shape/contract as src/rosma_data.py::feature_label_arrays, for
    a future train_self_collected_classifier.py to reuse the same
    early-vs-late RQ1-style check Phase 4 ran on ROSMA.
    """
    meta_cols = {"session_id", "task", "repetition", "completion_time_sec",
                 "error_count", "mastery_label", "n_accel_samples", "n_gyro_samples"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    if labelled_only:
        sub = df[df["mastery_label"].isin(["early", "late"])].reset_index(drop=True)
        X = sub[feature_cols].to_numpy()
        y = (sub["mastery_label"] == "late").astype(int).to_numpy()
        sessions = sub["session_id"].to_numpy()
        return X, y, sessions, sub

    X = df[feature_cols].to_numpy()
    y = df["repetition"].to_numpy()
    sessions = df["session_id"].to_numpy()
    return X, y, sessions, df


if __name__ == "__main__":
    # Quick self-check against the real (currently header-only) sessions.csv:
    # confirms the "no rows yet" path raises the intended, actionable error
    # rather than crashing obscurely. This is NOT a test against real or
    # fabricated recordings -- see tests/test_self_collected_data.py (run via
    # a synthetic fixture under a temp directory, never under
    # simulation/data/) for the parsing-logic check referenced in the
    # module docstring.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        load_self_collected(os.path.join(here, "data"))
        print("Unexpected: load_self_collected succeeded with no real sessions recorded yet.")
    except ValueError as e:
        print(f"OK -- expected error with no recordings yet: {e}")
    print(
        "Parsing-logic check (feature extraction + early/late labelling) against a "
        "clearly-labelled SYNTHETIC fixture was run ad hoc in a temp directory outside "
        "this repo, not committed here and not a real or fabricated-as-real dataset -- "
        "see Weekly Report for the date this was last verified."
    )
