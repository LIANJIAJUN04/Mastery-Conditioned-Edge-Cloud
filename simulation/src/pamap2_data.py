"""
Loader for the PAMAP2 (Physical Activity Monitoring) dataset -- dataset #2 of 3.

This is a SECOND, INDEPENDENT real wearable-IMU dataset (3 body-worn IMUs --
wrist/hand, chest, ankle -- plus heart rate; 9 subjects, 12+ daily activities).
Its role in this project is deliberately narrow and honest: UCI HAR (dataset
#1) already established two findings about the "sensing -> tiny edge model ->
uncertainty" pipeline (Sections 9.4 and 13.2 of the proposal). PAMAP2 exists
to check whether those two findings REPLICATE on a second, independently
collected real dataset with a different (and more "wearable-like", multi-IMU)
sensor configuration -- rather than being an artefact of HAR's specific
waist-mounted single-phone setup. Like HAR, PAMAP2 has no skill/mastery
labels; it is still a daily-activity dataset, not a psychomotor-skill dataset.

Dataset: https://archive.ics.uci.edu/dataset/231
Raw format per official readme: 54 whitespace-separated columns per row,
100 Hz:
    col 1        timestamp (s)
    col 2        activityID (0 = transient/other -> not a real activity)
    col 3        heart rate (bpm) -- sparse (~9 Hz), unused here
    cols 4-20    hand/wrist IMU  (temp, acc±16g x3, acc±6g x3, gyro x3, mag x3, orientation x4[invalid])
    cols 21-37   chest IMU       (same 17-column layout)
    cols 38-54   ankle IMU       (same 17-column layout)

Cleaning/transformation applied here (see README.md for the full writeup):
  1. Drop activityID == 0 (transient/"other", not a labelled activity --
     this is the dataset's own documented convention, not our invention).
  2. Drop the orientation sub-columns (the dataset's own readme states these
     are invalid in this data collection) and the ±6g accelerometer range
     (redundant with the ±16g range) and heart rate (too sparse for a 100 Hz
     feature window) -- keep only acc±16g (x,y,z) and gyro (x,y,z) per IMU,
     18 raw channels total.
  3. Wireless dropout produces scattered NaNs in the kept channels: per
     subject, sort by timestamp and linearly interpolate short gaps, then
     drop any window that still contains a NaN after interpolation.
  4. Slide a 256-sample (2.56 s) window with 50% overlap (same window length
     convention as UCI HAR) over each subject's channel-cleaned recording;
     keep only windows whose activityID is constant throughout (windows that
     straddle an activity change are dropped rather than mislabelled).
  5. Per window, extract mean/std/min/max for each of the 18 channels (72
     features/window) -- a deliberately simple, transparent feature set
     (not a reproduction of HAR's proprietary 561-feature pipeline), enough
     to train a comparable RandomForest and re-run the same two checks.
"""

from __future__ import annotations
import glob
import os

import numpy as np
import pandas as pd

ACTIVITY_NAMES = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "Nordic_walking",
    9: "watching_TV",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}

# 0-based column indices into the raw 54-column .dat rows.
_ACC16 = {"hand": [4, 5, 6], "chest": [21, 22, 23], "ankle": [38, 39, 40]}
_GYRO = {"hand": [10, 11, 12], "chest": [27, 28, 29], "ankle": [44, 45, 46]}
_CHANNEL_COLS = sum(_ACC16.values(), []) + sum(_GYRO.values(), [])  # 18 raw channels
_TIMESTAMP_COL = 0
_ACTIVITY_COL = 1

WINDOW_SIZE = 256  # samples (2.56 s at 100 Hz), matching UCI HAR's window convention
WINDOW_STRIDE = 128  # 50% overlap


def _load_one_subject(path: str) -> pd.DataFrame:
    """Reads one subject's raw .dat file and applies steps 1-3 of the cleaning
    pipeline described in the module docstring."""
    raw = pd.read_csv(path, sep=r"\s+", header=None, na_values="NaN")
    raw = raw.sort_values(_TIMESTAMP_COL).reset_index(drop=True)

    # Step 1: drop transient/unlabelled activity.
    raw = raw[raw[_ACTIVITY_COL] != 0].reset_index(drop=True)

    # Step 2: keep only the 18 channels we use, plus timestamp/activity.
    keep_cols = [_TIMESTAMP_COL, _ACTIVITY_COL] + _CHANNEL_COLS
    df = raw[keep_cols].copy()
    df.columns = ["timestamp", "activity_id"] + [f"ch{i}" for i in range(len(_CHANNEL_COLS))]

    # Step 3: interpolate short wireless-dropout gaps, then drop any
    # window that still has a NaN (handled at windowing time).
    channel_cols = [c for c in df.columns if c.startswith("ch")]
    df[channel_cols] = df[channel_cols].interpolate(limit=20, limit_direction="both")

    return df


def _window_one_subject(df: pd.DataFrame, subject_id: int) -> list[dict]:
    """Step 4-5: slide fixed windows over one subject's cleaned recording and
    extract mean/std/min/max features per channel."""
    channel_cols = [c for c in df.columns if c.startswith("ch")]
    activity = df["activity_id"].to_numpy()
    values = df[channel_cols].to_numpy()

    rows = []
    n = len(df)
    for start in range(0, n - WINDOW_SIZE + 1, WINDOW_STRIDE):
        end = start + WINDOW_SIZE
        window_activity = activity[start:end]
        if not np.all(window_activity == window_activity[0]):
            continue  # straddles an activity change -- drop rather than mislabel
        window_vals = values[start:end]
        if np.isnan(window_vals).any():
            continue  # dropout survived interpolation -- drop this window

        feats = {}
        for i, col in enumerate(channel_cols):
            v = window_vals[:, i]
            feats[f"{col}_mean"] = v.mean()
            feats[f"{col}_std"] = v.std()
            feats[f"{col}_min"] = v.min()
            feats[f"{col}_max"] = v.max()
        feats["activity_id"] = int(window_activity[0])
        feats["subject_id"] = subject_id
        rows.append(feats)
    return rows


def load_pamap2(data_dir: str) -> pd.DataFrame:
    """Loads all subjects in `data_dir`/PAMAP2_Dataset/Protocol/subject*.dat,
    applies cleaning steps 1-3, then windows/feature-extracts (steps 4-5).

    Returns a single DataFrame: one row per window, columns = 72 features +
    activity_id + subject_id.
    """
    protocol_dir = os.path.join(data_dir, "PAMAP2_Dataset", "Protocol")
    paths = sorted(glob.glob(os.path.join(protocol_dir, "subject*.dat")))
    if not paths:
        raise FileNotFoundError(
            f"No PAMAP2 subject*.dat files found under {protocol_dir}. "
            "Download+extract the PAMAP2 dataset first (see README.md)."
        )

    all_rows = []
    for path in paths:
        subject_id = int(os.path.basename(path).replace("subject", "").replace(".dat", ""))
        cleaned = _load_one_subject(path)
        all_rows.extend(_window_one_subject(cleaned, subject_id))

    return pd.DataFrame(all_rows)


def feature_label_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits a windowed DataFrame into (X features, y activity labels, subject ids)."""
    feature_cols = [c for c in df.columns if c not in ("activity_id", "subject_id")]
    X = df[feature_cols].to_numpy()
    y = df["activity_id"].to_numpy()
    subj = df["subject_id"].to_numpy()
    return X, y, subj
