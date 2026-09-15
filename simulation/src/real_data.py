"""
Loader for the UCI HAR (Human Activity Recognition Using Smartphones) dataset.

This is REAL wearable IMU data (waist-worn smartphone accelerometer + gyroscope,
30 subjects, 6 activities). It has no "mastery" or "skill" labels -- it is not
a psychomotor-skill dataset -- but it lets us test something the proposal's
RQ1 depends on: can a lightweight model consume real wearable IMU windows and
produce (a) reasonable accuracy and (b) a usable predictive-uncertainty signal?
That is the "sensing -> tiny edge model -> uncertainty" pipeline described in
Sections 9.1 and 9.4, independent of the skill-training domain.

Dataset: https://archive.ics.uci.edu/dataset/240
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def _read_whitespace_table(path: str) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None).to_numpy()


def load_uci_har(data_dir: str) -> dict:
    """Loads the pre-extracted 561-feature vectors (fast, no deep learning needed)
    plus the official subject-disjoint train/test split and subject IDs.

    Returns a dict with keys: X_train, y_train, subj_train, X_test, y_test, subj_test
    """
    root = os.path.join(data_dir, "UCI HAR Dataset")

    X_train = _read_whitespace_table(os.path.join(root, "train", "X_train.txt"))
    y_train = _read_whitespace_table(os.path.join(root, "train", "y_train.txt")).ravel()
    subj_train = _read_whitespace_table(os.path.join(root, "train", "subject_train.txt")).ravel()

    X_test = _read_whitespace_table(os.path.join(root, "test", "X_test.txt"))
    y_test = _read_whitespace_table(os.path.join(root, "test", "y_test.txt")).ravel()
    subj_test = _read_whitespace_table(os.path.join(root, "test", "subject_test.txt")).ravel()

    return {
        "X_train": X_train,
        "y_train": y_train,
        "subj_train": subj_train,
        "X_test": X_test,
        "y_test": y_test,
        "subj_test": subj_test,
    }


_UNCERTAINTY_SAMPLE_FILES = [
    "real_uncertainty_samples.csv",         # dataset #1: UCI HAR (train_activity_classifier.py)
    "real_uncertainty_samples_pamap2.csv",  # dataset #2: PAMAP2 (train_pamap2_classifier.py)
]


def load_uci_har_pooled(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pools the dataset's official train+test pre-extracted feature vectors
    into one array, keeping subject ids, so callers can re-split by subject
    into train/test/validation proportions of their own choosing (see
    `src/data_splits.py::split_groups`) instead of the original authors'
    fixed 21/9-subject split.
    """
    data = load_uci_har(data_dir)
    X = np.vstack([data["X_train"], data["X_test"]])
    y = np.concatenate([data["y_train"], data["y_test"]])
    subj = np.concatenate([data["subj_train"], data["subj_test"]])
    return X, y, subj


def load_empirical_uncertainty(results_dir: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Loads the (uncertainty, correct) samples written by the real-data
    classifier scripts and returns (uncertainty_when_correct,
    uncertainty_when_incorrect), both min-max rescaled to [0, 1] using the
    POOLED sample range so the arrays stay on a shared, comparable scale.

    Pools samples across BOTH real wearable-IMU datasets (UCI HAR + PAMAP2)
    when both are available -- two independently collected real datasets
    give a more robust empirical uncertainty distribution than either alone,
    and this is the only place PAMAP2 actually feeds INTO the simulation
    (its other role is a standalone replication check, see
    train_pamap2_classifier.py). Falls back to whichever single dataset is
    present, or None if neither has been run yet (Phase 2a not done) --
    callers should fall back to the synthetic uncertainty model in that case.
    """
    frames = []
    for fname in _UNCERTAINTY_SAMPLE_FILES:
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    unc = df["uncertainty"].to_numpy()
    correct = df["correct"].to_numpy().astype(bool)

    lo, hi = unc.min(), unc.max()
    unc_norm = (unc - lo) / (hi - lo) if hi > lo else unc

    return unc_norm[correct], unc_norm[~correct]
