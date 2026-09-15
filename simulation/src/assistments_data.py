"""
Loader/cleaner for the ASSISTments 2009-2010 "skill builder" dataset --
dataset #3 of 3.

Unlike UCI HAR and PAMAP2 (which are wearable-IMU/perception-layer datasets
with no mastery labels), this is REAL STUDENT LEARNING DATA: ~525k logged
problem-solving attempts, each tagged with a skill and correct/incorrect.
Its role here is different from the other two datasets: it is used to fit
the actual Bayesian Knowledge Tracing model (`src/bkt_fit.py`) against real
per-skill attempt sequences, so Phase 1's mastery simulator
(`src/mastery.py::BKTLearner`) can be re-parameterised with empirically
fitted p_init/p_learn/p_slip/p_guess ranges instead of hand-picked ones.

This is a different domain (cognitive/academic skills, not psychomotor
motor-skill training) -- it validates the BKT *machinery* against real
learner data, not the wearable-sensing pipeline. See README.md and
CLAUDE.md's "Next steps" section, which flagged this dataset as the
priority follow-up for exactly this reason.

Cleaning steps applied (each is a documented, standard step for this
specific, well-studied dataset -- see Xiong et al. 2016, "Going Deeper with
Deep Knowledge Tracing", for the known data-quality issues being addressed):
  1. Drop rows with a null/missing `skill_name` (a large fraction of this
     dataset is untagged; those rows cannot be used for per-skill BKT).
  2. Keep only `original == 1` (the first, non-scaffolding presentation of a
     problem) -- scaffolding follow-up problems are a different, easier
     construct and would contaminate a single skill's difficulty.
  3. Drop exact-duplicate rows (same user_id, assignment_id, problem_id,
     order_id) -- a small number of rows are duplicated by a known logging
     bug in this dataset.
  4. Sort each (user_id, skill_name) group by `order_id` to reconstruct the
     chronological attempt sequence BKT needs (order_id is this dataset's
     canonical global ordering, not wall-clock time, which is sometimes
     missing/inconsistent).
  5. Keep only skills with at least MIN_USERS_PER_SKILL distinct learners
     (sparse skills give unstable EM estimates), then further restrict to
     the TOP_N_SKILLS by total attempt count, purely to keep EM fitting
     time bounded -- documented, not hidden.
"""

from __future__ import annotations
import os

import numpy as np
import pandas as pd

from .data_splits import split_groups

MIN_USERS_PER_SKILL = 100
TOP_N_SKILLS = 25
MAX_USERS_PER_SKILL = 500  # cap on eligible users considered per skill, for EM runtime
RANDOM_SEED = 42


def load_and_clean_assistments(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "assistments", "as.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ASSISTments CSV not found at {path}. See README.md for the download step.")

    df = pd.read_csv(
        path,
        usecols=["order_id", "user_id", "assignment_id", "problem_id", "original", "correct", "skill_name"],
        low_memory=False,
        encoding="latin-1",  # this dataset contains non-UTF-8 bytes (e.g. in skill_name)
    )
    n_raw = len(df)

    # Step 1: drop untagged rows.
    df = df.dropna(subset=["skill_name"])
    n_after_skill = len(df)

    # Step 2: keep only original (non-scaffolding) presentations.
    df = df[df["original"] == 1]
    n_after_original = len(df)

    # Step 3: drop exact-duplicate logging rows.
    df = df.drop_duplicates(subset=["user_id", "assignment_id", "problem_id", "order_id"])
    n_after_dedup = len(df)

    df = df[["order_id", "user_id", "skill_name", "correct"]].copy()
    df["correct"] = df["correct"].astype(int)

    print(
        f"[assistments cleaning] raw={n_raw} -> skill-tagged={n_after_skill} "
        f"-> original-only={n_after_original} -> deduped={n_after_dedup}"
    )
    return df


def build_skill_sequences(df: pd.DataFrame) -> dict[str, dict[str, list[np.ndarray]]]:
    """Groups cleaned attempts into per-skill, per-STUDENT sequences, then
    splits each skill's students 70/20/10 into train/test/validation (step 5)
    using `src/data_splits.py::split_groups` -- splitting by student, not by
    row, for the same leakage reason UCI HAR/PAMAP2 split by subject.

    Returns {skill_name: {"train": [...], "test": [...], "val": [...]}}.
    """
    users_per_skill = df.groupby("skill_name")["user_id"].nunique()
    eligible_skills = users_per_skill[users_per_skill >= MIN_USERS_PER_SKILL]

    attempts_per_skill = df[df["skill_name"].isin(eligible_skills.index)].groupby("skill_name").size()
    top_skills = attempts_per_skill.sort_values(ascending=False).head(TOP_N_SKILLS).index.tolist()

    rng = np.random.default_rng(RANDOM_SEED)
    sequences: dict[str, dict[str, list[np.ndarray]]] = {}
    for skill in top_skills:
        skill_df = df[df["skill_name"] == skill]
        user_ids = skill_df["user_id"].unique()
        if len(user_ids) > MAX_USERS_PER_SKILL:
            user_ids = rng.choice(user_ids, size=MAX_USERS_PER_SKILL, replace=False)

        train_users, test_users, val_users = split_groups(user_ids, 0.7, 0.2, 0.1, seed=RANDOM_SEED)

        skill_df = skill_df[skill_df["user_id"].isin(set(user_ids))].sort_values(["user_id", "order_id"])
        seq_by_user = {uid: g["correct"].to_numpy() for uid, g in skill_df.groupby("user_id")}

        sequences[skill] = {
            "train": [seq_by_user[u] for u in train_users if len(seq_by_user[u]) >= 2],
            "test": [seq_by_user[u] for u in test_users if len(seq_by_user[u]) >= 2],
            "val": [seq_by_user[u] for u in val_users if len(seq_by_user[u]) >= 2],
        }

    return sequences
