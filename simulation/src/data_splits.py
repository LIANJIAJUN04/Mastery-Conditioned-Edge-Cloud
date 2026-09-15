"""
Shared, deterministic GROUP-disjoint train/test/validation splitting.

Every real dataset in this project is split by a GROUP id (subject id for
UCI HAR/PAMAP2, student/user id for ASSISTments), never by row/trial. This
is the same methodology point Section 13.2 validates empirically (a random
row-level split leaks information across splits when several rows come from
the same person) -- applying it consistently to all three datasets, and to
all three splits (not just train/test), keeps that finding from being
quietly violated the moment a validation set is added.

Default proportions: 70% train / 20% test / 10% validation.
"""

from __future__ import annotations
import numpy as np


def split_groups(
    group_ids,
    train_frac: float = 0.7,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[set, set, set]:
    """Splits the UNIQUE values of `group_ids` into (train_ids, test_ids, val_ids)
    sets, disjoint by group, deterministically shuffled by `seed`.

    Rounding favours giving every split at least 1 group when there are at
    least 3 unique groups (relevant for small-N datasets like PAMAP2's 9
    subjects), taking the extra unit from whichever of train/test is larger.
    """
    if abs(train_frac + test_frac + val_frac - 1.0) > 1e-9:
        raise ValueError("train_frac + test_frac + val_frac must sum to 1.0")

    unique = np.array(sorted(set(group_ids)))
    n = len(unique)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)

    n_train = max(1, round(n * train_frac))
    n_test = max(1, round(n * test_frac))
    n_val = n - n_train - n_test
    if n_val < 1 and n >= 3:
        n_val = 1
        if n_train >= n_test:
            n_train -= 1
        else:
            n_test -= 1

    train_ids = set(shuffled[:n_train].tolist())
    test_ids = set(shuffled[n_train : n_train + n_test].tolist())
    val_ids = set(shuffled[n_train + n_test : n_train + n_test + n_val].tolist())
    return train_ids, test_ids, val_ids
