"""
Shared train/validate/test helper for the two RandomForest edge-model
surrogates (UCI HAR, PAMAP2). Model SELECTION happens on the validation
split only; the test split is touched exactly once, for the final reported
number -- the standard discipline a 70/20/10 split exists to enforce.
"""

from __future__ import annotations
import itertools

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 20],
}


def fit_with_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
) -> dict:
    """Small grid search over RandomForest hyperparameters, selected by
    validation accuracy only. Returns the refit-on-train-only best model
    plus the grid search trace, so the choice is auditable rather than a
    black box.
    """
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))

    trace = []
    best_val_acc = -1.0
    best_params = None
    best_clf = None

    for combo in combos:
        params = dict(zip(keys, combo))
        clf = RandomForestClassifier(random_state=random_state, n_jobs=-1, **params)
        clf.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, clf.predict(X_val))
        trace.append({**params, "val_accuracy": float(val_acc)})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            best_clf = clf

    return {
        "best_clf": best_clf,
        "best_params": best_params,
        "best_val_accuracy": float(best_val_acc),
        "grid_search_trace": trace,
    }
