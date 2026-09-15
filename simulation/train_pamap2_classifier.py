"""
Phase 2a-replicate: re-run the exact two checks from train_activity_classifier.py
(Phase 2a, UCI HAR) on PAMAP2 -- a second, independently collected real
wearable-IMU dataset with a different sensor configuration (3 body-worn IMUs
instead of one waist-mounted phone). The question this answers: do the two
Phase 2a findings hold up as general properties of "real wearable IMU +
lightweight classifier", or were they specific to HAR's setup?

  1. (Section 13.2) cross-subject split vs improper random split -- does the
     random split still inflate accuracy on a different dataset?
  2. (Section 9.4) is raw predictive uncertainty still usable (separates
     correct/incorrect) and still imperfectly calibrated, and in which
     direction (over- vs under-confident)?

Usage:
    python train_pamap2_classifier.py
"""

from __future__ import annotations
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from src.pamap2_data import load_pamap2, feature_label_arrays, ACTIVITY_NAMES
from src.data_splits import split_groups
from src.model_selection import fit_with_validation
from train_activity_classifier import entropy_uncertainty, expected_calibration_error

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Held out entirely from training for the proper cross-subject split -- the
# two subjects with the largest recordings, so the held-out set is not tiny.
TEST_SUBJECTS = {108, 109}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading + cleaning + windowing PAMAP2 (this can take ~30-60s)...")
    df = load_pamap2(DATA_DIR)
    X, y, subj = feature_label_arrays(df)
    print(f"Loaded {len(df)} windows across {len(set(subj))} subjects, "
          f"{len(set(y))} activity classes.")

    # --- Experiment A: proper cross-subject split ---
    train_mask = ~np.isin(subj, list(TEST_SUBJECTS))
    test_mask = np.isin(subj, list(TEST_SUBJECTS))
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    clf_proper = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_proper.fit(X_train, y_train)
    proper_pred = clf_proper.predict(X_test)
    proper_proba = clf_proper.predict_proba(X_test)
    proper_acc = accuracy_score(y_test, proper_pred)

    # --- Experiment B: improper random split, ignoring subject identity ---
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf_random = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_random.fit(Xr_train, yr_train)
    random_pred = clf_random.predict(Xr_test)
    random_acc = accuracy_score(yr_test, random_pred)

    print("\n=== PAMAP2 replicate of Section 13.2 check ===")
    print(f"Proper cross-subject-split accuracy : {proper_acc:.4f}")
    print(f"Improper random-split accuracy       : {random_acc:.4f}")
    print(f"Inflation from the invalid split      : {100*(random_acc - proper_acc):.2f} percentage points")

    # --- Proper 70/20/10 subject-disjoint train/val/test split ---
    # Same discipline as train_activity_classifier.py: hyperparameters are
    # chosen on the validation subject(s) only; the test subjects are
    # touched exactly once. With only 9 subjects total this gives roughly
    # 6 train / 2 test / 1 validation.
    train_ids, test_ids, val_ids = split_groups(subj, 0.7, 0.2, 0.1, seed=42)

    def _subset(ids):
        mask = np.isin(subj, list(ids))
        return X[mask], y[mask]

    X_tr, y_tr = _subset(train_ids)
    X_val, y_val = _subset(val_ids)
    X_te, y_te = _subset(test_ids)

    print(f"\n=== 70/20/10 subject-disjoint split: {sorted(train_ids)} train / "
          f"{sorted(test_ids)} test / {sorted(val_ids)} validation subjects ===")

    selection = fit_with_validation(X_tr, y_tr, X_val, y_val)
    clf_final = selection["best_clf"]
    print(f"Best hyperparameters (chosen on validation subject): {selection['best_params']} "
          f"(validation accuracy {selection['best_val_accuracy']:.4f})")

    final_pred = clf_final.predict(X_te)
    final_proba = clf_final.predict_proba(X_te)
    final_acc = accuracy_score(y_te, final_pred)
    print(f"Held-out TEST accuracy (touched once): {final_acc:.4f}")

    confidences = final_proba.max(axis=1)
    uncertainty = entropy_uncertainty(final_proba)
    correct = (final_pred == y_te).astype(int)

    mean_unc_correct = float(uncertainty[correct == 1].mean())
    mean_unc_incorrect = float(uncertainty[correct == 0].mean()) if (correct == 0).any() else None
    ece = expected_calibration_error(confidences, correct, n_bins=10)
    mean_confidence = float(confidences.mean())
    calibration_direction = "overconfident" if mean_confidence > final_acc else "underconfident"

    print("\n=== PAMAP2 replicate of Section 9.4 check ===")
    print(f"Mean uncertainty on CORRECT predictions   : {mean_unc_correct:.4f}")
    print(f"Mean uncertainty on INCORRECT predictions : {mean_unc_incorrect}")
    print(f"Expected Calibration Error (10 bins)      : {ece:.4f}")
    print(f"Mean confidence {mean_confidence:.4f} vs accuracy {final_acc:.4f} -> model is {calibration_direction}")

    report = {
        "dataset": "PAMAP2",
        "section_13_2_check": {
            "test_subjects_held_out": sorted(TEST_SUBJECTS),
            "proper_cross_subject_accuracy": float(proper_acc),
            "improper_random_split_accuracy": float(random_acc),
            "accuracy_inflation_pp": float(100 * (random_acc - proper_acc)),
        },
        "split_70_20_10": {
            "train_subjects": sorted(train_ids),
            "test_subjects": sorted(test_ids),
            "val_subjects": sorted(val_ids),
            "n_train": int(len(y_tr)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_te)),
            "best_hyperparameters": selection["best_params"],
            "validation_accuracy": selection["best_val_accuracy"],
            "grid_search_trace": selection["grid_search_trace"],
            "test_accuracy": float(final_acc),
        },
        "mean_uncertainty_correct": mean_unc_correct,
        "mean_uncertainty_incorrect": mean_unc_incorrect,
        "expected_calibration_error": ece,
        "mean_confidence": mean_confidence,
        "calibration_direction": calibration_direction,
        "n_classes": int(len(set(y))),
        "activity_names_present": sorted(ACTIVITY_NAMES.get(int(a), str(a)) for a in set(y)),
    }
    with open(os.path.join(RESULTS_DIR, "pamap2_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    samples_df = np.column_stack([uncertainty, correct])
    np.savetxt(
        os.path.join(RESULTS_DIR, "real_uncertainty_samples_pamap2.csv"),
        samples_df,
        header="uncertainty,correct",
        delimiter=",",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(uncertainty[correct == 1], bins=30, alpha=0.6, label="Correct predictions", color="#2e9e5b")
    if (correct == 0).any():
        ax.hist(uncertainty[correct == 0], bins=30, alpha=0.6, label="Incorrect predictions", color="#e05a45")
    ax.set_xlabel("Predictive uncertainty (normalised entropy)")
    ax.set_ylabel("Count")
    ax.set_title("Real classifier uncertainty on held-out subjects (PAMAP2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "pamap2_uncertainty_distribution.png"), dpi=150)
    plt.close(fig)

    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_acc = [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_acc.append(correct[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.bar(bin_centers, bin_acc, width=0.08, alpha=0.75, color="#5b8dd6", label="Observed accuracy")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"PAMAP2 reliability diagram (ECE = {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "pamap2_reliability_diagram.png"), dpi=150)
    plt.close(fig)

    print(f"\nReport and plots written to: {RESULTS_DIR}")
    return report


if __name__ == "__main__":
    main()
