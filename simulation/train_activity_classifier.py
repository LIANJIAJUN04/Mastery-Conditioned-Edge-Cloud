"""
Phase 2a: train a lightweight classifier on REAL wearable IMU data (UCI HAR)
and check two things the proposal's methodology depends on:

  1. (Section 13.2) "Cross-learner splits, never random splits. Random splits
     leak the same learner's trials across train/test and inflate accuracy
     substantially." -- we test this directly by training the SAME model
     twice: once with the dataset's official subject-disjoint split, once
     with an improper random split, and comparing accuracy.

  2. (Section 9.4) Raw softmax/predict_proba confidence is not automatically
     trustworthy as a router signal. We compute real predictive uncertainty
     from a real trained model and check whether it is at least *usable*:
     does it separate correct from incorrect predictions, and how far off is
     its calibration (a crude reliability diagram + Expected Calibration Error)?

Usage:
    python train_activity_classifier.py
"""

from __future__ import annotations
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from src.real_data import load_uci_har, load_uci_har_pooled, ACTIVITY_NAMES
from src.data_splits import split_groups
from src.model_selection import fit_with_validation

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def entropy_uncertainty(proba: np.ndarray) -> np.ndarray:
    """Normalised Shannon entropy of predicted class probabilities, in [0, 1].
    0 = fully confident (one class has prob 1), 1 = maximally uncertain (uniform)."""
    eps = 1e-12
    n_classes = proba.shape[1]
    ent = -np.sum(proba * np.log(proba + eps), axis=1)
    return ent / np.log(n_classes)


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE: bin by predicted confidence, compare average confidence
    to actual accuracy within each bin, weight by bin size."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = load_uci_har(DATA_DIR)

    # --- Experiment A: proper cross-subject split (the dataset's official one) ---
    clf_proper = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_proper.fit(data["X_train"], data["y_train"])
    proper_pred = clf_proper.predict(data["X_test"])
    proper_proba = clf_proper.predict_proba(data["X_test"])
    proper_acc = accuracy_score(data["y_test"], proper_pred)

    # --- Experiment B: improper random split, IGNORING subject identity ---
    # This is exactly the mistake Section 13.2 warns against: the same subject's
    # trials can appear in both "train" and "test", which leaks information
    # (a model partially memorises that subject's gait/posture idiosyncrasies).
    X_all = np.vstack([data["X_train"], data["X_test"]])
    y_all = np.concatenate([data["y_train"], data["y_test"]])
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
    )
    clf_random = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_random.fit(Xr_train, yr_train)
    random_pred = clf_random.predict(Xr_test)
    random_acc = accuracy_score(yr_test, random_pred)

    print("=== Section 13.2 check: cross-subject vs random split ===")
    print(f"Proper cross-subject-split accuracy : {proper_acc:.4f}")
    print(f"Improper random-split accuracy       : {random_acc:.4f}")
    print(f"Inflation from the invalid split      : {100*(random_acc - proper_acc):.2f} percentage points")

    # --- Experiment C: proper 70/20/10 subject-disjoint train/val/test split ---
    # This is the ACTUAL training pipeline for the edge-model surrogate: model
    # selection (hyperparameters) happens on the validation subjects only, the
    # test subjects are touched exactly once, at the very end, for the number
    # that gets reported and for the uncertainty samples fed into simulate.py.
    X_all, y_all, subj_all = load_uci_har_pooled(DATA_DIR)
    train_ids, test_ids, val_ids = split_groups(subj_all, 0.7, 0.2, 0.1, seed=42)

    def _subset(ids):
        mask = np.isin(subj_all, list(ids))
        return X_all[mask], y_all[mask]

    X_tr, y_tr = _subset(train_ids)
    X_val, y_val = _subset(val_ids)
    X_te, y_te = _subset(test_ids)

    print(f"\n=== 70/20/10 subject-disjoint split: {len(train_ids)} train / "
          f"{len(test_ids)} test / {len(val_ids)} validation subjects ===")

    selection = fit_with_validation(X_tr, y_tr, X_val, y_val)
    clf_final = selection["best_clf"]
    print(f"Best hyperparameters (chosen on validation subjects): {selection['best_params']} "
          f"(validation accuracy {selection['best_val_accuracy']:.4f})")

    final_pred = clf_final.predict(X_te)
    final_proba = clf_final.predict_proba(X_te)
    final_acc = accuracy_score(y_te, final_pred)
    print(f"Held-out TEST accuracy (touched once): {final_acc:.4f}")

    # --- Uncertainty analysis on the FINAL model's held-out TEST predictions ---
    confidences = final_proba.max(axis=1)
    uncertainty = entropy_uncertainty(final_proba)
    correct = (final_pred == y_te).astype(int)

    mean_unc_correct = float(uncertainty[correct == 1].mean())
    mean_unc_incorrect = float(uncertainty[correct == 0].mean()) if (correct == 0).any() else None
    ece = expected_calibration_error(confidences, correct, n_bins=10)
    mean_confidence = float(confidences.mean())
    calibration_direction = "overconfident" if mean_confidence > final_acc else "underconfident"

    print("\n=== Section 9.4 check: is real predictive uncertainty usable as a router signal? ===")
    print(f"Mean uncertainty on CORRECT predictions   : {mean_unc_correct:.4f}")
    print(f"Mean uncertainty on INCORRECT predictions : {mean_unc_incorrect}")
    print(f"Expected Calibration Error (10 bins)      : {ece:.4f}")
    print(f"Mean confidence {mean_confidence:.4f} vs accuracy {final_acc:.4f} -> model is {calibration_direction}")

    report = {
        "section_13_2_check": {
            "proper_cross_subject_accuracy": float(proper_acc),
            "improper_random_split_accuracy": float(random_acc),
            "accuracy_inflation_pp": float(100 * (random_acc - proper_acc)),
        },
        "split_70_20_10": {
            "n_train_subjects": len(train_ids),
            "n_test_subjects": len(test_ids),
            "n_val_subjects": len(val_ids),
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
        "n_classes": len(ACTIVITY_NAMES),
    }
    with open(os.path.join(RESULTS_DIR, "real_data_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Save per-sample (uncertainty, correct) pairs so the Phase-1 policy
    # simulation can sample real-data-derived uncertainty instead of an
    # assumed Gaussian noise model (see src/mastery.py::empirical_sensor_uncertainty
    # and src/real_data.py::load_empirical_uncertainty).
    samples_df = np.column_stack([uncertainty, correct])
    np.savetxt(
        os.path.join(RESULTS_DIR, "real_uncertainty_samples.csv"),
        samples_df,
        header="uncertainty,correct",
        delimiter=",",
        comments="",
    )

    # --- Plot 1: uncertainty distribution, correct vs incorrect ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(uncertainty[correct == 1], bins=30, alpha=0.6, label="Correct predictions", color="#2e9e5b")
    if (correct == 0).any():
        ax.hist(uncertainty[correct == 0], bins=30, alpha=0.6, label="Incorrect predictions", color="#e05a45")
    ax.set_xlabel("Predictive uncertainty (normalised entropy)")
    ax.set_ylabel("Count")
    ax.set_title("Real classifier uncertainty on held-out subjects (UCI HAR)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "real_uncertainty_distribution.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: reliability diagram ---
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_acc, bin_count = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_acc.append(correct[mask].mean())
        bin_count.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.bar(bin_centers, bin_acc, width=0.08, alpha=0.75, color="#5b8dd6", label="Observed accuracy")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "reliability_diagram.png"), dpi=150)
    plt.close(fig)

    print(f"\nReport and plots written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
