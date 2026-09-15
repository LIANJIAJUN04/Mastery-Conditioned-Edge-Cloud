"""
Phase 4: ROSMA -- first test of RQ1 (can mastery be estimated from real
sensor data, in a psychomotor-skill domain) on real data. See
src/rosma_data.py for the full dataset/label/feature writeup and the
CLAUDE.md "Next steps / Phase 4" section for why ROSMA (JIGSAWS was
rejected; ROSMA is the documented backup).

Three checks, in order of how directly they speak to RQ1:

  1. (primary) Cross-USER binary classifier: predict "early" (rep 1-2) vs
     "late" (rep 5-6) attempt from whole-trial kinematic features alone,
     evaluated on 2-3 held-out users the model never saw. If this beats
     chance, real sensor data DOES carry a recoverable cross-person
     mastery signal in this domain -- the actual empirical gap RQ1 flags.
  2. Monotonic-trend check: on the held-out test users' FULL 6-repetition
     sequences (including the middle reps 3-4, never used in training),
     does the classifier's predicted P(late) rise monotonically with
     repetition number? A real progressive mastery signal should show a
     trend across all 6 reps, not just separate the two label endpoints.
  3. Section 13.2 replicate (does the split-leakage finding generalise to
     a 4th, very different dataset): proper user-disjoint split vs
     improper random split accuracy.

Usage:
    python train_rosma_classifier.py
"""

from __future__ import annotations
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from src.rosma_data import load_rosma, feature_label_arrays, TASKS
from src.data_splits import split_groups
from src.model_selection import fit_with_validation
from train_activity_classifier import entropy_uncertainty, expected_calibration_error

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading + feature-extracting ROSMA trials (this can take a couple of minutes)...")
    df = load_rosma(DATA_DIR)
    print(f"Loaded {len(df)} trials across {df['user_id'].nunique()} users, "
          f"{df['task'].nunique()} tasks.")

    X, y, users, labelled = feature_label_arrays(df, labelled_only=True)
    print(f"Labelled (early/late) subset: {len(y)} trials "
          f"({int(y.sum())} late, {len(y) - int(y.sum())} early).")

    # --- Cross-user 70/20/10 split (group = user_id) ---
    unique_users = sorted(set(users))
    train_ids, test_ids, val_ids = split_groups(unique_users, 0.7, 0.2, 0.1, seed=42)
    print(f"\n=== User-disjoint split: {sorted(train_ids)} train / "
          f"{sorted(test_ids)} test / {sorted(val_ids)} validation ===")

    def _subset(ids):
        mask = np.isin(users, list(ids))
        return X[mask], y[mask], mask

    X_tr, y_tr, _ = _subset(train_ids)
    X_val, y_val, _ = _subset(val_ids)
    X_te, y_te, te_mask = _subset(test_ids)

    selection = fit_with_validation(X_tr, y_tr, X_val, y_val)
    clf = selection["best_clf"]
    print(f"Best hyperparameters (chosen on validation users): {selection['best_params']} "
          f"(validation accuracy {selection['best_val_accuracy']:.4f})")

    pred = clf.predict(X_te)
    proba = clf.predict_proba(X_te)
    test_acc = accuracy_score(y_te, pred)
    print(f"\n=== Check 1: held-out-user early-vs-late accuracy: {test_acc:.4f} "
          f"(n_test={len(y_te)}, chance=0.50) ===")

    confidences = proba.max(axis=1)
    uncertainty = entropy_uncertainty(proba)
    correct = (pred == y_te).astype(int)
    ece = expected_calibration_error(confidences, correct, n_bins=10)
    mean_confidence = float(confidences.mean())
    calibration_direction = "overconfident" if mean_confidence > test_acc else "underconfident"
    print(f"Mean confidence {mean_confidence:.4f} vs accuracy {test_acc:.4f} -> {calibration_direction}, ECE={ece:.4f}")

    # Per-task breakdown on the same held-out users.
    per_task = {}
    test_tasks = labelled.loc[te_mask, "task"].to_numpy()
    for task in TASKS:
        tmask = test_tasks == task
        if tmask.sum() == 0:
            continue
        per_task[task] = {
            "n": int(tmask.sum()),
            "accuracy": float(accuracy_score(y_te[tmask], pred[tmask])),
        }
    print("Per-task accuracy on held-out users:", per_task)

    # --- Check 2: monotonic-trend across all 6 reps on held-out users ---
    X_all, rep_all, users_all, all_df = feature_label_arrays(df, labelled_only=False)
    te_all_mask = np.isin(users_all, list(test_ids))
    X_te_all = X_all[te_all_mask]
    rep_te_all = rep_all[te_all_mask]
    proba_late_all = clf.predict_proba(X_te_all)[:, 1]

    rho, pval = spearmanr(rep_te_all, proba_late_all)
    print(f"\n=== Check 2: Spearman correlation between repetition number (1-6) and "
          f"predicted P(late) on held-out users: rho={rho:.4f}, p={pval:.4g} ===")

    mean_proba_by_rep = {
        int(r): float(proba_late_all[rep_te_all == r].mean())
        for r in sorted(set(rep_te_all))
    }
    print("Mean predicted P(late) by repetition:", mean_proba_by_rep)

    # --- Check 3: Section 13.2 replicate -- proper vs improper split ---
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf_random = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_random.fit(Xr_train, yr_train)
    random_acc = accuracy_score(yr_test, clf_random.predict(Xr_test))
    print(f"\n=== Check 3: proper user-disjoint accuracy {test_acc:.4f} vs "
          f"improper random-split accuracy {random_acc:.4f} "
          f"(inflation {100*(random_acc - test_acc):.2f}pp) ===")

    # --- Check 4 (exploratory follow-up): per-(user,task) baseline-normalized
    # features. Check 1's raw whole-trial kinematics mix person-level baseline
    # differences (body size, motion style, overall skill) with the
    # within-person practice-related change we actually want the model to
    # learn; for a genuinely new held-out user, "which person is this" is not
    # a transferable cue, and check 1 came in at/below chance. This z-scores
    # every feature within its own (user, task) group before re-running the
    # identical cross-user classifier, isolating within-person practice
    # variation from between-person baseline variation, to test whether that
    # confound (rather than "no mastery signal exists") explains the null
    # result.
    meta_cols = {"user_id", "task", "repetition", "time_s", "errors", "score", "mastery_label"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    norm_df = df.copy()
    grp = norm_df.groupby(["user_id", "task"])[feature_cols]
    grp_mean = grp.transform("mean")
    grp_std = grp.transform("std").replace(0, 1.0).fillna(1.0)
    norm_df[feature_cols] = (norm_df[feature_cols] - grp_mean) / grp_std

    Xn, yn, usersn, _ = feature_label_arrays(norm_df, labelled_only=True)

    def _subset_n(ids):
        mask = np.isin(usersn, list(ids))
        return Xn[mask], yn[mask]

    Xn_tr, yn_tr = _subset_n(train_ids)
    Xn_val, yn_val = _subset_n(val_ids)
    Xn_te, yn_te = _subset_n(test_ids)

    selection_n = fit_with_validation(Xn_tr, yn_tr, Xn_val, yn_val)
    clf_n = selection_n["best_clf"]
    pred_n = clf_n.predict(Xn_te)
    test_acc_n = accuracy_score(yn_te, pred_n)
    print(f"\n=== Check 4 (exploratory): per-user-normalized held-out-user accuracy: "
          f"{test_acc_n:.4f} (raw Check 1 was {test_acc:.4f}, chance=0.50) ===")

    report = {
        "dataset": "ROSMA",
        "n_trials_loaded": int(len(df)),
        "n_users": int(df["user_id"].nunique()),
        "n_labelled_trials": int(len(y)),
        "split": {
            "train_users": sorted(train_ids),
            "test_users": sorted(test_ids),
            "val_users": sorted(val_ids),
            "best_hyperparameters": selection["best_params"],
            "validation_accuracy": selection["best_val_accuracy"],
        },
        "check1_early_vs_late_test_accuracy": float(test_acc),
        "check1_per_task_accuracy": per_task,
        "check1_mean_confidence": mean_confidence,
        "check1_calibration_direction": calibration_direction,
        "check1_expected_calibration_error": ece,
        "check2_spearman_rho_repetition_vs_p_late": float(rho),
        "check2_spearman_pvalue": float(pval),
        "check2_mean_p_late_by_repetition": mean_proba_by_rep,
        "check3_proper_split_accuracy": float(test_acc),
        "check3_improper_random_split_accuracy": float(random_acc),
        "check3_accuracy_inflation_pp": float(100 * (random_acc - test_acc)),
        "check4_per_user_normalized_test_accuracy": float(test_acc_n),
        "check4_hyperparameters": selection_n["best_params"],
        "check4_validation_accuracy": selection_n["best_val_accuracy"],
    }
    with open(os.path.join(RESULTS_DIR, "rosma_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Plot: mean predicted P(late) by repetition, held-out users only.
    reps_sorted = sorted(mean_proba_by_rep.keys())
    fig, ax = plt.subplots(figsize=(7, 5))
    box_data = [proba_late_all[rep_te_all == r] for r in reps_sorted]
    ax.boxplot(box_data, positions=reps_sorted, widths=0.5)
    ax.plot(reps_sorted, [mean_proba_by_rep[r] for r in reps_sorted], color="#5b8dd6", marker="o", label="mean")
    ax.axhline(0.5, linestyle="--", color="gray", label="chance")
    ax.set_xlabel("Repetition number within (user, task)")
    ax.set_ylabel("Predicted P(late / higher-mastery attempt)")
    ax.set_title(f"ROSMA: predicted mastery signal vs repetition (held-out users, rho={rho:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "rosma_mastery_trend.png"), dpi=150)
    plt.close(fig)

    # Reliability diagram (same style as HAR/PAMAP2).
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
    ax.set_title(f"ROSMA reliability diagram (ECE = {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "rosma_reliability_diagram.png"), dpi=150)
    plt.close(fig)

    print(f"\nReport and plots written to: {RESULTS_DIR}")
    return report


if __name__ == "__main__":
    main()
