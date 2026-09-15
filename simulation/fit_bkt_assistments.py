"""
Phase 3: fit the real Bayesian Knowledge Tracing model to ASSISTments
2009-2010 data (dataset #3) and compare the fitted parameters against the
hand-picked synthetic ranges Phase 1 has been using
(src/mastery.py::BKTLearner defaults: p_init 0.08-0.18, p_learn 0.10-0.14,
p_slip 0.08-0.12, p_guess 0.18-0.22).

This directly tests RQ1's assumption that BKT-estimated mastery is a
meaningful, learnable quantity -- previously only checked on synthetic data.
ASSISTments is a different domain (academic/cognitive skills, not
psychomotor motor skills), so treat this as validating the BKT *machinery*,
not as literal parameter values for network-cabling skills.

Writes:
  results/bkt_fitted_params_per_skill.csv  -- one row per fitted skill
  results/bkt_real_fit_report.json         -- aggregate summary + comparison
  results/bkt_param_distributions.png      -- 4-panel histogram vs synthetic ranges

Usage:
    python fit_bkt_assistments.py
"""

from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.assistments_data import load_and_clean_assistments, build_skill_sequences
from src.bkt_fit import fit_bkt_em, mean_loglik_per_obs, predictive_accuracy

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

SYNTHETIC_RANGES = {
    "p_init": (0.08, 0.18),
    "p_learn": (0.10, 0.14),
    "p_slip": (0.08, 0.12),
    "p_guess": (0.18, 0.22),
}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading + cleaning ASSISTments 2009-2010...")
    df = load_and_clean_assistments(DATA_DIR)
    print("Building per-skill, 70/20/10 student-disjoint train/test/validation sequences...")
    sequences = build_skill_sequences(df)
    print(f"Fitting BKT (EM, with validation-based early stopping) for {len(sequences)} skills...")

    rows = []
    for skill, splits in sequences.items():
        train_seqs, val_seqs, test_seqs = splits["train"], splits["val"], splits["test"]
        if len(train_seqs) == 0 or len(val_seqs) == 0 or len(test_seqs) == 0:
            continue
        try:
            fit = fit_bkt_em(train_seqs, n_iter=15, val_sequences=val_seqs)
        except ValueError:
            continue

        p = (fit["p_init"], fit["p_learn"], fit["p_slip"], fit["p_guess"])
        test_ll = mean_loglik_per_obs(test_seqs, *p)
        test_acc = predictive_accuracy(test_seqs, *p)
        majority_baseline = max(
            np.mean([np.mean(s) for s in test_seqs]),
            1 - np.mean([np.mean(s) for s in test_seqs]),
        )

        rows.append(
            {
                "skill_name": skill,
                "p_init": fit["p_init"],
                "p_learn": fit["p_learn"],
                "p_slip": fit["p_slip"],
                "p_guess": fit["p_guess"],
                "n_train_students": len(train_seqs),
                "n_val_students": len(val_seqs),
                "n_test_students": len(test_seqs),
                "n_observations": fit["n_observations"],
                "best_em_iteration": fit["best_iter"],
                "val_loglik_per_obs": fit["best_val_loglik_per_obs"],
                "test_loglik_per_obs": test_ll,
                "test_predictive_accuracy": test_acc,
                "test_majority_class_baseline": float(majority_baseline),
            }
        )
        print(
            f"  {skill!r}: p_init={fit['p_init']:.3f} p_learn={fit['p_learn']:.3f} "
            f"p_slip={fit['p_slip']:.3f} p_guess={fit['p_guess']:.3f} "
            f"(train={len(train_seqs)}/val={len(val_seqs)}/test={len(test_seqs)} students, "
            f"best_iter={fit['best_iter']}, test_acc={test_acc:.3f} vs majority={majority_baseline:.3f})"
        )

    fitted_df = pd.DataFrame(rows)
    fitted_df.to_csv(os.path.join(RESULTS_DIR, "bkt_fitted_params_per_skill.csv"), index=False)

    summary = {}
    for param in ["p_init", "p_learn", "p_slip", "p_guess"]:
        vals = fitted_df[param].to_numpy()
        summary[param] = {
            "median": float(np.median(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "synthetic_range_used_in_phase1": SYNTHETIC_RANGES[param],
        }

    generalization = {
        "mean_test_predictive_accuracy": float(fitted_df["test_predictive_accuracy"].mean()),
        "mean_test_majority_class_baseline": float(fitted_df["test_majority_class_baseline"].mean()),
        "mean_test_loglik_per_obs": float(fitted_df["test_loglik_per_obs"].mean()),
        "n_skills_where_model_beats_majority_baseline": int(
            (fitted_df["test_predictive_accuracy"] > fitted_df["test_majority_class_baseline"]).sum()
        ),
    }

    report = {
        "n_skills_fitted": len(fitted_df),
        "min_users_per_skill": 100,
        "split": "70% train / 20% test / 10% validation, by student, per skill",
        "param_summary": summary,
        "held_out_test_generalization": generalization,
    }
    with open(os.path.join(RESULTS_DIR, "bkt_real_fit_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Real BKT parameters vs Phase 1's synthetic ranges ===")
    print(json.dumps(summary, indent=2))
    print("\n=== Held-out TEST set generalization (70/20/10 split) ===")
    print(json.dumps(generalization, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, param in zip(axes.ravel(), ["p_init", "p_learn", "p_slip", "p_guess"]):
        vals = fitted_df[param].to_numpy()
        ax.hist(vals, bins=15, color="#5b8dd6", alpha=0.8)
        lo, hi = SYNTHETIC_RANGES[param]
        ax.axvspan(lo, hi, color="#e05a45", alpha=0.25, label="Phase 1 synthetic range")
        ax.axvline(np.median(vals), color="black", linestyle="--", label=f"real median={np.median(vals):.3f}")
        ax.set_title(param)
        ax.legend(fontsize=8)
    fig.suptitle("Real BKT parameters fitted per-skill (ASSISTments) vs Phase 1's synthetic ranges")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "bkt_param_distributions.png"), dpi=150)
    plt.close(fig)

    print(f"\nReport and plots written to: {RESULTS_DIR}")
    return report


if __name__ == "__main__":
    main()
