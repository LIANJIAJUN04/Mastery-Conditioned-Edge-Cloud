"""
Entry point: run the mastery-conditioned edge-cloud routing simulation and
report whether the results support the proposal's falsifiable predictions
(E1, E2, E3 in Section 12 of the proposal).

Usage (from the `simulation/` folder, with the virtual environment active):

    python run_experiment.py

If results/real_uncertainty_samples.csv does not exist yet, this script runs
Phase 2a (train_activity_classifier.py) first to generate it, then continues
with the Phase 1 + 2b simulation - a single command runs the whole pipeline.
"""

from __future__ import annotations
import json
import os

from src.simulate import run_experiment
from src.metrics import (
    summary_table,
    offload_rate_by_trial,
    offload_rate_by_mastery_bin,
    fit_quadratic_trend,
    e1_offload_reduction,
    e3_latency_energy_reduction,
)
from src.plot_results import generate_all_plots

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
HAR_UNCERTAINTY_CSV = os.path.join(RESULTS_DIR, "real_uncertainty_samples.csv")
PAMAP2_UNCERTAINTY_CSV = os.path.join(RESULTS_DIR, "real_uncertainty_samples_pamap2.csv")
BKT_FIT_REPORT = os.path.join(RESULTS_DIR, "bkt_real_fit_report.json")


def main():
    # Dataset #1: UCI HAR -- edge classifier + uncertainty/calibration (Phase 2a).
    if not os.path.exists(HAR_UNCERTAINTY_CSV):
        print("real_uncertainty_samples.csv not found - running Phase 2a (train_activity_classifier.py) first...\n")
        import train_activity_classifier
        train_activity_classifier.main()
        print("\nPhase 2a done.\n")

    # Dataset #2: PAMAP2 -- replicates Phase 2a's two checks on an independent
    # real wearable dataset, and its uncertainty samples get pooled into the
    # empirical distribution alongside HAR's (see src/real_data.py).
    if not os.path.exists(PAMAP2_UNCERTAINTY_CSV):
        print("real_uncertainty_samples_pamap2.csv not found - running Phase 2a-replicate "
              "(train_pamap2_classifier.py) first...\n")
        import train_pamap2_classifier
        train_pamap2_classifier.main()
        print("\nPAMAP2 replicate done.\n")

    # Dataset #3: ASSISTments -- real BKT parameter fit (Phase 3), feeds
    # mastery-transition ranges into build_trial_contexts (see src/mastery.py).
    if not os.path.exists(BKT_FIT_REPORT):
        print("bkt_real_fit_report.json not found - running Phase 3 (fit_bkt_assistments.py) first...\n")
        import fit_bkt_assistments
        fit_bkt_assistments.main()
        print("\nPhase 3 done.\n")

    print("Continuing with Phase 1 + 2b/2c simulation, now grounded in all 3 real datasets...\n")

    n_learners, n_steps, n_trials = 80, 3, 60
    print(f"Running simulation ({n_learners} learners x {n_steps} steps x {n_trials} trials, 4 policies)...")
    df = run_experiment(n_learners=n_learners, n_steps=n_steps, n_trials_per_step=n_trials, seed=42)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "raw_log.csv"), index=False)

    summary = summary_table(df)
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_by_policy.csv"), index=False)
    print("\n=== Summary by policy ===")
    print(summary.to_string(index=False))

    # E2: inverted-U test, computed only on B4's offload-rate-by-trial curve.
    rates = offload_rate_by_trial(df)
    b4_rates = rates[rates.policy == "B4_mastery_conditioned"]
    b3_rates = rates[rates.policy == "B3_uncertainty_network"]

    e2_b4_trial = fit_quadratic_trend(b4_rates["trial_idx"].to_numpy(), b4_rates["offload_rate"].to_numpy())
    e2_b3_trial = fit_quadratic_trend(b3_rates["trial_idx"].to_numpy(), b3_rates["offload_rate"].to_numpy())

    # Mechanistic check: offload rate binned directly by mastery level, which
    # isolates the mastery -> offload relationship from population timing noise
    # (different simulated learners cross the acquisition boundary at different
    # trial indices, which can wash out the hump in the trial-index view alone).
    mastery_bins = offload_rate_by_mastery_bin(df)
    b4_mastery = mastery_bins[mastery_bins.policy == "B4_mastery_conditioned"]
    b3_mastery = mastery_bins[mastery_bins.policy == "B3_uncertainty_network"]
    e2_b4_mastery = fit_quadratic_trend(b4_mastery["mastery"].to_numpy(), b4_mastery["offload_rate"].to_numpy())
    e2_b3_mastery = fit_quadratic_trend(b3_mastery["mastery"].to_numpy(), b3_mastery["offload_rate"].to_numpy())

    e1 = e1_offload_reduction(summary)
    e3 = e3_latency_energy_reduction(summary)

    results = {
        "datasets_used": {
            "1_uci_har": "edge classifier + uncertainty/calibration (Phase 2a)",
            "2_pamap2": "independent replication of Phase 2a's checks + pooled into empirical uncertainty",
            "3_assistments_2009_2010": "real EM-fitted BKT parameters (Phase 3), used in build_trial_contexts",
        },
        "E1_offload_reduction_vs_B3": e1,
        "E2_inverted_U_shape_by_trial_index": {
            "B4_mastery_conditioned": e2_b4_trial,
            "B3_uncertainty_network (should NOT show inverted-U)": e2_b3_trial,
        },
        "E2_inverted_U_shape_by_mastery_level": {
            "B4_mastery_conditioned": e2_b4_mastery,
            "B3_uncertainty_network (should NOT show inverted-U)": e2_b3_mastery,
        },
        "E3_latency_energy_reduction_vs_B3": e3,
    }

    with open(os.path.join(RESULTS_DIR, "falsifiable_predictions_report.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Falsifiable prediction checks (Section 12 of the proposal) ===")
    print(json.dumps(results, indent=2))

    generate_all_plots(df, RESULTS_DIR)
    print(f"\nPlots and CSVs written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
