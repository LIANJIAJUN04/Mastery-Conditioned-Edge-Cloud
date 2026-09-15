"""Generates the three charts that visually support the proposal's argument."""

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

from .metrics import offload_rate_by_trial, summary_table

POLICY_LABELS = {
    "B1_cloud_only": "B1: Cloud-only",
    "B2_edge_only": "B2: Edge-only (fixed)",
    "B3_uncertainty_network": "B3: Uncertainty + network",
    "B4_mastery_conditioned": "B4: Mastery-conditioned (proposed)",
}
POLICY_COLORS = {
    "B1_cloud_only": "#999999",
    "B2_edge_only": "#5b8dd6",
    "B3_uncertainty_network": "#e08e45",
    "B4_mastery_conditioned": "#2e9e5b",
}


def plot_offload_rate_vs_trial(df: pd.DataFrame, out_path: str) -> None:
    rates = offload_rate_by_trial(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy, group in rates.groupby("policy"):
        ax.plot(
            group["trial_idx"],
            group["offload_rate"],
            label=POLICY_LABELS.get(policy, policy),
            color=POLICY_COLORS.get(policy),
            linewidth=2,
        )
    ax.set_xlabel("Trial index (proxy for session / acquisition progress)")
    ax.set_ylabel("Offload rate (fraction routed to M/L class)")
    ax.set_title("Offload rate across the acquisition curve, by routing policy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mastery_trajectories(df: pd.DataFrame, out_path: str, n_sample_learners: int = 6) -> None:
    sample_ids = df["learner_id"].unique()[:n_sample_learners]
    subset = df[(df["policy"] == "B4_mastery_conditioned") & (df["learner_id"].isin(sample_ids))]

    fig, ax = plt.subplots(figsize=(8, 5))
    for (learner_id, step_id), group in subset.groupby(["learner_id", "step_id"]):
        ax.plot(group["trial_idx"], group["mastery"], alpha=0.6, linewidth=1.2)
    ax.set_xlabel("Trial index")
    ax.set_ylabel("Estimated mastery P(know)")
    ax.set_title(f"Sample BKT mastery trajectories ({n_sample_learners} learners x steps)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_latency_energy_bars(df: pd.DataFrame, out_path: str) -> None:
    summary = summary_table(df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    colors = [POLICY_COLORS.get(p) for p in summary["policy"]]
    labels = [POLICY_LABELS.get(p, p) for p in summary["policy"]]

    axes[0].bar(labels, summary["mean_latency_ms"], color=colors)
    axes[0].set_ylabel("Mean latency (ms)")
    axes[0].set_title("Mean end-to-end latency")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(labels, summary["mean_energy_j"], color=colors)
    axes[1].set_ylabel("Mean energy (J)")
    axes[1].set_title("Mean energy per inference")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_all_plots(df: pd.DataFrame, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    plot_offload_rate_vs_trial(df, os.path.join(results_dir, "offload_rate_vs_trial.png"))
    plot_mastery_trajectories(df, os.path.join(results_dir, "mastery_trajectories.png"))
    plot_latency_energy_bars(df, os.path.join(results_dir, "latency_energy_comparison.png"))
