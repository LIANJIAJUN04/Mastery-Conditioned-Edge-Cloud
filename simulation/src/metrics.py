"""
Analysis functions that turn the raw simulation log into the proposal's
falsifiable predictions (Section 12):

  E1 - B4 reduces cloud-offload rate vs B3 at ~equal accuracy
  E2 - B4's offload rate follows an inverted-U across trials, then decays
       (this is *the* signature prediction -- B3 cannot produce this shape)
  E3 - B4 reduces latency and energy vs B3
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per policy: mean offload rate, latency, energy, accuracy."""
    return (
        df.groupby("policy")
        .agg(
            offload_rate=("offloaded", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
            mean_energy_j=("energy_j", "mean"),
            mean_accuracy=("accuracy", "mean"),
        )
        .reset_index()
    )


def offload_rate_by_mastery_bin(df: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    """Offload rate binned directly by mastery level (cleaner mechanistic check
    than trial index, since learners progress through mastery at different rates
    -- this isolates the mastery -> offload relationship from population timing
    variance)."""
    out = df.copy()
    out["mastery_bin"] = pd.cut(out["mastery"], bins=n_bins, labels=False)
    binned = (
        out.groupby(["policy", "mastery_bin"])
        .agg(offload_rate=("offloaded", "mean"), mastery=("mastery", "mean"))
        .reset_index()
    )
    return binned


def offload_rate_by_trial(df: pd.DataFrame) -> pd.DataFrame:
    """Offload rate averaged over learners/steps, per trial index, per policy."""
    return (
        df.groupby(["policy", "trial_idx"])["offloaded"]
        .mean()
        .reset_index()
        .rename(columns={"offloaded": "offload_rate"})
    )


def fit_quadratic_trend(trial_idx: np.ndarray, offload_rate: np.ndarray) -> dict:
    """Fit offload_rate ~ a*trial_idx^2 + b*trial_idx + c.

    a < 0 with a peak inside the observed range is the numerical signature of
    the predicted inverted-U (E2). a >= 0, or a peak outside the range,
    means the curve is (at best) monotonic -- i.e. it looks like B3, not B4.
    """
    coeffs = np.polyfit(trial_idx, offload_rate, deg=2)
    a, b, c = coeffs
    vertex_x = -b / (2 * a) if abs(a) > 1e-12 else None
    is_inverted_u = (
        a < 0
        and vertex_x is not None
        and trial_idx.min() < vertex_x < trial_idx.max()
    )
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "vertex_trial_idx": float(vertex_x) if vertex_x is not None else None,
        "is_inverted_u": bool(is_inverted_u),
    }


def e1_offload_reduction(summary: pd.DataFrame) -> dict:
    b3 = summary.loc[summary.policy == "B3_uncertainty_network"].iloc[0]
    b4 = summary.loc[summary.policy == "B4_mastery_conditioned"].iloc[0]
    reduction_pct = 100 * (b3.offload_rate - b4.offload_rate) / b3.offload_rate
    accuracy_drop_pp = 100 * (b3.mean_accuracy - b4.mean_accuracy)
    return {
        "b3_offload_rate": float(b3.offload_rate),
        "b4_offload_rate": float(b4.offload_rate),
        "offload_reduction_pct": float(reduction_pct),
        "accuracy_drop_pp": float(accuracy_drop_pp),
        "meets_E1_target(35-60%, <=2pp drop)": bool(
            35 <= reduction_pct <= 60 and abs(accuracy_drop_pp) <= 2
        ),
    }


def e3_latency_energy_reduction(summary: pd.DataFrame) -> dict:
    b3 = summary.loc[summary.policy == "B3_uncertainty_network"].iloc[0]
    b4 = summary.loc[summary.policy == "B4_mastery_conditioned"].iloc[0]
    lat_reduction = 100 * (b3.mean_latency_ms - b4.mean_latency_ms) / b3.mean_latency_ms
    energy_reduction = 100 * (b3.mean_energy_j - b4.mean_energy_j) / b3.mean_energy_j
    return {
        "latency_reduction_pct": float(lat_reduction),
        "energy_reduction_pct": float(energy_reduction),
        "meets_E3_target(>=30% latency, >=25% energy)": bool(
            lat_reduction >= 30 and energy_reduction >= 25
        ),
    }
