"""
Trial-based simulation loop.

For N simulated learners, each practising K skill steps across T trials, this
generates one shared sequence of "trial contexts" (mastery, uncertainty,
network condition, ...) and then replays that exact same sequence through each
of the four routing policies. Replaying the identical sequence per policy is
what makes the comparison fair -- differences in the results are due only to
the policy's decisions, not to random seed differences between policies.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

from .mastery import (
    BKTLearner,
    sensor_uncertainty,
    empirical_sensor_uncertainty,
    engagement_latent,
    real_bkt_param_ranges,
)
from .real_data import load_empirical_uncertainty
from .policies import (
    POLICIES,
    sample_network,
    expected_latency_ms,
    BASE_ENERGY_J,
    ACCURACY,
    NEEDS_NETWORK,
)

P_CRITICAL_STEP = 0.15  # fraction of steps flagged safety-critical (Section 9.5 gate iii)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def build_trial_contexts(
    n_learners: int,
    n_steps: int,
    n_trials_per_step: int,
    seed: int = 42,
) -> list[dict]:
    """Pre-generate the full sequence of trial contexts shared across all policies."""
    rng = np.random.default_rng(seed)
    contexts = []

    empirical = load_empirical_uncertainty(RESULTS_DIR)
    if empirical is not None:
        emp_unc_correct, emp_unc_incorrect = empirical
        print(
            f"[simulate] Using REAL uncertainty distribution pooled from wearable-IMU datasets "
            f"({len(emp_unc_correct)} correct / {len(emp_unc_incorrect)} incorrect samples) "
            f"instead of the synthetic Gaussian model."
        )
    else:
        emp_unc_correct = emp_unc_incorrect = None
        print("[simulate] No real_uncertainty_samples.csv found -- run train_activity_classifier.py "
              "first to use real-data-grounded uncertainty. Falling back to the synthetic model.")

    bkt_ranges = real_bkt_param_ranges(RESULTS_DIR)
    if bkt_ranges is not None:
        print(
            f"[simulate] Using REAL BKT parameter ranges fitted on ASSISTments "
            f"(p_init={bkt_ranges['p_init']}, p_learn={bkt_ranges['p_learn']}, "
            f"p_slip={bkt_ranges['p_slip']}, p_guess={bkt_ranges['p_guess']}) "
            f"instead of the hand-picked synthetic ranges."
        )
    else:
        bkt_ranges = {
            "p_init": (0.08, 0.18),
            "p_learn": (0.10, 0.14),
            "p_slip": (0.08, 0.12),
            "p_guess": (0.18, 0.22),
        }
        print("[simulate] No bkt_real_fit_report.json found -- run fit_bkt_assistments.py "
              "first to use real-data-grounded BKT parameters. Falling back to synthetic ranges.")

    for learner_id in range(n_learners):
        for step_id in range(n_steps):
            learner = BKTLearner(
                p_init=rng.uniform(*bkt_ranges["p_init"]),
                p_learn=rng.uniform(*bkt_ranges["p_learn"]),
                p_slip=rng.uniform(*bkt_ranges["p_slip"]),
                p_guess=rng.uniform(*bkt_ranges["p_guess"]),
                rng=rng,
            )
            is_critical = rng.random() < P_CRITICAL_STEP

            for trial_idx in range(n_trials_per_step):
                correct, mastery = learner.attempt()
                if emp_unc_correct is not None:
                    uncertainty = empirical_sensor_uncertainty(correct, rng, emp_unc_correct, emp_unc_incorrect)
                else:
                    uncertainty = sensor_uncertainty(mastery, rng)
                engagement = engagement_latent(rng)
                network = sample_network(rng)
                feedback_type = "confirmation" if correct else "correction"

                contexts.append(
                    {
                        "learner_id": learner_id,
                        "step_id": step_id,
                        "trial_idx": trial_idx,
                        "mastery": mastery,
                        "uncertainty": uncertainty,
                        "engagement": engagement,
                        "network": network,
                        "is_critical": is_critical,
                        "feedback_type": feedback_type,
                        "correct": correct,
                    }
                )
    return contexts


def run_all_policies(contexts: list[dict]) -> pd.DataFrame:
    """Replay the same contexts through every policy and log the outcome of each decision."""
    rows = []
    for policy_name, policy_fn in POLICIES.items():
        for ctx in contexts:
            chosen_class = policy_fn(ctx)
            latency = expected_latency_ms(chosen_class, ctx["network"])
            energy = BASE_ENERGY_J[chosen_class]
            offloaded = NEEDS_NETWORK[chosen_class]  # left the personal-edge tier (M or L)

            rows.append(
                {
                    "policy": policy_name,
                    "learner_id": ctx["learner_id"],
                    "step_id": ctx["step_id"],
                    "trial_idx": ctx["trial_idx"],
                    "mastery": ctx["mastery"],
                    "uncertainty": ctx["uncertainty"],
                    "chosen_class": chosen_class,
                    "offloaded": offloaded,
                    "latency_ms": latency,
                    "energy_j": energy,
                    "accuracy": ACCURACY[chosen_class],
                }
            )
    return pd.DataFrame(rows)


def run_experiment(
    n_learners: int = 30,
    n_steps: int = 3,
    n_trials_per_step: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    contexts = build_trial_contexts(n_learners, n_steps, n_trials_per_step, seed)
    return run_all_policies(contexts)
