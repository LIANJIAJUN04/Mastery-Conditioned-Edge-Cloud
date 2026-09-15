"""
Learner mastery simulator based on Bayesian Knowledge Tracing (BKT).

This reproduces the "learner model" described in the proposal (Section 6.4 / 9.3):
each learner has a hidden ("latent") mastery probability for a skill step, which is
updated after every attempt using Bayes' rule, with explicit slip/guess parameters
so a single lucky/unlucky trial cannot swing the estimate too much.
"""

from __future__ import annotations
import json
import os
import numpy as np


class BKTLearner:
    """Simulates one learner's mastery trajectory on one skill step.

    Parameters
    ----------
    p_init : float
        Probability the learner already knows the step before the first attempt.
    p_learn : float
        Probability of transitioning from "not known" to "known" after one attempt.
    p_slip : float
        Probability of an incorrect attempt despite knowing the step.
    p_guess : float
        Probability of a correct attempt despite not knowing the step.
    """

    def __init__(
        self,
        p_init: float = 0.15,
        p_learn: float = 0.12,
        p_slip: float = 0.10,
        p_guess: float = 0.20,
        rng: np.random.Generator | None = None,
    ):
        self.p_know = p_init
        self.p_learn = p_learn
        self.p_slip = p_slip
        self.p_guess = p_guess
        self.rng = rng if rng is not None else np.random.default_rng()

    def attempt(self) -> tuple[bool, float]:
        """Simulate one attempt at the step.

        Returns
        -------
        correct : bool
            Whether the simulated attempt was correct.
        mastery_after : float
            Updated P(know) after observing this attempt.
        """
        p_correct = self.p_know * (1 - self.p_slip) + (1 - self.p_know) * self.p_guess
        correct = self.rng.random() < p_correct

        if correct:
            numerator = self.p_know * (1 - self.p_slip)
            denom = p_correct
        else:
            numerator = self.p_know * self.p_slip
            denom = 1 - p_correct

        posterior_know = numerator / denom if denom > 1e-9 else self.p_know
        # Learning transition: even if not yet known, each attempt gives a chance to learn.
        self.p_know = posterior_know + (1 - posterior_know) * self.p_learn
        self.p_know = float(np.clip(self.p_know, 0.0, 1.0))

        return correct, self.p_know


def sensor_uncertainty(mastery: float, rng: np.random.Generator, noise: float = 0.12) -> float:
    """Proxy for a classifier's predictive uncertainty on this trial's sensor window.

    Modelled as monotonically decreasing in mastery (novices produce high-variance,
    hesitant motion that is genuinely harder to classify; experts produce fluent,
    low-variance motion) plus zero-mean noise, clipped to [0, 1].

    This is what an "uncertainty-only" router (baseline B3) sees.
    """
    base = 1.0 - mastery
    value = base + rng.normal(0.0, noise)
    return float(np.clip(value, 0.0, 1.0))


def empirical_sensor_uncertainty(
    correct: bool,
    rng: np.random.Generator,
    emp_uncertainty_correct: np.ndarray,
    emp_uncertainty_incorrect: np.ndarray,
) -> float:
    """Real-data-grounded alternative to `sensor_uncertainty()`.

    Instead of assuming uncertainty ~ (1 - mastery) + Gaussian noise, this
    draws a real observed uncertainty value from a classifier actually trained
    on wearable IMU data (UCI HAR; see train_activity_classifier.py), matched
    to whether the *simulated* attempt was correct or not.

    This is a deliberate simplification: UCI HAR's "correct/incorrect
    activity classification" is not the same construct as "correct/incorrect
    skill execution", but both share the property that a wearable-IMU-derived
    classifier is less certain when it is wrong -- which is the property this
    simulation actually depends on. Treat this as evidence that the *shape* of
    the assumption is realistic, not as a literal transfer of the numbers.
    """
    pool = emp_uncertainty_correct if correct else emp_uncertainty_incorrect
    return float(rng.choice(pool))


def real_bkt_param_ranges(results_dir: str) -> dict[str, tuple[float, float]] | None:
    """Loads the real, EM-fitted BKT parameter ranges written by
    fit_bkt_assistments.py (dataset #3), if available.

    Uses each parameter's interquartile range (p25-p75) ACROSS SKILLS as the
    sampling range for simulated learners, replacing the hand-picked
    synthetic ranges hard-coded in simulate.py::build_trial_contexts. Returns
    None if fit_bkt_assistments.py hasn't been run yet, so callers can fall
    back to the synthetic ranges.
    """
    path = os.path.join(results_dir, "bkt_real_fit_report.json")
    if not os.path.exists(path):
        return None

    with open(path) as f:
        report = json.load(f)

    ranges = {}
    for param, stats in report["param_summary"].items():
        ranges[param] = (stats["p25"], stats["p75"])
    return ranges


def engagement_latent(rng: np.random.Generator, base: float = 0.85, noise: float = 0.1) -> float:
    """Proxy for the engagement latent e_s (Section 9.3): distinguishes 'cannot'
    from 'did not attempt'. Mostly high, occasionally dips."""
    value = base + rng.normal(0.0, noise)
    return float(np.clip(value, 0.0, 1.0))
