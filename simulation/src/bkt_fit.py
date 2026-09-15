"""
Standard Baum-Welch (EM) fitting of the 4-parameter Bayesian Knowledge Tracing
model (Corbett & Anderson, 1994) to real per-skill attempt sequences.

This is what makes dataset #3 (ASSISTments) actually "train" part of the
simulation, rather than just being descriptive: `src/mastery.py::BKTLearner`
already *simulates* a learner with hand-picked p_init/p_learn/p_slip/p_guess
ranges (0.08-0.18 / 0.10-0.14 / 0.08-0.12 / 0.18-0.22). Fitting these same
four parameters to real student data lets Phase 1's simulation be re-run with
empirically-grounded ranges instead of hand-picked ones (see
`real_bkt_param_ranges()` in mastery.py and its use in simulate.py).

Model (no forgetting, the standard BKT assumption):
    states:      0 = not-yet-known, 1 = known
    init:        P(state=1 at first opportunity)      = p_init
    transition:  P(0 -> 1 between opportunities)       = p_learn
                 P(1 -> 0)                              = 0 (no forgetting)
    emission:    P(correct | state=1)  = 1 - p_slip
                 P(correct | state=0)  = p_guess

Fitting is per-skill (parameters are shared across all learners attempting
that skill, which is the standard BKT assumption and matches how the
ASSISTments `skill_name` column groups attempts).
"""

from __future__ import annotations
import numpy as np

_EPS = 1e-9


def _forward_backward(obs: np.ndarray, p_init: float, p_learn: float, p_slip: float, p_guess: float):
    """obs: 1D array of 0/1 (incorrect/correct) for ONE learner's attempts on
    one skill, in order. Returns (gamma, xi, loglik) where gamma[t, s] =
    P(state_t = s | obs) and xi[t, s, s'] = P(state_t=s, state_{t+1}=s' | obs)."""
    T = len(obs)
    emit = np.zeros((T, 2))
    emit[:, 0] = np.where(obs == 1, p_guess, 1 - p_guess)
    emit[:, 1] = np.where(obs == 1, 1 - p_slip, p_slip)

    trans = np.array([[1 - p_learn, p_learn], [0.0, 1.0]])  # no forgetting
    init = np.array([1 - p_init, p_init])

    alpha = np.zeros((T, 2))
    c = np.zeros(T)  # scaling factors for numerical stability
    alpha[0] = init * emit[0]
    c[0] = alpha[0].sum() + _EPS
    alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ trans) * emit[t]
        c[t] = alpha[t].sum() + _EPS
        alpha[t] /= c[t]

    beta = np.zeros((T, 2))
    beta[T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (trans @ (emit[t + 1] * beta[t + 1])) / c[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True) + _EPS

    xi = np.zeros((T - 1, 2, 2))
    for t in range(T - 1):
        xi[t] = (alpha[t][:, None] * trans * (emit[t + 1] * beta[t + 1])[None, :]) / c[t + 1]
    xi /= (xi.sum(axis=(1, 2), keepdims=True) + _EPS)

    loglik = np.sum(np.log(c))
    return gamma, xi, loglik


def sequence_loglik(obs: np.ndarray, p_init: float, p_learn: float, p_slip: float, p_guess: float) -> float:
    """Log-likelihood of one observed 0/1 sequence under fixed BKT parameters
    (via the forward algorithm's scaling factors) -- used to evaluate a
    fitted model on held-out validation/test sequences, without touching
    the EM update rules."""
    _, _, loglik = _forward_backward(obs, p_init, p_learn, p_slip, p_guess)
    return loglik


def mean_loglik_per_obs(sequences: list[np.ndarray], p_init: float, p_learn: float, p_slip: float, p_guess: float) -> float:
    """Mean per-observation log-likelihood across a list of sequences --
    the metric used for validation-based early stopping and for reporting
    held-out test generalisation (comparable across skills with different
    numbers of sequences/observations)."""
    total_ll = 0.0
    total_n = 0
    for obs in sequences:
        if len(obs) < 2:
            continue
        total_ll += sequence_loglik(obs, p_init, p_learn, p_slip, p_guess)
        total_n += len(obs)
    return total_ll / total_n if total_n > 0 else float("nan")


def predictive_accuracy(sequences: list[np.ndarray], p_init: float, p_learn: float, p_slip: float, p_guess: float) -> float:
    """Standard next-attempt predictive accuracy: at each opportunity, predict
    "correct" if the current filtered P(know)-implied P(correct) >= 0.5,
    BEFORE seeing that opportunity's actual outcome, then Bayes-update on the
    real outcome (same update rule as BKTLearner.attempt(), but driven by
    real observed data instead of simulated draws). Used to report held-out
    test-set generalisation in a way that's easy to sanity-check against
    "always predict the majority class"."""
    n_correct_preds = 0
    n_total = 0
    for obs in sequences:
        if len(obs) < 2:
            continue
        p_know = p_init
        for o in obs:
            p_correct = p_know * (1 - p_slip) + (1 - p_know) * p_guess
            pred = 1 if p_correct >= 0.5 else 0
            n_correct_preds += int(pred == o)
            n_total += 1

            if o == 1:
                numerator, denom = p_know * (1 - p_slip), p_correct
            else:
                numerator, denom = p_know * p_slip, 1 - p_correct
            posterior = numerator / denom if denom > 1e-9 else p_know
            p_know = posterior + (1 - posterior) * p_learn

    return n_correct_preds / n_total if n_total > 0 else float("nan")


def fit_bkt_em(
    sequences: list[np.ndarray],
    n_iter: int = 15,
    init_params: tuple[float, float, float, float] = (0.15, 0.12, 0.10, 0.20),
    val_sequences: list[np.ndarray] | None = None,
) -> dict:
    """Fits (p_init, p_learn, p_slip, p_guess) by EM over a list of 0/1
    attempt sequences (one array per learner) that all share ONE skill.

    If `val_sequences` is given (a held-out, disjoint set of learners on the
    same skill), this does early stopping: after every EM iteration, the
    CURRENT parameters are scored on `val_sequences` by mean per-observation
    log-likelihood, and the returned parameters are those from whichever
    iteration scored best on validation -- not necessarily the last one.
    This is what makes the EM fit an actual train/validation split rather
    than fitting-and-hoping.

    Returns dict with fitted params, final/best log-likelihood, and
    iteration history (for a convergence + overfitting sanity check).
    """
    p_init, p_learn, p_slip, p_guess = init_params
    history = []

    sequences = [s for s in sequences if len(s) >= 2]
    if len(sequences) == 0:
        raise ValueError("Need at least one sequence of length >= 2 to fit BKT.")

    best_val_ll = -np.inf
    best_params = init_params
    best_iter = -1

    for it in range(n_iter):
        num_init = 0.0
        n_seq = len(sequences)
        num_learn_trans = 0.0
        denom_state0 = 0.0
        num_guess_correct = 0.0
        denom_guess = 0.0
        num_slip_incorrect = 0.0
        denom_slip = 0.0
        total_loglik = 0.0

        for obs in sequences:
            gamma, xi, loglik = _forward_backward(obs, p_init, p_learn, p_slip, p_guess)
            total_loglik += loglik

            num_init += gamma[0, 1]

            # p_learn: fraction of "state=0" opportunities that transition to state=1
            num_learn_trans += xi[:, 0, 1].sum()
            denom_state0 += gamma[:-1, 0].sum()

            # p_guess: P(correct | state=0)
            denom_guess += gamma[:, 0].sum()
            num_guess_correct += (gamma[:, 0] * (obs == 1)).sum()

            # p_slip: P(incorrect | state=1)
            denom_slip += gamma[:, 1].sum()
            num_slip_incorrect += (gamma[:, 1] * (obs == 0)).sum()

        p_init = float(np.clip(num_init / n_seq, 1e-3, 1 - 1e-3))
        p_learn = float(np.clip(num_learn_trans / (denom_state0 + _EPS), 1e-3, 1 - 1e-3))
        p_guess = float(np.clip(num_guess_correct / (denom_guess + _EPS), 1e-3, 0.5))
        p_slip = float(np.clip(num_slip_incorrect / (denom_slip + _EPS), 1e-3, 0.5))

        iter_record = {"p_init": p_init, "p_learn": p_learn, "p_slip": p_slip, "p_guess": p_guess, "loglik": total_loglik}

        if val_sequences is not None:
            val_ll = mean_loglik_per_obs(val_sequences, p_init, p_learn, p_slip, p_guess)
            iter_record["val_loglik_per_obs"] = val_ll
            if val_ll > best_val_ll:
                best_val_ll = val_ll
                best_params = (p_init, p_learn, p_slip, p_guess)
                best_iter = it

        history.append(iter_record)

    if val_sequences is not None:
        # Early stopping: use the iteration that scored best on the held-out
        # validation sequences, not necessarily the last EM iteration.
        p_init, p_learn, p_slip, p_guess = best_params

    return {
        "p_init": p_init,
        "p_learn": p_learn,
        "p_slip": p_slip,
        "p_guess": p_guess,
        "n_sequences": len(sequences),
        "n_observations": int(sum(len(s) for s in sequences)),
        "final_loglik": history[-1]["loglik"],
        "best_iter": best_iter if val_sequences is not None else n_iter - 1,
        "best_val_loglik_per_obs": best_val_ll if val_sequences is not None else None,
        "history": history,
    }
