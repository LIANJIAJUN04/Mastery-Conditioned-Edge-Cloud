"""
Model-selection policies for the edge-cloud routing simulation.

Implements the four baselines the proposal requires (Section 9.6):
  B1 cloud-only
  B2 edge-only fixed model
  B3 uncertainty + network routing   (DeepEdgeFM-equivalent)
  B4 mastery-conditioned routing     (the proposal's core contribution)

Each policy receives the same per-trial "context" (mastery, uncertainty,
engagement, network condition, whether the step is safety-critical) and must
return one of the model classes: "T", "S", "M", "L" (Section 9.1).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

MODEL_CLASSES = ["T", "S", "M", "L"]

# Indicative per-class profiles (Section 8/9.1). Latency is in milliseconds for
# an *unloaded* network; actual latency also depends on the sampled network
# condition for classes that must leave the personal edge tier (S/M/L here are
# treated as needing lab-edge or cloud reach, T is fully local).
BASE_LATENCY_MS = {"T": 40, "S": 90, "M": 220, "L": 600}
BASE_ENERGY_J = {"T": 0.05, "S": 0.15, "M": 0.45, "L": 0.9}  # local compute + radio energy
ACCURACY = {"T": 0.72, "S": 0.85, "M": 0.93, "L": 0.98}
NEEDS_NETWORK = {"T": False, "S": False, "M": True, "L": True}  # M/L leave the personal edge


@dataclass
class NetworkCondition:
    bandwidth_mbps: float
    rtt_ms: float
    loss_pct: float


def sample_network(rng: np.random.Generator) -> NetworkCondition:
    """Sample a network condition from the shaped-network sweep in Section 13.1."""
    bandwidth = rng.choice([1, 5, 20, 100])
    rtt = rng.uniform(0, 200)
    loss = rng.uniform(0, 5)
    return NetworkCondition(bandwidth_mbps=float(bandwidth), rtt_ms=float(rtt), loss_pct=float(loss))


def expected_latency_ms(model_class: str, net: NetworkCondition) -> float:
    base = BASE_LATENCY_MS[model_class]
    if not NEEDS_NETWORK[model_class]:
        return base
    # crude network penalty: RTT adds directly, low bandwidth adds a queueing penalty
    bandwidth_penalty = 200.0 / max(net.bandwidth_mbps, 1.0)
    loss_penalty = net.loss_pct * 15.0  # retransmits
    return base + net.rtt_ms + bandwidth_penalty + loss_penalty


def latency_budget_ms(feedback_type: str) -> float:
    """Section 7.1 (C4): 300ms for step-completion confirmation, 1.5s for corrective prompts."""
    return 300.0 if feedback_type == "confirmation" else 1500.0


# --------------------------------------------------------------------------- #
# B1 -- cloud-only
# --------------------------------------------------------------------------- #
def policy_cloud_only(ctx: dict) -> str:
    return "L"


# --------------------------------------------------------------------------- #
# B2 -- edge-only, fixed model
# --------------------------------------------------------------------------- #
def policy_edge_only(ctx: dict) -> str:
    return "S"


# --------------------------------------------------------------------------- #
# B3 -- uncertainty + network routing (ignores mastery entirely)
# --------------------------------------------------------------------------- #
def policy_uncertainty_network(ctx: dict) -> str:
    u = ctx["uncertainty"]
    net: NetworkCondition = ctx["network"]
    good_network = net.bandwidth_mbps >= 5 and net.rtt_ms <= 150

    if u < 0.25:
        return "T"
    if u < 0.5:
        return "S"
    if good_network:
        return "M" if u < 0.75 else "L"
    return "S"  # network too poor to escalate even though uncertainty is high


# --------------------------------------------------------------------------- #
# B4 -- mastery-conditioned routing (the proposal's contribution, Section 9.5)
# --------------------------------------------------------------------------- #
def mastery_utility_term(mastery: float) -> float:
    """Sharply peaked inverted-U centred on the acquisition boundary (mastery = 0.5).

    Squaring 4*m*(1-m) keeps it 0 at mastery=0/1 and 1 at mastery=0.5, but pulls
    it down faster away from the boundary -- a near-novice or near-expert trial
    should get almost none of the "diagnostic escalation" bonus below.
    """
    linear = 4.0 * mastery * (1.0 - mastery)
    return linear ** 3


# How much *extra* diagnostic value each class can unlock at the acquisition
# boundary, on top of its baseline verification value. T is purely local and
# never carries diagnostic value; L (cloud FM) carries the most.
ESCALATION_BONUS = {"T": 0.0, "S": 0.1, "M": 0.45, "L": 0.8}
BASE_VALUE_WEIGHT = 0.5
LAMBDA_COST = 0.20  # trade-off weight between educational utility and system cost

# Real uncertainty as a *weighted* utility input, not a gate (Discussion
# Notes Phase 4, Q2, option 2). Gated by the same boundary_value as
# escalation_value on purpose: the earlier hard "escalate if uncertainty is
# high" rule (tried, reverted) fired uniformly across mastery levels -- a BKT
# learner can "slip" at any mastery, not just near the boundary -- which
# swamped E2's inverted-U. Weight of 1.0 is un-tuned (equal footing with
# escalation_value, not a separately-fitted knob); a sweep confirmed the
# inverted-U survives up to ~1.5 and breaks at 2.0+, so this sits with margin.
UNCERTAINTY_BONUS = {"T": 0.0, "S": 0.1, "M": 0.45, "L": 0.8}
UNCERTAINTY_WEIGHT = 1.0


def cost(model_class: str, net: NetworkCondition) -> float:
    lat = expected_latency_ms(model_class, net)
    energy = BASE_ENERGY_J[model_class]
    # normalise roughly to [0,1]-ish so lambda is meaningful
    return (lat / 1000.0) * 0.5 + energy * 0.5


def policy_mastery_conditioned(ctx: dict) -> str:
    mastery = ctx["mastery"]
    engagement = ctx["engagement"]
    is_critical = ctx["is_critical"]
    net: NetworkCondition = ctx["network"]
    feedback_type = ctx["feedback_type"]
    uncertainty = ctx["uncertainty"]

    budget = latency_budget_ms(feedback_type)
    boundary_value = mastery_utility_term(mastery)
    engagement_factor = 0.3 + 0.7 * engagement

    best_class, best_score = None, -np.inf
    for cls in MODEL_CLASSES:
        # --- Gate (ii): latency gate ---
        if expected_latency_ms(cls, net) > budget:
            continue
        # --- Gate (iii): safety gate -- critical steps need at least S-class ---
        if is_critical and cls == "T":
            continue

        # Baseline verification value: every class can confirm a step reasonably
        # well, so this term does NOT depend on model size -- otherwise a bigger
        # model would always look slightly better even far from the boundary,
        # which contradicts the theory (Section 6.4): an autonomous performer
        # needs only verification, not extra accuracy. Only the escalation term
        # below (diagnostic value near the boundary) should be able to justify
        # paying the extra cost of a bigger class.
        base_value = BASE_VALUE_WEIGHT * engagement_factor
        escalation_value = ESCALATION_BONUS[cls] * boundary_value * engagement
        uncertainty_value = UNCERTAINTY_WEIGHT * UNCERTAINTY_BONUS[cls] * boundary_value * uncertainty
        educational_value = base_value + escalation_value + uncertainty_value

        score = educational_value - LAMBDA_COST * cost(cls, net)

        if score > best_score:
            best_score, best_class = score, cls

    return best_class if best_class is not None else "S"


POLICIES = {
    "B1_cloud_only": policy_cloud_only,
    "B2_edge_only": policy_edge_only,
    "B3_uncertainty_network": policy_uncertainty_network,
    "B4_mastery_conditioned": policy_mastery_conditioned,
}
