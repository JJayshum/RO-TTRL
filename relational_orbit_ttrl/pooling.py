"""Canonical mapped-view pooling and cross-fitted rewards."""

from collections import Counter
import math
from typing import Sequence


def _log_score(count: int, total: int, alpha: float, support_size: int) -> float:
    return math.log(count + alpha) - math.log(total + alpha * support_size)


def canonical_pool(samples: Sequence[Sequence], inverse_maps: Sequence, *, support=None,
                   alpha: float = 1.0, weights=None, temperature: float = 1.0):
    """Pool node samples after mapping each answer back to root coordinates."""
    if alpha <= 0 or temperature <= 0:
        raise ValueError("alpha and temperature must be positive")
    if len(samples) != len(inverse_maps):
        raise ValueError("one inverse map is required per view")
    canonical = [[inv(a) for a in view] for view, inv in zip(samples, inverse_maps)]
    if support is None:
        support = {None}
        for view in canonical:
            support.update(view)
    support = tuple(support)
    if weights is None:
        weights = [1.0 / len(canonical)] * len(canonical)
    if len(weights) != len(canonical) or any(w < 0 for w in weights):
        raise ValueError("weights must be nonnegative and match views")
    total_weight = sum(weights)
    if not math.isclose(total_weight, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("weights must sum to one")
    scores = {}
    for a in support:
        scores[a] = sum(w * _log_score(Counter(view)[a], len(view), alpha, len(support))
                        for view, w in zip(canonical, weights)) / temperature
    peak = max(scores.values())
    norm = sum(math.exp(v - peak) for v in scores.values())
    return {a: math.exp(v - peak) / norm for a, v in scores.items()}


def leave_one_out_rewards(samples: Sequence[Sequence], inverse_maps: Sequence, *, support,
                          alpha=1.0, weights=None, temperature=1.0):
    """Return stop-gradient q_i(answer) with that rollout removed from its view."""
    rewards = []
    for i, view in enumerate(samples):
        view_rewards = []
        for r in range(len(view)):
            reduced = [list(v) for v in samples]
            reduced[i].pop(r)
            q = canonical_pool(reduced, inverse_maps, support=support, alpha=alpha,
                               weights=weights, temperature=temperature)
            view_rewards.append(q[inverse_maps[i](view[r])])
        rewards.append(view_rewards)
    return rewards
