"""Controlled finite-answer testbed from the proposal's p,s,Q,nu model."""

from dataclasses import dataclass
import numpy as np

from .pooling import canonical_pool


@dataclass(frozen=True)
class OrbitRegime:
    name: str
    truth_probability: tuple[float, ...]
    shortcut_probability: tuple[float, ...]
    shortcut_destinations: tuple[int, ...]


REGIMES = (
    OrbitRegime("dispersed_shortcut", (0.36, 0.36, 0.36), (0.50, 0.50, 0.50), (1, 2, 3)),
    OrbitRegime("stable_shortcut", (0.36, 0.36, 0.36), (0.50, 0.50, 0.50), (1, 1, 1)),
    OrbitRegime("coherent_wrong", (0.20, 0.20, 0.20), (0.70, 0.70, 0.70), (1, 1, 1)),
)


def sample_orbit(regime: OrbitRegime, *, k=32, seed=0, n_answers=4):
    rng = np.random.default_rng(seed)
    truth = 0
    views = []
    maps = []
    for i, (p, s, dest) in enumerate(zip(regime.truth_probability, regime.shortcut_probability,
                                         regime.shortcut_destinations)):
        probs = np.zeros(n_answers)
        probs[truth] = p
        probs[dest] += s
        residual = max(0.0, 1.0 - probs.sum())
        for a in range(1, n_answers):
            if a != dest:
                probs[a] += residual / max(1, n_answers - 2)
        probs /= probs.sum()
        views.append(rng.choice(n_answers, size=k, p=probs).tolist())
        maps.append(lambda x, i=i: x)
    return views, maps, truth


def evaluate(regime, *, orbits=200, k=32, alpha=1.0, seed=0):
    root_correct = mapped_correct = 0
    for j in range(orbits):
        views, maps, truth = sample_orbit(regime, k=k, seed=seed + j)
        root = max(set(views[0]), key=views[0].count)
        q = canonical_pool(views, maps, support=range(4), alpha=alpha)
        root_correct += root == truth
        mapped_correct += max(q, key=q.get) == truth
    return {"regime": regime.name, "orbits": orbits, "root_accuracy": root_correct / orbits,
            "mapped_accuracy": mapped_correct / orbits,
            "delta": (mapped_correct - root_correct) / orbits}
