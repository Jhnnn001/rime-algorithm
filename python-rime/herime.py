"""HERIME: Li et al. (2025), doi:10.3390/biomimetics10010014."""

from collections.abc import Callable
from math import sqrt

import numpy as np
from numpy.typing import ArrayLike

from _base import BaseRIME


class HERIME(BaseRIME):
    """Sequential RIME with fitness-distance selection and Gaussian EDA fusion."""

    method = "herime"

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: ArrayLike,
        n_agents: int = 30,
        max_iter: int = 500,
        seed: int | np.random.Generator | None = None,
        callback: Callable[[BaseRIME], None] | None = None,
        *,
        lam: float = 0.5,
        n_eda: int | None = None,
    ) -> None:
        """Configure distance weight ``lam`` and EDA batch size ``n_eda``.

        ``lam`` is in [0, 1]. ``n_eda`` defaults to floor(n_agents/2), since
        the paper does not state a value. At least four agents are required.
        """
        super().__init__(objective, bounds, n_agents, max_iter, seed, callback)
        if self.n_agents < 4:
            raise ValueError("HERIME requires n_agents >= 4.")
        self.lam = float(lam)
        if not np.isfinite(self.lam) or not 0 <= self.lam <= 1:
            raise ValueError("lam must be finite and in [0, 1].")
        self.n_eda = self.n_agents // 2 if n_eda is None else n_eda
        if isinstance(self.n_eda, (bool, np.bool_)) or not isinstance(self.n_eda, (int, np.integer)) or not 1 <= self.n_eda <= self.n_agents:
            raise ValueError("n_eda must be an integer in [1, n_agents].")
        self.n_eda = int(self.n_eda)

    def _step(self, t):
        distance = np.linalg.norm(self.positions - self.best_x, axis=1)
        nf = np.zeros(self.n_agents)
        finite = np.isfinite(self.fitness)
        if np.any(finite):
            low, high = np.min(self.fitness[finite]), np.max(self.fitness[finite])
            if high > low:
                nf[finite] = (high - self.fitness[finite]) / (high - low)
        nf[np.isneginf(self.fitness)] = 1
        nd = distance / np.max(distance) if np.max(distance) > 0 else np.zeros(self.n_agents)
        score = (1 - self.lam) * nf + self.lam * nd
        k = self.rng.choice(self.n_agents, p=score / score.sum()) if score.sum() > 0 else self.rng.integers(self.n_agents)
        reference = self.positions[k].copy()

        factor = self._rime_factor(t)
        threshold = self._hard_rime_threshold()
        for i in range(self.n_agents):
            u1, u2, u3 = (self.rng.random(self.dim) for _ in range(3))
            soft = self.best_x + factor * (self._lb + u2 * (self._ub - self._lb))
            candidate = np.where(u1 < sqrt(t / self.max_iter), soft, self.positions[i])
            candidate = self._clip(np.where(2 * u3 - 1 < threshold[i], reference, candidate))
            value = self._evaluate(candidate[None, :])[0]
            if value < self.fitness[i]:
                self.positions[i], self.fitness[i] = candidate, value

        order = np.argsort(self.fitness, kind="stable")
        self.positions, self.fitness = self.positions[order], self.fitness[order]
        m = self.n_agents // 2
        weights = np.log(m + 0.5) - np.log(np.arange(1, m + 1))
        weights /= weights.sum()
        mean = weights @ self.positions[:m]
        delta = self.positions[:m] - mean
        covariance = delta.T @ delta / m  # Eq. 10 is unweighted despite the prose.
        gaussian = self.rng.multivariate_normal(mean, covariance, size=self.n_eda, method="svd", check_valid="ignore")
        candidates = self._clip(gaussian + self.rng.random((self.n_eda, 1)) * (mean - self.positions[:self.n_eda]))
        values = self._evaluate(candidates)
        union_x = np.concatenate([self.positions, candidates])
        union_f = np.concatenate([self.fitness, values])
        keep = np.argsort(union_f, kind="stable")[:self.n_agents]
        self.positions, self.fitness = union_x[keep], union_f[keep]
