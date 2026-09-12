"""QGRIME: Bai et al. (2025), doi:10.1093/jcde/qwaf016."""

from collections.abc import Callable
from math import sqrt

import numpy as np
from numpy.typing import ArrayLike

from _base import BaseRIME


class QGRIME(BaseRIME):
    """RIME with unit Gaussian mutation and a rotation gate on the updated candidate.

    ``theta`` defaults to the paper's 0.015*pi radians. Noise and rotations act
    in the supplied coordinates, so behavior depends on coordinate scale.
    """

    method = "qgrime"

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: ArrayLike,
        n_agents: int = 30,
        max_iter: int = 500,
        seed: int | np.random.Generator | None = None,
        callback: Callable[[BaseRIME], None] | None = None,
        *,
        theta: float = 0.015 * np.pi,
    ) -> None:
        """Configure the shared optimizer options and a finite positive rotation angle."""
        super().__init__(objective, bounds, n_agents, max_iter, seed, callback)
        self.theta = float(theta)
        if not np.isfinite(self.theta) or self.theta <= 0:
            raise ValueError("theta must be finite and > 0.")

    def _propose(self, t):
        factor = self._rime_factor(t)
        threshold = self._hard_rime_threshold()
        u1, u2, u3 = (self.rng.random(self.positions.shape) for _ in range(3))
        noise = self.rng.standard_normal(self.positions.shape)
        soft = self.best_x + factor * (self._lb + u2 * (self._ub - self._lb))
        candidates = np.where(u1 < sqrt(t / self.max_iter), soft, self.positions + noise)
        candidates = np.where(2 * u3 - 1 < threshold[:, None], self.best_x, candidates)

        # Table 1: sign multiplication avoids coordinate-product overflow/underflow.
        sign = np.sign(candidates) * np.sign(self.best_x)
        worse = self.fitness[:, None] > self.best_f
        better = self.fitness[:, None] < self.best_f
        angles = np.zeros_like(candidates)
        angles[(worse & (sign > 0)) | (better & (sign < 0))] = self.theta
        angles[(worse & (sign < 0)) | (better & (sign > 0))] = -self.theta
        random_sign = (worse & (candidates == 0)) | (better & (self.best_x == 0) & (candidates != 0))
        angles[random_sign] = self.theta * np.where(self.rng.random(np.count_nonzero(random_sign)) < 0.5, 1, -1)
        return candidates * np.cos(angles) - self.best_x * np.sin(angles)
