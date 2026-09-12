"""Shared RIME engine: Su et al. (2023), doi:10.1016/j.neucom.2023.02.010."""

from collections.abc import Callable
from dataclasses import dataclass
from math import cos, floor, pi, sqrt

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class Result:
    """Best evaluated position, fitness history, and objective-call accounting."""

    x: np.ndarray
    fun: float
    history: np.ndarray
    n_iter: int
    n_evals: int
    method: str


class BaseRIME:
    """Shared validation, bounded evaluation, best archive, and iteration loop."""

    method: str

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: ArrayLike,
        n_agents: int = 30,
        max_iter: int = 500,
        seed: int | np.random.Generator | None = None,
        callback: Callable[["BaseRIME"], None] | None = None,
    ) -> None:
        """Configure a minimizer without evaluating the objective.

        Parameters
        ----------
        objective : callable
            Receives one float64 coordinate vector and returns a scalar to minimize.
        bounds : array_like of shape (dim, 2)
            Finite, strictly increasing lower and upper bounds for each variable.
        n_agents, max_iter : int
            Population size (at least two) and number of complete iterations.
        seed : int, numpy.random.Generator, or None
            Random seed or an existing generator, used without reseeding.
        callback : callable or None
            Called with the optimizer after each iteration; return value is ignored.
        """
        if not callable(objective):
            raise TypeError("objective must be callable.")
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None.")
        box = np.asarray(bounds, dtype=float)
        if box.ndim != 2 or box.shape[1] != 2 or box.shape[0] == 0:
            raise ValueError("bounds must have shape (dim, 2), with dim >= 1.")
        with np.errstate(over="ignore", invalid="ignore"):
            widths = box[:, 1] - box[:, 0]
        if not np.all(np.isfinite(box)) or not np.all(np.isfinite(widths) & (widths > 0)):
            raise ValueError("bounds must have finite endpoints and finite positive widths.")
        for name, value, minimum in (("n_agents", n_agents, 2), ("max_iter", max_iter, 1)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        self.objective = objective
        self.bounds = box.copy()
        self._lb, self._ub = self.bounds[:, 0], self.bounds[:, 1]
        self.dim = len(box)
        self.n_agents, self.max_iter = int(n_agents), int(max_iter)
        self.rng = np.random.default_rng(seed)
        self.callback = callback
        self.iter = self.n_evals = 0
        self.positions = self.fitness = self.best_x = None
        self.best_f = float("inf")

    def _evaluate(self, candidates):
        values = np.empty(len(candidates))
        for i, row in enumerate(candidates):
            self.n_evals += 1
            value = float(self.objective(row.copy()))
            values[i] = float("inf") if np.isnan(value) else value
            if values[i] < self.best_f:
                self.best_f = float(values[i])
                self.best_x = row.copy()
        return values

    def _clip(self, positions):
        return np.clip(positions, self._lb, self._ub)

    def _init_population(self):
        return self._lb + self.rng.random((self.n_agents, self.dim)) * (self._ub - self._lb)

    def _rime_factor(self, t):
        beta = 1 - floor(5 * t / self.max_iter + 0.5) / 5
        return self.rng.uniform(-1, 1) * cos(pi * t / (10 * self.max_iter)) * beta

    def _hard_rime_threshold(self):
        """Return L2 thresholds q; puncture probability is (q + 1) / 2."""
        with np.errstate(over="ignore", invalid="ignore"):
            norm = np.linalg.norm(self.fitness)
        if norm == 0 or not np.isfinite(norm):
            return np.zeros(self.n_agents)
        return self.fitness / norm

    def _propose(self, t):
        factor = self._rime_factor(t)
        threshold = self._hard_rime_threshold()
        u1, u2, u3 = (self.rng.random(self.positions.shape) for _ in range(3))
        soft = self.best_x + factor * (self._lb + u2 * (self._ub - self._lb))
        candidates = np.where(u1 < sqrt(t / self.max_iter), soft, self.positions)
        return np.where(2 * u3 - 1 < threshold[:, None], self.best_x, candidates)

    def _greedy_select(self, candidates, values):
        improved = values < self.fitness
        self.positions[improved] = candidates[improved]
        self.fitness[improved] = values[improved]

    def _step(self, t):
        candidates = self._clip(self._propose(t))
        self._greedy_select(candidates, self._evaluate(candidates))

    def run(self) -> Result:
        """Start a fresh population and return the best result after ``max_iter`` iterations.

        The generator continues from its current state. NaN objective returns
        become positive infinity; objective and callback exceptions propagate.
        """
        self.iter = self.n_evals = 0
        self.best_f = float("inf")
        self.positions = self._clip(self._init_population())
        self.best_x = self.positions[0].copy()
        self.fitness = self._evaluate(self.positions)
        history = np.empty(self.max_iter + 1)
        history[0] = self.best_f
        for t in range(1, self.max_iter + 1):
            self.iter = t
            self._step(t)
            history[t] = self.best_f
            if self.callback is not None:
                self.callback(self)
        return Result(self.best_x.copy(), self.best_f, history, self.max_iter, self.n_evals, self.method)
