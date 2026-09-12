"""ACRIME: Abdel-Salam et al. (2024), doi:10.1016/j.compbiomed.2024.108803."""

from math import sqrt

import numpy as np

from _base import BaseRIME


class ACRIME(BaseRIME):
    """RIME with chaotic initialization, mutualism, best mutation, and restart.

    Mutualism uses sequential pairs and a frozen top-two guide. Mutation ties
    replace the archive; restart ties keep the incumbent.
    """

    method = "acrime"

    def _init_population(self):
        self.TR = np.zeros(self.n_agents, dtype=int)
        positions = np.empty((self.n_agents, self.dim))
        for j in range(self.dim):
            s = self.rng.uniform(-1, 1)
            for i in range(self.n_agents):
                if i:
                    s = np.clip(4 * s**3 - 3 * s, -1, 1)
                while s <= -1 or s >= 1 or s == 0:
                    s = self.rng.uniform(-1, 1)
                positions[i, j] = self._lb[j] + (self._ub[j] - self._lb[j]) * (s + 1) / 2
        return positions

    def _step(self, t):
        previous = self.fitness.copy()
        first, second = np.argsort(self.fitness, kind="stable")[:2]
        guide = (self.positions[first] + self.positions[second]) / 2
        for i in range(self.n_agents):
            z = self.rng.integers(self.n_agents)
            while z == i:
                z = self.rng.integers(self.n_agents)
            bf1 = 1 + self.rng.random()
            bf2 = 3 - bf1
            xi, xz = self.positions[i].copy(), self.positions[z].copy()
            mutual = (xi + xz) / 2
            target = guide if self.rng.random() < 0.5 else self.positions[self.rng.integers(self.n_agents)].copy()
            self.positions[i] = xi + self.rng.random(self.dim) * (target - mutual * bf1)
            self.positions[z] = xz + self.rng.random(self.dim) * (target - mutual * bf2)
        self.positions = self._clip(self.positions)
        self.fitness = self._evaluate(self.positions)

        super()._step(t)

        l2 = (t / self.max_iter) ** 2
        mutated = self.best_x * (1 + (1 - l2) * self.rng.standard_cauchy(self.dim) + l2 * self.rng.standard_normal(self.dim))
        old_best = self.best_f
        mutated = self._clip(mutated[None, :])
        value = self._evaluate(mutated)[0]
        if value <= old_best:
            self.best_x, self.best_f = mutated[0].copy(), float(value)

        self.TR = np.where(self.fitness == previous, self.TR + 1, 0)
        for i in np.flatnonzero(self.TR >= sqrt(t)):
            trial1 = self._lb + self.rng.random(self.dim) * (self._ub - self._lb)
            trial2 = self.rng.random(self.dim) * (self._ub + self._lb) - self.positions[i]
            repair = (trial2 <= self._lb) | (trial2 >= self._ub)
            trial2[repair] = self._lb[repair] + self.rng.random(np.count_nonzero(repair)) * (self._ub - self._lb)[repair]
            f1, f2 = self._evaluate(np.stack([trial1, trial2]))
            if f1 < f2:
                self.positions[i], self.fitness[i] = trial1, f1
            elif f2 < f1:
                self.positions[i], self.fitness[i] = trial2, f2
            self.TR[i] = 0
