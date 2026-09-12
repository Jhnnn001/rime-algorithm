"""Minimize a shifted quadratic; run with python example.py."""

import numpy as np

from minimize import minimize


def objective(x):
    return float(np.sum((x - [1.0, -2.0]) ** 2))


if __name__ == "__main__":
    result = minimize(objective, [(-5.0, 5.0), (-5.0, 5.0)],
                      method="rime", n_agents=30, max_iter=500, seed=0)
    print("Best coordinates:", result.x)
    print("Objective:", result.fun)
    print("Evaluations:", result.n_evals)
