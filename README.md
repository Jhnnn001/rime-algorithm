# RIME optimization

Python and MATLAB implementations of **RIME, ACRIME, QGRIME, and HERIME** for
minimizing a user-defined scalar objective within box bounds.
The Python algorithms are in `python-rime/`; MATLAB runs from a
single source file with local functions.

These are independent implementations of the cited methods, with contract
and equation checks. They do not reproduce the papers' feature-selection
pipelines or establish new optimizer performance claims.

## Layout

```
python-rime/   Python implementation and its tests
matlab-rime/   MATLAB implementation and its tests
```

Each folder stands alone. Copy the one for your language into your own
project, or clone the repository and work inside that folder.

## Dependencies

Use either language independently.

- **Python:** Python 3.10+ and NumPy 1.24+.
- **MATLAB:** MATLAB R2019b+; base MATLAB is sufficient.
- **Tests only:** pytest 8+ for Python; MATLAB's built-in test runner for MATLAB.

## Setup

Clone this repository and enter its directory:

```sh
git clone https://github.com/Jhnnn001/rime-algorithm.git
cd rime-algorithm
```

For Python, install the dependency into your chosen environment and check it:

```sh
python -m pip install "numpy>=1.24"
python --version
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

For MATLAB, open the repository as the current folder and run:

```matlab
addpath('matlab-rime');
```

The path addition lasts for the current MATLAB session. The Python commands
below assume that you run them from the `python-rime/` directory.

## Run

### Python

```sh
cd python-rime
python example.py
```

[python-rime/example.py](python-rime/example.py) minimizes `sum((x - [1, -2])**2)` on
`[-5, 5]^2` and prints the best coordinates, objective, and evaluation count.
Its known minimum is zero at `[1, -2]`; the numerical result is approximate.

For your own objective, use this pattern in a script inside `python-rime/`:

```python
import numpy as np
from minimize import minimize

def objective(x):
    return float(np.sum((x - [1.0, -2.0]) ** 2))

result = minimize(objective, [(-5.0, 5.0), (-5.0, 5.0)],
                  method="rime", n_agents=30, max_iter=500, seed=0)
print(result.x, result.fun, result.n_evals)
```

Replace the objective and one bounds pair per coordinate. Set `method` to
`"rime"`, `"acrime"`, `"qgrime"`, or `"herime"`.
For maximization, return the negative of your objective.

You can also call an algorithm directly:
`from rime import RIME`, followed by
`RIME(objective, bounds, seed=0).run()`.
The other modules export `ACRIME`, `QGRIME`, and `HERIME`, respectively.
Constructors validate options without evaluating the objective; each `run()`
starts a fresh population while continuing its random generator.

### MATLAB

From the repository root:

```matlab
addpath('matlab-rime');
objective = @(x) sum((x - [1, -2]).^2);
result = rime_minimize(objective, [-5 5; -5 5], ...
    'method', 'rime', 'n_agents', 30, 'max_iter', 500, 'seed', 0);
disp(result.x);
disp(result.fun);
disp(result.n_evals);
```

Change `'method'` to select another variant.
`help rime_minimize` lists the options. Each call returns a result struct;
MATLAB does not call Python.

## Module reference

Each language folder is self-contained and exposes the same four methods.

### Python — [`python-rime/`](python-rime/)

| Module | Summary |
| --- | --- |
| [`_base`](python-rime/_base.py) | Shared validation, movement, evaluation, history, and the `Result` type. |
| [`rime`](python-rime/rime.py) | `RIME`, the plain rime-ice optimizer [1]. |
| [`acrime`](python-rime/acrime.py) | `ACRIME`: chaotic initialization, mutualism, mutation, and restart [2]. |
| [`qgrime`](python-rime/qgrime.py) | `QGRIME`: Gaussian mutation and the quantum rotation gate [3]. |
| [`herime`](python-rime/herime.py) | `HERIME`: fitness-distance selection and Gaussian EDA fusion [4]. |
| [`minimize`](python-rime/minimize.py) | `minimize`, selecting an algorithm by name. |
| [`benchmarks`](python-rime/benchmarks.py) | Scalar test objectives used by the checks. |
| [`example`](python-rime/example.py) | Minimal runnable objective. |
| [`tests/`](python-rime/tests/) | Contract and paper-equation checks. |

### MATLAB — [`matlab-rime/`](matlab-rime/)

| Function | Summary |
| --- | --- |
| [`rime_minimize`](matlab-rime/rime_minimize.m) | All four methods and their helpers, as local functions in one file. |
| [`tests/`](matlab-rime/tests/) | Contract and paper-equation checks. |

### Repository files

| File | Summary |
| --- | --- |
| [`LICENSE`](LICENSE) | MIT license for this implementation. |

## Inputs, options, and results

| Input | Contract |
| --- | --- |
| `objective` | Callable/function handle receiving one bounded coordinate vector and returning one real scalar to **minimize**. Negate a maximization objective. |
| Coordinate vector | Python: float64 array of shape `(dim,)`. MATLAB: double row of shape `1 × dim`. Objective-side edits cannot change stored positions. |
| `bounds` | `dim × 2`, each row `[lower, upper]`. At least one dimension; endpoints and positive widths must be finite. Coordinates are used directly without rescaling. |
| `n_agents` | Integer ≥ 2, or ≥ 4 for HERIME. Default 30. Booleans are rejected. |
| `max_iter` | Integer ≥ 1. Default 500. Every iteration completes all method phases. |
| Python `seed` | `None`, nonnegative integer, or an existing NumPy `Generator`. |
| MATLAB `seed` | `[]`, integer in `[0, 2^32−1]`, or an existing `RandStream`. Numeric seeds create a local `mt19937ar` stream. |
| `callback` | Optional callable/function handle, invoked once after each completed iteration. Return values are ignored. |

Only QGRIME accepts `theta`, a finite positive angle in radians, default
`0.015*pi`. Only HERIME accepts `lam`, a finite distance weight in `[0, 1]`,
default `0.5`, and `n_eda`, an integer in `[1, n_agents]`, default
`floor(n_agents/2)`. Pass these as Python keywords or MATLAB name-value pairs.
Unknown options and options belonging to a different method are rejected.

The Python `Result` and MATLAB result struct have the same fields:

| Field | Meaning |
| --- | --- |
| `x` | Best evaluated coordinates, using the language's vector shape above. |
| `fun` | Objective value at `x`. |
| `history` | Best value after initialization and each iteration; length `max_iter + 1`. MATLAB returns a column. |
| `n_iter` | Completed iterations. |
| `n_evals` | Actual objective calls, including initialization and every extra phase. |
| `method` | Lowercase method name. |

The best value is an archive over all evaluations and need not belong to the
current population. ACRIME may replace the archived position on a fitness tie
under Eq. 19. Fitness history is non-increasing. NaN objective returns become
positive infinity, and either infinity is accepted. If all evaluations return
NaN or positive infinity, the result still contains an evaluated bounded
position with `fun = inf`. Objective and callback exceptions propagate.

The callback receives the Python optimizer or a MATLAB state struct, exposing
`iter`, `positions`, `fitness`, `best_x`, `best_f`, `n_evals`, `bounds`, `dim`,
`n_agents`, `max_iter`, `method`, and `rng`. `positions` is `n_agents × dim`;
`fitness` is a Python vector or MATLAB column. ACRIME also exposes `TR`, its
consecutive-stagnation counters. Treat callback state as read-only; consuming
the random stream changes subsequent search. The library performs no printing,
file writes, parallel evaluation, caching, or early termination.

Python also provides `benchmarks`: `sphere`, `rastrigin`, `schwefel`,
`zakharov`, and `bounds(name, dim)`. These are mathematical test functions;
the runnable example uses a separate shifted quadratic.

## Evaluation budgets and reproducibility

For population size `n`, iterations `T`, and HERIME EDA size `k`:

| Method | Total objective calls |
| --- | --- |
| RIME / QGRIME | `n(T + 1)` |
| ACRIME | `n + (2n + 1)T + 2R`, where `R` counts all triggered agent restarts |
| HERIME | `n(T + 1) + kT` |

Compare methods by actual evaluation counts; equal iteration limits can have
different costs. Seeded runs are reproducible with a deterministic objective
and unchanged software/environment. Python and MATLAB use different random
implementations, so identical seeds do not imply identical trajectories.
QGRIME's unit Gaussian noise and rotation operate in the supplied coordinates;
coordinate units and origin affect the search. Use numerically reasonable
coordinate and fitness scales to avoid floating-point overflow.

## Notes

Each method implements the equations of its source publication. Where the
authors' prose, pseudocode, and reference code disagree, this implementation
follows the cited equations; Python and MATLAB make identical choices.

| Method | Source | Implemented here |
| --- | --- | --- |
| `RIME` | [1] | Soft-rime search, hard-rime puncture, positive greedy selection. |
| `ACRIME` | [2] | Chaotic initialization, adaptive mutualism, mutation, and restart on top of `RIME`. |
| `QGRIME` | [3] | Gaussian mutation followed by a quantum rotation gate. |
| `HERIME` | [4] | Fitness-distance selection fused with a Gaussian estimation-of-distribution step. |

### Deviations from the published text

**RIME [1].** The cosine term is `cos(pi*t/(10*T))`, and the hard-rime
comparison is drawn from `(-1, 1)`. A zero L2 fitness threshold therefore
still gives a 50% puncture probability.

**ACRIME [2].** Mutualism updates pairs sequentially using a frozen top-two
guide. Equation 19 permits a tied best mutation, while tied restart trials
keep the incumbent. The best archive includes every evaluated trial.

**QGRIME [3].** Gaussian noise has unit scale in the supplied coordinates.
The rotation gate acts on the already updated candidate, and clipping follows.

**HERIME [4].** RIME updates are sequential. Equation 10's covariance is
unweighted despite the prose. The unspecified EDA batch size defaults to half
the population; normalization fallbacks and stable population merging are
conventions of this implementation.

## Checks

For Python:

```sh
cd python-rime
python -m pip install "pytest>=8"
python -m pytest -q tests
```

For MATLAB, from the repository root:

```matlab
addpath('matlab-rime');
results = runtests('matlab-rime/tests');
assertSuccess(results);
```

All **103 Python tests** pass on macOS arm64 with Python 3.14.7, NumPy 2.5.3,
and pytest 9.1.1. The example completes with 15,030 objective calls and
objective `5.826510471021917e-09` on the documented shifted quadratic.
All **8 MATLAB test functions** pass on MATLAB R2024a Update 9, where the
README example returns an objective of approximately `6.0289e-08`.
Other supported versions have not been tested.

The checks exercise bounds, objective-call accounting, seeds, degenerate
fitness, callbacks, paper operators, ACRIME ties and restarts, and singular
HERIME covariance.

## License

[MIT](LICENSE) for this implementation. The cited papers retain their own
copyrights and are not bundled.

## References

1. H. Su et al., "RIME: A physics-based optimization," *Neurocomputing*
   **532**, 183–214 (2023).
   [DOI](https://doi.org/10.1016/j.neucom.2023.02.010).
2. M. Abdel-Salam et al., "Chaotic RIME optimization algorithm with adaptive
   mutualism for feature selection problems," *Computers in Biology and
   Medicine* **179**, 108803 (2024).
   [DOI](https://doi.org/10.1016/j.compbiomed.2024.108803).
3. T. Bai et al., "QGRIME: A RIME optimizer with quantum rotation gate and
   Gaussian mutation for feature selection," *Journal of Computational Design
   and Engineering* **12**(2), 235–257 (2025).
   [DOI](https://doi.org/10.1093/jcde/qwaf016).
4. W. Li et al., "A Novel Hybrid Improved RIME Algorithm for Global Optimization
   Problems," *Biomimetics* **10**(1), 14 (2025).
   [DOI](https://doi.org/10.3390/biomimetics10010014).
