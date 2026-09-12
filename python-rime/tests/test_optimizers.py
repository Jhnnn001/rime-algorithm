"""Contract and equation checks, independent of stochastic performance claims."""

import numpy as np
import pytest
from types import SimpleNamespace

import benchmarks
from acrime import ACRIME
from herime import HERIME
from qgrime import QGRIME
from rime import RIME
from minimize import minimize

METHODS = [RIME, ACRIME, QGRIME, HERIME]


def test_benchmarks():
    zero = np.zeros(3)
    for objective in (benchmarks.sphere, benchmarks.rastrigin, benchmarks.zakharov):
        assert objective(zero) == 0
    assert benchmarks.zakharov(np.array([1.0, 2.0])) == 50.3125
    assert benchmarks.schwefel(np.full(2, 420.9687)) == pytest.approx(2.5455675e-5, abs=1e-10)
    assert np.array_equal(benchmarks.bounds("SPHERE", 2), [[-100, 100], [-100, 100]])
    for name, dim in (("unknown", 2), ("sphere", 0), ("sphere", 1.5), ("sphere", True)):
        with pytest.raises(ValueError):
            benchmarks.bounds(name, dim)


@pytest.mark.parametrize("optimizer", METHODS)
@pytest.mark.parametrize("kwargs", [
    {"bounds": [1, 2, 3]}, {"bounds": [[1, 2, 3]]}, {"bounds": []},
    {"bounds": [[1, 1]]}, {"bounds": [[2, 1]]}, {"bounds": [[0, np.inf]]},
    {"bounds": [[np.nan, 2]]}, {"bounds": [[-1e308, 1e308]]},
    {"n_agents": 1}, {"n_agents": 2.5}, {"n_agents": True},
    {"max_iter": 0}, {"max_iter": 1.5}, {"max_iter": False},
])
def test_validation(optimizer, kwargs):
    arguments = {"bounds": [[-1, 1]], **kwargs}
    with pytest.raises(ValueError):
        optimizer(benchmarks.sphere, **arguments)


def test_invalid_callable_and_method():
    with pytest.raises(TypeError):
        RIME(3, [[-1, 1]])
    with pytest.raises(TypeError):
        RIME(benchmarks.sphere, [[-1, 1]], callback=3)
    with pytest.raises(TypeError):
        RIME(benchmarks.sphere, [[-1, 1]], nope=1)
    for method in (None, 1, "unknown"):
        with pytest.raises(ValueError):
            minimize(benchmarks.sphere, [[-1, 1]], method=method)


@pytest.mark.parametrize("optimizer", METHODS)
def test_run_contract(optimizer):
    box = np.array([[-3, 4], [2, 6], [-5, -1], [0, 0.01]])
    calls, iterations, recorded_best = [], [], []

    def objective(x):
        assert x.shape == (4,) and x.dtype == np.float64
        assert np.all((box[:, 0] <= x) & (x <= box[:, 1]))
        calls.append(x.copy())
        return benchmarks.sphere(x)

    def callback(opt):
        assert opt.positions.shape == (10, 4)
        assert opt.fitness.shape == (10,)
        iterations.append(opt.iter)
        recorded_best.append(opt.best_f)

    opt = optimizer(objective, box, n_agents=10, max_iter=30, seed=0, callback=callback)
    assert not calls and opt.positions is None and opt.best_x is None
    result = opt.run()
    assert iterations == list(range(1, 31))
    assert np.all(np.diff(recorded_best) <= 0)
    assert result.n_evals == len(calls)
    assert result.n_iter == 30 and result.method == opt.method
    assert result.history.shape == (31,)
    assert np.all(result.history[1:] <= result.history[:-1])
    assert result.fun == result.history[-1] == benchmarks.sphere(result.x)
    assert not np.shares_memory(result.x, opt.best_x)
    assert not np.shares_memory(opt.bounds, box)


@pytest.mark.parametrize("optimizer", METHODS)
def test_seed_and_fresh_run(optimizer):
    box = np.tile([-10, 10], (4, 1))
    left = optimizer(benchmarks.sphere, box, n_agents=7, max_iter=11, seed=123)
    right = optimizer(benchmarks.sphere, box, n_agents=7, max_iter=11, seed=123)
    first = left.run()
    duplicate = right.run()
    assert np.array_equal(first.x, duplicate.x)
    assert np.array_equal(first.history, duplicate.history)
    second = left.run()
    assert np.array_equal(second.history, right.run().history)
    assert not np.array_equal(first.history, second.history)
    other = optimizer(benchmarks.sphere, box, n_agents=7, max_iter=11, seed=124).run()
    assert not np.array_equal(first.history, other.history)
    via_dispatch = minimize(benchmarks.sphere, box, method=left.method.upper(), n_agents=7, max_iter=11, seed=123)
    assert np.array_equal(first.history, via_dispatch.history)
    generator = np.random.default_rng(0)
    assert optimizer(benchmarks.sphere, box, seed=generator).rng is generator


@pytest.mark.parametrize("optimizer", METHODS)
def test_evaluation_counts(optimizer):
    result = optimizer(benchmarks.sphere, [[-3, 5]], n_agents=7, max_iter=11, seed=0).run()
    if result.method == "acrime":
        assert 172 <= result.n_evals <= 326
    elif result.method == "herime":
        assert result.n_evals == 117
    else:
        assert result.n_evals == 84


@pytest.mark.parametrize("optimizer", METHODS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0.0])
def test_degenerate_fitness(optimizer, value):
    result = optimizer(lambda x: value, [[2, 3]], n_agents=7, max_iter=5, seed=0).run()
    expected = np.inf if np.isnan(value) else value
    assert result.fun == expected
    assert np.all(result.history == expected)
    assert 2 <= result.x[0] <= 3


@pytest.mark.parametrize("optimizer", METHODS)
def test_nan_region_and_input_copy(optimizer):
    def objective(x):
        value = np.nan if x[0] > 0 else benchmarks.sphere(x)
        x[:] = 1e6
        return value

    result = optimizer(objective, [[-1, 1]], n_agents=10, max_iter=30, seed=0).run()
    assert np.isfinite(result.fun) and result.x[0] <= 0
    assert result.fun == benchmarks.sphere(result.x)


def test_exceptions_and_false_callback():
    def broken(x):
        raise RuntimeError("objective failed")

    opt = RIME(broken, [[0, 1]])
    with pytest.raises(RuntimeError, match="objective failed"):
        opt.run()
    assert opt.n_evals == 1
    with pytest.raises((TypeError, ValueError)):
        RIME(lambda x: object(), [[0, 1]]).run()

    class Callback:
        calls = 0

        def __bool__(self):
            return False

        def __call__(self, opt):
            self.calls += 1

    callback = Callback()
    RIME(benchmarks.sphere, [[0, 1]], max_iter=3, callback=callback).run()
    assert callback.calls == 3


def test_rime_equations():
    opt = RIME(benchmarks.sphere, [[0, 10], [0, 20]], n_agents=2, max_iter=10)
    opt.positions = np.array([[1.0, 2.0], [3.0, 4.0]])
    opt.best_x = opt.positions[0].copy()
    opt.fitness = np.zeros(2)
    draws = iter([np.ones((2, 2)), np.full((2, 2), 0.5), [[0.25, 0.75], [0.75, 0.25]]])
    opt.rng = SimpleNamespace(uniform=lambda low, high: 0.5, random=lambda shape: np.asarray(next(draws)))
    assert np.array_equal(opt._propose(1), [[1, 2], [3, 2]])
    assert opt._rime_factor(1) == pytest.approx(0.4 * np.cos(np.pi / 100))
    assert opt._rime_factor(9) == 0
    opt.fitness = np.array([-3.0, -4.0])
    assert np.allclose(opt._hard_rime_threshold(), [-0.6, -0.8])
    u2 = np.array([[0.25, 0.75], [0.75, 0.25]])
    draws = iter([np.zeros((2, 2)), u2, np.ones((2, 2))])
    expected = [1, 2] + 0.4 * np.cos(np.pi / 100) * u2 * [10, 20]
    assert np.allclose(opt._propose(1), expected)
    old = opt.positions.copy()
    opt._greedy_select(old + 10, np.array([-3.0, -5.0]))
    assert np.array_equal(opt.positions[0], old[0])
    assert np.array_equal(opt.positions[1], old[1] + 10)


def test_qgrime_validation():
    for theta in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            QGRIME(benchmarks.sphere, [[-1, 1]], theta=theta)
    with pytest.raises(TypeError):
        QGRIME(benchmarks.sphere, [[-1, 1]], gauss_scale=0.01)


def test_qgrime_rotation_table_and_candidate_order():
    # Table 1 rows: equal, worse, better; columns include both-zero precedence.
    candidate = np.array([[2, -2, 0, 2, 0], [2, -2, 0, 2, 0], [2, -2, 0, 2, 0]], dtype=float)
    best = np.array([1, 1, 1, 0, 0], dtype=float)
    opt = QGRIME(benchmarks.sphere, np.tile([-10, 10], (5, 1)), n_agents=3, max_iter=10, theta=0.1)
    opt.positions = candidate - 1
    opt.best_x = best.copy()
    opt.best_f = 0
    opt.fitness = np.array([0.0, 1.0, -1.0])
    draws = iter([np.ones((3, 5)), np.zeros((3, 5)), np.ones((3, 5)), np.array([0.25, 0.75, 0.25])])
    opt.rng = SimpleNamespace(
        uniform=lambda low, high: 0.0,
        random=lambda shape: np.asarray(next(draws)),
        standard_normal=lambda shape: np.ones(shape),
    )
    angles = np.array([[0, 0, 0, 0, 0], [0.1, -0.1, 0.1, 0, -0.1], [-0.1, 0.1, 0, 0.1, 0]])
    expected = candidate * np.cos(angles) - best * np.sin(angles)
    assert np.allclose(opt._propose(1), expected)
    assert np.array_equal(opt.best_x, best)


def test_qgrime_unit_noise_and_final_clip():
    for width in (1, 100):
        opt = QGRIME(benchmarks.sphere, [[-width, width]], n_agents=2, max_iter=10)
        opt.positions = np.zeros((2, 1))
        opt.best_x, opt.best_f, opt.fitness = np.zeros(1), 0.0, np.zeros(2)
        draws = iter([np.ones((2, 1)), np.zeros((2, 1)), np.ones((2, 1)), np.empty(0)])
        opt.rng = SimpleNamespace(uniform=lambda low, high: 0.0, random=lambda shape: next(draws), standard_normal=lambda shape: np.full(shape, 0.25))
        assert np.array_equal(opt._propose(1), [[0.25], [0.25]])
    opt = QGRIME(benchmarks.sphere, [[0, 100]], n_agents=2, max_iter=10)
    opt.positions = np.full((2, 1), 100.0)
    opt.best_x, opt.best_f, opt.fitness = np.array([100.0]), 0.0, np.ones(2)
    draws = iter([np.ones((2, 1)), np.zeros((2, 1)), np.ones((2, 1)), np.empty(0)])
    opt.rng = SimpleNamespace(uniform=lambda low, high: 0.0, random=lambda shape: next(draws), standard_normal=lambda shape: np.full(shape, 100.0))
    assert np.array_equal(opt._clip(opt._propose(1)), [[100], [100]])


def test_acrime_chaotic_initialization():
    opt = ACRIME(benchmarks.sphere, [[0, 2]], n_agents=3)
    draws = iter([0.0, -1.0, 1.0, 0.5, 0.25])
    opt.rng = SimpleNamespace(uniform=lambda low, high: next(draws))
    # 0.5 maps to -1, so the excluded state is redrawn as 0.25.
    assert np.array_equal(opt._init_population().ravel(), [1.5, 1.25, 0.3125])
    assert np.array_equal(opt.TR, [0, 0, 0])
    assert ACRIME(benchmarks.sphere, [[0, 2]], n_agents=2, max_iter=2, seed=0).run().n_iter == 2


def test_acrime_mutualism_uses_copied_pairs_and_frozen_guide():
    calls = []
    opt = ACRIME(lambda x: calls.append(x.copy()) or 1.0, [[-10, 10]], n_agents=2, max_iter=10)
    opt.positions = np.array([[2.0], [4.0]])
    opt.fitness, opt.best_f, opt.best_x, opt.TR = np.ones(2), 1.0, np.array([2.0]), np.zeros(2, dtype=int)
    partners = iter([1, 0])
    opt.rng = SimpleNamespace(
        integers=lambda n: next(partners), uniform=lambda low, high: 0.0,
        random=lambda size=None: 0.25 if size is None else np.full(size, 0.5),
        standard_cauchy=lambda size: np.zeros(size), standard_normal=lambda size: np.zeros(size),
    )
    opt._step(4)
    # First pair: (1.625, 2.875). Second pair uses that pair and guide=3.
    assert np.array_equal(np.asarray(calls[:2]).ravel(), [1.15625, 2.96875])
    assert opt.n_evals == 5  # No restart at t=4 with TR=1.
    assert np.array_equal(opt.TR, [1, 1])


def test_acrime_mutation_ties_and_restart_counters():
    calls = []
    opt = ACRIME(lambda x: calls.append(x.copy()) or 1.0, [[0, 10]], n_agents=2, max_iter=10)
    opt.positions = np.array([[2.0], [4.0]])
    opt.fitness, opt.best_f, opt.best_x, opt.TR = np.ones(2), 1.0, np.array([2.0]), np.zeros(2, dtype=int)
    partners = iter([1, 0, 1, 0])
    opt.rng = SimpleNamespace(
        integers=lambda n: next(partners), uniform=lambda low, high: 0.0,
        random=lambda size=None: 0.0 if size is None else np.zeros(size),
        standard_cauchy=lambda size: np.full(size, 2.0), standard_normal=lambda size: np.full(size, 3.0),
    )
    opt._step(1)
    assert calls[4][0] == pytest.approx(6.02)  # best * (1 + .99*Cauchy + .01*Gaussian)
    assert opt.best_x[0] == pytest.approx(6.02) and opt.best_f == 1
    assert opt.n_evals == 9
    assert np.array_equal(opt.positions.ravel(), [2, 4])  # Restart ties keep incumbents.
    assert np.array_equal(opt.TR, [0, 0])  # Boundary repair did not cancel either attempt.
    opt.TR[:] = 2
    opt.fitness[0] = 2  # A changed fitness resets its consecutive-stagnation counter.
    opt._step(4)
    assert opt.n_evals == 16  # Only the second agent restarts.
    assert np.array_equal(opt.TR, [0, 0])


def test_acrime_restart_strict_branches():
    opt = ACRIME(lambda x: x[0], [[0, 10]], n_agents=2, max_iter=10)
    opt.positions = np.array([[2.0], [4.0]])
    opt.fitness, opt.best_f, opt.best_x, opt.TR = np.array([2.0, 4.0]), 2.0, np.array([2.0]), np.zeros(2, dtype=int)
    partners = iter([1, 0])
    vectors = iter([0, 0, 0, 0, 0.1, 0.5, 0.8, 0.5])

    def random(size=None):
        if size is None:
            return 0.0
        if isinstance(size, tuple):
            return np.ones(size)
        return np.full(size, next(vectors)) if size else np.empty(0)

    opt.rng = SimpleNamespace(
        integers=lambda n: next(partners), uniform=lambda low, high: 0.0, random=random,
        standard_cauchy=lambda size: np.zeros(size), standard_normal=lambda size: np.zeros(size),
    )
    opt._step(1)
    assert np.array_equal(opt.positions.ravel(), [1, 1])  # T1 wins first; T2 wins second.
    assert np.array_equal(opt.fitness, [1, 1])
    assert opt.n_evals == 9


def test_herime_validation_and_eda_sizes():
    for kwargs in ({"n_agents": 3}, {"lam": -0.1}, {"lam": 1.1}, {"lam": np.nan}, {"lam": np.inf},
                   {"n_eda": 0}, {"n_eda": 5}, {"n_eda": 1.5}, {"n_eda": True}):
        with pytest.raises(ValueError):
            HERIME(benchmarks.sphere, [[-1, 1]], **{"n_agents": 4, **kwargs})
    for count in (1, 7):
        result = HERIME(benchmarks.sphere, [[-1, 1]], n_agents=7, max_iter=3, seed=0, n_eda=count).run()
        assert result.n_evals == 28 + 3 * count


def test_herime_sequential_archive_and_frozen_reference():
    calls = []
    opt = HERIME(lambda x: calls.append(x.copy()) or x[0], [[0, 20]], n_agents=4, max_iter=10, n_eda=1)
    opt.positions = np.arange(10.0, 14.0).reshape(4, 1)
    opt.fitness, opt.best_f, opt.best_x = np.arange(10.0, 14.0), 10.0, np.array([10.0])
    draws = iter([value for hard in (1, 1, 1, 0) for value in (0, 0.1, hard)] + [0])
    opt.rng = SimpleNamespace(
        choice=lambda n, p: 0, uniform=lambda low, high: -0.5 / (0.8 * np.cos(np.pi / 100)),
        random=lambda size: np.full(size, next(draws)),
        multivariate_normal=lambda mean, cov, **kwargs: np.array([[20.0]]),
    )
    opt._step(1)
    assert np.allclose(np.asarray(calls[:4]).ravel(), [9, 8, 7, 10])
    assert opt.best_f == pytest.approx(7) and opt.n_evals == 5


def test_herime_roulette_covariance_and_fusion():
    probabilities, model = [], []
    opt = HERIME(benchmarks.sphere, [[-10, 10]], n_agents=4, max_iter=10)
    opt.positions = np.arange(1.0, 5.0).reshape(4, 1)
    opt.fitness, opt.best_f, opt.best_x = np.array([1.0, 4.0, 9.0, 16.0]), 1.0, np.array([1.0])

    def gaussian(mean, cov, **kwargs):
        model.append((mean.copy(), cov.copy()))
        assert kwargs == {"size": 2, "method": "svd", "check_valid": "ignore"}
        return np.array([[0.0], [5.0]])

    opt.rng = SimpleNamespace(
        choice=lambda n, p: probabilities.append(p.copy()) or 3, uniform=lambda low, high: 0.0,
        random=lambda size: np.full(size, 0.5) if isinstance(size, tuple) else np.ones(size),
        multivariate_normal=gaussian,
    )
    opt._step(1)
    score = 0.5 * np.array([1, 12 / 15, 7 / 15, 0]) + 0.5 * np.array([0, 1 / 3, 2 / 3, 1])
    assert np.allclose(probabilities[0], score / score.sum())
    mean = (np.log(2.5) + 2 * np.log(1.25)) / (np.log(2.5) + np.log(1.25))
    assert model[0][0][0] == pytest.approx(mean)
    assert model[0][1][0, 0] == pytest.approx(((1 - mean)**2 + (2 - mean)**2) / 2)
    assert np.allclose(opt.positions.ravel(), [0.5 * (mean - 1), 1, 2, 3])
    assert opt.n_evals == 6


def test_herime_singular_covariance_and_stable_ties():
    opt = HERIME(lambda x: 0.0, [[-1, 1], [-1, 1]], n_agents=4, max_iter=10, n_eda=4, lam=0, seed=0)
    opt.positions = np.zeros((4, 2))
    opt.fitness, opt.best_f, opt.best_x = np.zeros(4), 0.0, np.zeros(2)
    opt._step(1)
    assert opt.n_evals == 8
    assert np.array_equal(opt.positions, np.zeros((4, 2)))
    # With a constant objective, even distinct EDA rows must lose ties to incumbents.
    opt.positions[:, 0] = [-0.75, -0.25, 0.25, 0.75]
    before = opt.positions.copy()
    opt._step(2)
    assert np.array_equal(opt.positions, before)
