function tests = test_optimizers
% Contract and paper-operator checks; run with MATLAB's built-in test runner.
tests = functiontests(localfunctions);
end

function testRunContract(testCase)
box = [-3 4; 2 6; -5 -1; 0 0.01];
methods = {'rime', 'acrime', 'qgrime', 'herime'};
calls = 0;
iterations = [];
for k = 1:numel(methods)
    calls = 0;
    iterations = [];
    result = rime_minimize(@objective, box, 'method', methods{k}, ...
        'n_agents', 10, 'max_iter', 30, 'seed', 0, 'callback', @callback);
    verifyEqual(testCase, result.n_evals, calls);
    verifyEqual(testCase, result.n_iter, 30);
    verifyEqual(testCase, result.method, methods{k});
    verifyEqual(testCase, iterations, 1:30);
    verifySize(testCase, result.history, [31 1]);
    verifyTrue(testCase, all(result.history(2:end) <= result.history(1:end-1)));
    verifyEqual(testCase, result.fun, result.history(end));
    verifyEqual(testCase, result.fun, sum(result.x.^2));
end
    function value = objective(x)
        assert(isa(x, 'double') && isequal(size(x), [1 4]));
        assert(all(x >= box(:, 1)' & x <= box(:, 2)'));
        calls = calls + 1;
        value = sum(x.^2);
    end
    function callback(state)
        assert(isequal(size(state.positions), [10 4]));
        assert(isequal(size(state.fitness), [10 1]));
        iterations(end + 1) = state.iter;
    end
end

function testSeedsAndCounts(testCase)
methods = {'rime', 'acrime', 'qgrime', 'herime'};
before = rng;
for k = 1:numel(methods)
    args = {'method', upper(methods{k}), 'n_agents', 7, 'max_iter', 11};
    first = rime_minimize(@(x) sum(x.^2), [-3 5; -2 7], args{:}, 'seed', 123);
    second = rime_minimize(@(x) sum(x.^2), [-3 5; -2 7], args{:}, 'seed', 123);
    other = rime_minimize(@(x) sum(x.^2), [-3 5; -2 7], args{:}, 'seed', 124);
    verifyEqual(testCase, first, second);
    verifyFalse(testCase, isequal(first.history, other.history));
    if strcmp(methods{k}, 'acrime')
        verifyTrue(testCase, first.n_evals >= 172 && first.n_evals <= 326);
    elseif strcmp(methods{k}, 'herime')
        verifyEqual(testCase, first.n_evals, 117);
    else
        verifyEqual(testCase, first.n_evals, 84);
    end
    left = RandStream('mt19937ar', 'Seed', 0);
    right = RandStream('mt19937ar', 'Seed', 0);
    for repeat = 1:2
        a = rime_minimize(@(x) sum(x.^2), [-3 5], args{:}, 'seed', left);
        b = rime_minimize(@(x) sum(x.^2), [-3 5], args{:}, 'seed', right);
        verifyEqual(testCase, a, b);
    end
end
verifyEqual(testCase, rng, before);
end

function testNonfiniteFitness(testCase)
methods = {'rime', 'acrime', 'qgrime', 'herime'};
for k = 1:numel(methods)
    for value = [NaN Inf -Inf 0]
        result = rime_minimize(@(x) value, [2 3], 'method', methods{k}, ...
            'n_agents', 7, 'max_iter', 5, 'seed', 0);
        expected = value;
        if isnan(value), expected = Inf; end
        verifyEqual(testCase, result.fun, expected);
        verifyTrue(testCase, all(result.history == expected));
        verifyTrue(testCase, result.x >= 2 && result.x <= 3);
    end
    result = rime_minimize(@partial, [-1 1], 'method', methods{k}, ...
        'n_agents', 10, 'max_iter', 30, 'seed', 0);
    verifyTrue(testCase, isfinite(result.fun) && result.x <= 0);
    verifyEqual(testCase, result.fun, result.x^2);
end
    function value = partial(x)
        value = x^2;
        if x > 0, value = NaN; end
    end
end

function testValidation(testCase)
objective = @(x) sum(x.^2);
for box = {[], [1 2 3], [1 1], [2 1], [0 Inf], [NaN 1], [-1e308 1e308], [1+1i 2]}
    verifyError(testCase, @() rime_minimize(objective, box{1}), 'rimeopt:InvalidBounds');
end
for pair = {{'n_agents', 1}, {'n_agents', 2.5}, {'n_agents', true}, ...
        {'max_iter', 0}, {'max_iter', 1.5}, {'max_iter', false}, {'seed', -1}, {'seed', 2^32}}
    args = pair{1};
    verifyError(testCase, @() rime_minimize(objective, [-1 1], args{:}), 'rimeopt:InvalidInteger');
end
verifyError(testCase, @() rime_minimize(3, [-1 1]), 'rimeopt:InvalidObjective');
verifyError(testCase, @() rime_minimize(objective, [-1 1], 'callback', 3), 'rimeopt:InvalidCallback');
verifyError(testCase, @() rime_minimize(objective, [-1 1], 'method', 'unknown'), 'rimeopt:InvalidMethod');
verifyError(testCase, @() rime_minimize(objective, [-1 1], 'theta', 0.1), 'rimeopt:InvalidOption');
for angle = [0 -1 NaN Inf]
    verifyError(testCase, @() rime_minimize(objective, [-1 1], 'method', 'qgrime', 'theta', angle), 'rimeopt:InvalidOption');
end
verifyError(testCase, @() rime_minimize(objective, [-1 1], 'method', 'herime', 'n_agents', 3), 'rimeopt:InvalidOption');
verifyError(testCase, @() rime_minimize(objective, [-1 1], 'method', 'herime', 'lam', 1.5), 'rimeopt:InvalidOption');
verifyError(testCase, @() rime_minimize(objective, [-1 1], 'method', 'herime', 'n_eda', 0), 'rimeopt:InvalidInteger');
verifyError(testCase, @() rime_minimize(@(x) [1 2], [-1 1]), 'rimeopt:InvalidValue');
verifyError(testCase, @() rime_minimize(@failingObjective, [-1 1]), 'test:Objective');
verifyError(testCase, @() rime_minimize(objective, [-1 1], 'callback', @(s) error('test:Callback', 'failed')), 'test:Callback');
end

function value = failingObjective(~) %#ok<STOUT>
error('test:Objective', 'failed');
end

function testRimePaperEquations(testCase)
calls = [];
seed = 42;
stream = RandStream('mt19937ar', 'Seed', seed);
initial = rand(stream, 2, 2) .* [10 20];
factor = (2 * rand(stream) - 1) * cos(pi / 100) * 0.8;
u1 = rand(stream, 2, 2);
u2 = rand(stream, 2, 2);
u3 = rand(stream, 2, 2);
expected = initial;
soft = initial(1, :) + factor * u2 .* [10 20];
attach = u1 < sqrt(0.1);
expected(attach) = soft(attach);
target = repmat(initial(1, :), 2, 1);
hard = 2 * u3 - 1 < 0; % Zero L2 threshold still means 50% puncture.
assert(any(hard(:)));
expected(hard) = target(hard);
expected = min([10 20], max([0 0], expected));
rime_minimize(@objective, [0 10; 0 20], 'n_agents', 2, 'max_iter', 10, 'seed', seed);
verifyEqual(testCase, calls(3:4, :), expected, 'AbsTol', 1e-12);
    function value = objective(x)
        calls(end + 1, :) = x;
        value = 0;
    end
end

function testQgrimeCandidateRotation(testCase)
calls = [];
seed = 42;
stream = RandStream('mt19937ar', 'Seed', seed);
initial = 1 + rand(stream, 3, 2) * 9;
factor = (2 * rand(stream) - 1) * cos(pi / 100) * 0.8;
u1 = rand(stream, 3, 2);
u2 = rand(stream, 3, 2);
u3 = rand(stream, 3, 2);
expected = initial + randn(stream, 3, 2);
soft = initial(1, :) + factor * (1 + u2 * 9);
attach = u1 < sqrt(0.1);
expected(attach) = soft(attach);
target = repmat(initial(1, :), 3, 1);
hard = 2 * u3 - 1 < [0; 1; 2] / sqrt(5);
expected(hard) = target(hard);
assert(all(expected(:) > 0));
expected(2:3, :) = expected(2:3, :) * cos(0.1) - initial(1, :) * sin(0.1);
expected = min(10, max(1, expected));
rime_minimize(@objective, [1 10; 1 10], 'method', 'qgrime', ...
    'n_agents', 3, 'max_iter', 10, 'seed', seed, 'theta', 0.1);
verifyEqual(testCase, calls(4:6, :), expected, 'AbsTol', 1e-12);
    function value = objective(x)
        calls(end + 1, :) = x;
        value = min(size(calls, 1) - 1, 3);
    end
end

function testAcrimeChaosAndTieRules(testCase)
calls = [];
state = [];
result = rime_minimize(@objective, [0 10; 0 10], 'method', 'acrime', ...
    'n_agents', 2, 'max_iter', 1, 'seed', 0, 'callback', @callback);
chaos = 2 * calls(1:2, :) / 10 - 1;
verifyEqual(testCase, chaos(2, :), 4 * chaos(1, :).^3 - 3 * chaos(1, :), 'AbsTol', 1e-12);
verifyEqual(testCase, result.n_evals, 11); % 2 initialization + 4 phases: 2+2+1+4.
verifyEqual(testCase, result.x, calls(7, :)); % Eq. 19 accepts the mutation tie.
verifyEqual(testCase, state.positions, calls(3:4, :)); % Restart ties retain mutualism rows.
verifyEqual(testCase, state.TR, zeros(2, 1));
    function value = objective(x)
        calls(end + 1, :) = x;
        value = 1;
    end
    function callback(value)
        state = value;
    end
end

function testHerimeSingularFusionAndSizes(testCase)
calls = [];
state = [];
result = rime_minimize(@objective, repmat([-1 1], 3, 1), 'method', 'herime', ...
    'n_agents', 4, 'max_iter', 1, 'seed', 0, 'n_eda', 4, 'callback', @callback);
verifyEqual(testCase, result.n_evals, 12);
verifyEqual(testCase, state.positions, calls(1:4, :)); % Stable union retains old rows on ties.
verifyTrue(testCase, all(isfinite(calls(:)))); % A 2-member dominant set has singular 3-D covariance.
for count = [1 7]
    result = rime_minimize(@(x) sum(x.^2), [-1 1], 'method', 'herime', ...
        'n_agents', 7, 'max_iter', 3, 'seed', 0, 'n_eda', count);
    verifyEqual(testCase, result.n_evals, 28 + 3 * count);
end
    function value = objective(x)
        calls(end + 1, :) = x;
        value = 0;
    end
    function callback(value)
        state = value;
    end
end
