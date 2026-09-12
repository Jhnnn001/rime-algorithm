function result = rime_minimize(objective, bounds, varargin)
%RIME_MINIMIZE Minimize a scalar objective inside a finite box using RIME variants.
%   RESULT = rime_minimize(OBJECTIVE, BOUNDS, Name, Value, ...) takes a
%   function handle accepting a 1-by-dim row and a dim-by-2 bounds matrix.
%
%   Options: method ('rime', 'acrime', 'qgrime', 'herime'), n_agents (30),
%   max_iter (500), seed ([] for a new stream, integer, or RandStream),
%   callback ([] or a function handle called with a state struct per iteration).
%   QGRIME: theta (0.015*pi). HERIME: lam (0.5), n_eda (floor(n_agents/2)).
%
%   RESULT contains x, fun, history (including initialization), n_iter,
%   n_evals, and method. NaN fitness becomes Inf; exceptions propagate.
%   All random draws use a local stream. No extra MATLAB toolboxes are needed.
%   See README.md for the API contract and the paper interpretations.
%
%   RIME:   doi:10.1016/j.neucom.2023.02.010
%   ACRIME: doi:10.1016/j.compbiomed.2024.108803
%   QGRIME: doi:10.1093/jcde/qwaf016
%   HERIME: doi:10.3390/biomimetics10010014

if ~isa(objective, 'function_handle')
    error('rimeopt:InvalidObjective', 'objective must be a function handle.');
end
if ~isnumeric(bounds) || ~isreal(bounds) || ~ismatrix(bounds) || isempty(bounds) || size(bounds, 2) ~= 2
    error('rimeopt:InvalidBounds', 'bounds must be a real dim-by-2 numeric matrix.');
end
bounds = double(bounds);
lb = bounds(:, 1)';
ub = bounds(:, 2)';
width = ub - lb;
if any(~isfinite(bounds(:))) || any(~isfinite(width) | width <= 0)
    error('rimeopt:InvalidBounds', 'bounds must have finite endpoints and finite positive widths.');
end
p = inputParser;
p.FunctionName = 'rime_minimize';
p.PartialMatching = false;
p.StructExpand = false;
addParameter(p, 'method', 'rime');
addParameter(p, 'n_agents', 30);
addParameter(p, 'max_iter', 500);
addParameter(p, 'seed', []);
addParameter(p, 'callback', []);
addParameter(p, 'theta', 0.015 * pi);
addParameter(p, 'lam', 0.5);
addParameter(p, 'n_eda', []);
parse(p, varargin{:});
options = p.Results;
if ~(ischar(options.method) && isrow(options.method)) && ~(isstring(options.method) && isscalar(options.method))
    error('rimeopt:InvalidMethod', 'method must be rime, acrime, qgrime, or herime.');
end
method = lower(char(options.method));
if ~ismember(method, {'rime', 'acrime', 'qgrime', 'herime'})
    error('rimeopt:InvalidMethod', 'method must be rime, acrime, qgrime, or herime.');
end
n = integer_option(options.n_agents, 2, Inf, 'n_agents');
T = integer_option(options.max_iter, 1, Inf, 'max_iter');
d = size(bounds, 1);
callback = options.callback;
if ~isempty(callback) && ~isa(callback, 'function_handle')
    error('rimeopt:InvalidCallback', 'callback must be empty or a function handle.');
end
if ~strcmp(method, 'qgrime') && ~ismember('theta', p.UsingDefaults)
    error('rimeopt:InvalidOption', 'theta is only available for qgrime.');
end
if ~strcmp(method, 'herime') && (~ismember('lam', p.UsingDefaults) || ~ismember('n_eda', p.UsingDefaults))
    error('rimeopt:InvalidOption', 'lam and n_eda are only available for herime.');
end
theta = options.theta;
if ~isnumeric(theta) || ~isreal(theta) || ~isscalar(theta) || ~isfinite(theta) || theta <= 0
    error('rimeopt:InvalidOption', 'theta must be finite and > 0.');
end
lam = options.lam;
if ~isnumeric(lam) || ~isreal(lam) || ~isscalar(lam) || ~isfinite(lam) || lam < 0 || lam > 1
    error('rimeopt:InvalidOption', 'lam must be finite and in [0, 1].');
end
n_eda = options.n_eda;
if isempty(n_eda)
    n_eda = floor(n / 2);
end
if strcmp(method, 'herime')
    if n < 4
        error('rimeopt:InvalidOption', 'HERIME requires n_agents >= 4.');
    end
    n_eda = integer_option(n_eda, 1, n, 'n_eda');
end
if isa(options.seed, 'RandStream')
    stream = options.seed;
elseif isempty(options.seed)
    stream = RandStream('mt19937ar', 'Seed', 'shuffle');
else
    seed = integer_option(options.seed, 0, 2^32 - 1, 'seed');
    stream = RandStream('mt19937ar', 'Seed', seed);
end

n_evals = 0;
best_f = Inf;
TR = zeros(n, 1);
if strcmp(method, 'acrime')
    positions = zeros(n, d);
    for j = 1:d
        s = 2 * rand(stream) - 1;
        for agent = 1:n
            if agent > 1
                s = min(1, max(-1, 4 * s^3 - 3 * s));
            end
            while s <= -1 || s >= 1 || s == 0
                s = 2 * rand(stream) - 1;
            end
            positions(agent, j) = lb(j) + width(j) * (s + 1) / 2;
        end
    end
else
    positions = lb + rand(stream, n, d) .* width;
end
positions = clip(positions);
best_x = positions(1, :);
fitness = evaluate(positions);
history = zeros(T + 1, 1);
history(1) = best_f;
for t = 1:T
    if strcmp(method, 'acrime')
        acrime_step(t);
    elseif strcmp(method, 'herime')
        herime_step(t);
    else
        rime_step(t);
    end
    history(t + 1) = best_f;
    if ~isempty(callback)
        state = struct('positions', positions, 'fitness', fitness, 'best_x', best_x, ...
            'best_f', best_f, 'iter', t, 'n_evals', n_evals, 'bounds', bounds, ...
            'dim', d, 'n_agents', n, 'max_iter', T, 'method', method, 'rng', stream);
        if strcmp(method, 'acrime')
            state.TR = TR;
        end
        callback(state);
    end
end
result = struct('x', best_x, 'fun', best_f, 'history', history, ...
    'n_iter', T, 'n_evals', n_evals, 'method', method);

    function values = evaluate(candidates)
        values = zeros(size(candidates, 1), 1);
        for row = 1:size(candidates, 1)
            n_evals = n_evals + 1;
            value = objective(candidates(row, :));
            if ~(isnumeric(value) || islogical(value)) || ~isscalar(value) || ~isreal(value)
                error('rimeopt:InvalidValue', 'objective must return a real numeric scalar.');
            end
            value = double(value);
            if isnan(value)
                value = Inf;
            end
            values(row) = value;
            if value < best_f
                best_f = value;
                best_x = candidates(row, :);
            end
        end
    end

    function x = clip(x)
        x = min(ub, max(lb, x));
    end

    function [factor, q] = rime_parameters(t)
        factor = (2 * rand(stream) - 1) * cos(pi * t / (10 * T)) * (1 - floor(5 * t / T + 0.5) / 5);
        length_f = norm(fitness);
        if length_f == 0 || ~isfinite(length_f)
            q = zeros(n, 1);
        else
            q = fitness / length_f;
        end
    end

    function rime_step(t)
        [factor, q] = rime_parameters(t);
        u1 = rand(stream, n, d);
        u2 = rand(stream, n, d);
        u3 = rand(stream, n, d);
        candidates = positions;
        if strcmp(method, 'qgrime')
            candidates = candidates + randn(stream, n, d);
        end
        soft = best_x + factor * (lb + u2 .* width);
        attach = u1 < sqrt(t / T);
        candidates(attach) = soft(attach);
        hard = 2 * u3 - 1 < q;
        target = repmat(best_x, n, 1);
        candidates(hard) = target(hard);
        if strcmp(method, 'qgrime')
            signs = sign(candidates) .* sign(best_x);
            worse = fitness > best_f;
            better = fitness < best_f;
            angles = zeros(n, d);
            angles((worse & signs > 0) | (better & signs < 0)) = theta;
            angles((worse & signs < 0) | (better & signs > 0)) = -theta;
            random_sign = (worse & candidates == 0) | (better & best_x == 0 & candidates ~= 0);
            angles(random_sign) = theta * (2 * (rand(stream, nnz(random_sign), 1) < 0.5) - 1);
            candidates = candidates .* cos(angles) - best_x .* sin(angles);
        end
        candidates = clip(candidates);
        values = evaluate(candidates);
        improved = values < fitness;
        positions(improved, :) = candidates(improved, :);
        fitness(improved) = values(improved);
    end

    function acrime_step(t)
        previous = fitness;
        [~, order] = sort(fitness);
        guide = (positions(order(1), :) + positions(order(2), :)) / 2;
        for i = 1:n
            z = randi(stream, n);
            while z == i
                z = randi(stream, n);
            end
            bf1 = 1 + rand(stream);
            bf2 = 3 - bf1;
            xi = positions(i, :);
            xz = positions(z, :);
            mutual = (xi + xz) / 2;
            if rand(stream) < 0.5
                target = guide;
            else
                target = positions(randi(stream, n), :);
            end
            positions(i, :) = xi + rand(stream, 1, d) .* (target - mutual * bf1);
            positions(z, :) = xz + rand(stream, 1, d) .* (target - mutual * bf2);
        end
        positions = clip(positions);
        fitness = evaluate(positions);
        rime_step(t);
        l2 = (t / T)^2;
        cauchy = tan(pi * (rand(stream, 1, d) - 0.5));
        mutated = clip(best_x .* (1 + (1 - l2) * cauchy + l2 * randn(stream, 1, d)));
        old_best = best_f;
        value = evaluate(mutated);
        if value <= old_best
            best_x = mutated;
            best_f = value;
        end
        unchanged = fitness == previous;
        TR(unchanged) = TR(unchanged) + 1;
        TR(~unchanged) = 0;
        for i = find(TR >= sqrt(t))'
            trial1 = lb + rand(stream, 1, d) .* width;
            trial2 = rand(stream, 1, d) .* (ub + lb) - positions(i, :);
            repair = trial2 <= lb | trial2 >= ub;
            trial2(repair) = lb(repair) + rand(stream, 1, nnz(repair)) .* width(repair);
            values = evaluate([trial1; trial2]);
            if values(1) < values(2)
                positions(i, :) = trial1;
                fitness(i) = values(1);
            elseif values(2) < values(1)
                positions(i, :) = trial2;
                fitness(i) = values(2);
            end
            TR(i) = 0;
        end
    end

    function herime_step(t)
        distance = sqrt(sum((positions - best_x).^2, 2));
        nf = zeros(n, 1);
        finite = isfinite(fitness);
        if any(finite)
            low = min(fitness(finite));
            high = max(fitness(finite));
            if high > low
                nf(finite) = (high - fitness(finite)) / (high - low);
            end
        end
        nf(fitness == -Inf) = 1;
        nd = zeros(n, 1);
        if max(distance) > 0
            nd = distance / max(distance);
        end
        score = (1 - lam) * nf + lam * nd;
        if sum(score) > 0
            cumulative = cumsum(score / sum(score));
            cumulative(end) = 1;
            k = find(rand(stream) < cumulative, 1);
        else
            k = randi(stream, n);
        end
        reference = positions(k, :);
        [factor, q] = rime_parameters(t);
        for i = 1:n
            u1 = rand(stream, 1, d);
            u2 = rand(stream, 1, d);
            u3 = rand(stream, 1, d);
            candidate = positions(i, :);
            soft = best_x + factor * (lb + u2 .* width);
            attach = u1 < sqrt(t / T);
            candidate(attach) = soft(attach);
            hard = 2 * u3 - 1 < q(i);
            candidate(hard) = reference(hard);
            candidate = clip(candidate);
            value = evaluate(candidate);
            if value < fitness(i)
                positions(i, :) = candidate;
                fitness(i) = value;
            end
        end
        [fitness, order] = sort(fitness);
        positions = positions(order, :);
        m = floor(n / 2);
        weights = log(m + 0.5) - log(1:m);
        weights = weights / sum(weights);
        mean_x = weights * positions(1:m, :);
        delta = positions(1:m, :) - mean_x;
        covariance = (delta' * delta) / m; % Eq. 10 uses unweighted covariance.
        [u, s, ~] = svd(covariance);
        gaussian = randn(stream, n_eda, d) * sqrt(s) * u' + mean_x;
        candidates = clip(gaussian + rand(stream, n_eda, 1) .* (mean_x - positions(1:n_eda, :)));
        values = evaluate(candidates);
        union_x = [positions; candidates];
        [union_f, order] = sort([fitness; values]);
        positions = union_x(order(1:n), :);
        fitness = union_f(1:n);
    end
end

function value = integer_option(value, minimum, maximum, name)
if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ~isfinite(value) || value ~= floor(value) || value < minimum || value > maximum
    error('rimeopt:InvalidInteger', '%s must be an integer in [%g, %g].', name, minimum, maximum);
end
value = double(value);
end
