import math
from collections.abc import Callable
from functools import partial
from hashlib import sha256
from typing import Any, NamedTuple

import numpy as np

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    numba_funcify_and_cache_key,
    register_funcify_and_cache_key,
)
from pytensor.link.numba.dispatch.optimize.utils import (
    _SOLVER_JIT_OPTIONS,
    _make_value_at,
    _scalar_types,
    _with_input_dtype,
)
from pytensor.tensor.optimize import RootScalarOp


def make_newton(objective, point, value, mixed):
    @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
    def newton(x0, args, xtol, maxiter):
        # The Newton branch of scipy's newton
        tolerance = point(xtol)
        x = following = x0
        for _ in range(maxiter):
            f, slope = objective(x, *args)
            f, slope = f.item(), slope.item()
            if f == 0.0:
                return x, True
            if slope == 0.0:
                return x, False
            following = x - f / slope
            if abs(following - x) <= tolerance:
                return following, True
            x = following
        return following, False

    return newton


def make_halley(objective, point, value, mixed):
    @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
    def halley(x0, args, xtol, maxiter):
        # The Halley branch of scipy's newton
        tolerance = point(xtol)
        x = following = x0
        for _ in range(maxiter):
            f, slope, curvature = objective(x, *args)
            f, slope, curvature = f.item(), slope.item(), curvature.item()
            if f == 0.0:
                return x, True
            if slope == 0.0:
                return x, False
            step = f / slope
            correction = step * curvature / slope / value(2)
            if abs(correction) < 1:
                step /= value(1.0) - correction
            following = x - step
            if abs(following - x) <= tolerance:
                return following, True
            x = following
        return following, False

    return halley


def make_secant(objective, point, value, mixed):
    value_at = _make_value_at(objective=objective)

    @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
    def secant(x0, args, xtol, maxiter):
        # The secant branch of scipy's newton
        tolerance = point(xtol)
        previous = x0
        current = x0 * point(1 + 1e-4)
        current += point(1e-4) if current >= 0 else point(-1e-4)
        previous_value = value_at(previous, args)
        current_value = value_at(current, args)
        if abs(current_value) < abs(previous_value):
            previous, current = current, previous
            previous_value, current_value = current_value, previous_value

        following = current
        for _ in range(maxiter):
            if current_value == previous_value:
                return (current + previous) / mixed(2.0), False
            if abs(current_value) > abs(previous_value):
                ratio = previous_value / current_value
                following = (-ratio * current + previous) / (value(1) - ratio)
            else:
                ratio = current_value / previous_value
                following = (-ratio * previous + current) / (value(1) - ratio)
            if abs(following - current) <= tolerance:
                return following, True
            previous, previous_value = current, current_value
            current = following
            current_value = value_at(current, args)
        return following, False

    return secant


@numba_basic.numba_njit(inline="always", **_SOLVER_JIT_OPTIONS)
def _same_sign(a, b):
    # The signbit comparison scipy's C solvers use, which cannot underflow
    return math.copysign(1.0, a) == math.copysign(1.0, b)


@numba_basic.numba_njit(inline="always", **_SOLVER_JIT_OPTIONS)
def _check_bracket(lower, upper, lower_value, upper_value):
    # The checks scipy's C solvers run on a bracket's ends before iterating
    if np.isnan(lower_value):
        return True, lower, False
    if np.isnan(upper_value):
        return True, upper, False
    if lower_value == 0.0:
        return True, lower, True
    if upper_value == 0.0:
        return True, upper, True
    if _same_sign(lower_value, upper_value):
        return True, np.nan, False
    return False, np.nan, False


def make_bisect(objective):
    value_at = _make_value_at(objective=objective)

    @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
    def bisect(lower, upper, args, xtol, rtol, maxiter):
        lower_value = value_at(lower, args)
        upper_value = value_at(upper, args)
        done, root, converged = _check_bracket(lower, upper, lower_value, upper_value)
        if done:
            return root, converged
        # As in scipy's bisect.c: only the lower end and the width are tracked
        width = upper - lower
        for _ in range(maxiter):
            width /= 2
            middle = lower + width
            middle_value = value_at(middle, args)
            if np.isnan(middle_value):
                return middle, False
            if _same_sign(middle_value, lower_value):
                lower = middle
            if middle_value == 0.0 or abs(width) < xtol + rtol * abs(middle):
                return middle, True
        return lower, False

    return bisect


@numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
def _brent_step(
    points, values, previous_step, current_step, tolerance, bisection, hyperbolic
):
    # An interpolation step when it is acceptable, a bisection step otherwise. With
    # hyperbolic, brenth's extrapolation replaces brentq's inverse quadratic one.
    previous, current, block = points
    previous_value, current_value, block_value = values
    if abs(previous_step) <= tolerance or abs(current_value) >= abs(previous_value):
        return bisection, bisection

    if previous == block:
        trial = -current_value * (current - previous) / (current_value - previous_value)
    else:
        previous_slope = (previous_value - current_value) / (previous - current)
        block_slope = (block_value - current_value) / (block - current)
        if hyperbolic:
            trial = (
                -current_value
                * (block_value - previous_value)
                / (block_value * previous_slope - previous_value * block_slope)
            )
        else:
            trial = (
                -current_value
                * (block_value * block_slope - previous_value * previous_slope)
                / (block_slope * previous_slope * (block_value - previous_value))
            )
    if 2 * abs(trial) < min(abs(previous_step), 3 * abs(bisection) - tolerance):
        return current_step, trial
    return bisection, bisection


def make_brent(objective, hyperbolic):
    value_at = _make_value_at(objective=objective)

    @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
    def brent(lower, upper, args, xtol, rtol, maxiter):
        # Brent's method as in scipy's brentq.c and brenth.c: interpolation or
        # extrapolation when it stays inside the bracket and shrinks fast enough,
        # bisection otherwise.
        previous, current = lower, upper
        previous_value, current_value = (
            value_at(previous, args),
            value_at(current, args),
        )
        done, root, converged = _check_bracket(
            previous, current, previous_value, current_value
        )
        if done:
            return root, converged

        block, block_value = previous, previous_value
        previous_step = current_step = 0.0
        for _ in range(maxiter):
            if (
                previous_value != 0.0
                and current_value != 0.0
                and not _same_sign(previous_value, current_value)
            ):
                block, block_value = previous, previous_value
                previous_step = current_step = current - previous
            if abs(block_value) < abs(current_value):
                previous, current, block = current, block, current
                previous_value, current_value, block_value = (
                    current_value,
                    block_value,
                    current_value,
                )

            tolerance = (xtol + rtol * abs(current)) / 2
            bisection = (block - current) / 2
            if current_value == 0.0 or abs(bisection) < tolerance:
                return current, True

            previous_step, current_step = _brent_step(
                (previous, current, block),
                (previous_value, current_value, block_value),
                previous_step,
                current_step,
                tolerance,
                bisection,
                hyperbolic,
            )

            previous, previous_value = current, current_value
            if abs(current_step) > tolerance:
                current += current_step
            else:
                current += tolerance if bisection > 0 else -tolerance
            current_value = value_at(current, args)
            if np.isnan(current_value):
                return current, False
        return current, False

    return brent


def make_ridder(objective):
    value_at = _make_value_at(objective=objective)

    @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
    def ridder(lower, upper, args, xtol, rtol, maxiter):
        # Ridders' method as in scipy's ridder.c
        tolerance = xtol + rtol * min(abs(lower), abs(upper))
        lower_value = value_at(lower, args)
        upper_value = value_at(upper, args)
        done, root, converged = _check_bracket(lower, upper, lower_value, upper_value)
        if done:
            return root, converged

        estimate = 0.0
        for _ in range(maxiter):
            half_width = (upper - lower) / 2
            middle = lower + half_width
            middle_value = value_at(middle, args)
            if np.isnan(middle_value):
                return middle, False
            if middle_value == 0.0:
                return middle, True

            ratio = middle_value / lower_value
            step = (
                half_width * ratio / np.sqrt(ratio * ratio - upper_value / lower_value)
            )
            step_sign = 1.0 if step > 0 else -1.0
            # C's MIN macro, which returns the limit when the step is NaN
            limit = abs(half_width) - tolerance / 2
            offset = abs(step) if abs(step) < limit else limit
            estimate = middle + step_sign * offset
            estimate_value = value_at(estimate, args)
            if np.isnan(estimate_value):
                return estimate, False

            if not _same_sign(estimate_value, middle_value):
                lower, lower_value = estimate, estimate_value
                upper, upper_value = middle, middle_value
            elif not _same_sign(estimate_value, lower_value):
                upper, upper_value = estimate, estimate_value
            else:
                lower, lower_value = estimate, estimate_value

            # ridder.c drops the fabs it uses for the initial tolerance, so negative
            # roots far from zero only converge on an exact zero. The port keeps this so
            # both backends agree.
            tolerance = xtol + rtol * estimate
            if estimate_value == 0.0 or abs(upper - lower) < tolerance:
                return estimate, True
        return estimate, False

    return ridder


@numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
def _inverse_cubic_zero(xs, fs, q, d):
    # scipy's _interpolated_poly at zero: Neville's algorithm on the cubic through
    # (fs[i], xs[i]), with q and d as 4x4 work buffers
    q[:, 0] = xs
    d[:, 0] = xs
    for k in range(1, 4):
        for i in range(k, 4):
            alpha = d[i, k - 1] - q[i - 1, k - 1]
            difference = fs[i - k] - fs[i]
            q[i, k] = fs[i] / difference * alpha
            d[i, k] = fs[i - k] / difference * alpha
    return q[3, 1] + q[3, 2] + q[3, 3] + q[3, 0]


@numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
def _newton_quadratic(ab, fab, d, fd, k):
    # scipy's _newton_quadratic: k Newton steps on the quadratic through (a, fa),
    # (b, fb) and (d, fd)
    a, b = ab[0], ab[1]
    fa, fb = fab[0], fab[1]
    slope = (fb - fa) / (b - a)
    curvature = ((fd - fb) / (d - b) - slope) / (d - a)
    if curvature == 0:
        return a - fa / slope

    r = a if np.sign(curvature) * np.sign(fa) > 0 else b
    for _ in range(k):
        value = (curvature * (r - b) + slope) * (r - a) + fa
        r1 = r - value / (slope + curvature * (2 * r - a - b))
        if not (a < r1 < b):
            if a < r < b:
                return r
            return (a + b) / 2.0
        r = r1
    return r


@numba_basic.numba_njit(inline="always", **_SOLVER_JIT_OPTIONS)
def _update_bracket(ab, fab, c, fc):
    # scipy's _update_bracket: c replaces one end, which is returned
    index = 0 if np.sign(fab[0]) * np.sign(fc) > 0 else 1
    discarded, discarded_value = ab[index], fab[index]
    ab[index], fab[index] = c, fc
    return discarded, discarded_value


@numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
def _separated(values, atol):
    # scipy's _notclose: nonzero, finite, and pairwise further than atol apart
    for i in range(values.shape[0]):
        if values[i] == 0 or not np.isfinite(values[i]):
            return False
        for j in range(i + 1, values.shape[0]):
            if abs(values[i] - values[j]) <= atol:
                return False
    return True


@numba_basic.numba_njit(inline="always", **_SOLVER_JIT_OPTIONS)
def _toms748_failure(x, fx):
    # A NaN fails at that point. scipy raises on an infinite value, which perform
    # reports as NaN.
    return x if np.isnan(fx) else np.nan


def make_toms748(objective):
    value_at = _make_value_at(objective=objective)

    @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
    def toms748(lower, upper, args, xtol, rtol, maxiter):
        # scipy's toms748 with its default k=1, checking each end as TOMS748Solver.start
        # does before it evaluates the next
        if not (np.isfinite(lower) and np.isfinite(upper)) or lower >= upper:
            return np.nan, False

        eps = np.finfo(np.float64).eps
        k = 1
        lower_value = value_at(lower, args)
        if not np.isfinite(lower_value):
            return _toms748_failure(lower, lower_value), False
        if lower_value == 0:
            return lower, True
        upper_value = value_at(upper, args)
        if not np.isfinite(upper_value):
            return _toms748_failure(upper, upper_value), False
        if upper_value == 0:
            return upper, True
        if np.sign(lower_value) * np.sign(upper_value) > 0:
            return np.nan, False

        ab = np.array([lower, upper])
        fab = np.array([lower_value, upper_value])

        if fab[0] == fab[1]:
            c = np.nan
        elif abs(fab[1]) > abs(fab[0]):
            c = (-fab[0] / fab[1] * ab[1] + ab[0]) / (1 - fab[0] / fab[1])
        else:
            c = (-fab[1] / fab[0] * ab[0] + ab[1]) / (1 - fab[1] / fab[0])
        if not ab[0] < c < ab[1]:
            c = (ab[0] + ab[1]) / 2.0
        fc = value_at(c, args)
        if not np.isfinite(fc):
            return _toms748_failure(c, fc), False
        if fc == 0:
            return c, True

        d, fd = _update_bracket(ab, fab, c, fc)
        e = fe = 0.0
        has_e = False
        iterations = 1
        points, values = np.empty(4), np.empty(4)
        neville_q, neville_d = np.empty((4, 4)), np.empty((4, 4))

        while True:
            iterations += 1
            ab_width = ab[1] - ab[0]
            has_c = False
            for steps in range(2, k + 2):
                if has_e:
                    values[0], values[1], values[2], values[3] = fab[0], fab[1], fd, fe
                    if _separated(values, 32 * eps):
                        points[0], points[1], points[2], points[3] = ab[0], ab[1], d, e
                        c0 = _inverse_cubic_zero(points, values, neville_q, neville_d)
                        if ab[0] < c0 < ab[1]:
                            c, has_c = c0, True
                if not has_c:
                    c, has_c = _newton_quadratic(ab, fab, d, fd, steps), True

                fc = value_at(c, args)
                if not np.isfinite(fc):
                    return _toms748_failure(c, fc), False
                if fc == 0:
                    return c, True
                e, fe, has_e = d, fd, True
                d, fd = _update_bracket(ab, fab, c, fc)

            u_index = 0 if abs(fab[0]) < abs(fab[1]) else 1
            u, fu = ab[u_index], fab[u_index]
            slope = (fab[1] - fab[0]) / (ab[1] - ab[0])
            c = u - 2 * fu / slope
            if abs(c - u) > 0.5 * (ab[1] - ab[0]):
                c = (ab[0] + ab[1]) / 2.0
            elif abs(c - u) <= eps * abs(u):
                if math.frexp(fab[u_index])[1] < math.frexp(fab[1 - u_index])[1] - 50:
                    c = (31 * ab[u_index] + ab[1 - u_index]) / 32
                else:
                    direction = 1 if u_index == 0 else -1
                    c = u + (direction * abs(c) * rtol + direction * xtol)
                if not ab[0] < c < ab[1]:
                    c = (ab[0] + ab[1]) / 2.0

            fc = value_at(c, args)
            if not np.isfinite(fc):
                return _toms748_failure(c, fc), False
            if fc == 0:
                return c, True
            e, fe = d, fd
            d, fd = _update_bracket(ab, fab, c, fc)

            if ab[1] - ab[0] > 0.5 * ab_width:
                e, fe = d, fd
                z = (ab[0] + ab[1]) / 2.0
                fz = value_at(z, args)
                if not np.isfinite(fz):
                    return _toms748_failure(z, fz), False
                if fz == 0:
                    return z, True
                d, fd = _update_bracket(ab, fab, z, fz)

            if abs(ab[0] - ab[1]) <= xtol + rtol * abs(ab[1]):
                return (ab[0] + ab[1]) / 2.0, True
            if iterations >= maxiter:
                return (ab[0] + ab[1]) / 2.0, False

    return toms748


class RootMethod(NamedTuple):
    make_solver: Callable
    defaults: dict
    bracketing: bool
    # Derivative outputs the objective must carry; None accepts any
    derivatives: int | None = None
    # scipy's lower limits on the static options
    min_rtol: float = 0.0
    min_maxiter: int = 1


BRACKET_DEFAULTS = {"xtol": 2e-12, "rtol": 4 * np.finfo(np.float64).eps, "maxiter": 100}
OPEN_DEFAULTS = {"xtol": 1.48e-8, "maxiter": 50}
C_SOLVER_LIMITS: dict[str, Any] = {
    "min_rtol": BRACKET_DEFAULTS["rtol"],
    "min_maxiter": 0,
}

ROOT_METHODS = {
    "bisect": RootMethod(
        make_solver=make_bisect,
        defaults=BRACKET_DEFAULTS,
        bracketing=True,
        **C_SOLVER_LIMITS,
    ),
    "brentq": RootMethod(
        make_solver=partial(make_brent, hyperbolic=False),
        defaults=BRACKET_DEFAULTS,
        bracketing=True,
        **C_SOLVER_LIMITS,
    ),
    "brenth": RootMethod(
        make_solver=partial(make_brent, hyperbolic=True),
        defaults=BRACKET_DEFAULTS,
        bracketing=True,
        **C_SOLVER_LIMITS,
    ),
    "ridder": RootMethod(
        make_solver=make_ridder,
        defaults=BRACKET_DEFAULTS,
        bracketing=True,
        **C_SOLVER_LIMITS,
    ),
    "toms748": RootMethod(
        make_solver=make_toms748,
        defaults=BRACKET_DEFAULTS,
        bracketing=True,
        min_rtol=BRACKET_DEFAULTS["rtol"] / 4,
    ),
    "newton": RootMethod(
        make_solver=make_newton,
        defaults=OPEN_DEFAULTS,
        bracketing=False,
        derivatives=1,
    ),
    "halley": RootMethod(
        make_solver=make_halley,
        defaults=OPEN_DEFAULTS,
        bracketing=False,
        derivatives=2,
    ),
    "secant": RootMethod(
        make_solver=make_secant, defaults=OPEN_DEFAULTS, bracketing=False
    ),
}


def solver_options(op: RootScalarOp) -> dict | None:
    """Return the options ``op`` solves with, scipy's defaults filled in.

    None means the numba solvers do not implement them, or scipy rejects them, so perform runs and raises scipy's
    own error.
    """
    method = ROOT_METHODS.get(op.method)
    if method is None:
        return None
    if method.bracketing and "bracket" not in op.bound_kwargs:
        return None
    if method.derivatives is not None and op.jac + op.hess != method.derivatives:
        return None
    if op.optimizer_kwargs.keys() - method.defaults.keys():
        return None

    settings = method.defaults | op.optimizer_kwargs
    if not isinstance(settings["maxiter"], int | np.integer):
        return None
    if (
        settings["xtol"] <= 0
        or settings.get("rtol", method.min_rtol) < method.min_rtol
        or settings["maxiter"] < method.min_maxiter
    ):
        return None
    return {
        name: int(value) if name == "maxiter" else float(value)
        for name, value in settings.items()
    }


@register_funcify_and_cache_key(RootScalarOp)
def numba_funcify_RootScalarOp(op, node, **kwargs):
    settings = solver_options(op=op)
    if settings is None:
        return generate_fallback_impl(op, node, **kwargs), None

    _ = kwargs.pop("storage_map", None)
    objective, objective_key = numba_funcify_and_cache_key(
        op.fgraph, fgraph_name="numba_root_scalar_objective", **kwargs
    )
    objective = _with_input_dtype(
        objective=objective, dtype=op.fgraph.inputs[0].type.dtype
    )

    as_output_dtype = node.outputs[0].type.numpy_dtype.type
    n_args = len(op.inner_inputs) - 1
    method = ROOT_METHODS[op.method]
    if method.bracketing:
        solve = method.make_solver(objective=objective)
        bracket_index = n_args + op.bound_kwargs.index("bracket")
        xtol, rtol, maxiter = settings["xtol"], settings["rtol"], settings["maxiter"]

        @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
        def root_scalar(_x0, *inputs):
            bracket = inputs[bracket_index]
            if bracket.shape[0] < 2:
                return np.asarray(as_output_dtype(np.nan)), False
            root, converged = solve(
                np.float64(bracket[0]),
                np.float64(bracket[1]),
                inputs[:n_args],
                xtol,
                rtol,
                maxiter,
            )
            return np.asarray(as_output_dtype(root)), converged

    else:
        point, value, mixed = _scalar_types(
            point_dtype=op.fgraph.inputs[0].type.dtype,
            value_dtype=op.fgraph.outputs[0].type.dtype,
        )
        solve = method.make_solver(
            objective=objective, point=point, value=value, mixed=mixed
        )
        xtol, maxiter = settings["xtol"], settings["maxiter"]

        @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
        def root_scalar(x0, *inputs):
            root, converged = solve(x0.item(), inputs[:n_args], xtol, maxiter)
            return np.asarray(as_output_dtype(root)), converged

    if objective_key is None:
        cache_key = None
    else:
        root_scalar_cache_version = 1
        cache_key = sha256(
            str(
                (
                    type(op),
                    op.method,
                    op.bound_kwargs,
                    sorted(settings.items()),
                    objective_key,
                    root_scalar_cache_version,
                )
            ).encode()
        ).hexdigest()
    return root_scalar, cache_key
