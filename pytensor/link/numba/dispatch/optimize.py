from hashlib import sha256

import numpy as np

from pytensor.link.numba.dispatch._linesearch import _more_thuente
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    numba_funcify_and_cache_key,
    numba_njit,
    register_funcify_and_cache_key,
)
from pytensor.tensor.optimize import MinimizeOp


@numba_njit(fastmath=False)
def _vecnorm(x, order):
    if order == np.inf:
        return np.max(np.abs(x))
    if order == -np.inf:
        return np.min(np.abs(x))
    return np.sum(np.abs(x) ** order) ** (1.0 / order)


@numba_njit(fastmath=False, inline="always")
def _zoom(evaluate, x, p, args, lo, hi, flo, fhi, dlo, f0, d0, g0, c1, c2):
    for _ in range(50):
        delta = hi - lo
        denominator = 2 * (fhi - flo - dlo * delta)
        alpha = lo + 0.5 * delta
        if denominator > 0 and np.isfinite(denominator):
            trial = lo - dlo * delta * delta / denominator
            margin = 0.1 * abs(delta)
            if min(lo, hi) + margin < trial < max(lo, hi) - margin:
                alpha = trial

        f, g = evaluate(x + alpha * p, *args)
        if not np.isfinite(f) or f > f0 + c1 * alpha * d0 or f >= flo:
            hi, fhi = alpha, f
        else:
            d = np.dot(g, p)
            if abs(d) <= -c2 * d0:
                return alpha, f, g, True
            if d * (hi - lo) >= 0:
                hi, fhi = lo, flo
            lo, flo, dlo = alpha, f, d

    return 0.0, f0, g0, False


@numba_njit(fastmath=False, inline="always")
def _line_search_wolfe2(evaluate, x, p, args, f0, g0, d0, alpha, c1, c2):
    lo, flo, dlo = 0.0, f0, d0
    for iteration in range(10):
        f, g = evaluate(x + alpha * p, *args)
        if (
            not np.isfinite(f)
            or f > f0 + c1 * alpha * d0
            or (iteration > 0 and f >= flo)
        ):
            return _zoom(
                evaluate, x, p, args, lo, alpha, flo, f, dlo, f0, d0, g0, c1, c2
            )
        d = np.dot(g, p)
        if abs(d) <= -c2 * d0:
            return alpha, f, g, True
        if d >= 0:
            return _zoom(evaluate, x, p, args, alpha, lo, f, flo, d, f0, d0, g0, c1, c2)
        lo, flo, dlo = alpha, f, d
        alpha *= 2

    return 0.0, f0, g0, False


@numba_njit(fastmath=False, inline="always")
def _line_search(evaluate, x, p, args, f0, previous_f, g0, c1, c2):
    d0 = np.dot(g0, p)
    if not np.isfinite(d0) or d0 >= 0:
        return 0.0, f0, g0, False

    alpha = min(1.0, 2.02 * (f0 - previous_f) / d0)
    if alpha <= 0 or not np.isfinite(alpha):
        alpha = 1.0
    result = _more_thuente(evaluate, x, p, args, f0, g0, d0, alpha, c1, c2)
    if result[3]:
        return result
    return _line_search_wolfe2(evaluate, x, p, args, f0, g0, d0, alpha, c1, c2)


@numba_njit(fastmath=False, inline="always")
def _bfgs(evaluate, x, args, gtol, maxiter, norm, xrtol, c1, c2):
    f, g = evaluate(x, *args)
    if maxiter is None:
        maxiter = x.size * 200
    if maxiter <= 0 or not np.isfinite(f) or not np.all(np.isfinite(g)):
        return x, False
    if _vecnorm(g, norm) <= gtol:
        return x, True

    h = np.eye(x.size)
    identity = np.eye(x.size)
    previous_f = f + np.linalg.norm(g) / 2
    for iteration in range(maxiter):
        p = -np.dot(h, g)
        alpha, new_f, new_g, found = _line_search(
            evaluate, x, p, args, f, previous_f, g, c1, c2
        )
        if not found:
            return x, False

        s = alpha * p
        x = x + s
        y = new_g - g
        previous_f, f, g = f, new_f, new_g
        if _vecnorm(g, norm) <= gtol:
            return x, iteration + 1 < maxiter
        if np.linalg.norm(s) <= xrtol * (xrtol + np.linalg.norm(x)):
            return x, iteration + 1 < maxiter
        if not np.isfinite(f) or not np.all(np.isfinite(g)):
            return x, False

        curvature = np.dot(y, s)
        rho = 1000.0 if curvature == 0 else 1.0 / curvature
        # Match SciPy's factored update and multiplication order. Expanding
        # these products can lose positive definiteness through cancellation.
        left = identity - s[:, None] * y[None, :] * rho
        right = identity - y[:, None] * s[None, :] * rho
        h = np.dot(left, np.dot(h, right)) + rho * s[:, None] * s[None, :]

    return x, False


def _can_use_native_bfgs(op):
    options = op.optimizer_kwargs.get("options") or {}
    maxiter = options.get("maxiter")
    return (
        isinstance(op.method, str)
        and op.method.lower() == "bfgs"
        and op.jac
        and not (op.hess or op.hessp)
        and op.inner_inputs[0].dtype in ("float32", "float64")
        and op.optimizer_kwargs.keys() <= {"tol", "options"}
        and options.keys() <= {"gtol", "maxiter", "norm", "xrtol", "c1", "c2"}
        and options.get("norm", np.inf) != 0
        and (maxiter is None or isinstance(maxiter, int | np.integer))
    )


def _can_use_native_lbfgsb(op):
    options = op.optimizer_kwargs.get("options") or {}
    return (
        op.optimizer_kwargs.get("bounds") is None
        and isinstance(op.method, str)
        and op.method.lower() == "l-bfgs-b"
        and op.jac
        and not (op.hess or op.hessp)
        and op.inner_inputs[0].dtype in ("float32", "float64")
        and op.optimizer_kwargs.keys() <= {"tol", "options", "bounds"}
        and options.keys() <= {"ftol", "gtol", "maxcor", "maxiter", "maxfun", "maxls"}
        and all(
            isinstance(options.get(name, default), int | np.integer)
            for name, default in (
                ("maxcor", 10),
                ("maxiter", 15000),
                ("maxfun", 15000),
                ("maxls", 20),
            )
        )
    )


def numba_funcify_lbfgsb(op, **kwargs):
    from pytensor.link.numba.dispatch._lbfgs import _lbfgs

    options = op.optimizer_kwargs.get("options") or {}
    tol = op.optimizer_kwargs.get("tol")
    ftol = float(
        options.get("ftol", np.finfo(np.float64).eps * 1e7 if tol is None else tol)
    )
    gtol = float(options.get("gtol", 1e-5 if tol is None else tol))
    maxcor = int(options.get("maxcor", 10))
    maxiter = int(options.get("maxiter", 15000))
    maxfun = int(options.get("maxfun", 15000))
    maxls = int(options.get("maxls", 20))
    kwargs.pop("storage_map", None)
    inner_fn, inner_key = numba_funcify_and_cache_key(op.fgraph, **kwargs)
    x_dtype = op.inner_inputs[0].dtype
    scalar_x = op.inner_inputs[0].ndim == 0

    @numba_njit(fastmath=False)
    def evaluate(x, *args):
        inner_x = x.reshape(()) if scalar_x else x
        if x_dtype != "float64":
            inner_x = inner_x.astype(x_dtype)
        value, gradient = inner_fn(inner_x, *args)
        return value.item(), gradient.astype(np.float64).reshape((-1,))

    @numba_njit(fastmath=False)
    def minimize_lbfgsb(x0, *args):
        x = x0.flatten().astype(np.float64)
        solution, success = _lbfgs(
            evaluate, x, args, maxcor, ftol, gtol, maxiter, maxfun, maxls
        )
        return solution.reshape(x0.shape).astype(x_dtype), success

    cache_key = None
    if inner_key is not None:
        cache_key = sha256(
            str(
                (
                    type(op),
                    "lbfgsb-4-native-unbounded",
                    inner_key,
                    scalar_x,
                    x_dtype,
                    maxcor,
                    ftol,
                    gtol,
                    maxiter,
                    maxfun,
                    maxls,
                )
            ).encode()
        ).hexdigest()
    return minimize_lbfgsb, cache_key


@register_funcify_and_cache_key(MinimizeOp)
def numba_funcify_MinimizeOp(op, node, **kwargs):
    if _can_use_native_lbfgsb(op):
        return numba_funcify_lbfgsb(op, **kwargs)
    if not _can_use_native_bfgs(op):
        return generate_fallback_impl(op, node, **kwargs), None

    options = op.optimizer_kwargs.get("options") or {}
    tol = op.optimizer_kwargs.get("tol")
    gtol = float(options.get("gtol", 1e-5 if tol is None else tol))
    maxiter = options.get("maxiter")
    maxiter = None if maxiter is None else int(maxiter)
    norm = float(options.get("norm", np.inf))
    xrtol = float(options.get("xrtol", 0))
    c1 = float(options.get("c1", 1e-4))
    c2 = float(options.get("c2", 0.9))
    if not 0 < c1 < c2 < 1:
        raise ValueError("'c1' and 'c2' must satisfy '0 < c1 < c2 < 1'.")

    kwargs.pop("storage_map", None)
    inner_fn, inner_key = numba_funcify_and_cache_key(op.fgraph, **kwargs)
    x_dtype = op.inner_inputs[0].dtype
    scalar_x = op.inner_inputs[0].ndim == 0

    @numba_njit(fastmath=False)
    def evaluate(x, *args):
        inner_x = x.reshape(()) if scalar_x else x
        if x_dtype != "float64":
            inner_x = inner_x.astype(x_dtype)
        value, gradient = inner_fn(inner_x, *args)
        return value.item(), gradient.astype(np.float64).reshape((-1,))

    @numba_njit(fastmath=False)
    def minimize_bfgs(x0, *args):
        x = x0.flatten().astype(np.float64)
        solution, success = _bfgs(evaluate, x, args, gtol, maxiter, norm, xrtol, c1, c2)
        return solution.reshape(x0.shape).astype(x_dtype), success

    cache_key = None
    if inner_key is not None:
        cache_key = sha256(
            str(
                (
                    type(op),
                    6,  # Cache version, including the solver and line search.
                    inner_key,
                    scalar_x,
                    x_dtype,
                    gtol,
                    maxiter,
                    norm,
                    xrtol,
                    c1,
                    c2,
                )
            ).encode()
        ).hexdigest()
    return minimize_bfgs, cache_key
