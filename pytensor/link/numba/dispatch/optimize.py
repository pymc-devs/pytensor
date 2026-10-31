from hashlib import sha256

import numpy as np

from pytensor.link.numba.dispatch._lbfgs import _lbfgs
from pytensor.link.numba.dispatch._linesearch import CONVERGED, _dcsrch
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    numba_funcify_and_cache_key,
    numba_njit,
    register_funcify_and_cache_key,
)
from pytensor.tensor.optimize import MinimizeOp


# Bump when the solvers or the line-search helpers they share change, since
# the Numba disk cache keys do not see helper sources.
_SOLVER_CACHE_VERSION = 1


@numba_njit(fastmath=False)
def _vecnorm(x, order):
    if order == np.inf:
        return np.max(np.abs(x))
    if order == -np.inf:
        return np.min(np.abs(x))
    return np.sum(np.abs(x) ** order) ** (1.0 / order)


# Ports of SciPy's _cubicmin, _quadmin, _zoom and scalar_search_wolfe2
# (scipy/optimize/_linesearch.py). NaN stands in for SciPy's None. The numpy
# error model turns SciPy's ArithmeticError cases into non-finite results,
# which the isfinite checks reject the same way.
@numba_njit(fastmath=False, error_model="numpy")
def _cubicmin(a, fa, fpa, b, fb, c, fc):
    db = b - a
    dc = c - a
    denom = (db * dc) ** 2 * (db - dc)
    t0 = fb - fa - fpa * db
    t1 = fc - fa - fpa * dc
    cubic = (dc**2 * t0 - db**2 * t1) / denom
    quadratic = (db**3 * t1 - dc**3 * t0) / denom
    radical = quadratic * quadratic - 3 * cubic * fpa
    xmin = a + (-quadratic + np.sqrt(radical)) / (3 * cubic)
    return xmin if np.isfinite(xmin) else np.nan


@numba_njit(fastmath=False, error_model="numpy")
def _quadmin(a, fa, fpa, b, fb):
    db = b - a
    curvature = (fb - fa - fpa * db) / (db * db)
    xmin = a - fpa / (2.0 * curvature)
    return xmin if np.isfinite(xmin) else np.nan


@numba_njit(fastmath=False, inline="always")
def _zoom(
    evaluate, x, p, args, a_lo, a_hi, phi_lo, phi_hi, derphi_lo, f0, d0, g0, c1, c2
):
    phi_rec, a_rec = f0, 0.0
    # SciPy's zoom evaluates until i > maxiter, with maxiter = 10.
    for i in range(11):
        dalpha = a_hi - a_lo
        a, b = (a_hi, a_lo) if dalpha < 0 else (a_lo, a_hi)
        a_j = np.nan
        if i > 0:
            cchk = 0.2 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
            if a_j > b - cchk or a_j < a + cchk:
                a_j = np.nan
        if np.isnan(a_j):
            qchk = 0.1 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if np.isnan(a_j) or a_j > b - qchk or a_j < a + qchk:
                a_j = a_lo + 0.5 * dalpha

        f, g = evaluate(x + a_j * p, *args)
        if f > f0 + c1 * a_j * d0 or f >= phi_lo:
            phi_rec, a_rec = phi_hi, a_hi
            a_hi, phi_hi = a_j, f
        else:
            d = np.dot(g, p)
            if abs(d) <= -c2 * d0:
                return a_j, f, g, True
            if d * (a_hi - a_lo) >= 0:
                phi_rec, a_rec = phi_hi, a_hi
                a_hi, phi_hi = a_lo, phi_lo
            else:
                phi_rec, a_rec = phi_lo, a_lo
            a_lo, phi_lo, derphi_lo = a_j, f, d

    return 0.0, f0, g0, False


@numba_njit(fastmath=False, inline="always")
def _line_search_wolfe2(evaluate, x, p, args, f0, g0, d0, alpha, c1, c2):
    alpha0, alpha1 = 0.0, alpha
    phi_a0, derphi_a0 = f0, d0
    f, g = evaluate(x + alpha1 * p, *args)
    for i in range(10):
        if f > f0 + c1 * alpha1 * d0 or (f >= phi_a0 and i > 0):
            return _zoom(
                evaluate,
                x,
                p,
                args,
                alpha0,
                alpha1,
                phi_a0,
                f,
                derphi_a0,
                f0,
                d0,
                g0,
                c1,
                c2,
            )
        d = np.dot(g, p)
        if abs(d) <= -c2 * d0:
            return alpha1, f, g, True
        if d >= 0:
            return _zoom(
                evaluate, x, p, args, alpha1, alpha0, f, phi_a0, d, f0, d0, g0, c1, c2
            )
        alpha0, alpha1 = alpha1, min(2 * alpha1, 1e100)
        phi_a0, derphi_a0 = f, d
        f, g = evaluate(x + alpha1 * p, *args)

    # SciPy accepts the last trial step when the doubling phase runs out.
    return alpha1, f, g, True


@numba_njit(fastmath=False, inline="always")
def _line_search_wolfe12(evaluate, x, p, args, f0, previous_f, g0, c1, c2):
    d0 = np.dot(g0, p)
    if not np.isfinite(d0) or d0 >= 0:
        return 0.0, f0, g0, False

    alpha = min(1.0, 2.02 * (f0 - previous_f) / d0)
    # SciPy keeps a zero step, which fails both of its searches.
    if alpha <= 0 or not np.isfinite(alpha):
        alpha = 1.0
    # SciPy's line_search_wolfe12 treats DCSRCH warnings as failures.
    step, f, g, status, _ = _dcsrch(
        evaluate,
        x,
        p,
        args,
        f0,
        g0,
        d0,
        alpha,
        c1,
        c2,
        1e-14,
        1e-100,
        1e100,
        100,
        np.empty_like(x),
    )
    if status == CONVERGED:
        return step, f, g, True
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

    identity = np.eye(x.size)
    h = identity
    updated = False
    previous_f = f + np.linalg.norm(g) / 2
    # Like SciPy's k, steps counts accepted steps only. Restarts cannot run away:
    # a second failure before the next accepted update returns.
    steps = 0
    while steps < maxiter:
        p = -np.dot(h, g)
        alpha, new_f, new_g, found = _line_search_wolfe12(
            evaluate, x, p, args, f, previous_f, g, c1, c2
        )
        if not found:
            if not updated:
                return x, False
            # Like SciPy's L-BFGS-B, discard the curvature history and retry
            # with steepest descent before giving up.
            h = identity
            updated = False
            continue

        steps += 1
        s = alpha * p
        descent = -np.dot(g, s)
        x = x + s
        y = new_g - g
        previous_f, f, g = f, new_f, new_g
        if _vecnorm(g, norm) <= gtol:
            return x, steps < maxiter
        if np.linalg.norm(s) <= xrtol * (xrtol + np.linalg.norm(x)):
            return x, steps < maxiter
        if not np.isfinite(f) or not np.all(np.isfinite(g)):
            return x, False

        curvature = np.dot(y, s)
        if curvature <= np.finfo(np.float64).eps * descent:
            continue
        if not updated:
            # Scale the identity to the observed curvature (Nocedal & Wright
            # eq. 6.20). Otherwise the update must cancel O(1) entries down to
            # 1 / curvature, and rounding can leave h indefinite.
            h = identity * (curvature / np.dot(y, y))
            updated = True
        # Expanded form of SciPy's (I - rho s y') h (I - rho y s') + rho s s',
        # which costs O(n^2) instead of two n x n matrix products. It updates h
        # in place, which is safe because h was rebound to a fresh scaled array
        # when updated became True.
        rho = 1.0 / curvature
        hy = np.dot(h, y)
        w = (0.5 * rho * rho * (curvature + np.dot(y, hy))) * s - rho * hy
        for i in range(x.size):
            for j in range(x.size):
                h[i, j] = (h[i, j] + s[i] * w[j]) + w[i] * s[j]

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


def _compile_objective(op, **kwargs):
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
        # astype always copies. The gradient output can be x itself
        # (0.5 * sum(x**2)), so the copy guarantees the solvers never hold a
        # gradient that aliases the x they update in place.
        return value.item(), gradient.astype(np.float64).reshape((-1,))

    return evaluate, inner_key


def _solver_cache_key(op, inner_key, solver, *settings):
    if inner_key is None:
        return None
    x = op.inner_inputs[0]
    return sha256(
        str(
            (
                type(op),
                _SOLVER_CACHE_VERSION,
                solver,
                inner_key,
                x.ndim == 0,
                x.dtype,
                *settings,
            )
        ).encode()
    ).hexdigest()


def _funcify_bfgs(op, **kwargs):
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

    evaluate, inner_key = _compile_objective(op, **kwargs)
    x_dtype = op.inner_inputs[0].dtype

    @numba_njit(fastmath=False)
    def minimize_bfgs(x0, *args):
        x = x0.ravel().astype(np.float64)
        solution, success = _bfgs(evaluate, x, args, gtol, maxiter, norm, xrtol, c1, c2)
        return solution.reshape(x0.shape).astype(x_dtype), success

    cache_key = _solver_cache_key(
        op, inner_key, "bfgs", gtol, maxiter, norm, xrtol, c1, c2
    )
    return minimize_bfgs, cache_key


def _funcify_lbfgsb(op, **kwargs):
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
    if maxcor <= 0:
        raise ValueError("'maxcor' must be positive.")
    if maxls <= 0:
        raise ValueError("'maxls' must be positive.")
    if ftol < 0:
        raise ValueError("'ftol' must be non-negative.")
    if gtol < 0:
        raise ValueError("'gtol' must be non-negative.")

    evaluate, inner_key = _compile_objective(op, **kwargs)
    x_dtype = op.inner_inputs[0].dtype

    @numba_njit(fastmath=False)
    def minimize_lbfgsb(x0, *args):
        x = x0.ravel().astype(np.float64)
        solution, success = _lbfgs(
            evaluate, x, args, maxcor, ftol, gtol, maxiter, maxfun, maxls
        )
        return solution.reshape(x0.shape).astype(x_dtype), success

    cache_key = _solver_cache_key(
        op, inner_key, "lbfgsb", maxcor, ftol, gtol, maxiter, maxfun, maxls
    )
    return minimize_lbfgsb, cache_key


@register_funcify_and_cache_key(MinimizeOp)
def numba_funcify_MinimizeOp(op, node, **kwargs):
    if _can_use_native_bfgs(op):
        return _funcify_bfgs(op, **kwargs)
    if _can_use_native_lbfgsb(op):
        return _funcify_lbfgsb(op, **kwargs)
    return generate_fallback_impl(op, node, **kwargs), None
