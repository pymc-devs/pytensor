from hashlib import sha256

import numpy as np

from pytensor.graph.fg import FunctionGraph
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    numba_funcify_and_cache_key,
    numba_njit,
    register_funcify_and_cache_key,
)
from pytensor.link.numba.dispatch.optimize.minimize.bfgs import _bfgs
from pytensor.link.numba.dispatch.optimize.minimize.lbfgs import _lbfgs
from pytensor.link.numba.dispatch.optimize.minimize.nelder_mead import _nelder_mead
from pytensor.tensor.optimize import MinimizeOp


# Bump when the solvers or the line-search helpers they share change, since
# the Numba disk cache keys do not see helper sources.
_SOLVER_CACHE_VERSION = 2


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


def _can_use_native_nelder_mead(op):
    options = op.optimizer_kwargs.get("options") or {}
    return (
        op.optimizer_kwargs.get("bounds") is None
        and isinstance(op.method, str)
        and op.method.lower() == "nelder-mead"
        and not (op.hess or op.hessp)
        and op.inner_inputs[0].dtype in ("float32", "float64")
        and op.optimizer_kwargs.keys() <= {"tol", "options", "bounds"}
        and options.keys() <= {"maxiter", "maxfev", "xatol", "fatol", "adaptive"}
        and all(
            options.get(name) is None or isinstance(options.get(name), int | np.integer)
            for name in ("maxiter", "maxfev")
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


def _compile_value(op, **kwargs):
    fgraph = op.fgraph
    if len(fgraph.outputs) > 1:
        # Nelder-Mead uses only the objective, so drop the gradient outputs.
        full = fgraph.unfreeze()
        fgraph = FunctionGraph(full.inputs, full.outputs[:1], clone=True).freeze()
    kwargs.pop("storage_map", None)
    inner_fn, inner_key = numba_funcify_and_cache_key(fgraph, **kwargs)
    x_dtype = op.inner_inputs[0].dtype
    scalar_x = op.inner_inputs[0].ndim == 0

    @numba_njit(fastmath=False)
    def evaluate(x, *args):
        inner_x = x.reshape(()) if scalar_x else x
        if x_dtype != "float64":
            inner_x = inner_x.astype(x_dtype)
        return inner_fn(inner_x, *args)[0].item()

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


def _funcify_nelder_mead(op, **kwargs):
    options = op.optimizer_kwargs.get("options") or {}
    tol = op.optimizer_kwargs.get("tol")
    maxiter = options.get("maxiter")
    maxiter = np.nan if maxiter is None else float(maxiter)
    maxfev = options.get("maxfev")
    maxfev = np.nan if maxfev is None else float(maxfev)
    xatol = float(options.get("xatol", 1e-4 if tol is None else tol))
    fatol = float(options.get("fatol", 1e-4 if tol is None else tol))
    adaptive = bool(options.get("adaptive", False))

    evaluate, inner_key = _compile_value(op, **kwargs)
    x_dtype = op.inner_inputs[0].dtype

    @numba_njit(fastmath=False)
    def minimize_nelder_mead(x0, *args):
        x = x0.ravel().astype(np.float64)
        solution, success = _nelder_mead(
            evaluate, x, args, maxiter, maxfev, xatol, fatol, adaptive
        )
        return solution.reshape(x0.shape).astype(x_dtype), success

    cache_key = _solver_cache_key(
        op, inner_key, "nelder-mead", maxiter, maxfev, xatol, fatol, adaptive
    )
    return minimize_nelder_mead, cache_key


@register_funcify_and_cache_key(MinimizeOp)
def numba_funcify_MinimizeOp(op, node, **kwargs):
    if _can_use_native_bfgs(op):
        return _funcify_bfgs(op, **kwargs)
    if _can_use_native_lbfgsb(op):
        return _funcify_lbfgsb(op, **kwargs)
    if _can_use_native_nelder_mead(op):
        return _funcify_nelder_mead(op, **kwargs)
    return generate_fallback_impl(op, node, **kwargs), None
