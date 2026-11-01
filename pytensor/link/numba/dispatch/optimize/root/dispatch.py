from hashlib import sha256

import numpy as np

from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    numba_funcify_and_cache_key,
    numba_njit,
    register_funcify_and_cache_key,
)
from pytensor.link.numba.dispatch.optimize.root.hybr import _hybrj
from pytensor.tensor.optimize import RootOp


# Bump when the solvers or the MINPACK helpers they share change, since the
# Numba disk cache keys do not see helper sources.
_SOLVER_CACHE_VERSION = 1


def _can_use_native_hybr(op):
    options = op.optimizer_kwargs.get("options") or {}
    maxfev = options.get("maxfev")
    return (
        isinstance(op.method, str)
        and op.method.lower() == "hybr"
        and op.jac
        and op.inner_inputs[0].dtype in ("float32", "float64")
        and op.optimizer_kwargs.keys() <= {"tol", "options"}
        and options.keys() <= {"xtol", "maxfev", "factor"}
        and (maxfev is None or isinstance(maxfev, int | np.integer))
    )


def _compile_residuals(op, **kwargs):
    kwargs.pop("storage_map", None)
    inner_fn, inner_key = numba_funcify_and_cache_key(op.fgraph, **kwargs)
    x_dtype = op.inner_inputs[0].dtype
    scalar_x = op.inner_inputs[0].ndim == 0
    jacobian_float64 = op.inner_outputs[1].dtype == "float64"

    @numba_njit(fastmath=False)
    def evaluate(x, *args):
        inner_x = x.reshape(()) if scalar_x else x
        if x_dtype != "float64":
            inner_x = inner_x.astype(x_dtype)
        equations, jacobian = inner_fn(inner_x, *args)
        # The solver writes into the residuals, and astype always copies, so it
        # owns them. It only reads the Jacobian, which needs no copy in float64.
        residuals = equations.astype(np.float64).reshape((-1,))
        if jacobian_float64:
            return residuals, jacobian
        return residuals, jacobian.astype(np.float64)

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


def _funcify_hybr(op, **kwargs):
    options = op.optimizer_kwargs.get("options") or {}
    tol = op.optimizer_kwargs.get("tol")
    xtol = float(options.get("xtol", 1.49012e-08 if tol is None else tol))
    # SciPy treats a non-positive maxfev as unset.
    maxfev = int(options.get("maxfev", 0))
    factor = float(options.get("factor", 100))
    if xtol < 0:
        raise ValueError("'xtol' must be non-negative.")
    if factor <= 0:
        raise ValueError("'factor' must be positive.")

    evaluate, inner_key = _compile_residuals(op, **kwargs)
    x_dtype = op.inner_inputs[0].dtype

    @numba_njit(fastmath=False)
    def root_hybr(x0, *args):
        x = x0.ravel().astype(np.float64)
        limit = maxfev if maxfev > 0 else 100 * (x.size + 1)
        solution, info = _hybrj(evaluate, x, args, xtol, limit, factor)
        return solution.reshape(x0.shape).astype(x_dtype), info == 1

    cache_key = _solver_cache_key(op, inner_key, "hybr", xtol, maxfev, factor)
    return root_hybr, cache_key


@register_funcify_and_cache_key(RootOp)
def numba_funcify_RootOp(op, node, **kwargs):
    if _can_use_native_hybr(op):
        return _funcify_hybr(op, **kwargs)
    return generate_fallback_impl(op, node, **kwargs), None
