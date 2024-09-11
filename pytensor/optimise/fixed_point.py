from collections.abc import Callable
from functools import partial

import pytensor
import pytensor.tensor as pt
from pytensor.scan.utils import until


def _check_convergence(f_x, tol):
    # TODO What convergence criterion? Norm of grad etc...
    converged = pt.lt(pt.linalg.norm(f_x, ord=1), tol)
    return converged


def fwd_solver(x_prev, *args, func, tol):
    x = func(x_prev, *args)
    is_converged = _check_convergence(x - x_prev, tol)
    return x, is_converged


def newton_solver(x_prev, *args, func, tol):
    f_root = func(x_prev, *args) - x_prev
    jac = pt.jacobian(f_root, x_prev)

    # TODO It would be nice to return the factored matrix for the pullback
    # TODO Handle errors of the factorization
    # 1D: x - f(x) / f'(x)
    # general: x - J^-1 f(x)

    # TODO: consider `grad = pt.linalg.solve(jac, f_root, assume_a="sym")``
    grad = pt.linalg.solve(jac, f_root)
    x = x_prev - grad

    # TODO: consider if this can all be done as a single call to `fwd_solver` as in the jax test case
    is_converged = _check_convergence(x - x_prev, tol)

    return x, is_converged


def fixed_point_solver(
    f: Callable,
    solver: Callable,
    x0: pt.TensorVariable,
    *args: tuple[pt.Variable, ...],
    max_iter: int = 1000,
    tol: float = 1e-5,
):
    args = [pt.as_tensor(arg) for arg in args]

    def _scan_step(x, n_steps, *args, func, solver, tol):
        x, is_converged = solver(x, *args, func=func, tol=tol)
        return (x, n_steps + 1), until(is_converged)

    partial_step = partial(
        _scan_step,
        func=f,
        solver=solver,
        tol=tol,
    )

    outputs, updates = pytensor.scan(
        partial_step,
        outputs_info=[x0, pt.constant(0, dtype="int64")],
        non_sequences=list(args),
        n_steps=max_iter,
        strict=True,
    )

    x_trace, n_steps_trace = outputs
    assert not updates

    return x_trace[-1], n_steps_trace[-1]
