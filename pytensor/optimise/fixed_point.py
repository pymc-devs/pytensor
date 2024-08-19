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

    # grad = pt.linalg.solve(jac, f_root, assume_a="sym")
    grad = pt.linalg.solve(jac, f_root)
    x = x_prev - grad

    is_converged = _check_convergence(x - x_prev, tol)

    return x, is_converged


def fixed_point_solver(
    f: callable,
    solver: callable,
    x0: pt.TensorVariable,
    *args: tuple[pt.Variable, ...],
    max_iter: int = 1000,
    tol: float = 1e-5,
):
    args = [pt.as_tensor(arg) for arg in args]
    print(len(args))

    def _scan_step(x, *args, func, solver, tol):
        print(x.type)
        x, is_converged = solver(x, *args, func=func, tol=tol)
        print(x.type)
        return x, until(is_converged)

    partial_step = partial(
        _scan_step,
        func=f,
        solver=solver,
        tol=tol,
    )

    x_sequence, updates = pytensor.scan(
        partial_step,
        outputs_info=[x0],
        non_sequences=list(args),
        n_steps=max_iter,
        strict=True,
    )

    assert not updates

    x = x_sequence[-1]
    return x, x_sequence.shape[0]


# %%
# x_star, n_steps = fixed_point_solver(
#     g,
#     fwd_solver,
#     pt.zeros_like(b),
#     W, b,
# )
# print(x_star.eval(), n_steps.eval())

# %%
# x_star, n_steps = fixed_point_solver(
#     g,
#     newton_solver,
#     pt.zeros_like(b),
#     W, b,
#     max_n_steps=10,
# )
# print(x_star.eval(), n_steps.eval())


# def _newton_solver(x_prev, *args, func, tol):
#     f_root = lambda x: func(x) - x
#     g = lambda x: x - pt.linalg.solve(pt.jacobian(f_root)(x), f_root(x))
#     return fwd_solver(g, x)

# # %%
# def jax_newton_solver(f, z_init):
#     f_root = lambda z: f(z) - z
#     grad = jax.numpy.linalg.solve(jax.jacobian(f_root)(z_init), f_root(z_init))
#     # print(np.linalg.solve(grad, f_root_z))
#     # print(sp.linalg.solve(grad, f_root_z))
#     # print(jax.numpy.linalg.solve(grad, f_root_z))
#     # print(pt.linalg.solve(grad, f_root_z).eval())
#     g = lambda z: z - jax.numpy.linalg.solve(jax.jacobian(f_root)(z), f_root(z))
#     return jax_fwd_solver(g, z_init)
#     # return grad

# def jax_fwd_solver(f, z_init):
#     z_prev, z = z_init, f(z_init)
#     i = 1
#     while jax.numpy.linalg.norm(z_prev - z) > 1e-5:
#         z_prev, z = z, f(z)
#         i += 1
#     print(i)
#     return z

# def _jax_g(x, W, b):
#     return jax.numpy.tanh(jax.numpy.dot(W, x) + b)


# jax_g = partial(_jax_g, W=W, b=b)

# print(jax_newton_solver(jax_g, jax.numpy.zeros_like(b)))

# Array([-0.02879991, -0.8708013 , -1.4001148 , -0.1013868 , -0.641474  ,
#        -0.7552165 ,  0.62554246,  0.9438805 , -0.05192749,  1.430574  ],      dtype=float32)


# %%
