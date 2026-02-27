import jax
import numpy as np
from numpy.testing import assert_array_almost_equal

import pytensor.tensor as pt
from pytensor.optimise.fixed_point import fixed_point_solver, fwd_solver, newton_solver


jax.config.update("jax_enable_x64", True)


def jax_newton_solver(f, x0):
    def f_root(x):
        return f(x) - x

    def g(x):
        return x - jax.numpy.linalg.solve(jax.jacobian(f_root)(x), f_root(x))

    return jax_fwd_solver(g, x0)


def jax_fwd_solver(f, x0, tol=1e-5):
    x_prev, x = x0, f(x0)
    while jax.numpy.linalg.norm(x_prev - x) > tol:
        x_prev, x = x, f(x)
    return x


def jax_fixed_point_solver(solver, f, params, x0, **solver_kwargs):
    x_star = solver(lambda x: f(x, *params), x0=x0, **solver_kwargs)
    return x_star


def test_fixed_point_forward():
    """Test taken from the [Deep Implicit Layers workshop](https://implicit-layers-tutorial.org/implicit_functions/)."""

    def g(x, W, b):
        return pt.tanh(pt.dot(W, x) + b)

    def jax_g(x, W, b):
        return jax.numpy.tanh(jax.numpy.dot(W, x) + b)

    ndim = 10
    W = jax.random.normal(jax.random.PRNGKey(0), (ndim, ndim)) / jax.numpy.sqrt(ndim)
    b = jax.random.normal(jax.random.PRNGKey(1), (ndim,))

    W, b = np.asarray(W), np.asarray(b)

    jax_solution = jax_fixed_point_solver(
        jax_fwd_solver,
        jax_g,
        (W, b),
        x0=jax.numpy.zeros_like(b),
    )

    pytensor_solution, _ = fixed_point_solver(
        g,
        fwd_solver,
        pt.zeros_like(b),
        W,
        b,
    )
    assert_array_almost_equal(jax_solution, pytensor_solution.eval(), decimal=5)


def test_fixed_point_newton():
    def g(x, W, b):
        return pt.tanh(pt.dot(W, x) + b)

    def jax_g(x, W, b):
        return jax.numpy.tanh(jax.numpy.dot(W, x) + b)

    ndim = 10
    W = jax.random.normal(jax.random.PRNGKey(0), (ndim, ndim)) / jax.numpy.sqrt(ndim)
    b = jax.random.normal(jax.random.PRNGKey(1), (ndim,))

    W, b = np.asarray(W), np.asarray(b)

    jax_solution = jax_fixed_point_solver(
        jax_newton_solver,
        jax_g,
        (W, b),
        x0=jax.numpy.zeros_like(b),
    )

    pytensor_solution, _ = fixed_point_solver(
        g,
        newton_solver,
        pt.zeros_like(b),
        W,
        b,
    )
    assert_array_almost_equal(jax_solution, pytensor_solution.eval(), decimal=5)


# TODO: test the grad is the same as naive grad from propagating through each step of the solver (`pt.grad`)
# and adjoint implicit function theorem rewritten grad
# see the [notes](https://theorashid.github.io/notes/fixed-point-iteration
# and the [Deep Implicit Layers workshop](https://implicit-layers-tutorial.org/implicit_functions/)

# %%
# import jax
# import numpy as np

# def grad_test_fixed_point_forward():
#     def jax_g(x, W, b):
#         return jax.numpy.tanh(jax.numpy.dot(W, x) + b)

#     ndim = 10
#     W = jax.random.normal(jax.random.PRNGKey(0), (ndim, ndim)) / jax.numpy.sqrt(ndim)
#     b = jax.random.normal(jax.random.PRNGKey(1), (ndim,))

#     W, b = np.asarray(W), np.asarray(b)  # params

#     # gradient of the sum of the outputs with respect to the parameter matrix
#     jax_grad = jax.grad(
#         lambda W: jax_fixed_point_solver(
#             jax_fwd_solver,
#             jax_g,
#             (W, b),  # wrt W
#             x0=jax.numpy.zeros_like(b),
#         ).sum()
#     )(W)
#     print(jax_grad[0])

# grad_test_fixed_point_forward()

#     # params -> W
#     # z -> x
#     # x -> b
#     # f = lambda W, b, x: jnp.tanh(jnp.dot(W, x) + b)
#     # x_star = solver(lambda x: f(params, b, x), x_init=jnp.zeros_like(b))
#     # x_star = fixed_point_layer(fwd_solver, f, W, b)
#     # g = jax.grad(lambda W: fixed_point_layer(fwd_solver, f, W, b).sum())(W)
# %%
# def implicit_gradients_vjp(solver, f, res, x_soln):
#     params, x, x_star = res
#     # find adjoint u^T via solver
#     # u^T = w^T + u^T \delta_{x_star} f(x_star, params)
#     _, vjp_x = jax.vjp(lambda : f(x, *params), x_star)  # diff wrt x
#     _, vjp_par = jax.vjp(lambda params: f(x, *params), *params)  # diff wrt params
#     u = solver(lambda u: vjp_x(u)[0] + x_soln, x0=jax.numpy.zeros_like(x_soln))

#     # then compute vjp u^T \delta_{params} f(x_star, params)
#     return vjp_par(u)
