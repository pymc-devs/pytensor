import functools as ft

import jax
import numpy as np
from numpy.testing import assert_array_almost_equal

import pytensor.tensor as pt
from pytensor.optimise.fixed_point import fixed_point_solver, fwd_solver, newton_solver


jax.config.update("jax_enable_x64", True)


def jax_newton_solver(f, z_init):
    def f_root(z):
        return f(z) - z

    def g(z):
        return z - jax.numpy.linalg.solve(jax.jacobian(f_root)(z), f_root(z))

    return jax_fwd_solver(g, z_init)


def jax_fwd_solver(f, z_init, tol=1e-5):
    z_prev, z = z_init, f(z_init)
    while jax.numpy.linalg.norm(z_prev - z) > tol:
        z_prev, z = z, f(z)
    return z


def test_fixed_point_forward():
    def g(x, W, b):
        return pt.tanh(pt.dot(W, x) + b)

    def _jax_g(x, W, b):
        return jax.numpy.tanh(jax.numpy.dot(W, x) + b)

    ndim = 10
    W = jax.random.normal(jax.random.PRNGKey(0), (ndim, ndim)) / jax.numpy.sqrt(ndim)
    b = jax.random.normal(jax.random.PRNGKey(1), (ndim,))

    W, b = np.asarray(W), np.asarray(b)

    jax_g = ft.partial(_jax_g, W=W, b=b)

    jax_solution = jax_fwd_solver(jax_g, jax.numpy.zeros_like(b))
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

    def _jax_g(x, W, b):
        return jax.numpy.tanh(jax.numpy.dot(W, x) + b)

    ndim = 10
    W = jax.random.normal(jax.random.PRNGKey(0), (ndim, ndim)) / jax.numpy.sqrt(ndim)
    b = jax.random.normal(jax.random.PRNGKey(1), (ndim,))

    W, b = np.asarray(W), np.asarray(b)

    jax_g = ft.partial(_jax_g, W=W, b=b)

    jax_solution = jax_newton_solver(jax_g, jax.numpy.zeros_like(b))
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
