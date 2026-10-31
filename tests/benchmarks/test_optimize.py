"""Compare Numba optimizers with SciPy, JAX BFGS and Optax L-BFGS.

Cheap objectives expose the cost of running the solver in Python. SciPy calls
the same Numba-compiled objective and gradient, so its timings isolate the
solver. All solvers use the same symbolic objective, but their line searches
can take different steps. Timings exclude compilation and synchronize JAX;
extra_info records preparation through the first solve separately.
"""

import time

import numpy as np
import pytest
import scipy.optimize

import pytensor
import pytensor.tensor as pt
from pytensor import config, function
from pytensor.tensor.optimize import minimize


def quadratic(n):
    x = pt.vector("x", shape=(n,), dtype="float64")
    target = pt.vector("target", shape=(n,), dtype="float64")
    weights = pt.vector("weights", shape=(n,), dtype="float64")
    inputs = (
        np.zeros(n, dtype="float64"),
        np.linspace(-1, 1, n, dtype="float64"),
        np.geomspace(0.1, 1, n, dtype="float64"),
    )
    return [x, target, weights], (weights * (x - target) ** 2).sum(), inputs, inputs[1]


def rosenbrock(n):
    x = pt.vector("x", shape=(n,), dtype="float64")
    odd, even = x[::2], x[1::2]
    objective = (100 * (even - odd**2) ** 2 + (1 - odd) ** 2).sum()
    return [x], objective, (np.tile([-1.2, 1.0], n // 2),), np.ones(n)


@pytest.mark.parametrize("n", [10, 100, 1000])
@pytest.mark.parametrize("problem", [quadratic, rosenbrock], ids=lambda p: p.__name__)
@pytest.mark.parametrize(
    "method, backend",
    [
        ("BFGS", "SCIPY"),
        ("BFGS", "NUMBA"),
        ("BFGS", "JAX"),
        ("L-BFGS-B", "SCIPY"),
        ("L-BFGS-B", "NUMBA"),
        ("L-BFGS-B", "OPTAX"),
    ],
)
def test_minimize_benchmark(n, problem, method, backend, benchmark):
    if backend == "JAX" and problem is rosenbrock:
        pytest.skip("JAX BFGS stops with a line-search failure (status 3)")
    benchmark.group = f"minimize-{problem.__name__}-{n}"
    start = time.perf_counter()
    symbolic_inputs, objective, inputs, expected = problem(n)
    x = symbolic_inputs[0]
    value_and_grad = function(
        symbolic_inputs,
        [objective, pytensor.grad(objective, x)],
        mode="NUMBA",
        trust_input=True,
    )
    options = {"gtol": 1e-6, "maxiter": 200 * n}  # SciPy's BFGS default
    if method == "L-BFGS-B":
        options["ftol"] = 0  # Compare at the same gradient tolerance.

    if backend == "SCIPY":

        def solve(*inputs):
            result = scipy.optimize.minimize(
                value_and_grad,
                inputs[0],
                args=inputs[1:],
                jac=True,
                method=method,
                options=options,
            )
            return result.x, result.success

        assert solve(*inputs)[1]  # Compile and check convergence before timing.
        benchmark.extra_info["ready_seconds"] = time.perf_counter() - start
        result = benchmark(solve, *inputs)

    elif backend == "NUMBA":
        solution, success = minimize(
            objective, x, method=method, optimizer_kwargs={"options": options}
        )
        solve = function(
            symbolic_inputs,
            [solution, success],
            mode="NUMBA",
            trust_input=True,
        )
        assert solve(*inputs)[1]  # Compile and check convergence before timing.
        benchmark.extra_info["ready_seconds"] = time.perf_counter() - start
        result = benchmark(solve, *inputs)

    else:
        jax = pytest.importorskip("jax")
        from jax.scipy.optimize import minimize as jax_minimize

        with config.change_flags(floatX="float64"), jax.enable_x64():
            objective_fn = function(symbolic_inputs, objective, mode="JAX").vm.jit_fn

            def jax_objective(x, *args):
                return objective_fn(x, *args)[0]

            if backend == "OPTAX":
                optax = pytest.importorskip("optax")
                solver = optax.lbfgs(memory_size=10)

                @jax.jit
                def jax_solve(x0, *args):
                    def value_fn(x):
                        return jax_objective(x, *args)

                    value_and_grad = optax.value_and_grad_from_state(value_fn)
                    state = solver.init(x0)
                    value, gradient = value_and_grad(x0, state=state)

                    def condition(carry):
                        _, _, value, gradient, iteration = carry
                        return (
                            (iteration < options["maxiter"])
                            & jax.numpy.isfinite(value)
                            & (jax.numpy.max(jax.numpy.abs(gradient)) > options["gtol"])
                        )

                    def step(carry):
                        x, state, value, gradient, iteration = carry
                        updates, state = solver.update(
                            gradient,
                            state,
                            x,
                            value=value,
                            grad=gradient,
                            value_fn=value_fn,
                        )
                        x = optax.apply_updates(x, updates)
                        value, gradient = value_and_grad(x, state=state)
                        return x, state, value, gradient, iteration + 1

                    x, _, value, gradient, _ = jax.lax.while_loop(
                        condition, step, (x0, state, value, gradient, 0)
                    )
                    success = jax.numpy.isfinite(value) & (
                        jax.numpy.max(jax.numpy.abs(gradient)) <= options["gtol"]
                    )
                    return x, success

            else:

                @jax.jit
                def jax_solve(x0, *args):
                    result = jax_minimize(
                        jax_objective,
                        x0,
                        args=args,
                        method="BFGS",
                        options=options,
                    )
                    return result.x, result.success

            def solve(*inputs):
                return jax.block_until_ready(jax_solve(*inputs))

            # Keep inputs on the CPU; host-to-device copies are outside timing.
            cpu = jax.devices("cpu")[0]
            jax_inputs = tuple(jax.device_put(value, cpu) for value in inputs)
            assert solve(*jax_inputs)[1]  # Compile, synchronize, and check convergence.
            benchmark.extra_info["ready_seconds"] = time.perf_counter() - start
            result = benchmark(solve, *jax_inputs)

    solution, success = result
    assert success
    solution = np.asarray(solution)
    _, gradient = value_and_grad(solution, *inputs[1:])
    assert np.max(np.abs(gradient)) <= options["gtol"]
    np.testing.assert_allclose(solution, expected, rtol=0, atol=1e-4)
