"""Compare native Numba and JAX BFGS with the Numba/SciPy fallback.

Cheap objectives expose the cost of running the solver and its callbacks in
Python. All solvers use the same symbolic objective, but their line searches
can take different steps. Timings exclude compilation and synchronize JAX.
"""

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config, function
from pytensor.tensor.optimize import minimize


@pytest.mark.parametrize(
    "n",
    [
        pytest.param(10, marks=pytest.mark.benchmark(group="bfgs-10")),
        pytest.param(100, marks=pytest.mark.benchmark(group="bfgs-100")),
    ],
)
@pytest.mark.parametrize("backend", ["SCIPY", "NUMBA", "JAX"])
def test_minimize_bfgs_benchmark(n, backend, benchmark, monkeypatch):
    x = pt.vector("x", shape=(n,), dtype="float64")
    target = pt.vector("target", shape=(n,), dtype="float64")
    weights = pt.vector("weights", shape=(n,), dtype="float64")
    objective = (weights * (x - target) ** 2).sum()
    inputs = (
        np.zeros(n, dtype="float64"),
        np.linspace(-1, 1, n, dtype="float64"),
        np.geomspace(0.1, 1, n, dtype="float64"),
    )
    options = {"gtol": 1e-6, "maxiter": 1000}

    if backend in ("SCIPY", "NUMBA"):
        if backend == "SCIPY":
            from pytensor.link.numba.dispatch import optimize as numba_optimize

            monkeypatch.setattr(
                numba_optimize, "_can_use_native_bfgs", lambda op: False
            )

        solution, success = minimize(
            objective, x, method="BFGS", optimizer_kwargs={"options": options}
        )
        solve = function(
            [x, target, weights],
            [solution, success],
            mode="NUMBA",
            trust_input=True,
        )
        assert solve(*inputs)[1]  # Compile and check convergence before timing.
        result = benchmark(solve, *inputs)

    else:
        jax = pytest.importorskip("jax")
        from jax.scipy.optimize import minimize as jax_minimize

        with config.change_flags(floatX="float64"), jax.enable_x64():
            objective_fn = function(
                [x, target, weights], objective, mode="JAX"
            ).vm.jit_fn

            def jax_objective(x, target, weights):
                return objective_fn(x, target, weights)[0]

            @jax.jit
            def jax_solve(x0, target, weights):
                result = jax_minimize(
                    jax_objective,
                    x0,
                    args=(target, weights),
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
            result = benchmark(solve, *jax_inputs)

    solution, success = result
    assert success
    np.testing.assert_allclose(
        solution, inputs[1], rtol=0, atol=options["gtol"] / (2 * inputs[2].min())
    )
    gradient = 2 * inputs[2] * (np.asarray(solution) - inputs[1])
    assert np.max(np.abs(gradient)) <= options["gtol"]
