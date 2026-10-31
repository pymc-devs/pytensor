import inspect
from functools import partial

import numpy as np
import pytest
import scipy.optimize._optimize as scipy_optimizers

import pytensor
import pytensor.tensor as pt
from pytensor.link.numba.dispatch.basic import numba_funcify_and_cache_key
from pytensor.link.numba.dispatch.optimize.scalar_minimize import MINIMIZE_METHODS
from pytensor.tensor.optimize import minimize_scalar
from tests.link.numba.test_basic import compare_numba_and_py


pytestmark = [
    pytest.mark.filterwarnings("error"),
    # scipy warns on these in the perform reference runs
    pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning"),
]

pytest.importorskip("numba")

# Minima and flags match scipy's up to ulp differences between numba's and numpy's math
# functions in the objective, which a minimizer amplifies more than a root finder does.
matches_scipy = partial(np.testing.assert_allclose, rtol=1e-8)


def _defaults(function):
    return {
        name: parameter.default
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


# cos(x) + 0.1 a x is minimized where sin(x) = 0.1 a, at x = pi - arcsin(0.1 a) near pi.
MINIMIZE_CASES = [
    pytest.param("brent", lambda a: {}, id="brent"),
    pytest.param(
        "brent", lambda a: {"bracket": pt.stack([a - 1.0, a])}, id="brent_search"
    ),
    pytest.param(
        "brent",
        lambda a: {"bracket": pt.stack([2.0, 3.0 * a, 4.0])},
        id="brent_bracket",
    ),
    pytest.param(
        "golden", lambda a: {"bracket": pt.stack([2.0, 3.0 * a, 4.0])}, id="golden"
    ),
    pytest.param(
        "bounded", lambda a: {"bounds": pt.stack([0.0, 5.0 * a])}, id="bounded"
    ),
]


def _minimize_graph(method, bound, optimizer_kwargs=None):
    x = pt.dscalar("x")
    a = pt.dscalar("a")
    solution, success = minimize_scalar(
        objective=pt.cos(x) + 0.1 * a * x,
        x=x,
        method=method,
        optimizer_kwargs=optimizer_kwargs,
        **bound(a),
    )
    return [x, a], [solution, success, pytensor.grad(solution, a)]


def _minimize_function(method, bound, optimizer_kwargs=None):
    inputs, outputs = _minimize_graph(
        method=method, bound=bound, optimizer_kwargs=optimizer_kwargs
    )
    return pytensor.function(inputs, outputs, mode="NUMBA")


@pytest.mark.parametrize("method, bound", MINIMIZE_CASES)
def test_minimize_scalar(method, bound):
    """dx/da = -0.1 / sqrt(1 - (0.1 a)^2)"""
    inputs, outputs = _minimize_graph(method=method, bound=bound)

    _, (_, success, gradient) = compare_numba_and_py(
        graph_inputs=inputs,
        graph_outputs=outputs,
        test_inputs=[0.0, 1.0],
        assert_fn=matches_scipy,
        eval_obj_mode=False,
    )

    assert success
    np.testing.assert_allclose(gradient, -0.1 / np.sqrt(0.99), rtol=1e-4)


@pytest.mark.parametrize("method, bound", MINIMIZE_CASES)
def test_minimize_scalar_maxiter(method, bound):
    fn = _minimize_function(
        method=method, bound=bound, optimizer_kwargs={"options": {"maxiter": 1}}
    )

    _, success, _ = fn(0.0, 1.0)

    assert not success


@pytest.mark.parametrize(
    "method, bound",
    [
        ("brent", lambda a: {"bracket": pt.stack([2.0, 4.0 * a, 3.0])}),
        ("golden", lambda a: {"bracket": pt.stack([1.0, 2.0 * a, 3.0, 4.0])}),
        ("bounded", lambda a: {"bounds": pt.stack([5.0 * a, 0.0])}),
    ],
    ids=["unordered_bracket", "long_bracket", "reversed_bounds"],
)
def test_minimize_scalar_invalid_bracket(method, bound):
    fn = _minimize_function(method=method, bound=bound)

    minimum, success, _ = fn(0.0, 1.0)

    assert not success
    assert np.isnan(minimum)


@pytest.mark.parametrize(
    "method, bound",
    [
        ("brent", {"bracket": (2.0, 3.0)}),
        ("golden", {"bracket": (2.0, 3.0, 4.0)}),
        (
            "golden",
            {
                "bracket": (2.0, 3.0, 4.0),
                "optimizer_kwargs": {"options": {"xtol": 1e-6}},
            },
        ),
        ("bounded", {"bounds": (0.0, 5.0)}),
    ],
    ids=["brent", "golden", "golden_given_xtol", "bounded"],
)
def test_minimize_scalar_float32_matches_scipy(method, bound):
    """With float32 bounds, scipy runs these minimizers in float32."""
    x = pt.fscalar("x")
    solution, success = minimize_scalar(
        objective=(x * x - 9) ** 2 + x, x=x, method=method, **bound
    )

    _, (minimum, _) = compare_numba_and_py(
        graph_inputs=[x],
        graph_outputs=[solution, success],
        test_inputs=[np.float32(0.0)],
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-6),
        eval_obj_mode=False,
    )

    assert minimum.dtype == np.float32


@pytest.mark.parametrize("method", ["brent", "golden"])
@pytest.mark.parametrize(
    "objective",
    [lambda x: x * 0 + 1.0, lambda x: pt.switch(x > 0.5, np.nan, x), lambda x: x],
    ids=["constant", "nan_beyond_start", "unbounded_below"],
)
def test_minimize_scalar_failed_bracket_search(method, objective):
    """A bracket search that finds no minimum reports scipy's recovered point as a failure."""
    x = pt.dscalar("x")
    solution, success = minimize_scalar(objective=objective(x), x=x, method=method)

    _, (_, succeeded) = compare_numba_and_py(
        graph_inputs=[x],
        graph_outputs=[solution, success],
        test_inputs=[0.0],
        assert_fn=matches_scipy,
        eval_obj_mode=False,
    )

    assert not succeeded


@pytest.mark.parametrize(
    "method, bound",
    [
        ("brent", {"bracket": (2.0, 2.9, 4.0)}),
        ("golden", {"bracket": (2.0, 2.9, 4.0)}),
        ("bounded", {"bounds": (2.0, 4.0)}),
    ],
    ids=["brent", "golden", "bounded"],
)
def test_minimize_scalar_nan_inside_bracket(method, bound):
    """NaN values met after the bracket is found are handled as scipy's comparisons handle them."""
    x = pt.dscalar("x")
    objective = pt.switch((x > 3.0) & (x < 3.3), np.nan, pt.cos(x))
    solution, success = minimize_scalar(
        objective=objective, x=x, method=method, **bound
    )

    compare_numba_and_py(
        graph_inputs=[x],
        graph_outputs=[solution, success],
        test_inputs=[0.0],
        assert_fn=matches_scipy,
        eval_obj_mode=False,
    )


def test_golden_cache_key_distinguishes_default_xtol():
    """The default xtol is a NumPy float64 and a given one adopts the point dtype, so they compile differently."""
    x = pt.fscalar("x")
    keys = []
    for optimizer_kwargs in (None, {"options": {"xtol": np.sqrt(np.finfo(float).eps)}}):
        solution, _ = minimize_scalar(
            objective=(x * x - 9) ** 2 + x,
            x=x,
            method="golden",
            bracket=(2.0, 3.0, 4.0),
            optimizer_kwargs=optimizer_kwargs,
        )
        _, key = numba_funcify_and_cache_key(solution.owner.op, solution.owner)
        keys.append(key)

    assert keys[0] != keys[1]


@pytest.mark.parametrize(
    "method, minimizer",
    [
        ("brent", scipy_optimizers._minimize_scalar_brent),
        ("golden", scipy_optimizers._minimize_scalar_golden),
        ("bounded", scipy_optimizers._minimize_scalar_bounded),
    ],
)
def test_minimize_solver_defaults_match_scipy(method, minimizer):
    method_defaults = MINIMIZE_METHODS[method].defaults
    assert {
        name: _defaults(function=minimizer)[name] for name in method_defaults
    } == method_defaults
