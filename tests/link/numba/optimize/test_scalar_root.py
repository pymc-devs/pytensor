import inspect
from functools import partial

import numpy as np
import pytest
import scipy.optimize as scipy_optimize

import pytensor
import pytensor.tensor as pt
from pytensor.link.numba.dispatch.optimize.scalar_root import (
    BRACKET_DEFAULTS,
    OPEN_DEFAULTS,
)
from pytensor.tensor.optimize import root_scalar
from tests.link.numba.test_basic import compare_numba_and_py


pytestmark = [
    pytest.mark.filterwarnings("error"),
    # scipy warns on these in the perform reference runs
    pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:Tolerance of:RuntimeWarning"),
]

pytest.importorskip("numba")

# Roots and flags match scipy's up to ulp differences between numba's and numpy's math
# functions in the objective.
matches_scipy = partial(np.testing.assert_allclose, rtol=1e-10)


def _defaults(function):
    return {
        name: parameter.default
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _bracket(a):
    return {"bracket": pt.stack([-2 * a, 0.0])}


# The root of x + 2 * a * cos(x) near zero at a = 1.
ROOT = -1.02986653
METHODS = [
    pytest.param("newton", lambda a: {"jac": True}, id="newton"),
    pytest.param("halley", lambda a: {"jac": True, "hess": True}, id="halley"),
    pytest.param("secant", lambda a: {}, id="secant"),
    pytest.param("brentq", _bracket, id="brentq"),
    pytest.param("bisect", _bracket, id="bisect"),
    pytest.param("brenth", _bracket, id="brenth"),
    pytest.param("ridder", _bracket, id="ridder"),
    pytest.param("toms748", _bracket, id="toms748"),
]


def _root_graph(method, options):
    x = pt.dscalar("x")
    a = pt.dscalar("a")
    solution, success = root_scalar(
        equation=x + 2 * a * pt.cos(x), variable=x, method=method, **options(a)
    )
    return [x, a], [solution, success, pytensor.grad(solution, a)]


def _root_function(method, options):
    inputs, outputs = _root_graph(method=method, options=options)
    return pytensor.function(inputs, outputs, mode="NUMBA")


@pytest.mark.parametrize("method, options", METHODS)
def test_root_scalar(method, options):
    """At the root of f(x, a) = x + 2 a cos(x), dx/da = -f_a / f_x = -2 cos(x) / (1 - 2 a sin(x))."""
    inputs, outputs = _root_graph(method=method, options=options)

    _, (_, success, gradient) = compare_numba_and_py(
        graph_inputs=inputs,
        graph_outputs=outputs,
        test_inputs=[0.0, 1.0],
        assert_fn=matches_scipy,
        eval_obj_mode=False,
    )

    assert success
    np.testing.assert_allclose(
        gradient, -2 * np.cos(ROOT) / (1 - 2 * np.sin(ROOT)), rtol=1e-6
    )


@pytest.mark.parametrize("method, options", METHODS)
def test_root_scalar_maxiter(method, options):
    fn = _root_function(
        method=method,
        options=lambda a: {**options(a), "optimizer_kwargs": {"maxiter": 1}},
    )

    _, success, _ = fn(0.0, 1.0)

    assert not success


BRACKET_METHODS = ["brentq", "bisect", "brenth", "ridder", "toms748"]


@pytest.mark.parametrize("method", BRACKET_METHODS)
@pytest.mark.parametrize(
    "equation, bracket",
    [
        (lambda x: x + 2 * pt.cos(x), (0.0, 1.0)),
        (lambda x: 1e-170 * (x - 5), (0.0, 1.0)),
        (lambda x: pt.log(x) - 0.5, (-1.0, 3.0)),
        (lambda x: 1 / x - 1, (0.0, 2.0)),
        (lambda x: x, (-1e-160, 1e160)),
        (lambda x: x - 1, (-1e308, 1e308)),
        (lambda x: pt.expm1((x + 3000.3) / 1000) - 1e-7, (-4000.0, -2000.0)),
        (lambda x: pt.switch((x > 0.5) & (x < 2.5), np.nan, x - 3), (-1.0, 5.0)),
        (lambda x: x + 2 * pt.cos(x), (-2.0,)),
        (lambda x: pt.switch(x > 1.5, np.nan, x), (0.0, 2.0)),
        (lambda x: pt.switch(x > 1.5, np.inf, x), (0.0, 2.0)),
        (
            lambda x: pt.switch(x < 0.5, np.inf, pt.switch(x > 1.5, np.nan, x - 1)),
            (0.0, 2.0),
        ),
        (lambda x: pt.switch((x > 0.6) & (x < 1.2), np.inf, x * x - 2), (0.0, 2.0)),
    ],
    ids=[
        "no_sign_change",
        "no_sign_change_underflow",
        "nan_at_an_end",
        "inf_at_an_end",
        "ratio_overflow",
        "huge_bracket",
        "negative_root_far_from_zero",
        "nan_inside_bracket",
        "short_bracket",
        "zero_then_nan_end",
        "zero_then_inf_end",
        "inf_then_nan_end",
        "inf_inside_bracket",
    ],
)
def test_root_scalar_bracket_edge_cases(method, equation, bracket):
    """Bad brackets, non-finite values and extreme scales give scipy's result, including where scipy itself fails."""
    x = pt.dscalar("x")
    solution, success = root_scalar(
        equation=equation(x), variable=x, method=method, bracket=bracket
    )

    compare_numba_and_py(
        graph_inputs=[x],
        graph_outputs=[solution, success],
        test_inputs=[0.0],
        assert_fn=matches_scipy,
        eval_obj_mode=False,
    )


@pytest.mark.parametrize(
    "method, options, error, message",
    [
        (
            "brentq",
            lambda a: {**_bracket(a=a), "optimizer_kwargs": {"xtol": 0.0}},
            ValueError,
            "xtol too small",
        ),
        (
            "brentq",
            lambda a: {**_bracket(a=a), "optimizer_kwargs": {"rtol": 1e-20}},
            ValueError,
            "rtol too small",
        ),
        (
            "secant",
            lambda a: {"optimizer_kwargs": {"maxiter": 0}},
            ValueError,
            "maxiter must be greater than 0",
        ),
        (
            "brentq",
            lambda a: {**_bracket(a=a), "optimizer_kwargs": {"maxiter": 10.5}},
            TypeError,
            "cannot be interpreted as an integer",
        ),
    ],
    ids=["xtol", "rtol", "maxiter", "fractional_maxiter"],
)
def test_root_scalar_invalid_static_option_raises(method, options, error, message):
    with pytest.warns(UserWarning, match="Numba will use object mode"):
        fn = _root_function(method=method, options=options)

    with pytest.raises(error, match=message):
        fn(0.0, 1.0)


def test_root_scalar_method_name_is_case_insensitive():
    fn = _root_function(method="BrentQ", options=_bracket)

    root, success, _ = fn(0.0, 1.0)

    assert success
    np.testing.assert_allclose(root, ROOT, atol=1e-8)


@pytest.mark.parametrize(
    "method, options",
    [
        ("secant", {}),
        ("newton", {"jac": True}),
        ("halley", {"jac": True, "hess": True}),
    ],
)
@pytest.mark.parametrize(
    "dtype, equation, x0",
    [("float32", lambda x: x**3 - 7, 1.0), ("float64", lambda x: x**3 - 1e30, 2e10)],
    ids=["float32", "root_near_1e10"],
)
def test_open_methods_match_scipy(method, options, dtype, equation, x0):
    """scipy iterates in the variable's dtype and stops when the iterate stops moving."""
    x = pt.scalar("x", dtype=dtype)
    solution, success = root_scalar(
        equation=equation(x), variable=x, method=method, **options
    )

    compare_numba_and_py(
        graph_inputs=[x],
        graph_outputs=[solution, success],
        test_inputs=[np.array(x0, dtype)],
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-6),
        eval_obj_mode=False,
    )


def test_secant_flat_region():
    x = pt.dscalar("x")
    solution, success = root_scalar(
        equation=pt.switch(x > 5, x - 6, 1.0), variable=x, method="secant"
    )

    compare_numba_and_py(
        graph_inputs=[x],
        graph_outputs=[solution, success],
        test_inputs=[0.0],
        assert_fn=matches_scipy,
        eval_obj_mode=False,
    )


@pytest.mark.parametrize(
    "solver",
    [
        scipy_optimize.bisect,
        scipy_optimize.brentq,
        scipy_optimize.brenth,
        scipy_optimize.ridder,
        scipy_optimize.toms748,
    ],
    ids=["bisect", "brentq", "brenth", "ridder", "toms748"],
)
def test_bracket_solver_defaults_match_scipy(solver):
    assert {
        name: _defaults(function=solver)[name] for name in BRACKET_DEFAULTS
    } == BRACKET_DEFAULTS


def test_open_solver_defaults_match_scipy():
    newton_defaults = _defaults(function=scipy_optimize.newton)
    assert {
        "xtol": newton_defaults["tol"],
        "maxiter": newton_defaults["maxiter"],
    } == OPEN_DEFAULTS


@pytest.mark.parametrize(
    "method, options",
    [("secant", lambda a: {}), ("toms748", _bracket)],
    ids=["secant", "toms748"],
)
def test_root_scalar_keeps_variable_dtype(method, options):
    x = pt.fscalar("x")
    a = pt.fscalar("a")
    solution, _ = root_scalar(
        equation=x + 2 * a * pt.cos(x), variable=x, method=method, **options(a)
    )

    _, (root,) = compare_numba_and_py(
        graph_inputs=[x, a],
        graph_outputs=[solution],
        test_inputs=[np.float32(0.0), np.float32(1.0)],
        eval_obj_mode=False,
    )

    assert root.dtype == np.float32
    np.testing.assert_allclose(root, ROOT, rtol=1e-6)


def test_root_scalar_evaluates_objective_in_variable_dtype():
    """In float32, x - 0.1 is exactly zero on an interval that bisect reaches within 30 steps; in float64 it is not."""
    x = pt.fscalar("x")
    solution, success = root_scalar(
        equation=x - np.float32(0.1),
        variable=x,
        method="bisect",
        bracket=(0.0, 1.0),
        optimizer_kwargs={"maxiter": 30},
    )

    _, (_, converged) = compare_numba_and_py(
        graph_inputs=[x],
        graph_outputs=[solution, success],
        test_inputs=[np.float32(0.0)],
        assert_fn=matches_scipy,
        eval_obj_mode=False,
    )

    assert converged


@pytest.mark.parametrize(
    "method, options",
    [
        ("newton", lambda a: {"jac": True, "optimizer_kwargs": {"rtol": 1e-12}}),
        ("newton", lambda a: {}),
    ],
    ids=["unsupported_option", "newton_without_jac"],
)
def test_root_scalar_object_mode_fallback(method, options):
    with pytest.warns(UserWarning, match="Numba will use object mode"):
        fn = _root_function(method=method, options=options)

    root, success, _ = fn(0.0, 1.0)

    assert success
    np.testing.assert_allclose(root, ROOT, atol=1e-8)
