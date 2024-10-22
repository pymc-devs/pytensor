from functools import partial
from typing import Literal

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor import nlinalg as pt_nlinalg
from pytensor.tensor import slinalg as pt_slinalg
from pytensor.tensor import subtensor as pt_subtensor
from pytensor.tensor.math import clip, cosh
from pytensor.tensor.type import matrix, vector
from tests.link.jax.test_basic import compare_jax_and_py


def test_jax_basic():
    rng = np.random.default_rng(28494)

    x = matrix("x")
    y = matrix("y")
    b = vector("b")

    # `ScalarOp`
    z = cosh(x**2 + y / 3.0)

    # `[Inc]Subtensor`
    out = pt_subtensor.set_subtensor(z[0], -10.0)
    out = pt_subtensor.inc_subtensor(out[0, 1], 2.0)
    out = out[:5, :3]

    out_fg = FunctionGraph([x, y], [out])

    test_input_vals = [
        np.tile(np.arange(10), (10, 1)).astype(config.floatX),
        np.tile(np.arange(10, 20), (10, 1)).astype(config.floatX),
    ]
    _, [jax_res] = compare_jax_and_py(out_fg, test_input_vals)

    # Confirm that the `Subtensor` slice operations are correct
    assert jax_res.shape == (5, 3)

    # Confirm that the `IncSubtensor` operations are correct
    assert jax_res[0, 0] == -10.0
    assert jax_res[0, 1] == -8.0

    out = clip(x, y, 5)
    out_fg = FunctionGraph([x, y], [out])
    compare_jax_and_py(out_fg, test_input_vals)

    out = pt.diagonal(x, 0)
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg, [np.arange(10 * 10).reshape((10, 10)).astype(config.floatX)]
    )

    out = pt_slinalg.cholesky(x)
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg,
        [
            (np.eye(10) + rng.standard_normal(size=(10, 10)) * 0.01).astype(
                config.floatX
            )
        ],
    )

    # not sure why this isn't working yet with lower=False
    out = pt_slinalg.Cholesky(lower=False)(x)
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg,
        [
            (np.eye(10) + rng.standard_normal(size=(10, 10)) * 0.01).astype(
                config.floatX
            )
        ],
    )

    out = pt_slinalg.solve(x, b)
    out_fg = FunctionGraph([x, b], [out])
    compare_jax_and_py(
        out_fg,
        [
            np.eye(10).astype(config.floatX),
            np.arange(10).astype(config.floatX),
        ],
    )

    out = pt.diag(b)
    out_fg = FunctionGraph([b], [out])
    compare_jax_and_py(out_fg, [np.arange(10).astype(config.floatX)])

    out = pt_nlinalg.det(x)
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg, [np.arange(10 * 10).reshape((10, 10)).astype(config.floatX)]
    )

    out = pt_nlinalg.matrix_inverse(x)
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg,
        [
            (np.eye(10) + rng.standard_normal(size=(10, 10)) * 0.01).astype(
                config.floatX
            )
        ],
    )


@pytest.mark.parametrize("check_finite", [False, True])
@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize("trans", [0, 1, 2])
def test_jax_SolveTriangular(trans, lower, check_finite):
    x = matrix("x")
    b = vector("b")

    out = pt_slinalg.solve_triangular(
        x,
        b,
        trans=trans,
        lower=lower,
        check_finite=check_finite,
    )
    out_fg = FunctionGraph([x, b], [out])
    compare_jax_and_py(
        out_fg,
        [
            np.eye(10).astype(config.floatX),
            np.arange(10).astype(config.floatX),
        ],
    )


def test_jax_block_diag():
    A = matrix("A")
    B = matrix("B")
    C = matrix("C")
    D = matrix("D")

    out = pt_slinalg.block_diag(A, B, C, D)
    out_fg = FunctionGraph([A, B, C, D], [out])

    compare_jax_and_py(
        out_fg,
        [
            np.random.normal(size=(5, 5)).astype(config.floatX),
            np.random.normal(size=(3, 3)).astype(config.floatX),
            np.random.normal(size=(2, 2)).astype(config.floatX),
            np.random.normal(size=(4, 4)).astype(config.floatX),
        ],
    )


def test_jax_block_diag_blockwise():
    A = pt.tensor3("A")
    B = pt.tensor3("B")
    out = pt_slinalg.block_diag(A, B)
    out_fg = FunctionGraph([A, B], [out])
    compare_jax_and_py(
        out_fg,
        [
            np.random.normal(size=(5, 5, 5)).astype(config.floatX),
            np.random.normal(size=(5, 3, 3)).astype(config.floatX),
        ],
    )


@pytest.mark.parametrize("lower", [False, True])
def test_jax_eigvalsh(lower):
    A = matrix("A")
    B = matrix("B")

    out = pt_slinalg.eigvalsh(A, B, lower=lower)
    out_fg = FunctionGraph([A, B], [out])

    with pytest.raises(NotImplementedError):
        compare_jax_and_py(
            out_fg,
            [
                np.array(
                    [[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]]
                ).astype(config.floatX),
                np.array(
                    [[10, 0, 1, 3], [0, 12, 7, 8], [1, 7, 14, 2], [3, 8, 2, 16]]
                ).astype(config.floatX),
            ],
        )
    compare_jax_and_py(
        out_fg,
        [
            np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]]).astype(
                config.floatX
            ),
            None,
        ],
    )


@pytest.mark.parametrize("method", ["direct", "bilinear"])
@pytest.mark.parametrize("shape", [(5, 5), (5, 5, 5)], ids=["matrix", "batch"])
def test_jax_solve_discrete_lyapunov(
    method: Literal["direct", "bilinear"], shape: tuple[int]
):
    A = pt.tensor(name="A", shape=shape)
    B = pt.tensor(name="B", shape=shape)
    out = pt_slinalg.solve_discrete_lyapunov(A, B, method=method)
    out_fg = FunctionGraph([A, B], [out])

    atol = rtol = 1e-8 if config.floatX == "float64" else 1e-3
    compare_jax_and_py(
        out_fg,
        [
            np.random.normal(size=shape).astype(config.floatX),
            np.random.normal(size=shape).astype(config.floatX),
        ],
        jax_mode="JAX",
        assert_fn=partial(np.testing.assert_allclose, atol=atol, rtol=rtol),
    )
