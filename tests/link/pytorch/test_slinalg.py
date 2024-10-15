import numpy as np
import pytest

from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor import slinalg as pt_slinalg
from pytensor.tensor.type import matrix, vector
from tests.link.pytorch.test_basic import compare_pytorch_and_py


@pytest.mark.parametrize("lower", [False, True])
def test_pytorch_eigvalsh(lower):
    A = matrix("A")
    B = matrix("B")

    out = pt_slinalg.eigvalsh(A, B, lower=lower)
    out_fg = FunctionGraph([A, B], [out])

    with pytest.raises(NotImplementedError):
        compare_pytorch_and_py(
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
    compare_pytorch_and_py(
        out_fg,
        [
            np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]]).astype(
                config.floatX
            ),
            None,
        ],
    )


def test_pytorch_cholesky():
    rng = np.random.default_rng(28494)

    x = matrix("x")

    out = pt_slinalg.cholesky(x)
    out_fg = FunctionGraph([x], [out])
    compare_pytorch_and_py(
        out_fg,
        [
            (np.eye(10) + rng.standard_normal(size=(10, 10)) * 0.01).astype(
                config.floatX
            )
        ],
    )

    out = pt_slinalg.cholesky(x, lower=False)
    out_fg = FunctionGraph([x], [out])
    compare_pytorch_and_py(
        out_fg,
        [
            (np.eye(10) + rng.standard_normal(size=(10, 10)) * 0.01).astype(
                config.floatX
            )
        ],
    )


def test_pytorch_solve():
    x = matrix("x")
    b = vector("b")

    out = pt_slinalg.solve(x, b)
    out_fg = FunctionGraph([x, b], [out])
    compare_pytorch_and_py(
        out_fg,
        [
            np.eye(10).astype(config.floatX),
            np.arange(10).astype(config.floatX),
        ],
    )


@pytest.mark.parametrize(
    "check_finite",
    (False, pytest.param(True, marks=pytest.mark.xfail(raises=NotImplementedError))),
)
@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize("trans", [0, 1, 2, "S"])
def test_pytorch_SolveTriangular(trans, lower, check_finite):
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
    compare_pytorch_and_py(
        out_fg,
        [
            np.eye(10).astype(config.floatX),
            np.arange(10).astype(config.floatX),
        ],
    )


def test_pytorch_block_diag():
    A = matrix("A")
    B = matrix("B")
    C = matrix("C")
    D = matrix("D")

    out = pt_slinalg.block_diag(A, B, C, D)
    out_fg = FunctionGraph([A, B, C, D], [out])

    compare_pytorch_and_py(
        out_fg,
        [
            np.random.normal(size=(5, 5)).astype(config.floatX),
            np.random.normal(size=(3, 3)).astype(config.floatX),
            np.random.normal(size=(2, 2)).astype(config.floatX),
            np.random.normal(size=(4, 4)).astype(config.floatX),
        ],
    )
