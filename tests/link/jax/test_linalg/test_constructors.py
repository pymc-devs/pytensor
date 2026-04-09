import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.tensor._linalg.constructors import block_diag
from pytensor.tensor.type import matrix
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def test_jax_block_diag():
    A = matrix("A")
    B = matrix("B")
    C = matrix("C")
    D = matrix("D")

    out = block_diag(A, B, C, D)

    compare_jax_and_py(
        [A, B, C, D],
        [out],
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
    out = block_diag(A, B)

    compare_jax_and_py(
        [A, B],
        [out],
        [
            np.random.normal(size=(5, 5, 5)).astype(config.floatX),
            np.random.normal(size=(5, 3, 3)).astype(config.floatX),
        ],
    )
