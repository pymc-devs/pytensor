import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from tests.link.numba.test_basic import compare_numba_and_py


pytestmark = pytest.mark.filterwarnings("error")

numba = pytest.importorskip("numba")

floatX = config.floatX


def test_block_diag():
    A = pt.matrix("A")
    B = pt.matrix("B")
    C = pt.matrix("C")
    D = pt.matrix("D")
    X = pt.linalg.block_diag(A, B, C, D)

    A_val = np.random.normal(size=(5, 5)).astype(floatX)
    B_val = np.random.normal(size=(3, 3)).astype(floatX)
    C_val = np.random.normal(size=(2, 2)).astype(floatX)
    D_val = np.random.normal(size=(4, 4)).astype(floatX)
    compare_numba_and_py([A, B, C, D], [X], [A_val, B_val, C_val, D_val])


def test_block_diag_with_read_only_inp():
    # Regression test where numba would complain a about *args containing both read-only and regular inputs
    # Currently, constants are read-only for numba, but for future-proofing we add an explicitly read-only input as well
    x = pt.tensor("x", shape=(2, 2))
    x_read_only = pt.tensor("x", shape=(2, 2))
    x_const = pt.constant(np.ones((2, 2), dtype=x.type.dtype), name="x_read_only")
    out = pt.linalg.block_diag(x, x_read_only, x_const)

    x_test = np.ones((2, 2), dtype=x.type.dtype)
    x_read_only_test = x_test.copy()
    x_read_only_test.flags.writeable = False
    compare_numba_and_py(
        [x, x_read_only],
        [out],
        [x_test, x_read_only_test],
    )
