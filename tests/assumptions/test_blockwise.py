import pytest

import pytensor.tensor as pt
from pytensor.assumptions import (
    DIAGONAL,
    LOWER_TRIANGULAR,
    SYMMETRIC,
    UPPER_TRIANGULAR,
)
from pytensor.assumptions.specify import assume
from pytensor.tensor.basic import alloc_diag
from pytensor.tensor.blockwise import Blockwise
from tests.assumptions.conftest import make_fgraph


def test_blockwise_cholesky_is_lower_triangular():
    x = pt.tensor("x", shape=(5, 3, 3))
    L = pt.linalg.cholesky(x, lower=True)
    _, af = make_fgraph(L)
    assert af.check(L, LOWER_TRIANGULAR)


def test_blockwise_diagonal_propagation():
    x = pt.tensor("x", shape=(5, 3, 3))
    x_diag = assume(x, diagonal=True)
    L = pt.linalg.cholesky(x_diag, lower=True)
    _, af = make_fgraph(L)
    assert af.check(L, DIAGONAL)


@pytest.mark.parametrize(
    "key", [DIAGONAL, SYMMETRIC, LOWER_TRIANGULAR, UPPER_TRIANGULAR]
)
def test_blockwise_alloc_diag_properties(key):
    v_core = pt.vector("v", shape=(3,))
    d_core = alloc_diag(v_core, offset=0, axis1=0, axis2=1)

    bw = Blockwise(d_core.owner.op, signature="(n)->(n,n)")
    v_batch = pt.matrix("v_batch", shape=(5, 3))
    result = bw(v_batch)

    _, af = make_fgraph(result)

    assert af.check(result, key)
