import numpy as np
import pytest

from pytensor import function
from pytensor.compile import get_default_mode
from pytensor.tensor import matmul, tensor, vectorize
from pytensor.tensor.blas import BatchedDot
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.rewriting.blas import specialize_matmul_to_batched_dot


@pytest.mark.parametrize("valid_case", (True, False))
def test_specialize_matmul_to_batched_dot(valid_case):
    signature = BatchedDot.gufunc_signature
    rewrite = specialize_matmul_to_batched_dot.__name__

    def core_pt(x, y):
        return matmul(x, y)

    def core_np(x, y):
        return np.matmul(x, y)

    x = tensor(shape=(7, 5, 3, 3))
    if valid_case:
        y = tensor(shape=(7, 5, 3, 3))
    else:
        y = tensor(shape=(5, 3, 3))

    vectorize_pt = function(
        [x, y],
        vectorize(core_pt, signature=signature)(x, y),
        mode=get_default_mode().including(rewrite),
    )
    blocwkise_node = any(
        isinstance(node.op, Blockwise) for node in vectorize_pt.maker.fgraph.apply_nodes
    )
    if valid_case:
        assert not blocwkise_node
    else:
        assert blocwkise_node

    x_test = np.random.normal(size=x.type.shape).astype(x.type.dtype)
    y_test = np.random.normal(size=y.type.shape).astype(y.type.dtype)
    vectorize_np = np.vectorize(core_np, signature=signature)
    np.testing.assert_allclose(
        vectorize_pt(x_test, y_test),
        vectorize_np(x_test, y_test),
    )
