import numpy as np
import scipy

from pytensor import tensor as pt
from pytensor.configdefaults import config
from pytensor.tensor.linalg import block_diag
from tests import unittest_tools as utt


def test_block_diagonal():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    result = block_diag(A, B)
    assert result.type.shape == (4, 4)
    assert result.owner.op.core_op._props_dict() == {"n_inputs": 2}

    np.testing.assert_allclose(result.eval(), scipy.linalg.block_diag(A, B))


def test_block_diagonal_static_shape():
    A = pt.dmatrix("A", shape=(5, 5))
    B = pt.dmatrix("B", shape=(3, 10))
    result = block_diag(A, B)
    assert result.type.shape == (8, 15)

    A = pt.dmatrix("A", shape=(5, 5))
    B = pt.dmatrix("B", shape=(3, None))
    result = block_diag(A, B)
    assert result.type.shape == (8, None)

    A = pt.dmatrix("A", shape=(None, 5))
    result = block_diag(A, B)
    assert result.type.shape == (None, None)


def test_block_diagonal_grad():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])

    utt.verify_grad(block_diag, pt=[A, B], rng=np.random.default_rng())


def test_block_diagonal_blockwise():
    batch_size = 5
    A = np.random.normal(size=(batch_size, 2, 2)).astype(config.floatX)
    B = np.random.normal(size=(batch_size, 4, 4)).astype(config.floatX)
    result = block_diag(A, B).eval()
    assert result.shape == (batch_size, 6, 6)
    for i in range(batch_size):
        np.testing.assert_allclose(
            result[i],
            scipy.linalg.block_diag(A[i], B[i]),
            atol=1e-4 if config.floatX == "float32" else 1e-8,
            rtol=1e-4 if config.floatX == "float32" else 1e-8,
        )

    # Test broadcasting
    A = np.random.normal(size=(10, batch_size, 2, 2)).astype(config.floatX)
    B = np.random.normal(size=(1, batch_size, 4, 4)).astype(config.floatX)
    result = block_diag(A, B).eval()
    assert result.shape == (10, batch_size, 6, 6)
