import numpy as np
import pytest


sp = pytest.importorskip("scipy", minversion="0.7.0")

from pytensor.sparse.utils import hash_from_sparse
from tests.sparse.test_basic import as_sparse_format


def test_hash_from_sparse():
    hashes = []
    x = np.random.random((5, 5))

    for format in ["csc", "csr"]:
        x = as_sparse_format(x, format)
        for data in [
            [[-2]],
            [[-1]],
            [[0]],
            [[1]],
            [[2]],
            np.zeros((1, 5)),
            np.zeros((1, 6)),
            # Data buffer empty but different shapes
            # np.zeros((1, 0)), np.zeros((2, 0)),
            # Same data buffer and shapes but different strides
            np.arange(25).reshape(5, 5),
            np.arange(25).reshape(5, 5).T,
            # Same data buffer, shapes and strides
            # but different dtypes
            np.zeros((5, 5), dtype="uint32"),
            np.zeros((5, 5), dtype="int32"),
            # Test slice
            x,
            x[1:],
            x[:4],
            x[1:3],
            # Don't test step as they are not supported by sparse
            # rng[::2], rng[::-1]
        ]:
            data = as_sparse_format(data, format)

            hashes.append(hash_from_sparse(data))

        # test that different type of views and their copy give the same hash
        assert hash_from_sparse(x[1:]) == hash_from_sparse(x[1:].copy())
        assert hash_from_sparse(x[1:3]) == hash_from_sparse(x[1:3].copy())
        assert hash_from_sparse(x[:4]) == hash_from_sparse(x[:4].copy())

    assert len(set(hashes)) == len(hashes)
