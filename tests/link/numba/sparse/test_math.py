import numpy as np
import pytest
import scipy

import pytensor.sparse as ps
import pytensor.tensor as pt
from tests.link.numba.sparse.test_basic import compare_numba_and_py_sparse


pytestmark = pytest.mark.filterwarnings("error")


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("y_ndim", [0, 1, 2])
def test_sparse_dense_multiply(y_ndim, format):
    ps_matrix = ps.csr_matrix if format == "csr" else ps.csc_matrix
    x = ps_matrix("x", shape=(3, 3))
    y = pt.tensor("y", shape=(3,) * y_ndim)
    z = x * y

    rng = np.random.default_rng((155, y_ndim, format == "csr"))
    x_test = scipy.sparse.random(3, 3, density=0.5, format=format, random_state=rng)
    y_test = rng.normal(size=(3,) * y_ndim)

    compare_numba_and_py_sparse(
        [x, y],
        z,
        [x_test, y_test],
    )
