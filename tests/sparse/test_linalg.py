import numpy as np
import pytest

from pytensor.configdefaults import config
from pytensor.sparse.linalg import block_diag


sp = pytest.importorskip("scipy", minversion="0.7.0")


@pytest.mark.parametrize("format", ["csc", "csr"], ids=["csc", "csr"])
@pytest.mark.parametrize("sparse_input", [True, False], ids=["sparse", "dense"])
def test_block_diagonal(format, sparse_input):
    from scipy import sparse as sp_sparse

    f_array = sp_sparse.csr_matrix if sparse_input else np.array
    A = f_array([[1, 2], [3, 4]]).astype(config.floatX)
    B = f_array([[5, 6], [7, 8]]).astype(config.floatX)

    result = block_diag(A, B, format=format)
    assert result.owner.op._props_dict() == {"n_inputs": 2, "format": format}

    sp_result = sp_sparse.block_diag([A, B], format=format)

    assert isinstance(result.eval(), type(sp_result))
    np.testing.assert_allclose(result.eval().toarray(), sp_result.toarray())
