import numpy as np
import scipy as sp

import pytensor
from pytensor.sparse.sharedvar import SparseTensorSharedVariable


def test_shared_basic():
    x = pytensor.shared(
        sp.sparse.csr_matrix(np.eye(100), dtype=np.float64), name="blah", borrow=True
    )

    assert isinstance(x, SparseTensorSharedVariable)
    assert x.format == "csr"
    assert x.dtype == "float64"
