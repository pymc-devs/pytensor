import numpy as np
import pytest

import pytensor
from pytensor.compile import SharedVariable


sp = pytest.importorskip("scipy", minversion="0.7.0")


def test_shared_basic():
    x = pytensor.shared(sp.sparse.csr_matrix(np.eye(100)), name="blah", borrow=True)

    assert isinstance(x, SharedVariable)
