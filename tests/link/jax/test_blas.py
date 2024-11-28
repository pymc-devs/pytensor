import numpy as np
import pytest

from pytensor.compile.function import function
from pytensor.compile.mode import Mode
from pytensor.configdefaults import config
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.link.jax import JAXLinker
from pytensor.tensor import blas as pt_blas
from pytensor.tensor.type import tensor3
from tests.link.jax.test_basic import compare_jax_and_py


def test_jax_BatchedDot():
    # tensor3 . tensor3
    a = tensor3("a")
    a_test_value = (
        np.linspace(-1, 1, 10 * 5 * 3).astype(config.floatX).reshape((10, 5, 3))
    )
    b = tensor3("b")
    b_test_value = (
        np.linspace(1, -1, 10 * 3 * 2).astype(config.floatX).reshape((10, 3, 2))
    )
    out = pt_blas.BatchedDot()(a, b)
    compare_jax_and_py([a, b], [out], [a_test_value, b_test_value])

    # A dimension mismatch should raise a TypeError for compatibility
    inputs = [a_test_value[:-1], b_test_value]
    opts = RewriteDatabaseQuery(include=[None], exclude=["cxx_only", "BlasOpt"])
    jax_mode = Mode(JAXLinker(), opts)
    pytensor_jax_fn = function([a, b], [out], mode=jax_mode)
    with pytest.raises(TypeError):
        pytensor_jax_fn(*inputs)
