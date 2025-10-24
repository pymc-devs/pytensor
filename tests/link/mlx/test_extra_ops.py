import numpy as np
import pytest

from pytensor.configdefaults import config
from pytensor.tensor import extra_ops as pt_extra_ops
from pytensor.tensor.type import matrix
from tests.link.mlx.test_basic import compare_mlx_and_py


mx = pytest.importorskip("mlx.core")


def test_extra_ops():
    a = matrix("a")
    a_test = np.arange(6, dtype=config.floatX).reshape((3, 2))

    out = pt_extra_ops.cumsum(a, axis=0)
    compare_mlx_and_py([a], [out], [a_test])

    out = pt_extra_ops.cumprod(a, axis=1)
    compare_mlx_and_py([a], [out], [a_test])

    out = pt_extra_ops.repeat(a, 3, axis=1)
    compare_mlx_and_py([a], [out], [a_test])
