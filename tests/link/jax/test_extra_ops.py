import numpy as np
import pytest

import pytensor.tensor.basic as ptb
from pytensor.configdefaults import config
from pytensor.tensor import extra_ops as pt_extra_ops
from pytensor.tensor.sort import argsort
from pytensor.tensor.type import matrix, tensor
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def test_extra_ops():
    a = matrix("a")
    a_test = np.arange(6, dtype=config.floatX).reshape((3, 2))

    out = pt_extra_ops.cumsum(a, axis=0)
    compare_jax_and_py([a], [out], [a_test])

    out = pt_extra_ops.cumprod(a, axis=1)
    compare_jax_and_py([a], [out], [a_test])

    out = pt_extra_ops.diff(a, n=2, axis=1)
    compare_jax_and_py([a], [out], [a_test])

    out = pt_extra_ops.repeat(a, (3, 3), axis=1)
    compare_jax_and_py([a], [out], [a_test])

    c = ptb.as_tensor(5)
    out = pt_extra_ops.fill_diagonal(a, c)
    compare_jax_and_py([a], [out], [a_test])

    with pytest.raises(NotImplementedError):
        out = pt_extra_ops.fill_diagonal_offset(a, c, c)
        compare_jax_and_py([a], [out], [a_test])

    with pytest.raises(NotImplementedError):
        out = pt_extra_ops.Unique(axis=1)(a)
        compare_jax_and_py([a], [out], [a_test])

    indices = np.arange(np.prod((3, 4)))
    out = pt_extra_ops.unravel_index(indices, (3, 4), order="C")
    compare_jax_and_py([], out, [], must_be_device_array=False)

    v = ptb.as_tensor_variable(6.0)
    sorted_idx = argsort(a.ravel())

    out = pt_extra_ops.searchsorted(a.ravel()[sorted_idx], v)
    compare_jax_and_py([a], [out], [a_test])


@pytest.mark.xfail(reason="Jitted JAX does not support dynamic shapes")
def test_bartlett_dynamic_shape():
    c = tensor(shape=(), dtype=int)
    out = pt_extra_ops.bartlett(c)
    compare_jax_and_py([], [out], [np.array(5)])


@pytest.mark.xfail(reason="Jitted JAX does not support dynamic shapes")
def test_ravel_multi_index_dynamic_shape():
    x_test, y_test = np.unravel_index(np.arange(np.prod((3, 4))), (3, 4))

    x = tensor(shape=(None,), dtype=int)
    y = tensor(shape=(None,), dtype=int)
    out = pt_extra_ops.ravel_multi_index((x, y), (3, 4))
    compare_jax_and_py([], [out], [x_test, y_test])


@pytest.mark.xfail(reason="Jitted JAX does not support dynamic shapes")
def test_unique_dynamic_shape():
    a = matrix("a")
    a_test = np.arange(6, dtype=config.floatX).reshape((3, 2))

    out = pt_extra_ops.Unique()(a)
    compare_jax_and_py([a], [out], [a_test])
