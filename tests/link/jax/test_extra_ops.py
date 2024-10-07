import numpy as np
import pytest

import pytensor.tensor.basic as ptb
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import get_test_value
from pytensor.tensor import extra_ops as pt_extra_ops
from pytensor.tensor.type import matrix, tensor
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def test_extra_ops():
    a = matrix("a")
    a_test = np.arange(6, dtype=config.floatX).reshape((3, 2))

    out = pt_extra_ops.cumsum(a, axis=0)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [a_test])

    out = pt_extra_ops.cumprod(a, axis=1)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [a_test])

    out = pt_extra_ops.diff(a, n=2, axis=1)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [a_test])

    out = pt_extra_ops.repeat(a, (3, 3), axis=1)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [a_test])

    c = ptb.as_tensor(5)
    out = pt_extra_ops.fill_diagonal(a, c)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [a_test])

    with pytest.raises(NotImplementedError):
        out = pt_extra_ops.fill_diagonal_offset(a, c, c)
        fgraph = FunctionGraph([a], [out])
        compare_jax_and_py(fgraph, [a_test])

    with pytest.raises(NotImplementedError):
        out = pt_extra_ops.Unique(axis=1)(a)
        fgraph = FunctionGraph([a], [out])
        compare_jax_and_py(fgraph, [a_test])

    indices = np.arange(np.prod((3, 4)))
    out = pt_extra_ops.unravel_index(indices, (3, 4), order="C")
    fgraph = FunctionGraph([], out)
    compare_jax_and_py(
        fgraph, [get_test_value(i) for i in fgraph.inputs], must_be_device_array=False
    )


@pytest.mark.xfail(reason="Jitted JAX does not support dynamic shapes")
def test_bartlett_dynamic_shape():
    c = tensor(shape=(), dtype=int)
    out = pt_extra_ops.bartlett(c)
    fgraph = FunctionGraph([], [out])
    compare_jax_and_py(fgraph, [np.array(5)])


@pytest.mark.xfail(reason="Jitted JAX does not support dynamic shapes")
def test_ravel_multi_index_dynamic_shape():
    x_test, y_test = np.unravel_index(np.arange(np.prod((3, 4))), (3, 4))

    x = tensor(shape=(None,), dtype=int)
    y = tensor(shape=(None,), dtype=int)
    out = pt_extra_ops.ravel_multi_index((x, y), (3, 4))
    fgraph = FunctionGraph([], [out])
    compare_jax_and_py(fgraph, [x_test, y_test])


@pytest.mark.xfail(reason="Jitted JAX does not support dynamic shapes")
def test_unique_dynamic_shape():
    a = matrix("a")
    a_test = np.arange(6, dtype=config.floatX).reshape((3, 2))

    out = pt_extra_ops.Unique()(a)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [a_test])
