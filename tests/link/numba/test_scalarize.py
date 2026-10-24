import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function
from pytensor.compile.mode import get_mode
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.basic import MakeVector, TensorFromScalar
from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise
from pytensor.tensor.rewriting.scalarize import (
    ScalarizedElemwise,
    ScalarJoin,
    ScalarSubtensor,
)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_scalar_index_reduction_and_mixed_join(dtype):
    parameters = pt.vector("parameters", dtype=dtype, shape=(3,))
    data = pt.vector("data", dtype=dtype)
    value = ((data - parameters[0]) * parameters[1]).sum()
    output = pt.join(0, value[None], parameters[2:])
    fn = function([parameters, data], output, mode="NUMBA")
    nodes = fn.maker.fgraph.toposort()
    assert any(isinstance(n.op, ScalarSubtensor) for n in nodes)
    assert any(isinstance(n.op, ScalarJoin) for n in nodes)
    assert any(
        isinstance(n.op, FusedElemwise)
        and any(isinstance(out.type, ScalarType) for out in n.outputs)
        for n in nodes
    )
    for size in (0, 7):
        x = np.array([0.2, 1.5, -0.3], dtype=dtype)
        y = np.arange(size, dtype=dtype)[::-1]
        expected = np.array([((y - x[0]) * x[1]).sum(), x[2]], dtype=dtype)
        np.testing.assert_allclose(fn(x, y), expected, rtol=1e-6)


def test_make_vector_mixed_entries():
    parameters = pt.vector("parameters", shape=(2,))
    counts = pt.lvector("counts", shape=(1,))
    offset = pt.scalar("offset")
    output = pt.stack(
        [pt.exp(parameters[0]) * parameters[1], counts[0], offset, parameters[1]]
    )
    fn = function([parameters, counts, offset], output, mode="NUMBA")
    nodes = fn.maker.fgraph.toposort()
    assert any(isinstance(n.op, ScalarJoin) for n in nodes)
    assert not any(isinstance(n.op, MakeVector | TensorFromScalar) for n in nodes)
    x = np.array([0.5, -2.0])
    np.testing.assert_allclose(
        fn(x, np.array([3]), np.array(1.5)), [np.exp(x[0]) * x[1], 3, 1.5, x[1]]
    )


def test_fused_scalar_accumulator_cast_and_output_ownership():
    parameters = pt.vector("parameters", dtype="float32", shape=(1,))
    data = pt.matrix("data", dtype="float32")
    reduced = pt.sum(data * parameters[0], dtype="float32", acc_dtype="float64")
    fn = function([parameters, data], reduced, mode="NUMBA")
    x = np.array([2.0], dtype="float32")
    y = np.arange(12, dtype="float32").reshape(3, 4)
    first = fn(x, y[:, ::-1])
    assert first.dtype == np.float32
    np.testing.assert_allclose(first, 132)
    second = fn(x, np.zeros((0, 4), dtype="float32"))
    np.testing.assert_allclose(second, 0)
    np.testing.assert_allclose(first, 132)


def test_scalarized_graph_python_fallback():
    parameters = pt.vector("parameters", shape=(2,))
    data = pt.vector("data")
    output = pt.join(0, ((data - parameters[0]) ** 2).sum()[None], parameters[1:])
    mode = get_mode("NUMBA").clone(linker=get_mode("FAST_COMPILE").linker)
    fn = function([parameters, data], output, mode=mode)
    np.testing.assert_allclose(fn(np.array([2.0, 4.0]), np.array([1.0, 3.0])), [2, 4])


def test_unit_tensor_inputs_and_post_fusion_boundaries():
    parameters = pt.vector("parameters", shape=(2,))
    data = pt.vector("data")
    unit = parameters[1:]
    reduced = (data * parameters[0]).sum()
    output = pt.join(0, reduced[None] * unit, reduced[None] + unit)
    fn = function([parameters, data], output, mode="NUMBA")
    nodes = fn.maker.fgraph.toposort()
    assert any(isinstance(n.op, ScalarJoin) for n in nodes)
    assert not any(isinstance(n.op, ScalarizedElemwise) for n in nodes)
    for x in (np.array([2.0, 3.0]), np.array([-0.5, -1.0])):
        y = np.arange(7.0)[::-1]
        reduced_value = (y * x[0]).sum()
        np.testing.assert_allclose(
            fn(x, y), [reduced_value * x[1], reduced_value + x[1]]
        )
