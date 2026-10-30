import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import In
from pytensor import scalar as ps
from pytensor.gradient import DisconnectedType
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.basic import scalar_from_tensor, tensor_from_scalar
from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise
from pytensor.tensor.rewriting.scalarize import (
    ScalarCAReduce,
    ScalarizedAlloc,
    ScalarizedElemwise,
    ScalarizedIncSubtensor,
    ScalarizedJoin,
    ScalarizeRewriter,
    ScalarSubtensor,
)
from tests.link.numba.test_basic import compare_numba_and_py
from tests.unittest_tools import RewriteTester


def test_scalarized_allocempty_not_constant_folded():
    # A constant shape does not make uninitialized contents a constant value.
    size = ps.int64("size")
    empty = ScalarizedAlloc(
        [size], [pt.empty(tensor_from_scalar(size), dtype="float32")]
    )(ps.constant(2, dtype="int64"))
    RewriteTester([], [empty]).assert_graph(empty)


@pytest.mark.parametrize("linker", ["numba", "py"])
def test_scalarized_allocempty(linker):
    # Dynamic shape arithmetic stays native while the static middle axis survives.
    n, m = pt.lscalars("n", "m")
    empty = pt.empty((n + 1, 2, m), dtype="float32")
    shape0, shape2 = ps.int64("shape0"), ps.int64("shape2")
    inner = pt.empty(
        (tensor_from_scalar(shape0), 2, tensor_from_scalar(shape2)), dtype="float32"
    )
    addition = empty.owner.inputs[0].owner
    native_sum = addition.op.scalar_op(
        *(scalar_from_tensor(inp) for inp in addition.inputs)
    )
    expected = ScalarizedAlloc([shape0, shape2], [inner])(
        native_sum, scalar_from_tensor(m)
    )
    result = RewriteTester(
        [n, m],
        [empty.shape],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
        linker=linker,
    )
    result.assert_graph(expected.shape)
    for values in ((2, 3), (-1, 3), (2, 0)):
        result.assert_eval(*values)
    allocation_error = ValueError if linker == "numba" else (ValueError, MemoryError)
    for fn in (result.orig_fn, result.rewr_fn):
        for values in ((-2, 3), (2, -1)):
            with pytest.raises(allocation_error):
                fn(*values)
    [node] = [
        node
        for node in result.rewr_fg.toposort()
        if isinstance(node.op, ScalarizedAlloc)
    ]
    assert all(isinstance(inp.type, ScalarType) for inp in node.inputs)
    assert all(
        isinstance(derivative.type, DisconnectedType)
        for derivative in pytensor.grad(
            node.outputs[0].sum(),
            node.inputs,
            disconnected_inputs="ignore",
            return_disconnected="disconnected",
        )
    )


@pytest.mark.parametrize("set_value", [False, True])
def test_scalarized_incsubtensor(set_value):
    # A scalar read feeds the update directly, without boxing its value again.
    index = slice(None, None, -2)
    x = pt.dvector("x")
    values = pt.dvector("values")
    scalar = ps.float64("scalar")
    read_index = ps.int64("read_index")
    indexed_value = values[0]
    output = x[index].set(indexed_value) if set_value else x[index].inc(indexed_value)
    indexed = ScalarSubtensor(
        [values, read_index], [scalar_from_tensor(values[read_index])]
    )(values, indexed_value.owner.inputs[1])
    indices = output.owner.inputs[2:]
    inner_indices = [inp.type() for inp in indices]
    inner_indexed = output.owner.op(x, tensor_from_scalar(scalar), *inner_indices)
    expected = ScalarizedIncSubtensor([x, scalar, *inner_indices], [inner_indexed])(
        x, indexed, *indices
    )

    result = RewriteTester(
        [x, values],
        [output],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
        linker="numba",
    )
    result.assert_graph(expected)
    result.assert_eval(np.arange(6, dtype="float64"), np.array([1.25]))


def test_scalarized_fused_singleton_reduction():
    # The fused reduction produces a scalar while retaining a singleton axis.
    x = pt.fmatrix("x", shape=(None, 1))
    y = pt.fmatrix("y", shape=(None, 1))
    reduction = (x + y).sum(axis=0, dtype="float32", acc_dtype="float64")
    specs = ((ps.add, (0,), 0.0, "float64"),)
    output = FusedElemwise(
        [x, y],
        [reduction],
        reduced_outputs=specs,
    )(x, y)
    client = scalar_from_tensor(output.squeeze())
    expected = FusedElemwise(
        [x, y],
        [scalar_from_tensor((x + y).sum(dtype="float32", acc_dtype="float64"))],
        reduced_outputs=((ps.add, (0, 1), 0.0, "float64"),),
    )(x, y)

    result = RewriteTester(
        [x, y],
        [client],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
        linker="numba",
    )
    result.assert_graph(expected)
    for size in (3, 0):
        result.assert_eval(
            np.arange(size, dtype="float32")[:, None],
            np.ones((size, 1), dtype="float32"),
        )


def test_scalarized_elemwise():
    # Elemwise consumes an existing scalar while keeping its array output.
    scalar = ps.float64("scalar")
    x = pt.dvector("x")
    output = tensor_from_scalar(scalar) - x
    expected = ScalarizedElemwise([scalar, x], [output])(scalar, x)

    result = RewriteTester(
        [scalar, x],
        [output],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
        linker="numba",
    )
    result.assert_graph(expected)
    result.assert_eval(2.0, np.array([1.0, 3.0]))

    # Elemwise with unit inputs produces a scalar for its client.
    unit = pt.dmatrix("unit", shape=(1, 1))
    output = pt.exp(unit)
    client = scalar_from_tensor(output.squeeze())
    expected = ps.exp(scalar_from_tensor(unit.squeeze()))

    result = RewriteTester(
        [unit],
        [client],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
        linker="numba",
    )
    result.assert_graph(expected)
    result.assert_eval(np.array([[0.5]]))

    # An isolated Elemwise keeps its tensor output.
    result = RewriteTester(
        [unit],
        [output],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
    )
    result.assert_graph(output)


@pytest.mark.parametrize("trailing_shape", [(), (1, 1)])
def test_scalar_subtensor(trailing_shape):
    # Subtensor consumes a scalar index and produces a scalar for its client.
    x = pt.tensor("x", dtype="float64", shape=(None, *trailing_shape))
    index = ps.int64("index")
    output = x[index]
    client = scalar_from_tensor(output.squeeze())
    indexed = x[(index,) + (0,) * len(trailing_shape)]
    expected = ScalarSubtensor(
        indexed.owner.inputs,
        [scalar_from_tensor(indexed)],
    )(*indexed.owner.inputs)

    result = RewriteTester(
        [x, index],
        [client],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
        linker="numba",
    )
    result.assert_graph(expected)
    result.assert_eval(np.array([-3.0, 2.0]).reshape(2, *trailing_shape), 1)

    # Without a scalar client, Subtensor keeps its tensor output.
    result = RewriteTester(
        [x, index],
        [output],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
    )
    result.assert_graph(output)


@pytest.mark.parametrize("trailing_shape", [(), (1, 1)])
def test_scalar_careduce(trailing_shape):
    # A scalar client receives a float32 reduction accumulated in float64.
    x = pt.tensor("x", dtype="float32", shape=(None, *trailing_shape))
    output = x.sum(axis=0, dtype="float32", acc_dtype="float64")
    client = scalar_from_tensor(output.squeeze())
    expected = ScalarCAReduce(
        [x],
        [scalar_from_tensor(x.sum(dtype="float32", acc_dtype="float64"))],
    )(x)

    result = RewriteTester(
        [x],
        [client],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
        linker="numba",
    )
    result.assert_graph(expected)
    result.assert_eval(
        np.array([2.0, -1.0, 3.0], dtype="float32").reshape(3, *trailing_shape)
    )
    result.assert_eval(np.empty((0, *trailing_shape), dtype="float32"))

    # Without a scalar client, the reduction keeps its tensor output.
    result = RewriteTester(
        [x],
        [output],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
    )
    result.assert_graph(output)


def test_fused_elemwise():
    # The fused indexed read accepts a scalar input and returns an array.
    x = pt.dvector("x")
    offset = pt.dscalar("offset")
    indices = pt.lvector("indices")
    scalar = ps.float64("scalar")
    indexed_inputs = (((0,), 0),)
    op = FusedElemwise(
        [x, offset, indices],
        [x[indices] + offset],
        indexed_inputs=indexed_inputs,
    )
    output = op(x, tensor_from_scalar(scalar), indices)
    expected = FusedElemwise(
        [x, scalar, indices],
        [x[indices] + tensor_from_scalar(scalar)],
        indexed_inputs=indexed_inputs,
    )(x, scalar, indices)

    result = RewriteTester(
        [x, scalar, indices],
        [output],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
        linker="numba",
    )
    result.assert_graph(expected)
    result.assert_eval(np.array([1.0, 2.0, 3.0]), 2.0, np.array([2, 0]))

    # The fused reduction produces a scalar without needing scalar inputs.
    x, y = pt.fvectors("x", "y")
    reduction = (x + y).sum(dtype="float32", acc_dtype="float64")
    reduced_outputs = ((ps.add, (0,), 0.0, "float64"),)
    op = FusedElemwise([x, y], [reduction], reduced_outputs=reduced_outputs)
    output = op(x, y)
    client = scalar_from_tensor(output)
    expected = FusedElemwise(
        [x, y],
        [scalar_from_tensor(reduction)],
        reduced_outputs=reduced_outputs,
    )(x, y)

    result = RewriteTester(
        [x, y],
        [client],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
        linker="numba",
    )
    result.assert_graph(expected)
    result.assert_eval(
        np.array([2.0, -1.0, 3.0], dtype="float32"),
        np.array([1.0, 2.0, 4.0], dtype="float32"),
    )

    # Without a scalar client, the fused reduction keeps its tensor output.
    result = RewriteTester(
        [x, y],
        [output],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
    )
    result.assert_graph(output)


def test_scalarized_join():
    # MakeVector passes an existing scalar directly and converts the tensor entry.
    scalar = ps.float64("scalar")
    other = ps.float64("other")
    offset = pt.dscalar("offset")
    output = pt.stack([tensor_from_scalar(scalar), offset])
    op = ScalarizedJoin(
        [scalar, other],
        [pt.join(0, tensor_from_scalar(scalar)[None], tensor_from_scalar(other)[None])],
    )
    expected = op(scalar, scalar_from_tensor(offset))

    result = RewriteTester(
        [scalar, offset],
        [output],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
        linker="numba",
    )
    result.assert_graph(expected)
    result.assert_eval(2.0, np.array(3.0))

    # Join accepts vector segments on both sides of a scalar input.
    x, y = pt.dvectors("x", "y")
    output = pt.join(0, x, tensor_from_scalar(scalar)[None], y)
    expected = ScalarizedJoin([x, scalar, y], [output])(x, scalar, y)

    result = RewriteTester(
        [x, scalar, y],
        [output],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
        linker="numba",
    )
    result.assert_graph(expected)
    result.assert_eval(np.array([-1.0, 3.0]), 2.0, np.array([4.0, 5.0, 6.0]))

    # Without a scalar input, Join stays unchanged.
    output = pt.join(0, offset[None], x)
    result = RewriteTester(
        [offset, x],
        [output],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
    )
    result.assert_graph(output)


def test_scalarized_graph():
    # Subtensor feeds the same scalar to both Elemwise and Join.
    x, y = pt.dvectors("x", "y")
    index = ps.int64("index")
    scalar = ps.float64("scalar")
    indexed = x[index]
    output = pt.join(0, indexed[None], indexed - y)

    indexed_scalar = ScalarSubtensor([x, index], [scalar_from_tensor(indexed)])(
        x, index
    )
    difference = ScalarizedElemwise(
        [scalar, y],
        [tensor_from_scalar(scalar) - y],
    )(indexed_scalar, y)
    expected = ScalarizedJoin(
        [scalar, y],
        [pt.join(0, tensor_from_scalar(scalar)[None], y)],
    )(indexed_scalar, difference)

    result = RewriteTester(
        [x, y, index],
        [output],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
        linker="numba",
    )
    result.assert_graph(expected)
    result.assert_eval(np.array([-3.0, 0.0]), np.array([1.0, 4.0]), 0)

    # Returning the indexed tensor too preserves one TensorFromScalar bridge.
    outputs = [indexed, output]
    result = RewriteTester(
        [x, y, index],
        outputs,
        include=None,
        custom_rewrite=ScalarizeRewriter(),
        linker="numba",
    )
    result.assert_graph(tensor_from_scalar(indexed_scalar), expected)
    result.assert_eval(np.array([-3.0, 0.0]), np.array([1.0, 4.0]), 0)


def test_scalarized_elemwise_inplace():
    x = pt.dvector("x")
    parameters = pt.dvector("parameters", shape=(2,))
    output = parameters[0] - x

    fn, _ = compare_numba_and_py(
        [In(x, mutable=True), parameters],
        output,
        [np.array([2.0, -1.0, 3.0]), np.array([3.0, 0.0])],
        numba_mode="NUMBA",
        inplace=True,
    )

    [node] = [
        n for n in fn.maker.fgraph.apply_nodes if isinstance(n.op, ScalarizedElemwise)
    ]
    assert node.op.destroy_map == {0: [1]}


def test_recompile_scalarized_graph():
    parameters = pt.dvector("parameters", shape=(2,))
    data = pt.dvector("data")
    output = pt.join(0, (data * parameters[0]).sum()[None], parameters[0] - data)

    # Initial compilation prepares the scalarized graph.
    inputs = [np.array([2.0, 4.0]), np.array([1.0, 3.0])]
    fn, _ = compare_numba_and_py([parameters, data], output, inputs, numba_mode="NUMBA")

    # Recompilation preserves the wrappers' inner graphs.
    scalarized_outputs = fn.maker.fgraph.outputs
    compare_numba_and_py(
        [parameters, data], scalarized_outputs, inputs, numba_mode="NUMBA"
    )
