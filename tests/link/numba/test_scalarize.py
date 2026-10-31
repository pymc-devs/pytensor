import numpy as np

import pytensor.tensor as pt
from pytensor import In
from pytensor import scalar as ps
from pytensor.compile.mode import get_mode
from pytensor.tensor.basic import scalar_from_tensor, tensor_from_scalar
from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise
from pytensor.tensor.rewriting.scalarize import (
    ScalarCAReduce,
    ScalarizedElemwise,
    ScalarizedJoin,
    ScalarizeRewriter,
    ScalarSubtensor,
)
from tests.link.numba.test_basic import compare_numba_and_py
from tests.unittest_tools import RewriteTester


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


def test_scalarized_elemwise_preserves_input_trace():
    # Regression test for https://github.com/pymc-devs/pytensor/issues/2585
    # Scalarizing several Elemwise clients of one 0d input must not grow its trace.
    x = pt.dscalar("x")
    x.tag.trace = [[("file.py", 1, "function", "line")]]
    tensor_ops = (pt.exp, pt.log, pt.sin, pt.cos)
    scalar_ops = (ps.exp, ps.log, ps.sin, ps.cos)
    clients = [scalar_from_tensor(op(x)) for op in tensor_ops]
    expected = [op(scalar_from_tensor(x)) for op in scalar_ops]

    result = RewriteTester(
        [x],
        clients,
        include=None,
        custom_rewrite=ScalarizeRewriter(),
    )
    result.assert_graph(*expected)
    [rewritten_x] = result.rewr_fg.inputs
    assert rewritten_x.tag.trace == x.tag.trace


def test_scalar_subtensor():
    # Subtensor consumes a scalar index and produces a scalar for its client.
    x = pt.dvector("x")
    index = ps.int64("index")
    output = x[index]
    client = scalar_from_tensor(output)
    expected = ScalarSubtensor([x, index], [client])(x, index)

    result = RewriteTester(
        [x, index],
        [client],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
        linker="numba",
    )
    result.assert_graph(expected)
    result.assert_eval(np.array([-3.0, 2.0]), 1)

    # Without a scalar client, Subtensor keeps its tensor output.
    result = RewriteTester(
        [x, index],
        [output],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
    )
    result.assert_graph(output)


def test_scalar_careduce():
    # A scalar client receives a float32 reduction accumulated in float64.
    x = pt.fvector("x")
    output = x.sum(dtype="float32", acc_dtype="float64")
    client = scalar_from_tensor(output)
    expected = ScalarCAReduce([x], [client])(x)

    result = RewriteTester(
        [x],
        [client],
        include=None,
        custom_rewrite=ScalarizeRewriter(),
        linker="numba",
    )
    result.assert_graph(expected)
    result.assert_eval(np.array([2.0, -1.0, 3.0], dtype="float32"))

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


def test_fused_indexed_write_of_scalarized_input():
    # The second use of exp(x) scalarizes it, so the fused indexed write has no
    # array input to read the loop rank from (pytensor#2591)
    x = pt.dscalar("x")
    s = pt.exp(x)
    updated = pt.set_subtensor(pt.zeros(1)[[0]], s[None] ** 2)

    fn, _ = compare_numba_and_py(
        [x], [updated, s * 2], [0.1], numba_mode=get_mode("NUMBA")
    )
    [fused] = [
        node
        for node in fn.maker.fgraph.apply_nodes
        if isinstance(node.op, FusedElemwise)
    ]
    assert isinstance(fused.inputs[0].type, ps.ScalarType)


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
