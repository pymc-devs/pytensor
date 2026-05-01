import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function
from pytensor.compile import optdb
from pytensor.compile.aliasing import add_supervisor_to_fgraph
from pytensor.compile.io import In
from pytensor.compile.mode import get_mode
from pytensor.graph.basic import equal_computations
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.basic import MakeVector, alloc, constant
from pytensor.tensor.blockwise import Blockwise, BlockwiseWithCoreShape
from pytensor.tensor.math import add as tensor_add
from pytensor.tensor.math import maximum, minimum
from pytensor.tensor.rewriting.elemwise import InplaceElemwiseOptimizer
from pytensor.tensor.rewriting.shape import ShapeFeature
from pytensor.tensor.shape import Shape_i
from pytensor.tensor.signal import convolve1d
from pytensor.tensor.signal.conv import Convolve1d


def count_ops(fgraph, op_type):
    return sum(isinstance(node.op, op_type) for node in fgraph.apply_nodes)


def rewrite_for_numba(inputs, outputs):
    fg = FunctionGraph(inputs, outputs)
    add_supervisor_to_fgraph(fg, [In(inp) for inp in fg.inputs])
    get_mode("NUMBA").optimizer.rewrite(fg)
    return fg


def core_shape_of(fgraph):
    [bwcs_node] = [
        n for n in fgraph.apply_nodes if isinstance(n.op, BlockwiseWithCoreShape)
    ]
    *functional_inputs, core_shape = bwcs_node.inputs
    return core_shape, functional_inputs


@pytest.mark.parametrize(
    "mode, x_shape, k_shape",
    [
        ("valid", (3, 10), (3, 4)),
        ("full", (3, 10), (3, 4)),
        ("valid", (None, None), (None, None)),
        ("full", (None, None), (None, None)),
    ],
)
def test_blockwise_core_shape_simplified(mode, x_shape, k_shape):
    x = pt.tensor("x", shape=x_shape)
    k = pt.tensor("k", shape=k_shape)
    out = Blockwise(Convolve1d())(
        x, k, constant(mode == "full", dtype="bool"), return_list=True
    )
    fn = function([x, k], out, mode="NUMBA")

    core_shape, _ = core_shape_of(fn.maker.fgraph)

    static = all(s is not None for s in x_shape + k_shape)
    if static:
        n, kk = x_shape[1], k_shape[1]
        expected_len = (n + kk - 1) if mode == "full" else (n - kk + 1)
        expected = constant(np.array([expected_len]))
        assert equal_computations([core_shape], [expected])
    else:
        n = Shape_i(1)(x)
        kk = Shape_i(1)(k)
        if mode == "full":
            expected = MakeVector("int64")(
                tensor_add(constant(-1, dtype="int64"), n, kk)
            )
        else:
            expected = MakeVector("int64")(
                constant(1, dtype="int64") + maximum(n, kk) - minimum(n, kk)
            )
        assert equal_computations([core_shape], [expected], in_xs=[x, k], in_ys=[x, k])


def test_introduce_core_shape_aliasing():
    """Graphs whose shape arithmetic gets inplaced, destroying variables that
    recursive core shape derivations used to read; they must simply lower.
    """
    larger = pt.matrix("larger", shape=(8, None))
    smaller = pt.matrix("smaller", shape=(8, None))
    a = alloc(pt.zeros((1, 1)), 1, larger.shape[1] + smaller.shape[1] - 1)
    out = convolve1d(a, larger[:, ::-1], mode="full")

    fg = rewrite_for_numba([larger, smaller], [out])
    assert count_ops(fg, BlockwiseWithCoreShape) == 1
    fg.toposort()

    # Crossed variant, where each core shape mixes dimensions of both inputs
    x1 = pt.matrix("x1", shape=(8, None))
    x2 = pt.matrix("x2", shape=(8, None))
    a1 = alloc(pt.zeros((1, 1)), 1, x1.shape[1] + 3)
    a2 = alloc(pt.zeros((1, 1)), 1, x2.shape[1] + 5)
    convA = convolve1d(a1, x2[:, ::-1], mode="full")
    convB = convolve1d(a2, x1[:, ::-1], mode="full")

    fg = rewrite_for_numba([x1, x2], [convA, convB])
    assert count_ops(fg, BlockwiseWithCoreShape) == 2
    fg.toposort()


def test_core_shape_simplify_keeps_fgraph_intact():
    """simplify_core_shape_graphs must not rewrite the fgraph's own applies,
    like the non-canonical chain feeding the alloc dim here.
    """
    x = pt.matrix("x", shape=(8, None))
    a = alloc(pt.zeros((1, 1)), 1, (x.shape[1] + 1) - 1)
    out = convolve1d(a, x[:, ::-1], mode="full")

    fg = FunctionGraph([x], [out], clone=False)
    fg.attach_feature(ShapeFeature())
    InplaceElemwiseOptimizer().rewrite(fg)
    assert any(node.op.destroy_map for node in fg.apply_nodes)
    optdb.query("+introduce_explicit_core_shape_blockwise").rewrite(fg)
    fg.toposort()
