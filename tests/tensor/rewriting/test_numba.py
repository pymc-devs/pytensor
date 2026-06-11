import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function
from pytensor.compile import optdb
from pytensor.compile.aliasing import add_supervisor_to_fgraph
from pytensor.compile.io import In
from pytensor.compile.mode import get_mode
from pytensor.compile.ops import DeepCopyOp
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

    [bwcs_node] = [
        n
        for n in fn.maker.fgraph.apply_nodes
        if isinstance(n.op, BlockwiseWithCoreShape)
    ]
    core_shape = bwcs_node.inputs[-1]

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
