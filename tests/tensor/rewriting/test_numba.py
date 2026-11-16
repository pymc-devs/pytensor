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


def test_chained_blockwise_core_shape():
    """A Blockwise consuming another Blockwise's output must still be lowered.

    Its core shape legitimately reads ``Shape_i`` of that input; only core
    shapes that depend on the node's own outputs are impossible to introduce.

    Regression test for https://github.com/pymc-devs/pytensor/issues/2360
    """
    x = pt.tensor("x", shape=(3, None))
    k = pt.tensor("k", shape=(3, None))
    out = convolve1d(convolve1d(x, k, mode="full"), k, mode="full")

    fg = rewrite_for_numba([x, k], [out])
    assert count_ops(fg, Blockwise) == 0
    assert count_ops(fg, BlockwiseWithCoreShape) == 2


def test_self_referential_core_shape_not_introduced():
    """When the core shape can only be obtained by evaluating the node itself
    (here the boolean mode varies across batch dims), the rewrite must bail.
    """
    x = pt.tensor("x", shape=(5, None))
    k = pt.tensor("k", shape=(5, None))
    m = pt.tensor("m", shape=(5,), dtype=bool)
    out = Blockwise(Convolve1d())(x, k, m)

    fg = rewrite_for_numba([x, k, m], [out])
    assert count_ops(fg, Blockwise) == 1
    assert count_ops(fg, BlockwiseWithCoreShape) == 0


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


@pytest.mark.parametrize("kind", ["set", "inc"])
@pytest.mark.parametrize("indexing", ["basic", "advanced"])
def test_basic_slice_write_fusion_2192(kind, indexing):
    from pytensor import config
    from pytensor.tensor.elemwise import Elemwise
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    x = pt.vector("x")
    destination = pt.vector("destination")
    indices = pt.lvector("indices")

    base = destination + 1
    region = base[1:] if indexing == "basic" else base[indices]
    values = pt.exp(x)

    result = region.set(values) if kind == "set" else region.inc(values)
    inputs = [x, destination]
    args = [
        np.array([0.0, 1.0, 2.0]),
        np.zeros(4),
    ]

    if indexing == "advanced":
        inputs.append(indices)
        args.append(np.array([1, 2, 3], dtype=np.int64))

    mode = get_mode("NUMBA")
    with config.change_flags(numba__cache=False):
        fn = function(inputs, result, mode=mode)
        actual = fn(*args)
        if indexing == "basic":
            fn_unfused = function(
                inputs,
                result,
                mode=mode.excluding("fuse_indexed_into_elemwise"),
            )
            fn_unfused(*args)
    expected = np.r_[
        1.0,
        np.exp(args[0]) + (0 if kind == "set" else 1),
    ]

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(args[1], np.zeros(4))

    assert any(
        isinstance(node.op, FusedElemwise) for node in fn.maker.fgraph.apply_nodes
    ), f"{kind}/{indexing}: expected fused slice write"

    if indexing == "basic":
        assert not any(
            isinstance(node.op, Elemwise) and type(node.op.scalar_op).__name__ == "Exp"
            for node in fn.maker.fgraph.apply_nodes
        )

        def allocations(compiled_fn):
            llvm = "\n".join(compiled_fn.vm.jit_fn.inspect_llvm().values())
            return llvm.count("NRT_MemInfo_alloc_aligned")

        assert allocations(fn) + 1 == allocations(fn_unfused), (
            "basic slice fusion should eliminate the standalone Exp output allocation"
        )


@pytest.mark.parametrize(
    ("slice_spec", "x_values"),
    [
        (slice(2, 4), np.array([0.0, 1.0])),
        (slice(1, 99), np.array([0.0, 1.0, 2.0])),
        (slice(9, None), np.array([])),
        (slice(2, 2), np.array([])),
        (slice(1, None), np.array([2.0])),
    ],
)
def test_basic_slice_write_fusion_2192_boundaries(slice_spec, x_values):
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    x = pt.vector("x")
    destination = pt.vector("destination")
    result = pt.set_subtensor((destination + 1)[slice_spec], pt.exp(x))
    fn = function([x, destination], result, mode="NUMBA")

    x_value = x_values.astype("float64")
    destination_value = np.zeros(4)
    actual = fn(x_value, destination_value)
    expected = destination_value + 1
    expected[slice_spec] = np.exp(x_value)
    np.testing.assert_allclose(actual, expected)
    assert any(
        isinstance(node.op, FusedElemwise) for node in fn.maker.fgraph.apply_nodes
    )


@pytest.mark.parametrize(
    ("slice_spec", "x_value"),
    [
        (slice(-2, None), np.array([0.0, 1.0])),
        (slice(None, None, 2), np.array([0.0, 1.0, 2.0])),
    ],
)
def test_basic_slice_write_fusion_2192_unsupported_slices_are_not_fused(
    slice_spec, x_value
):
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    x = pt.vector("x")
    destination = pt.vector("destination")
    result = pt.set_subtensor((destination + 1)[slice_spec], pt.exp(x))
    fn = function([x, destination], result, mode="NUMBA")

    destination_value = np.zeros(5)
    actual = fn(x_value, destination_value)
    expected = destination_value + 1
    expected[slice_spec] = np.exp(x_value)
    np.testing.assert_allclose(actual, expected)
    assert not any(
        isinstance(node.op, FusedElemwise) for node in fn.maker.fgraph.apply_nodes
    )


def test_basic_slice_write_fusion_2192_set_dtype_conversion():
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    x = pt.vector("x")
    destination = pt.vector("destination", dtype="int32")
    result = pt.set_subtensor((destination + 1)[1:], pt.exp(x))
    fn = function([x, destination], result, mode="NUMBA")

    actual = fn(np.array([0.0, 1.0]), np.array([4, 5, 6], dtype=np.int32))
    np.testing.assert_array_equal(actual, np.array([5, 1, 2], dtype=np.int32))
    assert any(
        isinstance(node.op, FusedElemwise) for node in fn.maker.fgraph.apply_nodes
    )


def test_basic_slice_write_fusion_2192_inc_dtype_mismatch_is_not_fused():
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    x = pt.vector("x")
    destination = pt.vector("destination", dtype="int32")
    result = pt.inc_subtensor((destination + 1)[1:], pt.exp(x))
    fn = function([x, destination], result, mode="NUMBA")

    actual = fn(np.array([0.0, 1.0]), np.array([4, 5, 6], dtype=np.int32))
    np.testing.assert_array_equal(actual, np.array([5, 7, 9], dtype=np.int32))
    assert not any(
        isinstance(node.op, FusedElemwise) for node in fn.maker.fgraph.apply_nodes
    )


def test_basic_slice_write_fusion_2192_shape_mismatch():
    x = pt.vector("x")
    destination = pt.vector("destination")
    result = pt.set_subtensor((destination + 1)[1:], pt.exp(x))
    fn = function([x, destination], result, mode="NUMBA")

    with pytest.raises(ValueError, match="broadcast"):
        fn(np.array([0.0, 1.0]), np.zeros(4))


def test_basic_slice_write_fusion_2192_dynamic_bounds_are_not_fused():
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    x = pt.vector("x")
    destination = pt.vector("destination")
    start = pt.lscalar("start")
    stop = pt.lscalar("stop")
    result = pt.set_subtensor((destination + 1)[start:stop], pt.exp(x))
    fn = function([x, destination, start, stop], result, mode="NUMBA")

    assert not any(
        isinstance(node.op, FusedElemwise) for node in fn.maker.fgraph.apply_nodes
    )
    x_value = np.array([0.0, 1.0])
    destination_value = np.zeros(4)
    for start_value, stop_value in ((0, 2), (2, 4)):
        actual = fn(x_value, destination_value, start_value, stop_value)
        expected = destination_value + 1
        expected[start_value:stop_value] = np.exp(x_value)
        np.testing.assert_allclose(actual, expected)


def test_basic_slice_write_fusion_2192_inc_shape_mismatch():
    x = pt.vector("x")
    destination = pt.vector("destination")
    result = pt.inc_subtensor((destination + 1)[1:], pt.exp(x))
    fn = function([x, destination], result, mode="NUMBA")

    with pytest.raises(ValueError, match="broadcast"):
        fn(np.array([0.0, 1.0]), np.zeros(4))


def test_basic_slice_write_fusion_2192_rejects_output_handling_core():
    from numba import TypingError

    from pytensor.link.numba.dispatch.vectorize_codegen import (
        _BasicOutputSlice,
        _CoreCall,
    )

    basic_output = object.__new__(_BasicOutputSlice)
    with pytest.raises(TypingError, match="not supported with basic slice writes"):
        _CoreCall(None, None, None, None, True, [basic_output])


@pytest.mark.parametrize("inplace", [False, True])
def test_basic_slice_write_fusion_2192_input_aliasing(inplace):
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    x = pt.vector("x")
    result = pt.set_subtensor(x[1:], pt.exp(x[:-1]), inplace=inplace)
    fn = function(
        [In(x, mutable=inplace)],
        result,
        mode="NUMBA",
        accept_inplace=inplace,
    )

    x_value = np.array([0.0, 1.0, 2.0, 3.0])
    original = x_value.copy()
    actual = fn(x_value)
    expected = original.copy()
    expected[1:] = np.exp(original[:-1])
    np.testing.assert_allclose(actual, expected)

    fused = any(
        isinstance(node.op, FusedElemwise) for node in fn.maker.fgraph.apply_nodes
    )
    assert fused is not inplace
    if inplace:
        np.testing.assert_allclose(x_value, expected)
    else:
        np.testing.assert_array_equal(x_value, original)
