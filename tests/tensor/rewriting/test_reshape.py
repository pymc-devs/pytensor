import numpy as np

import pytensor
from pytensor.graph import FunctionGraph, rewrite_graph
from pytensor.graph.traversal import apply_ancestors
from pytensor.tensor.basic import constant, expand_dims
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.extra_ops import squeeze
from pytensor.tensor.math import exp
from pytensor.tensor.reshape import JoinDims, SplitDims, join_dims, split_dims
from pytensor.tensor.shape import Reshape, specify_shape
from pytensor.tensor.type import tensor
from tests.unittest_tools import assert_equal_computations


def _count(out, cls):
    return sum(isinstance(node.op, cls) for node in apply_ancestors([out]))


def test_local_split_dims_general_persists():
    x = tensor("x", shape=(2, 10, 3))
    x_split = split_dims(x, axis=1, shape=(2, 5, 1))

    fg = FunctionGraph(inputs=[x], outputs=[x_split])

    assert sum([1 for node in fg.toposort() if isinstance(node.op, SplitDims)]) == 1

    rewrite_graph(fg, include=("canonicalize",))

    # The general split is dispatched natively; it is not lowered to Reshape.
    assert sum([1 for node in fg.toposort() if isinstance(node.op, SplitDims)]) == 1
    assert sum([1 for node in fg.toposort() if isinstance(node.op, Reshape)]) == 0
    assert fg.outputs[0].type.shape == (2, 2, 5, 1, 3)


def test_local_join_dims_general_persists():
    x = tensor("x", shape=(2, 2, 5, 1, 3))
    x_join = join_dims(x, start_axis=1, n_axes=3)

    fg = FunctionGraph(inputs=[x], outputs=[x_join])

    assert sum([1 for node in fg.toposort() if isinstance(node.op, JoinDims)]) == 1

    rewrite_graph(fg, include=("canonicalize",))

    # The general join is dispatched natively; it is not lowered to Reshape.
    assert sum([1 for node in fg.toposort() if isinstance(node.op, JoinDims)]) == 1
    assert sum([1 for node in fg.toposort() if isinstance(node.op, Reshape)]) == 0
    assert fg.outputs[0].type.shape == (2, 10, 3)


def test_local_join_dims_noop():
    """Test that join_dims with n_axes=1 becomes identity (no-op)."""
    x = tensor("x", shape=(2, 3, 4))
    x_join = join_dims(x, start_axis=1, n_axes=1)

    assert (
        sum([isinstance(node.op, JoinDims) for node in apply_ancestors([x_join])]) == 1
    )
    rewritten = rewrite_graph(x_join, include=("canonicalize",))
    assert_equal_computations([rewritten], [x])


def test_local_join_dims_to_expand_dims():
    """Test that join_dims with n_axes=0 becomes expand_dims."""
    x = tensor("x", shape=(2, 3, 4))
    x_join = join_dims(x, start_axis=1, n_axes=0)

    assert (
        sum([isinstance(node.op, JoinDims) for node in apply_ancestors([x_join])]) == 1
    )
    rewritten = rewrite_graph(x_join, include=("canonicalize",))
    # Output shape should be (2, 1, 3, 4) - new dimension of size 1 inserted at axis 1
    expected = expand_dims(x, axis=1)
    assert_equal_computations([rewritten], [expected])


def test_local_split_dims_to_squeeze():
    """Test that split_dims with shape tensor of static shape (0,) becomes squeeze via merged rewrite."""
    x = tensor("x", shape=(2, 1, 3, 4))
    x_split = split_dims(x, axis=1, shape=())

    assert (
        sum([isinstance(node.op, SplitDims) for node in apply_ancestors([x_split])])
        == 1
    )
    rewritten = rewrite_graph(x_split, include=("canonicalize",))
    # Output shape should be (2, 3, 4) - dimension 1 removed
    expected = squeeze(x, axis=1)
    assert_equal_computations([rewritten], [expected])


def test_local_split_dims_to_specify_shape():
    """Test that split_dims with shape tensor of static shape (0,) becomes squeeze via merged rewrite."""
    x = tensor("x", shape=(2, None, 4))
    x_split = split_dims(x, axis=1, shape=(5,))

    assert (
        sum([isinstance(node.op, SplitDims) for node in apply_ancestors([x_split])])
        == 1
    )
    rewritten = rewrite_graph(x_split, include=("canonicalize",))
    # Output shape should be (2, 3, 4) - dimension 1 removed
    expected = specify_shape(x, (None, 5, None))
    assert_equal_computations([rewritten], [expected], strict_dtype=False)


def test_local_join_dims_squeeze():
    """A size-1 dim inside a join span is squeezed out before joining."""
    x = tensor("x", shape=(2, 1, 3, 4))
    out = join_dims(x, start_axis=1, n_axes=2)  # merge the size-1 and the '3'
    rewritten = rewrite_graph(out, include=("canonicalize",))
    # Only the size-1 dim is removed; the remaining single axis needs no join.
    assert _count(rewritten, JoinDims) == 0
    expected = squeeze(x, axis=1)
    assert_equal_computations([rewritten], [expected])

    # With two genuine dims in the span, a join survives after the squeeze.
    y = tensor("y", shape=(2, 1, 3, 4))
    out2 = join_dims(y, start_axis=1, n_axes=3)  # merge size-1, '3', '4'
    rewritten2 = rewrite_graph(out2, include=("canonicalize",))
    assert _count(rewritten2, JoinDims) == 1
    fn = pytensor.function([y], rewritten2)
    yv = np.arange(24.0).reshape(2, 1, 3, 4)
    np.testing.assert_allclose(fn(yv), yv.reshape(2, 12))


def test_local_split_dims_expand():
    """Size-1 split factors become expand_dims around a split of the real factors."""
    x = tensor("x", shape=(6, 4))
    out = split_dims(x, shape=(2, 1, 3), axis=0)  # (2, 1, 3, 4)
    rewritten = rewrite_graph(out, include=("canonicalize",))
    assert _count(rewritten, SplitDims) == 1
    fn = pytensor.function([x], rewritten)
    xv = np.arange(24.0).reshape(6, 4)
    np.testing.assert_allclose(fn(xv), xv.reshape(2, 1, 3, 4))

    # A single genuine factor collapses to a pure expand_dims (no split left).
    y = tensor("y", shape=(6, 4))
    out2 = split_dims(y, shape=(1, 6, 1), axis=0)  # (1, 6, 1, 4)
    rewritten2 = rewrite_graph(out2, include=("canonicalize",))
    assert _count(rewritten2, SplitDims) == 0
    expected = expand_dims(y, axis=(0, 2))
    assert_equal_computations([rewritten2], [expected])


def test_local_join_of_split_cancels():
    """``JoinDims(SplitDims(x, s))`` over the same span cancels to ``x``."""
    x = tensor("x", shape=(2, 12, 5))
    out = join_dims(split_dims(x, shape=(3, 4), axis=1), start_axis=1, n_axes=2)
    rewritten = rewrite_graph(out, include=("canonicalize",))
    assert rewritten is x


def test_local_join_of_split_span_mismatch_stays():
    """A join straddling a split boundary is a genuine mix and does not cancel."""
    x = tensor("x", shape=(2, 12, 5))
    out = join_dims(split_dims(x, shape=(3, 4), axis=1), start_axis=0, n_axes=2)
    rewritten = rewrite_graph(out, include=("canonicalize",))
    assert _count(rewritten, SplitDims) == 1
    assert _count(rewritten, JoinDims) == 1


def test_local_join_of_split_superset_merges():
    """A join covering the split span plus neighbours becomes one join of ``x``."""
    x = tensor("x", shape=(2, 6, 5))
    out = join_dims(split_dims(x, shape=(2, 3), axis=1), start_axis=0, n_axes=3)
    rewritten = rewrite_graph(out, include=("canonicalize",))
    assert _count(rewritten, SplitDims) == 0
    assert _count(rewritten, JoinDims) == 1
    fn = pytensor.function([x], rewritten)
    xv = np.arange(60.0).reshape(2, 6, 5)
    np.testing.assert_allclose(fn(xv), xv.reshape(12, 5))


def test_local_join_of_split_subset_splits():
    """A join inside the split span becomes one split with sizes multiplied."""
    x = tensor("x", shape=(12, 5))
    out = join_dims(split_dims(x, shape=(2, 2, 3), axis=0), start_axis=1, n_axes=2)
    rewritten = rewrite_graph(out, include=("canonicalize",))
    assert _count(rewritten, JoinDims) == 0
    assert _count(rewritten, SplitDims) == 1
    fn = pytensor.function([x], rewritten)
    xv = np.arange(60.0).reshape(12, 5)
    np.testing.assert_allclose(fn(xv), xv.reshape(2, 6, 5))


def test_local_join_of_join_merges():
    """Contiguous nested joins collapse into a single join."""
    x = tensor("x", shape=(2, 3, 4, 5))
    out = join_dims(join_dims(x, start_axis=1, n_axes=2), start_axis=0, n_axes=2)
    rewritten = rewrite_graph(out, include=("canonicalize",))
    assert _count(rewritten, JoinDims) == 1
    [join] = [n.op for n in apply_ancestors([rewritten]) if isinstance(n.op, JoinDims)]
    assert (join.start_axis, join.n_axes) == (0, 3)
    fn = pytensor.function([x], rewritten)
    xv = np.random.default_rng(0).normal(size=(2, 3, 4, 5))
    np.testing.assert_allclose(fn(xv), xv.reshape(24, 5))


def test_local_join_of_join_disjoint_stays():
    """Disjoint nested joins are left as two joins."""
    x = tensor("x", shape=(2, 3, 4, 5))
    out = join_dims(join_dims(x, start_axis=0, n_axes=2), start_axis=1, n_axes=2)
    rewritten = rewrite_graph(out, include=("canonicalize",))
    assert _count(rewritten, JoinDims) == 2


def test_local_split_of_join_cancels():
    """``SplitDims(JoinDims(x), s)`` cancels to ``x`` when ``s`` restores dims."""
    x = tensor("x", shape=(2, 3, 4))
    joined = join_dims(x, start_axis=0, n_axes=2)
    out = split_dims(joined, shape=[x.shape[0], x.shape[1]], axis=0)
    rewritten = rewrite_graph(out, include=("canonicalize",))
    assert rewritten is x


def test_local_split_of_join_unprovable_stays():
    """A split of a join with sizes not provably the pre-join dims is left."""
    x = tensor("x", shape=(2, 3, 4))
    joined = join_dims(x, start_axis=0, n_axes=2)  # (6, 4)
    out = split_dims(joined, shape=constant(np.array([3, 2], "int64")), axis=0)
    rewritten = rewrite_graph(out, include=("canonicalize",))
    assert _count(rewritten, JoinDims) == 1
    assert _count(rewritten, SplitDims) == 1


def test_local_split_of_split_merges():
    """An outer split refining an inner-produced axis collapses to one split."""
    x = tensor("x", shape=(6, 4))
    inner = split_dims(x, shape=(2, 3), axis=0)  # (2, 3, 4)
    out = split_dims(inner, shape=(1, 3), axis=1)  # split the '3' -> (2, 1, 3, 4)
    rewritten = rewrite_graph(out, include=("canonicalize",))
    assert _count(rewritten, SplitDims) == 1
    fn = pytensor.function([x], rewritten)
    xv = np.arange(24.0).reshape(6, 4)
    np.testing.assert_allclose(fn(xv), xv.reshape(2, 1, 3, 4))


def test_local_split_of_split_disjoint_stays():
    """Splits of different original axes stay as two independent splits."""
    x = tensor("x", shape=(6, 20))
    inner = split_dims(x, shape=(2, 3), axis=0)  # (2, 3, 20)
    out = split_dims(inner, shape=(4, 5), axis=2)  # split the untouched '20'
    rewritten = rewrite_graph(out, include=("canonicalize",))
    assert _count(rewritten, SplitDims) == 2


def test_local_join_split_dims_lift():
    """JoinDims/SplitDims lift through a unary Elemwise to reach the elemwise."""
    x = tensor("x", shape=(2, 3, 4))
    rewritten = rewrite_graph(join_dims(exp(x)), include=("canonicalize",))
    assert isinstance(rewritten.owner.op, Elemwise)
    assert isinstance(rewritten.owner.inputs[0].owner.op, JoinDims)

    xs = tensor("xs", shape=(6, 4))
    rs = rewrite_graph(
        split_dims(exp(xs), shape=(2, 3), axis=0), include=("canonicalize",)
    )
    assert isinstance(rs.owner.op, Elemwise)
    assert isinstance(rs.owner.inputs[0].owner.op, SplitDims)


def test_local_join_dims_transpose_lift():
    """A transpose over a JoinDims lifts to a transpose of the input (Join stays)."""
    x = tensor("x", shape=(2, 3, 4, 5))
    out = join_dims(x, start_axis=1, n_axes=2).dimshuffle(2, 0, 1)  # (5, 2, 12)
    rewritten = rewrite_graph(out, include=("canonicalize",))
    assert _count(rewritten, JoinDims) == 1
    # the JoinDims now sits above the (lifted) DimShuffle, not below it
    assert isinstance(rewritten.owner.op, JoinDims)
    fn = pytensor.function([x], rewritten)
    xv = np.arange(120.0).reshape(2, 3, 4, 5)
    np.testing.assert_allclose(fn(xv), xv.reshape(2, 12, 5).transpose(2, 0, 1))


def test_local_split_dims_transpose_lift_enables_cancel():
    """A whole-group transpose between a split and join lets the pair cancel."""
    x = tensor("x", shape=(6, 4))
    split = split_dims(x, shape=(2, 3), axis=0)  # (2, 3, 4)
    out = join_dims(split.dimshuffle(2, 0, 1), start_axis=1, n_axes=2)  # (4, 6)
    rewritten = rewrite_graph(out, include=("canonicalize",))
    assert _count(rewritten, JoinDims) == 0
    assert _count(rewritten, SplitDims) == 0
    fn = pytensor.function([x], rewritten)
    xv = np.arange(24.0).reshape(6, 4)
    np.testing.assert_allclose(
        fn(xv), xv.reshape(2, 3, 4).transpose(2, 0, 1).reshape(4, 6)
    )


def test_local_split_dims_transpose_within_group_irreducible():
    """A transpose *within* the split group is a genuine reshuffle and is kept."""
    x = tensor("x", shape=(6, 4))
    split = split_dims(x, shape=(2, 3), axis=0)  # (2, 3, 4)
    out = join_dims(split.dimshuffle(1, 0, 2), start_axis=0, n_axes=2)  # (6, 4)
    rewritten = rewrite_graph(out, include=("canonicalize",))
    assert _count(rewritten, JoinDims) == 1
    assert _count(rewritten, SplitDims) == 1
    fn = pytensor.function([x], rewritten)
    xv = np.arange(24.0).reshape(6, 4)
    np.testing.assert_allclose(
        fn(xv), xv.reshape(2, 3, 4).transpose(1, 0, 2).reshape(6, 4)
    )


def _reshape_canon(x, new_shape):
    out = x.reshape(new_shape)
    return rewrite_graph(out, include=("canonicalize", "specialize"))


def test_local_reshape_to_join_split_join():
    """A reshape that only merges adjacent dims becomes JoinDims (no Reshape)."""
    x = tensor("x", shape=(2, 3, 4))
    r = _reshape_canon(x, (6, 4))
    assert _count(r, JoinDims) == 1
    assert _count(r, SplitDims) == 0
    assert _count(r, Reshape) == 0
    fn = pytensor.function([x], r)
    xv = np.arange(24.0).reshape(2, 3, 4)
    np.testing.assert_allclose(fn(xv), xv.reshape(6, 4))


def test_local_reshape_to_join_split_split():
    """A reshape that only splits a dim becomes SplitDims (no Reshape)."""
    x = tensor("x", shape=(6, 4))
    r = _reshape_canon(x, (2, 3, 4))
    assert _count(r, JoinDims) == 0
    assert _count(r, SplitDims) == 1
    assert _count(r, Reshape) == 0
    fn = pytensor.function([x], r)
    xv = np.arange(24.0).reshape(6, 4)
    np.testing.assert_allclose(fn(xv), xv.reshape(2, 3, 4))


def test_local_reshape_to_join_split_straddle():
    """A regroup whose boundaries don't align becomes a JoinDims + SplitDims pair."""
    x = tensor("x", shape=(3, 6))
    r = _reshape_canon(x, (6, 3))
    assert _count(r, JoinDims) == 1
    assert _count(r, SplitDims) == 1
    assert _count(r, Reshape) == 0
    fn = pytensor.function([x], r)
    xv = np.arange(18.0).reshape(3, 6)
    np.testing.assert_allclose(fn(xv), xv.reshape(6, 3))


def test_local_reshape_to_join_split_passthrough_anchor():
    """An unknown dim passed through as x.shape[k] anchors the decomposition."""
    x = tensor("x", shape=(None, 3, 4))
    r = _reshape_canon(x, (x.shape[0], 12))
    assert _count(r, JoinDims) == 1
    assert _count(r, SplitDims) == 0
    assert _count(r, Reshape) == 0
    fn = pytensor.function([x], r)
    xv = np.arange(24.0).reshape(2, 3, 4)
    np.testing.assert_allclose(fn(xv), xv.reshape(2, 12))


def test_local_reshape_to_join_split_opaque_stays():
    """An opaque runtime shape (unknown leading dims + -1) stays a Reshape."""
    x = tensor("x", shape=(None, None, 4))
    r = _reshape_canon(x, (-1, 4))
    assert _count(r, Reshape) == 1
    assert _count(r, JoinDims) == 0
    assert _count(r, SplitDims) == 0
