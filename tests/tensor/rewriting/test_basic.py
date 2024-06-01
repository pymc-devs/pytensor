import copy

import numpy as np
import pytest

import pytensor
import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor import graph_replace, shared
from pytensor.compile import optdb
from pytensor.compile.function import function
from pytensor.compile.mode import get_default_mode, get_mode
from pytensor.compile.ops import DeepCopyOp, deep_copy_op
from pytensor.configdefaults import config
from pytensor.graph.basic import equal_computations
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import check_stack_trace, out2in
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.printing import debugprint, pprint
from pytensor.raise_op import Assert, CheckAndRaise
from pytensor.scalar import Composite, float64
from pytensor.tensor.basic import (
    Alloc,
    Join,
    MakeVector,
    ScalarFromTensor,
    Split,
    TensorFromScalar,
    as_tensor,
    cast,
    join,
    tile,
)
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import (
    add,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    dot,
    eq,
    exp,
    floor_div,
    ge,
    gt,
    int_div,
    le,
    log,
    lt,
    maximum,
    minimum,
    mul,
    neq,
    softplus,
    sqrt,
    sub,
    true_div,
)
from pytensor.tensor.math import pow as pt_pow
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.rewriting.basic import (
    assert_op,
    local_alloc_sink_dimshuffle,
    local_merge_alloc,
    local_useless_alloc,
    local_useless_elemwise,
    topological_fill_sink,
)
from pytensor.tensor.rewriting.math import local_lift_transpose_through_dot
from pytensor.tensor.rewriting.shape import ShapeFeature
from pytensor.tensor.shape import (
    Reshape,
    Shape_i,
    SpecifyShape,
    Unbroadcast,
    specify_shape,
    unbroadcast,
)
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor1,
    Subtensor,
    advanced_inc_subtensor,
    advanced_inc_subtensor1,
    inc_subtensor,
)
from pytensor.tensor.type import (
    TensorType,
    dmatrix,
    dscalar,
    dvector,
    fmatrix,
    fscalar,
    imatrices,
    iscalar,
    iscalars,
    ivector,
    lscalar,
    lvector,
    matrices,
    matrix,
    row,
    scalar,
    scalars,
    tensor,
    tensor3,
    tensor4,
    values_eq_approx_remove_nan,
    vector,
)
from tests import unittest_tools as utt


rewrite_mode = config.mode
if rewrite_mode == "FAST_COMPILE":
    rewrite_mode = "FAST_RUN"
rewrite_mode = get_mode(rewrite_mode)

_stabilize_rewrites = RewriteDatabaseQuery(include=["fast_run"])
_stabilize_rewrites.position_cutoff = 1.51
_stabilize_rewrites = optdb.query(_stabilize_rewrites)

_specialize_rewrites = RewriteDatabaseQuery(include=["fast_run"])
_specialize_rewrites.position_cutoff = 2.01
_specialize_rewrites = optdb.query(_specialize_rewrites)

_fast_run_rewrites = RewriteDatabaseQuery(include=["fast_run"])
_fast_run_rewrites = optdb.query(_fast_run_rewrites)


def rewrite(g, level="fast_run"):
    if level == "fast_run":
        _fast_run_rewrites.rewrite(g)
    elif level == "specialize":
        _specialize_rewrites.rewrite(g)
    elif level == "stabilize":
        _stabilize_rewrites.rewrite(g)
    else:
        raise ValueError(level)
    return g


def test_local_useless_slice():
    # test a simple matrix
    x = matrix("x")
    mode_excluding = get_default_mode().excluding(
        "local_useless_slice", "local_mul_canonizer"
    )
    mode_including = (
        get_default_mode()
        .including("local_useless_slice")
        .excluding("local_mul_canonizer")
    )

    # test with and without the useless slice
    o = 2 * x[0, :]
    f_excluding = function([x], o, mode=mode_excluding)
    f_including = function([x], o, mode=mode_including)
    rng = np.random.default_rng(utt.fetch_seed())
    test_inp = rng.integers(-10, 10, (4, 4)).astype("float32")
    assert all(f_including(test_inp) == f_excluding(test_inp))
    # test to see if the slice is truly gone
    apply_node = f_including.maker.fgraph.toposort()[0]
    subtens = apply_node.op
    assert not any(isinstance(idx, slice) for idx in subtens.idx_list)

    # Now test that the stack trace is copied over properly,
    # before before and after rewriting.
    assert check_stack_trace(f_excluding, ops_to_check="all")
    assert check_stack_trace(f_including, ops_to_check="all")

    # test a 4d tensor
    z = tensor4("z")
    o2 = z[1, :, :, 1]
    o3 = z[0, :, :, :]
    f_including_check = function([z], o2, mode=mode_including)
    f_including_check_apply = function([z], o3, mode=mode_including)

    # The rewrite shouldn't apply here
    apply_node = f_including_check.maker.fgraph.toposort()[0]
    subtens = apply_node.op
    assert [isinstance(idx, slice) for idx in subtens.idx_list].count(True) == 2
    # But it should here
    apply_node = f_including_check_apply.maker.fgraph.toposort()[0]
    subtens = apply_node.op
    assert not any(isinstance(idx, slice) for idx in subtens.idx_list)

    # Finally, test that the stack trace is copied over properly,
    # before before and after rewriting.
    assert check_stack_trace(f_including_check, ops_to_check=Subtensor)
    assert check_stack_trace(f_including_check_apply, ops_to_check=Subtensor)


def test_local_useless_fill():
    x = dvector()
    y = dvector()
    z = lvector()

    x_ = np.random.random((5,))
    y_ = np.random.random((5,))
    z_ = (np.random.random((5,)) * 5).astype("int64")

    # basic case
    f = function([x], pt.fill(x, x) * 2, mode=rewrite_mode)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    res = f(x_)
    exp_res = np.broadcast_to(x_, x_.shape) * 2
    assert np.array_equal(res, exp_res)

    # basic case
    f = function([x, y], pt.second(y, x) * 2, mode=rewrite_mode)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    res = f(x_, y_)
    exp_res = np.broadcast_to(x_, y_.shape) * 2
    assert np.array_equal(res, exp_res)

    # basic case
    f = function([x, y], pt.fill(x, y) * 2, mode=rewrite_mode)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    res = f(x_, y_)
    exp_res = np.broadcast_to(y_, x_.shape) * 2
    assert np.array_equal(res, exp_res)

    # now with different type(cast)
    f = function([x, z], pt.fill(z, x) * 2, mode=rewrite_mode)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    res = f(x_, z_)
    exp_res = np.broadcast_to(x_, z_.shape) * 2
    assert np.array_equal(res, exp_res)

    # now with different type(cast)
    f = function([x, z], pt.fill(x, z) * 2, mode=rewrite_mode)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    res = f(x_, z_)
    exp_res = np.broadcast_to(z_, x_.shape) * 2
    assert np.array_equal(res, exp_res)

    # now cutting out the input ??
    f = function([x, y], pt.fill(x, y) * 2, mode=rewrite_mode)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    res = f(x_, y_)
    exp_res = np.broadcast_to(y_, x_.shape) * 2
    assert np.array_equal(res, exp_res)


def test_local_fill_to_alloc():
    x = dvector()
    m = dmatrix()

    x_ = np.random.random((5,))
    m_ = np.random.random((5, 5))

    y = pt.fill(m, x)

    mode = rewrite_mode.including("stabilize", "local_fill_to_alloc").excluding(
        "useless", "local_useless_fill"
    )

    f = function([m, x], y, mode=mode)
    assert Alloc in [node.op.__class__ for node in f.maker.fgraph.toposort()]

    res = f(m_, x_)
    exp_res = np.broadcast_to(x_, m_.shape)
    assert np.array_equal(res, exp_res)

    y = pt.fill(x, m)

    f = function([m, x], y, mode=mode)

    assert Alloc not in [node.op.__class__ for node in f.maker.fgraph.toposort()]

    res = f(m_, x_)
    assert np.array_equal(res, m_)


class TestLocalCanonicalizeAlloc:
    def setup_method(self):
        self.rng = np.random.default_rng(utt.fetch_seed())

    @pytest.mark.parametrize("shape_unsafe", (True, False))
    def test_inconsistent_shared(self, shape_unsafe):
        # These shapes don't match!
        x = shared(self.rng.standard_normal((3, 7)))
        a = pt.alloc(x, 6, 7)

        assert a.owner and isinstance(a.owner.op, Alloc)

        mode = rewrite_mode if shape_unsafe else rewrite_mode.excluding("shape_unsafe")
        f = function([], a, mode=mode)

        has_alloc = any(
            isinstance(node.op, Alloc) for node in f.maker.fgraph.toposort()
        )
        if shape_unsafe:
            assert not has_alloc
            # Error raised by SpecifyShape that is introduced due to static shape inference
            with pytest.raises(
                AssertionError,
                match="SpecifyShape: dim 0 of input has shape 3, expected 6.",
            ):
                f()
        else:
            assert has_alloc
            # Error raised by Alloc Op
            with pytest.raises(
                ValueError,
                match=r"could not broadcast input array from shape \(3,7\) into shape \(6,7\)",
            ):
                f()

        good_x_val = self.rng.standard_normal((6, 7))
        x.set_value(good_x_val)

        assert np.array_equal(f(), good_x_val)

    def test_basic_fill(self):
        x = matrix("x")
        y = pt.fill(x, x)

        # The rewrite `locall_fill_to_alloc` should call `pt.alloc`,
        # which should return `x` and not `alloc(x, ...)`
        f = function([x], [y], mode=rewrite_mode.including("local_fill_to_alloc"))
        assert not any(isinstance(node.op, Alloc) for node in f.maker.fgraph.toposort())

    def test_basic_tile(self):
        x = matrix("x")
        y = pt.tile(x, (1,) * 2)

        mode = rewrite_mode.including(
            "local_dimshuffle_lift",
            "local_useless_dimshuffle_in_reshape",
            "local_alloc_sink_dimshuffle",
        )
        f = function([x], [y], mode=mode)

        assert not any(isinstance(node.op, Alloc) for node in f.maker.fgraph.toposort())

    @pytest.mark.parametrize(
        "x, has_alloc",
        [
            (pt.alloc(np.ones((2,)), 1, 3, 2), True),
            (pt.alloc(np.array(1.0), 1, 1), False),
            (pt.alloc(np.ones((1, 1)), 1, 1, 2), True),
            (pt.alloc(np.ones((1, 1)), 1, 2), True),
        ],
    )
    def test_useless_alloc_with_shape_one(self, x, has_alloc):
        g = FunctionGraph(outputs=[x])
        assert any(isinstance(node.op, Alloc) for node in g.toposort())

        alloc_lift = out2in(local_alloc_sink_dimshuffle)
        alloc_lift.rewrite(g)

        if has_alloc:
            assert any(isinstance(node.op, Alloc) for node in g.toposort())
        else:
            assert not any(isinstance(node.op, Alloc) for node in g.toposort())


class TestLocalUselessIncSubtensorAlloc:
    rewrite_name = "local_useless_inc_subtensor_alloc"

    def setup_method(self):
        # The rewrite requires the shape feature so we need to compile in
        # FAST_RUN mode.
        mode = config.mode
        if mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        self.mode = get_mode(mode)
        self.rng = np.random.default_rng(utt.fetch_seed())

    def test_advanced_inc_subtensor(self):
        x = vector("x")
        y = scalar("y")
        i = matrix("i", dtype="int64")
        z = advanced_inc_subtensor(x, pt.alloc(y, *i.shape), i)
        mode1 = self.mode.excluding(self.rewrite_name)
        mode2 = self.mode.including(self.rewrite_name)
        f1 = function([x, i, y], z, mode=mode1)
        f2 = function([x, i, y], z, mode=mode2)

        # the alloc op should still be there
        assert (
            len([n for n in f1.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 1
        )
        # the alloc op should have been removed
        assert (
            len([n for n in f2.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 0
        )

        x_value = np.random.standard_normal(5).astype(config.floatX)
        y_value = np.random.standard_normal()
        i_value = self.rng.integers(0, 3, size=(2, 3))

        r1 = f1(x_value, i_value, y_value)
        r2 = f2(x_value, i_value, y_value)

        utt.assert_allclose(r1, r2)

        # Check stacktrace was copied over correctly after rewrite was applied
        assert check_stack_trace(f1, ops_to_check=AdvancedIncSubtensor1)
        assert check_stack_trace(f2, ops_to_check=AdvancedIncSubtensor1)

    def test_advanced_inc_subtensor1(self):
        x = vector("x")
        y = scalar("y")
        i = vector("i", dtype="int64")
        z = advanced_inc_subtensor1(x, pt.alloc(y, *i.shape), i)
        mode1 = self.mode.excluding(self.rewrite_name)
        mode2 = self.mode.including(self.rewrite_name)
        f1 = function([x, i, y], z, mode=mode1)
        f2 = function([x, i, y], z, mode=mode2)

        # the alloc op should still be there
        assert (
            len([n for n in f1.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 1
        )
        # the alloc op should have been removed
        assert (
            len([n for n in f2.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 0
        )

        x_value = np.random.standard_normal(5).astype(config.floatX)
        y_value = np.random.standard_normal()
        i_value = self.rng.integers(0, 3, size=2)

        r1 = f1(x_value, i_value, y_value)
        r2 = f2(x_value, i_value, y_value)

        utt.assert_allclose(r1, r2)

        assert check_stack_trace(f1, ops_to_check=AdvancedIncSubtensor1)
        assert check_stack_trace(f2, ops_to_check="all")

    def test_incsubtensor(self):
        x = vector("x")
        y = scalar("y")
        i = scalar("i", dtype="int64")
        z = inc_subtensor(x[:i], pt.alloc(y, i))
        mode1 = self.mode.excluding(self.rewrite_name)
        mode2 = self.mode.including(self.rewrite_name)
        f1 = function([x, i, y], z, mode=mode1)
        f2 = function([x, i, y], z, mode=mode2)

        # the alloc op should still be there
        assert (
            len([n for n in f1.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 1
        )
        # the alloc op should have been removed
        assert (
            len([n for n in f2.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 0
        )

        x_value = np.random.standard_normal(5).astype(config.floatX)
        y_value = np.random.standard_normal()
        i_value = 3

        r1 = f1(x_value, i_value, y_value)
        r2 = f2(x_value, i_value, y_value)

        utt.assert_allclose(r1, r2)

        assert check_stack_trace(f1, ops_to_check="last")
        assert check_stack_trace(f2, ops_to_check="last")


class TestUselessCheckAndRaise:
    def test_basic(self):
        mode = get_default_mode().including(
            "canonicalize", "local_remove_useless_assert"
        )
        x = scalar()
        y = scalar()
        f = function([x, y], assert_op(x, eq(x, y)), mode=mode)
        assert f(1, 1) == 1
        with pytest.raises(AssertionError):
            f(1, 0)

    def test_local_remove_useless_1(self):
        """Remove `CheckAndRaise`s when all the conditions are always true."""
        x = scalar()
        fg = FunctionGraph(outputs=[assert_op(x, 1)], clone=False)
        fg_res = rewrite_graph(fg, include=["canonicalize", "specialize"])
        topo = fg_res.toposort()
        assert not any(isinstance(node.op, CheckAndRaise) for node in topo)

    def test_local_remove_useless_2(self):
        """Remove `CheckAndRaise` conditions that are always true."""
        x = scalar()
        y = scalar()
        fg = FunctionGraph(outputs=[assert_op(x, y, 1)], clone=False)
        fg_res = rewrite_graph(fg, include=["canonicalize", "specialize"])
        topo = fg_res.toposort()
        (assert_node,) = (node for node in topo if isinstance(node.op, CheckAndRaise))
        assert assert_node.inputs == [x, y]

    def test_local_remove_useless_3(self):
        """Don't remove `CheckAndRaise` conditions that are always false."""
        x = scalar()
        y = scalar()
        fg = FunctionGraph(outputs=[assert_op(x, y, 0)], clone=False)
        fg_res = rewrite_graph(fg, include=["canonicalize", "specialize"])
        topo = fg_res.toposort()
        (assert_node,) = (node for node in topo if isinstance(node.op, CheckAndRaise))
        assert assert_node.inputs[:2] == [x, y]
        assert assert_node.inputs[-1].data == 0


def test_local_remove_all_assert():
    r"""Remove all `Assert`\s."""
    mode = get_default_mode().including("canonicalize", "local_remove_all_assert")

    x = scalar()
    y = scalar()
    f = function([x, y], assert_op(x, y), mode=mode)
    # Without the rewrite, this would fail
    assert f(1, 0) == 1
    topo = f.maker.fgraph.toposort()
    assert not any(isinstance(node.op, CheckAndRaise) for node in topo)

    mode = get_default_mode()
    a = assert_op(x, eq(x, 0).any())
    f = function([x], a, mode=mode.excluding("unsafe"))
    topo = f.maker.fgraph.toposort()
    a_op = [n for n in topo if isinstance(n.op, Assert)]
    assert len(a_op) == 1


class TestTile:
    def test_local_useless_tile(self):
        v = vector()
        m = matrix()
        mode = None
        if config.mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        for var, data in [(v, [1, 2, 3]), (m, [[1, 2], [3, 4]])]:
            # When len(repeat pattern) <= var.ndim, everything is removed
            # for ndim in range(1, var.ndim):
            for ndim in range(var.ndim + 1):
                f = function([var], tile(var, (1,) * ndim), mode=mode)
                topo = f.maker.fgraph.toposort()
                assert len(topo) == 1
                assert isinstance(topo[0].op, DeepCopyOp)
                f(data)
                # In this case, the rewrite only removes nodes;
                # no need to `check_stack_trace`
            # When len(repeat pattern) > var.ndim, only a dimshuffle should be
            # left, but there can be a DeepCopy as well
            for ndim in range(var.ndim + 1, var.ndim + 3):
                f = function([var], tile(var, (1,) * ndim), mode=mode)
                topo = f.maker.fgraph.toposort()
                assert len(topo) <= 2
                assert isinstance(topo[0].op, DimShuffle)
                assert check_stack_trace(f, ops_to_check=[DimShuffle])
                f(data)


class TestUnbroadcast:
    def setup_method(self):
        self.mode = get_default_mode().including("canonicalize")

    def test_local_useless_unbroadcast(self):
        x1 = tensor(dtype="float64", shape=(1, 2))
        x2 = tensor(dtype="float64", shape=(2, 1))
        unbroadcast_op = Unbroadcast(0)

        f = function([x1], unbroadcast_op(x1), mode=self.mode)
        assert (
            sum(isinstance(node.op, Unbroadcast) for node in f.maker.fgraph.toposort())
            == 1
        )

        f = function([x2], unbroadcast_op(x2), mode=self.mode)
        assert (
            sum(isinstance(node.op, Unbroadcast) for node in f.maker.fgraph.toposort())
            == 0
        )

    def test_local_unbroadcast_lift(self):
        x = tensor(dtype="float64", shape=(1, 1))
        y = unbroadcast(pt.exp(unbroadcast(x, 0)), 1)

        assert (
            sum(
                isinstance(node.op, Unbroadcast)
                for node in FunctionGraph([x], [y], copy_inputs=False).toposort()
            )
            == 2
        )

        f = function([x], y, mode=self.mode)
        assert (
            sum(isinstance(node.op, Unbroadcast) for node in f.maker.fgraph.toposort())
            == 1
        )

        np.testing.assert_almost_equal(f([[1]]), np.exp([[1]]))


class TestUselessElemwise:
    def setup_method(self):
        self.mode = get_default_mode().including("canonicalize", "local_fill_to_alloc")

    def test_eq(self):
        x = dmatrix()
        y = dmatrix()
        f = function([x, y], eq(x, y), mode=self.mode)
        vx = np.random.random((5, 4))
        vy = np.random.random((5, 4))
        f(vx, vy)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, Elemwise)
        assert isinstance(topo[0].op.scalar_op, ps.EQ)
        f2 = function([x], eq(x, x), mode=self.mode)
        assert np.all(f2(vx) == np.ones((5, 4)))
        topo2 = f2.maker.fgraph.toposort()
        # Shape_i{1}(<TensorType(float64, (?, ?))>),
        # Shape_i{0}(<TensorType(float64, (?, ?))>), Alloc([[1]], Shape_i{0}.0,
        # Shape_i{1}.0
        assert len(topo2) == 3
        assert isinstance(topo2[-1].op, Alloc)

    def test_neq(self):
        x = dmatrix()
        y = dmatrix()
        f = function([x, y], neq(x, y), mode=self.mode)
        vx = np.random.random((5, 4))
        vy = np.random.random((5, 4))
        f(vx, vy)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, Elemwise)
        assert isinstance(topo[0].op.scalar_op, ps.NEQ)
        f2 = function([x], neq(x, x), mode=self.mode)
        assert np.all(f2(vx) == np.zeros((5, 4)))
        topo2 = f2.maker.fgraph.toposort()
        assert len(topo2) == 3
        assert isinstance(topo2[-1].op, Alloc)

    def test_mul(self):
        x = dmatrix()
        y = dmatrix()
        f = function([x], mul(x), mode=self.mode)
        vx = np.random.random((5, 4))
        vy = np.random.random((5, 4))
        f(vx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op
        f2 = function([x, y], mul(x, y), mode=self.mode)
        assert np.all(f2(vx, vy) == vx * vy)
        topo2 = f2.maker.fgraph.toposort()
        assert len(topo2) == 1
        assert isinstance(topo2[0].op, Elemwise)
        assert isinstance(topo2[0].op.scalar_op, ps.Mul)

    def test_add(self):
        x = dmatrix()
        y = dmatrix()
        f = function([x], add(x), mode=self.mode)
        vx = np.random.random((5, 4))
        vy = np.random.random((5, 4))
        f(vx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op
        f2 = function([x, y], add(x, y), mode=self.mode)
        assert np.all(f2(vx, vy) == vx + vy)
        topo2 = f2.maker.fgraph.toposort()
        assert len(topo2) == 1
        assert isinstance(topo2[0].op, Elemwise)
        assert isinstance(topo2[0].op.scalar_op, ps.Add)

    def test_identity(self):
        # ps.identity is used in 2 Elemwise functions:
        # tensor_copy, and view
        x = matrix()
        f = function([x], pt.tensor_copy(x), mode=self.mode)
        vx = np.random.random((5, 4)).astype(config.floatX)
        f(vx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op


class TestCastCast:
    def setup_method(self):
        mode = get_default_mode()
        self.mode = mode.including("local_cast_cast")

    def test_consecutive(self):
        x = fmatrix()
        o = Elemwise(ps.Cast(ps.ScalarType("float64")))(x.astype("float64"))
        f = function([x], o, mode=self.mode)
        dx = np.random.random((5, 4)).astype("float32")
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op.scalar_op, ps.basic.Cast)

        x = dmatrix()
        o = Elemwise(ps.Cast(ps.ScalarType("float32")))(x.astype("float32"))
        f = function([x], o, mode=self.mode)
        dx = np.random.random((5, 4))
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op.scalar_op, ps.basic.Cast)

    def test_upcast(self):
        # Upcast followed by any other cast
        x = fmatrix()
        o = Elemwise(ps.Cast(ps.ScalarType("complex128")))(x.astype("complex64"))
        f = function([x], o, mode=self.mode)
        dx = np.random.random((5, 4)).astype("float32")
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op.scalar_op, ps.basic.Cast)

        # Upcast followed by a downcast back to the base type
        x = fmatrix()
        o = Elemwise(ps.Cast(ps.ScalarType("float32")))(x.astype("float64"))
        f = function([x], o, mode=self.mode)
        dx = np.random.random((5, 4)).astype("float32")
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, DeepCopyOp)

        # Downcast followed by an upcast back to the base type
        # The rewrite shouldn't be applied
        x = dmatrix()
        o = Elemwise(ps.Cast(ps.ScalarType("float64")))(x.astype("float32"))
        f = function([x], o, mode=self.mode)
        dx = np.random.random((5, 4))
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert (
            len(topo) == 1 and isinstance(topo[0].op.scalar_op, ps.basic.Composite)
        ) or (len(topo) > 1)


def test_constant_folding():
    # Test that constant folding get registered at fast_compile
    # An error removed that registration during the registration.
    x = dvector()
    mode = get_mode("FAST_COMPILE").excluding("fusion")
    f = function([x], [x * 2, x + x], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 2

    # Test that we do not crash when constant folding elemwise scalar
    # as they should not generate c code.

    x = pt.constant(3)
    assert x.ndim == 0
    mode = get_mode("FAST_COMPILE").excluding("fusion")
    f = function([], [x * 2, x + x], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 2
    assert all(isinstance(n.op, DeepCopyOp) for n in topo)


@pytest.mark.xfail(
    reason="PyTensor rewrites constants before stabilization. "
    "This breaks stabilization rewrites in some cases. See #504.",
    raises=AssertionError,
)
def test_constant_get_stabilized():
    # Currently PyTensor enables the `constant_folding` rewrite before stabilization rewrites.
    # This caused some stabilization rewrites to not be activated and that
    # caused inf values to appear when they should not.

    # We can't simply move the `constant_folding` rewrite to
    # specialize since this will break other rewrites.  We will need to
    # partially duplicate some canonicalize rewrites to fix this issue.

    x2 = scalar()
    y2 = log(1 + exp(x2))
    mode = get_default_mode()
    mode.check_isfinite = False
    f2 = function([x2], y2, mode=mode)

    assert len(f2.maker.fgraph.toposort()) == 1
    assert f2.maker.fgraph.toposort()[0].op == softplus
    assert f2(800) == 800

    x = pt.as_tensor_variable(800)
    y = log(1 + exp(x))
    f = function([], y, mode=mode)
    # When this error is fixed, the following line should be ok.
    assert f() == 800, f()


class TestLocalSwitchSink:
    def setup_method(self):
        # condition values
        self.condm = np.asarray([[0.1, 0, 1, -1], [0.0, 0.0, 0.0, 0.0], [1, 1, 1, 1]])
        self.condv = np.asarray([0.1, 0, 1, -1])
        self.conds = [0.1, 0, 1, -1]

        # x values
        self.xm = np.ones((3, 4))
        self.xv = np.ones((4,))
        self.xs = 1.0

        # expected results
        self.resm = (
            [np.asarray([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]])] * 3
            + [np.asarray([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]])]
            + 2 * [np.asarray([[1, 0, 1, 0]])]
            + [[np.ones((3, 4)), np.zeros((3, 4)), np.ones((3, 4)), np.zeros((3, 4))]]
            + [[np.ones((4,)), np.zeros((4,)), np.ones((4,)), np.zeros((4,))]]
            + [[np.asarray(1.0), np.asarray(0.0), np.asarray(1.0), np.asarray(0.0)]]
        )

        self.mode = (
            get_default_mode()
            .including("canonicalize", "fast_run")
            .excluding("gpu", "fusion")
        )
        self.mode = copy.copy(self.mode)
        self.mode.check_isfinite = False

    def function_remove_nan(self, *args, **kwargs):
        """
        Wrapper around function for this test.

        It disables checking for NaN removed by rewrites in `DebugMode`
        (it has false positives in that case).
        """
        f = function(*args, **kwargs)

        def wrapped_f(*args, **kwargs):
            # This is a bit ugly since it changes the global value of
            # TensorType.values_eq_approx.
            old_values_eq_approx = staticmethod(TensorType.values_eq_approx)
            TensorType.values_eq_approx = staticmethod(values_eq_approx_remove_nan)
            try:
                out = f(*args, **kwargs)
            finally:
                TensorType.values_eq_approx = old_values_eq_approx
            return out

        return wrapped_f

    def test_local_mul_switch_sink(self):
        c = dscalar()
        idx = 0
        for condition in [
            (dmatrix("cond"), self.condm),
            (dvector("cond"), self.condv),
            (dscalar("cond"), self.conds),
        ]:
            for x in [
                (dmatrix("x"), self.xm),
                (dvector("x"), self.xv),
                (dscalar("x"), self.xs),
            ]:
                y = mul(
                    pt.switch(condition[0] > 0, 1.0 * x[0], 0.0 * x[0]),
                    pt.switch(condition[0] > 0, 1.0 * x[0], log(c) * x[0]),
                )
                f = self.function_remove_nan(
                    [condition[0], x[0], c], [y], mode=self.mode
                )
                if isinstance(condition[1], list):
                    for i in range(len(condition[1])):
                        res = f(condition[1][i], x[1], -1)
                        assert (
                            res == np.asarray(self.resm[idx][i])
                        ).sum() == self.resm[idx][i].size
                else:
                    res = f(condition[1], x[1], -1)
                    assert (res == np.asarray(self.resm[idx])).sum() == self.resm[
                        idx
                    ].size
                idx += 1

        # This case caused a missed rewrite in the past.
        x = dscalar("x")
        y = pt.switch(x < 7, x, sqrt(x - 7))
        f = self.function_remove_nan([x], pytensor.gradient.grad(y, x), self.mode)
        assert f(5) == 1, f(5)

    @pytest.mark.slow
    def test_local_div_switch_sink(self):
        c = dscalar()
        idx = 0
        for condition in [
            (dmatrix("cond"), self.condm),
            (dvector("cond"), self.condv),
            (dscalar("cond"), self.conds),
        ]:
            for x in [
                (dmatrix("x"), self.xm),
                (dvector("x"), self.xv),
                (dscalar("x"), self.xs),
            ]:
                y = true_div(
                    pt.switch(condition[0] > 0, 1.0 * x[0], 0.0 * x[0]),
                    pt.switch(condition[0] > 0, 1.0 * x[0], log(c) * x[0]),
                )
                f = self.function_remove_nan(
                    [condition[0], x[0], c], [y], mode=self.mode
                )
                if isinstance(condition[1], list):
                    for i in range(len(condition[1])):
                        res = f(condition[1][i], x[1], -1)
                        assert (
                            res == np.asarray(self.resm[idx][i])
                        ).sum() == self.resm[idx][i].size
                else:
                    res = f(condition[1], x[1], -1)
                    assert (res == np.asarray(self.resm[idx])).sum() == self.resm[
                        idx
                    ].size
                idx += 1


class TestLocalUselessSwitch:
    def setup_method(self):
        self.mode = rewrite_mode.excluding("constant_folding")

    @pytest.mark.parametrize(
        "dtype1",
        ["int32", "int64"],
    )
    @pytest.mark.parametrize(
        "dtype2",
        ["int32", "int64"],
    )
    @pytest.mark.parametrize(
        "cond",
        [0, 1, np.array([True])],
    )
    def test_const(self, dtype1, dtype2, cond):
        x = matrix("x", dtype=dtype1)
        y = matrix("y", dtype=dtype2)
        z = pt.switch(cond, x, y)
        f = function([x, y], z, mode=self.mode)
        assert not any(
            node.op
            for node in f.maker.fgraph.toposort()
            if (
                isinstance(node.op, Elemwise)
                and isinstance(node.op.scalar_op, ps.basic.Switch)
            )
        )
        vx = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype1)
        vy = np.array([[7, 8, 9], [10, 11, 12]], dtype=dtype2)
        np_res = np.where(cond, vx, vy)
        assert np.array_equal(f(vx, vy), np_res)

    @pytest.mark.parametrize(
        "dtype1",
        ["int32", "int64"],
    )
    def test_left_is_right(self, dtype1):
        x = matrix("x", dtype=dtype1)
        varc = matrix("varc", dtype=dtype1)
        z1 = pt.switch(1, x, x)
        z0 = pt.switch(0, x, x)
        z2 = pt.switch(varc, x, x)
        f1 = function([x], z1, mode=self.mode)
        f0 = function([x], z0, mode=self.mode)
        f2 = function([x, varc], z2, mode=self.mode)

        topo = f1.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op

        topo = f0.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op

        topo = f2.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op

        vx = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype1)
        vc = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype1)
        assert np.array_equal(f1(vx), vx)
        assert np.array_equal(f0(vx), vx)
        assert np.array_equal(f2(vx, vc), vx)

    def test_left_is_right_constant(self):
        int8_one = as_tensor(np.int8(1))
        int8_zero = as_tensor(np.int8(0))
        int64_zero = as_tensor(np.int64(0))
        cond = scalar("cond", dtype=bool)

        out = pt.switch(cond, int8_zero, int64_zero)
        assert equal_computations([rewrite_graph(out)], [int64_zero])

        out = pt.switch(cond, int64_zero, int8_zero)
        assert equal_computations([rewrite_graph(out)], [int64_zero])

        out = pt.switch(cond, int8_one, int8_zero)
        assert equal_computations([rewrite_graph(out)], [out])

    @pytest.mark.parametrize(
        "dtype1",
        ["float32", "float64"],
    )
    def test_shape_le_0(self, dtype1):
        x = matrix("x", dtype=dtype1)
        z0 = pt.switch(le(x.shape[0], 0), 0, x.shape[0])
        f0 = function([x], z0, mode=self.mode)
        assert isinstance(f0.maker.fgraph.toposort()[0].op, Shape_i)

        z1 = pt.switch(le(x.shape[1], 0), 0, x.shape[1])
        f1 = function([x], z1, mode=self.mode)
        assert isinstance(f1.maker.fgraph.toposort()[0].op, Shape_i)

        vx = np.random.standard_normal((0, 5)).astype(dtype1)
        assert f0(vx) == 0
        assert f1(vx) == 5

    def test_broadcasting_1(self):
        # test switch(cst, matrix, row)
        x = matrix("x", dtype="int32")
        y = vector("y", dtype="int64")

        z = pt.switch(1, x, y)
        f = function([x, y], z, mode=self.mode)

        start_var = f.maker.fgraph.outputs[0]
        assert isinstance(start_var.owner.op, Elemwise)
        assert isinstance(start_var.owner.op.scalar_op, ps.basic.Cast)
        assert not any(node.op == pt.switch for node in f.maker.fgraph.toposort())

        vx = np.array([[1, 2, 3], [4, 5, 6]], dtype="int32")
        vy = np.array([10, 11, 12], dtype="int64")
        np_res = np.where(1, vx, vy)
        assert np.array_equal(f(vx, vy), np_res)

        z = pt.switch(0, x, y)
        f = function([x, y], z, mode=self.mode)

        assert isinstance(f.maker.fgraph.outputs[0].owner.op, Alloc)
        assert f.maker.fgraph.inputs[1] == f.maker.fgraph.outputs[0].owner.inputs[0]
        assert not any(node.op == pt.switch for node in f.maker.fgraph.toposort())

        vx = np.array([[1, 2, 3], [4, 5, 6]], dtype="int32")
        vy = np.array([10, 11, 12], dtype="int64")
        np_res = np.where(0, vx, vy)
        assert np.array_equal(f(vx, vy), np_res)

    def test_broadcasting_2(self):
        # test switch(cst, vector, matrix)

        x = vector("x", dtype="int32")
        y = matrix("y", dtype="int64")

        z = pt.switch(1, x, y)
        f = function([x, y], z, mode=self.mode)

        assert isinstance(f.maker.fgraph.outputs[0].owner.op, Alloc)
        assert not any(node.op == pt.switch for node in f.maker.fgraph.toposort())

        vx = np.array([4, 5, 6], dtype="int32")
        vy = np.array([[7, 8, 9], [10, 11, 12]], dtype="int64")
        np_res = np.where(1, vx, vy)
        assert np.array_equal(f(vx, vy), np_res)

        z = pt.switch(0, x, y)
        f = function([x, y], z, mode=self.mode)

        assert isinstance(f.maker.fgraph.outputs[0].owner.op, DeepCopyOp)
        assert not any(node.op == pt.switch for node in f.maker.fgraph.toposort())

        vx = np.array([4, 5, 6], dtype="int32")
        vy = np.array([[7, 8, 9], [10, 11, 12]], dtype="int64")
        np_res = np.where(0, vx, vy)
        assert np.array_equal(f(vx, vy), np_res)

    def test_broadcasting_3(self):
        # test switch(matrix, same_vector, same_vector)

        x = matrix("x", dtype="int32")
        y = vector("y", dtype="int64")
        z = pt.switch(x, y, y)
        f = function([x, y], z, mode=self.mode)
        vx = np.array([[0, 1], [1, 0]], dtype="int32")
        vy = np.array([7, 8], dtype="int64")
        utt.assert_allclose(f(vx, vy), np.where(vx, vy, vy))

        assert isinstance(f.maker.fgraph.outputs[0].owner.op, Alloc)
        assert not any(node.op == pt.switch for node in f.maker.fgraph.toposort())

    def test_broadcasting_different_dtype(self):
        cond = vector("x", dtype="bool")
        float32_branch = as_tensor(np.array([0], dtype="float32"))
        float64_branch = as_tensor(np.array([0], dtype="float64"))

        out = pt.switch(cond, float32_branch, float64_branch)
        expected_out = pt.alloc(float64_branch, cond.shape)

        rewritten_out = rewrite_graph(
            out, include=("canonicalize", "stabilize", "specialize")
        )
        assert equal_computations([rewritten_out], [expected_out])

        out = pt.switch(cond, float64_branch, float32_branch)
        rewritten_out = rewrite_graph(
            out, include=("canonicalize", "stabilize", "specialize")
        )
        assert equal_computations([rewritten_out], [expected_out])


class TestLocalMergeSwitchSameCond:
    @pytest.mark.parametrize(
        "op",
        [
            add,
            sub,
            mul,
            true_div,
            int_div,
            floor_div,
            minimum,
            maximum,
            gt,
            lt,
            ge,
            le,
            eq,
            neq,
            pt_pow,
        ],
    )
    def test_elemwise_float_ops(self, op):
        # float Ops
        mats = matrices("cabxy")
        c, a, b, x, y = mats
        s1 = pt.switch(c, a, b)
        s2 = pt.switch(c, x, y)

        g = rewrite(FunctionGraph(mats, [op(s1, s2)]))
        assert debugprint(g, file="str").count("Switch") == 1

    @pytest.mark.parametrize(
        "op",
        [
            bitwise_and,
            bitwise_or,
            bitwise_xor,
        ],
    )
    def test_elemwise_int_ops(self, op):
        # integer Ops
        mats = imatrices("cabxy")
        c, a, b, x, y = mats
        s1 = pt.switch(c, a, b)
        s2 = pt.switch(c, x, y)
        g = rewrite(FunctionGraph(mats, [op(s1, s2)]))
        assert debugprint(g, file="str").count("Switch") == 1

    @pytest.mark.parametrize("op", [add, mul])
    def test_elemwise_multi_inputs(self, op):
        # add/mul with more than two inputs
        mats = imatrices("cabxy")
        c, a, b, x, y = mats
        s1 = pt.switch(c, a, b)
        s2 = pt.switch(c, x, y)
        u, v = matrices("uv")
        s3 = pt.switch(c, u, v)
        g = rewrite(FunctionGraph([*mats, u, v], [op(s1, s2, s3)]))
        assert debugprint(g, file="str").count("Switch") == 1


class TestLocalOptAlloc:
    """
    TODO FIXME: These tests are incomplete; they need to `assert` something.
    """

    dtype = "float32"

    def test_sum_upcast(self):
        s = lscalar()
        a = pt.alloc(np.asarray(5, dtype=self.dtype), s, s)
        with config.change_flags(warn_float64="raise"):
            f = function([s], a.sum())
            f(5)

    def test_prod_upcast(self):
        s = lscalar()
        a = pt.alloc(np.asarray(5, dtype=self.dtype), s, s)

        with config.change_flags(warn_float64="raise"):
            f = function([s], a.prod())
            f(5)

    @config.change_flags(on_opt_error="raise")
    def test_sum_bool_upcast(self):
        s = lscalar()
        a = pt.alloc(np.asarray(True, dtype="bool"), s, s)
        f = function([s], a.sum())
        f(5)
        # test with user specified dtype
        f = function([s], a.sum(dtype=self.dtype))
        f(5)
        # test only 1 axis summed
        f = function([s], a.sum(axis=0, dtype=self.dtype))
        f(5)


class TestLocalOptAllocF16(TestLocalOptAlloc):
    dtype = "float16"


def test_local_join_1():
    # test for vector
    a = vector("a")
    s = pt.stack([a])
    f = function([a], s, mode=rewrite_mode)
    val = f([1])
    assert np.all(val == [1])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 0
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    # test for matrix join(0,a)
    a = matrix("a")
    s = join(0, a)
    f = function([a], s, mode=rewrite_mode)
    val = f([[1]])
    assert np.all(val == [[1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 0
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    # test for matrix join(1,a)
    s = join(1, a)
    f = function([a], s, mode=rewrite_mode)
    val = f([[1]])
    assert np.all(val == [[1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 0
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    # test we don't apply when their is 2 inputs
    s = join(1, a, a)
    f = function([a], s, mode=rewrite_mode)
    val = f([[1]])
    assert np.all(val == [[1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert f.maker.fgraph.outputs[0].dtype == config.floatX


def test_local_join_empty():
    # test for vector, vector, empty to vector
    empty_vec = np.asarray([], dtype=config.floatX)
    a = vector("a")
    s = pt.join(0, a, a, empty_vec)
    f = function([a], s, mode=rewrite_mode)
    val = f([1])
    assert np.all(val == [1])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all(
        not isinstance(n.op, Join) or len(n.inputs) == 3
        for n in e
        if isinstance(n.op, Join)
    )
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    # test for matrix join(1,a)
    empty_mat = np.asarray([[]], dtype=config.floatX)
    m = matrix("m")
    s = join(1, empty_mat, m, m, m)
    f = function([m], s, mode=rewrite_mode)
    val = f([[1]])
    assert np.all(val == [[1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all(
        not isinstance(n.op, Join) or len(n.inputs) == 4
        for n in e
        if isinstance(n.op, Join)
    )
    assert f.maker.fgraph.outputs[0].dtype == config.floatX
    # test for vector, vector, empty to matrix
    # We can't rewrite this case.
    s = pt.stack([a, a, empty_vec])
    f = function([a], s, mode=rewrite_mode)
    val = f([])
    assert np.all(val == [1])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all(
        not isinstance(n.op, Join) or len(n.inputs) == 4
        for n in e
        if isinstance(n.op, Join)
    )
    assert f.maker.fgraph.outputs[0].dtype == config.floatX
    # test for matrix join(0,a)
    # We can't rewrite this case.
    s = join(0, m, np.asarray([[2.0]], dtype=config.floatX), m)
    f = function([m], s, mode=rewrite_mode)
    val = f([[1]])
    assert np.all(val == [[1], [2], [1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all(
        not isinstance(n.op, Join) or len(n.inputs) == 4
        for n in e
        if isinstance(n.op, Join)
    )
    assert f.maker.fgraph.outputs[0].dtype == config.floatX


def test_local_join_make_vector():
    a, b, c, d, e = scalars("abcde")
    v = vector("v")
    mv = MakeVector(config.floatX)
    s = pt.join(0, mv(a), v, mv(b, c), mv(d, e))
    f = function([a, b, c, d, e, v], s, mode=rewrite_mode)
    val = f(1, 2, 3, 4, 6, [7, 8])
    assert np.all(val == [1, 7, 8, 2, 3, 4, 6])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all(
        not isinstance(n.op, Join) or len(n.inputs) == 4
        for n in e
        if isinstance(n.op, Join)
    )
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    assert check_stack_trace(f, ops_to_check="all")


def test_local_sum_make_vector():
    # To check that rewrite is applied, we must enforce dtype to
    # allow rewrite to occur even if floatX != "float64"
    a, b, c = scalars("abc")
    mv = MakeVector(config.floatX)
    output = mv(a, b, c).sum(dtype="float64")
    rewrite_output = rewrite_graph(output)
    expected_output = cast(
        add(*[cast(value, "float64") for value in [a, b, c]]), dtype="float64"
    )
    assert equal_computations([expected_output], [rewrite_output])

    # Empty axes should return input vector since no sum is applied
    a, b, c = scalars("abc")
    mv = MakeVector(config.floatX)
    output = mv(a, b, c).sum(axis=[])
    rewrite_output = rewrite_graph(output)
    expected_output = mv(a, b, c)
    assert equal_computations([expected_output], [rewrite_output])

    # Empty input should return 0
    mv = MakeVector(config.floatX)
    output = mv().sum()
    rewrite_output = rewrite_graph(output)
    expected_output = pt.as_tensor(0, dtype=config.floatX)
    assert equal_computations([expected_output], [rewrite_output])

    # Single element input should return element value
    a = scalars("a")
    mv = MakeVector(config.floatX)
    output = mv(a).sum()
    rewrite_output = rewrite_graph(output)
    expected_output = cast(a, config.floatX)
    assert equal_computations([expected_output], [rewrite_output])

    # This is a regression test for #653. Ensure that rewrite is NOT
    # applied when user requests float32
    with config.change_flags(floatX="float32", warn_float64="raise"):
        a, b, c = scalars("abc")
        mv = MakeVector(config.floatX)
        output = mv(a, b, c).sum()
        rewrite_output = rewrite_graph(output)
        assert equal_computations([output], [rewrite_output])


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ],
)
def test_local_tensor_scalar_tensor(dtype):
    t_type = TensorType(dtype=dtype, shape=())
    t = t_type()
    s = pt.scalar_from_tensor(t)
    t2 = pt.tensor_from_scalar(s)

    f = function([t], t2, mode=rewrite_mode)
    e = f.maker.fgraph.toposort()
    assert not any(
        n for n in e if isinstance(n.op, TensorFromScalar | ScalarFromTensor)
    )


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ],
)
def test_local_scalar_tensor_scalar(dtype):
    s_type = ps.ScalarType(dtype=dtype)
    s = s_type()
    t = pt.tensor_from_scalar(s)
    s2 = pt.scalar_from_tensor(t)

    f = function([s], s2, mode=rewrite_mode)
    e = f.maker.fgraph.toposort()
    assert not any(
        n for n in e if isinstance(n.op, TensorFromScalar | ScalarFromTensor)
    )


def test_local_useless_split():
    x = matrix("x")
    splits = ivector("splits")
    rewritten = pt.split(x, splits, n_splits=1)
    not_rewritten = pt.split(x, splits, n_splits=3)

    mode = get_default_mode().including("local_useless_split")
    f_rewritten = function([x, splits], rewritten, mode=mode)
    f_not_rewritten = function([x, splits], not_rewritten, mode=mode)

    f_rewritten(np.random.random((4, 4)).astype(config.floatX), [4])
    f_not_rewritten(np.random.random((4, 4)).astype(config.floatX), [1, 2, 1])
    graph_rewritten = f_rewritten.maker.fgraph
    graph_not_rewritten = f_not_rewritten.maker.fgraph

    assert all(
        isinstance(out.owner.op, DeepCopyOp) for out in graph_not_rewritten.outputs
    )
    assert all(isinstance(out.owner.op, DeepCopyOp) for out in graph_rewritten.outputs)

    assert sum(isinstance(node.op, Split) for node in graph_rewritten.apply_nodes) == 0
    assert (
        sum(isinstance(node.op, Split) for node in graph_not_rewritten.apply_nodes) == 1
    )

    assert sum(isinstance(node.op, Assert) for node in graph_rewritten.apply_nodes) == 2
    assert (
        sum(isinstance(node.op, Assert) for node in graph_not_rewritten.apply_nodes)
        == 0
    )

    # The DeepCopy Ops don't have traces, so we can't check "all"
    assert check_stack_trace(f_rewritten, ops_to_check=[Assert])
    assert check_stack_trace(f_not_rewritten, ops_to_check=[Split])


@pytest.mark.parametrize("i", list(range(1, 4)))
def test_local_flatten_lift(i):
    x = tensor4()
    out = pt.flatten(exp(x), i)
    assert out.ndim == i
    mode = get_default_mode()
    mode = mode.including("local_reshape_lift")
    f = function([x], out, mode=mode)
    x_np = np.random.random((5, 4, 3, 2)).astype(config.floatX)
    out_np = f(x_np)
    topo = f.maker.fgraph.toposort()
    shape_out_np = (*x_np.shape[: i - 1], np.prod(x_np.shape[i - 1 :]))
    assert shape_out_np == out_np.shape

    reshape_nodes = [n for n in topo if isinstance(n.op, Reshape)]
    assert len(reshape_nodes) == 1 and pt.is_flat(reshape_nodes[0].outputs[0], ndim=i)
    assert isinstance(topo[-1].op, Elemwise)


class TestLiftTransposeThroughDot:
    def simple_rewrite(self, g):
        out2in(local_useless_elemwise).rewrite(g)
        out2in(local_lift_transpose_through_dot).rewrite(g)
        out2in(local_useless_elemwise).rewrite(g)
        return g

    def test_matrix_matrix(self):
        a, b = matrices("ab")
        g = self.simple_rewrite(FunctionGraph([a, b], [dot(a, b).T], clone=False))
        assert equal_computations(g.outputs, [dot(b.T, a.T)])
        assert check_stack_trace(g, ops_to_check="all")

    def test_row_matrix(self):
        a = vector("a")
        b = matrix("b")
        g = rewrite(
            FunctionGraph([a, b], [dot(a.dimshuffle("x", 0), b).T], clone=False),
            level="stabilize",
        )
        assert equal_computations(g.outputs, [dot(b.T, a.dimshuffle(0, "x"))])
        assert check_stack_trace(g, ops_to_check="all")

    def test_matrix_col(self):
        a = vector("a")
        b = matrix("b")
        g = rewrite(
            FunctionGraph([a, b], [dot(b, a.dimshuffle(0, "x")).T], clone=False),
            level="stabilize",
        )
        assert equal_computations(g.outputs, [dot(a.dimshuffle("x", 0), b.T)])
        assert check_stack_trace(g, ops_to_check="all")


def test_local_upcast_elemwise_constant_inputs():
    s = dvector("s")
    x = pt_sum(log(10**s))
    f = function([s], [pytensor.gradient.grad(x, s)])
    f([-42, -2.1, -1, -0.5, 0, 0.2, 1, 2, 12])

    # This tests a corner case for which the rewrite should not be applied.
    with config.change_flags(floatX="float32"):
        v = lvector()
        function([v], true_div(v, 2))


def test_assert_op_gradient():
    x = vector("x")
    assert_op = Assert()
    cost = pt_sum(assert_op(x, x.size < 2))
    grad = pytensor.gradient.grad(cost, x)
    func = function([x], grad)

    x_val = np.ones(shape=(1,), dtype=config.floatX)
    assert func(x_val) == 1


def test_local_merge_alloc():
    # Add this rewrite to the default mode; otherwise, FAST_COMPILE fails.
    default_mode = get_default_mode()
    rewrite_mode = default_mode.including("local_merge_alloc")

    x = iscalar("x")
    y = iscalar("y")
    y2 = iscalar("y2")
    z = iscalar("z")
    w = iscalar("w")
    m = fscalar("m")
    # case 1
    # Alloc(Alloc(m, x, 1, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    output = pt.alloc(pt.alloc(m, 1, y, 1, 1), x, y, z, w)
    f = function([m, x, y, z, w], output, mode=rewrite_mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, Alloc)
    o = f(0.0, 1, 2, 3, 4)
    assert o.shape == (1, 2, 3, 4)

    # case 2
    # Alloc(Alloc(m, y, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    output = pt.alloc(pt.alloc(m, y, 1, 1), x, y, z, w)
    f = function([m, x, y, z, w], output, mode=rewrite_mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, Alloc)
    o = f(0.0, 1, 2, 3, 4)
    assert o.shape == (1, 2, 3, 4)

    # case 3
    # Alloc(Alloc(m, y1, 1, 1), x, y2, z, w) ->
    #   Alloc(m, x, assert(y1, y1==y2), z, w)
    output = pt.alloc(pt.alloc(m, y, 1, 1), x, y2, z, w)
    f = function([m, x, y, y2, z, w], output, mode=rewrite_mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 3
    assert isinstance(topo[-2].op, Assert)
    assert isinstance(topo[-1].op, Alloc)
    o = f(0.0, 1, 2, 2, 3, 4)
    assert o.shape == (1, 2, 3, 4)
    with pytest.raises((AssertionError, ValueError)):
        f(0.0, 1, 2, 5, 3, 4)


def test_local_useless_alloc():
    useless_alloc = out2in(local_useless_alloc)
    merge_alloc = out2in(local_merge_alloc)

    x = iscalar("x")
    y = iscalar("y")
    y2 = iscalar("y2")
    z = iscalar("z")
    w = iscalar("w")
    m = fscalar("m")

    # case 1
    # Alloc(Alloc(m, x, 1, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    output = pt.alloc(pt.alloc(m, 1, y, 1, 1), x, y, z, w)
    g = FunctionGraph([m, x, y, z, w], [output])

    useless_alloc.rewrite(g)
    merge_alloc.rewrite(g)
    useless_alloc.rewrite(g)

    topo = g.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, Alloc)

    # case 2
    # Alloc(Alloc(m, y, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    output = pt.alloc(pt.alloc(m, y, 1, 1), x, y, z, w)
    g = FunctionGraph([m, x, y, z, w], [output])

    useless_alloc.rewrite(g)
    merge_alloc.rewrite(g)
    useless_alloc.rewrite(g)

    topo = g.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, Alloc)

    # case 3
    # Alloc(Alloc(m, y1, 1, 1), x, y2, z, w) ->
    #   Alloc(m, x, assert(y1, y1==y2), z, w)
    output = pt.alloc(pt.alloc(m, y, 1, 1), x, y2, z, w)
    g = FunctionGraph([m, x, y, y2, z, w], [output])

    useless_alloc.rewrite(g)
    merge_alloc.rewrite(g)
    useless_alloc.rewrite(g)

    topo = g.toposort()
    assert len(topo) == 3
    assert isinstance(topo[-2].op, Assert)
    assert isinstance(topo[-1].op, Alloc)


def test_local_merge_consecutive_specify_shape():
    x = matrix()
    s = pt.as_tensor([iscalar(), iscalar()])
    y = specify_shape(specify_shape(x, s), s)

    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_rewritten_fg = rewrite_graph(
        y_fg,
        clone=False,
        include=["canonicalize", "local_merge_consecutive_specify_shape"],
    )
    y_rewritten = y_rewritten_fg.outputs[0]

    assert isinstance(y_rewritten.owner.op, SpecifyShape)
    assert y_rewritten.owner.inputs[0] == x


def test_local_merge_consecutive_specify_shape2():
    x = tensor3()
    s1, s2, s3, s4 = iscalars("s1", "s2", "s3", "s4")
    y = specify_shape(specify_shape(x, [s1, s2, None]), [None, s3, s4])

    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_rewritten_fg = rewrite_graph(
        y_fg,
        clone=False,
        include=["canonicalize", "local_merge_consecutive_specify_shape"],
    )
    y_rewritten = y_rewritten_fg.outputs[0]

    assert isinstance(y_rewritten.owner.op, SpecifyShape)
    assert tuple(y_rewritten.owner.inputs) == (x, s1, s3, s4)


def test_printing():
    a, b = scalars("ab")
    mv = MakeVector(config.floatX)
    v = mv(a, b)
    assert pprint(v) == "[a, b]"


class TestLocalElemwiseAlloc:
    """

    TODO FIXME: Remove redundant tests.

    """

    dtype = config.floatX

    def setup_method(self):
        self.fast_compile_mode = get_mode("FAST_COMPILE")
        self.fast_run_mode = get_mode("FAST_RUN")

        self.vec = vector("vec", dtype=self.dtype)
        self.mat = matrix("mat", dtype=self.dtype)
        self.tens = tensor3("tens", dtype=self.dtype)

        self.alloc_wo_dep = pt.alloc(self.vec, 2, 2)
        self.alloc_wo_dep_broad = pt.alloc(self.vec, 1, 2)
        self.alloc_w_dep = pt.alloc(self.vec, *self.mat.shape)
        self.alloc_w_dep_broad = pt.alloc(self.vec, 1, *self.mat.shape)
        self.alloc_w_dep_broad2 = pt.alloc(
            self.vec, self.mat.shape[0], self.mat.shape[1], 1
        )
        self.alloc_w_dep_tens = pt.alloc(
            self.vec, self.tens.shape[0], self.tens.shape[1]
        )
        self.tv_wo_dep = pt.alloc(self.vec, 5, 5)
        self.tm_wo_dep = pt.alloc(self.mat, 5, 5, 5)
        self.s = iscalar("s")
        self.tv_w_dep = pt.alloc(self.vec, self.s, self.s)
        self.tm_w_dep = pt.alloc(self.mat, 5, 5, 5)
        self.row = row(dtype=self.dtype)
        self.o = pt.alloc(self.row, 5, 5)

    @staticmethod
    def verify_op_count(f, count, cls):
        assert (
            sum(
                isinstance(elem.op, cls)
                for elem in f.maker.fgraph.toposort()
                if elem.op is not None
            )
            == count
        )

    @pytest.mark.parametrize(
        "expr, x_shape, y_shape, needs_alloc",
        [
            (lambda x, y: pt.mul(pt.alloc(1, *y.shape), x), (1, 2), (3, 2), True),
            (lambda x, y: pt.mul(pt.alloc(1, *y.shape), x), (1, 1), (1, 1), False),
            (lambda x, y: pt.mul(x, pt.alloc(y, 2, 3)), (1, 3), (2, 3), False),
            (
                lambda x, y: pt.mul(
                    pt.alloc(x, 3).dimshuffle("x", 0), y.dimshuffle("x", "x")
                ),
                (),
                (),
                True,
            ),
            (lambda x, y: pt.mul(y, pt.alloc(1, x)), (), (), True),
            (lambda x, y: pt.mul(pt.alloc(x, 15, 1), y), (15, 1), (15, 1), False),
            (lambda x, y: pt.mul(pt.alloc(x, 15, 2), y), (15, 2), (15, 2), False),
            (
                lambda x, y: pt.mul(pt.alloc(x, 15, 1), pt.alloc(y, 15, 1)),
                (15, 1),
                (15, 1),
                False,
            ),
            (
                lambda x, y: pt.mul(pt.alloc(x, 15, 2), pt.alloc(y, 15, 2)),
                (15, 2),
                (15, 2),
                False,
            ),
            (
                lambda x, y: pt.mul(pt.alloc(x, 15, 2).dimshuffle(1, 0), y),
                (15, 2),
                (2, 15),
                False,
            ),
            (lambda x, y: pt.mul(pt.alloc(x, 1, 15, 2), y), (15, 2), (15, 2), False),
            (
                lambda x, y: pt.mul(pt.alloc(x, 1, 15, 2).dimshuffle(0, 2, 1), y),
                (15, 2),
                (2, 15),
                False,
            ),
        ],
    )
    def test_basic(self, expr, x_shape, y_shape, needs_alloc):
        x = pt.tensor(
            dtype="int64", shape=(1 if val == 1 else None for val in x_shape), name="x"
        )
        y = pt.tensor(
            dtype="int64", shape=(1 if val == 1 else None for val in y_shape), name="y"
        )
        z = expr(x, y)

        z_opt = pytensor.function(
            [x, y],
            z,
            mode=get_default_mode().including("local_elemwise_alloc"),
            on_unused_input="ignore",
        )

        nodes = z_opt.maker.fgraph.toposort()
        if needs_alloc:
            # When the final result needs an Alloc, this should be the last node
            # x = scalar; y = vector; mul(x, ones_like(y)) -> alloc(x, y.shape)
            assert isinstance(nodes[-1].op, Alloc)
            nodes = nodes[:-1]

        assert not any(isinstance(node.op, Alloc) for node in nodes)

        # Check results are the same without the optimization
        z_no_opt = pytensor.function(
            [x, y],
            z,
            mode=get_default_mode().excluding("local_elemwise_alloc"),
            on_unused_input="ignore",
        )

        x_val = np.arange(np.prod(x_shape), dtype=np.int64).reshape(x_shape)
        y_val = np.arange(np.prod(y_shape), dtype=np.int64).reshape(y_shape)

        res = z_opt(x_val, y_val)
        exp_res = z_no_opt(x_val, y_val)
        assert np.array_equal(res, exp_res)

    def test_single_input(self):
        """Test that rewrite is not triggered when there is only one `Alloc` in an `Elemwise`."""
        x = pt.matrix("x")
        z = pt.exp(pt.alloc(x, 15, 1))

        z_fg = FunctionGraph(outputs=[z], copy_inputs=False, features=[ShapeFeature()])

        z_opt_fg = rewrite_graph(z_fg, clone=False, include=["local_elemwise_alloc"])
        assert any(isinstance(node.op, Alloc) for node in z_opt_fg.apply_nodes)

    def test_remove_alloc_wo_dimshuffle(self):
        # Exclude `local_useless_alloc`, since it does not introduce
        # `Assert` in all the same cases.
        self.fast_run_mode = self.fast_run_mode.excluding(
            "local_useless_alloc", "local_alloc_sink_dimshuffle"
        )
        func = function(
            [self.vec, self.mat],
            self.alloc_wo_dep + self.mat,
            mode=self.fast_compile_mode,
        )
        self.verify_op_count(func, 1, Alloc)
        self.verify_op_count(func, 0, Assert)
        assert check_stack_trace(func, ops_to_check="all")

        func = function(
            [self.vec, self.mat], self.alloc_wo_dep + self.mat, mode=self.fast_run_mode
        )
        self.verify_op_count(func, 0, Alloc)
        self.verify_op_count(func, 1, SpecifyShape)

        func = function(
            [self.vec, self.mat],
            self.alloc_wo_dep_broad + self.mat,
            mode=self.fast_run_mode,
        )
        self.verify_op_count(func, 0, Alloc)
        self.verify_op_count(func, 1, SpecifyShape)

        # No optimization on alloc without assert
        func = function(
            [self.vec, self.mat],
            self.alloc_w_dep + self.mat,
            mode=self.fast_compile_mode,
        )
        self.verify_op_count(func, 1, Alloc)
        self.verify_op_count(func, 0, Assert)

        func = function(
            [self.vec, self.mat], self.alloc_w_dep + self.mat, mode=self.fast_run_mode
        )
        self.verify_op_count(func, 0, Alloc)
        self.verify_op_count(func, 0, Assert)

        func = function(
            [self.vec, self.mat],
            self.alloc_w_dep_broad + self.mat,
            mode=self.fast_run_mode,
        )
        self.verify_op_count(func, 0, Alloc)
        self.verify_op_count(func, 0, Assert)

        # This was previously not rewritten, but it is now that we
        # have `BroadcastTo`.
        func = function(
            [self.vec, self.mat],
            self.alloc_w_dep_broad2 + self.mat,
            mode=self.fast_run_mode,
        )
        # This graph requires one outer Alloc and an Assert
        # To make sure `mat` is square since we end up doing
        # broadcast_to(x, mat[..., None].shape) + mat[None, ...]
        self.verify_op_count(func, 1, Alloc)
        self.verify_op_count(func, 1, Assert)

    def test_remove_alloc_w_dimshuffle(self):
        func = function(
            [self.vec, self.tens],
            self.alloc_wo_dep.dimshuffle(0, 1, "x") + self.tens,
            mode=self.fast_compile_mode,
        )
        self.verify_op_count(func, 1, Alloc)
        self.verify_op_count(func, 0, Assert)

        func = function(
            [self.vec, self.tens],
            self.alloc_wo_dep.dimshuffle(0, 1, "x") + self.tens,
            mode=self.fast_run_mode,
        )
        self.verify_op_count(func, 0, Alloc)
        self.verify_op_count(func, 1, SpecifyShape)

        func = function(
            [self.vec, self.tens],
            self.alloc_w_dep_tens.dimshuffle(0, 1, "x") + self.tens,
            mode=self.fast_compile_mode,
        )
        self.verify_op_count(func, 1, Alloc)
        self.verify_op_count(func, 0, Assert)

        func = function(
            [self.vec, self.tens],
            self.alloc_w_dep_tens.dimshuffle(0, 1, "x") + self.tens,
            mode=self.fast_run_mode,
        )
        self.verify_op_count(func, 0, Alloc)
        self.verify_op_count(func, 0, Assert)

    def test_multi_input_single_alloc(self):
        # No optimization on dimshuffle with assert
        func = function(
            [self.vec, self.mat],
            self.tv_wo_dep + self.tm_wo_dep,
            mode=self.fast_compile_mode,
        )
        self.verify_op_count(func, 2, Alloc)
        self.verify_op_count(func, 0, Assert)

        func = function(
            [self.vec, self.mat],
            self.tv_wo_dep + self.tm_wo_dep,
            mode=self.fast_run_mode,
        )
        # It still needs an outer alloc to broadcast final shape
        self.verify_op_count(func, 1, Alloc)
        self.verify_op_count(func, 0, Assert)

        # No optimization on dimshuffle without assert
        func = function(
            [self.vec, self.mat, self.s],
            self.tv_w_dep + self.tm_w_dep,
            mode=self.fast_compile_mode,
        )
        self.verify_op_count(func, 2, Alloc)
        self.verify_op_count(func, 0, Assert)

        func = function(
            [self.vec, self.mat, self.s],
            self.tv_w_dep + self.tm_w_dep,
            mode=self.fast_run_mode,
        )
        # It still needs an outer alloc to broadcast final shape
        self.verify_op_count(func, 1, Alloc)
        self.verify_op_count(func, 0, Assert)

    def test_misc(self):
        x = row("x", dtype=self.dtype)
        y = tensor("y", dtype=self.dtype, shape=(None, None, 1))

        out = pt.alloc(x, 5, 5).dimshuffle(0, 1, "x") + y
        func = function([y, x], out, mode=self.fast_run_mode)

        self.verify_op_count(func, 0, Alloc)
        self.verify_op_count(func, 1, SpecifyShape)

        y_val = np.random.random((5, 5, 1)).astype(self.dtype)
        x_val = np.random.random((1, 5)).astype(self.dtype)
        exp_res = np.broadcast_to(x_val, (5, 5))[..., None] + y_val
        assert np.array_equal(func(y_val, x_val), exp_res)


def test_shape_unsafe_tag():
    mode = get_mode("FAST_RUN")
    x = vector("x")
    y = vector("y")
    out = x * y / y

    fn = function([x, y], out, mode=mode)
    np.testing.assert_equal(fn([0, 1], [2, 3, 4]), [0, 1])

    fn = function([x, y], out, mode=mode.excluding("shape_unsafe"))
    with pytest.raises(ValueError):
        fn([0, 1], [2, 3, 4]), [0, 1]


def test_topological_fill_sink_multi_output_client():
    x = float64("x")
    elem_op_with_2_outputs = Elemwise(Composite([x], [x + 1, x + 2]))

    x = pt.vector("x", shape=(1,))
    z = pt.vector("z", shape=(None,))
    bcast_x = pt.full_like(z, x)
    out = pt.add(*elem_op_with_2_outputs(pt.exp(bcast_x)))

    fg = FunctionGraph([x, z], [out], copy_inputs=False)
    topological_fill_sink.rewrite(fg)
    [new_out] = fg.outputs
    expected_out = pt.full_like(z, pt.add(*elem_op_with_2_outputs(pt.exp(x))))
    assert equal_computations([new_out], [expected_out])


def test_topological_fill_sink_broadcastable_change():
    """Test rewrite doesn't fail after a graph replacement that provides a broadcastable change."""
    a = vector("a", shape=(1,))
    b = vector("b", shape=(1,))
    zeros = pt.vector("zeros", shape=(None,))
    initial_out = pt.full_like(zeros, a) + b

    # Make broadcast to zeros irrelevant
    out = graph_replace(initial_out, {zeros: pt.zeros((1,))}, strict=False)

    fg = FunctionGraph([a, b], [out], copy_inputs=False)
    topological_fill_sink.rewrite(fg)
    [new_out] = fg.outputs
    assert equal_computations([new_out], [a + b])
