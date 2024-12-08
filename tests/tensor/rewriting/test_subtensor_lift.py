import numpy as np
import pytest
import unittest_tools as utt

from pytensor import (
    Mode,
    Variable,
    config,
    function,
    shared,
)
from pytensor import scalar as ps
from pytensor import tensor as pt
from pytensor.compile import DeepCopyOp, get_default_mode, get_mode
from pytensor.graph import (
    Constant,
    FunctionGraph,
    RewriteDatabaseQuery,
    Type,
    rewrite_graph,
)
from pytensor.graph.rewriting.basic import check_stack_trace
from pytensor.tensor import (
    add,
    exp,
    inplace,
    iscalar,
    iscalars,
    lscalar,
    lscalars,
    matrix,
    row,
    scalar,
    shape,
    slicetype,
    specify_shape,
    tensor,
    tensor3,
    vector,
)
from pytensor.tensor.basic import MakeVector, make_vector
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.rewriting.subtensor_lift import (
    local_subtensor_make_vector,
    local_subtensor_shape_constant,
)
from pytensor.tensor.shape import SpecifyShape, Unbroadcast, _shape
from pytensor.tensor.subtensor import Subtensor


mode_opt = config.mode
if mode_opt == "FAST_COMPILE":
    mode_opt = "FAST_RUN"
mode_opt = get_mode(mode_opt)


class TestLocalSubtensorLift:
    def test_basic(self):
        # basic test that the Op works
        x = matrix("x")
        f = function([x], exp(x)[0], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check="all")

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)  # first subtensor
        assert prog[1].op == exp
        assert len(prog) == 2
        f([[0, 1], [2, 3]])  # let debugmode test something

    def test_basic_1(self):
        # as test0, but we reuse the output of the elemwise
        # So we should not lift the subtensor
        x = matrix("x")
        f = function([x], [exp(x)[0], exp(x)], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=[Subtensor, Elemwise])

        prog = f.maker.fgraph.toposort()
        assert prog[0].op == exp
        assert isinstance(prog[1].op, Subtensor)  # first subtensor
        assert isinstance(prog[2].op, DeepCopyOp)
        assert len(prog) == 3
        f([[0, 1], [2, 3]])  # let debugmode test something

    def test_basic_2(self):
        # basic test that the optimization work with scalar broadcasted
        x = matrix("x")
        y = scalar("y")
        z = matrix("z")
        f = function([x, y, z], exp(x + y + z)[0], mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, DimShuffle)
        assert isinstance(prog[2].op, Subtensor)
        assert isinstance(prog[3].op.scalar_op, ps.Composite)  # Composite{add,add}
        assert len(prog) == 4

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=[Subtensor])

        # let debugmode test something
        f([[0, 1], [2, 3]], 4, [[4, 5], [6, 7]])

    def test_basic_3(self):
        # as 1, but take a slice
        x = matrix("x")
        y = scalar("y")
        z = matrix("z")
        f = function([x, y, z], exp(x + y + z)[0:2], mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, DimShuffle)
        assert isinstance(prog[2].op, Subtensor)
        assert isinstance(prog[3].op.scalar_op, ps.Composite)  # Composite{add,add}
        assert len(prog) == 4

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=[Subtensor])

        # let debugmode test something
        f([[0, 1], [2, 3]], 4, [[4, 5], [6, 7]])

    def test_basic_4(self):
        # basic test that the optimization does work with broadcasting
        # for unary elemwise.
        y = vector("y")
        f = function([y], exp(y.dimshuffle(0, "x"))[0], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check="all")

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, DimShuffle)
        assert isinstance(prog[1].op, Subtensor)
        assert prog[2].op == exp
        assert len(prog) == 3
        f([4, 5])  # let debugmode test something

    @utt.assertFailure_fast
    def test_basic_5(self):
        # basic test that the optimization doesn't work with broadcasting
        # ... It *could* be extended to,
        # ... but right now it doesn't, so it shouldn't try.
        x = matrix("x")
        y = vector("y")
        f = function([x, y], exp(x + y)[0], mode=mode_opt)

        # Opt doesn't apply, so no need for check_stack_trace
        # assert check_stack_trace(f, ops_to_check='all')

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, DimShuffle)
        assert prog[1].op == add
        assert isinstance(prog[2].op, Subtensor)  # first subtensor
        assert prog[3].op == inplace.exp_inplace
        assert len(prog) == 4
        f([[0, 1], [2, 3]], [4, 5])  # let debugmode test something

    def test_basic_6(self):
        # test that we don't lift when we reuse the output of the
        # elemwise for other computation.
        x = matrix("x")
        y = vector("y")
        f = function([x, y], [exp(x + y)[0], exp(x + y) + x], mode=mode_opt)

        # Opt doesn't apply, so no need for check_stack_trace
        # assert check_stack_trace(f, ops_to_check=Subtensor)

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, DimShuffle)
        assert isinstance(prog[1].op.scalar_op, ps.Composite)  # Composite{add,exp}
        # first subtensor
        assert isinstance(prog[2].op, Subtensor)
        assert len(prog) == 3
        f([[0, 1], [2, 3]], [4, 5])  # let debugmode test something

    def test_basic_7(self):
        # basic test that the optimization works with a scalar as input,
        # and a scalar as output (no broadcasting of the scalar needed).
        # The optimization used to fail and display an ERROR message.

        x = vector("x")
        y = scalar("y")
        f = function([x, y], exp(x + y)[0], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        # Composite{add,exp}
        assert isinstance(prog[1].op.scalar_op, ps.Composite)
        assert len(prog) == 2
        f([1, 2, 3], 4)  # let debugmode test something

    def test_basic_8(self):
        # Test that Subtensor(Unbroadcast(x)) gets optimized into
        # Unbroadcast(Subtensor(x)).

        # test basic case
        x = row("x")
        xval = np.random.random((1, 10)).astype(config.floatX)
        assert x.broadcastable == (True, False)
        newx = Unbroadcast(0)(x)
        assert newx.broadcastable == (False, False)

        f1 = function([x], newx[:2, :5], mode=mode_opt)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f1, ops_to_check=[Subtensor, Unbroadcast])
        prog = f1.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, Unbroadcast)
        assert (f1(xval) == xval[:2, :5]).all()

        # corner case 1: Unbroadcast changes dims which are dropped through subtensor
        y = tensor(dtype="float64", shape=(1, 10, 1, 3), name="x")
        yval = np.random.random((1, 10, 1, 3)).astype(config.floatX)
        assert y.broadcastable == (True, False, True, False)
        newy = Unbroadcast(0, 2)(y)
        assert newy.broadcastable == (False, False, False, False)

        f2 = function([y], newy[:, 3, 0, :], mode=mode_opt)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f2, ops_to_check=[Subtensor, Unbroadcast])
        prog = f2.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, Unbroadcast)
        assert (f2(yval) == yval[:, 3, 0, :]).all()

        # corner case 2: subtensor idx_list is shorter than resulting broadcast pattern
        f3 = function([y], newy[:, 3, 0], mode=mode_opt)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f3, ops_to_check=[Subtensor, Unbroadcast])
        prog = f3.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, Unbroadcast)
        assert (f3(yval) == yval[:, 3, 0]).all()

        # corner case 3: subtensor idx_list is shorter than Unbroadcast.axis
        z = tensor(dtype="float64", shape=(4, 10, 3, 1), name="x")
        zval = np.random.random((4, 10, 3, 1)).astype(config.floatX)
        assert z.broadcastable == (False, False, False, True)
        newz = Unbroadcast(3)(z)
        assert newz.broadcastable == (False, False, False, False)

        f4 = function([z], newz[:, 3, 0], mode=mode_opt)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f4, ops_to_check=[Subtensor, Unbroadcast])
        prog = f4.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, Unbroadcast)
        assert (f4(zval) == zval[:, 3, 0]).all()


def test_local_subtensor_of_alloc():
    # DebugMode should detect if something goes wrong.
    # test shape combination of odd and event shape.
    for s in [(3, 5), (4, 6), (3, 8), (4, 7), (1, 5), (5, 1)]:
        x = tensor(
            dtype=config.floatX,
            shape=(1 if s[0] == 1 else None, 1 if s[1] == 1 else None),
        )

        xval = np.zeros(s, dtype=config.floatX)
        yval = np.arange(s[1], dtype=config.floatX)

        for y in [shared(yval), pt.constant([1.0])]:
            # The rows of yx are copies of y
            yx = pt.alloc(y, x.shape[0], x.shape[1])

            # Slice of each row
            z_mat = yx[:, 3:]
            assert z_mat.ndim == 2

            # Only one column
            z_vec = yx[:, 3]
            assert z_vec.ndim == 1
            # results are vector
            slicess = []
            if s[0] != 1:
                slicess.append((2, slice(None)))
            if s[1] != 1:
                slicess.append((slice(None), 3))

            # results are matrix
            slicess += [
                (slice(None), slice(3, None)),
                (slice(3, None),),
                (slice(3, None), slice(3, None)),
                (slice(1, 3), slice(None, -1)),
                (slice(None, None, 2)),
                (slice(1, None, 2)),
            ]
            for slices in slicess:
                z = yx.__getitem__(slices)
                f = function([x], z)
                if config.mode != "FAST_COMPILE":
                    # Subtensor can be in the input of Alloc
                    assert not isinstance(f.maker.fgraph.toposort()[-1].op, Subtensor)
                val = f(xval)
                assert xval.__getitem__(slices).shape == val.shape


@pytest.mark.parametrize(
    "x, s, idx, x_val, s_val",
    [
        (
            vector(),
            (iscalar(),),
            (1,),
            np.array([1, 2], dtype=config.floatX),
            np.array([2], dtype=np.int64),
        ),
        (
            matrix(),
            (iscalar(), iscalar()),
            (1,),
            np.array([[1, 2], [3, 4]], dtype=config.floatX),
            np.array([2, 2], dtype=np.int64),
        ),
        (
            matrix(),
            (iscalar(), iscalar()),
            (0,),
            np.array([[1, 2, 3], [4, 5, 6]], dtype=config.floatX),
            np.array([2, 3], dtype=np.int64),
        ),
        (
            matrix(),
            (iscalar(), iscalar()),
            (1, 1),
            np.array([[1, 2, 3], [4, 5, 6]], dtype=config.floatX),
            np.array([2, 3], dtype=np.int64),
        ),
        (
            tensor3(),
            (iscalar(), iscalar(), iscalar()),
            (-1,),
            np.arange(2 * 3 * 5, dtype=config.floatX).reshape((2, 3, 5)),
            np.array([2, 3, 5], dtype=np.int64),
        ),
        (
            tensor3(),
            (iscalar(), iscalar(), iscalar()),
            (-1, 0),
            np.arange(2 * 3 * 5, dtype=config.floatX).reshape((2, 3, 5)),
            np.array([2, 3, 5], dtype=np.int64),
        ),
    ],
)
def test_local_subtensor_SpecifyShape_lift(x, s, idx, x_val, s_val):
    y = specify_shape(x, s)[idx]
    assert isinstance(y.owner.inputs[0].owner.op, SpecifyShape)

    rewrites = RewriteDatabaseQuery(include=[None])
    no_rewrites_mode = Mode(optimizer=rewrites)

    y_val_fn = function([x, *s], y, on_unused_input="ignore", mode=no_rewrites_mode)
    y_val = y_val_fn(*([x_val, *s_val]))

    # This optimization should appear in the canonicalizations
    y_opt = rewrite_graph(y, clone=False)

    if y.ndim == 0:
        # SpecifyShape should be removed altogether
        assert isinstance(y_opt.owner.op, Subtensor)
        assert y_opt.owner.inputs[0] is x
    else:
        assert isinstance(y_opt.owner.op, SpecifyShape)

    y_opt_fn = function([x, *s], y_opt, on_unused_input="ignore")
    y_opt_val = y_opt_fn(*([x_val, *s_val]))

    assert np.allclose(y_val, y_opt_val)


@pytest.mark.parametrize(
    "x, s, idx",
    [
        (
            matrix(),
            (iscalar(), iscalar()),
            (slice(1, None),),
        ),
        (
            matrix(),
            (iscalar(), iscalar()),
            (slicetype(),),
        ),
        (
            matrix(),
            (iscalar(), iscalar()),
            (1, 0),
        ),
    ],
)
def test_local_subtensor_SpecifyShape_lift_fail(x, s, idx):
    y = specify_shape(x, s)[idx]

    # This optimization should appear in the canonicalizations
    y_opt = rewrite_graph(y, clone=False)

    assert not isinstance(y_opt.owner.op, SpecifyShape)


class TestLocalSubtensorMakeVector:
    mode = get_mode("FAST_RUN").including("local_subtensor_make_vector")

    def test_scalar_idx(self):
        x, y, z = lscalars("xyz")
        v = make_vector(x, y, z)
        f = function([x, y, z], v[0], mode=self.mode)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, DeepCopyOp)
        assert f(0, 1, 2) == 0

    def test_idx_symbolic(self):
        x, y, z = iscalars("xyz")
        v = MakeVector("int32")(x, y, z)
        idx = pt.as_tensor([0], dtype=np.int64)
        f = function([x, y, z], v[idx], mode=self.mode)

        opt_fgraph = f.maker.fgraph
        assert opt_fgraph.outputs[0].dtype == "int32"
        assert isinstance(opt_fgraph.outputs[0].owner.op, MakeVector)
        assert f(0, 1, 2) == np.array([0], dtype=np.int32)

    def test_slice_idx_start(self):
        x, y, z = iscalars("xyz")
        v = MakeVector("int32")(x, y, z)
        f = function([x, y, z], v[1:], mode=self.mode, on_unused_input="ignore")

        opt_fgraph = f.maker.fgraph
        assert opt_fgraph.outputs[0].dtype == "int32"
        assert isinstance(opt_fgraph.outputs[0].owner.op, MakeVector)
        assert len(opt_fgraph.outputs[0].owner.inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 1 and r[1] == 2

    def test_slice_idx_stop(self):
        x, y, z = lscalars("xyz")
        v = make_vector(x, y, z)
        f = function([x, y, z], v[:2], mode=self.mode)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, MakeVector)
        assert len(prog[0].inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 0 and r[1] == 1

    def test_slice_idx_step(self):
        x, y, z = lscalars("xyz")
        v = make_vector(x, y, z)
        f = function([x, y, z], v[::2], mode=self.mode)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, MakeVector)
        assert len(prog[0].inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 0 and r[1] == 2

    def test_AdvancedSubtensor1_idx(self):
        x, y, z = lscalars("xyz")
        v = make_vector(x, y, z)
        f = function([x, y, z], v[[0, 2]], mode=self.mode)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, MakeVector)
        assert len(prog[0].inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 0 and r[1] == 2

    def test_MakeVector_idx(self):
        x, y, z, q = lscalars("xyzq")
        v = make_vector(x, y, z)
        q = make_vector(0, 2)
        f = function([x, y, z], v[q], mode=self.mode)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, MakeVector)
        assert len(prog[0].inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 0 and r[1] == 2

    def test_stack_trace(self):
        x, y, z = lscalars("xyz")
        v = make_vector(x, y, z)

        mode = get_default_mode().including("local_subtensor_make_vector")

        # list of subtensor cases, where local_subtensor_make_vector
        # inserts a new MakeVector node
        v_subtensors = [v[:2], v[::2], v[[0, 2]]]

        for v_subtensor in v_subtensors:
            f = function([x, y, z], v_subtensor, mode=mode)
            assert check_stack_trace(f, ops_to_check="all")

    def test_empty_subtensor(self):
        x, y = lscalars("xy")
        v = make_vector(x, y)
        out = v[()]

        fgraph = FunctionGraph(outputs=[out], clone=False)
        node = fgraph.outputs[0].owner
        assert isinstance(node.op, Subtensor)

        assert local_subtensor_make_vector.transform(fgraph, node) == [v]


def test_local_subtensor_shape_constant():
    x = tensor(dtype=np.float64, shape=(1, None)).shape[0]
    (res,) = local_subtensor_shape_constant.transform(None, x.owner)
    assert isinstance(res, Constant)
    assert res.data == 1

    # Make sure it's part of the canonicalizations
    res = rewrite_graph(x)
    assert isinstance(res, Constant)
    assert res.data == 1

    x = _shape(tensor(dtype=np.float64, shape=(1, None)))[lscalar()]
    assert not local_subtensor_shape_constant.transform(None, x.owner)

    x = _shape(tensor(dtype=np.float64, shape=(1, None)))[0:]
    assert not local_subtensor_shape_constant.transform(None, x.owner)

    x = _shape(tensor(dtype=np.float64, shape=(1, None)))[lscalar() :]
    assert not local_subtensor_shape_constant.transform(None, x.owner)

    x = _shape(tensor(dtype=np.float64, shape=(1, 1)))[1:]
    (res,) = local_subtensor_shape_constant.transform(None, x.owner)
    assert isinstance(res, Constant)
    assert np.array_equal(res.data, [1])

    x = _shape(tensor(dtype=np.float64, shape=(None, 1, 1)))[1:]
    (res,) = local_subtensor_shape_constant.transform(None, x.owner)
    assert isinstance(res, Constant)
    assert np.array_equal(res.data, [1, 1])

    # A test for a non-`TensorType`
    class MyType(Type):
        def filter(self, *args, **kwargs):
            raise NotImplementedError()

        def __eq__(self, other):
            return isinstance(other, MyType) and other.thingy == self.thingy

    x = shape(Variable(MyType(), None, None))[0]

    assert not local_subtensor_shape_constant.transform(None, x.owner)
