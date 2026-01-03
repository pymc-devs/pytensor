import numpy as np
import pytest

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
    Op,
    RewriteDatabaseQuery,
    Type,
    rewrite_graph,
)
from pytensor.graph.basic import equal_computations
from pytensor.graph.rewriting.basic import check_stack_trace
from pytensor.printing import debugprint
from pytensor.tensor import (
    add,
    dvector,
    exp,
    iscalar,
    iscalars,
    lscalar,
    lscalars,
    matrix,
    shape,
    slicetype,
    specify_shape,
    tensor,
    tensor3,
    vector,
)
from pytensor.tensor.basic import MakeVector, concatenate, expand_dims, make_vector
from pytensor.tensor.blas import Dot22, Gemv
from pytensor.tensor.blas_c import CGemv
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import Dot
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.rewriting.subtensor_lift import (
    local_subtensor_make_vector,
    local_subtensor_of_batch_dims,
    local_subtensor_shape_constant,
)
from pytensor.tensor.shape import SpecifyShape, _shape
from pytensor.tensor.special import softmax
from pytensor.tensor.subtensor import AdvancedSubtensor, Subtensor


mode_opt = config.mode
if mode_opt == "FAST_COMPILE":
    mode_opt = "FAST_RUN"
mode_opt = get_mode(mode_opt)


NO_OPTIMIZATION_MODE = Mode(linker="py", optimizer=None)


class TestLocalSubtensorOfBatchDims:
    def test_unary_multiple_clients(self):
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

        x_test = [[0, 1], [2, 3]]
        res1, res2 = f(x_test)
        np.testing.assert_allclose(
            res1,
            np.exp(x_test)[0],
        )
        np.testing.assert_allclose(res2, np.exp(x_test))

    def test_multinary_multiple_clients(self):
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

        x_test = np.array([[0, 1], [2, 3]]).astype(x.dtype)
        y_test = np.array([4, 5]).astype(y.dtype)
        res1, res2 = f(x_test, y_test)
        np.testing.assert_allclose(
            res1,
            np.exp(x_test + y_test)[0],
        )
        np.testing.assert_allclose(
            res2,
            np.exp(x_test + y_test) + x_test,
        )

    @pytest.mark.parametrize(
        "original_fn, expected_fn",
        [
            # Unary integer indexing
            (lambda x, y: exp(x)[0], lambda x, y: exp(x[0])),
            # Unary integer with expand_dims
            (lambda x, y: exp(x[:, None])[0], lambda x, y: exp(x[0][None])),
            # Integer indexing on non-broadcastable dimension
            (lambda x, y: add(x, y)[0], lambda x, y: add(x[0], y[0])),
            # Slice indexing on non-broadcastable dimension
            (lambda x, y: add(x, y)[1:], lambda x, y: add(x[1:], y[1:])),
            # Integer indexing on broacastable dimension
            (lambda x, y: add(x[None], y[None])[0], lambda x, y: add(x, y)),
            (lambda x, y: add(x[None], y[None])[0, 1], lambda x, y: add(x[1], y[1])),
            (
                lambda x, y: add(x[None, :], y[:, None])[2],
                lambda x, y: add(x, y[2][None]),
            ),
            (
                lambda x, y: add(x[:, None], y[None, :])[:, 2],
                lambda x, y: add(x, y[2][None]),
            ),
            # Slice indexing on broadcastable dimension
            (
                lambda x, y: add(x[None], y[None])[1:],
                lambda x, y: add(x[None][1:], y[None][1:]),
            ),
            (
                lambda x, y: add(x[None, :], y[:, None])[1:],
                lambda x, y: add(x[None, :], y[1:][:, None]),
            ),
        ],
    )
    def test_elemwise(self, original_fn, expected_fn):
        rng = np.random.default_rng(257)
        x = pt.matrix("x", shape=(5, 3))
        y = pt.matrix("y", shape=(5, 3))
        x_test = rng.normal(size=x.type.shape).astype(x.dtype)
        y_test = rng.normal(size=y.type.shape).astype(y.dtype)

        out = original_fn(x, y)
        expected_opt_out = expected_fn(x, y)
        opt_out = rewrite_graph(out)
        assert equal_computations([opt_out], [expected_opt_out]), debugprint(
            [expected_opt_out, opt_out], print_type=True
        )
        eval_kwargs = dict(mode=NO_OPTIMIZATION_MODE, on_unused_input="ignore")
        np.testing.assert_allclose(
            opt_out.eval({x: x_test, y: y_test}, **eval_kwargs),
            out.eval({x: x_test, y: y_test}, **eval_kwargs),
        )

    def test_elemwise_multiple_clients(self):
        x = pt.matrix("x", shape=(5, 3))
        y = pt.matrix("y", shape=(5, 3))
        out1 = add(x, y)
        out2 = out1[0]

        # Rewrite should fail when another node uses out1 directly (in this case it's an extra output)
        fgraph = FunctionGraph([x, y], [out1, out2], clone=False)
        assert local_subtensor_of_batch_dims.transform(fgraph, out2.owner) is None

        # Otherwise it should work
        fgraph.remove_output(0)
        assert local_subtensor_of_batch_dims.transform(fgraph, out2.owner) is not None

    def test_blockwise(self):
        class CoreTestOp(Op):
            itypes = [dvector, dvector]
            otypes = [dvector]

            def perform(self, node, inputs, output_storage):
                output_storage[0][0] = np.convolve(*inputs, mode="valid")

        core_test_op = CoreTestOp()
        block_test_op = Blockwise(core_test_op, signature="(a),(b)->(c)")

        x = tensor3("x", shape=(7, 5, 11), dtype="float64")
        y = tensor("y", shape=(7, 33), dtype="float64")
        out = block_test_op(x, y[:, None, :])
        assert isinstance(out.owner.op, Blockwise)

        out_sliced = out[2:][:, 3:]
        rewritten_out_sliced = rewrite_graph(out_sliced)
        expected_out_sliced = block_test_op(x[2:, 3:], y[2:][:, None, :])
        assert equal_computations([rewritten_out_sliced], [expected_out_sliced])

        rng = np.random.default_rng(191)
        x_test = rng.normal(size=x.type.shape).astype(x.type.dtype)
        y_test = rng.normal(size=y.type.shape).astype(y.type.dtype)
        np.testing.assert_allclose(
            rewritten_out_sliced.eval(
                {x: x_test, y: y_test}, mode=NO_OPTIMIZATION_MODE
            ),
            out_sliced.eval({x: x_test, y: y_test}, mode=NO_OPTIMIZATION_MODE),
        )

        # Check slice on core dims
        out_sliced = out[2:][:, 0][:, 4:]
        rewritten_out_sliced = rewrite_graph(out_sliced)
        expected_out_sliced = block_test_op(x[2:, 0], y[2:])[:, 4:]
        assert equal_computations([rewritten_out_sliced], [expected_out_sliced])


def test_local_subtensor_of_dot():
    m1 = matrix()
    m2 = matrix()
    d1 = np.arange(6).reshape((3, 2)).astype(config.floatX)
    d2 = np.arange(8).reshape((2, 4)).astype(config.floatX) + 10
    mode = get_default_mode().including("local_subtensor_of_dot")

    def test_equality(a, b):
        return a.shape == b.shape and np.allclose(a, b)

    # [cst]
    f = function([m1, m2], pt.dot(m1, m2)[1], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert test_equality(f(d1, d2), np.dot(d1, d2)[1])
    # DimShuffle happen in FAST_COMPILE
    assert isinstance(topo[-1].op, CGemv | Gemv | DimShuffle)

    # slice
    f = function([m1, m2], pt.dot(m1, m2)[1:2], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert test_equality(f(d1, d2), np.dot(d1, d2)[1:2])
    assert isinstance(topo[-1].op, Dot | Dot22)

    m1 = tensor3()
    m2 = tensor3()
    idx = iscalar()
    d1 = np.arange(30).reshape(2, 5, 3).astype(config.floatX)
    d2 = np.arange(72).reshape(4, 3, 6).astype(config.floatX) + 100

    f = function([m1, m2, idx], pt.dot(m1, m2)[idx, 1:4, :, idx:], mode=mode)
    assert test_equality(f(d1, d2, 1), np.dot(d1, d2)[1, 1:4, :, 1:])
    # if we return the gradients. We need to use same mode as before.
    assert check_stack_trace(f, ops_to_check="last")

    f = function([m1, m2, idx], pt.dot(m1, m2)[1:4, :, idx:, idx], mode=mode)
    assert test_equality(f(d1, d2, 1), np.dot(d1, d2)[1:4, :, 1:, 1])

    # Now test that the stack trace is copied over properly,
    # if we return the gradients. We need to use same mode as before.
    assert check_stack_trace(f, ops_to_check="last")


@pytest.mark.parametrize(
    "original_fn, expected_fn",
    [
        # Indexing before axis of reduction
        (lambda x: pt_sum(x, axis=2)[0], lambda x: pt_sum(x[0], axis=1)),
        (lambda x: pt_sum(x, axis=2)[0, 1], lambda x: pt_sum(x[0, 1], axis=None)),
        (lambda x: pt_sum(x, axis=2)[1:], lambda x: pt_sum(x[1:], axis=2)),
        # Indexing "at" axis of reduction
        (lambda x: pt_sum(x, axis=0)[2], lambda x: pt_sum(x[:, 2], axis=0)),
        (lambda x: pt_sum(x, axis=0)[:-2], lambda x: pt_sum(x[:, :-2], axis=0)),
        # Index after axis of reduction
        (lambda x: pt_sum(x, axis=0)[:, 1:], lambda x: pt_sum(x[:, :, 1:], axis=0)),
        # Index before and after axis reduction
        (lambda x: pt_sum(x, axis=1)[-2, 1:], lambda x: pt_sum(x[-2, :, 1:], axis=0)),
        (lambda x: pt_sum(x, axis=1)[1:, -2], lambda x: pt_sum(x[1:, :, -2], axis=1)),
    ],
)
def test_local_subtensor_of_reduce(original_fn, expected_fn):
    rng = np.random.default_rng(245)
    x = pt.tensor("x", shape=(5, 3, 2))
    x_test = rng.normal(size=x.type.shape).astype(x.dtype)

    out = original_fn(x)
    expected_opt_out = expected_fn(x)
    opt_out = rewrite_graph(out, exclude=("local_convert_negative_indices",))
    assert equal_computations([opt_out], [expected_opt_out]), debugprint(
        [expected_opt_out, opt_out], print_type=True
    )
    np.testing.assert_allclose(
        opt_out.eval({x: x_test}, mode=NO_OPTIMIZATION_MODE),
        out.eval({x: x_test}, mode=NO_OPTIMIZATION_MODE),
    )


@pytest.mark.parametrize(
    "original_fn, expected_fn",
    [
        # Lift single index that does not ovelap with axis of softmax
        (lambda x: softmax(x, axis=1)[0], lambda x: softmax(x[0], axis=0)),
        (lambda x: softmax(x, axis=1)[1:], lambda x: softmax(x[1:], axis=1)),
        (lambda x: softmax(x, axis=0)[:, 0], lambda x: softmax(x[:, 0], axis=0)),
        (lambda x: softmax(x, axis=0)[:, 1:], lambda x: softmax(x[:, 1:], axis=0)),
        # Do nothing to single index over axis of softmax
        (lambda x: softmax(x, axis=0)[0], lambda x: softmax(x, axis=0)[0]),
        (lambda x: softmax(x, axis=1)[:, 1:], lambda x: softmax(x, axis=1)[:, 1:]),
        # Split indexing on axis of softmax
        (lambda x: softmax(x, axis=0)[1:, 0], lambda x: softmax(x[:, 0], axis=0)[1:]),
        (lambda x: softmax(x, axis=1)[1:, 0], lambda x: softmax(x[1:], axis=1)[:, 0]),
        (
            lambda x: softmax(x, axis=0)[0, :5:2],
            lambda x: softmax(x[:, :5:2], axis=0)[0],
        ),
        (lambda x: softmax(x, axis=1)[0, :5:2], lambda x: softmax(x[0], axis=0)[:5:2]),
    ],
)
def test_local_subtensor_of_softmax(original_fn, expected_fn):
    rng = np.random.default_rng(230)
    x = pt.matrix("x", shape=(5, 3))
    x_test = rng.normal(size=x.type.shape).astype(x.dtype)

    out = original_fn(x)
    expected_opt_out = expected_fn(x)
    opt_out = rewrite_graph(out)
    assert equal_computations([opt_out], [expected_opt_out]), debugprint(
        [expected_opt_out, opt_out], print_type=True
    )
    np.testing.assert_allclose(
        opt_out.eval({x: x_test}, mode=NO_OPTIMIZATION_MODE),
        out.eval({x: x_test}, mode=NO_OPTIMIZATION_MODE),
    )


@pytest.mark.parametrize(
    "original_fn, expected_fn",
    [
        # Integer indexing
        (lambda x: expand_dims(x, axis=0)[0], lambda x: x),
        (
            lambda x: expand_dims(x, axis=1)[0],
            lambda x: expand_dims(x[0], axis=0),
        ),
        (
            lambda x: expand_dims(x, axis=(1, 3))[0],
            lambda x: expand_dims(x[0], axis=(0, 2)),
        ),
        # Slice indexing
        (
            lambda x: expand_dims(x, axis=1)[1:],
            lambda x: expand_dims(x[1:], axis=1),
        ),
        (
            lambda x: expand_dims(x, axis=(1, 3))[1:],
            lambda x: expand_dims(x[1:], axis=(1, 3)),
        ),
        # Not supported, slice indexing on expanded dimension
        (
            lambda x: expand_dims(x, axis=0)[1:],
            lambda x: expand_dims(x, axis=0)[1:],
        ),
        # Mixed indexing
        (
            lambda x: expand_dims(x, axis=1)[0, :, 1:],
            lambda x: expand_dims(x[0, 1:], axis=0),
        ),
        (
            lambda x: expand_dims(x, axis=1)[1:, :, 0],
            lambda x: expand_dims(x[1:, 0], axis=1),
        ),
        (
            lambda x: expand_dims(x, axis=(1, 2))[1:, :, 0],
            lambda x: expand_dims(x[1:], axis=1),
        ),
    ],
)
def test_local_subtensor_of_expand_dims(original_fn, expected_fn):
    rng = np.random.default_rng(232)
    x = tensor("x", shape=(5, 3))
    x_test = rng.normal(size=x.type.shape).astype(x.dtype)

    out = original_fn(x)
    expected_opt_out = expected_fn(x)
    opt_out = rewrite_graph(out)
    assert equal_computations([opt_out], [expected_opt_out]), debugprint(
        [opt_out, expected_opt_out], print_type=True
    )
    np.testing.assert_allclose(
        opt_out.eval({x: x_test}, mode=NO_OPTIMIZATION_MODE),
        out.eval({x: x_test}, mode=NO_OPTIMIZATION_MODE),
    )


@pytest.mark.parametrize(
    "original_fn, expected_fn",
    [
        (lambda x: x.transpose(2, 1, 0)[0], lambda x: x[:, :, 0].transpose(1, 0)),
        (lambda x: x.transpose(2, 1, 0)[:, :, 1:], lambda x: x[1:].transpose(2, 1, 0)),
        (
            lambda x: x.transpose(2, 1, 0)[0, :1, 1:],
            lambda x: x[1:, :1, 0].transpose(1, 0),
        ),
        (lambda x: x.transpose(2, 1, 0)[0, :1, 1], lambda x: x[1, :1, 0]),
    ],
)
def test_local_subtensor_of_transpose(original_fn, expected_fn):
    rng = np.random.default_rng(232)
    x = tensor("x", shape=(7, 5, 3))
    x_test = rng.normal(size=x.type.shape).astype(x.dtype)

    out = original_fn(x)
    expected_opt_out = expected_fn(x)
    opt_out = rewrite_graph(out)
    assert equal_computations([opt_out], [expected_opt_out]), debugprint(
        [expected_opt_out, opt_out], print_type=True
    )
    np.testing.assert_allclose(
        opt_out.eval({x: x_test}, mode=NO_OPTIMIZATION_MODE),
        out.eval({x: x_test}, mode=NO_OPTIMIZATION_MODE),
    )


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


class TestLocalSubtensorSpecifyShapeLift:
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
    def test_local_subtensor_SpecifyShape_lift(self, x, s, idx, x_val, s_val):
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
    def test_local_subtensor_SpecifyShape_lift_fail(self, x, s, idx):
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


shared_axis = shared(1, "axis")


@pytest.mark.parametrize(
    "original_fn, expected_fn",
    [
        (
            lambda x, y: concatenate([x, y], axis=1)[1],
            lambda x, y: concatenate([x[1], y[1]], axis=0),
        ),
        (
            lambda x, y: concatenate([x, y], axis=-1)[1:],
            lambda x, y: concatenate([x[1:], y[1:]], axis=1),
        ),
        # Indexing on both axis of concatenation and somewhere else:
        (
            lambda x, y: concatenate([x, y], axis=1)[0, 1:],
            lambda x, y: concatenate([x[0], y[0]], axis=0)[1:],
        ),
        # Not supported, indexing on axis of concatenation
        (
            lambda x, y: concatenate([x, y], axis=0)[0],
            lambda x, y: concatenate([x, y], axis=0)[0],
        ),
        (
            lambda x, y: concatenate([x, y], axis=1)[:, 1:],
            lambda x, y: concatenate([x, y], axis=1)[:, 1:],
        ),
        # Not supported, axis of concatenation is dynamically determined
        (
            lambda x, y: concatenate([x, y], axis=shared_axis)[1],
            lambda x, y: concatenate([x, y], axis=shared_axis)[1],
        ),
    ],
)
def test_local_subtensor_of_join(original_fn, expected_fn):
    rng = np.random.default_rng(257)
    x = pt.matrix("x", shape=(5, 3))
    y = pt.matrix("y", shape=(5, 3))
    x_test = rng.normal(size=x.type.shape).astype(x.dtype)
    y_test = rng.normal(size=y.type.shape).astype(y.dtype)

    out = original_fn(x, y)
    expected_opt_out = expected_fn(x, y)
    opt_out = rewrite_graph(out)
    assert equal_computations([opt_out], [expected_opt_out]), debugprint(
        [expected_opt_out, opt_out], print_type=True
    )
    np.testing.assert_allclose(
        opt_out.eval({x: x_test, y: y_test}, mode=NO_OPTIMIZATION_MODE),
        out.eval({x: x_test, y: y_test}, mode=NO_OPTIMIZATION_MODE),
    )


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


@pytest.mark.parametrize(
    "original_fn, supported",
    [
        (lambda x: x[:, [0, 1]][0], True),
        (lambda x: x[:, [0, 1], [0, 0]][1:], True),
        (lambda x: x[:, [[0, 1], [0, 0]]][1:], True),
        # Not supported, basic indexing on advanced indexing dim
        (lambda x: x[[0, 1]][0], False),
        # Not implemented, basic indexing on the right of advanced indexing
        (lambda x: x[[0, 1]][:, 0], False),
        # Not implemented, complex flavors of advanced indexing
        (lambda x: x[:, None, [0, 1]][0], False),
        (lambda x: x[:, 5:, [0, 1]][0], False),
        (lambda x: x[:, :, np.array([True, False, False])][0], False),
        (lambda x: x[[0, 1], :, [0, 1]][:, 0], False),
    ],
)
def test_local_subtensor_of_adv_subtensor(original_fn, supported):
    rng = np.random.default_rng(257)
    x = pt.tensor3("x", shape=(7, 5, 3))
    x_test = rng.normal(size=x.type.shape).astype(x.dtype)

    out = original_fn(x)
    opt_out = rewrite_graph(
        out, include=("canonicalize", "local_subtensor_of_adv_subtensor")
    )
    # The graphs generated are too complicated to assert
    # We simply check that the happens before the advanced subtensor
    toposort = FunctionGraph(outputs=[opt_out], clone=False).toposort()
    [idx_subtensor] = [
        i for i, node in enumerate(toposort) if isinstance(node.op, Subtensor)
    ]
    [idx_adv_subtensor] = [
        i for i, node in enumerate(toposort) if isinstance(node.op, AdvancedSubtensor)
    ]
    swapped = idx_subtensor < idx_adv_subtensor
    correct = swapped if supported else not swapped
    assert correct, debugprint(opt_out, print_type=True)
    np.testing.assert_allclose(
        opt_out.eval({x: x_test}, mode=NO_OPTIMIZATION_MODE),
        out.eval({x: x_test}, mode=NO_OPTIMIZATION_MODE),
    )


def test_local_subtensor_of_squeeze():
    def find_squeeze_and_index_ops(fg):
        squeeze_op = next(
            node
            for node in fg.toposort()
            if isinstance(node.op, DimShuffle) and node.op.is_squeeze
        )
        index_op = next(
            node for node in fg.toposort() if isinstance(node.op, Subtensor)
        )
        return squeeze_op, index_op

    rng = np.random.default_rng()

    x = pt.tensor("x", shape=(1, 5, 2, 1))
    z = x.squeeze(0)[0]
    fg = FunctionGraph(outputs=[z], clone=False)
    squeeze_op, index_op = find_squeeze_and_index_ops(fg)

    sorted_ops = list(fg.toposort())
    assert sorted_ops.index(squeeze_op) < sorted_ops.index(index_op)

    x_indexed = rewrite_graph(
        z,
        include=(
            "canonicalize",
            "local_subtensor_of_squeeze",
        ),
    )

    fg = FunctionGraph(outputs=[x_indexed], clone=False)
    squeeze_op, index_op = find_squeeze_and_index_ops(fg)
    sorted_ops = list(fg.toposort())
    assert sorted_ops.index(squeeze_op) > sorted_ops.index(index_op)

    fn = function([x], x_indexed)
    x_val = rng.normal(size=x.type.shape).astype(x.type.dtype)
    np.testing.assert_allclose(fn(x_val), x_val[0, 0])

    # Regression test for https://github.com/pymc-devs/pytensor/issues/1818
    x = pt.tensor("x", shape=(1, 1, 2, 1, 3))
    z = x.squeeze((0, 1, -2))[:, 0]
    fg = FunctionGraph(outputs=[z], clone=False)

    squeeze_op, index_op = find_squeeze_and_index_ops(fg)
    sorted_ops = list(fg.toposort())
    assert sorted_ops.index(squeeze_op) < sorted_ops.index(index_op)

    x_indexed = rewrite_graph(
        z,
        include=(
            "canonicalize",
            "local_subtensor_of_squeeze",
        ),
    )

    fg = FunctionGraph(outputs=[x_indexed], clone=False)
    squeeze_op, index_op = find_squeeze_and_index_ops(fg)
    sorted_ops = list(fg.toposort())
    assert sorted_ops.index(squeeze_op) > sorted_ops.index(index_op)

    fn = function([x], x_indexed)
    x_val = rng.normal(size=x.type.shape).astype(x.type.dtype)
    np.testing.assert_allclose(fn(x_val), x_val[0, 0, :, 0, 0])
