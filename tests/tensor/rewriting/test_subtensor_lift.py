import random

import numpy as np
import pytest

from pytensor import (
    Mode,
    config,
    function,
    shared,
)
from pytensor import scalar as ps
from pytensor import tensor as pt
from pytensor.assumptions import assume
from pytensor.compile import get_default_mode, get_mode
from pytensor.compile.ops import DeepCopyOp
from pytensor.graph import (
    Constant,
    FunctionGraph,
    Op,
    RewriteDatabaseQuery,
    rewrite_graph,
)
from pytensor.graph.basic import equal_computations
from pytensor.graph.rewriting.basic import check_stack_trace, out2in
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
    specify_shape,
    tensor,
    tensor3,
    vector,
)
from pytensor.tensor.basic import MakeVector, concatenate, expand_dims, make_vector
from pytensor.tensor.blas import Dot22, Gemv
from pytensor.tensor.blas.blas_c import CGemv
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import Dot
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.rewriting.assumptions import DrainSpecifyAssumptions
from pytensor.tensor.rewriting.subtensor import (
    local_adv_idx_to_diagonal,
)
from pytensor.tensor.rewriting.subtensor_lift import (
    _diag_indices,
    local_subtensor_make_vector,
    local_subtensor_of_batch_dims,
    local_subtensor_of_expand_dims,
    local_subtensor_shape_constant,
)
from pytensor.tensor.shape import Shape_i, SpecifyShape, _shape
from pytensor.tensor.special import softmax
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    Subtensor,
)
from tests.unittest_tools import RewriteTester, assert_equal_computations


mode_opt = config.mode
if mode_opt == "FAST_COMPILE":
    mode_opt = "FAST_RUN"
mode_opt = get_mode(mode_opt)


NO_OPTIMIZATION_MODE = Mode(linker="py", optimizer=None)


class TestLocalSubtensorOfBatchDims:
    rewrite_kw = dict(
        include=("ShapeOpt", "canonicalize", "specialize"),
        exclude=("local_replace_AdvancedSubtensor",),
        clone=True,
    )

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
                lambda x, y: pt.alloc(pt.add(x[None], y[None]), 0, *x.type.shape),
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
        assert_equal_computations([opt_out], [expected_opt_out], strict_dtype=False)
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

    def test_elemwise_adv_index_provably_smaller(self):
        """An adv index provably smaller than the non-broadcast Elemwise inputs lifts through Elemwise."""
        x = pt.matrix("x", shape=(10, 5))
        y = pt.matrix("y", shape=(10, 5))
        idx = pt.constant(np.array([0, 2, 5]), dtype="int64")
        out = pt.add(x, y)[idx]
        rewritten = rewrite_graph(out)
        expected = pt.add(x[idx], y[idx])
        assert_equal_computations([rewritten], [expected])

    def test_elemwise_adv_index_not_provably_smaller_bails(self):
        """An adv index whose size cannot be bounded does not lift through Elemwise."""
        x = pt.matrix("x", shape=(10, 5))
        y = pt.matrix("y", shape=(10, 5))
        idx = pt.tensor("idx", shape=(None,), dtype="int64")
        out = pt.add(x, y)[idx]
        rewritten = rewrite_graph(out)
        assert equal_computations([rewritten], [out])

    def test_elemwise_adv_index_assumed_unique_lifts(self):
        """An unbounded adv index asserted unique_indices can never enlarge, so it lifts."""
        x = pt.matrix("x")
        y = pt.matrix("y")
        idx = pt.lvector("idx")
        idx_unique = assume(idx, unique_indices=True)
        out = (x + y)[idx_unique]
        # Drain resolves the asserted fact onto idx, then canonicalize lifts the index.
        result = RewriteTester(
            [x, y, idx],
            [out],
            include="canonicalize",
            custom_rewrite=DrainSpecifyAssumptions(),
        )
        result.assert_graph(x[idx] + y[idx])

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

    @pytest.mark.parametrize(
        "idx", [np.zeros(20, dtype="int64"), slice(0, 5)], ids=["advanced", "slice"]
    )
    def test_stale_elemwise_output_type(self, idx):
        """When every input is length 1 on an indexed dim but elem's output type
        stale-claims non-bcast, the lift still succeeds: the broadcast-back shape
        is derived from the inputs (which are never stale), not from elem.type."""
        x_input = pt.tensor("x_input", shape=(None, 3, 3), dtype="float64")
        x_new_input = pt.tensor("x_new", shape=(1, 3, 3), dtype="float64")
        x = pt.identity(x_input)
        out = x * x
        indexed = out[idx]
        fgraph = FunctionGraph([x_input, x_new_input], [indexed], clone=False)

        # Forge a stale state: inputs are broadcastable on dim 0, but elem
        # output type is NOT. This happens naturally when upstream rewrites
        # call fgraph.replace(x, x_new_input).
        fgraph.replace(x, x_new_input)

        # Confirm the state is genuinely stale: the Elemwise output type still
        # claims dim 0 is non-broadcastable, while its (replaced) inputs are now
        # length 1 there.
        elem = indexed.owner.inputs[0]
        assert not elem.type.broadcastable[0]
        assert all(inp.type.broadcastable[0] for inp in elem.owner.inputs)

        [new_out] = local_subtensor_of_batch_dims.transform(fgraph, indexed.owner)
        # The lifted graph must be type-compatible with the (stale) original.
        fgraph.replace(indexed, new_out)

        rewritten = rewrite_graph(new_out, **self.rewrite_kw)
        if isinstance(idx, slice):
            # slice(0, 5) on a length-1 dim stays length 1, so no Alloc is needed.
            expected = pt.sqr(x_new_input)
        else:
            expected = pt.alloc(pt.sqr(x_new_input), idx.shape[0], 3, 3)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

    def test_advanced_index_on_broadcast_dim_does_not_expand_inputs(self):
        """An advanced index on a dim that is length 1 in every input must not be
        applied to those inputs (it would expand them 1->K and duplicate the
        computation). They stay length 1 and the K-sized dim comes from one Alloc."""
        x = pt.tensor("x", shape=(1, 3), dtype="float64")
        y = pt.tensor("y", shape=(1, 3), dtype="float64")
        out = (x + y)[np.zeros(5, dtype="int64")]

        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(x + y, 5, 3)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

    def test_broadcast_dim_does_not_block_lift(self):
        """An advanced index on a broadcast dim (which needs an Alloc back) must
        not stop the lift: the shrinking index on the other, non-broadcast dim
        still pushes the Elemwise inside. Here ``exp`` runs on a single row
        instead of a million."""
        x = pt.matrix("x", shape=(1_000_000, 1))
        out = pt.exp(x)[0, np.array([0, 0, 0])]

        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(pt.exp(x[0]), 3)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)


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
            lambda x: softmax(x, axis=0)[0, :2:2],
            lambda x: softmax(x[:, :2:2], axis=0)[0],
        ),
        (lambda x: softmax(x, axis=1)[0, :2:2], lambda x: softmax(x[0], axis=0)[:2:2]),
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


def test_local_subtensor_of_expand_dims_stale_output_type():
    """The re-expansion axes must be derived from the indices, not from the
    Subtensor's (possibly stale) output broadcastable pattern.

    When an upstream rewrite turns an indexed dim from non-broadcastable to
    length 1, the cached Subtensor output type still claims non-broadcastable.
    Comparing the freshly indexed result against that stale type would misplace
    the expand_dims and yield a mis-shaped (here: wrong-ndim) graph.
    """
    x = pt.tensor("x", shape=(None, None, None), dtype="float64")
    x_new = pt.tensor("x_new", shape=(None, 1, None), dtype="float64")
    out = expand_dims(x, 3)[7]

    fgraph = FunctionGraph([x], [out], clone=True)
    [cloned_out] = fgraph.outputs
    [cloned_x] = fgraph.inputs
    node = cloned_out.owner

    # Forge a stale state: dim 1 becomes length 1, but the Subtensor's cached
    # output type still claims dim 1 is non-broadcastable.
    fgraph.replace(cloned_x, x_new, import_missing=True)
    assert not cloned_out.type.broadcastable[1]
    assert node.inputs[0].owner.inputs[0].type.broadcastable[1]

    [new_out] = local_subtensor_of_expand_dims.transform(fgraph, node)
    # `new_out` (on x_new) must match the untouched `out` (on x) for the same data,
    # in particular keep the same ndim (the bug produced ndim 4).
    assert new_out.type.ndim == out.type.ndim
    x_test = np.random.default_rng(232).normal(size=(10, 1, 3))
    np.testing.assert_allclose(
        new_out.eval({x_new: x_test}, mode=NO_OPTIMIZATION_MODE),
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


class TestSubtensorOfAlloc:
    """Coverage for ``local_subtensor_of_alloc`` — basic and advanced indexing."""

    rewrite_kw = dict(
        include=("ShapeOpt", "canonicalize", "specialize"),
        exclude=("local_replace_AdvancedSubtensor",),
        clone=True,
    )

    def test_basic_subtensor(self):
        for s in [(3, 5), (4, 6), (3, 8), (4, 7), (1, 5), (5, 1)]:
            x = tensor(
                dtype=config.floatX,
                shape=(1 if s[0] == 1 else None, 1 if s[1] == 1 else None),
            )

            xval = np.zeros(s, dtype=config.floatX)
            yval = np.arange(s[1], dtype=config.floatX)

            for y in [shared(yval), pt.constant([1.0])]:
                yx = pt.alloc(y, x.shape[0], x.shape[1])

                slicess: list = []
                if s[0] != 1:
                    slicess.append((2, slice(None)))
                if s[1] != 1:
                    slicess.append((slice(None), 3))
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
                        assert not isinstance(
                            f.maker.fgraph.toposort()[-1].op, Subtensor
                        )
                    val = f(xval)
                    assert xval.__getitem__(slices).shape == val.shape

    @pytest.mark.parametrize("idx_ndim", (0, 1, 2))
    def test_adv_int_indexing(self, idx_ndim):
        v = pt.vector("v", shape=(7,))
        # ``n`` is uint so the slice rewrite can prove non-negativity for the
        # ``arange(n) -> slice(0, n)`` round-trip; ``uint32`` (not ``uint64``)
        # because ``ShapeFeature`` rejects ``uint64`` shape dims.
        n = pt.scalar("n", dtype="uint32")
        idx = pt.tensor("idx", shape=(None,) * idx_ndim, dtype="int64")

        # On broadcasted axis idx is a no-op, only contributes shape
        out = pt.alloc(v, 5, 5, 7)[idx]
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        match idx_ndim:
            case 0:
                expected = pt.alloc(v, 5, 7)
            case 1:
                expected = pt.alloc(v, Shape_i(0)(idx), 5, 7)
            case 2:
                expected = pt.alloc(v, Shape_i(0)(idx), Shape_i(1)(idx), 5, 7)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

        # On real axis it could cause more work
        out = pt.alloc(v, 5, 5, 7)[..., idx]
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        match idx_ndim:
            case 0:
                expected = pt.alloc(v[idx], 5, 5)
            case 1 | 2:
                expected = out
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

        if idx_ndim == 0:
            return

        # But if we know it's less work, we're back on game
        core_shape = (2,) * idx_ndim
        out = pt.alloc(v, 5, 5, 7)[..., pt.specify_shape(idx, core_shape)]
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(v[pt.specify_shape(idx, core_shape)], 5, 5, *core_shape)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

        cast_n = n.astype("int64")
        match idx_ndim:
            case 1:
                arange_idx = pt.arange(n)
                # The lift's inner slice uses ``cast_n`` directly (slice clips
                # at runtime), but the outer Alloc dim uses
                # ``minimum(cast_n, dim_length)`` since arange-to-slice clamps
                # against ``v.shape[0]`` to keep the gather in bounds.
                expected_v_indexed = v[:cast_n]
                index_shape = (pt.minimum(cast_n, v.type.shape[0]),)
            case 2:
                arange_idx = pt.arange(n).reshape((2, -1))
                # The reshape-of-arange index isn't recognized by
                # ``arange-to-slice``, so the lifted gather keeps its
                # ``AdvancedSubtensor`` form. The outer Alloc shape uses
                # ``cast_n // 2`` because ``arange`` casts the uint stop.
                expected_v_indexed = v[arange_idx]
                index_shape = (2, cast_n // 2)
        out = pt.alloc(v, 5, 5, 7)[..., arange_idx]
        expected = pt.alloc(expected_v_indexed, 5, 5, *index_shape)
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

    def test_multiple_adv_int_indexing(self):
        # Index on kept dims
        val = pt.matrix("val", shape=(5, 7))
        idx_a = pt.constant(np.array([0, 2], dtype=np.int64))
        idx_b = pt.constant(np.array([1, 1], dtype=np.int64))

        out = pt.alloc(val, 3, 4, 5, 7)[idx_a, idx_b, :, :]
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(val, 2, 5, 7)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

        out = pt.alloc(val, 3, 4, 5, 7)[:, idx_a, idx_b, :]
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(val[idx_b], 3, 2, 7)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

        out = pt.alloc(val, 3, 4, 5, 7)[:, :, idx_a, idx_b]
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(val[idx_a, idx_b], 3, 4, 2)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

        # Non-consecutive advanced indices make everything harder, we don't try
        out = pt.alloc(val, 3, 4, 5, 7)[:, idx_a, :, idx_b]
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = out
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

    def test_index_on_broadcast_val_dim(self):
        v = pt.tensor("v", shape=(7, 1))

        out = pt.alloc(v, 7, 5)[:3, :2]
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(v[:3], 3, 2)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

        out = pt.alloc(v, 7, 5)[:3, 2]
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = v[:3].squeeze(axis=1)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

        idx_a = pt.constant(np.array([0, 2], dtype=np.int64))
        idx_b = pt.constant(np.array([0, 3], dtype=np.int64))

        out = pt.alloc(v, 7, 5)[idx_a, idx_b]
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        rng = np.random.default_rng(0)
        v_test = rng.normal(size=(7, 1))
        np.testing.assert_allclose(
            rewritten.eval({v: v_test}, mode=NO_OPTIMIZATION_MODE),
            np.broadcast_to(v_test, (7, 5))[[0, 2], [0, 3]],
        )

        out = pt.alloc(v, 7, 5)[:3, idx_b]
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(v[:3], 3, 2)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

    def test_boolean_idx_bails(self):
        """A boolean mask on a non-broadcast val dim does not lift through Alloc."""
        val = pt.vector("val", shape=(7,))
        mask = pt.tensor("mask", shape=(None,), dtype="bool")

        out = pt.alloc(val, 5, 7)[:, mask]
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        assert_equal_computations([rewritten], [out], strict_dtype=False)

    def test_const_idx_with_duplicates_bails(self):
        """A constant integer index larger than the val dim does not lift through Alloc."""
        val = pt.vector("val", shape=(3,))
        idx = pt.constant(np.array([0, 0, 0, 0, 0], dtype=np.int64))

        out = pt.alloc(val, 5, 3)[:, idx]
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        assert_equal_computations([rewritten], [out], strict_dtype=False)

    def test_negative_step_idx_to_slice(self):
        """Negative-step constant arange ``[7, 5, 3, 1]`` rewrites to ``x[7::-2]``."""
        x = pt.vector("x", shape=(10,))
        rng = np.random.default_rng(0)
        x_test = rng.normal(size=10)

        out = x[pt.constant(np.array([7, 5, 3, 1], dtype=np.int64))]
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        np.testing.assert_allclose(
            rewritten.eval({x: x_test}, mode=NO_OPTIMIZATION_MODE),
            x_test[7::-2],
        )

    def test_mixed_slice_and_adv_index(self):
        """A mix of slice and adv index across alloc dims lifts each to its own dim."""
        val = pt.matrix("val", shape=(4, 6))
        idx = pt.constant(np.array([0, 1, 3], dtype=np.int64))

        out = pt.alloc(val, 3, 4, 6)[:2, idx]
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(val[idx], 2, 3, 6)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)


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
                (slice(iscalar(), iscalar(), iscalar()),),
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
        assert_equal_computations(
            [opt_fgraph.outputs[0].owner.inputs[0]], [expand_dims(x, 0)]
        )
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

    # Any static dim folds, not just broadcastable ones
    x = tensor(dtype=np.float64, shape=(7, None)).shape[0]
    (res,) = local_subtensor_shape_constant.transform(None, x.owner)
    assert isinstance(res, Constant)
    assert res.data == 7

    x = _shape(tensor(dtype=np.float64, shape=(None, 3, 7)))[1:]
    (res,) = local_subtensor_shape_constant.transform(None, x.owner)
    assert isinstance(res, Constant)
    assert np.array_equal(res.data, [3, 7])


@pytest.mark.parametrize(
    "original_fn, supported",
    [
        # Use non-constant-step index permutations ([2, 0, 1] not [1, 0]) so
        # ``local_adv_idx_to_slice`` doesn't rewrite the
        # AdvancedSubtensor to a Subtensor before this rewrite gets a chance
        # to fire.
        (lambda x: x[:, [2, 0, 1]][0], True),
        (lambda x: x[:, [2, 0, 1], [0, 0, 0]][1:], True),
        (lambda x: x[:, [[2, 0], [0, 1]]][1:], True),
        (lambda x: x[:, None, [2, 0, 1]][0], True),
        # Not supported, basic indexing on advanced indexing dim
        (lambda x: x[[0, 1]][0], False),
        # Not supported, basic indexing on the right of advanced indexing
        (lambda x: x[[0, 1]][:, 0], False),
        # Not implemented, complex flavors of advanced indexing
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
        out,
        include=("canonicalize", "local_subtensor_of_adv_subtensor"),
        exclude=(
            "local_advanced_read_of_write_constant_indices",
            "local_adv_idx_to_slice",
        ),
    )
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


@pytest.mark.parametrize(
    "original_fn, expected_fn, x_shape",
    [
        (
            lambda x: x.squeeze(0)[0],
            lambda x: x[:, 0].squeeze(0),
            (1, 5, 2, 1),
        ),
        # Regression test for https://github.com/pymc-devs/pytensor/issues/1818
        # Squeeze multiple axes then index
        (
            lambda x: x.squeeze((0, 1, -2))[:, 0],
            lambda x: x[:, :, :, :, 0].squeeze((0, 1, 3)),
            (1, 1, 2, 1, 3),
        ),
    ],
)
def test_local_subtensor_of_squeeze(original_fn, expected_fn, x_shape):
    rng = np.random.default_rng()
    x = pt.tensor("x", shape=x_shape)
    x_test = rng.normal(size=x.type.shape).astype(x.dtype)

    out = original_fn(x)
    expected_opt_out = expected_fn(x)
    opt_out = rewrite_graph(out)
    assert_equal_computations([opt_out], [expected_opt_out])
    np.testing.assert_allclose(
        opt_out.eval({x: x_test}, mode=NO_OPTIMIZATION_MODE),
        out.eval({x: x_test}, mode=NO_OPTIMIZATION_MODE),
    )


class TestExtractDiagLiftPass:
    """Coverage for ``extract_diag_lift_pass`` and its constituent rewrites:
    ``local_extract_diag_of_alloc_diag``, ``local_extract_diag_of_eye``,
    ``local_extract_diag_lift``, and ``local_slice_read_of_write``.
    """

    rewrite_kw = dict(include=("ShapeOpt", "canonicalize", "specialize"))

    @pytest.mark.parametrize(
        "v_len, k_alloc, offset",
        [(5, 0, 0), (4, 1, 1)],
    )
    def test_extract_diag_of_alloc_diag_match(self, v_len, k_alloc, offset):
        v = pt.vector("v", shape=(v_len,))
        out = pt.diagonal(pt.diag(v, k=k_alloc), offset=offset)
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        assert_equal_computations([rewritten], [v])

    def test_extract_diag_of_alloc_diag_offset_mismatch(self):
        """When offsets differ, the diagonals don't overlap, so the result is zeros."""
        v = pt.vector("v", shape=(4,))
        out = pt.diagonal(pt.diag(v, k=1), offset=0)
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(np.asarray(0.0, dtype=out.dtype), np.int64(5))
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

    @pytest.mark.parametrize(
        "n, m, k_eye, diag_offset, expected_val, expected_len",
        [
            (5, 5, 0, 0, 1, 5),
            (3, 5, 0, 0, 1, 3),
            (5, 5, 0, 1, 0, 4),
        ],
    )
    def test_extract_diag_of_eye_static(
        self, n, m, k_eye, diag_offset, expected_val, expected_len
    ):
        out = pt.diagonal(pt.eye(n, m, k_eye), offset=diag_offset)
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.as_tensor_variable(
            np.full(expected_len, expected_val, dtype=out.dtype)
        )
        assert_equal_computations([rewritten], [expected])

    def test_extract_diag_of_eye_symbolic(self):
        n = pt.iscalar("n")
        out = pt.diagonal(pt.eye(n, n, 0))
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(np.asarray(1.0, dtype=out.dtype), n)
        assert_equal_computations([rewritten], [expected])

    def test_extract_diag_of_eye_partial_static_n(self):
        """Diagonal of ``eye(n, 5, 0)`` with symbolic ``n`` is ``alloc(1, min(n, 5))``."""
        n = pt.iscalar("n")
        out = pt.diagonal(pt.eye(n, 5, 0))
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(
            np.asarray(1.0, dtype=out.dtype), pt.minimum(n, np.int64(5))
        )
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

    def test_extract_diag_of_eye_partial_static_m(self):
        """Diagonal of ``eye(5, m, 0)`` with symbolic ``m`` is ``alloc(1, min(5, m))``."""
        m = pt.iscalar("m")
        out = pt.diagonal(pt.eye(5, m, 0))
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(
            np.asarray(1.0, dtype=out.dtype), pt.minimum(np.int64(5), m)
        )
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

    def test_extract_diag_of_alloc_zeros(self):
        n = pt.lscalar("n")
        out = pt.diagonal(pt.alloc(np.float64(0.0), n, n))
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(np.float64(0.0), n)
        assert_equal_computations([rewritten], [expected])

    def test_extract_diag_of_alloc_ones(self):
        n = pt.lscalar("n")
        m = pt.lscalar("m")
        out = pt.diagonal(pt.alloc(np.float64(1.0), n, m))
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(np.float64(1.0), pt.minimum(n, m))
        assert_equal_computations([rewritten], [expected])

    def test_extract_diag_of_alloc_symbolic_scalar(self):
        v = pt.scalar("v")
        out = pt.diagonal(pt.alloc(v, 5, 5))
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(v, 5)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

    def test_extract_diag_of_alloc_row_broadcast(self):
        """Diagonal of a row-broadcast alloc reduces to a slice of the row."""
        v = pt.vector("v", shape=(None,))
        out = pt.diagonal(pt.alloc(v, 5, v.shape[0]))
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = v[:5]
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

    def test_extract_diag_of_alloc_matrix(self):
        """Diagonal lifts through Alloc onto the underlying matrix input."""
        x = pt.matrix("x", shape=(None, None))
        out = pt.diagonal(pt.alloc(x, 3, x.shape[0], x.shape[1]), axis1=-2, axis2=-1)
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        d = pt.minimum(Shape_i(0)(x), Shape_i(1)(x))
        expected = pt.alloc(pt.diagonal(x), 3, d)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

    def test_extract_diag_of_eye_mul_matrix(self):
        x = pt.matrix("x", shape=(5, 5))
        out = pt.diagonal(pt.eye(5) * x)
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.diagonal(x)
        assert_equal_computations([rewritten], [expected])

    def test_extract_diag_of_eye_mul_scalar(self):
        """Diagonal of ``eye(5) * s`` reduces to a length-5 broadcast of ``s``."""
        s = pt.scalar("s")
        out = pt.diagonal(pt.eye(5) * s)
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(s[None], 5)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

    def test_extract_diag_of_eye_mul_row(self):
        v = pt.row("v", shape=(1, 5))
        out = pt.diagonal(pt.eye(5) * v)
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = v.squeeze(axis=0)
        assert_equal_computations([rewritten], [expected])

    def test_extract_diag_of_eye_mul_col(self):
        v = pt.col("v", shape=(5, 1))
        out = pt.diagonal(pt.eye(5) * v)
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = v.squeeze(axis=1)
        assert_equal_computations([rewritten], [expected])

    def test_extract_diag_of_eye_mul_nonzero_offset(self):
        """Off-diagonal of ``eye(5) * x`` is zero everywhere."""
        x = pt.matrix("x", shape=(5, 5))
        out = pt.diagonal(pt.eye(5) * x, offset=1)
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.alloc(np.asarray(0.0, dtype=out.dtype), 4)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

    def test_extract_diag_of_elemwise_unary(self):
        x = pt.matrix("x", shape=(5, 4))
        out = pt.diagonal(pt.exp(x))
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.exp(pt.diagonal(x))
        assert_equal_computations([rewritten], [expected])

    def test_extract_diag_of_elemwise_binary(self):
        x = pt.matrix("x", shape=(5, 5))
        y = pt.matrix("y", shape=(5, 5))
        out = pt.diagonal(x + y)
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.diagonal(x) + pt.diagonal(y)
        assert_equal_computations([rewritten], [expected])

    @pytest.mark.parametrize(
        "bcast_shape, squeeze_axis",
        [
            ((), None),
            ((1, 5), 0),
            ((5, 1), 1),
        ],
        ids=["scalar", "row", "col"],
    )
    def test_extract_diag_of_elemwise_broadcast(self, bcast_shape, squeeze_axis):
        x = pt.matrix("x", shape=(5, 5))
        b = pt.tensor("b", shape=bcast_shape)
        out = pt.diagonal(x + b) if bcast_shape else pt.diagonal(x * b)
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        b_diag = b.squeeze(axis=squeeze_axis) if squeeze_axis is not None else b
        expected = (pt.diagonal(x) + b_diag) if bcast_shape else (pt.diagonal(x) * b)
        assert_equal_computations([rewritten], [expected])

    def test_extract_diag_of_elemwise_row_broadcast_offset(self):
        """Row-broadcast input with positive offset contributes ``r[:, offset:]``."""
        x = pt.matrix("x", shape=(5, 5))
        r = pt.row("r", shape=(1, 5))
        out = pt.diagonal(x + r, offset=1)
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        expected = pt.diagonal(x, offset=1) + r[:, 1:].squeeze(axis=0)
        assert_equal_computations([rewritten], [expected], strict_dtype=False)

    def test_extract_diag_of_elemwise_unary_full_broadcast(self):
        """Diagonal of a unary Elemwise on a fully-broadcast (1, 1) matrix is length-1."""
        s = pt.scalar("s")
        m = s.dimshuffle("x", "x")
        out = pt.diagonal(-m)
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        rng = np.random.default_rng(0)
        s_test = float(rng.normal())
        np.testing.assert_allclose(rewritten.eval({s: s_test}), [-s_test])

    def test_extract_diag_of_elemwise_binary_with_full_broadcast(self):
        """Diagonal of a binary Elemwise mixing a matrix and a (1, 1) input."""
        x = pt.matrix("x", shape=(4, 4))
        s = pt.scalar("s")
        m = s.dimshuffle("x", "x")
        out = pt.diagonal(x + m)
        rewritten = rewrite_graph(out, **self.rewrite_kw)
        rng = np.random.default_rng(0)
        x_test = rng.normal(size=(4, 4))
        s_test = float(rng.normal())
        np.testing.assert_allclose(
            rewritten.eval({x: x_test, s: s_test}),
            np.diagonal(x_test + s_test),
        )

    @pytest.mark.parametrize("offset", [0, 2, -2])
    def test_diag_indices_roundtrip(self, offset):
        """``diagonal(x, k)`` -> ``_diag_indices`` -> rewrite back -> ``diagonal(x, k)``."""
        x = pt.matrix("x", shape=(5, 5))
        diag_form = pt.diagonal(x, offset=offset)

        row_off, col_off = max(0, -offset), max(0, offset)
        d = min(5 - row_off, 5 - col_off)
        idxs = _diag_indices(2, 0, 1, d, row_off, col_off)
        arange_form = x[tuple(idxs)]

        ar = pt.arange(d, dtype="int64")
        rows = ar + row_off if row_off else ar
        cols = ar + col_off if col_off else ar
        expected_lowered = x[rows, cols]
        assert_equal_computations([arange_form], [expected_lowered])

        folded = rewrite_graph(
            arange_form,
            include=(),
            custom_rewrite=out2in(local_adv_idx_to_diagonal),
        )
        assert_equal_computations([folded], [diag_form])

    @pytest.mark.skipif(
        config.mode == "FAST_COMPILE", reason="Test requires specialization rewrites"
    )
    def test_extract_diag_of_write(self):
        """Diagonal reads collapse fully-covered diagonal writes and keep partial-coverage writes."""
        A = pt.full((2, 6, 6), np.nan)
        rows = pt.arange(A.shape[-2])
        cols = pt.arange(A.shape[-1])
        write_offsets = [-2, -1, 0, 1, 2]
        random.shuffle(write_offsets)
        for offset in write_offsets:
            value = offset + 0.1 * offset
            if offset == 0:
                A = A[..., rows, cols].set(value)
            elif offset > 0:
                A = A[..., rows[:-offset], cols[offset:]].set(value)
            else:
                offset = -offset
                A = A[..., rows[offset:], cols[:-offset]].set(value)
        # Partial write along offset 3
        A = A[..., rows[1:-3], cols[4:]].set(np.pi)

        read_offsets = [-2, -1, 0, 1, 2, 3]
        outs = [
            A.diagonal(offset=offset, axis1=-2, axis2=-1) for offset in read_offsets
        ]

        f_on = function([], outs)
        f_off = function(
            [],
            outs,
            mode=get_default_mode().excluding("extract_diag_lift_pass"),
        )

        # The partial-coverage read at offset=3 keeps exactly one scatter write of pi.
        on_topo = f_on.maker.fgraph.toposort()
        n_writes = sum(
            isinstance(n.op, AdvancedIncSubtensor | AdvancedIncSubtensor1)
            for n in on_topo
        )
        assert n_writes == 1

        for got, ref in zip(f_on(), f_off(), strict=True):
            np.testing.assert_allclose(got, ref, equal_nan=True)
