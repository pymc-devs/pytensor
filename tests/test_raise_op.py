import numpy as np
import pytest
import scipy.sparse

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import OPT_FAST_RUN, Mode
from pytensor.graph import vectorize_graph
from pytensor.graph.basic import Constant, equal_computations
from pytensor.raise_op import Assert, CheckAndRaise, assert_op
from pytensor.scalar.basic import ScalarType, float64
from pytensor.sparse import as_sparse_variable
from pytensor.tensor.basic import second
from pytensor.tensor.elemwise import DimShuffle
from tests import unittest_tools as utt


class CustomException(ValueError):
    """A custom user-created exception to throw."""


def test_CheckAndRaise_str():
    exc_msg = "this is the exception"
    check_and_raise = CheckAndRaise(CustomException, exc_msg)
    assert (
        str(check_and_raise)
        == f"CheckAndRaise{{raises=CustomException, msg='{exc_msg}'}}"
    )


def test_CheckAndRaise_pickle():
    import pickle

    exc_msg = "this is the exception"
    check_and_raise = CheckAndRaise(CustomException, exc_msg)

    y = check_and_raise(pt.as_tensor(1), pt.as_tensor(0))
    y_str = pickle.dumps(y)
    new_y = pickle.loads(y_str)

    assert y.owner.op == new_y.owner.op
    assert y.owner.op.msg == new_y.owner.op.msg
    assert y.owner.op.exc_type == new_y.owner.op.exc_type


def test_CheckAndRaise_equal():
    x, y = pt.vectors("xy")
    g1 = assert_op(x, (x > y).all())
    g2 = assert_op(x, (x > y).all())

    assert equal_computations([g1], [g2])


def test_CheckAndRaise_validation():
    with pytest.raises(ValueError):
        CheckAndRaise(str)

    g1 = assert_op(np.array(1.0))
    assert isinstance(g1.owner.inputs[0], Constant)


@pytest.mark.parametrize(
    "linker",
    [
        pytest.param(
            "cvm",
            marks=pytest.mark.skipif(
                not pytensor.config.cxx,
                reason="G++ not available, so we need to skip this test.",
            ),
        ),
        "py",
    ],
)
def test_CheckAndRaise_basic_c(linker):
    exc_msg = "this is the exception"
    check_and_raise = CheckAndRaise(CustomException, exc_msg)

    conds = pt.scalar()
    y = check_and_raise(pt.as_tensor(1), conds)
    y_fn = pytensor.function([conds], y, mode=Mode(linker))

    with pytest.raises(CustomException, match=exc_msg):
        y_fn(0)

    x = pt.vector()
    y = check_and_raise(x, conds)
    y_fn = pytensor.function([conds, x], y.shape, mode=Mode(linker, OPT_FAST_RUN))

    x_val = np.array([1.0], dtype=pytensor.config.floatX)
    assert np.array_equal(y_fn(0, x_val), x_val)

    y = check_and_raise(x, pt.as_tensor(0))
    y_grad = pytensor.grad(y.sum(), [x])
    y_fn = pytensor.function([x], y_grad, mode=Mode(linker, OPT_FAST_RUN))

    assert np.array_equal(y_fn(x_val), [x_val])


@pytest.mark.parametrize(
    "linker",
    [
        pytest.param(
            "cvm",
            marks=pytest.mark.skipif(
                not pytensor.config.cxx,
                reason="G++ not available, so we need to skip this test.",
            ),
        ),
        "py",
    ],
)
def test_perform_CheckAndRaise_scalar(linker):
    exc_msg = "this is the exception"
    check_and_raise = CheckAndRaise(CustomException, exc_msg)

    val = float64("val")
    conds = (val > 0, val > 3)
    y = check_and_raise(val, *conds)

    assert all(isinstance(i.type, ScalarType) for i in y.owner.inputs)
    assert isinstance(y.type, ScalarType)

    mode = Mode(linker=linker)
    y_fn = pytensor.function([val], y, mode=mode)

    with pytest.raises(CustomException, match=exc_msg):
        y_fn(0.0)

    assert y_fn(4.0) == 4.0

    if linker == "cvm":
        assert isinstance(
            y_fn.maker.fgraph.outputs[0].owner.inputs[0].owner.op, CheckAndRaise
        )
        assert hasattr(y_fn.vm.thunks[-2], "cthunk")

    (y_grad,) = pytensor.grad(y, [val])
    y_fn = pytensor.function([val], y_grad, mode=Mode(linker, OPT_FAST_RUN))

    assert np.array_equal(y_fn(4.0), 1.0)


class TestCheckAndRaiseInferShape(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()

    def test_infer_shape(self):
        adscal = pt.dscalar()
        bdscal = pt.dscalar()
        adscal_val = np.random.random()
        bdscal_val = np.random.random() + 1
        out = assert_op(adscal, bdscal)
        self._compile_and_check(
            [adscal, bdscal], [out], [adscal_val, bdscal_val], Assert
        )

        admat = pt.dmatrix()
        admat_val = np.random.random((3, 4))
        adscal_val += 1
        out = assert_op(admat, adscal, bdscal)
        self._compile_and_check(
            [admat, adscal, bdscal], [out], [admat_val, adscal_val, bdscal_val], Assert
        )

    def test_infer_shape_scalar(self):
        adscal = float64("adscal")
        bdscal = float64("bdscal")
        adscal_val = np.random.random()
        bdscal_val = np.random.random() + 1
        out = assert_op(adscal, bdscal)
        self._compile_and_check(
            [adscal, bdscal], [out], [adscal_val, bdscal_val], Assert
        )


def test_CheckAndRaise_sparse_variable():
    check_and_raise = CheckAndRaise(ValueError, "sparse_check")

    spe1 = scipy.sparse.csc_matrix(scipy.sparse.eye(5, 3))
    aspe1 = as_sparse_variable(spe1)
    a1 = check_and_raise(aspe1, aspe1.sum() > 2)
    assert a1.sum().eval() == 3

    spe2 = scipy.sparse.csc_matrix(scipy.sparse.eye(5, 1))
    aspe2 = as_sparse_variable(spe2)
    a2 = check_and_raise(aspe1, aspe2.sum() > 2)
    with pytest.raises(ValueError, match="sparse_check"):
        a2.sum().eval()


@pytensor.config.change_flags(cxx="")  # For speed-up
def test_vectorize():
    floatX = pytensor.config.floatX
    x = pt.vector("x")
    y = pt.vector("y")
    cond = pt.all(y >= 0)
    out = assert_op(x, cond)

    batch_x = pt.matrix("batch_x", shape=(2, None))
    batch_y = pt.matrix("batch_y", shape=(2, None))

    test_x = np.arange(3).astype(floatX)
    test_y = np.arange(4).astype(floatX)
    test_batch_x = np.arange(6).reshape(2, 3).astype(floatX)
    test_batch_y = np.arange(8).reshape(2, 4).astype(floatX)

    # Only x is batched
    vect_out = vectorize_graph(out, {x: batch_x, y: y})
    assert vect_out.type.shape == (2, None)
    assert isinstance(vect_out.owner.op, CheckAndRaise)
    np.testing.assert_array_equal(
        vect_out.eval({batch_x: test_batch_x, y: test_y}),
        test_batch_x,
    )
    with pytest.raises(AssertionError):
        vect_out.eval({batch_x: test_batch_x, y: -test_y})

    # Only y is batched
    vect_out = vectorize_graph(out, {x: x, y: batch_y})
    assert vect_out.type.shape == (2, None)
    assert vect_out.owner.op == second  # broadcast
    assert isinstance(vect_out.owner.inputs[1].owner.op, DimShuffle)
    assert isinstance(vect_out.owner.inputs[1].owner.inputs[0].owner.op, CheckAndRaise)
    np.testing.assert_array_equal(
        vect_out.eval({x: test_x, batch_y: test_batch_y}),
        np.broadcast_to(test_x, (2, *test_x.shape)),
    )
    with pytest.raises(AssertionError):
        vect_out.eval({x: test_x, batch_y: -test_batch_y})

    # Both x, and y are batched
    vect_out = vectorize_graph(out, {x: batch_x, y: batch_y})
    assert vect_out.type.shape == (2, None)
    assert vect_out.owner.op == second
    assert isinstance(vect_out.owner.inputs[1].owner.op, CheckAndRaise)
    np.testing.assert_array_equal(
        vect_out.eval({batch_x: test_batch_x, batch_y: test_batch_y}),
        test_batch_x,
    )
    with pytest.raises(AssertionError):
        vect_out.eval({batch_x: test_batch_x, batch_y: -test_batch_y})

    # Both x, and y are batched and broadcast each other
    vect_out = vectorize_graph(out, {x: batch_x[:, None, :], y: batch_y[None, :, :]})
    assert vect_out.type.shape == (2, 2, None)
    assert vect_out.owner.op == second
    assert isinstance(vect_out.owner.inputs[1].owner.op, CheckAndRaise)
    np.testing.assert_array_equal(
        vect_out.eval({batch_x: test_batch_x, batch_y: test_batch_y}),
        np.broadcast_to(test_batch_x[:, None, :], (2, *test_batch_x.shape)),
    )
    with pytest.raises(AssertionError):
        vect_out.eval({batch_x: test_batch_x, batch_y: -test_batch_y})
