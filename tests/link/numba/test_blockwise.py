import numpy as np
import pytest

from pytensor import function
from pytensor.compile.mode import Mode
from pytensor.graph import Apply
from pytensor.scalar import ScalarOp
from pytensor.tensor import TensorVariable, lvector, tensor, tensor3, vector
from pytensor.tensor.basic import Alloc, ARange, constant
from pytensor.tensor.blockwise import Blockwise, BlockwiseWithCoreShape
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.decomposition.svd import SVD
from pytensor.tensor.linalg.summary import Det
from pytensor.tensor.signal import convolve1d
from tests.link.numba.test_basic import compare_numba_and_py, numba_mode


# Fails if object mode warning is issued when not expected
pytestmark = pytest.mark.filterwarnings("error")


@pytest.mark.parametrize("shape_opt", [True, False], ids=str)
@pytest.mark.parametrize("core_op", [Det(), Cholesky(), SVD(compute_uv=True)], ids=str)
def test_blockwise(core_op, shape_opt):
    x = tensor(shape=(5, None, None))
    outs = Blockwise(core_op=core_op)(x, return_list=True)

    mode = (
        numba_mode.including("ShapeOpt")
        if shape_opt
        else numba_mode.excluding("ShapeOpt")
    )
    x_test = np.eye(3) * np.arange(1, 6)[:, None, None]
    compare_numba_and_py(
        [x],
        outs,
        [x_test],
        numba_mode=mode,
        eval_obj_mode=False,
    )


def test_non_square_blockwise():
    """Test that Op that cannot always be blockwised at runtime fails gracefully."""
    x = tensor(shape=(3,), dtype="int64")
    out = Blockwise(core_op=ARange(dtype="int64"), signature="(),(),()->(a)")(0, x, 1)

    with pytest.warns(UserWarning, match="Numba will use object mode"):
        fn = function([x], out, mode="NUMBA")

    np.testing.assert_allclose(fn([5, 5, 5]), np.broadcast_to(np.arange(5), (3, 5)))

    with pytest.raises(ValueError):
        fn([3, 4, 5])


def test_repeated_args():
    x = tensor3("x")
    x_test = np.full((1, 1, 1), 2.0, dtype=x.type.dtype)
    out = x @ x
    fn, _ = compare_numba_and_py([x], [out], [x_test], eval_obj_mode=False)

    # Confirm we are testing a Blockwise with repeated inputs
    final_node = fn.maker.fgraph.outputs[0].owner
    assert isinstance(final_node.op, BlockwiseWithCoreShape)
    assert final_node.inputs[0] is final_node.inputs[1]


def test_blockwise_alloc():
    val = lvector("val")
    out = Blockwise(Alloc(), signature="(),(),()->(2,3)")(
        val, constant(2, dtype="int64"), constant(3, dtype="int64")
    )
    assert out.type.ndim == 3

    compare_numba_and_py([val], [out], [np.arange(5)], eval_obj_mode=False)


def test_blockwise_scalar_dimshuffle():
    x = lvector("x")
    blockwise_scalar_ds = Blockwise(
        DimShuffle(input_ndim=0, new_order=["x", "x"]), signature="()->(1,1)"
    )
    out = blockwise_scalar_ds(x)
    compare_numba_and_py([x], [out], [np.arange(9)], eval_obj_mode=False)


@pytest.mark.parametrize("signal_layout", ["C", "F", "strided"])
@pytest.mark.parametrize("kernel_layout", ["C", "F", "strided"])
def test_blockwise_non_c_contiguous_inputs(signal_layout, kernel_layout):
    # Regression test for https://github.com/pymc-devs/pytensor/issues/2228
    # When a batched input is non-C-contiguous (F-order, or strided batch/core dims)
    # its per-batch core slice is strided. The core type must not claim a contiguous
    # layout, or numba reads contiguous memory off a strided buffer (wrong values).
    signal = tensor("signal", shape=(3, 16))
    kernel = tensor("kernel", shape=(3, 5))
    out = convolve1d(signal, kernel, mode="valid")
    assert isinstance(out.owner.op, Blockwise)

    rng = np.random.default_rng(2228)

    def array_with_layout(core, layout):
        """A (3, core) test array in the requested memory layout."""
        if layout == "C":
            return rng.normal(size=(3, core))
        if layout == "F":
            # Transpose of a C array -> the per-batch core slice is strided.
            return np.asfortranarray(rng.normal(size=(3, core)))
        # "strided": [::2, ::2] view -> both the batch and core dims are strided.
        return rng.normal(size=(6, core * 2))[::2, ::2]

    signal_test = array_with_layout(16, signal_layout)
    kernel_test = array_with_layout(5, kernel_layout)

    fn = function([signal, kernel], out, mode="NUMBA")
    ref_fn = function([signal, kernel], out, mode=Mode(linker="py", optimizer=None))
    np.testing.assert_allclose(
        fn(signal_test, kernel_test), ref_fn(signal_test, kernel_test)
    )


def test_blockwise_vs_elemwise_scalar_op():
    # Regression test for https://github.com/pymc-devs/pytensor/issues/1760

    class TestScalarOp(ScalarOp):
        def make_node(self, x):
            return Apply(self, [x], [x.type()])

        def perform(self, node, inputs, outputs):
            [x] = inputs
            if isinstance(node.inputs[0], TensorVariable):
                assert isinstance(x, np.ndarray)
            else:
                assert isinstance(x, np.number | float)
            out = x + 1
            if isinstance(node.outputs[0], TensorVariable):
                out = np.asarray(out)
            outputs[0][0] = out

    x = vector("x")
    y = Elemwise(TestScalarOp())(x)
    with pytest.warns(
        UserWarning,
        match="Numba will use object mode to run TestScalarOp's perform method",
    ):
        fn = function([x], y, mode="NUMBA")
    np.testing.assert_allclose(fn(np.zeros((3,))), [1, 1, 1])

    z = Blockwise(TestScalarOp(), signature="()->()")(x)
    with pytest.warns(
        UserWarning,
        match="Numba will use object mode to run TestScalarOp's perform method",
    ):
        fn = function([x], z, mode="NUMBA")
    np.testing.assert_allclose(fn(np.zeros((3,))), [1, 1, 1])
