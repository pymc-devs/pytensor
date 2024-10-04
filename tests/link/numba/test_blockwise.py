import numpy as np
import pytest

from pytensor import function
from pytensor.tensor import tensor
from pytensor.tensor.basic import ARange
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.nlinalg import SVD, Det
from pytensor.tensor.slinalg import Cholesky, cholesky
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
        ([x], outs),
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


def test_blockwise_benchmark(benchmark):
    x = tensor(shape=(5, 3, 3))
    out = cholesky(x)
    assert isinstance(out.owner.op, Blockwise)

    fn = function([x], out, mode="NUMBA")
    x_test = np.eye(3) * np.arange(1, 6)[:, None, None]
    fn(x_test)  # JIT compile
    benchmark(fn, x_test)
