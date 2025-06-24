import numpy as np

import pytensor.tensor as pt
from pytensor import config
from pytensor.tensor.blas import banded_gemv
from tests.link.numba.test_basic import compare_numba_and_py, numba_inplace_mode
from tests.tensor.test_slinalg import _make_banded_A


def test_banded_dot():
    rng = np.random.default_rng()

    A = pt.tensor("A", shape=(10, 10), dtype=config.floatX)
    A_val = _make_banded_A(rng.normal(size=(10, 10)), kl=1, ku=1).astype(config.floatX)

    x = pt.tensor("x", shape=(10,), dtype=config.floatX)
    x_val = rng.normal(size=(10,)).astype(config.floatX)

    output = banded_gemv(A, x, upper_diags=1, lower_diags=1)

    fn, _ = compare_numba_and_py(
        [A, x],
        output,
        test_inputs=[A_val, x_val],
        numba_mode=numba_inplace_mode,
        eval_obj_mode=False,
    )

    for stride in [2, -1, -2]:
        x_shape = (10 * abs(stride),)
        x_val = rng.normal(size=x_shape).astype(config.floatX)
        x_val = x_val[::stride]

        nb_output = fn(A_val, x_val)
        expected = A_val @ x_val

        np.testing.assert_allclose(
            nb_output,
            expected,
            strict=True,
            err_msg=f"Test failed for stride = {stride}",
        )
