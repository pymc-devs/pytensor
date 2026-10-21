import numpy as np
import pytest
import scipy.special.cython_special


numba = pytest.importorskip("numba")


from numba.types import float32, float64, int32, int64

from pytensor.link.numba.dispatch.cython_support import wrap_cython_function


@pytest.mark.parametrize(
    "func, output, inputs, expected",
    [
        (
            scipy.special.cython_special.agm,
            np.float64,
            [np.float64, np.float64],
            float64(float64, float64, int32),
        ),
        (
            scipy.special.cython_special.erfc,
            np.float64,
            [np.float64],
            float64(float64, int32),
        ),
        (
            scipy.special.cython_special.expit,
            np.float32,
            [np.float32],
            float32(float32, int32),
        ),
        (
            scipy.special.cython_special.expit,
            np.float64,
            [np.float64],
            float64(float64, int32),
        ),
        (
            # expn doesn't have a float32 implementation
            scipy.special.cython_special.expn,
            np.float32,
            [np.float32, np.float32],
            float64(float64, float64, int32),
        ),
        (
            # We choose the integer implementation if possible
            scipy.special.cython_special.expn,
            np.float32,
            [np.int64, np.float32],
            float64(int64, float64, int32),
        ),
    ],
)
def test_choose_signature(func, output, inputs, expected):
    wrapper = wrap_cython_function(func, output, inputs)
    assert wrapper.signature.res_dtype == wrapper.output_dtype
    assert wrapper.signature.arg_dtypes == wrapper.input_dtypes
    assert wrapper.numba_signature() == expected
