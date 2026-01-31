from functools import partial

import pytest
import scipy.signal


pytest.importorskip("xarray")
pytestmark = pytest.mark.filterwarnings("error")

from xarray import apply_ufunc

from pytensor.xtensor.signal import convolve1d
from pytensor.xtensor.type import xtensor
from tests.xtensor.util import xr_arange_like, xr_assert_allclose, xr_function


@pytest.mark.parametrize("mode", ("full", "valid", "same"))
def test_convolve_1d(mode):
    in1 = xtensor("in1", dims=("batch_a", "time", "batch_b"), shape=(2, 11, 3))
    in2 = xtensor("in2", dims=("batch_c", "time", "batch_b"), shape=(5, 17, 3))

    out = convolve1d(in1, in2, mode=mode, dim="time")
    assert out.type.dims == ("batch_a", "batch_b", "batch_c", "time")
    assert out.type.shape == (2, 3, 5, None)

    fn = xr_function([in1, in2], out)
    in1_test = xr_arange_like(in1)
    in2_test = xr_arange_like(in2)

    eval_out = fn(in1_test, in2_test)
    expected_out = apply_ufunc(
        partial(scipy.signal.convolve, mode=mode),
        in1_test,
        in2_test,
        input_core_dims=[("time",), ("time",)],
        output_core_dims=[("time",)],
        exclude_dims={"time"},
        vectorize=True,
    )
    xr_assert_allclose(eval_out, expected_out)
