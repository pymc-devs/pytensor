import re
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
    in2 = xtensor("in2", dims=("batch_c", "kernel", "batch_b"), shape=(5, 17, 3))

    out = convolve1d(in1, in2, mode=mode, dims=("time", "kernel"))
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
        input_core_dims=[("time",), ("kernel",)],
        output_core_dims=[("time",)],
        exclude_dims={"time"},  # Output time isn't aligned with input
        vectorize=True,
    )
    xr_assert_allclose(eval_out, expected_out)


def test_convolve_1d_invalid():
    in1 = xtensor("x", dims=("time", "batch"))
    in2 = xtensor("x", dims=("batch", "kernel"))

    # Check valid case doesn't raise
    convolve1d(in1, in2, dims=("time", "kernel"))

    with pytest.raises(ValueError, match=r"mode must be one of .*, got parisian"):
        convolve1d(in1, in2, mode="parisian", dims=("time", "kernel"))

    with pytest.raises(ValueError, match="Two dims required"):
        convolve1d(in1, in2, dims=("time",))

    with pytest.raises(ValueError, match="The two dims must be unique"):
        convolve1d(in1, in2, dims=("batch", "batch"))

    with pytest.raises(
        ValueError,
        match=re.escape("Input 0 has invalid core dims ['kernel']. Allowed: ('time',)"),
    ):
        convolve1d(in1.rename({"batch": "kernel"}), in2, dims=("time", "kernel"))

    with pytest.raises(
        ValueError,
        match=re.escape("Input 1 has invalid core dims ['time']. Allowed: ('kernel',)"),
    ):
        convolve1d(in1, in2.rename({"batch": "time"}), dims=("time", "kernel"))
