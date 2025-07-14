# ruff: noqa: E402
import pytest


pytest.importorskip("xarray")

import numpy as np
from xarray import DataArray
from xarray.testing import assert_allclose

from pytensor import function
from pytensor.xtensor.type import XTensorType


def xr_function(*args, **kwargs):
    """Compile and wrap a PyTensor function to return xarray DataArrays."""
    fn = function(*args, **kwargs)
    symbolic_outputs = fn.maker.fgraph.outputs
    assert all(
        isinstance(out.type, XTensorType) for out in symbolic_outputs
    ), "All outputs must be xtensor"

    def xfn(*xr_inputs):
        np_inputs = [
            inp.values if isinstance(inp, DataArray) else inp for inp in xr_inputs
        ]
        np_outputs = fn(*np_inputs)
        if not isinstance(np_outputs, tuple | list):
            return DataArray(np_outputs, dims=[dim.name for dim in symbolic_outputs[0].type.dims])
        else:
            return tuple(
                DataArray(res, dims=[dim.name for dim in out.type.dims])
                for res, out in zip(np_outputs, symbolic_outputs)
            )

    xfn.fn = fn
    return xfn


def xr_assert_allclose(x, y, *args, **kwargs):
    # Assert that two xarray DataArrays are close, ignoring coordinates
    x = x.drop_vars(x.coords)
    y = y.drop_vars(y.coords)
    assert_allclose(x, y, *args, **kwargs)


def xr_arange_like(x):
    data = np.arange(np.prod(x.type.shape), dtype=x.type.dtype)
    dtype = x.type.dtype

    return DataArray(
        data.reshape(x.type.shape),
        dims=[dim.name for dim in x.type.dims],
    )


def xr_random_like(x, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    return DataArray(
        rng.standard_normal(size=x.type.shape, dtype=x.type.dtype), dims=x.type.dims
    )
