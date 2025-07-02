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
            return DataArray(np_outputs, dims=symbolic_outputs[0].type.dims)
        else:
            return tuple(
                DataArray(res, dims=out.type.dims)
                for res, out in zip(np_outputs, symbolic_outputs)
            )

    xfn.fn = fn
    return xfn


def xr_assert_allclose(x, y, check_dtype=False, *args, **kwargs):
    """Assert that two xarray DataArrays are close, ignoring coordinates.

    Mostly a wrapper around xarray.testing.assert_allclose,
    but with the option to check the dtype.

    Parameters
    ----------
    x : xarray.DataArray
        The first xarray DataArray to compare.
    y : xarray.DataArray
        The second xarray DataArray to compare.
    check_dtype : bool, optional
        If True, check that the dtype of the two DataArrays is the same.
    *args :
        Additional arguments to pass to xarray.testing.assert_allclose.
    **kwargs :
        Additional keyword arguments to pass to xarray.testing.assert_allclose.
    """
    x = x.drop_vars(x.coords)
    y = y.drop_vars(y.coords)
    assert_allclose(x, y, *args, **kwargs)
    if check_dtype:
        assert x.dtype == y.dtype


def xr_arange_like(x):
    return DataArray(
        np.arange(np.prod(x.type.shape), dtype=x.type.dtype).reshape(x.type.shape),
        dims=x.type.dims,
    )


def xr_random_like(x, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    return DataArray(
        rng.standard_normal(size=x.type.shape, dtype=x.type.dtype), dims=x.type.dims
    )
