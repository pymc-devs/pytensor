import pytest


xr = pytest.importorskip("xarray")

from itertools import chain

import numpy as np
from xarray import DataArray
from xarray.testing import assert_allclose

from pytensor import function
from pytensor.xtensor.type import XTensorType, as_xtensor
from pytensor.xtensor.vectorization import vectorize_graph


def xr_function(*args, **kwargs):
    """Compile and wrap a PyTensor function to return xarray DataArrays."""
    fn = function(*args, **kwargs)
    symbolic_outputs = fn.maker.fgraph.outputs
    assert all(isinstance(out.type, XTensorType) for out in symbolic_outputs), (
        "All outputs must be xtensor"
    )

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


def check_vectorization(inputs, outputs, input_vals=None, rng=None):
    # Create core graph and function
    if not isinstance(inputs, list | tuple):
        inputs = (inputs,)

    if not isinstance(outputs, list | tuple):
        outputs = (outputs,)

    # apply_ufunc isn't happy with list output or single entry
    _core_fn = function(inputs, outputs)

    def core_fn(*args, _core_fn=_core_fn):
        res = _core_fn(*args)
        if len(res) == 1:
            return res[0]
        else:
            return tuple(res)

    if input_vals is None:
        rng = np.random.default_rng(rng)
        input_vals = [xr_random_like(inp, rng) for inp in inputs]

    # Create vectorized inputs
    batch_inputs = []
    batch_input_vals = []
    for i, (inp, val) in enumerate(zip(inputs, input_vals)):
        new_val = val.expand_dims({f"batch_{i}": 2 ** (i + 1)})
        new_inp = as_xtensor(new_val).type(f"batch_{inp.name or f'input{i}'}")
        batch_inputs.append(new_inp)
        batch_input_vals.append(new_val)

    # Create vectorized function
    new_outputs = vectorize_graph(outputs, dict(zip(inputs, batch_inputs)))
    vec_fn = xr_function(batch_inputs, new_outputs)
    vec_res = vec_fn(*batch_input_vals)

    # xarray.apply_ufunc with vectorize=True loops over non-core dims
    input_core_dims = [i.dims for i in inputs]
    output_core_dims = [o.dims for o in outputs]
    expected_res = xr.apply_ufunc(
        core_fn,
        *batch_input_vals,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        exclude_dims=set(chain.from_iterable((*input_core_dims, *output_core_dims))),
        vectorize=True,
    )
    if not isinstance(expected_res, list | tuple):
        expected_res = (expected_res,)

    for v_r, e_r in zip(vec_res, expected_res):
        xr_assert_allclose(v_r, e_r.transpose(*v_r.dims))
