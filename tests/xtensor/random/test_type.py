import pytest


pytest.importorskip("xarray")

import numpy as np

from pytensor.compile import function
from pytensor.tensor.random.type import random_generator_type
from pytensor.tensor.random.variable import RandomGeneratorVariable
from pytensor.xtensor.random.type import rng_to_xrng, xrng_to_rng
from pytensor.xtensor.random.variable import XRandomGeneratorVariable


def test_rng_type_casting():
    """Test casting between RandomGeneratorType and XRandomGeneratorType, including roundtrip eval."""
    # rng -> xrng -> rng roundtrip
    rng_var = random_generator_type("rng")
    assert isinstance(rng_var, RandomGeneratorVariable)
    assert not isinstance(rng_var, XRandomGeneratorVariable)

    xrng_var = rng_to_xrng(rng_var)
    assert not isinstance(xrng_var, RandomGeneratorVariable)
    assert isinstance(xrng_var, XRandomGeneratorVariable)

    next_xrng_var, x = xrng_var.normal(0, 1, extra_dims={"a": 3})
    assert isinstance(next_xrng_var, XRandomGeneratorVariable)

    rng_back = xrng_to_rng(next_xrng_var)
    assert isinstance(rng_back, RandomGeneratorVariable)
    final_rng_var, y = rng_back.normal(0, 1, size=(2,))

    fn = function([rng_var], [final_rng_var, x, y])
    _next_rng, x_result, y_result = fn(np.random.default_rng(42))
    assert x_result.shape == (3,)
    assert y_result.shape == (2,)
    assert np.unique([*x_result, *y_result]).size == 5
