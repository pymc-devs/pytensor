"""
This test is for testing the NanGuardMode.
"""

import logging

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.compile import shared
from pytensor.compile.function import function
from pytensor.compile.nanguardmode import NanGuardMode
from pytensor.configdefaults import config
from pytensor.tensor.math import dot
from pytensor.tensor.type import matrix, tensor3


def test_NanGuardMode():
    # Tests if NanGuardMode is working by feeding in numpy.inf and numpy.nans
    # intentionally. A working implementation should be able to capture all
    # the abnormalties.
    rng = np.random.default_rng(2482)
    x = matrix()
    w = shared(rng.standard_normal((5, 7)).astype(config.floatX))
    y = dot(x, w)

    fun = function([x], y, mode=NanGuardMode(nan_is_error=True, inf_is_error=True))
    a = rng.standard_normal((3, 5)).astype(config.floatX)

    with pytest.warns(RuntimeWarning):
        infa = np.tile((np.asarray(100.0) ** 1000000).astype(config.floatX), (3, 5))

    nana = np.tile(np.asarray(np.nan).astype(config.floatX), (3, 5))

    biga = np.tile(np.asarray(1e20).astype(config.floatX), (3, 5))

    fun(a)  # normal values

    # Temporarily silence logger
    _logger = logging.getLogger("pytensor.compile.nanguardmode")
    try:
        _logger.propagate = False
        with pytest.raises(AssertionError):
            fun(infa)  # INFs
        with pytest.raises(AssertionError), pytest.warns(RuntimeWarning):
            fun(nana)  # NANs
        with pytest.raises(AssertionError):
            fun(biga)  # big values
    finally:
        _logger.propagate = True

    # slices
    a = rng.standard_normal((3, 4, 5)).astype(config.floatX)

    with pytest.warns(RuntimeWarning):
        infa = np.tile((np.asarray(100.0) ** 1000000).astype(config.floatX), (3, 4, 5))

    nana = np.tile(np.asarray(np.nan).astype(config.floatX), (3, 4, 5))

    biga = np.tile(np.asarray(1e20).astype(config.floatX), (3, 4, 5))

    x = tensor3()
    y = x[:, pt.arange(2), pt.arange(2), None]
    fun = function([x], y, mode=NanGuardMode(nan_is_error=True, inf_is_error=True))
    fun(a)  # normal values
    try:
        _logger.propagate = False
        with pytest.raises(AssertionError):
            fun(infa)  # INFs
        with pytest.raises(AssertionError), pytest.warns(RuntimeWarning):
            fun(nana)  # NANs
        with pytest.raises(AssertionError):
            fun(biga)  # big values
    finally:
        _logger.propagate = True
