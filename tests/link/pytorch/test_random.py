import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.compile.function import function
from pytensor.compile.sharedvalue import shared
from tests.link.pytorch.test_basic import pytorch_mode


torch = pytest.importorskip("torch")


@pytest.mark.parametrize("size", [(), (4,)])
def test_random_bernoulli(size):
    rng = shared(np.random.default_rng(123))

    g = pt.random.bernoulli(0.5, size=(1000, *size), rng=rng)
    g_fn = function([], g, mode=pytorch_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(axis=0), 0.5, 1)
