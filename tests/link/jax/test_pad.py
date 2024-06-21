import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from pytensor.graph import FunctionGraph
from pytensor.tensor.pad import PadMode
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")
floatX = config.floatX
RTOL = ATOL = 1e-6 if floatX.endswith("64") else 1e-3

test_kwargs = {
    "constant": {"constant_values": 0},
    "linear_ramp": {"end_values": 0},
    "maximum": {"stat_length": None},
    "mean": {"stat_length": [[1, 2], [3, 3]]},
    "median": {"stat_length": 2},
    "reflect": {"reflect_type": "even"},
    "symmetric": {"reflect_type": "even"},
}


@pytest.mark.parametrize(
    "mode",
    [
        "constant",
        "edge",
        "linear_ramp",
        "wrap",
        "symmetric",
        "mean",
        "maximum",
        "minimum",
    ],
)
def test_jax_pad(mode: PadMode):
    x_pt = pt.dmatrix("x")
    x = np.random.normal(size=(3, 3))
    kwargs = test_kwargs.get(mode, {})

    res = pt.pad(x_pt, mode=mode, pad_width=3, **kwargs)
    res_fg = FunctionGraph([x_pt], [res])

    compare_jax_and_py(
        res_fg,
        [x],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL, atol=ATOL),
        py_mode="FAST_COMPILE",
    )
