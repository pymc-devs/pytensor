import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from pytensor.graph import FunctionGraph
from pytensor.tensor.pad import PadMode
from tests.link.numba.test_basic import compare_numba_and_py


floatX = config.floatX
RTOL = ATOL = 1e-6 if floatX.endswith("64") else 1e-3


@pytest.mark.parametrize(
    "mode, kwargs",
    [
        ("constant", {"constant_values": 0}),
        ("constant", {"constant_values": (1, 2)}),
        pytest.param(
            "edge",
            {},
            marks=pytest.mark.skip(
                "This is causing a segfault in NUMBA mode, but I have no idea why"
            ),
        ),
        ("linear_ramp", {"end_values": 0}),
        ("linear_ramp", {"end_values": (1, 2)}),
        ("reflect", {"reflect_type": "even"}),
        ("wrap", {}),
        ("symmetric", {"reflect_type": "even"}),
        ("mean", {"stat_length": None}),
        ("mean", {"stat_length": (10, 2)}),
        ("maximum", {"stat_length": None}),
        ("maximum", {"stat_length": (10, 2)}),
        ("minimum", {"stat_length": None}),
        ("minimum", {"stat_length": (10, 2)}),
    ],
    ids=[
        "constant_default",
        "constant_tuple",
        "edge",
        "linear_ramp_default",
        "linear_ramp_tuple",
        "reflect",
        "wrap",
        "symmetric",
        "mean_default",
        "mean_tuple",
        "maximum_default",
        "maximum_tuple",
        "minimum_default",
        "minimum_tuple",
    ],
)
def test_numba_pad(mode: PadMode, kwargs):
    x_pt = pt.tensor("x", shape=(3, 3))
    x = np.random.normal(size=(3, 3))

    res = pt.pad(x_pt, mode=mode, pad_width=3, **kwargs)
    res_fg = FunctionGraph([x_pt], [res])

    compare_numba_and_py(
        res_fg,
        [x],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL, atol=ATOL),
        py_mode="FAST_RUN",
    )
