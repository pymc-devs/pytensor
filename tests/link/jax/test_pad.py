import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from pytensor.gradient import grad
from pytensor.tensor.pad import PadMode
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")
floatX = config.floatX
RTOL = ATOL = 1e-6 if floatX.endswith("64") else 1e-3


@pytest.mark.parametrize(
    "mode, kwargs",
    [
        ("constant", {"constant_values": 0}),
        ("constant", {"constant_values": (1, 2)}),
        ("edge", {}),
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
def test_jax_pad(mode: PadMode, kwargs):
    x_pt = pt.tensor("x", shape=(3, 3))
    x = np.random.normal(size=(3, 3))

    res = pt.pad(x_pt, mode=mode, pad_width=3, **kwargs)

    compare_jax_and_py(
        [x_pt],
        [res],
        [x],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL, atol=ATOL),
        py_mode="FAST_RUN",
    )


@pytest.mark.parametrize(
    "mode",
    [
        "constant",
        "edge",
        "linear_ramp",
        "mean",
        "maximum",
        "minimum",
        "wrap",
        "symmetric",
        "reflect",
    ],
)
def test_jax_pad_grad(mode: PadMode):
    x_pt = pt.tensor("x", shape=(8, 8))
    x = np.random.normal(size=(8, 8))

    res = pt.pad(x_pt, mode=mode, pad_width=[[1, 1], [2, 2]])
    grad_x = grad(res.sum(), x_pt)

    # The gradient allocates its zero buffer from the padded shape, which jax can
    # only trace once constant folding has resolved that shape to literals, so the
    # module's thinner `jax_mode` is not enough here.
    compare_jax_and_py(
        [x_pt],
        [grad_x],
        [x],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL, atol=ATOL),
        jax_mode="JAX",
    )
