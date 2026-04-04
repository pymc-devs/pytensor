import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from tests.link.mlx.test_basic import compare_mlx_and_py


mx = pytest.importorskip("mlx.core")
floatX = config.floatX
RTOL = ATOL = 1e-6 if floatX.endswith("64") else 1e-3


@pytest.mark.parametrize(
    "mode, kwargs",
    [
        ("constant", {"constant_values": 0}),
        ("edge", {}),
    ],
    ids=["constant_default", "edge"],
)
def test_mlx_pad(mode, kwargs):
    x_pt = pt.tensor("x", shape=(3, 3))
    x = np.random.normal(size=(3, 3))

    res = pt.pad(x_pt, mode=mode, pad_width=3, **kwargs)

    compare_mlx_and_py(
        [x_pt],
        [res],
        [x],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL, atol=ATOL),
    )


def test_mlx_pad_unsupported_mode():
    x_pt = pt.tensor("x", shape=(3, 3))
    res = pt.pad(x_pt, mode="reflect", pad_width=3)

    with pytest.raises(NotImplementedError, match="MLX does not support pad mode"):
        compare_mlx_and_py([x_pt], [res], [np.ones((3, 3))])


def test_mlx_pad_non_scalar_constant_values():
    x_pt = pt.tensor("x", shape=(3, 3))
    res = pt.pad(x_pt, mode="constant", pad_width=3, constant_values=(1, 2))

    with pytest.raises(
        NotImplementedError, match="only accepts a scalar constant_values"
    ):
        compare_mlx_and_py([x_pt], [res], [np.ones((3, 3))])
