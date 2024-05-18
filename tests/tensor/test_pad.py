import numpy as np
import pytest

import pytensor
from pytensor.tensor.pad import PadMode, flip, pad


floatX = pytensor.config.floatX
RTOL = ATOL = 1e-8 if floatX.endswith("64") else 1e-4


def test_unknown_mode_raises():
    x = np.random.normal(size=(3, 3)).astype(floatX)
    with pytest.raises(ValueError, match="Invalid mode: unknown"):
        pad(x, 1, mode="unknown")


@pytest.mark.parametrize(
    "size", [(3,), (3, 3), (3, 3, 3)], ids=["1d", "2d square", "3d square"]
)
@pytest.mark.parametrize("constant", [0, 0.0], ids=["int", "float"])
@pytest.mark.parametrize("pad_width", [1, (1, 2)], ids=["symmetrical", "asymmetrical"])
def test_constant_pad(
    size: tuple, constant: int | float, pad_width: int | tuple[int, ...]
):
    x = np.random.normal(size=size).astype(floatX)
    expected = np.pad(x, pad_width, mode="constant", constant_values=constant)
    z = pad(x, pad_width, mode="constant", constant_values=constant)
    assert z.pad_mode == "constant"

    f = pytensor.function([], z, mode="FAST_COMPILE")

    np.testing.assert_allclose(expected, f(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "size", [(3,), (3, 3), (3, 5, 5)], ids=["1d", "2d square", "3d square"]
)
@pytest.mark.parametrize(
    "pad_width", [1, (1, 2)], ids=["symmetrical", "asymmetrical_1d"]
)
def test_edge_pad(size: tuple, pad_width: int | tuple[int, ...]):
    x = np.random.normal(size=size).astype(floatX)
    expected = np.pad(x, pad_width, mode="edge")
    z = pad(x, pad_width, mode="edge")
    assert z.pad_mode == "edge"

    f = pytensor.function([], z, mode="FAST_COMPILE")

    np.testing.assert_allclose(expected, f(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "size", [(3,), (3, 3), (3, 5, 5)], ids=["1d", "2d square", "3d square"]
)
@pytest.mark.parametrize(
    "pad_width", [1, (1, 2)], ids=["symmetrical", "asymmetrical_1d"]
)
@pytest.mark.parametrize("end_values", [0, -1], ids=["0", "-1"])
def test_linear_ramp_pad(
    size: tuple,
    pad_width: int | tuple[int, ...],
    end_values: int | float | tuple[int | float, ...],
):
    x = np.random.normal(size=size).astype(floatX)
    expected = np.pad(x, pad_width, mode="linear_ramp", end_values=end_values)
    z = pad(x, pad_width, mode="linear_ramp", end_values=end_values)
    assert z.pad_mode == "linear_ramp"

    f = pytensor.function([], z, mode="FAST_COMPILE")

    np.testing.assert_allclose(expected, f(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "size", [(3,), (3, 3), (3, 5, 5)], ids=["1d", "2d square", "3d square"]
)
@pytest.mark.parametrize(
    "pad_width", [1, (1, 2)], ids=["symmetrical", "asymmetrical_1d"]
)
@pytest.mark.parametrize("stat", ["mean", "minimum", "maximum"])
@pytest.mark.parametrize("stat_length", [None, 2])
def test_stat_pad(
    size: tuple,
    pad_width: int | tuple[int, ...],
    stat: PadMode,
    stat_length: int | None,
):
    x = np.random.normal(size=size).astype(floatX)
    expected = np.pad(x, pad_width, mode=stat, stat_length=stat_length)
    z = pad(x, pad_width, mode=stat, stat_length=stat_length)
    assert z.pad_mode == stat
    assert z.stat_length_input == (stat_length is not None)

    f = pytensor.function([], z, mode="FAST_COMPILE")

    np.testing.assert_allclose(expected, f(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "size", [(3,), (3, 3), (3, 5, 5)], ids=["1d", "2d square", "3d square"]
)
@pytest.mark.parametrize(
    "pad_width", [1, (1, 2)], ids=["symmetrical", "asymmetrical_1d"]
)
def test_wrap_pad(size: tuple, pad_width: int | tuple[int, ...]):
    x = np.random.normal(size=size).astype(floatX)
    expected = np.pad(x, pad_width, mode="wrap")
    z = pad(x, pad_width, mode="wrap")
    assert z.pad_mode == "wrap"
    f = pytensor.function([], z, mode="FAST_COMPILE")

    np.testing.assert_allclose(expected, f(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "size", [(3,), (3, 3), (3, 5, 5)], ids=["1d", "2d square", "3d square"]
)
@pytest.mark.parametrize(
    "pad_width", [1, (1, 2)], ids=["symmetrical", "asymmetrical_1d"]
)
@pytest.mark.parametrize(
    "reflect_type",
    ["even", pytest.param("odd", marks=pytest.mark.xfail(raises=NotImplementedError))],
    ids=["even", "odd"],
)
def test_symmetric_pad(size, pad_width, reflect_type):
    x = np.random.normal(size=size).astype(floatX)
    expected = np.pad(x, pad_width, mode="symmetric", reflect_type=reflect_type)
    z = pad(x, pad_width, mode="symmetric", reflect_type=reflect_type)
    assert z.pad_mode == "symmetric"
    f = pytensor.function([], z, mode="FAST_COMPILE")

    np.testing.assert_allclose(expected, f(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "size", [(3,), (3, 3), (3, 5, 5)], ids=["1d", "2d square", "3d square"]
)
def test_flip(size: tuple[int]):
    from itertools import combinations

    x = np.random.normal(size=size).astype(floatX)
    x_pt = pytensor.tensor.tensor(shape=size, name="x")
    expected = np.flip(x, axis=None)
    z = flip(x_pt, axis=None)
    f = pytensor.function([x_pt], z, mode="FAST_COMPILE")
    np.testing.assert_allclose(expected, f(x), atol=ATOL, rtol=RTOL)

    # Test all combinations of axes
    flip_options = [
        axes for i in range(1, x.ndim + 1) for axes in combinations(range(x.ndim), r=i)
    ]
    for axes in flip_options:
        expected = np.flip(x, axis=list(axes))
        z = flip(x_pt, axis=list(axes))
        f = pytensor.function([x_pt], z, mode="FAST_COMPILE")
        np.testing.assert_allclose(expected, f(x), atol=ATOL, rtol=RTOL)
