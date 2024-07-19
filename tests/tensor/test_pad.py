from typing import Literal

import numpy as np
import pytest

import pytensor
from pytensor.tensor.pad import PadMode, pad


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
@pytest.mark.parametrize(
    "pad_width",
    [10, (10, 0), (0, 10)],
    ids=["symmetrical", "asymmetrical_left", "asymmetric_right"],
)
def test_constant_pad(
    size: tuple, constant: int | float, pad_width: int | tuple[int, ...]
):
    x = np.random.normal(size=size).astype(floatX)
    expected = np.pad(x, pad_width, mode="constant", constant_values=constant)
    z = pad(x, pad_width, mode="constant", constant_values=constant)
    assert z.owner.op.pad_mode == "constant"

    f = pytensor.function([], z, mode="FAST_COMPILE")

    np.testing.assert_allclose(expected, f(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "size", [(3,), (3, 3), (3, 5, 5)], ids=["1d", "2d square", "3d square"]
)
@pytest.mark.parametrize(
    "pad_width",
    [10, (10, 0), (0, 10)],
    ids=["symmetrical", "asymmetrical_left", "asymmetric_right"],
)
def test_edge_pad(size: tuple, pad_width: int | tuple[int, ...]):
    x = np.random.normal(size=size).astype(floatX)
    expected = np.pad(x, pad_width, mode="edge")
    z = pad(x, pad_width, mode="edge")
    assert z.owner.op.pad_mode == "edge"

    f = pytensor.function([], z, mode="FAST_COMPILE")

    np.testing.assert_allclose(expected, f(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "size", [(3,), (3, 3), (3, 5, 5)], ids=["1d", "2d square", "3d square"]
)
@pytest.mark.parametrize(
    "pad_width",
    [10, (10, 0), (0, 10)],
    ids=["symmetrical", "asymmetrical_left", "asymmetric_right"],
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
    assert z.owner.op.pad_mode == "linear_ramp"

    f = pytensor.function([], z, mode="FAST_COMPILE")

    np.testing.assert_allclose(expected, f(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "size", [(3,), (3, 3), (3, 5, 5)], ids=["1d", "2d square", "3d square"]
)
@pytest.mark.parametrize(
    "pad_width",
    [10, (10, 0), (0, 10)],
    ids=["symmetrical", "asymmetrical_left", "asymmetric_right"],
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
    assert z.owner.op.pad_mode == stat

    f = pytensor.function([], z, mode="FAST_COMPILE")

    np.testing.assert_allclose(expected, f(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "size", [(3,), (3, 3), (3, 5, 5)], ids=["1d", "2d square", "3d square"]
)
@pytest.mark.parametrize(
    "pad_width",
    [10, (10, 0), (0, 10)],
    ids=["symmetrical", "asymmetrical_left", "asymmetric_right"],
)
def test_wrap_pad(size: tuple, pad_width: int | tuple[int, ...]):
    x = np.random.normal(size=size).astype(floatX)
    expected = np.pad(x, pad_width, mode="wrap")
    z = pad(x, pad_width, mode="wrap")
    assert z.owner.op.pad_mode == "wrap"
    f = pytensor.function([], z, mode="FAST_COMPILE")

    np.testing.assert_allclose(expected, f(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "size", [(3,), (3, 3), (3, 5, 5)], ids=["1d", "2d square", "3d square"]
)
@pytest.mark.parametrize(
    "pad_width",
    [10, (10, 0), (0, 10)],
    ids=["symmetrical", "asymmetrical_left", "asymmetric_right"],
)
@pytest.mark.parametrize(
    "reflect_type",
    ["even", pytest.param("odd", marks=pytest.mark.xfail(raises=NotImplementedError))],
    ids=["even", "odd"],
)
def test_symmetric_pad(
    size,
    pad_width,
    reflect_type: Literal["even", "odd"],
):
    x = np.random.normal(size=size).astype(floatX)
    expected = np.pad(x, pad_width, mode="symmetric", reflect_type=reflect_type)
    z = pad(x, pad_width, mode="symmetric", reflect_type=reflect_type)
    assert z.owner.op.pad_mode == "symmetric"
    f = pytensor.function([], z, mode="FAST_COMPILE")

    np.testing.assert_allclose(expected, f(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "size", [(3,), (3, 3), (3, 5, 5)], ids=["1d", "2d square", "3d square"]
)
@pytest.mark.parametrize(
    "pad_width",
    [10, (10, 0), (0, 10)],
    ids=["symmetrical", "asymmetrical_left", "asymmetric_right"],
)
@pytest.mark.parametrize(
    "reflect_type",
    ["even", pytest.param("odd", marks=pytest.mark.xfail(raises=NotImplementedError))],
    ids=["even", "odd"],
)
def test_reflect_pad(
    size,
    pad_width,
    reflect_type: Literal["even", "odd"],
):
    x = np.random.normal(size=size).astype(floatX)
    expected = np.pad(x, pad_width, mode="reflect", reflect_type=reflect_type)
    z = pad(x, pad_width, mode="reflect", reflect_type=reflect_type)
    assert z.owner.op.pad_mode == "reflect"
    f = pytensor.function([], z, mode="FAST_COMPILE")

    np.testing.assert_allclose(expected, f(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "mode",
    [
        "constant",
        "edge",
        "linear_ramp",
        "wrap",
        "symmetric",
        "reflect",
        "mean",
        "maximum",
        "minimum",
    ],
)
@pytest.mark.parametrize("padding", ["symmetric", "asymmetric"])
def test_nd_padding(mode, padding):
    rng = np.random.default_rng()
    n = rng.integers(3, 5)
    if padding == "symmetric":
        pad_width = [(i, i) for i in rng.integers(1, 5, size=n)]
        stat_length = [(i, i) for i in rng.integers(1, 5, size=n)]
    else:
        pad_width = rng.integers(1, 5, size=(n, 2)).tolist()
        stat_length = rng.integers(1, 5, size=(n, 2)).tolist()

    test_kwargs = {
        "constant": {"constant_values": 0},
        "linear_ramp": {"end_values": 0},
        "maximum": {"stat_length": stat_length},
        "mean": {"stat_length": stat_length},
        "minimum": {"stat_length": stat_length},
        "reflect": {"reflect_type": "even"},
        "symmetric": {"reflect_type": "even"},
    }

    x = np.random.normal(size=(2,) * n).astype(floatX)
    kwargs = test_kwargs.get(mode, {})
    expected = np.pad(x, pad_width, mode=mode, **kwargs)
    z = pad(x, pad_width, mode=mode, **kwargs)
    f = pytensor.function([], z, mode="FAST_COMPILE")

    np.testing.assert_allclose(expected, f(), atol=ATOL, rtol=RTOL)
