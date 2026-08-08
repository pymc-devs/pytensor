from typing import Literal

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.mode import Mode
from pytensor.gradient import grad
from pytensor.graph.basic import Constant
from pytensor.tensor.pad import PadMode, pad
from pytensor.tensor.subtensor import Subtensor
from tests import unittest_tools as utt


floatX = pytensor.config.floatX
RTOL = ATOL = 1e-8 if floatX.endswith("64") else 1e-4


@pytest.mark.parametrize("kwargs", [{}, {"constant_values": 0}], ids=["none", "valid"])
def test_unknown_mode_raises(kwargs):
    """An unknown mode is reported as such, before its options are looked up."""
    x = np.random.normal(size=(3, 3)).astype(floatX)
    with pytest.raises(ValueError, match="Invalid mode: unknown"):
        pad(x, 1, mode="unknown", **kwargs)


@pytest.mark.parametrize(
    "mode, kwargs",
    [
        ("edge", {"constant_values": 1}),
        ("constant", {"end_values": 1}),
        ("wrap", {"stat_length": 2}),
        ("mean", {"reflect_type": "even"}),
    ],
    ids=str,
)
def test_option_not_valid_for_mode_raises(mode, kwargs):
    """An option a mode does not accept is rejected rather than silently ignored."""
    x = np.random.normal(size=(3, 3)).astype(floatX)
    with pytest.raises(
        ValueError, match=f"Invalid keyword arguments for mode '{mode}'"
    ):
        pad(x, 1, mode=mode, **kwargs)


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


ALL_MODES = [
    "constant",
    "edge",
    "linear_ramp",
    "mean",
    "maximum",
    "minimum",
    "wrap",
    "symmetric",
    "reflect",
]

GATHER_BASED_MODES = ["wrap", "symmetric", "reflect"]


def _pad_width_as_symbolic_expression(ndim, width):
    """Build a uniform pad_width of ``width`` that is symbolic but constant-foldable."""
    pad_width = pt.zeros((ndim, 2), dtype="int64")
    for axis in range(ndim):
        pad_width = pad_width[axis, 0].set(width)
        pad_width = pad_width[axis, 1].set(width)
    return pad_width


def _dynamic_subtensors(inputs, outputs):
    """Collect the ``idx_list`` of every Subtensor whose indices are not all constants.

    Descends into ``OpFromGraph`` inner graphs, where ``Pad`` keeps the interior
    slice it takes in its gradient.
    """
    fn = pytensor.function(
        inputs, outputs, mode=Mode(linker="py", optimizer="fast_run")
    )
    found = []

    def walk(nodes):
        for node in nodes:
            if isinstance(node.op, Subtensor) and any(
                not isinstance(i, Constant) for i in node.inputs[1:]
            ):
                found.append(node.op.idx_list)
            if isinstance(node.op, OpFromGraph):
                walk(node.op.fgraph.apply_nodes)

    walk(fn.maker.fgraph.apply_nodes)
    return found


@pytest.mark.parametrize("mode", ALL_MODES)
def test_pad_static_shape(mode):
    """A statically known pad_width should give a statically known output shape."""
    x = pt.tensor("x", shape=(8, 8))
    z = pad(x, [[1, 1], [2, 2]], mode=mode)

    assert z.type.shape == (10, 12)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_pad_grad_has_static_slice_bounds(mode):
    """The gradient must not index the padded interior with runtime values.

    Backends that require static slice bounds cannot compile a gradient whose
    interior slice is bounded by a runtime value.
    """
    x = pt.tensor("x", shape=(8, 8))
    z = pad(x, [[1, 1], [2, 2]], mode=mode)
    grad_x = grad(z.sum(), x)

    assert _dynamic_subtensors([x], grad_x) == []


@pytest.mark.xfail(reason="pad does not constant-fold a symbolic pad_width")
@pytest.mark.parametrize("mode", ["constant", "edge"])
def test_pad_static_shape_from_foldable_pad_width(mode):
    """A constant-foldable pad_width expression should still give static shapes."""
    x = pt.tensor("x", shape=(8, 8))
    pad_width = _pad_width_as_symbolic_expression(2, 1)
    z = pad(x, pad_width, mode=mode)

    assert z.type.shape == (10, 10)


@pytest.mark.parametrize("mode", GATHER_BASED_MODES)
@pytest.mark.parametrize(
    "pad_width",
    [(1, 1), (7, 7), (13, 4), (0, 9)],
    ids=["within_axis", "wider_than_axis", "asymmetric_wide", "one_sided_wide"],
)
@pytest.mark.parametrize("size", [(1,), (2,), (5,), (3, 4)], ids=str)
def test_gather_pad_wider_than_axis(mode, pad_width, size):
    """Padding wider than the axis wraps/reflects repeatedly."""
    x = np.arange(np.prod(size), dtype=floatX).reshape(size)
    expected = np.pad(x, pad_width, mode=mode)

    z = pad(pt.as_tensor(x), pad_width, mode=mode)
    assert z.type.shape == expected.shape

    np.testing.assert_allclose(z.eval(), expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("mode", GATHER_BASED_MODES)
@pytest.mark.parametrize(
    "width", [(1, 1), (7, 2)], ids=["within_axis", "wider_than_axis"]
)
@pytest.mark.parametrize("size", [5, 1], ids=["size_5", "size_1"])
def test_gather_pad_symbolic_pad_width(mode, width, size):
    """A symbolic pad_width still works, without static shapes.

    A size-1 axis degenerates the reflect period, which the symbolic branch of the
    index map has to clamp separately from the static one.
    """
    x = pt.tensor("x", shape=(size,))
    pad_width = pt.vector("pad_width", shape=(2,), dtype="int64")
    z = pad(x, pad_width, mode=mode)
    assert z.type.shape == (None,)

    x_val = np.arange(size, dtype=floatX)
    np.testing.assert_allclose(
        z.eval({x: x_val, pad_width: np.array(width, dtype="int64")}),
        np.pad(x_val, width, mode=mode),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("mode", ALL_MODES)
def test_pad_rejects_non_integral_pad_width(mode):
    """A float pad_width must be rejected, not silently truncated."""
    x = pt.tensor("x", shape=(5,))

    with pytest.raises(TypeError, match="must be of integral type"):
        pad(x, (1.5, 1.5), mode=mode)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_pad_grad_matches_numeric(mode):
    """Every mode's gradient must match finite differences."""
    rng = np.random.default_rng(11)
    x_val = rng.normal(size=(4, 4))

    utt.verify_grad(
        lambda x: pad(x, ((1, 2), (2, 1)), mode=mode),
        [x_val],
        rng=np.random.default_rng(11),
    )


@pytest.mark.parametrize(
    "pad_width",
    [((1, 1), (1, 1)), ((1, 1), (2, 2)), ((0, 3), (2, 1)), 2, ((5, 5), (5, 5))],
    ids=["all_ones", "axes", "sides", "scalar", "wider_than_axis"],
)
def test_edge_pad_grad(pad_width):
    """Each input element is copied a fixed number of times into the output, so
    ``d(sum(pad(x)))/dx`` is exactly that copy count.
    """
    x = pt.tensor("x", shape=(8, 8))
    grad_x = grad(pad(x, pad_width, mode="edge").sum(), x)

    x_val = np.arange(64, dtype=floatX).reshape(8, 8)
    got = pytensor.function([x], grad_x, mode=Mode(linker="py", optimizer="fast_run"))(
        x_val
    )

    width = pad_width if not isinstance(pad_width, int) else ((pad_width,) * 2,) * 2
    expected = np.zeros_like(x_val)
    for i in range(8):
        for j in range(8):
            probe = np.zeros_like(x_val)
            probe[i, j] = 1.0
            expected[i, j] = np.pad(probe, width, mode="edge").sum()
    np.testing.assert_allclose(got, expected, atol=ATOL, rtol=RTOL)
