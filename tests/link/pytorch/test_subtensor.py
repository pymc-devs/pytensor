import contextlib

import numpy as np
import pytest

import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor import inc_subtensor, set_subtensor
from pytensor.tensor import subtensor as pt_subtensor
from tests.link.pytorch.test_basic import compare_pytorch_and_py


def test_pytorch_Subtensor():
    shape = (3, 4, 5)
    x_pt = pt.tensor("x", shape=shape, dtype="int")
    x_np = np.arange(np.prod(shape)).reshape(shape)

    out_pt = x_pt[1, 2, 0]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np])

    out_pt = x_pt[1:, 1, :]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np])

    out_pt = x_pt[:2, 1, :]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np])

    out_pt = x_pt[1:2, 1, :]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np])

    # symbolic index
    a_pt = ps.int64("a")
    a_np = 1
    out_pt = x_pt[a_pt, 2, a_pt:2]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    out_fg = FunctionGraph([x_pt, a_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np, a_np])

    with pytest.raises(
        NotImplementedError, match="Negative step sizes are not supported in Pytorch"
    ):
        out_pt = x_pt[::-1]
        out_fg = FunctionGraph([x_pt], [out_pt])
        compare_pytorch_and_py(out_fg, [x_np])


def test_pytorch_AdvSubtensor():
    shape = (3, 4, 5)
    x_pt = pt.tensor("x", shape=shape, dtype="int")
    x_np = np.arange(np.prod(shape)).reshape(shape)

    out_pt = pt_subtensor.advanced_subtensor1(x_pt, [1, 2])
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor1)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np])

    out_pt = x_pt[[1, 2], [2, 3]]
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np])

    out_pt = x_pt[[1, 2], 1:]
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np])

    out_pt = x_pt[[1, 2], :, [3, 4]]
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np])

    out_pt = x_pt[[1, 2], None]
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np])

    a_pt = ps.int64("a")
    a_np = 2
    out_pt = x_pt[[1, a_pt], a_pt]
    out_fg = FunctionGraph([x_pt, a_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np, a_np])

    # boolean indices
    out_pt = x_pt[np.random.binomial(1, 0.5, size=(3, 4, 5)).astype(bool)]
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np])

    a_pt = pt.tensor3("a", dtype="bool")
    a_np = np.random.binomial(1, 0.5, size=(3, 4, 5)).astype(bool)
    out_pt = x_pt[a_pt]
    out_fg = FunctionGraph([x_pt, a_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np, a_np])

    with pytest.raises(
        NotImplementedError, match="Negative step sizes are not supported in Pytorch"
    ):
        out_pt = x_pt[[1, 2], ::-1]
        out_fg = FunctionGraph([x_pt], [out_pt])
        assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)
        compare_pytorch_and_py(out_fg, [x_np])


@pytest.mark.parametrize("subtensor_op", [set_subtensor, inc_subtensor])
def test_pytorch_IncSubtensor(subtensor_op):
    x_pt = pt.tensor3("x")
    x_test = np.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(config.floatX)

    st_pt = pt.as_tensor_variable(np.array(-10.0, dtype=config.floatX))
    out_pt = subtensor_op(x_pt[1, 2, 3], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_test])

    # Test different type update
    st_pt = pt.as_tensor_variable(np.r_[-1.0, 0.0].astype("float32"))
    out_pt = subtensor_op(x_pt[:2, 0, 0], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_test])

    out_pt = subtensor_op(x_pt[0, 1:3, 0], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_test])


def inc_subtensor_ignore_duplicates(x, y):
    return inc_subtensor(x, y, ignore_duplicates=True)


@pytest.mark.parametrize(
    "advsubtensor_op", [set_subtensor, inc_subtensor, inc_subtensor_ignore_duplicates]
)
def test_pytorch_AvdancedIncSubtensor(advsubtensor_op):
    rng = np.random.default_rng(42)

    x_pt = pt.tensor3("x")
    x_test = (np.arange(3 * 4 * 5) + 1).reshape((3, 4, 5)).astype(config.floatX)

    st_pt = pt.as_tensor_variable(
        rng.uniform(-1, 1, size=(2, 4, 5)).astype(config.floatX)
    )
    out_pt = advsubtensor_op(x_pt[np.r_[0, 2]], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_test])

    # Repeated indices
    out_pt = advsubtensor_op(x_pt[np.r_[0, 0]], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_test])

    # Mixing advanced and basic indexing
    if advsubtensor_op is inc_subtensor:
        # PyTorch does not support `np.add.at` equivalent with slices
        expectation = pytest.raises(NotImplementedError)
    else:
        expectation = contextlib.nullcontext()
    st_pt = pt.as_tensor_variable(x_test[[0, 2], 0, :3])
    out_pt = advsubtensor_op(x_pt[[0, 0], 0, :3], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    with expectation:
        compare_pytorch_and_py(out_fg, [x_test])

    # Test different dtype update
    st_pt = pt.as_tensor_variable(np.r_[-1.0, 0.0].astype("float32"))
    out_pt = advsubtensor_op(x_pt[[0, 2], 0, 0], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_test])

    # Boolean indices
    out_pt = advsubtensor_op(x_pt[x_pt > 5], 1.0)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_test])
