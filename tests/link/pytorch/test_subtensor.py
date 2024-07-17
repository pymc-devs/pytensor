import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor import subtensor as pt_subtensor
from tests.link.pytorch.test_basic import compare_pytorch_and_py


def test_pytorch_Subtensor():
    shape = (3, 4, 5)
    x_pt = pt.tensor("x", shape=shape, dtype="int")
    x_np = np.arange(np.prod(shape)).reshape(shape)

    # Basic indices
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

    # Advanced indexing
    out_pt = pt_subtensor.advanced_subtensor1(x_pt, [1, 2])
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor1)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np])

    out_pt = x_pt[[1, 2], [2, 3]]
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np])

    # Advanced and basic indexing
    out_pt = x_pt[[1, 2], :]
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np])

    with pytest.raises(RuntimeError):
        out_pt = x_pt[[1, 2], :, [3, 4]]
        assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)
        out_fg = FunctionGraph([x_pt], [out_pt])
        compare_pytorch_and_py(out_fg, [x_np])

    # Flipping
    with pytest.raises(
        NotImplementedError, match="Negative step sizes are not supported in Pytorch"
    ):
        out_pt = x_pt[::-1]
        out_fg = FunctionGraph([x_pt], [out_pt])
        compare_pytorch_and_py(out_fg, [x_np])

    out_pt = x_pt[np.random.binomial(1, 0.5, size=(3, 4, 5)).astype(bool)]
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_pytorch_and_py(out_fg, [x_np])


def test_pytorch_IncSubtensor():
    rng = np.random.default_rng(42)

    x_np = rng.uniform(-1, 1, size=(3, 4, 5)).astype(config.floatX)
    x_pt = pt.constant(np.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(config.floatX))

    # "Set" basic indices
    st_pt = pt.as_tensor_variable(np.array(-10.0, dtype=config.floatX))
    out_pt = pt_subtensor.set_subtensor(x_pt[1, 2, 3], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_pytorch_and_py(out_fg, [])

    st_pt = pt.as_tensor_variable(np.r_[-1.0, 0.0].astype(config.floatX))
    out_pt = pt_subtensor.set_subtensor(x_pt[:2, 0, 0], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_pytorch_and_py(out_fg, [])

    out_pt = pt_subtensor.set_subtensor(x_pt[0, 1:3, 0], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_pytorch_and_py(out_fg, [])

    # "Set" advanced indices
    st_pt = pt.as_tensor_variable(
        rng.uniform(-1, 1, size=(2, 4, 5)).astype(config.floatX)
    )
    out_pt = pt_subtensor.set_subtensor(x_pt[np.r_[0, 2]], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_pytorch_and_py(out_fg, [])

    st_pt = pt.as_tensor_variable(np.r_[-1.0, 0.0].astype(config.floatX))
    out_pt = pt_subtensor.set_subtensor(x_pt[[0, 2], 0, 0], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_pytorch_and_py(out_fg, [])

    # "Set" boolean indices
    mask_pt = pt.constant(x_np > 0)
    out_pt = pt_subtensor.set_subtensor(x_pt[mask_pt], 0.0)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_pytorch_and_py(out_fg, [])

    # "Increment" basic indices
    st_pt = pt.as_tensor_variable(np.array(-10.0, dtype=config.floatX))
    out_pt = pt_subtensor.inc_subtensor(x_pt[1, 2, 3], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_pytorch_and_py(out_fg, [])

    st_pt = pt.as_tensor_variable(np.r_[-1.0, 0.0].astype(config.floatX))
    out_pt = pt_subtensor.inc_subtensor(x_pt[:2, 0, 0], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_pytorch_and_py(out_fg, [])

    out_pt = pt_subtensor.set_subtensor(x_pt[0, 1:3, 0], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_pytorch_and_py(out_fg, [])

    # "Increment" advanced indices
    st_pt = pt.as_tensor_variable(
        rng.uniform(-1, 1, size=(2, 4, 5)).astype(config.floatX)
    )
    out_pt = pt_subtensor.inc_subtensor(x_pt[np.r_[0, 2]], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_pytorch_and_py(out_fg, [])

    st_pt = pt.as_tensor_variable(np.r_[-1.0, 0.0].astype(config.floatX))
    out_pt = pt_subtensor.inc_subtensor(x_pt[[0, 2], 0, 0], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_pytorch_and_py(out_fg, [])

    # "Increment" boolean indices
    mask_pt = pt.constant(x_np > 0)
    out_pt = pt_subtensor.set_subtensor(x_pt[mask_pt], 1.0)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_pytorch_and_py(out_fg, [])

    st_pt = pt.as_tensor_variable(x_np[[0, 2], 0, :3])
    out_pt = pt_subtensor.set_subtensor(x_pt[[0, 2], 0, :3], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_pytorch_and_py(out_fg, [])

    st_pt = pt.as_tensor_variable(x_np[[0, 2], 0, :3])
    out_pt = pt_subtensor.inc_subtensor(x_pt[[0, 2], 0, :3], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_pytorch_and_py(out_fg, [])
