import numpy as np
import pytest

from pytensor.graph.fg import FunctionGraph
from pytensor.tensor import subtensor as pt_subtensor
from pytensor.tensor import tensor
from tests.link.pytorch.test_basic import compare_pytorch_and_py


def test_pytorch_Subtensor():
    shape = (3, 4, 5)
    x_pt = tensor("x", shape=shape, dtype="int")
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

    # out_pt = x_pt[[1, 2], :, [3, 4]]
    # assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)
    # out_fg = FunctionGraph([x_pt], [out_pt])
    # compare_pytorch_and_py(out_fg, [x_np])

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
