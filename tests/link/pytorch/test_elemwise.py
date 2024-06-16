import numpy as np
import pytest
import torch

import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor import elemwise as pt_elemwise
from pytensor.tensor.type import matrix, tensor, vector
from tests.link.pytorch.test_basic import compare_pytorch_and_py


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pytorch_Dimshuffle(device):
    with torch.device(device):
        a_pt = matrix("a")

        x = a_pt.T
        x_fg = FunctionGraph([a_pt], [x])
        compare_pytorch_and_py(
            x_fg, [np.c_[[1.0, 2.0], [3.0, 4.0]].astype(config.floatX)]
        )

        x = a_pt.dimshuffle([0, 1, "x"])
        x_fg = FunctionGraph([a_pt], [x])
        compare_pytorch_and_py(
            x_fg, [np.c_[[1.0, 2.0], [3.0, 4.0]].astype(config.floatX)]
        )

        a_pt = tensor(dtype=config.floatX, shape=(None, 1))
        x = a_pt.dimshuffle((0,))
        x_fg = FunctionGraph([a_pt], [x])
        compare_pytorch_and_py(
            x_fg, [np.c_[[1.0, 2.0, 3.0, 4.0]].astype(config.floatX)]
        )

        a_pt = tensor(dtype=config.floatX, shape=(None, 1))
        x = pt_elemwise.DimShuffle([False, True], (0,))(a_pt)
        x_fg = FunctionGraph([a_pt], [x])
        compare_pytorch_and_py(
            x_fg, [np.c_[[1.0, 2.0, 3.0, 4.0]].astype(config.floatX)]
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_multiple_input_output(device):
    with torch.device(device):
        x = vector("x")
        y = vector("y")
        out = pt.mul(x, y)

        fg = FunctionGraph(outputs=[out], clone=False)
        compare_pytorch_and_py(fg, [[1.5], [2.5]])

        x = vector("x")
        y = vector("y")
        div = pt.int_div(x, y)
        pt_sum = pt.add(y, x)

        fg = FunctionGraph(outputs=[div, pt_sum], clone=False)
        compare_pytorch_and_py(fg, [[1.5], [2.5]])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pytorch_elemwise(device):
    with torch.device(device):
        x = pt.vector("x")
        out = pt.log(1 - x)

        fg = FunctionGraph([x], [out])
        compare_pytorch_and_py(fg, [[0.9, 0.9]])
