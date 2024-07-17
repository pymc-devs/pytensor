import numpy as np
import pytest

from pytensor.compile.mode import Mode
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import get_test_value
from pytensor.link.pytorch.linker import PytorchLinker
from pytensor.tensor import blas as pt_blas
from pytensor.tensor.type import tensor3
from tests.link.pytorch.test_basic import compare_pytorch_and_py


def test_pytorch_BatchedDot():
    # tensor3 . tensor3
    a = tensor3("a")
    a.tag.test_value = (
        np.linspace(-1, 1, 10 * 5 * 3).astype(config.floatX).reshape((10, 5, 3))
    )
    b = tensor3("b")
    b.tag.test_value = (
        np.linspace(1, -1, 10 * 3 * 2).astype(config.floatX).reshape((10, 3, 2))
    )
    out = pt_blas.BatchedDot()(a, b)
    fgraph = FunctionGraph([a, b], [out])
    pytensor_pytorch_fn, _ = compare_pytorch_and_py(
        fgraph, [get_test_value(i) for i in fgraph.inputs]
    )

    # A dimension mismatch should raise a TypeError for compatibility
    inputs = [get_test_value(a)[:-1], get_test_value(b)]
    pytorch_mode_no_rewrites = Mode(PytorchLinker(), None)
    pytensor_pytorch_fn.mode = pytorch_mode_no_rewrites
    with pytest.raises(TypeError):
        pytensor_pytorch_fn(*inputs)
