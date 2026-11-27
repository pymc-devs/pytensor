import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from tests.link.pytorch.test_basic import compare_pytorch_and_py


@pytest.mark.parametrize(
    "shapes",
    [[(5, 5), (3, 2), (2, 4)], [(4, 5, 5), (4, 3, 3)]],
    ids=["core", "batched"],
)
def test_block_diag(shapes):
    rng = np.random.default_rng(15)
    inputs = [pt.tensor(f"x{i}", shape=shape) for i, shape in enumerate(shapes)]
    values = [rng.normal(size=shape).astype(config.floatX) for shape in shapes]

    compare_pytorch_and_py(inputs, [pt.linalg.block_diag(*inputs)], values)
