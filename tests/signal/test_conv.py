import numpy as np

from pytensor.signal.conv import Conv1d
from tests import unittest_tools as utt


def test_conv1d_grads():
    rng = np.random.default_rng()

    data_val = rng.normal(size=(3,))
    kernel_val = rng.normal(size=(5,))

    op = Conv1d(mode="full")

    utt.verify_grad(op=op, pt=[data_val, kernel_val])
