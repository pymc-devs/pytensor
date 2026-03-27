import numpy as np

import pytensor
from pytensor.tensor import slinalg
from tests import unittest_tools as utt


def test_eigvalsh_grad_no_b():
    rng = np.random.default_rng(utt.fetch_seed())
    a = rng.standard_normal((5, 5))
    a = a + a.T

    def func(x):
        return slinalg.eigvalsh(x, pytensor.tensor.type_other.NoneConst)

    utt.verify_grad(func, [a])


def test_eigvalsh_grad_with_b():
    rng = np.random.default_rng(utt.fetch_seed())
    a = rng.standard_normal((5, 5))
    a = a + a.T
    b = rng.standard_normal((5, 5))
    b = b.dot(b.T) + np.eye(5)

    utt.verify_grad(lambda x, y: slinalg.eigvalsh(x, y), [a, b])
