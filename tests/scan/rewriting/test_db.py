import numpy as np

from pytensor import function, scan
from pytensor.configdefaults import config
from pytensor.tensor.blas import Dot22
from pytensor.tensor.math import dot
from pytensor.tensor.type import matrix
from tests import unittest_tools as utt


def test_opt_order():
    """
    Verify that scan optimizations are applied before blas
    optimizations.

    This is needed as otherwise, the dot won't become a dot22
    so it will be slower and won't get transferred to the gpu.
    """

    x = matrix("x")
    A = matrix("A")

    z = scan(dot, sequences=[], non_sequences=[x, A], n_steps=2, return_updates=False)
    f = function([x, A], z, mode="CVM")
    topo = f.maker.fgraph.toposort()

    assert any(isinstance(node.op, Dot22) for node in topo)

    vx = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=config.floatX)
    vA = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=config.floatX)
    vR = np.array([[[2, 1], [4, 2]], [[2, 1], [4, 2]]], dtype=config.floatX)
    utt.assert_allclose(f(vx, vA), vR)
