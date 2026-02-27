import numpy as np
import pytest
import scipy.signal as scipy_signal

from pytensor import function
from pytensor.tensor.ssignal import GaussSpline, gauss_spline
from pytensor.tensor.type import matrix
from tests import unittest_tools as utt


class TestGaussSpline(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = GaussSpline
        self.op = gauss_spline

    @pytest.mark.parametrize("n", [-1, 1.5, None, "string"])
    def test_make_node_raises(self, n):
        a = matrix()
        with pytest.raises(ValueError, match="n must be a non-negative integer"):
            self.op(a, n=n)

    def test_perform(self):
        a = matrix()
        f = function([a], self.op(a, n=10))
        a = np.random.random((8, 6))
        assert np.allclose(f(a), scipy_signal.gauss_spline(a, 10))

    def test_infer_shape(self):
        a = matrix()
        self._compile_and_check(
            [a], [self.op(a, 16)], [np.random.random((12, 4))], self.op_class
        )
