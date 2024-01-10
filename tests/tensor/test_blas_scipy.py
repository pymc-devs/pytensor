import pickle

import numpy as np
import pytest

import pytensor
from pytensor import tensor as pt
from pytensor.tensor.blas_scipy import ScipyGer
from pytensor.tensor.math import outer
from pytensor.tensor.type import tensor
from tests.tensor.test_blas import TestBlasStrides, gemm_no_inplace
from tests.unittest_tools import OptimizationTestMixin


@pytest.mark.skipif(not pytensor.tensor.blas_scipy.have_fblas, reason="fblas needed")
class TestScipyGer(OptimizationTestMixin):
    def setup_method(self):
        self.mode = pytensor.compile.get_default_mode()
        self.mode = self.mode.including("fast_run")
        self.mode = self.mode.excluding("c_blas")  # c_blas trumps scipy Ops
        dtype = self.dtype = "float64"  # optimization isn't dtype-dependent
        self.A = tensor(dtype=dtype, shape=(None, None))
        self.a = tensor(dtype=dtype, shape=())
        self.x = tensor(dtype=dtype, shape=(None,))
        self.y = tensor(dtype=dtype, shape=(None,))
        self.Aval = np.ones((2, 3), dtype=dtype)
        self.xval = np.asarray([1, 2], dtype=dtype)
        self.yval = np.asarray([1.5, 2.7, 3.9], dtype=dtype)

    def function(self, inputs, outputs):
        return pytensor.function(inputs, outputs, self.mode)

    def run_f(self, f):
        f(self.Aval, self.xval, self.yval)
        f(self.Aval[::-1, ::-1], self.xval[::-1], self.yval[::-1])

    def b(self, bval):
        return pt.as_tensor_variable(np.asarray(bval, dtype=self.dtype))

    def test_outer(self):
        f = self.function([self.x, self.y], outer(self.x, self.y))
        self.assertFunctionContains(f, ScipyGer(destructive=True))

    def test_A_plus_outer(self):
        f = self.function([self.A, self.x, self.y], self.A + outer(self.x, self.y))
        self.assertFunctionContains(f, ScipyGer(destructive=False))
        self.run_f(f)  # DebugMode tests correctness

    def test_A_plus_scaled_outer(self):
        f = self.function(
            [self.A, self.x, self.y], self.A + 0.1 * outer(self.x, self.y)
        )
        self.assertFunctionContains(f, ScipyGer(destructive=False))
        self.run_f(f)  # DebugMode tests correctness

    def test_scaled_A_plus_scaled_outer(self):
        f = self.function(
            [self.A, self.x, self.y], 0.2 * self.A + 0.1 * outer(self.x, self.y)
        )
        self.assertFunctionContains(f, gemm_no_inplace)
        self.run_f(f)  # DebugMode tests correctness

    def test_pickle(self):
        out = ScipyGer(destructive=False)(self.A, self.a, self.x, self.y)
        f = pytensor.function([self.A, self.a, self.x, self.y], out)
        new_f = pickle.loads(pickle.dumps(f))

        assert isinstance(new_f.maker.fgraph.toposort()[-1].op, ScipyGer)
        assert np.allclose(
            f(self.Aval, 1.0, self.xval, self.yval),
            new_f(self.Aval, 1.0, self.xval, self.yval),
        )


class TestBlasStridesScipy(TestBlasStrides):
    mode = pytensor.compile.get_default_mode()
    mode = mode.including("fast_run").excluding("gpu", "c_blas")
