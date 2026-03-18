import numpy as np

import pytensor
import pytensor.tensor as pt
from pytensor import config, function
from pytensor.compile.io import In
from pytensor.tensor.slinalg import cholesky
from pytensor.tensor.type import matrix


def _check_blas_c():
    try:
        from pytensor.tensor.blas_c import CGemv  # noqa: F401
    except ImportError:
        raise NotImplementedError("C BLAS not available")


class GemvVectorDot:
    """Benchmark CGemv used as a vector dot product."""

    def setup(self):
        _check_blas_c()
        from pytensor.tensor.blas_c import CGemv

        n = 400_000
        a = pt.vector("A", shape=(n,))
        b = pt.vector("x", shape=(n,))
        out = CGemv(inplace=True)(pt.empty((1,)), 1.0, a[None], b, 0.0)
        self.fn = pytensor.function([a, b], out, accept_inplace=True, trust_input=True)
        rng = np.random.default_rng(430)
        self.test_a = rng.normal(size=n)
        self.test_b = rng.normal(size=n)

    def time_gemv_vector_dot(self):
        self.fn(self.test_a, self.test_b)


class GemvNegativeStrides:
    """Benchmark CGemv with negative strides and Fortran layout."""

    params = [[True, False], [True, False], [True, False]]
    param_names = ["neg_stride0", "neg_stride1", "F_layout"]

    def setup(self, neg_stride0, neg_stride1, F_layout):
        _check_blas_c()
        from pytensor.tensor.blas_c import CGemv

        A = pt.matrix("A", shape=(512, 512))
        x = pt.vector("x", shape=(512,))
        y = pt.vector("y", shape=(512,))
        out = CGemv(inplace=False)(y, 1.0, A, x, 1.0)
        self.fn = pytensor.function([A, x, y], out, trust_input=True)

        rng = np.random.default_rng(430)
        test_A = rng.normal(size=(512, 512))
        self.test_x = rng.normal(size=(512,))
        self.test_y = rng.normal(size=(512,))

        if F_layout:
            test_A = np.asfortranarray(test_A)
        if neg_stride0:
            test_A = test_A[::-1]
        if neg_stride1:
            test_A = test_A[:, ::-1]
        self.test_A = test_A

    def time_gemv_negative_strides(self, neg_stride0, neg_stride1, F_layout):
        self.fn(self.test_A, self.test_x, self.test_y)


class Ger:
    """Benchmark general rank-1 update (ger)."""

    params = [[2**7, 2**9, 2**13], [True, False]]
    param_names = ["n", "inplace"]

    def setup(self, n, inplace):
        alpha = pt.dscalar("alpha")
        x = pt.dvector("x")
        y = pt.dvector("y")
        A = pt.dmatrix("A")
        out = alpha * pt.outer(x, y) + A
        self.fn = pytensor.function(
            [alpha, x, y, In(A, mutable=inplace)], out, trust_input=True
        )

        rng = np.random.default_rng([2274, n])
        self.alpha_test = rng.normal(size=())
        self.x_test = rng.normal(size=(n,))
        self.y_test = rng.normal(size=(n,))
        self.A_test = rng.normal(size=(n, n))

    def time_ger(self, n, inplace):
        self.fn(self.alpha_test, self.x_test, self.y_test, self.A_test)


class Cholesky:
    """Benchmark Cholesky decomposition."""

    def setup(self):
        rng = np.random.default_rng(1234)
        r = rng.standard_normal((10, 10)).astype(config.floatX)
        self.pd = np.dot(r, r.T)
        x = matrix()
        chol = cholesky(x)
        self.fn = function([x], chol)

    def time_cholesky(self):
        self.fn(self.pd)
