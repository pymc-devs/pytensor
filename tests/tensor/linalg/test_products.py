import numpy as np
import pytest
import scipy

import pytensor
from pytensor import function
from pytensor.configdefaults import config
from pytensor.tensor.linalg import (
    expm,
    kron,
    matrix_dot,
    matrix_power,
    pinv,
)
from pytensor.tensor.math import _allclose
from pytensor.tensor.type import matrix, tensor, vector
from tests import unittest_tools as utt


def test_matrix_dot():
    rng = np.random.default_rng(utt.fetch_seed())
    n = rng.integers(4) + 2
    rs = []
    xs = []
    for k in range(n):
        rs += [rng.standard_normal((4, 4)).astype(config.floatX)]
        xs += [matrix()]
    sol = matrix_dot(*xs)

    pytensor_sol = function(xs, sol)(*rs)
    numpy_sol = rs[0]
    for r in rs[1:]:
        numpy_sol = np.dot(numpy_sol, r)

    assert _allclose(numpy_sol, pytensor_sol)


class TestMatrixPower:
    @pytest.mark.parametrize("n", [-1, 0, 1, 2, 3, 4, 5, 11])
    def test_numpy_compare(self, n):
        a = np.array([[0.1231101, 0.72381381], [0.28748201, 0.43036511]]).astype(
            config.floatX
        )
        A = matrix("A", dtype=config.floatX)
        Q = matrix_power(A, n)
        n_p = np.linalg.matrix_power(a, n)
        assert np.allclose(n_p, Q.eval({A: a}))

    def test_non_square_matrix(self):
        A = matrix("A", dtype=config.floatX)
        Q = matrix_power(A, 3)
        f = function([A], [Q])
        a = np.array(
            [
                [0.47497769, 0.81869379],
                [0.74387558, 0.31780172],
                [0.54381007, 0.28153101],
            ]
        ).astype(config.floatX)
        with pytest.raises(ValueError):
            f(a)


class TestKron(utt.InferShapeTester):
    rng = np.random.default_rng(43)

    def setup_method(self):
        self.op = kron
        super().setup_method()

    def test_vec_vec_kron_raises(self):
        """Ensure kron raises an error for 1D inputs."""
        x = vector()
        y = vector()
        with pytest.raises(
            TypeError, match="kron: inputs dimensions must sum to 3 or more"
        ):
            kron(x, y)

    @pytest.mark.parametrize("static_shape", [True, False])
    @pytest.mark.parametrize("shp0", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
    @pytest.mark.parametrize("shp1", [(6,), (6, 7), (6, 7, 8), (6, 7, 8, 9)])
    def test_perform(self, static_shape, shp0, shp1):
        """Test kron execution and symbolic shape inference."""
        if len(shp0) + len(shp1) == 2:
            pytest.skip("Sum of shp0 and shp1 must be more than 2")

        a = np.asarray(self.rng.random(shp0)).astype(config.floatX)
        b = self.rng.random(shp1).astype(config.floatX)

        # Using np.kron to evaluate expected numerical output and dimensionality
        np_val = np.kron(a, b)

        # Determine tensor shapes
        shape_x = shp0 if static_shape else (None,) * len(shp0)
        shape_y = shp1 if static_shape else (None,) * len(shp1)
        shape_out = np_val.shape if static_shape else (None,) * np_val.ndim

        x = tensor(dtype="floatX", shape=shape_x)
        y = tensor(dtype="floatX", shape=shape_y)

        kron_xy = kron(x, y)

        # Assert symbolic shape inference immediately after node creation
        assert kron_xy.type.shape == shape_out

        f = function([x, y], kron_xy)
        out = f(a, b)

        np.testing.assert_allclose(out, np_val)

    @pytest.mark.parametrize(
        "i, shp0, shp1",
        [(0, (2, 3), (6, 7)), (1, (2, 3), (4, 3, 5)), (2, (2, 4, 3), (4, 3, 5))],
    )
    def test_kron_commutes_with_inv(self, i, shp0, shp1):
        if (pytensor.config.floatX == "float32") & (i == 2):
            pytest.skip("Half precision insufficient for test 3 to pass")
        x = tensor(dtype="floatX", shape=(None,) * len(shp0))
        a = np.asarray(self.rng.random(shp0)).astype(config.floatX)
        y = tensor(dtype="floatX", shape=(None,) * len(shp1))
        b = self.rng.random(shp1).astype(config.floatX)
        lhs_f = function([x, y], pinv(kron(x, y)))
        rhs_f = function([x, y], kron(pinv(x), pinv(y)))
        atol = 1e-4 if config.floatX == "float32" else 1e-12
        np.testing.assert_allclose(lhs_f(a, b), rhs_f(a, b), atol=atol)


def test_expm():
    rng = np.random.default_rng(utt.fetch_seed())
    A = rng.standard_normal((5, 5)).astype(config.floatX)

    ref = scipy.linalg.expm(A)

    x = matrix()
    m = expm(x)
    expm_f = function([x], m)

    val = expm_f(A)
    np.testing.assert_array_almost_equal(val, ref)


@pytest.mark.parametrize(
    "mode", ["symmetric", "nonsymmetric_real_eig", "nonsymmetric_complex_eig"][-1:]
)
def test_expm_grad(mode):
    rng = np.random.default_rng([898, sum(map(ord, mode))])

    match mode:
        case "symmetric":
            A = rng.standard_normal((5, 5))
            A = A + A.T
        case "nonsymmetric_real_eig":
            A = rng.standard_normal((5, 5))
            w = rng.standard_normal(5) ** 2
            A = (np.diag(w**0.5)).dot(A + A.T).dot(np.diag(w ** (-0.5)))
        case "nonsymmetric_complex_eig":
            A = rng.standard_normal((5, 5))
        case _:
            raise ValueError(f"Invalid mode: {mode}")

    utt.verify_grad(expm, [A], rng=rng, abs_tol=1e-5, rel_tol=1e-5)
