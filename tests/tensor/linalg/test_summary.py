import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytensor
from pytensor import function
from pytensor.configdefaults import config
from pytensor.tensor.linalg import det, norm, slogdet, trace
from pytensor.tensor.type import matrix, scalar, tensor, vector
from tests import unittest_tools as utt


def test_det():
    rng = np.random.default_rng(utt.fetch_seed())

    r = rng.standard_normal((5, 5)).astype(config.floatX)
    x = matrix()
    f = pytensor.function([x], det(x))
    assert np.allclose(np.linalg.det(r), f(r))


def test_det_non_square_raises():
    with pytest.raises(ValueError, match="Determinant not defined"):
        det(tensor("x", shape=(5, 7)))


def test_det_grad():
    rng = np.random.default_rng(utt.fetch_seed())

    r = rng.standard_normal((5, 5)).astype(config.floatX)
    utt.verify_grad(det, [r], rng=np.random)


def test_det_shape():
    x = matrix()
    assert det(x).type.shape == ()


def test_slogdet():
    rng = np.random.default_rng(utt.fetch_seed())

    r = rng.standard_normal((5, 5)).astype(config.floatX)
    x = matrix()
    f = pytensor.function([x], slogdet(x))
    f_sign, f_det = f(r)
    sign, det_val = np.linalg.slogdet(r)
    assert np.equal(sign, f_sign)
    assert np.allclose(det_val, f_det)
    # check numpy array types is returned
    # see https://github.com/pymc-devs/pytensor/issues/799
    sign, logdet = slogdet(x)
    det_result = sign * pytensor.tensor.exp(logdet)
    assert_array_almost_equal(det_result.eval({x: [[1]]}), np.array(1.0))


def test_trace():
    rng = np.random.default_rng(utt.fetch_seed())
    x = matrix()
    with pytest.warns(FutureWarning):
        g = trace(x)
    f = pytensor.function([x], g)

    for shp in [(2, 3), (3, 2), (3, 3)]:
        m = rng.random(shp).astype(config.floatX)
        v = np.trace(m)
        assert v == f(m)

    xx = vector()
    ok = False
    try:
        with pytest.warns(FutureWarning):
            trace(xx)
    except TypeError:
        ok = True
    except ValueError:
        ok = True

    assert ok


class TestNorm:
    def test_wrong_type_of_ord_for_vector(self):
        with pytest.raises(ValueError, match="Invalid norm order 'fro' for vectors"):
            norm([2, 1], "fro")

    def test_wrong_type_of_ord_for_matrix(self):
        ord = 0
        with pytest.raises(ValueError, match=f"Invalid norm order for matrices: {ord}"):
            norm([[2, 1], [3, 4]], ord)

    def test_non_tensorial_input(self):
        with pytest.raises(
            ValueError,
            match="Cannot compute norm when core_dims < 1 or core_dims > 3, found: core_dims = 0",
        ):
            norm(3, ord=2)

    def test_invalid_axis_input(self):
        axis = scalar("i", dtype="int")
        with pytest.raises(
            TypeError, match="'axis' must be None, an integer, or a tuple of integers"
        ):
            norm([[1, 2], [3, 4]], axis=axis)

    @pytest.mark.parametrize(
        "ord",
        [None, np.inf, -np.inf, 1, -1, 2, -2],
        ids=["None", "inf", "-inf", "1", "-1", "2", "-2"],
    )
    @pytest.mark.parametrize("core_dims", [(4,), (4, 3)], ids=["vector", "matrix"])
    @pytest.mark.parametrize("batch_dims", [(), (2,)], ids=["no_batch", "batch"])
    @pytest.mark.parametrize("test_imag", [True, False], ids=["complex", "real"])
    @pytest.mark.parametrize(
        "keepdims", [True, False], ids=["keep_dims=True", "keep_dims=False"]
    )
    def test_numpy_compare(
        self,
        ord: float,
        core_dims: tuple[int, ...],
        batch_dims: tuple[int, ...],
        test_imag: bool,
        keepdims: bool,
        axis=None,
    ):
        is_matrix = len(core_dims) == 2
        has_batch = len(batch_dims) > 0
        if ord in [np.inf, -np.inf] and not is_matrix:
            pytest.skip("Infinity norm not defined for vectors")
        if test_imag and is_matrix and ord == -2:
            pytest.skip("Complex matrices not supported")
        if has_batch and not is_matrix:
            # Handle batched vectors by row-normalizing a matrix
            axis = (-1,)

        rng = np.random.default_rng(utt.fetch_seed())

        if test_imag:
            x_real, x_imag = rng.standard_normal((2, *batch_dims, *core_dims)).astype(
                config.floatX
            )
            dtype = "complex128" if config.floatX.endswith("64") else "complex64"
            X = (x_real + 1j * x_imag).astype(dtype)
        else:
            X = rng.standard_normal(batch_dims + core_dims).astype(config.floatX)

        if batch_dims == ():
            np_norm = np.linalg.norm(X, ord=ord, axis=axis, keepdims=keepdims)
        else:
            np_norm = np.stack(
                [np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims) for x in X]
            )

        pt_norm = norm(X, ord=ord, axis=axis, keepdims=keepdims)
        f = function([], pt_norm, mode="FAST_COMPILE")

        utt.assert_allclose(np_norm, f())
