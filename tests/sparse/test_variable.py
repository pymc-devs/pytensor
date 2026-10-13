from contextlib import ExitStack

import numpy as np
import pytest
import scipy.sparse as scipy_sparse
from scipy.sparse.csr import csr_matrix

import pytensor
import pytensor.sparse as sparse
import pytensor.tensor as pt
from pytensor.sparse.type import SparseTensorType
from pytensor.tensor.type import DenseTensorType


class TestSparseVariable:
    @pytest.mark.parametrize(
        "method",
        [
            "__abs__",
            "__neg__",
            "__ceil__",
            "__floor__",
            "__trunc__",
            "any",
            "all",
            "flatten",
            "ravel",
            "arccos",
            "arcsin",
            "arctan",
            "arccosh",
            "arcsinh",
            "arctanh",
            "ceil",
            "cos",
            "cosh",
            "deg2rad",
            "exp",
            "exp2",
            "expm1",
            "floor",
            "log",
            "log10",
            "log1p",
            "log2",
            "rad2deg",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
            "copy",
            "sum",
            "prod",
            "mean",
            "var",
            "std",
            "min",
            "max",
            "argmin",
            "argmax",
            "nonzero",
            "nonzero_values",
            "argsort",
            "conj",
            "round",
            "trace",
            "zeros_like",
            "ones_like",
            "cumsum",
            "cumprod",
            "ptp",
            "squeeze",
            "diagonal",
        ],
    )
    def test_unary(self, method):
        x = pt.dmatrix("x") if method != "conj" else pt.cmatrix("x")
        x = sparse.csr_from_dense(x)

        method_to_call = getattr(x, method)
        z = method_to_call()

        if not isinstance(z, tuple):
            z_outs = (z,)
        else:
            z_outs = z

        f = pytensor.function(
            [x], z, on_unused_input="ignore", allow_input_downcast=True
        )

        res = f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])

        if not isinstance(res, list):
            res_outs = [res]
        else:
            res_outs = res

        # TODO: Make a separate test for methods that always reduce to dense (only sum for now)
        if getattr(method_to_call, "_is_dense_override", False) or method == "sum":
            assert all(isinstance(out.type, DenseTensorType) for out in z_outs)
            assert all(isinstance(out, np.ndarray) for out in res_outs)
        else:
            assert all(isinstance(out.type, SparseTensorType) for out in z_outs)
            assert all(isinstance(out, csr_matrix) for out in res_outs)

    @pytest.mark.parametrize(
        "method",
        [
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
            "__and__",
            "__or__",
            "__xor__",
            "__add__",
            "__sub__",
            "__mul__",
            "__pow__",
            "__mod__",
            "__divmod__",
            "__truediv__",
            "__floordiv__",
        ],
    )
    def test_binary(self, method):
        x = pt.lmatrix("x")
        y = pt.lmatrix("y")
        x = sparse.csr_from_dense(x)
        y = sparse.csr_from_dense(y)

        method_to_call = getattr(x, method)

        exp_type = (
            DenseTensorType
            if getattr(method_to_call, "_is_dense_override", False)
            else SparseTensorType
        )

        if exp_type == SparseTensorType:
            exp_res_type = csr_matrix
            cm = ExitStack()
        else:
            exp_res_type = np.ndarray
            cm = pytest.warns(UserWarning, match=".*converted to dense.*")

        with cm:
            z = method_to_call(y)

        if not isinstance(z, tuple):
            z_outs = (z,)
        else:
            z_outs = z

        assert all(isinstance(out.type, exp_type) for out in z_outs)

        f = pytensor.function([x, y], z)
        res = f(
            [[1, 0, 2], [-1, 0, 0]],
            [[1, 1, 2], [1, 4, 1]],
        )

        if not isinstance(res, list):
            res_outs = [res]
        else:
            res_outs = res

        assert all(isinstance(out, exp_res_type) for out in res_outs)

    def test_reshape(self):
        x = pt.dmatrix("x")
        x = sparse.csr_from_dense(x)

        with pytest.warns(UserWarning, match=".*converted to dense.*"):
            z = x.reshape((3, 2))

        assert isinstance(z.type, DenseTensorType)

        f = pytensor.function([x], z)
        exp_res = f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])
        assert isinstance(exp_res, np.ndarray)

    def test_dimshuffle(self):
        x = pt.dmatrix("x")
        x = sparse.csr_from_dense(x)

        with pytest.warns(UserWarning, match=".*converted to dense.*"):
            z = x.dimshuffle((1, 0))

        assert isinstance(z.type, DenseTensorType)

        f = pytensor.function([x], z)
        exp_res = f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])
        assert isinstance(exp_res, np.ndarray)

    def test_getitem(self):
        x = pt.dmatrix("x")
        x = sparse.csr_from_dense(x)

        z = x[:, :2]
        assert isinstance(z.type, SparseTensorType)

        f = pytensor.function([x], z)
        exp_res = f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])
        assert isinstance(exp_res, csr_matrix)

    def test_dot(self):
        x = pt.lmatrix("x")
        y = pt.lmatrix("y")
        x = sparse.csr_from_dense(x)
        y = sparse.csr_from_dense(y)

        z = x.__dot__(y)
        assert isinstance(z.type, SparseTensorType)

        f = pytensor.function([x, y], z)
        exp_res = f(
            [[1, 0, 2], [-1, 0, 0]],
            [[-1], [2], [1]],
        )
        assert isinstance(exp_res, csr_matrix)

    @pytest.mark.parametrize("format", ["csr", "csc"])
    @pytest.mark.parametrize(
        "method",
        [
            "dot",
            "tensor_dot",
            "reflected_dot",
            "structured_dot",
            "reflected_structured_dot",
        ],
    )
    def test_dot_vector(self, format, method):
        x = SparseTensorType(format, dtype="float64")("x")
        y = pt.dvector("y")
        weights = pt.dvector("weights")
        reflected = method.startswith("reflected")
        if method == "dot":
            z = x.dot(y)
        elif "structured_dot" in method:
            z = (
                sparse.structured_dot(y, x)
                if reflected
                else sparse.structured_dot(x, y)
            )
        else:
            z = pt.dot(y, x) if reflected else pt.dot(x, y)

        assert z.ndim == 1
        sparse_grad, dense_grad = pt.grad(pt.dot(z, weights), [x, y])
        assert sparse_grad.type == x.type
        f = pytensor.function([x, y, weights], [z, sparse_grad, dense_grad])

        matrix = scipy_sparse.csr_matrix(
            ([1.0, 0.0, 2.0, -1.0], [0, 1, 2, 0], [0, 3, 4]), shape=(2, 3)
        )
        singleton = scipy_sparse.csr_matrix(([0.0], [0], [0, 1]), shape=(1, 1))
        for matrix_value, y_value, weights_value in [
            (matrix, np.array([-1.0, 2.0, 1.0]), np.array([2.0, -3.0])),
            (singleton, np.array([2.0]), np.array([3.0])),
        ]:
            x_value = (matrix_value.T if reflected else matrix_value).asformat(format)
            coordinates = x_value.tocoo()
            if reflected:
                expected = y_value @ x_value
                expected_dense_grad = x_value @ weights_value
                expected_sparse_data = (
                    y_value[coordinates.row] * weights_value[coordinates.col]
                )
            else:
                expected = x_value @ y_value
                expected_dense_grad = x_value.T @ weights_value
                expected_sparse_data = (
                    weights_value[coordinates.row] * y_value[coordinates.col]
                )

            result, sparse_gradient, dense_gradient = f(x_value, y_value, weights_value)
            np.testing.assert_allclose(result, expected)
            np.testing.assert_allclose(dense_gradient, expected_dense_grad)
            assert sparse_gradient.shape == x_value.shape
            np.testing.assert_array_equal(sparse_gradient.indices, x_value.indices)
            np.testing.assert_array_equal(sparse_gradient.indptr, x_value.indptr)
            np.testing.assert_allclose(sparse_gradient.data, expected_sparse_data)

    def test_repeat(self):
        x = pt.dmatrix("x")
        x = sparse.csr_from_dense(x)

        with pytest.warns(UserWarning, match=".*converted to dense.*"):
            z = x.repeat(2, axis=1)

        assert isinstance(z.type, DenseTensorType)

        f = pytensor.function([x], z)
        exp_res = f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])
        assert isinstance(exp_res, np.ndarray)
