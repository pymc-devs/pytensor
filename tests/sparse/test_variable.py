from contextlib import ExitStack

import numpy as np
import pytest
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

    def test_repeat(self):
        x = pt.dmatrix("x")
        x = sparse.csr_from_dense(x)

        with pytest.warns(UserWarning, match=".*converted to dense.*"):
            z = x.repeat(2, axis=1)

        assert isinstance(z.type, DenseTensorType)

        f = pytensor.function([x], z)
        exp_res = f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])
        assert isinstance(exp_res, np.ndarray)
