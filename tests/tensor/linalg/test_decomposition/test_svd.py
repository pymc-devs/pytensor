from functools import partial

import numpy as np
import pytest

import pytensor
from pytensor import function
from pytensor.configdefaults import config
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.linalg import SVD, svd
from pytensor.tensor.type import matrix, tensor
from tests import unittest_tools as utt


class TestSvd(utt.InferShapeTester):
    op_class = SVD

    def setup_method(self):
        super().setup_method()
        self.rng = np.random.default_rng(utt.fetch_seed())
        self.A = matrix(dtype=config.floatX)
        self.op = svd

    @pytest.mark.parametrize(
        "core_shape", [(3, 3), (4, 3), (3, 4)], ids=["square", "tall", "wide"]
    )
    @pytest.mark.parametrize(
        "full_matrix", [True, False], ids=["full=True", "full=False"]
    )
    @pytest.mark.parametrize(
        "compute_uv", [True, False], ids=["compute_uv=True", "compute_uv=False"]
    )
    @pytest.mark.parametrize(
        "batched", [True, False], ids=["batched=True", "batched=False"]
    )
    @pytest.mark.parametrize(
        "test_imag", [True, False], ids=["test_imag=True", "test_imag=False"]
    )
    def test_svd(self, core_shape, full_matrix, compute_uv, batched, test_imag):
        dtype = config.floatX
        if test_imag:
            dtype = "complex128" if dtype.endswith("64") else "complex64"
        shape = core_shape if not batched else (10, *core_shape)
        A = tensor("A", shape=shape, dtype=dtype)
        a = self.rng.random(shape).astype(dtype)

        outputs = svd(A, compute_uv=compute_uv, full_matrices=full_matrix)
        outputs = outputs if isinstance(outputs, list) else [outputs]
        fn = function(inputs=[A], outputs=outputs)

        np_fn = np.vectorize(
            partial(np.linalg.svd, compute_uv=compute_uv, full_matrices=full_matrix),
            signature=outputs[0].owner.op.core_op.gufunc_signature,
        )

        np_outputs = np_fn(a)
        pt_outputs = fn(a)

        np_outputs = np_outputs if isinstance(np_outputs, tuple) else [np_outputs]
        if compute_uv:
            # In this case we sometimes get a sign flip on some columns in one impl and not the thore
            # The results are both correct, and we test that by reconstructing the original input
            U, S, Vh = pt_outputs
            S_diag = np.expand_dims(S, -2) * np.eye(S.shape[-1])

            diff = a.shape[-2] - a.shape[-1]
            if full_matrix:
                if diff > 0:
                    S_diag = np.pad(S_diag, [(0, 0), (0, diff), (0, 0)][-a.ndim :])
                elif diff < 0:
                    S_diag = np.pad(S_diag, [(0, 0), (0, 0), (0, -diff)][-a.ndim :])

            a_r = U @ S_diag @ Vh
            rtol = 1e-3 if config.floatX == "float32" else 1e-7
            np.testing.assert_allclose(a_r, a, rtol=rtol)

            for np_val, pt_val in zip(np_outputs, pt_outputs, strict=True):
                np.testing.assert_allclose(np.abs(np_val), np.abs(pt_val), rtol=rtol)

        else:
            rtol = 1e-5 if config.floatX == "float32" else 1e-7
            for np_val, pt_val in zip(np_outputs, pt_outputs, strict=True):
                np.testing.assert_allclose(np_val, pt_val, rtol=rtol)

    def test_svd_infer_shape(self):
        self.validate_shape((4, 4), full_matrices=True, compute_uv=True)
        self.validate_shape((4, 4), full_matrices=False, compute_uv=True)
        self.validate_shape((2, 4), full_matrices=False, compute_uv=True)
        self.validate_shape((4, 2), full_matrices=False, compute_uv=True)
        self.validate_shape((4, 4), compute_uv=False)

    def validate_shape(self, shape, compute_uv=True, full_matrices=True):
        A = self.A
        A_v = self.rng.random(shape).astype(config.floatX)
        outputs = self.op(A, full_matrices=full_matrices, compute_uv=compute_uv)
        if not compute_uv:
            outputs = [outputs]
        self._compile_and_check([A], outputs, [A_v], self.op_class, warn=False)

    @pytest.mark.parametrize(
        "compute_uv, full_matrices, gradient_test_case",
        [(False, False, 0)]
        + [(True, False, i) for i in range(8)]
        + [(True, True, i) for i in range(8)],
        ids=(
            ["compute_uv=False, full_matrices=False"]
            + [
                f"compute_uv=True, full_matrices=False, gradient={grad}"
                for grad in ["U", "s", "V", "U+s", "s+V", "U+V", "U+s+V", "None"]
            ]
            + [
                f"compute_uv=True, full_matrices=True, gradient={grad}"
                for grad in ["U", "s", "V", "U+s", "s+V", "U+V", "U+s+V", "None"]
            ]
        ),
    )
    @pytest.mark.parametrize(
        "shape", [(3, 3), (4, 3), (3, 4)], ids=["(3,3)", "(4,3)", "(3,4)"]
    )
    @pytest.mark.parametrize(
        "batched", [True, False], ids=["batched=True", "batched=False"]
    )
    def test_grad(self, compute_uv, full_matrices, gradient_test_case, shape, batched):
        rng = np.random.default_rng(utt.fetch_seed())
        if batched:
            shape = (4, *shape)

        A_v = self.rng.normal(size=shape).astype(config.floatX)
        if full_matrices:
            with pytest.raises(
                NotImplementedError,
                match="Gradient of svd not implemented for full_matrices=True",
            ):
                _U, s, _V = svd(
                    self.A, compute_uv=compute_uv, full_matrices=full_matrices
                )
                pytensor.grad(s.sum(), self.A)

        elif compute_uv:

            def svd_fn(A, case=0):
                U, s, V = svd(A, compute_uv=compute_uv, full_matrices=full_matrices)
                if case == 0:
                    return U.sum()
                elif case == 1:
                    return s.sum()
                elif case == 2:
                    return V.sum()
                elif case == 3:
                    return U.sum() + s.sum()
                elif case == 4:
                    return s.sum() + V.sum()
                elif case == 5:
                    return U.sum() + V.sum()
                elif case == 6:
                    return U.sum() + s.sum() + V.sum()
                elif case == 7:
                    return as_tensor_variable(3.0)

            utt.verify_grad(
                partial(svd_fn, case=gradient_test_case),
                [A_v],
                rng=rng,
            )

        else:
            utt.verify_grad(
                partial(svd, compute_uv=compute_uv, full_matrices=full_matrices),
                [A_v],
                rng=rng,
            )
