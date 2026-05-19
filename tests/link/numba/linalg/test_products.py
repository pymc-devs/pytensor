import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import In, config
from pytensor.tensor.linalg.products import Expm, expm
from tests.link.numba.test_basic import compare_numba_and_py, numba_inplace_mode


pytestmark = [
    pytest.mark.filterwarnings("error"),
    pytest.mark.filterwarnings("ignore::numba.core.errors.NumbaPerformanceWarning"),
]

numba = pytest.importorskip("numba")

floatX = config.floatX

rng = np.random.default_rng(42849)


class TestExpm:
    @pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
    @pytest.mark.parametrize(
        "overwrite_a", [False, True], ids=["no_overwrite", "overwrite_a"]
    )
    def test_expm(self, overwrite_a: bool, dtype: str):
        A = pt.matrix("A", dtype=dtype)
        y = Expm(overwrite_a=overwrite_a)(A)

        x = rng.normal(size=(4, 4)) * 5.0
        if np.dtype(dtype).kind == "c":
            x = x + 1j * rng.normal(size=(4, 4)) * 5.0
        val = x.astype(dtype)
        rtol = 1e-3 if np.dtype(dtype).char in "fF" else 1e-10

        def assert_fn(actual, expected):
            np.testing.assert_allclose(actual, expected, rtol=rtol)

        fn, res = compare_numba_and_py(
            [In(A, mutable=overwrite_a)],
            [y],
            [val],
            numba_mode=numba_inplace_mode,
            inplace=True,
            assert_fn=assert_fn,
        )

        op = fn.maker.fgraph.outputs[0].owner.op
        assert isinstance(op, Expm)
        assert overwrite_a == (op.destroy_map == {0: [0]})

        # F-contiguous input is mutated when overwrite_a=True (kernel uses
        # A's buffer directly as scratch during scaling).
        val_f_contig = np.copy(val, order="F")
        res_f_contig = fn(val_f_contig)
        np.testing.assert_allclose(res_f_contig, res, rtol=rtol)
        assert (val == val_f_contig).all() == (not overwrite_a)

        # C-contiguous input is also mutated when overwrite_a=True: the kernel
        # takes A.T (f-contig view of A's buffer) and computes expm(A.T) =
        # expm(A).T, scaling A's buffer in place along the way.
        val_c_contig = np.copy(val, order="C")
        res_c_contig = fn(val_c_contig)
        np.testing.assert_allclose(res_c_contig, res, rtol=rtol)
        assert (val == val_c_contig).all() == (not overwrite_a)

        # Non-contiguous (strided) input is also never mutated.
        val_not_contig = np.repeat(val, 2, axis=0)[::2]
        res_not_contig = fn(val_not_contig)
        np.testing.assert_allclose(res_not_contig, res, rtol=rtol)
        np.testing.assert_allclose(val_not_contig, val)

    def test_expm_size_zero(self):
        A = pt.matrix("A", dtype=floatX)
        y = expm(A)
        compare_numba_and_py([A], [y], [np.zeros((0, 0), dtype=floatX)])

    def test_expm_integer_input(self):
        A = pt.matrix("A", dtype="int64")
        y = expm(A)
        assert y.type.dtype == "float64"

        val = rng.integers(-2, 3, size=(4, 4)).astype("int64")
        original = val.copy()
        _, res = compare_numba_and_py([A], [y], [val])
        np.testing.assert_array_equal(val, original)
        assert res[0].dtype == np.float64
