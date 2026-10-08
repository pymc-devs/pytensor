import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import In, config, function
from pytensor.link.numba.dispatch.basic import numba_njit
from pytensor.link.numba.dispatch.linalg.products import (
    _blas_operand,
    _blas_vector,
    _gemm,
    _gemv_operands,
    _normalize_gemm_strides,
)
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


@numba_njit(final_function=True)
def _gemm_jit(A, B, C, transa, transb, alpha, beta):
    return _gemm(A, B, C, transa, transb, alpha, beta)


@numba_njit(final_function=True)
def _gemv_buffers_jit(A, x, y):
    A, x, y, _, _ = _gemv_operands(A, x, y)
    x, x_base, incx = _blas_vector(x)
    y, y_base, incy = _blas_vector(y)
    return A, x, x_base, y, y_base, incx, incy


@numba_njit(final_function=True)
def _gemm_buffers_jit(A, B, C, beta=0.3):
    A, B, C = _normalize_gemm_strides(A, B, C, beta)
    A, _, _ = _blas_operand(A, False)
    B, _, _ = _blas_operand(B, False)
    C, _, _ = _blas_operand(C, False)
    return A, B, C


@pytest.mark.parametrize("poison", [np.inf, np.nan], ids=["inf", "nan"])
@pytest.mark.parametrize(
    "shape", [(3, 2, 4), (3, 2, 1), (1, 2, 4), (1, 2, 1), (3, 1, 4)]
)
def test_gemm_ignores_C_when_beta_is_zero(poison, shape):
    """`BatchedDot` allocates C with `np.empty` and passes beta=0, so C holds whatever the
    allocator returned. BLAS does not read C in that case and neither may the reference, or stray
    non-finite bytes would multiply by zero into nan."""
    m, k, n = shape
    A = np.ascontiguousarray(rng.normal(size=(m, k)))
    B = np.ascontiguousarray(rng.normal(size=(k, n)))
    uninitialized = np.full((m, n), poison)

    np.testing.assert_allclose(
        _gemm(A, B, uninitialized.copy(), False, False, 1.0, 0.0), A @ B
    )
    np.testing.assert_allclose(
        _gemm_jit(A, B, uninitialized.copy(), False, False, 1.0, 0.0), A @ B
    )


def _matrix_views(rows, cols):
    wide = rng.normal(size=(rows, 2 * cols))
    full = np.ascontiguousarray(wide[:, :cols])
    return {
        "column_slice": wide[:, :cols],
        "row_step": np.ascontiguousarray(np.vstack([full, full]))[::2],
        "reversed_rows": full[::-1],
        "reversed_columns": full[:, ::-1],
        "reversed_both": full[::-1, ::-1],
        "fortran_reversed_rows": np.asfortranarray(full)[::-1],
    }


@pytest.mark.parametrize(
    "layout",
    [
        "column_slice",
        "row_step",
        "reversed_rows",
        "reversed_columns",
        "reversed_both",
        "fortran_reversed_rows",
    ],
)
def test_gemm_reads_strided_operands(layout):
    A = _matrix_views(6, 5)[layout]
    B = _matrix_views(5, 4)[layout]
    C = _matrix_views(6, 4)[layout]
    expected = 0.5 * C + 2.0 * (A @ B)

    np.testing.assert_allclose(_gemm_jit(A, B, C, False, False, 2.0, 0.5), expected)


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("transpose", [False, True], ids=["matvec", "vecmat"])
def test_gemm_vector_strides(dtype, transpose):
    matrix = rng.normal(size=(4, 3))
    x_storage = rng.normal(size=6)
    y_storage = rng.normal(size=8)
    if np.dtype(dtype).kind == "c":
        matrix = matrix + 1j * rng.normal(size=matrix.shape)
        x_storage = x_storage + 1j * rng.normal(size=x_storage.shape)
        y_storage = y_storage + 1j * rng.normal(size=y_storage.shape)
    matrix = matrix.astype(dtype)
    x_storage = x_storage.astype(dtype)
    y_storage = y_storage.astype(dtype)
    rtol = 1e-5 if np.dtype(dtype).char in "fF" else 1e-10

    for order in ["C", "F"]:
        # Retain a unit-stride axis, but leave gaps on the other axis.
        axis = 0 if order == "C" else 1
        padded = np.array(np.repeat(matrix, 2, axis=axis), order=order)
        padded = padded[::2] if axis == 0 else padded[:, ::2]
        for row_step, col_step in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            A = padded[::row_step, ::col_step]
            for step in [2, -2]:
                x = x_storage[::step]
                y = y_storage.copy()[::step]
                expected = 1.7 * (A @ x) + 0.3 * y
                if transpose:
                    result = _gemm_jit(x[None], A, y[None], False, True, 1.7, 0.3)
                    assert np.shares_memory(result, y)
                    result = result[0]
                else:
                    result = _gemm_jit(
                        A, x[:, None], y[:, None], False, False, 1.7, 0.3
                    )[:, 0]
                    assert np.shares_memory(result, y)
                np.testing.assert_allclose(result, expected, rtol=rtol, atol=1e-6)

    # Dot is unconjugated, including when both vectors have non-unit strides.
    a = x_storage[::2][None]
    b = x_storage[::-2][:, None]
    out = np.full((1, 1), np.nan, dtype=dtype)
    expected = 1.7 * (a @ b)
    np.testing.assert_allclose(
        _gemm_jit(a, b, out, False, False, 1.7, 0.0),
        expected,
        rtol=rtol,
        atol=1e-6,
    )
    a, b = b, a
    out = np.full((6, 6), np.nan, dtype=dtype)[::-2, ::2]
    np.testing.assert_allclose(
        _gemm_jit(a, b, out, False, False, 1.7, 0.0),
        1.7 * (a @ b),
        rtol=rtol,
        atol=1e-6,
    )


def test_gemv_keeps_operand_buffers():
    matrix = rng.normal(size=(4, 3))
    x = rng.normal(size=6)[::-2]
    y = rng.normal(size=8)[::-2]
    layouts = [
        np.array(matrix, order="C"),
        np.array(matrix, order="F"),
        np.repeat(matrix, 2, axis=0)[::2],
        np.asfortranarray(np.repeat(matrix, 2, axis=1))[:, ::2],
    ]
    for A in layouts:
        for row_step, col_step in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            view = A[::row_step, ::col_step]
            work, x_view, x_base, y_view, y_base, incx, incy = _gemv_buffers_jit(
                view, x, y
            )
            assert np.shares_memory(work, A)
            for logical, base, inc, original in [
                (x_view, x_base, incx, x),
                (y_view, y_base, incy, y),
            ]:
                assert np.shares_memory(logical, original)
                assert np.shares_memory(base, original)
                assert abs(inc) == 2


def test_gemm_cancels_reversals():
    A_base = rng.normal(size=(4, 3))
    B_base = rng.normal(size=(3, 5))
    C_base = rng.normal(size=(4, 5))
    for row_step in [1, -1]:
        for inner_step in [1, -1]:
            for col_step in [1, -1]:
                A = A_base[::row_step, ::inner_step]
                B = B_base[::inner_step, ::col_step]
                C = C_base[::row_step, ::col_step]
                for work, original in zip(
                    _gemm_buffers_jit(A, B, C), (A, B, C), strict=True
                ):
                    assert np.shares_memory(work, original)
                for transa, transb in [
                    (False, False),
                    (True, False),
                    (False, True),
                    (True, True),
                ]:
                    expected = 1.7 * (A @ B) + 0.3 * C
                    result = _gemm_jit(
                        A.T if transa else A,
                        B.T if transb else B,
                        C,
                        transa,
                        transb,
                        1.7,
                        0.3,
                    )
                    assert np.shares_memory(result, C)
                    np.testing.assert_allclose(result, expected)


def test_gemm_moves_copy_to_smaller_operand():
    # Move an unmatched contraction reversal from the large A to the small B.
    A = rng.normal(size=(20, 3))[:, ::-1]
    B = rng.normal(size=(3, 2))
    C = rng.normal(size=(20, 2))
    work_a, work_b, work_c = _gemm_buffers_jit(A, B, C)
    assert np.shares_memory(work_a, A)
    assert not np.shares_memory(work_b, B)
    assert np.shares_memory(work_c, C)
    expected = 1.7 * (A @ B) + 0.3 * C
    np.testing.assert_allclose(_gemm_jit(A, B, C, False, False, 1.7, 0.3), expected)

    # Both matrices can be read forwards when a smaller output is reversed.
    A = rng.normal(size=(3, 20))[::-1, ::-1]
    B = rng.normal(size=(20, 2))[::-1, ::-1]
    C = rng.normal(size=(3, 2))
    work_a, work_b, work_c = _gemm_buffers_jit(A, B, C)
    assert np.shares_memory(work_a, A)
    assert np.shares_memory(work_b, B)
    assert not np.shares_memory(work_c, C)
    expected = 1.7 * (A @ B) + 0.3 * C
    np.testing.assert_allclose(_gemm_jit(A, B, C, False, False, 1.7, 0.3), expected)


@pytest.mark.parametrize(
    "shape", [(0, 3, 4), (4, 3, 0), (4, 0, 3), (4, 0, 1), (1, 0, 4), (1, 0, 1)]
)
def test_gemm_empty(shape):
    m, k, n = shape
    A = np.empty((m, k))
    B = np.empty((k, n))
    C = np.full((m, n), np.nan)
    np.testing.assert_array_equal(
        _gemm_jit(A, B, C, False, False, 1.0, 0.0), np.zeros((m, n))
    )


@pytest.mark.parametrize("operand", [1, 2], ids=["B", "C"])
def test_gemm_requires_matching_dtypes(operand):
    arrays = [np.ones((3, 2)), np.ones((2, 4)), np.ones((3, 4))]
    arrays[operand] = arrays[operand].astype("float32")
    with pytest.raises(numba.TypingError, match="gemm only supported for float64"):
        _gemm_jit(*arrays, False, False, 1.0, 0.0)


@pytest.mark.parametrize("batched", [False, True])
def test_dot_vector_strides(batched):
    A = pt.tensor("A", shape=(None,) * (3 if batched else 2))
    B = A.type("B")
    x = pt.tensor("x", shape=(None,) * (2 if batched else 1))
    if batched:
        outputs = [
            pt.matmul(A, x[..., None])[..., 0],
            pt.matmul(x[..., None, :], B)[..., 0, :],
        ]
    else:
        outputs = [pt.dot(A, x), pt.dot(x, B)]
    fn = function([A, x, B], outputs, mode="NUMBA")
    values = rng.normal(size=(2, 4, 3)).astype(config.floatX)
    vectors = rng.normal(size=(2, 6)).astype(config.floatX)
    layouts = [
        values,
        values.swapaxes(-1, -2).copy().swapaxes(-1, -2),
        np.repeat(values, 2, axis=-2)[..., ::2, :],
    ]
    for layout in layouts:
        for row_step, col_step in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            for step in [2, -2]:
                a = layout[..., ::row_step, ::col_step]
                v = vectors[..., ::step]
                if not batched:
                    a, v = a[0], v[0]
                expected = (a @ v[..., None])[..., 0]
                results = fn(a, v, a.swapaxes(-1, -2))
                for result in results:
                    np.testing.assert_allclose(
                        result,
                        expected,
                        rtol=1e-5 if config.floatX == "float32" else 1e-10,
                        atol=1e-6 if config.floatX == "float32" else 1e-10,
                    )
