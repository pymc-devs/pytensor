import numpy as np
from numba.core import types
from numba.core.extending import overload
from numba.core.types import Complex, Float
from numba.np.linalg import _copy_to_fortran_order, ensure_blas, ensure_lapack
from scipy import linalg

from pytensor import config
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.link.numba.dispatch.linalg._BLAS import _BLAS
from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    _get_underlying_float,
    int_ptr_to_val,
    val_to_cptr,
    val_to_dptr,
    val_to_int_ptr,
    val_to_sptr,
    val_to_zptr,
)
from pytensor.link.numba.dispatch.linalg.utils import _check_linalg_matrix
from pytensor.tensor.linalg.products import Expm


@numba_basic.numba_njit(inline="always")
def _poly2_id(c0, A0, c1, A1, id_c, out):
    n = out.shape[0]
    for i in range(n):
        for j in range(n):
            out[i, j] = c0 * A0[i, j] + c1 * A1[i, j]
    for i in range(n):
        out[i, i] += id_c


@numba_basic.numba_njit(inline="always")
def _poly3(c0, A0, c1, A1, c2, A2, out):
    n = out.shape[0]
    for i in range(n):
        for j in range(n):
            out[i, j] = c0 * A0[i, j] + c1 * A1[i, j] + c2 * A2[i, j]


@numba_basic.numba_njit(inline="always")
def _poly3_id(c0, A0, c1, A1, c2, A2, id_c, out):
    n = out.shape[0]
    for i in range(n):
        for j in range(n):
            out[i, j] = c0 * A0[i, j] + c1 * A1[i, j] + c2 * A2[i, j]
    for i in range(n):
        out[i, i] += id_c


@numba_basic.numba_njit(inline="always")
def _poly4_id(c0, A0, c1, A1, c2, A2, c3, A3, id_c, out):
    n = out.shape[0]
    for i in range(n):
        for j in range(n):
            out[i, j] = c0 * A0[i, j] + c1 * A1[i, j] + c2 * A2[i, j] + c3 * A3[i, j]
    for i in range(n):
        out[i, i] += id_c


def _expm(A, overwrite_a=False):
    return linalg.expm(A)


@overload(_expm)
def _expm_impl(A, overwrite_a):
    # Al-Mohy & Higham 2009 Pade scaling-and-squaring (Tables 2.3, 3.1).
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=(Float, Complex), func_name="expm")

    real_dtype = _get_underlying_float(A.dtype)
    is_single = real_dtype == np.float32

    numba_xgetrf = _LAPACK().numba_xgetrf(A.dtype)
    numba_xgetrs = _LAPACK().numba_xgetrs(A.dtype)

    if is_single:
        theta_max = real_dtype.type(3.925724783138660)
        theta_3 = real_dtype.type(4.258730016922831e-01)
        theta_5 = real_dtype.type(1.880152677804762e00)
        theta_7 = real_dtype.type(3.925724783138660)
        theta_9 = real_dtype.type(3.925724783138660)
    else:
        theta_max = real_dtype.type(5.371920351148152)
        theta_3 = real_dtype.type(1.495585217958292e-02)
        theta_5 = real_dtype.type(2.539398330063230e-01)
        theta_7 = real_dtype.type(9.504178996162932e-01)
        theta_9 = real_dtype.type(2.097847961257068e00)

    b3 = tuple(real_dtype.type(x) for x in (120.0, 60.0, 12.0, 1.0))
    b5 = tuple(real_dtype.type(x) for x in (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0))
    b7 = tuple(
        real_dtype.type(x)
        for x in (
            17297280.0,
            8648640.0,
            1995840.0,
            277200.0,
            25200.0,
            1512.0,
            56.0,
            1.0,
        )
    )
    b9 = tuple(
        real_dtype.type(x)
        for x in (
            17643225600.0,
            8821612800.0,
            2075673600.0,
            302702400.0,
            30270240.0,
            2162160.0,
            110880.0,
            3960.0,
            90.0,
            1.0,
        )
    )
    b13 = tuple(
        real_dtype.type(x)
        for x in (
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        )
    )

    def impl(A, overwrite_a):
        n = A.shape[-1]

        A_L1 = np.linalg.norm(A, 1)

        if A_L1 > theta_max:
            s = int(np.ceil(np.log2(A_L1 / theta_max)))
        else:
            s = 0

        # expm(X.T) = expm(X).T -- run the kernel on A.T when A is c-contig so
        # we get an f-contig view of the input buffer for free.
        transposed = False
        if A.flags.c_contiguous:
            A_s = A.T if overwrite_a else A.copy().T
            transposed = True
        elif overwrite_a and A.flags.f_contiguous:
            A_s = A
        else:
            A_s = _copy_to_fortran_order(A)

        A_s = np.asfortranarray(A_s)

        if s > 0:
            A_s /= real_dtype.type(2.0) ** s

        norm_scaled = A_L1 / (real_dtype.type(2.0) ** s)

        dtype = A_s.dtype
        A2 = np.empty((n, n), dtype=dtype)
        np.dot(A_s, A_s, A2)
        U = np.empty((n, n), dtype=dtype)
        V = np.empty((n, n), dtype=dtype)
        S = np.empty((n, n), dtype=dtype)
        T = np.empty((n, n), dtype=dtype).T  # f-contig, consumed by getrs

        if is_single:
            if norm_scaled <= theta_3:
                # U = A_s @ (b3[3]*A2 + b3[1]*I);  V = b3[2]*A2 + b3[0]*I
                np.multiply(b3[3], A2, S)
                for i in range(n):
                    S[i, i] += b3[1]
                np.dot(A_s, S, U)
                np.multiply(b3[2], A2, V)
                for i in range(n):
                    V[i, i] += b3[0]
            elif norm_scaled <= theta_5:
                # U = A_s @ (b5[5]*A4 + b5[3]*A2 + b5[1]*I)
                # V = b5[4]*A4 + b5[2]*A2 + b5[0]*I
                A4 = np.empty((n, n), dtype=dtype)
                np.dot(A2, A2, A4)
                _poly2_id(b5[5], A4, b5[3], A2, b5[1], S)
                np.dot(A_s, S, U)
                _poly2_id(b5[4], A4, b5[2], A2, b5[0], V)
            else:
                # U = A_s @ (b7[7]*A6 + b7[5]*A4 + b7[3]*A2 + b7[1]*I)
                # V =        b7[6]*A6 + b7[4]*A4 + b7[2]*A2 + b7[0]*I
                A4 = np.empty((n, n), dtype=dtype)
                np.dot(A2, A2, A4)
                A6 = np.empty((n, n), dtype=dtype)
                np.dot(A4, A2, A6)
                _poly3_id(b7[7], A6, b7[5], A4, b7[3], A2, b7[1], S)
                np.dot(A_s, S, U)
                _poly3_id(b7[6], A6, b7[4], A4, b7[2], A2, b7[0], V)
        else:
            if norm_scaled <= theta_3:
                # U = A_s @ (b3[3]*A2 + b3[1]*I);  V = b3[2]*A2 + b3[0]*I
                np.multiply(b3[3], A2, S)
                for i in range(n):
                    S[i, i] += b3[1]
                np.dot(A_s, S, U)
                np.multiply(b3[2], A2, V)
                for i in range(n):
                    V[i, i] += b3[0]
            elif norm_scaled <= theta_5:
                # U = A_s @ (b5[5]*A4 + b5[3]*A2 + b5[1]*I)
                # V = b5[4]*A4 + b5[2]*A2 + b5[0]*I
                A4 = np.empty((n, n), dtype=dtype)
                np.dot(A2, A2, A4)
                _poly2_id(b5[5], A4, b5[3], A2, b5[1], S)
                np.dot(A_s, S, U)
                _poly2_id(b5[4], A4, b5[2], A2, b5[0], V)
            elif norm_scaled <= theta_7:
                # U = A_s @ (b7[7]*A6 + b7[5]*A4 + b7[3]*A2 + b7[1]*I)
                # V =        b7[6]*A6 + b7[4]*A4 + b7[2]*A2 + b7[0]*I
                A4 = np.empty((n, n), dtype=dtype)
                np.dot(A2, A2, A4)
                A6 = np.empty((n, n), dtype=dtype)
                np.dot(A4, A2, A6)
                _poly3_id(b7[7], A6, b7[5], A4, b7[3], A2, b7[1], S)
                np.dot(A_s, S, U)
                _poly3_id(b7[6], A6, b7[4], A4, b7[2], A2, b7[0], V)
            elif norm_scaled <= theta_9:
                # U = A_s @ (b9[9]*A8 + b9[7]*A6 + b9[5]*A4 + b9[3]*A2 + b9[1]*I)
                # V =        b9[8]*A8 + b9[6]*A6 + b9[4]*A4 + b9[2]*A2 + b9[0]*I
                A4 = np.empty((n, n), dtype=dtype)
                np.dot(A2, A2, A4)
                A6 = np.empty((n, n), dtype=dtype)
                np.dot(A4, A2, A6)
                A8 = np.empty((n, n), dtype=dtype)
                np.dot(A6, A2, A8)
                _poly4_id(b9[9], A8, b9[7], A6, b9[5], A4, b9[3], A2, b9[1], S)
                np.dot(A_s, S, U)
                _poly4_id(b9[8], A8, b9[6], A6, b9[4], A4, b9[2], A2, b9[0], V)
            else:
                # Pade 13 via Horner (Higham 2005 eqs. 2.2-2.3), so we never
                # form A^8/A^10/A^12 explicitly.
                #   W1 = b13[13]*A6 + b13[11]*A4 + b13[9]*A2
                #   W2 = b13[7]*A6  + b13[5]*A4  + b13[3]*A2 + b13[1]*I
                #   U  = A_s @ (A6 @ W1 + W2)
                #   Z1 = b13[12]*A6 + b13[10]*A4 + b13[8]*A2
                #   Z2 = b13[6]*A6  + b13[4]*A4  + b13[2]*A2 + b13[0]*I
                #   V  = A6 @ Z1 + Z2
                A4 = np.empty((n, n), dtype=dtype)
                np.dot(A2, A2, A4)
                A6 = np.empty((n, n), dtype=dtype)
                np.dot(A4, A2, A6)
                _poly3(b13[13], A6, b13[11], A4, b13[9], A2, S)  # S = W1
                _poly3_id(b13[7], A6, b13[5], A4, b13[3], A2, b13[1], U)  # U = W2
                np.dot(A6, S, V)  # V = A6 @ W1
                V += U  # V = A6 @ W1 + W2
                np.dot(A_s, V, U)  # U = A_s @ V (final U)
                _poly3(b13[12], A6, b13[10], A4, b13[8], A2, S)  # S = Z1
                np.dot(A6, S, V)  # V = A6 @ Z1
                # V += Z2 fused with the np.dot output
                for i in range(n):
                    for j in range(n):
                        V[i, j] += (
                            b13[6] * A6[i, j] + b13[4] * A4[i, j] + b13[2] * A2[i, j]
                        )
                for i in range(n):
                    V[i, i] += b13[0]

        np.add(U, V, T)  # T = P = U + V
        V -= U  # V = Q = V - U

        # Solve Q R = P -> V is c-contig; pass V.T as A and undo with TRANS='T'.
        n_i32 = np.int32(n)
        N_PTR = val_to_int_ptr(n_i32)
        LDA = val_to_int_ptr(n_i32)
        LDB = val_to_int_ptr(n_i32)
        NRHS = val_to_int_ptr(n_i32)
        TRANS = val_to_int_ptr(np.int32(ord("T")))
        INFO_RF = val_to_int_ptr(np.int32(0))
        INFO_RS = val_to_int_ptr(np.int32(0))
        IPIV = np.empty(n, dtype=np.int32)
        V_T = V.T

        numba_xgetrf(N_PTR, N_PTR, V_T.ctypes, LDA, IPIV.ctypes, INFO_RF)
        numba_xgetrs(
            TRANS, N_PTR, NRHS, V_T.ctypes, LDA, IPIV.ctypes, T.ctypes, LDB, INFO_RS
        )

        R = T
        if int_ptr_to_val(INFO_RF) != 0 or int_ptr_to_val(INFO_RS) != 0:
            R[:] = np.nan

        if s > 0:
            A2[:] = R
            R = A2
            R_buf = U
            for _ in range(s):
                np.dot(R, R, R_buf)
                R, R_buf = R_buf, R

        if transposed:
            return R.T
        return R

    return impl


@register_funcify_default_op_cache_key(Expm)
def numba_funcify_Expm(op, node, **kwargs):
    overwrite_a = op.overwrite_a

    inp_dtype = node.inputs[0].type.numpy_dtype
    discrete_input = inp_dtype.kind in "ibu"
    if discrete_input and config.compiler_verbose:
        print("Expm requires casting discrete input to float")  # noqa: T201

    out_dtype = node.outputs[0].type.numpy_dtype
    effective_overwrite_a = overwrite_a or discrete_input

    @numba_basic.numba_njit
    def expm(a):
        if a.size == 0:
            return np.zeros(a.shape, dtype=out_dtype)
        if discrete_input:
            a = a.astype(out_dtype)
        return _expm(a, effective_overwrite_a)

    cache_version = 1
    return expm, cache_version


def _gemm(A, B, C, transa=False, transb=False, alpha=1.0, beta=0.0):
    r"""
    Overwrite ``C`` with :math:`\alpha \, op(A) \, op(B) + \beta C`.

    Parameters
    ----------
    A, B, C : ndarray
        2-d arrays of one dtype. ``C`` is the output and is returned.
    transa, transb : bool, optional
        Read ``A`` or ``B`` transposed. Both default to False.
    alpha, beta : scalar, optional
        Default 1.0 and 0.0. With ``beta == 0`` the incoming ``C`` is not read.

    Returns
    -------
    C : ndarray
    """
    op_a = A.T if transa else A
    op_b = B.T if transb else B
    product = alpha * (op_a @ op_b)
    # An uninitialized C may hold inf or nan bytes, which beta * C would keep.
    C[:] = product if beta == 0 else product + beta * C
    return C


@numba_basic.numba_njit(inline="always")
def _leading_dim(unit_stride, other_stride, unit_extent, other_extent, itemsize):
    """Leading dimension BLAS can walk the matrix with, or 0 when it cannot."""
    if unit_extent != 1 and unit_stride != itemsize:
        return 0
    if other_extent == 1:
        return max(1, unit_extent)
    if other_stride <= 0 or other_stride % itemsize != 0:
        return 0
    ld = other_stride // itemsize
    return ld if ld >= unit_extent else 0


@numba_basic.numba_njit(inline="always")
def _column_major_ld(X):
    return _leading_dim(X.strides[0], X.strides[1], X.shape[0], X.shape[1], X.itemsize)


@numba_basic.numba_njit(inline="always")
def _row_major_ld(X):
    return _leading_dim(X.strides[1], X.strides[0], X.shape[1], X.shape[0], X.itemsize)


@numba_basic.numba_njit(inline="always")
def _blas_operand(X, trans):
    """Buffer, transpose flag and leading dimension for one gemm operand.

    A row-major buffer is its own transpose to BLAS, so its flag flips. A layout
    with no unit-stride axis is copied to C order.
    """
    ld = _column_major_ld(X)
    if ld:
        return X, trans, np.int32(ld)
    ld = _row_major_ld(X)
    if ld:
        return X, not trans, np.int32(ld)
    return np.ascontiguousarray(X), not trans, np.int32(max(1, X.shape[1]))


# Stack slots; a one-element array would cost a heap allocation per call
_SCALAR_PTR_INTRINSICS = {
    types.float32: val_to_sptr,
    types.float64: val_to_dptr,
    types.complex64: val_to_cptr,
    types.complex128: val_to_zptr,
}


@numba_basic.numba_njit(inline="always")
def _blas_vector(x):
    """Logical vector, lowest-address view and signed BLAS increment."""
    if x.size == 1:
        return x, x, np.int32(1)
    if x.strides[0] == 0 or x.strides[0] % x.itemsize:
        x = x.copy()
    inc = np.int32(x.strides[0] // x.itemsize)
    base = x[::-1] if inc < 0 else x
    return x, base, inc


@numba_basic.numba_njit(inline="always")
def _gemv_operands(A, x, y):
    # Reversing a matrix axis also reverses its corresponding vector.
    if A.strides[0] < 0:
        A = A[::-1]
        y = y[::-1]
    if A.strides[1] < 0:
        A = A[:, ::-1]
        x = x[::-1]
    A, trans, ld = _blas_operand(A, False)
    return A, x, y, trans, ld


def _gemv(A, x, y, alpha, beta):
    product = alpha * (A @ x)
    y[:] = product if beta == 0 else product + beta * y
    return y


@overload(_gemv)
def _gemv_impl(A, x, y, alpha, beta):
    ensure_blas()
    _check_linalg_matrix(A, ndim=2, dtype=(Float, Complex), func_name="gemv")
    _check_linalg_matrix(x, ndim=1, dtype=A.dtype, func_name="gemv")
    _check_linalg_matrix(y, ndim=1, dtype=A.dtype, func_name="gemv")
    numba_gemv = _BLAS().numba_xgemv(A.dtype)
    scalar_ptr = _SCALAR_PTR_INTRINSICS[A.dtype]

    if isinstance(A.dtype, Complex):

        @numba_basic.numba_njit(inline="always")
        def vector_dot(a, x):
            # Cython's complex return ABI varies by platform; stay in Numba.
            value = 0j
            for i in range(x.size):
                value += a[i] * x[i]
            return value

    else:
        numba_dot = _BLAS().numba_xdot(A.dtype)

        @numba_basic.numba_njit(inline="always")
        def vector_dot(a, x):
            _, a_base, inca = _blas_vector(a)
            _, x_base, incx = _blas_vector(x)
            return numba_dot(
                val_to_int_ptr(np.int32(x.size)),
                a_base.ctypes,
                val_to_int_ptr(inca),
                x_base.ctypes,
                val_to_int_ptr(incx),
            )

    def impl(A, x, y, alpha, beta):
        if A.shape[1] != x.size or A.shape[0] != y.size:
            raise ValueError("gemv: operands have mismatched dimensions")
        if y.size == 0:
            return y
        if x.size == 0 or alpha == 0:
            for i in range(y.size):
                y[i] = 0 if beta == 0 else beta * y[i]
            return y

        if y.size == 1:
            value = alpha * vector_dot(A[0], x)
            y[0] = value if beta == 0 else value + beta * y[0]
            return y

        A_work, x, y_view, trans, ld = _gemv_operands(A, x, y)
        _, x_base, incx = _blas_vector(x)
        y_work, y_base, incy = _blas_vector(y_view)
        m = np.int32(A_work.shape[1] if trans else A_work.shape[0])
        n = np.int32(A_work.shape[0] if trans else A_work.shape[1])
        numba_gemv(
            val_to_int_ptr(ord("T") if trans else ord("N")),
            val_to_int_ptr(m),
            val_to_int_ptr(n),
            scalar_ptr(alpha),
            A_work.ctypes,
            val_to_int_ptr(ld),
            x_base.ctypes,
            val_to_int_ptr(incx),
            scalar_ptr(beta),
            y_base.ctypes,
            val_to_int_ptr(incy),
        )
        if y_work is not y_view:
            y_view[:] = y_work
        return y

    return impl


@numba_basic.numba_njit(inline="always")
def _blas_abs_layout(X):
    return bool(
        _leading_dim(
            abs(X.strides[0]), abs(X.strides[1]), X.shape[0], X.shape[1], X.itemsize
        )
        or _leading_dim(
            abs(X.strides[1]), abs(X.strides[0]), X.shape[1], X.shape[0], X.itemsize
        )
    )


@numba_basic.numba_njit(inline="always")
def _normalize_gemm_strides(A, B, C, beta):
    ar, ak = A.strides[0] < 0, A.strides[1] < 0
    bk, bc = B.strides[0] < 0, B.strides[1] < 0
    cr, cc = C.strides[0] < 0, C.strides[1] < 0
    if not (ar or ak or bk or bc or cr or cc):
        return A, B, C

    a_layout, b_layout, c_layout = (
        _blas_abs_layout(A),
        _blas_abs_layout(B),
        _blas_abs_layout(C),
    )
    # A contraction reversal cancels against B; free axes cancel against C.
    # Prefer cancelling pairs on ties, then minimize the elements copied. C
    # needs a writeback too, but its incoming contents are unused with beta=0.
    best = int(ar and cr) | (int(ak and bk) << 1) | (int(bc and cc) << 2)
    best_cost = A.size + B.size + 2 * C.size + 1
    for candidate in range(9):
        flips = best if candidate == 0 else candidate - 1
        r, k, c = bool(flips & 1), bool(flips & 2), bool(flips & 4)
        cost = (
            A.size * (not a_layout or ar != r or ak != k)
            + B.size * (not b_layout or bk != k or bc != c)
            + C.size * (1 if beta == 0 else 2) * (not c_layout or cr != r or cc != c)
        )
        if cost < best_cost:
            best, best_cost = flips, cost
            if cost == 0:
                break

    if best & 1:
        A = A[::-1]
        C = C[::-1]
    if best & 2:
        A = A[:, ::-1]
        B = B[::-1]
    if best & 4:
        B = B[:, ::-1]
        C = C[:, ::-1]
    return A, B, C


@overload(_gemm)
def _gemm_impl(A, B, C, transa, transb, alpha, beta):
    ensure_blas()
    _check_linalg_matrix(A, ndim=2, dtype=(Float, Complex), func_name="gemm")
    _check_linalg_matrix(B, ndim=2, dtype=A.dtype, func_name="gemm")
    _check_linalg_matrix(C, ndim=2, dtype=A.dtype, func_name="gemm")

    numba_gemm = _BLAS().numba_xgemm(A.dtype)
    dtype = A.dtype
    scalar_ptr = _SCALAR_PTR_INTRINSICS[dtype]

    def impl(A, B, C, transa, transb, alpha, beta):
        out = C
        if transa:
            A = A.T
        if transb:
            B = B.T
        M = np.int32(A.shape[0])
        K = np.int32(A.shape[1])
        N = np.int32(B.shape[1])

        # BLAS trusts the extents it is handed, so a mismatch here reads past the end
        # of an operand rather than failing.
        if B.shape[0] != K:
            raise ValueError("gemm: operands have mismatched contraction dimensions")
        if C.shape[0] != M or C.shape[1] != N:
            raise ValueError("gemm: output shape does not match the product")

        if M == 0 or N == 0:
            return out
        if K == 0 or alpha == 0:
            for i in range(M):
                for j in range(N):
                    C[i, j] = 0 if beta == 0 else beta * C[i, j]
            return out
        if N == 1:
            _gemv(A, B[:, 0], C[:, 0], alpha, beta)
            return out
        if M == 1:
            _gemv(B.T, A[0], C[0], alpha, beta)
            return out
        if K == 1:
            for i in range(M):
                scaled = alpha * A[i, 0]
                for j in range(N):
                    value = scaled * B[0, j]
                    C[i, j] = value if beta == 0 else value + beta * C[i, j]
            return out

        A, B, C = _normalize_gemm_strides(A, B, C, beta)
        A_work, A_trans, LDA = _blas_operand(A, False)
        B_work, B_trans, LDB = _blas_operand(B, False)

        # A row-major C is C^T to BLAS, and C^T = op(B)^T op(A)^T, so the operands
        # swap places and each flag flips. A C with no unit-stride axis goes through
        # a copy.
        LDC = _column_major_ld(C)
        if LDC:
            C_work = C
            swap = False
        else:
            LDC = _row_major_ld(C)
            swap = True
            if LDC:
                C_work = C
            else:
                C_work = (
                    np.empty(C.shape, dtype=dtype)
                    if beta == 0
                    else np.ascontiguousarray(C)
                )
                LDC = max(1, C.shape[1])

        if swap:
            A_work, B_work = B_work, A_work
            A_trans, B_trans = not B_trans, not A_trans
            LDA, LDB = LDB, LDA
            M, N = N, M

        numba_gemm(
            val_to_int_ptr(ord("T") if A_trans else ord("N")),
            val_to_int_ptr(ord("T") if B_trans else ord("N")),
            val_to_int_ptr(M),
            val_to_int_ptr(N),
            val_to_int_ptr(K),
            scalar_ptr(alpha),
            A_work.ctypes,
            val_to_int_ptr(LDA),
            B_work.ctypes,
            val_to_int_ptr(LDB),
            scalar_ptr(beta),
            C_work.ctypes,
            val_to_int_ptr(np.int32(LDC)),
        )
        if C_work is not C:
            C[:] = C_work
        return out

    return impl
