import numpy as np
from numba.core.extending import overload
from numba.core.types import Complex, Float
from numba.np.linalg import _copy_to_fortran_order, ensure_lapack
from scipy import linalg

from pytensor import config
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    _get_underlying_float,
    int_ptr_to_val,
    val_to_int_ptr,
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
    effective_overwrite_a = overwrite_a and not discrete_input

    @numba_basic.numba_njit
    def expm(a):
        if a.size == 0:
            return np.zeros(a.shape, dtype=out_dtype)
        if discrete_input:
            a = a.astype(out_dtype)
        return _expm(a, effective_overwrite_a)

    cache_version = 1
    return expm, cache_version
