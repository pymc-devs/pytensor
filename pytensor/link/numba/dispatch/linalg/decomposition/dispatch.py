import warnings

import numba
import numpy as np

from pytensor import config
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    get_numba_type,
    register_funcify_default_op_cache_key,
)
from pytensor.link.numba.dispatch.linalg.decomposition.cholesky import _cholesky
from pytensor.link.numba.dispatch.linalg.decomposition.lu import (
    _lu_1,
    _lu_2,
    _lu_3,
    _pivot_to_permutation,
)
from pytensor.link.numba.dispatch.linalg.decomposition.lu_factor import _lu_factor
from pytensor.link.numba.dispatch.linalg.decomposition.qr import (
    _qr_full_no_pivot,
    _qr_full_pivot,
    _qr_r_no_pivot,
    _qr_r_pivot,
    _qr_raw_no_pivot,
    _qr_raw_pivot,
)
from pytensor.link.numba.dispatch.linalg.decomposition.qz import (
    _qz_complex_nosort_eig,
    _qz_complex_nosort_noeig,
    _qz_complex_sort_eig,
    _qz_complex_sort_noeig,
    _qz_real_nosort_eig,
    _qz_real_nosort_noeig,
    _qz_real_sort_eig,
    _qz_real_sort_noeig,
)
from pytensor.link.numba.dispatch.linalg.decomposition.schur import (
    schur_complex,
    schur_real,
)
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.decomposition.eigen import Eig, Eigh
from pytensor.tensor.linalg.decomposition.lu import LU, LUFactor, PivotToPermutations
from pytensor.tensor.linalg.decomposition.qr import QR
from pytensor.tensor.linalg.decomposition.schur import QZ, Schur
from pytensor.tensor.linalg.decomposition.svd import SVD


@register_funcify_default_op_cache_key(SVD)
def numba_funcify_SVD(op, node, **kwargs):
    full_matrices = op.full_matrices
    compute_uv = op.compute_uv
    out_dtype = np.dtype(node.outputs[0].dtype)

    discrete_input = node.inputs[0].type.numpy_dtype.kind in "ibu"
    if discrete_input and config.compiler_verbose:
        print("SVD requires casting discrete input to float")  # noqa: T201

    # np.linalg.svd always returns real-valued singular values, even for complex input.
    # The Op may declare s as complex (matching input dtype), but numba returns the real
    # component dtype, so we must match that to avoid type unification errors.
    matrix_dtype = out_dtype
    if out_dtype.kind == "c":
        s_dtype = np.dtype(f"f{out_dtype.itemsize // 2}")
    else:
        s_dtype = out_dtype

    if not compute_uv:

        @numba_basic.numba_njit
        def svd(x):
            if x.size == 0:
                m, n = x.shape
                k = min(m, n)
                return np.zeros((k,), dtype=s_dtype)
            if discrete_input:
                x = x.astype(out_dtype)
            _, ret, _ = np.linalg.svd(x, full_matrices)
            return ret

    else:

        @numba_basic.numba_njit
        def svd(x):
            if x.size == 0:
                m, n = x.shape
                k = min(m, n)
                # The LAPACK dispatch returns matrices in fortran order. To match this for the empty cases,
                # build flip the shape inputs to np.zeros and transpose.
                if full_matrices:
                    return (
                        np.zeros((m, m), dtype=matrix_dtype).T,
                        np.zeros((k,), dtype=s_dtype),
                        np.zeros((n, n), dtype=matrix_dtype).T,
                    )
                else:
                    return (
                        np.zeros((k, m), dtype=matrix_dtype).T,
                        np.zeros((k,), dtype=s_dtype),
                        np.zeros((n, k), dtype=matrix_dtype).T,
                    )
            if discrete_input:
                x = x.astype(out_dtype)
            return np.linalg.svd(x, full_matrices)

    cache_version = 1
    return svd, cache_version


@register_funcify_default_op_cache_key(Eig)
def numba_funcify_Eig(op, node, **kwargs):
    w_dtype = node.outputs[0].type.numpy_dtype
    non_complex_input = node.inputs[0].type.numpy_dtype.kind != "c"
    if non_complex_input and config.compiler_verbose:
        print("Eig requires casting input to complex")  # noqa: T201

    @numba_basic.numba_njit
    def eig(x):
        if non_complex_input:
            # Even floats are better cast to complex, otherwise numba may raise
            # ValueError: eig() argument must not cause a domain change.
            x = x.astype(w_dtype)
        w, v = np.linalg.eig(x)
        return w.astype(w_dtype), v.astype(w_dtype)

    cache_version = 2
    return eig, cache_version


@register_funcify_default_op_cache_key(Eigh)
def numba_funcify_Eigh(op, node, **kwargs):
    uplo = op.UPLO

    if uplo != "L":
        warnings.warn(
            (
                "Numba will use object mode to allow the "
                "`UPLO` argument to `numpy.linalg.eigh`."
            ),
            UserWarning,
        )

        out_dtypes = tuple(o.type.numpy_dtype for o in node.outputs)
        ret_sig = numba.types.Tuple(
            [get_numba_type(node.outputs[0].type), get_numba_type(node.outputs[1].type)]
        )

        @numba_basic.numba_njit
        def eigh(x):
            with numba.objmode(ret=ret_sig):
                out = np.linalg.eigh(x, UPLO=uplo)
                ret = (out[0].astype(out_dtypes[0]), out[1].astype(out_dtypes[1]))
            return ret

    else:

        @numba_basic.numba_njit
        def eigh(x):
            return np.linalg.eigh(x)

    return eigh


@register_funcify_default_op_cache_key(Cholesky)
def numba_funcify_Cholesky(op, node, **kwargs):
    """
    Overload scipy.linalg.cholesky with a numba function.

    Note that np.linalg.cholesky is already implemented in numba, but it does not support additional keyword arguments.
    In particular, the `inplace` argument is not supported, which is why we choose to implement our own version.
    """
    lower = op.lower
    overwrite_a = op.overwrite_a

    inp_dtype = node.inputs[0].type.numpy_dtype
    discrete_inp = inp_dtype.kind in "ibu"
    if discrete_inp and config.compiler_verbose:
        print("Cholesky requires casting discrete input to float")  # noqa: T201

    out_dtype = node.outputs[0].type.numpy_dtype

    @numba_basic.numba_njit
    def cholesky(a):
        if a.size == 0:
            return np.zeros(a.shape, dtype=out_dtype)

        if discrete_inp:
            a = a.astype(out_dtype)

        return _cholesky(a, lower, overwrite_a)

    cache_version = 2
    return cholesky, cache_version


@register_funcify_default_op_cache_key(PivotToPermutations)
def pivot_to_permutation(op, node, **kwargs):
    inverse = op.inverse

    @numba_basic.numba_njit
    def numba_pivot_to_permutation(piv):
        p_inv = _pivot_to_permutation(piv)

        if inverse:
            return p_inv

        return np.argsort(p_inv)

    cache_version = 2
    return numba_pivot_to_permutation, cache_version


@register_funcify_default_op_cache_key(LU)
def numba_funcify_LU(op, node, **kwargs):
    inp_dtype = node.inputs[0].type.numpy_dtype
    if inp_dtype.kind == "c":
        return generate_fallback_impl(op, node=node, **kwargs)
    discrete_inp = inp_dtype.kind in "ibu"
    if discrete_inp and config.compiler_verbose:
        print("LU requires casting discrete input to float")  # noqa: T201

    out_dtype = node.outputs[0].type.numpy_dtype
    permute_l = op.permute_l
    p_indices = op.p_indices
    overwrite_a = op.overwrite_a

    @numba_basic.numba_njit
    def lu(a):
        if a.size == 0:
            L = np.zeros(a.shape, dtype=a.dtype)
            U = np.zeros(a.shape, dtype=a.dtype)
            if permute_l:
                return L, U
            elif p_indices:
                P = np.zeros(a.shape[0], dtype="int32")
                return P, L, U
            else:
                P = np.zeros(a.shape, dtype=a.dtype)
                return P, L, U

        if discrete_inp:
            a = a.astype(out_dtype)

        if p_indices:
            res = _lu_1(
                a,
                permute_l=permute_l,
                p_indices=p_indices,
                overwrite_a=overwrite_a,
            )
        elif permute_l:
            res = _lu_2(
                a,
                permute_l=permute_l,
                p_indices=p_indices,
                overwrite_a=overwrite_a,
            )
        else:
            res = _lu_3(
                a,
                permute_l=permute_l,
                p_indices=p_indices,
                overwrite_a=overwrite_a,
            )

        return res

    cache_version = 2
    return lu, cache_version


@register_funcify_default_op_cache_key(LUFactor)
def numba_funcify_LUFactor(op, node, **kwargs):
    inp_dtype = node.inputs[0].type.numpy_dtype
    discrete_inp = inp_dtype.kind in "ibu"
    if discrete_inp and config.compiler_verbose:
        print("LUFactor requires casting discrete input to float")  # noqa: T201

    out_dtype = node.outputs[0].type.numpy_dtype
    overwrite_a = op.overwrite_a

    @numba_basic.numba_njit
    def lu_factor(a):
        if a.size == 0:
            return (
                np.zeros(a.shape, dtype=out_dtype),
                np.zeros(a.shape[0], dtype="int32"),
            )

        if discrete_inp:
            a = a.astype(out_dtype)

        LU, piv = _lu_factor(a, overwrite_a)

        return LU, piv

    cache_version = 3
    return lu_factor, cache_version


@register_funcify_default_op_cache_key(QR)
def numba_funcify_QR(op, node, **kwargs):
    mode = op.mode
    pivoting = op.pivoting
    overwrite_a = op.overwrite_a

    in_dtype = node.inputs[0].type.numpy_dtype
    integer_input = in_dtype.kind in "ibu"
    if integer_input and config.compiler_verbose:
        print("QR requires casting discrete input to float")  # noqa: T201

    out_dtype = node.outputs[0].type.numpy_dtype

    @numba_basic.numba_njit
    def qr(a):
        if a.size == 0:
            m, n = a.shape
            k = min(m, n)
            if (mode == "full" or mode == "economic") and pivoting:
                Q = np.zeros(
                    (k, m) if mode == "economic" else (m, m), dtype=out_dtype
                ).T
                R = np.zeros((k, n) if mode == "economic" else (m, n), dtype=out_dtype)
                P = np.zeros(n, dtype=np.int32)
                return Q, R, P
            elif (mode == "full" or mode == "economic") and not pivoting:
                Q = np.zeros(
                    (k, m) if mode == "economic" else (m, m), dtype=out_dtype
                ).T
                R = np.zeros((k, n) if mode == "economic" else (m, n), dtype=out_dtype)
                return Q, R
            elif mode == "r" and pivoting:
                R = np.zeros((m, n), dtype=out_dtype)
                P = np.zeros(n, dtype=np.int32)
                return R, P
            elif mode == "r" and not pivoting:
                R = np.zeros((m, n), dtype=out_dtype)
                return R
            elif mode == "raw" and pivoting:
                H = np.zeros((m, m), dtype=out_dtype)
                tau = np.zeros(k, dtype=out_dtype)
                R = np.zeros((m, n), dtype=out_dtype)
                P = np.zeros(n, dtype=np.int32)
                return H, tau, R, P
            else:
                H = np.zeros((m, m), dtype=out_dtype)
                tau = np.zeros(k, dtype=out_dtype)
                R = np.zeros((m, n), dtype=out_dtype)
                return H, tau, R

        if integer_input:
            a = a.astype(out_dtype)

        if (mode == "full" or mode == "economic") and pivoting:
            Q, R, P = _qr_full_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
            )
            return Q, R, P

        elif (mode == "full" or mode == "economic") and not pivoting:
            Q, R = _qr_full_no_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
            )
            return Q, R

        elif mode == "r" and pivoting:
            R, P = _qr_r_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
            )
            return R, P

        elif mode == "r" and not pivoting:
            (R,) = _qr_r_no_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
            )
            return R

        elif mode == "raw" and pivoting:
            H, tau, R, P = _qr_raw_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
            )
            return H, tau, R, P

        elif mode == "raw" and not pivoting:
            H, tau, R = _qr_raw_no_pivot(
                a,
                mode=mode,
                pivoting=pivoting,
                overwrite_a=overwrite_a,
            )
            return H, tau, R

        else:
            raise NotImplementedError(
                f"QR mode={mode}, pivoting={pivoting} not supported in numba mode."
            )

    cache_version = 2
    return qr, cache_version


@register_funcify_default_op_cache_key(Schur)
def numba_funcify_Schur(op, node, **kwargs):
    output = op.output
    overwrite_a = op.overwrite_a
    sort = op.sort

    if sort is not None:
        if config.compiler_verbose:
            print(  # noqa: T201
                "Schur is not implemented in numba mode when `sort` is not None, "
                "falling back to object mode"
            )
        return generate_fallback_impl(op, node=node, **kwargs)

    in_dtype = node.inputs[0].type.numpy_dtype
    out_dtype = node.outputs[0].type.numpy_dtype
    integer_input = in_dtype.kind in "ibu"
    complex_input = in_dtype.kind in "cz"
    needs_complex_cast = in_dtype.kind in "fd" and output == "complex"

    # Disable overwrite_a for dtype conversion (real->complex upcast)
    if needs_complex_cast:
        overwrite_a = False
        if config.compiler_verbose:
            print(  # noqa: T201
                "Schur: disabling overwrite_a due to dtype conversion (casting prevents in-place operation)"
            )

    if integer_input and config.compiler_verbose:
        print("Schur requires casting discrete input to float")  # noqa: T201

    # Complex input always produces complex output, and output == "complex" forces complex output
    if complex_input or output == "complex":

        @numba_basic.numba_njit
        def schur(a):
            if a.size == 0:
                n = a.shape[0]
                return np.zeros((n, n), dtype=out_dtype), np.zeros(
                    (n, n), dtype=out_dtype
                )
            if integer_input:
                a = a.astype(out_dtype)
            elif needs_complex_cast:
                a = a.astype(out_dtype)
            T, Z = schur_complex(a, lwork=None, overwrite_a=overwrite_a)
            return T, Z
    else:
        # Real input with real output
        @numba_basic.numba_njit
        def schur(a):
            if a.size == 0:
                n = a.shape[0]
                return np.zeros((n, n), dtype=out_dtype), np.zeros(
                    (n, n), dtype=out_dtype
                )
            if integer_input:
                a = a.astype(out_dtype)
            T, Z = schur_real(a, lwork=None, overwrite_a=overwrite_a)
            return T, Z

    cache_version = 1
    return schur, cache_version


@register_funcify_default_op_cache_key(QZ)
def numba_funcify_QZ(op, node, **kwargs):
    complex_output = op.complex_output
    sort = op.sort
    return_eigenvalues = op.return_eigenvalues
    overwrite_a = op.overwrite_a
    overwrite_b = op.overwrite_b

    in_dtype_a = node.inputs[0].type.numpy_dtype
    in_dtype_b = node.inputs[1].type.numpy_dtype
    out_dtype = node.outputs[0].type.numpy_dtype

    integer_input_a = in_dtype_a.kind in "ibu"
    integer_input_b = in_dtype_b.kind in "ibu"
    complex_input = in_dtype_a.kind == "c" or in_dtype_b.kind == "c"
    needs_complex_cast = (
        in_dtype_a.kind in "fd" or in_dtype_b.kind in "fd"
    ) and complex_output

    # Disable overwrite for dtype conversion (real->complex upcast)
    if needs_complex_cast:
        overwrite_a = False
        overwrite_b = False
        if config.compiler_verbose:
            print(  # noqa: T201
                "QZ: disabling overwrite_a/b due to dtype conversion (casting prevents in-place operation)"
            )

    if (integer_input_a or integer_input_b) and config.compiler_verbose:
        print("QZ requires casting discrete input to float")  # noqa: T201

    alpha_dtype = node.outputs[2].type.numpy_dtype if return_eigenvalues else out_dtype

    use_complex = complex_input or complex_output
    use_sort = sort is not None

    if use_complex:
        if use_sort:
            if return_eigenvalues:
                qz_fn = _qz_complex_sort_eig
            else:
                qz_fn = _qz_complex_sort_noeig
        else:
            if return_eigenvalues:
                qz_fn = _qz_complex_nosort_eig
            else:
                qz_fn = _qz_complex_nosort_noeig
    else:
        if use_sort:
            if return_eigenvalues:
                qz_fn = _qz_real_sort_eig
            else:
                qz_fn = _qz_real_sort_noeig
        else:
            if return_eigenvalues:
                qz_fn = _qz_real_nosort_eig
            else:
                qz_fn = _qz_real_nosort_noeig

    if use_sort:

        @numba_basic.numba_njit
        def qz(a, b):
            if a.size == 0:
                n = a.shape[0]
                if return_eigenvalues:
                    return (
                        np.zeros((n, n), dtype=out_dtype),
                        np.zeros((n, n), dtype=out_dtype),
                        np.zeros((n,), dtype=alpha_dtype),
                        np.zeros((n,), dtype=out_dtype),
                        np.zeros((n, n), dtype=out_dtype),
                        np.zeros((n, n), dtype=out_dtype),
                    )
                return (
                    np.zeros((n, n), dtype=out_dtype),
                    np.zeros((n, n), dtype=out_dtype),
                    np.zeros((n, n), dtype=out_dtype),
                    np.zeros((n, n), dtype=out_dtype),
                )
            if integer_input_a:
                a = a.astype(out_dtype)
            elif needs_complex_cast:
                a = a.astype(out_dtype)
            if integer_input_b:
                b = b.astype(out_dtype)
            elif needs_complex_cast:
                b = b.astype(out_dtype)
            return qz_fn(a, b, sort, overwrite_a, overwrite_b)
    else:

        @numba_basic.numba_njit
        def qz(a, b):
            if a.size == 0:
                n = a.shape[0]
                if return_eigenvalues:
                    return (
                        np.zeros((n, n), dtype=out_dtype),
                        np.zeros((n, n), dtype=out_dtype),
                        np.zeros((n,), dtype=alpha_dtype),
                        np.zeros((n,), dtype=out_dtype),
                        np.zeros((n, n), dtype=out_dtype),
                        np.zeros((n, n), dtype=out_dtype),
                    )
                return (
                    np.zeros((n, n), dtype=out_dtype),
                    np.zeros((n, n), dtype=out_dtype),
                    np.zeros((n, n), dtype=out_dtype),
                    np.zeros((n, n), dtype=out_dtype),
                )
            if integer_input_a:
                a = a.astype(out_dtype)
            elif needs_complex_cast:
                a = a.astype(out_dtype)
            if integer_input_b:
                b = b.astype(out_dtype)
            elif needs_complex_cast:
                b = b.astype(out_dtype)
            return qz_fn(a, b, overwrite_a, overwrite_b)

    cache_version = 1
    return qz, cache_version
