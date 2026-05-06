/*
 * GEMM helper functions for PyTensor.
 *
 * This file contains the core GEMM dispatch logic extracted from the
 * Python code generation templates. The goal is to have real C code
 * that IDEs can parse, with minimal dynamic parts.
 */

#ifndef PYTENSOR_GEMM_HELPER_H
#define PYTENSOR_GEMM_HELPER_H

#include <Python.h>
#include <numpy/arrayobject.h>

/* Include BLAS declarations */
#include "fortran_blas.h"

#ifndef MOD
#define MOD %
#endif

/*
 * Compute strides for a contiguous array.
 * Used when PyArray_STRIDES returns invalid values (e.g., for size-0 arrays).
 */
static inline void compute_strides(npy_intp *shape, int ndim, int type_size, npy_intp *res) {
    res[ndim - 1] = type_size;
    for (int i = ndim - 1; i > 0; i--) {
        npy_intp s = shape[i];
        res[i - 1] = res[i] * (s > 0 ? s : 1);
    }
}

/*
 * Encode the stride structure of three 2D arrays into a single integer.
 *
 * For each array, we encode:
 *   0x0 = row-major (last stride == type_size) or single column
 *   0x1 = column-major (first stride == type_size) or single row
 *   0x2 = neither (will trigger error)
 *
 * The encoding is: (x_code << 8) | (y_code << 4) | (z_code << 0)
 */
static inline int pytensor_encode_gemm_strides(
    npy_intp *Nx, npy_intp *Sx,
    npy_intp *Ny, npy_intp *Sy,
    npy_intp *Nz, npy_intp *Sz,
    int type_size
) {
    int unit = 0;
    unit |= ((Sx[1] == type_size || Nx[1] == 1) ? 0x0 : (Sx[0] == type_size || Nx[0] == 1) ? 0x1 : 0x2) << 8;
    unit |= ((Sy[1] == type_size || Ny[1] == 1) ? 0x0 : (Sy[0] == type_size || Ny[0] == 1) ? 0x1 : 0x2) << 4;
    unit |= ((Sz[1] == type_size || Nz[1] == 1) ? 0x0 : (Sz[0] == type_size || Nz[0] == 1) ? 0x1 : 0x2) << 0;
    return unit;
}

/*
 * Compute BLAS-compatible strides from NumPy strides.
 *
 * BLAS requires leading dimensions to be >= 1 and not smaller than
 * the number of elements in that dimension. For vectors or empty
 * matrices, we need to compute valid dummy strides.
 */
static inline void pytensor_compute_gemm_strides(
    npy_intp *Nx, npy_intp *Sx, int *sx_0, int *sx_1,
    npy_intp *Ny, npy_intp *Sy, int *sy_0, int *sy_1,
    npy_intp *Nz, npy_intp *Sz, int *sz_0, int *sz_1,
    int type_size
) {
    *sx_0 = (Nx[0] > 1) ? Sx[0] / type_size : (Nx[1] + 1);
    *sx_1 = (Nx[1] > 1) ? Sx[1] / type_size : (Nx[0] + 1);
    *sy_0 = (Ny[0] > 1) ? Sy[0] / type_size : (Ny[1] + 1);
    *sy_1 = (Ny[1] > 1) ? Sy[1] / type_size : (Ny[0] + 1);
    *sz_0 = (Nz[0] > 1) ? Sz[0] / type_size : (Nz[1] + 1);
    *sz_1 = (Nz[1] > 1) ? Sz[1] / type_size : (Nz[0] + 1);
}

/*
 * Call sgemm_ with the appropriate transpose flags based on stride encoding.
 *
 * Returns 0 on success, -1 on error (with Python exception set).
 */
static inline int pytensor_sgemm_dispatch(
    int unit,
    float *x, float *y, float *z,
    float a, float b,
    int Nz0, int Nz1, int Nx1,
    int sx_0, int sx_1, int sy_0, int sy_1, int sz_0, int sz_1
) {
    char N = 'N';
    char T = 'T';

    switch (unit) {
        case 0x000: sgemm_(&N, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_0, &b, z, &sz_0); break;
        case 0x100: sgemm_(&N, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_1, &b, z, &sz_0); break;
        case 0x010: sgemm_(&T, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_0, &b, z, &sz_0); break;
        case 0x110: sgemm_(&T, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_1, &b, z, &sz_0); break;
        case 0x001: sgemm_(&T, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_0, &b, z, &sz_1); break;
        case 0x101: sgemm_(&N, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_0, &b, z, &sz_1); break;
        case 0x011: sgemm_(&T, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_1, &b, z, &sz_1); break;
        case 0x111: sgemm_(&N, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_1, &b, z, &sz_1); break;
        default:
            PyErr_SetString(PyExc_ValueError, "some matrix has no unit stride");
            return -1;
    }
    return 0;
}

/*
 * Call dgemm_ with the appropriate transpose flags based on stride encoding.
 *
 * Returns 0 on success, -1 on error (with Python exception set).
 */
static inline int pytensor_dgemm_dispatch(
    int unit,
    double *x, double *y, double *z,
    double a, double b,
    int Nz0, int Nz1, int Nx1,
    int sx_0, int sx_1, int sy_0, int sy_1, int sz_0, int sz_1
) {
    char N = 'N';
    char T = 'T';

    switch (unit) {
        case 0x000: dgemm_(&N, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_0, &b, z, &sz_0); break;
        case 0x100: dgemm_(&N, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_1, &b, z, &sz_0); break;
        case 0x010: dgemm_(&T, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_0, &b, z, &sz_0); break;
        case 0x110: dgemm_(&T, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_1, &b, z, &sz_0); break;
        case 0x001: dgemm_(&T, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_0, &b, z, &sz_1); break;
        case 0x101: dgemm_(&N, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_0, &b, z, &sz_1); break;
        case 0x011: dgemm_(&T, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_1, &b, z, &sz_1); break;
        case 0x111: dgemm_(&N, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_1, &b, z, &sz_1); break;
        default:
            PyErr_SetString(PyExc_ValueError, "some matrix has no unit stride");
            return -1;
    }
    return 0;
}

/*
 * Check if an array needs to be copied to make it BLAS-compatible.
 *
 * BLAS requires arrays to have at least one unit stride and valid
 * (non-negative, properly aligned) strides.
 */
static inline int pytensor_needs_copy_for_blas(npy_intp *N, npy_intp *S, int type_size) {
    return (S[0] < 1) || (S[1] < 1)
        || (S[0] MOD type_size) || (S[1] MOD type_size)
        || ((S[0] != type_size) && (S[1] != type_size));
}

/*
 * Ensure an array is BLAS-compatible, copying if necessary.
 *
 * If the array needs copying, *arr is updated to point to the copy
 * and *S is updated to the new strides. The caller must handle
 * reference counting appropriately.
 *
 * Returns 0 on success, -1 on error (with Python exception set).
 */
static inline int pytensor_ensure_blas_compatible(
    PyArrayObject **arr, npy_intp *N, npy_intp **S, int type_size
) {
    if (pytensor_needs_copy_for_blas(N, *S, type_size)) {
        PyArrayObject *copy = (PyArrayObject *)PyArray_Copy(*arr);
        if (!copy) {
            return -1;
        }
        Py_DECREF(*arr);
        *arr = copy;
        *S = PyArray_STRIDES(*arr);
        if ((*S)[0] < 1 || (*S)[1] < 1) {
            compute_strides(N, 2, type_size, *S);
        }
    }
    return 0;
}

#endif /* PYTENSOR_GEMM_HELPER_H */

