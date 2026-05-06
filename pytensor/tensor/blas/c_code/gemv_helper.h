/*
 * GEMV helper functions for PyTensor.
 *
 * This file contains GEMV dispatch logic extracted from Python code generation
 * templates. The goal is to have real C code that IDEs can parse.
 *
 * GEMV computes: z <- beta * y + alpha * dot(A, x)
 * where A is a matrix, x and y are vectors.
 */

#ifndef PYTENSOR_GEMV_HELPER_H
#define PYTENSOR_GEMV_HELPER_H

#include <Python.h>
#include <numpy/arrayobject.h>

/* Include BLAS declarations */
#include "fortran_blas.h"

/*
 * Compute BLAS-compatible strides for a matrix.
 *
 * For row or column matrices, the stride in the dummy dimension doesn't matter,
 * but BLAS requires it to be no smaller than the number of elements.
 */
static inline void pytensor_gemv_compute_matrix_strides(
    int NA0, int NA1,
    npy_intp stride0, npy_intp stride1,
    int elemsize,
    int *SA0, int *SA1
) {
    *SA0 = (NA0 > 1) ? (stride0 / elemsize) : NA1;
    *SA1 = (NA1 > 1) ? (stride1 / elemsize) : NA0;
}

/*
 * Check if a matrix needs to be copied to be BLAS-compatible.
 *
 * Returns 1 if copy needed, 0 if matrix can be used directly.
 * A matrix can be used directly if:
 * - It's C-contiguous (SA1 == 1) or F-contiguous (SA0 == 1)
 * - OR strides are negative but can be handled by reversing iteration
 */
static inline int pytensor_gemv_needs_copy(int SA0, int SA1) {
    /* Can handle negative strides by reversing iteration if one stride is ±1 */
    if ((SA0 < 0 || SA1 < 0) && (abs(SA0) == 1 || abs(SA1) == 1)) {
        return 0;
    }
    /* Otherwise need copy if neither stride is 1 or if strides are negative */
    return (SA0 < 0) || (SA1 < 0) || ((SA0 != 1) && (SA1 != 1));
}

/*
 * Call sgemv_ for float matrix-vector multiply.
 *
 * Handles both C-contiguous and F-contiguous layouts.
 * For C-contiguous (SA1 == 1): uses transpose
 * For F-contiguous (SA0 == 1): no transpose
 *
 * Returns 0 on success, -1 on error.
 */
static inline int pytensor_sgemv_dispatch(
    int NA0, int NA1,
    int SA0, int SA1,
    float *A_data, float *x_data, float *z_data,
    float alpha, float beta,
    int Sx, int Sz
) {
    if (SA0 == 1) {
        /* F-contiguous */
        char NOTRANS = 'N';
        sgemv_(&NOTRANS, &NA0, &NA1,
               &alpha, A_data, &SA1,
               x_data, &Sx,
               &beta, z_data, &Sz);
    } else if (SA1 == 1) {
        /* C-contiguous */
        char TRANS = 'T';
        sgemv_(&TRANS, &NA1, &NA0,
               &alpha, A_data, &SA0,
               x_data, &Sx,
               &beta, z_data, &Sz);
    } else {
        PyErr_SetString(PyExc_AssertionError,
                        "A is neither C nor F-contiguous, it should have been copied");
        return -1;
    }
    return 0;
}

/*
 * Call dgemv_ for double matrix-vector multiply.
 *
 * Handles both C-contiguous and F-contiguous layouts.
 * For C-contiguous (SA1 == 1): uses transpose
 * For F-contiguous (SA0 == 1): no transpose
 *
 * Returns 0 on success, -1 on error.
 */
static inline int pytensor_dgemv_dispatch(
    int NA0, int NA1,
    int SA0, int SA1,
    double *A_data, double *x_data, double *z_data,
    double alpha, double beta,
    int Sx, int Sz
) {
    if (SA0 == 1) {
        /* F-contiguous */
        char NOTRANS = 'N';
        dgemv_(&NOTRANS, &NA0, &NA1,
               &alpha, A_data, &SA1,
               x_data, &Sx,
               &beta, z_data, &Sz);
    } else if (SA1 == 1) {
        /* C-contiguous */
        char TRANS = 'T';
        dgemv_(&TRANS, &NA1, &NA0,
               &alpha, A_data, &SA0,
               x_data, &Sx,
               &beta, z_data, &Sz);
    } else {
        PyErr_SetString(PyExc_AssertionError,
                        "A is neither C nor F-contiguous, it should have been copied");
        return -1;
    }
    return 0;
}

/*
 * Handle vector-vector dot product case (when A has only 1 row).
 *
 * Computes: z[0] = beta * z[0] + alpha * dot(A[0,:], x)
 *
 * This is faster than calling gemv for a single row.
 */
static inline void pytensor_sgemv_dot_case(
    int NA1, int SA1,
    float *A_data, float *x_data, float *z_data,
    float alpha, float beta, int Sx
) {
    z_data[0] = (beta != 0.0f) ? beta * z_data[0] : 0.0f;
    z_data[0] += alpha * sdot_(&NA1, A_data, &SA1, x_data, &Sx);
}

static inline void pytensor_dgemv_dot_case(
    int NA1, int SA1,
    double *A_data, double *x_data, double *z_data,
    double alpha, double beta, int Sx
) {
    z_data[0] = (beta != 0.0) ? beta * z_data[0] : 0.0;
    z_data[0] += alpha * ddot_(&NA1, A_data, &SA1, x_data, &Sx);
}

/*
 * Adjust data pointers and strides for negative stride iteration.
 *
 * When strides are negative but abs(stride) == 1 for one dimension,
 * we can handle this by:
 * 1. Jumping to the "first" element (which is at the end of the array)
 * 2. Negating the stride so we iterate backwards
 * 3. Negating corresponding vector strides
 *
 * This avoids making a copy of the array.
 */
static inline void pytensor_gemv_handle_negative_strides(
    int NA0, int NA1,
    int *SA0, int *SA1,
    int *Sx, int *Sz,
    void **A_data, int elemsize
) {
    char *A = (char*)*A_data;
    if (*SA0 < 0) {
        A += (NA0 - 1) * (*SA0) * elemsize;
        *SA0 = -(*SA0);
        *Sz = -(*Sz);
    }
    if (*SA1 < 0) {
        A += (NA1 - 1) * (*SA1) * elemsize;
        *SA1 = -(*SA1);
        *Sx = -(*Sx);
    }
    *A_data = A;
}

#endif /* PYTENSOR_GEMV_HELPER_H */

