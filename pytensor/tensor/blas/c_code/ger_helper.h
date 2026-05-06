/*
 * GER helper functions for PyTensor.
 *
 * This file contains GER (rank-1 update) dispatch logic extracted from
 * Python code generation templates.
 *
 * GER computes: A <- A + alpha * x * y^T
 * where A is a matrix and x, y are vectors.
 */

#ifndef PYTENSOR_GER_HELPER_H
#define PYTENSOR_GER_HELPER_H

#include <Python.h>
#include <numpy/arrayobject.h>

/* Include BLAS declarations */
#include "fortran_blas.h"

/*
 * Check if a matrix needs to be copied for GER.
 *
 * GER requires the matrix to have at least one unit stride.
 * Returns 1 if copy needed, 0 if matrix can be used directly.
 */
static inline int pytensor_ger_needs_copy(npy_intp stride0, npy_intp stride1, int elemsize) {
    return (stride0 < 0) || (stride1 < 0) ||
           ((stride0 != elemsize) && (stride1 != elemsize));
}

/*
 * Manual float GER with copy: Z = A + alpha * x * y^T
 *
 * Used when A needs to be copied (non-contiguous or non-destructive).
 * Reads from A (zdata), writes to Z (zoutdata).
 */
static inline void pytensor_sger_manual_copy(
    int dims0, int dims1,
    const float *zdata, int Ai, int Aj,
    float *zoutdata, int Zi, int Zj,
    const float *xdata, int xi,
    const float *ydata, int yj,
    float alpha
) {
    for (int i = 0; i < dims0; ++i) {
        float xx = alpha * xdata[xi * i];
        for (int j = 0; j < dims1; ++j) {
            float tmp = zdata[Ai*i + Aj*j];
            tmp += xx * ydata[yj * j];
            zoutdata[Zi*i + Zj*j] = tmp;
        }
    }
}

/*
 * Manual double GER with copy: Z = A + alpha * x * y^T
 */
static inline void pytensor_dger_manual_copy(
    int dims0, int dims1,
    const double *zdata, int Ai, int Aj,
    double *zoutdata, int Zi, int Zj,
    const double *xdata, int xi,
    const double *ydata, int yj,
    double alpha
) {
    for (int i = 0; i < dims0; ++i) {
        double xx = alpha * xdata[xi * i];
        for (int j = 0; j < dims1; ++j) {
            double tmp = zdata[Ai*i + Aj*j];
            tmp += xx * ydata[yj * j];
            zoutdata[Zi*i + Zj*j] = tmp;
        }
    }
}

/*
 * Manual float GER inplace: Z += alpha * x * y^T
 *
 * Used for small matrices where calling BLAS has overhead.
 */
static inline void pytensor_sger_manual_inplace(
    int dims0, int dims1,
    float *zoutdata, int Zi, int Zj,
    const float *xdata, int xi,
    const float *ydata, int yj,
    float alpha
) {
    for (int i = 0; i < dims0; ++i) {
        float axi = alpha * xdata[xi * i];
        for (int j = 0; j < dims1; ++j) {
            zoutdata[Zi*i + Zj*j] += axi * ydata[yj * j];
        }
    }
}

/*
 * Manual double GER inplace: Z += alpha * x * y^T
 */
static inline void pytensor_dger_manual_inplace(
    int dims0, int dims1,
    double *zoutdata, int Zi, int Zj,
    const double *xdata, int xi,
    const double *ydata, int yj,
    double alpha
) {
    for (int i = 0; i < dims0; ++i) {
        double axi = alpha * xdata[xi * i];
        for (int j = 0; j < dims1; ++j) {
            zoutdata[Zi*i + Zj*j] += axi * ydata[yj * j];
        }
    }
}

/*
 * Call sger_ for float rank-1 update.
 *
 * Handles both C-contiguous and F-contiguous layouts.
 * For F-contiguous (stride[0] == elemsize): sger_(Nz0, Nz1, ..., x, y, ...)
 * For C-contiguous (stride[1] == elemsize): sger_(Nz1, Nz0, ..., y, x, ...)
 *
 * Returns 0 on success, -1 on error.
 */
static inline int pytensor_sger_dispatch(
    int Nz0, int Nz1,
    npy_intp stride0, npy_intp stride1,
    int elemsize,
    float *z_data, float *x_data, float *y_data,
    float alpha, int Sx, int Sy
) {
    /* Compute BLAS-compatible strides */
    int Sz0 = (Nz0 > 1) ? (stride0 / elemsize) : (Nz1 + 1);
    int Sz1 = (Nz1 > 1) ? (stride1 / elemsize) : (Nz0 + 1);

    if (stride0 == elemsize) {
        /* F-contiguous */
        sger_(&Nz0, &Nz1, &alpha, x_data, &Sx, y_data, &Sy, z_data, &Sz1);
    } else if (stride1 == elemsize) {
        /* C-contiguous: swap dimensions and vectors */
        sger_(&Nz1, &Nz0, &alpha, y_data, &Sy, x_data, &Sx, z_data, &Sz0);
    } else {
        PyErr_SetString(PyExc_AssertionError,
            "A is a double-strided matrix, and should have been copied "
            "into a memory-contiguous one.");
        return -1;
    }
    return 0;
}

/*
 * Call dger_ for double rank-1 update.
 *
 * Returns 0 on success, -1 on error.
 */
static inline int pytensor_dger_dispatch(
    int Nz0, int Nz1,
    npy_intp stride0, npy_intp stride1,
    int elemsize,
    double *z_data, double *x_data, double *y_data,
    double alpha, int Sx, int Sy
) {
    /* Compute BLAS-compatible strides */
    int Sz0 = (Nz0 > 1) ? (stride0 / elemsize) : (Nz1 + 1);
    int Sz1 = (Nz1 > 1) ? (stride1 / elemsize) : (Nz0 + 1);

    if (stride0 == elemsize) {
        /* F-contiguous */
        dger_(&Nz0, &Nz1, &alpha, x_data, &Sx, y_data, &Sy, z_data, &Sz1);
    } else if (stride1 == elemsize) {
        /* C-contiguous: swap dimensions and vectors */
        dger_(&Nz1, &Nz0, &alpha, y_data, &Sy, x_data, &Sx, z_data, &Sz0);
    } else {
        PyErr_SetString(PyExc_AssertionError,
            "A is a double-strided matrix, and should have been copied "
            "into a memory-contiguous one.");
        return -1;
    }
    return 0;
}

/*
 * Threshold for using manual loop vs BLAS GER.
 * For small matrices, the overhead of calling BLAS is not worth it.
 */
#define PYTENSOR_GER_BLAS_THRESHOLD 100000

#endif /* PYTENSOR_GER_HELPER_H */

