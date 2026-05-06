/*
 * Complete GER operation for PyTensor.
 *
 * This file contains a top-level GER function that handles:
 * - Input validation (rank, dtype, shape checks)
 * - Output allocation (with proper reference counting)
 * - Computation dispatch (manual loops or BLAS)
 *
 * This enables minimal Python codegen - just a single function call.
 */

#ifndef PYTENSOR_GER_OP_H
#define PYTENSOR_GER_OP_H

#include <Python.h>
#include <numpy/arrayobject.h>

/* Include the helper functions */
#include "ger_helper.h"

/*
 * Perform complete GER operation: Z = A + alpha * outer(x, y)
 *
 * Parameters:
 *   A          - Input matrix (2D)
 *   a          - Scalar alpha (0D)
 *   x          - Input vector (1D), length must match A.shape[0]
 *   y          - Input vector (1D), length must match A.shape[1]
 *   Z_ptr      - Pointer to output array pointer (will be set/updated)
 *   destructive - If true and A is contiguous, operate in-place on A
 *
 * Returns:
 *   0 on success, -1 on error (Python exception set)
 *
 * Note: Caller is responsible for reference counting. If *Z_ptr is changed,
 * the old value is DECREF'd and the new value is returned with a new reference.
 */
static int pytensor_ger(
    PyArrayObject *A,
    PyArrayObject *a,
    PyArrayObject *x,
    PyArrayObject *y,
    PyArrayObject **Z_ptr,
    int destructive
) {
    int elemsize;
    npy_intp dims[2];
    PyArrayObject *Z = *Z_ptr;

    /* Validate ranks */
    if (PyArray_NDIM(A) != 2) {
        PyErr_SetString(PyExc_NotImplementedError, "rank(A) != 2");
        return -1;
    }
    if (PyArray_NDIM(x) != 1) {
        PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 1");
        return -1;
    }
    if (PyArray_NDIM(y) != 1) {
        PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 1");
        return -1;
    }
    if (PyArray_NDIM(a) != 0) {
        PyErr_SetString(PyExc_NotImplementedError, "rank(a) != 0");
        return -1;
    }

    /* Validate dtypes match */
    if (PyArray_DESCR(A)->type_num != PyArray_DESCR(x)->type_num) {
        PyErr_SetString(PyExc_TypeError, "A vs. x dtype mismatch");
        return -1;
    }
    if (PyArray_DESCR(A)->type_num != PyArray_DESCR(y)->type_num) {
        PyErr_SetString(PyExc_TypeError, "A vs. y dtype mismatch");
        return -1;
    }

    /* Validate shapes */
    if (PyArray_DIMS(A)[0] != PyArray_DIMS(x)[0]) {
        PyErr_SetString(PyExc_ValueError, "Shape mismatch: A.shape[0] != x.shape[0]");
        return -1;
    }
    if (PyArray_DIMS(A)[1] != PyArray_DIMS(y)[0]) {
        PyErr_SetString(PyExc_ValueError, "Shape mismatch: A.shape[1] != y.shape[0]");
        return -1;
    }

    /* Determine element size */
    if (PyArray_DESCR(A)->type_num == NPY_DOUBLE) {
        elemsize = 8;
    } else if (PyArray_DESCR(A)->type_num == NPY_FLOAT) {
        elemsize = 4;
    } else {
        PyErr_SetString(PyExc_NotImplementedError, "complex CGer not implemented");
        return -1;
    }

    dims[0] = PyArray_DIMS(A)[0];
    dims[1] = PyArray_DIMS(A)[1];

    /* Decide: copy A or operate in-place */
    if (!destructive || pytensor_ger_needs_copy(
            PyArray_STRIDES(A)[0], PyArray_STRIDES(A)[1], elemsize)) {
        /*
         * Need to copy: either non-destructive mode or A has bad strides.
         * Allocate Z if needed, then copy A into Z and compute.
         */
        int need_alloc = (Z == NULL)
            || (PyArray_DIMS(Z)[0] != dims[0])
            || (PyArray_DIMS(Z)[1] != dims[1])
            || pytensor_ger_needs_copy(
                   PyArray_STRIDES(Z)[0], PyArray_STRIDES(Z)[1], elemsize);

        if (need_alloc) {
            Py_XDECREF(Z);
            Z = (PyArrayObject *)PyArray_SimpleNew(2, dims, PyArray_TYPE(A));
            if (!Z) {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc ger output");
                return -1;
            }
            *Z_ptr = Z;
        }

        if (Z == A) {
            PyErr_SetString(PyExc_AssertionError, "Z should not be A in copy path");
            return -1;
        }

        /* Copy A to Z and add outer product */
        if (PyArray_DESCR(Z)->type_num == NPY_FLOAT) {
            const float *zdata = (float *)PyArray_DATA(A);
            float *zoutdata = (float *)PyArray_DATA(Z);
            const float *xdata = (float *)PyArray_DATA(x);
            const float *ydata = (float *)PyArray_DATA(y);
            float alpha = ((float *)PyArray_DATA(a))[0];
            int Ai = PyArray_STRIDES(A)[0] / sizeof(float);
            int Aj = PyArray_STRIDES(A)[1] / sizeof(float);
            int Zi = PyArray_STRIDES(Z)[0] / sizeof(float);
            int Zj = PyArray_STRIDES(Z)[1] / sizeof(float);
            int xi = PyArray_STRIDES(x)[0] / sizeof(float);
            int yj = PyArray_STRIDES(y)[0] / sizeof(float);
            pytensor_sger_manual_copy(dims[0], dims[1],
                zdata, Ai, Aj, zoutdata, Zi, Zj,
                xdata, xi, ydata, yj, alpha);
        } else {
            const double *zdata = (double *)PyArray_DATA(A);
            double *zoutdata = (double *)PyArray_DATA(Z);
            const double *xdata = (double *)PyArray_DATA(x);
            const double *ydata = (double *)PyArray_DATA(y);
            double alpha = ((double *)PyArray_DATA(a))[0];
            int Ai = PyArray_STRIDES(A)[0] / sizeof(double);
            int Aj = PyArray_STRIDES(A)[1] / sizeof(double);
            int Zi = PyArray_STRIDES(Z)[0] / sizeof(double);
            int Zj = PyArray_STRIDES(Z)[1] / sizeof(double);
            int xi = PyArray_STRIDES(x)[0] / sizeof(double);
            int yj = PyArray_STRIDES(y)[0] / sizeof(double);
            pytensor_dger_manual_copy(dims[0], dims[1],
                zdata, Ai, Aj, zoutdata, Zi, Zj,
                xdata, xi, ydata, yj, alpha);
        }
    } else {
        /*
         * Destructive mode with good strides: operate in-place on A.
         */
        if (Z != A) {
            Py_XDECREF(Z);
            Z = A;
            Py_INCREF(Z);
            *Z_ptr = Z;
        }

        if ((dims[0] * dims[1]) < PYTENSOR_GER_BLAS_THRESHOLD) {
            /* Small matrix: use manual loop */
            if (PyArray_DESCR(Z)->type_num == NPY_FLOAT) {
                float *zoutdata = (float *)PyArray_DATA(Z);
                const float *xdata = (float *)PyArray_DATA(x);
                const float *ydata = (float *)PyArray_DATA(y);
                float alpha = ((float *)PyArray_DATA(a))[0];
                int Zi = PyArray_STRIDES(Z)[0] / sizeof(float);
                int Zj = PyArray_STRIDES(Z)[1] / sizeof(float);
                int xi = PyArray_STRIDES(x)[0] / sizeof(float);
                int yj = PyArray_STRIDES(y)[0] / sizeof(float);
                pytensor_sger_manual_inplace(dims[0], dims[1],
                    zoutdata, Zi, Zj, xdata, xi, ydata, yj, alpha);
            } else {
                double *zoutdata = (double *)PyArray_DATA(Z);
                const double *xdata = (double *)PyArray_DATA(x);
                const double *ydata = (double *)PyArray_DATA(y);
                double alpha = ((double *)PyArray_DATA(a))[0];
                int Zi = PyArray_STRIDES(Z)[0] / sizeof(double);
                int Zj = PyArray_STRIDES(Z)[1] / sizeof(double);
                int xi = PyArray_STRIDES(x)[0] / sizeof(double);
                int yj = PyArray_STRIDES(y)[0] / sizeof(double);
                pytensor_dger_manual_inplace(dims[0], dims[1],
                    zoutdata, Zi, Zj, xdata, xi, ydata, yj, alpha);
            }
        } else {
            /* Large matrix: use BLAS */
            int Nz0 = dims[0];
            int Nz1 = dims[1];
            int Sx = PyArray_STRIDES(x)[0] / elemsize;
            int Sy = PyArray_STRIDES(y)[0] / elemsize;

            /* Handle negative strides */
            void *x_data = PyArray_DATA(x);
            void *y_data = PyArray_DATA(y);
            if (Sx < 0) {
                x_data = (char *)x_data + (Nz0 - 1) * Sx * elemsize;
            }
            if (Sy < 0) {
                y_data = (char *)y_data + (Nz1 - 1) * Sy * elemsize;
            }

            if (PyArray_DESCR(Z)->type_num == NPY_FLOAT) {
                float alpha = ((float *)PyArray_DATA(a))[0];
                if (pytensor_sger_dispatch(Nz0, Nz1,
                        PyArray_STRIDES(Z)[0], PyArray_STRIDES(Z)[1], elemsize,
                        (float *)PyArray_DATA(Z), (float *)x_data, (float *)y_data,
                        alpha, Sx, Sy) != 0) {
                    return -1;
                }
            } else {
                double alpha = ((double *)PyArray_DATA(a))[0];
                if (pytensor_dger_dispatch(Nz0, Nz1,
                        PyArray_STRIDES(Z)[0], PyArray_STRIDES(Z)[1], elemsize,
                        (double *)PyArray_DATA(Z), (double *)x_data, (double *)y_data,
                        alpha, Sx, Sy) != 0) {
                    return -1;
                }
            }
        }
    }

    return 0;
}

#endif /* PYTENSOR_GER_OP_H */

