from pytensor.link.c.op import COp
from pytensor.link.c.params_type import ParamsType
from pytensor.scalar import bool as bool_t
from pytensor.tensor.blas import (
    Gemv,
    Ger,
    blas_header_text,
    blas_header_version,
    ldflags,
)


class BaseBLAS(COp):
    def c_libraries(self, **kwargs):
        return ldflags()

    def c_compile_args(self, **kwargs):
        return ldflags(libs=False, flags=True)

    def c_lib_dirs(self, **kwargs):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self, **kwargs):
        return ldflags(libs=False, include_dir=True)

    def c_support_code(self, **kwargs):
        return blas_header_text()


# ##### ####### #######
# GER
# ##### ####### #######


def ger_c_code(A, a, x, y, Z, fail, params):
    return f"""

    int elemsize ;

    if (PyArray_NDIM({A}) != 2)
    {{PyErr_SetString(PyExc_NotImplementedError, "rank(A) != 2"); {fail};}}
    if (PyArray_NDIM({x}) != 1)
    {{PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 1"); {fail};}}
    if (PyArray_NDIM({y}) != 1)
    {{PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 1"); {fail};}}
    if (PyArray_NDIM({a}) != 0)
    {{PyErr_SetString(PyExc_NotImplementedError, "rank(a) != 0"); {fail};}}

    if (PyArray_DESCR({A})->type_num != PyArray_DESCR({x})->type_num)
    {{ PyErr_SetString(PyExc_TypeError, "A vs. x"); {fail}; }}
    if (PyArray_DESCR({A})->type_num != PyArray_DESCR({y})->type_num)
    {{ PyErr_SetString(PyExc_TypeError, "A vs. y"); {fail}; }}

    if (PyArray_DIMS({A})[0] != PyArray_DIMS({x})[0])
    {{
        PyErr_SetString(PyExc_ValueError,
                        "Shape mismatch: A.shape[0] != x.shape[0]");
        {fail};
    }}
    if (PyArray_DIMS({A})[1] != PyArray_DIMS({y})[0])
    {{
        PyErr_SetString(PyExc_ValueError,
                        "Shape mismatch: A.shape[1] != y.shape[0]");
        {fail};
    }}

    if  (PyArray_DESCR({A})->type_num == NPY_DOUBLE) {{ elemsize = 8; }}
    else if (PyArray_DESCR({A})->type_num == NPY_FLOAT) {{ elemsize = 4;}}
    else
    {{
        PyErr_SetString(PyExc_NotImplementedError, "complex CGer");
        {fail};
    }}

    // copy A if !self.destructive or A is fully strided
    if (!{params}->destructive
        || (PyArray_STRIDES({A})[0] < 0)
        || (PyArray_STRIDES({A})[1] < 0)
        || ((PyArray_STRIDES({A})[0] != elemsize)
            && (PyArray_STRIDES({A})[1] != elemsize)))
    {{
        npy_intp dims[2];
        dims[0] = PyArray_DIMS({A})[0];
        dims[1] = PyArray_DIMS({A})[1];

        if ((NULL == {Z})
            || (PyArray_DIMS({Z})[0] != PyArray_DIMS({A})[0])
            || (PyArray_DIMS({Z})[1] != PyArray_DIMS({A})[1])
            || (PyArray_STRIDES({Z})[0] < 0)
            || (PyArray_STRIDES({Z})[1] < 0)
            || ((PyArray_STRIDES({Z})[0] != elemsize)
                && (PyArray_STRIDES({Z})[1] != elemsize)))
        {{
            Py_XDECREF({Z});
            {Z} = (PyArrayObject*) PyArray_SimpleNew(2, dims,
                                                       PyArray_TYPE({A}));
            if(!{Z}) {{
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc ger output");
                {fail}
            }}
        }}
        if ({Z} == {A})
        {{
            PyErr_SetString(PyExc_AssertionError, "{Z} != {A}");
            {fail}
        }}
        if (PyArray_DESCR({Z})->type_num == NPY_FLOAT)
        {{
            float * zoutdata = (float*)PyArray_DATA({Z});
            const float * zdata = (float*)PyArray_DATA({A});
            const float * xdata = (float*)PyArray_DATA({x});
            const float * ydata = (float*)PyArray_DATA({y});
            const float * adata = (float*)PyArray_DATA({a});
            const float alpha = adata[0];
            float tmp, xx;
            int Ai = PyArray_STRIDES({A})[0]/sizeof(float);
            int Aj = PyArray_STRIDES({A})[1]/sizeof(float);
            int Zi = PyArray_STRIDES({Z})[0]/sizeof(float);
            int Zj = PyArray_STRIDES({Z})[1]/sizeof(float);
            int xi = PyArray_STRIDES({x})[0]/sizeof(float);
            int yj = PyArray_STRIDES({y})[0]/sizeof(float);
            for (int i = 0; i < dims[0]; ++i)
            {{
                xx = alpha * xdata[xi * i];
                for (int j = 0; j < dims[1]; ++j)
                {{
                    tmp = zdata[Ai*i+Aj*j];
                    tmp += xx * ydata[yj * j];
                    zoutdata[Zi*i+Zj*j] = tmp;
                }}
            }}
        }}
        else if (PyArray_DESCR({Z})->type_num == NPY_DOUBLE)
        {{
            double * zoutdata = (double*) PyArray_DATA({Z});
            const double * zdata = (double*)PyArray_DATA({A});
            const double * xdata = (double*)PyArray_DATA({x});
            const double * ydata = (double*)PyArray_DATA({y});
            const double * adata = (double*)PyArray_DATA({a});
            const double alpha = adata[0];
            double tmp, xx;

            int Ai = PyArray_STRIDES({A})[0]/sizeof(double);
            int Aj = PyArray_STRIDES({A})[1]/sizeof(double);
            int Zi = PyArray_STRIDES({Z})[0]/sizeof(double);
            int Zj = PyArray_STRIDES({Z})[1]/sizeof(double);
            int xi = PyArray_STRIDES({x})[0]/sizeof(double);
            int yj = PyArray_STRIDES({y})[0]/sizeof(double);
            for (int i = 0; i < dims[0]; ++i)
            {{
                xx = alpha * xdata[xi * i];
                for (int j = 0; j < dims[1]; ++j)
                {{
                    tmp = zdata[Ai*i+Aj*j];
                    tmp += xx * ydata[yj * j];
                    zoutdata[Zi*i+Zj*j] = tmp;
                }}
            }}
        }}
        else
        {{
            PyErr_SetString(PyExc_AssertionError,
                            "neither float nor double dtype");
            {fail}
        }}
    }}
    else
    {{
        if ({Z} != {A})
        {{
            if ({Z}) {{ Py_DECREF({Z}); }}
            {Z} = {A};
            Py_INCREF({Z});
        }}
        npy_intp dims[2];
        dims[0] = PyArray_DIMS({A})[0];
        dims[1] = PyArray_DIMS({A})[1];
        if ((dims[0] * dims[1]) < 100000)
        {{
            if (PyArray_DESCR({Z})->type_num == NPY_FLOAT)
            {{
                float * zoutdata = (float*)PyArray_DATA({Z});
                const float * xdata = (float*)PyArray_DATA({x});
                const float * ydata = (float*)PyArray_DATA({y});
                const float * adata = (float*)PyArray_DATA({a});
                const float alpha = adata[0];
                float tmp, axi;
                int Zi = PyArray_STRIDES({Z})[0]/sizeof(float);
                int Zj = PyArray_STRIDES({Z})[1]/sizeof(float);
                int xi = PyArray_STRIDES({x})[0]/sizeof(float);
                int yj = PyArray_STRIDES({y})[0]/sizeof(float);
                for (int i = 0; i < dims[0]; ++i)
                {{
                    axi = alpha * xdata[xi * i];
                    for (int j = 0; j < dims[1]; ++j)
                    {{
                        zoutdata[Zi*i+Zj*j] += axi * ydata[yj * j];
                    }}
                }}
            }}
            else if (PyArray_DESCR({Z})->type_num == NPY_DOUBLE)
            {{
                double * zoutdata = (double*) PyArray_DATA({Z});
                const double * xdata = (double*)PyArray_DATA({x});
                const double * ydata = (double*)PyArray_DATA({y});
                const double * adata = (double*)PyArray_DATA({a});
                const double alpha = adata[0];
                double tmp, axi;

                int Zi = PyArray_STRIDES({Z})[0]/sizeof(double);
                int Zj = PyArray_STRIDES({Z})[1]/sizeof(double);
                int xi = PyArray_STRIDES({x})[0]/sizeof(double);
                int yj = PyArray_STRIDES({y})[0]/sizeof(double);
                for (int i = 0; i < dims[0]; ++i)
                {{
                    axi = alpha * xdata[xi * i];
                    for (int j = 0; j < dims[1]; ++j)
                    {{
                        zoutdata[Zi*i+Zj*j] += axi * ydata[yj * j];
                    }}
                }}
            }}
        }}
        else
        {{
            int Nz0 = PyArray_DIMS({Z})[0];
            int Nz1 = PyArray_DIMS({Z})[1];
            int Sx = PyArray_STRIDES({x})[0] / elemsize;
            int Sy = PyArray_STRIDES({y})[0] / elemsize;

            /* create appropriate strides for Z, if it is a row or column matrix.
             * In that case, the value of the stride does not really matter, but
             * some versions of BLAS insist that:
             *  - they are not smaller than the number of elements in the array,
             *  - they are not 0.
             */
            int Sz0 = (Nz0 > 1) ? (PyArray_STRIDES({Z})[0] / elemsize) : (Nz1 + 1);
            int Sz1 = (Nz1 > 1) ? (PyArray_STRIDES({Z})[1] / elemsize) : (Nz0 + 1);

            dtype_{x}* x_data = (dtype_{x}*) PyArray_DATA({x});
            dtype_{y}* y_data = (dtype_{y}*) PyArray_DATA({y});
            // gemv expects pointers to the beginning of memory arrays,
            // but numpy provides provides a pointer to the first element,
            // so when the stride is negative, we need to get the last one.
            if (Sx < 0)
                x_data += (Nz0 - 1) * Sx;
            if (Sy < 0)
                y_data += (Nz1 - 1) * Sy;

            if (PyArray_STRIDES({Z})[0] == elemsize)
            {{
                if (PyArray_DESCR({Z})->type_num == NPY_FLOAT)
                {{
                    float alpha = ((dtype_{a}*)PyArray_DATA({a}))[0];
                    sger_(&Nz0, &Nz1, &alpha,
                        (float*)x_data, &Sx,
                        (float*)y_data, &Sy,
                        (float*)(PyArray_DATA({Z})), &Sz1);
                }}
                else if (PyArray_DESCR({Z})->type_num == NPY_DOUBLE)
                {{
                    double alpha = ((dtype_{a}*)PyArray_DATA({a}))[0];
                    dger_(&Nz0, &Nz1, &alpha,
                        (double*)x_data, &Sx,
                        (double*)y_data, &Sy,
                        (double*)(PyArray_DATA({Z})), &Sz1);


                }}
                else {{
                    PyErr_SetString(PyExc_NotImplementedError,
                                    "not float nor double");
                    {fail}
                }}
            }}
            else if (PyArray_STRIDES({Z})[1] == elemsize)
            {{
                if (PyArray_DESCR({Z})->type_num == NPY_FLOAT)
                {{
                    float alpha = ((dtype_{a}*)(PyArray_DATA({a})))[0];
                    sger_(&Nz1, &Nz0, &alpha,
                        (float*)y_data, &Sy,
                        (float*)x_data, &Sx,
                        (float*)(PyArray_DATA({Z})), &Sz0);
                }}
                else if (PyArray_DESCR({Z})->type_num == NPY_DOUBLE)
                {{
                    double alpha = ((dtype_{a}*)PyArray_DATA({a}))[0];
                    dger_(&Nz1, &Nz0, &alpha,
                        (double*)y_data, &Sy,
                        (double*)x_data, &Sx,
                        (double*)(PyArray_DATA({Z})), &Sz0);
                }}
                else
                {{
                    PyErr_SetString(PyExc_NotImplementedError,
                                    "not float nor double");
                    {fail}
                }}
            }}
            else
            {{
                PyErr_SetString(PyExc_AssertionError,
                    "A is a double-strided matrix, and should have been copied "
                    "into a memory-contiguous one.");
                {fail}
            }}
        }}
    }}

    """


class CGer(BaseBLAS, Ger):
    params_type = ParamsType(
        destructive=bool_t,
    )

    def c_code(self, node, name, inp, out, sub):
        A, a, x, y = inp
        (Z,) = out
        code = ger_c_code(A, a, x, y, Z, fail=sub["fail"], params=sub["params"])
        return code

    def c_code_cache_version(self):
        return (11, blas_header_version())


cger_inplace = CGer(True)
cger_no_inplace = CGer(False)


# ##### ####### #######
# GEMV
# ##### ####### #######


def gemv_c_code(y, A, x, z, alpha, beta, fail, must_initialize_y=False, params=None):
    """
    z <- beta * y + alpha * dot(A, x)

    where A is a matrix, y and x are vectors (ergo z is vector)
    z = y if inplace else y.copy()
    """
    code = """

    bool is_float;
    int elemsize;
    float fbeta;
    double dbeta;

    if (PyArray_DIMS(%(A)s)[0] != PyArray_DIMS(%(y)s)[0])
    {
        PyErr_SetString(PyExc_ValueError,
                        "Shape mismatch: A.shape[0] != y.shape[0]");
        %(fail)s;
    }
    if (PyArray_DIMS(%(A)s)[1] != PyArray_DIMS(%(x)s)[0])
    {
        PyErr_SetString(PyExc_ValueError,
                        "Shape mismatch: A.shape[1] != x.shape[0]");
        %(fail)s;
    }

    if ((PyArray_DESCR(%(y)s)->type_num != PyArray_DESCR(%(x)s)->type_num)
        || (PyArray_DESCR(%(y)s)->type_num != PyArray_DESCR(%(A)s)->type_num))
    {
        PyErr_SetString(PyExc_TypeError, "GEMV: dtypes of A, x, y do not match");
        %(fail)s;
    }
    if  (PyArray_DESCR(%(y)s)->type_num == NPY_DOUBLE) {
        is_float = 0;
        elemsize = 8;
    }
    else if (PyArray_DESCR(%(y)s)->type_num == NPY_FLOAT) {
        elemsize = 4;
        is_float = 1;
    }
    else {
        %(fail)s;
        PyErr_SetString(PyExc_NotImplementedError, "GEMV: Inputs must be float or double");
    }

    fbeta = dbeta = ((dtype_%(beta)s*)PyArray_DATA(%(beta)s))[0];

    // copy y if not destructive
    if (!%(params)s->inplace)
    {
        if ((NULL == %(z)s)
            || (PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(y)s)[0]))
        {
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*)PyArray_SimpleNew(1,
                PyArray_DIMS(%(y)s), PyArray_TYPE(%(y)s));
            if(!%(z)s) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemv output");
                %(fail)s
            }
        }
        if (dbeta != 0)
        {
            // If dbeta is zero, we avoid doing the copy
            if (PyArray_CopyInto(%(z)s, %(y)s) != 0) {
                %(fail)s
            }
        }
    }
    else
    {
        if (%(z)s != %(y)s)
        {
            Py_XDECREF(%(z)s);
            %(z)s = %(y)s;
            Py_INCREF(%(z)s);
        }
    }

    {
        int NA0 = PyArray_DIMS(%(A)s)[0];
        int NA1 = PyArray_DIMS(%(A)s)[1];

        if (NA0 * NA1)
        {
            // Non-empty A matrix

            if (%(must_initialize_y)d && dbeta == 0)
            {
                // Most BLAS implementations of GEMV ignore y=nan when beta=0
                // PyTensor considers that the correct behavior,
                // and even exploits it to avoid copying or initializing outputs.
                // By deciding to exploit this, however, it becomes our responsibility
                // to ensure the behavior even in the rare cases BLAS deviates,
                // or users will get errors, even for graphs that had no nan to begin with.
                PyArray_FILLWBYTE(%(z)s, 0);
            }

            /* In the case where A is actually a row or column matrix,
             * the strides corresponding to the dummy dimension don't matter,
             * but BLAS requires these to be no smaller than the number of elements in the array.
             */
            int SA0 = (NA0 > 1) ? (PyArray_STRIDES(%(A)s)[0] / elemsize) : NA1;
            int SA1 = (NA1 > 1) ? (PyArray_STRIDES(%(A)s)[1] / elemsize) : NA0;
            int Sz = PyArray_STRIDES(%(z)s)[0] / elemsize;
            int Sx = PyArray_STRIDES(%(x)s)[0] / elemsize;

            dtype_%(A)s* A_data = (dtype_%(A)s*) PyArray_DATA(%(A)s);
            dtype_%(x)s* x_data = (dtype_%(x)s*) PyArray_DATA(%(x)s);
            dtype_%(z)s* z_data = (dtype_%(z)s*) PyArray_DATA(%(z)s);

            // gemv expects pointers to the beginning of memory arrays,
            // but numpy provides a pointer to the first element,
            // so when the stride is negative, we need to get the last one.
            if (Sx < 0)
                x_data += (NA1 - 1) * Sx;
            if (Sz < 0)
                z_data += (NA0 - 1) * Sz;

            if ( ((SA0 < 0) || (SA1 < 0)) && (abs(SA0) == 1 || (abs(SA1) == 1)) )
            {
                // We can treat the array A as C-or F-contiguous by changing the order of iteration
                // printf("GEMV: Iterating in reverse NA0=%%d, NA1=%%d, SA0=%%d, SA1=%%d\\n", NA0, NA1, SA0, SA1);
                if (SA0 < 0){
                    A_data += (NA0 -1) * SA0;  // Jump to first row
                    SA0 = -SA0;  // Iterate over rows in reverse
                    Sz = -Sz;  // Iterate over y in reverse
                }
                if (SA1 < 0){
                    A_data += (NA1 -1) * SA1;  // Jump to first column
                    SA1 = -SA1;  // Iterate over columns in reverse
                    Sx = -Sx;  // Iterate over x in reverse
                }
            } else if ((SA0 < 0) || (SA1 < 0) || ((SA0 != 1) && (SA1 != 1)))
            {
                // Array isn't contiguous, we have to make a copy
                // - if the copy is too long, maybe call vector/vector dot on each row instead
                // printf("GEMV: Making a copy NA0=%%d, NA1=%%d, SA0=%%d, SA1=%%d\\n", NA0, NA1, SA0, SA1);
                npy_intp dims[2];
                dims[0] = NA0;
                dims[1] = NA1;
                PyArrayObject * A_copy = (PyArrayObject *) PyArray_Copy(%(A)s);
                if (!A_copy)
                    %(fail)s
                Py_XDECREF(%(A)s);
                %(A)s = A_copy;
                SA0 = (NA0 > 1) ? (PyArray_STRIDES(%(A)s)[0] / elemsize) : NA1;
                SA1 = (NA1 > 1) ? (PyArray_STRIDES(%(A)s)[1] / elemsize) : NA0;
                A_data = (dtype_%(A)s*) PyArray_DATA(%(A)s);
            }
            //else {printf("GEMV: Using the original array NA0=%%d, NA1=%%d, SA0=%%d, SA1=%%d\\n", NA0, NA1, SA0, SA1);}

            if (NA0 == 1)
            {
                // Vector-vector dot product, it seems faster to avoid GEMV
                dtype_%(alpha)s alpha = ((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0];

                if (is_float)
                {
                    z_data[0] = dbeta != 0 ? dbeta * z_data[0] : 0.f;
                    z_data[0] += alpha * sdot_(&NA1,  (float*)(A_data), &SA1,
                                              (float*)x_data, &Sx);
                }
                else
                {
                    z_data[0] = dbeta != 0 ? dbeta * z_data[0] : 0.;
                    z_data[0] += alpha * ddot_(&NA1,  (double*)(A_data), &SA1,
                                              (double*)x_data, &Sx);
                }
            }
            else if (SA0 == 1)
            {
                // F-contiguous
                char NOTRANS = 'N';
                if (is_float)
                {
                    float alpha = ((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0];
                    sgemv_(&NOTRANS, &NA0, &NA1,
                        &alpha,
                        (float*)(A_data), &SA1,
                        (float*)x_data, &Sx,
                        &fbeta,
                        (float*)z_data, &Sz);
                }
                else
                {
                    double alpha = ((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0];
                    dgemv_(&NOTRANS, &NA0, &NA1,
                        &alpha,
                        (double*)(A_data), &SA1,
                        (double*)x_data, &Sx,
                        &dbeta,
                        (double*)z_data, &Sz);
                }
            }
            else if (SA1 == 1)
            {
                // C-contiguous
                char TRANS = 'T';
                if (is_float)
                {
                    float alpha = ((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0];
                    sgemv_(&TRANS, &NA1, &NA0,
                        &alpha,
                        (float*)(A_data), &SA0,
                        (float*)x_data, &Sx,
                        &fbeta,
                        (float*)z_data, &Sz);
                }
                else
                {
                    double alpha = ((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0];
                    dgemv_(&TRANS, &NA1, &NA0,
                        &alpha,
                        (double*)(A_data), &SA0,
                        (double*)x_data, &Sx,
                        &dbeta,
                        (double*)z_data, &Sz);
                }
            }
            else
            {
                PyErr_SetString(PyExc_AssertionError,
                                "A is neither C nor F-contiguous, it should have been copied into a memory-contiguous array;");
                %(fail)s
            }
        } else
        {
            // Empty A matrix, just scale y by beta
            if (dbeta != 1.0)
            {
                npy_intp Sz = PyArray_STRIDES(%(z)s)[0] / elemsize;
                dtype_%(z)s* z_data = (dtype_%(z)s*) PyArray_DATA(%(z)s);
                for (npy_intp i = 0; i < NA0; ++i)
                {
                    z_data[i * Sz] = (dbeta == 0.0) ? 0 : z_data[i * Sz] * dbeta;
                }
            }
        }
    }
    """
    return code % locals()


class CGemv(BaseBLAS, Gemv):
    params_type = ParamsType(
        inplace=bool_t,
    )

    def __init__(self, inplace):
        super().__init__(inplace)

    def c_code(self, node, name, inp, out, sub):
        y, alpha, A, x, beta = inp
        (z,) = out
        code = gemv_c_code(
            y,
            A,
            x,
            z,
            alpha,
            beta,
            fail=sub["fail"],
            must_initialize_y=must_initialize_y_gemv(),
            params=sub["params"],
        )
        return code

    def c_code_cache_version(self):
        return (18, blas_header_version(), must_initialize_y_gemv())


cgemv_inplace = CGemv(inplace=True)
cgemv_no_inplace = CGemv(inplace=False)


def must_initialize_y_gemv():
    if must_initialize_y_gemv._force_init_beta is None:
        from pytensor.link.c.cmodule import GCC_compiler

        """
        Test issue 1569.
        Namely when evaluating

            beta*y + alpha*dot(A, x)

        where we set y * beta = zeros of the correct dimensions we
        do not actually set y = zeros and instead let the BLAS
        perform beta*y with uninitialized memory for
        speed. Occasionally the memory contains values that are
        equivalent to NaN in which case the product beta*y contains
        NaN's for correctly implemented BLAS libraries. In this
        situation, since we are introducing the NaN's, we need to test
        whether the BLAS performs correctly. If it *does*, i.e. it
        actually performs the multiplication beta*y which will result
        in NaN's in the result, then we need initialize the memory to
        zeros.
        """
        test_code = """
#include <math.h>
extern "C" void dgemv_(char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int *);
int main() {
  double A[2][2] = {{1., 1.}, {1., 1.}};
  double x[2] = {1., 1.};
  double y[2] = {NAN, NAN};
  const int s = 2;
  const int inc = 1;
  const double alpha = 1.0;
  const double beta = 0.0;

  dgemv_("T", &s, &s, &alpha, A, &s, x, &inc, &beta, &y, &inc);

  return (isnan(y[0]) || isnan(y[1]) ? 1 : 0;
}
"""
        res = GCC_compiler.try_compile_tmp(
            test_code,
            tmp_prefix="check_beta_",
            flags=ldflags(libs=True, flags=True, libs_dir=True),
            try_run=True,
        )
        if res:
            if res[0]:
                must_initialize_y_gemv._force_init_beta = res[1]
            else:
                must_initialize_y_gemv._force_init_beta = False
        else:
            must_initialize_y_gemv._force_init_beta = False

    return must_initialize_y_gemv._force_init_beta


must_initialize_y_gemv._force_init_beta = None
