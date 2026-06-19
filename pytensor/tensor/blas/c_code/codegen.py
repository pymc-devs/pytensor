# C-code generators for the BLAS ops, kept out of the Op classes so the Op files
# stay small. Each generator takes the Apply ``node`` plus the C variable names
# and returns a C source string, matching the ``c_code(node, name, inputs,
# outputs, sub)`` contract.
#
# !!! CACHE DISCIPLINE !!!
# The CLinker module cache does NOT hash the generated C text (see
# ``cmodule_key_`` in pytensor/link/c/basic.py). A node's cache key is built
# from ``c_code_cache_version_apply`` + ``__props__`` + the input/output type
# versions. Any change to the emitted C must be reflected by BUMPING the leading
# version int in the corresponding op's cache version -- otherwise callers
# silently get stale compiled binaries.


# ##### ####### #######
# GEMM family (Gemm, Dot22, Dot22Scalar)
# ##### ####### #######

_DECLARE_NS = """
        int unit = 0;

        int type_num = PyArray_DESCR(%(_x)s)->type_num;
        int type_size = PyArray_ITEMSIZE(%(_x)s); // in bytes

        npy_intp* Nx = PyArray_DIMS(%(_x)s);
        npy_intp* Ny = PyArray_DIMS(%(_y)s);
        npy_intp* Nz = 0; //PyArray_DIMS(%(_zout)s);

        npy_intp* Sx = PyArray_STRIDES(%(_x)s);
        npy_intp* Sy = PyArray_STRIDES(%(_y)s);
        npy_intp* Sz = 0; //PyArray_STRIDES(%(_zout)s);

        //strides for x, y, z in dimensions 0, 1
        int sx_0, sx_1, sy_0, sy_1, sz_0, sz_1;
        """

_CHECK_XYZ_RANK2 = """
        if (PyArray_NDIM(%(_x)s) != 2) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(x) != 2. rank(x) is %%d.",
                         PyArray_NDIM(%(_x)s));
            %(fail)s;
        }
        if (PyArray_NDIM(%(_y)s) != 2) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(y) != 2. rank(y) is %%d.", PyArray_NDIM(%(_y)s));
            %(fail)s;
        }
        if (%(_zout)s && PyArray_NDIM(%(_zout)s) != 2) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(z) != 2. rank(z) is %%d.", PyArray_NDIM(%(_zout)s));
            %(fail)s;
        }
        """

_CHECK_XYZ_DOUBLE_OR_FLOAT = """
        if ((PyArray_DESCR(%(_x)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_x)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(x) is not double or float"); %(fail)s;}

        if ((PyArray_DESCR(%(_y)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_y)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(y) is not double or float"); %(fail)s;}

        if ((PyArray_DESCR(%(_zout)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_zout)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(z) is not double or float"); %(fail)s;}

        if ((PyArray_DESCR(%(_x)s)->type_num != PyArray_DESCR(%(_y)s)->type_num)
            ||(PyArray_DESCR(%(_x)s)->type_num != PyArray_DESCR(%(_zout)s)->type_num))
        { PyErr_SetString(PyExc_NotImplementedError, "type(x), type(y), type(z) are not all the same"); %(fail)s; }
        """

_GEMM_CHECK_AB_DOUBLE_OR_FLOAT = """
        if ((PyArray_DESCR(%(_a)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_a)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(a) is not double or float"); %(fail)s;}

        if ((PyArray_DESCR(%(_b)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_b)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(b) is not double or float"); %(fail)s;}
        """

_CHECK_DIMS = """
        if (Nx[0] !=1 && Nz[0] != 1 && Nx[0] != Nz[0])
        {
            PyErr_Format(PyExc_ValueError,
                "Shape mismatch: x has %%ld rows but z has %%ld rows",
                (long int)Nx[0], (long int)Nz[0]);
            %(fail)s;
        }
        if (Nx[1] != Ny[0])
        {
            PyErr_Format(PyExc_ValueError,
                "Shape mismatch: x has %%ld cols (and %%ld rows) but y has %%ld rows (and %%ld cols)",
                (long int)Nx[1], (long int)Nx[0], (long int)Ny[0], (long int)Ny[1]);
            %(fail)s;
        }
        if (Ny[1] != 1 && Nz[1]!= 1 && Ny[1] != Nz[1])
        {
            PyErr_Format(PyExc_ValueError,
                "Shape mismatch: y has %%ld cols but z has %%ld cols",
                (long int)Ny[1], (long int)Nz[1]);
            %(fail)s;
        }

        // We must not raise an error when Nx[1] == 0. This would disable cases
        // that numpy.dot accept.
        """

_CHECK_STRIDES = """
        /*
        If some matrices are not contiguous on either dimensions,
        or have invalid strides, copy their content into a contiguous one
        */
        if ((Sx[0] < 1) || (Sx[1] < 1) || (Sx[0] MOD type_size) || (Sx[1] MOD type_size)
            || ((Sx[0] != type_size) && (Sx[1] != type_size)))
        {
            PyArrayObject * _x_copy = (PyArrayObject *) PyArray_Copy(%(_x)s);
            if (!_x_copy)
                %(fail)s
            Py_XDECREF(%(_x)s);
            %(_x)s = _x_copy;
            Sx = PyArray_STRIDES(%(_x)s);
            if ((Sx[0] < 1) || (Sx[1] < 1)) {
                compute_strides(Nx, 2, type_size, Sx);
            }
        }

        if ((Sy[0] < 1) || (Sy[1] < 1) || (Sy[0] MOD type_size) || (Sy[1] MOD type_size)
            || ((Sy[0] != type_size) && (Sy[1] != type_size)))
        {
            PyArrayObject * _y_copy = (PyArrayObject *) PyArray_Copy(%(_y)s);
            if (!_y_copy)
                %(fail)s
            Py_XDECREF(%(_y)s);
            %(_y)s = _y_copy;
            Sy = PyArray_STRIDES(%(_y)s);
            if ((Sy[0] < 1) || (Sy[1] < 1)) {
                compute_strides(Ny, 2, type_size, Sy);
            }
        }

        if ((Sz[0] < 1) || (Sz[1] < 1) || (Sz[0] MOD type_size) || (Sz[1] MOD type_size)
            || ((Sz[0] != type_size) && (Sz[1] != type_size)))
        {
            PyArrayObject * _z_copy = (PyArrayObject *) PyArray_Copy(%(_zout)s);
            if (!_z_copy)
                %(fail)s
            Py_XDECREF(%(_zout)s);
            %(_zout)s = _z_copy;
            Sz = PyArray_STRIDES(%(_zout)s);
            if ((Sz[0] < 1) || (Sz[1] < 1)) {
                compute_strides(Nz, 2, type_size, Sz);
            }
        }
        """

_ENCODE_STRIDES_IN_UNIT = """
        /*
        encode the stride structure of _x,_y,_zout into a single integer
        */
        unit |= ((Sx[1] == type_size || Nx[1]==1) ? 0x0 : (Sx[0] == type_size || Nx[0]==1) ? 0x1 : 0x2) << 8;
        unit |= ((Sy[1] == type_size || Ny[1]==1) ? 0x0 : (Sy[0] == type_size || Ny[0]==1) ? 0x1 : 0x2) << 4;
        unit |= ((Sz[1] == type_size || Nz[1]==1) ? 0x0 : (Sz[0] == type_size || Nz[0]==1) ? 0x1 : 0x2) << 0;
        """

_COMPUTE_STRIDES = """
        /* create appropriate strides for malformed matrices that are row or column
         * vectors, or empty matrices.
         * In that case, the value of the stride does not really matter, but
         * some versions of BLAS insist that:
         *  - they are not smaller than the number of elements in the array,
         *  - they are not 0.
         */
        sx_0 = (Nx[0] > 1) ? Sx[0]/type_size : (Nx[1] + 1);
        sx_1 = (Nx[1] > 1) ? Sx[1]/type_size : (Nx[0] + 1);
        sy_0 = (Ny[0] > 1) ? Sy[0]/type_size : (Ny[1] + 1);
        sy_1 = (Ny[1] > 1) ? Sy[1]/type_size : (Ny[0] + 1);
        sz_0 = (Nz[0] > 1) ? Sz[0]/type_size : (Nz[1] + 1);
        sz_1 = (Nz[1] > 1) ? Sz[1]/type_size : (Nz[0] + 1);
        """

_BEGIN_SWITCH_TYPENUM = """
        switch (type_num)
        {
        """

_CASE_FLOAT = """
            case NPY_FLOAT:
            {
        """

_CASE_FLOAT_GEMM = """
                float* x = (float*)PyArray_DATA(%(_x)s);
                float* y = (float*)PyArray_DATA(%(_y)s);
                float* z = (float*)PyArray_DATA(%(_zout)s);
                char N = 'N';
                char T = 'T';
                int Nz0 = Nz[0], Nz1 = Nz[1], Nx1 = Nx[1];
                switch(unit)
                {
                    case 0x000: sgemm_(&N, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_0, &b, z, &sz_0); break;
                    case 0x100: sgemm_(&N, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_1, &b, z, &sz_0); break;
                    case 0x010: sgemm_(&T, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_0, &b, z, &sz_0); break;
                    case 0x110: sgemm_(&T, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_1, &b, z, &sz_0); break;
                    case 0x001: sgemm_(&T, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_0, &b, z, &sz_1); break;
                    case 0x101: sgemm_(&N, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_0, &b, z, &sz_1); break;
                    case 0x011: sgemm_(&T, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_1, &b, z, &sz_1); break;
                    case 0x111: sgemm_(&N, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_1, &b, z, &sz_1); break;
                    default: PyErr_SetString(PyExc_ValueError, "some matrix has no unit stride"); %(fail)s;
                };
        """

_CASE_DOUBLE = """
            }
            break;
            case NPY_DOUBLE:
            {
        """

_CASE_DOUBLE_GEMM = """
                double* x = (double*)PyArray_DATA(%(_x)s);
                double* y = (double*)PyArray_DATA(%(_y)s);
                double* z = (double*)PyArray_DATA(%(_zout)s);
                char N = 'N';
                char T = 'T';
                int Nz0 = Nz[0], Nz1 = Nz[1], Nx1 = Nx[1];
                switch(unit)
                {
                    case 0x000: dgemm_(&N, &N, &Nz1, &Nz0, &Nx1, &a, y,
                                       &sy_0, x, &sx_0, &b, z, &sz_0); break;
                    case 0x100: dgemm_(&N, &T, &Nz1, &Nz0, &Nx1, &a, y,
                                       &sy_0, x, &sx_1, &b, z, &sz_0); break;
                    case 0x010: dgemm_(&T, &N, &Nz1, &Nz0, &Nx1, &a, y,
                                       &sy_1, x, &sx_0, &b, z, &sz_0); break;
                    case 0x110: dgemm_(&T, &T, &Nz1, &Nz0, &Nx1, &a, y,
                                       &sy_1, x, &sx_1, &b, z, &sz_0); break;
                    case 0x001: dgemm_(&T, &T, &Nz0, &Nz1, &Nx1, &a, x,
                                       &sx_0, y, &sy_0, &b, z, &sz_1); break;
                    case 0x101: dgemm_(&N, &T, &Nz0, &Nz1, &Nx1, &a, x,
                                       &sx_1, y, &sy_0, &b, z, &sz_1); break;
                    case 0x011: dgemm_(&T, &N, &Nz0, &Nz1, &Nx1, &a, x,
                                       &sx_0, y, &sy_1, &b, z, &sz_1); break;
                    case 0x111: dgemm_(&N, &N, &Nz0, &Nz1, &Nx1, &a, x,
                                       &sx_1, y, &sy_1, &b, z, &sz_1); break;
                    default: PyErr_SetString(PyExc_ValueError,
                                             "some matrix has no unit stride");
                             %(fail)s;
                };
        """

_END_SWITCH_TYPENUM = """
            }
            break;
        }
        """

_GEMM_SETUP_Z_INPLACE = """
        // Needs broadcasting
        if (PyArray_DIMS(%(_z)s)[0] < Nx[0] || PyArray_DIMS(%(_z)s)[1] < Ny[1]){

            npy_intp dims[2];
            dims[0] = (PyArray_DIMS(%(_z)s)[0] >= Nx[0]) ? PyArray_DIMS(%(_z)s)[0] : Nx[0];
            dims[1] = (PyArray_DIMS(%(_z)s)[1] >= Ny[1]) ? PyArray_DIMS(%(_z)s)[1] : Ny[1];

            // Check if we need to allocate new array
            if((NULL == %(_zout)s)
                || (PyArray_DIMS(%(_zout)s)[0] != dims[0])
                || (PyArray_DIMS(%(_zout)s)[1] != dims[1]))
            {
                // fprintf(stderr, "Gemm Allocating z output array with shape (%%i %%i)\\n", dims[0], dims[1]);
                Py_XDECREF(%(_zout)s);
                %(_zout)s = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(_z)s));
            }

            // fprintf(stderr, "Gemm Broadcasting Z into shape (%%i %%i)\\n", dims[0], dims[1]);
            if(PyArray_CopyInto(%(_zout)s, %(_z)s) == -1)
            {
                %(fail)s;
            }

        } else {
            if (%(_zout)s != %(_z)s)
            {
                Py_XDECREF(%(_zout)s);
                %(_zout)s = %(_z)s;
                Py_INCREF(%(_zout)s);
            }
        }

        Nz = PyArray_DIMS(%(_zout)s);
        Sz = PyArray_STRIDES(%(_zout)s);
        """

_GEMM_SETUP_Z_OUTPLACE = """
        npy_intp dims[2];
        dims[0] = (PyArray_DIMS(%(_z)s)[0] >= Nx[0]) ? PyArray_DIMS(%(_z)s)[0] : Nx[0];
        dims[1] = (PyArray_DIMS(%(_z)s)[1] >= Ny[1]) ? PyArray_DIMS(%(_z)s)[1] : Ny[1];

        // Check if we need to allocate new array
        if ((NULL == %(_zout)s)
            || (PyArray_DIMS(%(_zout)s)[0] != dims[0])
            || (PyArray_DIMS(%(_zout)s)[1] != dims[1]))
        {
            Py_XDECREF(%(_zout)s);
            %(_zout)s = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(_z)s));
            // fprintf(stderr, "Gemm Allocating z output array with shape (%%i %%i)\\n", dims[0], dims[1]);
            if(!%(_zout)s) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemm_no_inplace output");
                %(fail)s
            }
        }

        // fprintf(stderr, "Gemm Broadcasting Z into shape (%%i %%i)\\n", dims[0], dims[1]);
        if(PyArray_CopyInto(%(_zout)s, %(_z)s) == -1)
        {
            %(fail)s
        }

        Nz = PyArray_DIMS(%(_zout)s);
        Sz = PyArray_STRIDES(%(_zout)s);
        """

_GEMM_BROADCAST_XY = """
        // Broadcast X if needed
        if (Nz[0] > Nx[0])
        {
            npy_intp dims[2];
            dims[0] = Nz[0];
            dims[1] = Nx[1];
            // fprintf(stderr, "Gemm Broadcasting X into shape (%%i %%i)\\n", dims[0], dims[1]);
            PyArrayObject *x_new = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(_x)s));
            if(!x_new) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemm_inplace input");
                %(fail)s
            }

            if(PyArray_CopyInto(x_new, %(_x)s) == -1)
            {
                %(fail)s
            }

            Py_DECREF(%(_x)s);
            %(_x)s = x_new;

            Nx = PyArray_DIMS(%(_x)s);
            Sx = PyArray_STRIDES(%(_x)s);
        }

        // Broadcast Y if needed
        if (Nz[1] > Ny[1])
        {
            npy_intp dims[2];
            dims[0] = Ny[0];
            dims[1] = Nz[1];
            // fprintf(stderr, "Gemm Broadcasting Y into shape (%%i %%i)\\n", dims[0], dims[1]);
            PyArrayObject *y_new = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(_x)s));
            if(!y_new) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemm_inplace input");
                %(fail)s
            }

            if(PyArray_CopyInto(y_new, %(_y)s) == -1)
            {
                %(fail)s
            }

            Py_DECREF(%(_y)s);
            %(_y)s = y_new;

            Ny = PyArray_DIMS(%(_y)s);
            Sy = PyArray_STRIDES(%(_y)s);
        }

    """

_GEMM_CASE_FLOAT_AB_CONSTANTS = """
        #define REAL float
        float a = (PyArray_DESCR(%(_a)s)->type_num == NPY_FLOAT)
        ? (REAL)(((float*)PyArray_DATA(%(_a)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_a)s))[0]);
        float b = (PyArray_DESCR(%(_b)s)->type_num == NPY_FLOAT) ?
        (REAL)(((float*)PyArray_DATA(%(_b)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_b)s))[0]);
        #undef REAL
        """

_GEMM_CASE_DOUBLE_AB_CONSTANTS = """
        #define REAL double
        double a = (PyArray_DESCR(%(_a)s)->type_num == NPY_FLOAT)
        ? (REAL)(((float*)PyArray_DATA(%(_a)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_a)s))[0]);
        double b = (PyArray_DESCR(%(_b)s)->type_num == NPY_FLOAT) ?
        (REAL)(((float*)PyArray_DATA(%(_b)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_b)s))[0]);
        #undef REAL
        """

_DOT22_SETUP_Z = """
        if ((NULL == %(_zout)s)
            || (PyArray_DIMS(%(_zout)s)[0] != PyArray_DIMS(%(_x)s)[0])
            || (PyArray_DIMS(%(_zout)s)[1] != PyArray_DIMS(%(_y)s)[1]))
        {
            if (NULL != %(_zout)s) Py_XDECREF(%(_zout)s);
            npy_intp dims[2];
            dims[0] = PyArray_DIMS(%(_x)s)[0];
            dims[1] = PyArray_DIMS(%(_y)s)[1];
            %(_zout)s = (PyArrayObject*)PyArray_SimpleNew(2, dims,
                            PyArray_TYPE(%(_x)s));
            //fprintf(stderr, "Dot Allocating %%i %%i\\n", dims[0], dims[1]);
            if(!%(_zout)s) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc dot22 output");
                %(fail)s
            }
        }
        Nz = PyArray_DIMS(%(_zout)s);
        Sz = PyArray_STRIDES(%(_zout)s);

        """

_DOT22_CASE_FLOAT_AB_CONSTANTS = """
                float a = 1.0;
                float b = 0.0;
        """

_DOT22_CASE_DOUBLE_AB_CONSTANTS = """
                double a = 1.0;
                double b = 0.0;
        """

_DOT22SCALAR_CHECK_AB = """
        if ((PyArray_DESCR(%(_a)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_a)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError,
                         "type(a) is not double or float"); %(fail)s;}

        """

_DOT22SCALAR_CASE_FLOAT_AB_CONSTANTS = """
        #define REAL float
        float a = (PyArray_DESCR(%(_a)s)->type_num == NPY_FLOAT)
        ? (REAL)(((float*)PyArray_DATA(%(_a)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_a)s))[0]);
        #undef REAL
        float b = 0.0;
        """

_DOT22SCALAR_CASE_DOUBLE_AB_CONSTANTS = """
        #define REAL double
        double a = (PyArray_DESCR(%(_a)s)->type_num == NPY_FLOAT)
        ? (REAL)(((float*)PyArray_DATA(%(_a)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_a)s))[0]);
        #undef REAL
        double b = 0.0;
        """


def _assemble_gemm_call(
    *, setup_z, check_ab, broadcast_xy, ab_constants_float, ab_constants_double
):
    """Concatenate the GEMM template fragments in execution order."""
    return "".join(
        (
            _DECLARE_NS,
            _CHECK_XYZ_RANK2,
            setup_z,
            _CHECK_XYZ_DOUBLE_OR_FLOAT,
            check_ab,
            broadcast_xy,
            _CHECK_DIMS,
            _CHECK_STRIDES,
            _ENCODE_STRIDES_IN_UNIT,
            _COMPUTE_STRIDES,
            _BEGIN_SWITCH_TYPENUM,
            _CASE_FLOAT,
            ab_constants_float,
            _CASE_FLOAT_GEMM,
            _CASE_DOUBLE,
            ab_constants_double,
            _CASE_DOUBLE_GEMM,
            _END_SWITCH_TYPENUM,
        )
    )


def gemm_c_code(node, name, inputs, outputs, sub):
    r"""C code for ``Gemm``: :math:`z \leftarrow b\,z + a\,xy` (in/out-of-place)."""
    _z, _a, _x, _y, _b = inputs
    (_zout,) = outputs
    setup_z = (
        f"if(%(params)s->inplace){{{_GEMM_SETUP_Z_INPLACE}}}"
        f"else{{{_GEMM_SETUP_Z_OUTPLACE}}}"
    )
    code = _assemble_gemm_call(
        setup_z=setup_z,
        check_ab=_GEMM_CHECK_AB_DOUBLE_OR_FLOAT,
        broadcast_xy=_GEMM_BROADCAST_XY,
        ab_constants_float=_GEMM_CASE_FLOAT_AB_CONSTANTS,
        ab_constants_double=_GEMM_CASE_DOUBLE_AB_CONSTANTS,
    )
    return code % dict(_z=_z, _a=_a, _x=_x, _y=_y, _b=_b, _zout=_zout, **sub)


def dot22_c_code(node, name, inputs, outputs, sub):
    r"""C code for ``Dot22``: :math:`z \leftarrow xy`, allocating a fresh output."""
    _x, _y = inputs
    (_zout,) = outputs
    code = _assemble_gemm_call(
        setup_z=_DOT22_SETUP_Z,
        check_ab="",
        broadcast_xy="",
        ab_constants_float=_DOT22_CASE_FLOAT_AB_CONSTANTS,
        ab_constants_double=_DOT22_CASE_DOUBLE_AB_CONSTANTS,
    )
    return code % dict(_x=_x, _y=_y, _zout=_zout, **sub)


def dot22scalar_c_code(node, name, inputs, outputs, sub):
    r"""C code for ``Dot22Scalar``: :math:`z \leftarrow a\,xy`, allocating a fresh output."""
    _x, _y, _a = inputs
    (_zout,) = outputs
    code = _assemble_gemm_call(
        setup_z=_DOT22_SETUP_Z,
        check_ab=_DOT22SCALAR_CHECK_AB,
        broadcast_xy="",
        ab_constants_float=_DOT22SCALAR_CASE_FLOAT_AB_CONSTANTS,
        ab_constants_double=_DOT22SCALAR_CASE_DOUBLE_AB_CONSTANTS,
    )
    return code % dict(_x=_x, _y=_y, _a=_a, _zout=_zout, **sub)


# ##### ####### #######
# GEMV
# ##### ####### #######


def gemv_c_code(node, name, inputs, outputs, sub):
    r"""C code for ``CGemv``: :math:`z \leftarrow \beta\,y + \alpha\,Ax`.

    ``z`` aliases ``y`` when inplace, otherwise a fresh copy; :math:`A` is a
    matrix and :math:`x`, :math:`y` are vectors.
    """
    # Imported lazily to avoid an import cycle (blas_c imports this module).
    from pytensor.tensor.blas.blas_c import must_initialize_y_gemv

    y, alpha, A, x, beta = inputs
    (z,) = outputs
    must_initialize_y = must_initialize_y_gemv()
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
    return code % dict(
        y=y,
        A=A,
        x=x,
        z=z,
        alpha=alpha,
        beta=beta,
        must_initialize_y=must_initialize_y,
        **sub,
    )


# ##### ####### #######
# GER
# ##### ####### #######


def ger_c_code(node, name, inputs, outputs, sub):
    r"""C code for ``CGer``: rank-1 update :math:`Z = A + \alpha\,x y^{\top}`."""
    A, a, x, y = inputs
    (Z,) = outputs
    fail = sub["fail"]
    params = sub["params"]
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


# ##### ####### #######
# BatchedDot
# ##### ####### #######

# C++ template that loops over the batch axis, calling gemm on each matrix pair.
# Emitted as BatchedDot's c_support_code (after the BLAS headers).
BATCH_GEMM = """
        template<typename dtype>
        bool batch_gemm(void (*gemm)(char*, char*, const int*, const int*, const int*, const dtype*, const dtype*, const int*, const dtype*, const int*, const dtype*, dtype*, const int*),
                        int type_size, PyArrayObject* xs, PyArrayObject* ys,
                        PyArrayObject* zs) {
            npy_intp *Nx = PyArray_DIMS(xs), *Sx = PyArray_STRIDES(xs);
            npy_intp *Ny = PyArray_DIMS(ys), *Sy = PyArray_STRIDES(ys);
            npy_intp *Nz = PyArray_DIMS(zs), *Sz = PyArray_STRIDES(zs);

            if (Nx[0] != Ny[0]) {
                PyErr_Format(PyExc_ValueError,
                             "Shape mismatch: batch sizes unequal."
                             " x.shape is (%d, %d, %d),"
                             " y.shape is (%d, %d, %d).",
                             Nx[0], Nx[1], Nx[2],
                             Ny[0], Ny[1], Ny[2]);
                return 1;
            }

            if (Nx[2] != Ny[1]) {
                PyErr_Format(PyExc_ValueError,
                             "Shape mismatch: summation axis sizes unequal."
                             " x.shape is (%d, %d, %d),"
                             " y.shape is (%d, %d, %d).",
                             Nx[0], Nx[1], Nx[2],
                             Ny[0], Ny[1], Ny[2]);
                return 1;
            }

            /* encode the stride structure of _x,_y,_z into a single integer. */
            int unit = 0;
            unit |= ((Sx[2] == type_size || Nx[2] == 1) ? 0x0 : (Sx[1] == type_size || Nx[1]==1) ? 0x1 : 0x2) << 8;
            unit |= ((Sy[2] == type_size || Ny[2] == 1) ? 0x0 : (Sy[1] == type_size || Ny[1]==1) ? 0x1 : 0x2) << 4;
            unit |= ((Sz[2] == type_size || Nz[2] == 1) ? 0x0 : (Sz[1] == type_size || Nz[1]==1) ? 0x1 : 0x2) << 0;

            /* create appropriate strides for malformed matrices that are row or column
             * vectors, or empty matrices.
             * In that case, the value of the stride does not really matter, but
             * some versions of BLAS insist that:
             *  - they are not smaller than the number of elements in the array,
             *  - they are not 0.
             */
            int sx_1 = (Nx[1] > 1) ? Sx[1]/type_size : (Nx[2] + 1);
            int sx_2 = (Nx[2] > 1) ? Sx[2]/type_size : (Nx[1] + 1);
            int sy_1 = (Ny[1] > 1) ? Sy[1]/type_size : (Ny[2] + 1);
            int sy_2 = (Ny[2] > 1) ? Sy[2]/type_size : (Ny[1] + 1);
            int sz_1 = (Nz[1] > 1) ? Sz[1]/type_size : (Nz[2] + 1);
            int sz_2 = (Nz[2] > 1) ? Sz[2]/type_size : (Nz[1] + 1);

            dtype* x = (dtype*)PyArray_DATA(xs);
            dtype* y = (dtype*)PyArray_DATA(ys);
            dtype* z = (dtype*)PyArray_DATA(zs);

            dtype a = 1.0;
            dtype b = 0.0;
            char N = 'N';
            char T = 'T';
            int Nz1 = Nz[1], Nz2 = Nz[2], Nx2 = Nx[2];

            // loop over batch axis
            for (int i = 0; i < Nz[0]; i++) {
                switch(unit)
                {
                    case 0x000: gemm(&N, &N, &Nz2, &Nz1, &Nx2, &a, y, &sy_1, x, &sx_1, &b, z, &sz_1); break;
                    case 0x100: gemm(&N, &T, &Nz2, &Nz1, &Nx2, &a, y, &sy_1, x, &sx_2, &b, z, &sz_1); break;
                    case 0x010: gemm(&T, &N, &Nz2, &Nz1, &Nx2, &a, y, &sy_2, x, &sx_1, &b, z, &sz_1); break;
                    case 0x110: gemm(&T, &T, &Nz2, &Nz1, &Nx2, &a, y, &sy_2, x, &sx_2, &b, z, &sz_1); break;
                    case 0x001: gemm(&T, &T, &Nz1, &Nz2, &Nx2, &a, x, &sx_1, y, &sy_1, &b, z, &sz_2); break;
                    case 0x101: gemm(&N, &T, &Nz1, &Nz2, &Nx2, &a, x, &sx_2, y, &sy_1, &b, z, &sz_2); break;
                    case 0x011: gemm(&T, &N, &Nz1, &Nz2, &Nx2, &a, x, &sx_1, y, &sy_2, &b, z, &sz_2); break;
                    case 0x111: gemm(&N, &N, &Nz1, &Nz2, &Nx2, &a, x, &sx_2, y, &sy_2, &b, z, &sz_2); break;
                    default: PyErr_SetString(PyExc_ValueError, "some matrix has no unit stride"); return 1;
                };
                x += Sx[0] / type_size;
                y += Sy[0] / type_size;
                z += Sz[0] / type_size;
            }

            return 0;
        }
        """
