#section support_code_apply

int APPLY_SPECIFIC(cpu_dimshuffle)(PyArrayObject *input, PyArrayObject **res, PARAMS_TYPE *params) {
    npy_int64* new_order;
    npy_intp nd_in;
    npy_intp nd_out;
    npy_intp* dimensions;
    npy_intp* strides;

    if (!PyArray_IS_C_CONTIGUOUS(params->_new_order)) {
        PyErr_SetString(PyExc_RuntimeError, "DimShuffle: param _new_order must be C-contiguous.");
        return 1;
    }
    new_order = (npy_int64*) PyArray_DATA(params->_new_order);
    nd_in = (npy_intp)(params->input_ndim);
    nd_out = PyArray_SIZE(params->_new_order);

    if (PyArray_NDIM(input) != nd_in) {
        PyErr_SetString(PyExc_ValueError, "DimShuffle: Input has less dimensions than expected.");
        return 1;
    }

    // Compute new dimensions and strides
    dimensions = (npy_intp*) malloc(nd_out * sizeof(npy_intp));
    strides = (npy_intp*) malloc(nd_out * sizeof(npy_intp));
    if (dimensions == NULL || strides == NULL) {
        PyErr_NoMemory();
        free(dimensions);
        free(strides);
        return 1;
    };

    npy_intp original_size = PyArray_SIZE(input);
    npy_intp new_size = 1;
    for (npy_intp i = 0; i < nd_out; ++i) {
        // We set the strides of length 1 dimensions to PyArray_ITEMSIZE(input).
        // The value is arbitrary, because there is never a next element.
        // np.expand_dims(x, 0) and x[None] do different things here.
        // I would prefer zero, but there are some poorly implemented BLAS operations
        // That don't handle zero strides correctly. At least they won't fail because of DimShuffle.
        if (new_order[i] != -1) {
            dimensions[i] = PyArray_DIMS(input)[new_order[i]];
            strides[i] = PyArray_DIMS(input)[new_order[i]] == 1 ? PyArray_ITEMSIZE(input) : PyArray_STRIDES(input)[new_order[i]];
        } else {
            dimensions[i] = 1;
            strides[i] = PyArray_ITEMSIZE(input);
        }
        new_size *= dimensions[i];
    }

    if (original_size != new_size) {
        PyErr_SetString(PyExc_ValueError, "DimShuffle: Attempting to squeeze axes with size not equal to one.");
        free(dimensions);
        free(strides);
        return 1;
    }

    if (*res)
        Py_XDECREF(*res);

    // Create the new array.
    *res = (PyArrayObject*)PyArray_New(&PyArray_Type, nd_out, dimensions,
                                       PyArray_TYPE(input), strides,
                                       PyArray_DATA(input), PyArray_ITEMSIZE(input),
                                       // borrow only the writable flag from the base
                                       // the NPY_OWNDATA flag will default to 0.
                                       (NPY_ARRAY_WRITEABLE * PyArray_ISWRITEABLE(input)),
                                       NULL);

    if (*res == NULL) {
        free(dimensions);
        free(strides);
        return 1;
    }

    // Declare it a view of the original input
    Py_INCREF((PyObject*)input);
    PyArray_SetBaseObject(*res, (PyObject*)input);

    // recalculate flags: CONTIGUOUS, FORTRAN, ALIGNED
    PyArray_UpdateFlags(*res, NPY_ARRAY_UPDATE_ALL);

    free(strides);
    free(dimensions);
    return 0;
}