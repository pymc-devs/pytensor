#section support_code_apply

int APPLY_SPECIFIC(cpu_dimshuffle)(PyArrayObject *input, PyArrayObject **res, PARAMS_TYPE *params) {
    npy_int64* new_order;
    npy_intp nd_in;
    npy_intp nd_out;
    npy_intp* dimensions;
    npy_intp* strides;

    // This points to either the original input or a copy we create below.
    // Either way, this is what we should be working on/with.
    PyArrayObject *_input;

    if (!PyArray_IS_C_CONTIGUOUS(params->_new_order)) {
        PyErr_SetString(PyExc_RuntimeError, "DimShuffle: param _new_order must be C-contiguous.");
        return 1;
    }
    new_order = (npy_int64*) PyArray_DATA(params->_new_order);
    nd_in = (npy_intp)(params->input_ndim);
    nd_out = PyArray_SIZE(params->_new_order);

    if (PyArray_NDIM(input) != nd_in) {
        PyErr_SetString(PyExc_NotImplementedError, "DimShuffle: Input has less dimensions than expected.");
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

    npy_intp original_size = PyArray_SIZE(_input);
    npy_intp new_size = 1;
    for (npy_intp i = 0; i < nd_out; ++i) {
        if (new_order[i] != -1) {
            dimensions[i] = PyArray_DIMS(_input)[new_order[i]];
            strides[i] = PyArray_DIMS(_input)[new_order[i]] == 1 ? 0 : PyArray_STRIDES(_input)[new_order[i]];
        } else {
            dimensions[i] = 1;
            strides[i] = 0;
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

    if (params->inplace) {
        _input = input;
        Py_INCREF((PyObject*)_input);
    } else {
        _input = (PyArrayObject *)PyArray_FromAny(
            (PyObject *)input, NULL, 0, 0, NPY_ARRAY_ALIGNED | NPY_ARRAY_ENSURECOPY,
            NULL);
    }

    // Create the new array.
    *res = (PyArrayObject*)PyArray_New(&PyArray_Type, nd_out, dimensions,
                                       PyArray_TYPE(_input), strides,
                                       PyArray_DATA(_input), PyArray_ITEMSIZE(_input),
                                       // borrow only the writable flag from the base
                                       // the NPY_OWNDATA flag will default to 0.
                                       (NPY_ARRAY_WRITEABLE * PyArray_ISWRITEABLE(_input)),
                                       NULL);

    if (*res == NULL) {
        free(dimensions);
        free(strides);
        return 1;
    }

    // recalculate flags: CONTIGUOUS, FORTRAN, ALIGNED
    PyArray_UpdateFlags(*res, NPY_ARRAY_UPDATE_ALL);

    // we are making a view in both inplace and non-inplace cases
    PyArray_SetBaseObject(*res, (PyObject*)_input);

    free(strides);
    free(dimensions);
    return 0;
}