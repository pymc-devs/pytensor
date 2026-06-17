import numpy as np

import pytensor.scalar
from pytensor.gradient import DisconnectedType, disconnected_type
from pytensor.graph.basic import Apply
from pytensor.link.c.op import COp
from pytensor.tensor.basic import as_tensor_variable, cast
from pytensor.tensor.blas._codegen import BATCH_GEMM
from pytensor.tensor.blas._core import ldflags
from pytensor.tensor.blas.blas_headers import blas_header_text, blas_header_version
from pytensor.tensor.shape import specify_broadcastable
from pytensor.tensor.type import DenseTensorType, tensor


class BatchedDot(COp):
    """
    Computes a batch matrix-matrix dot with tensor3 variables

        out[i] = dot(a[i], b[i])
    """

    __props__ = ()
    gufunc_signature = "(b,m,k),(b,k,n)->(b,m,n)"

    def make_node(self, x, y):
        x = as_tensor_variable(x)
        y = as_tensor_variable(y)

        if not (
            isinstance(x.type, DenseTensorType) and isinstance(y.type, DenseTensorType)
        ):
            raise NotImplementedError("Only dense tensor types are supported")

        if not (x.type.ndim == 3 and y.type.ndim == 3):
            raise TypeError(
                f"Inputs must have 3 ndim, but got {x.type.ndim} and {y.type.ndim}. "
                "Consider using `dot` with `pytensor.tensor.vectorize` or "
                "`pytensor.graph.replace.vectorize_graph` instead."
            )

        def extract_static_dim(dim_x, dim_y):
            dims = {dim_x, dim_y} - {None}
            if len(dims) > 1:
                # BatchedDot doesn't allow broadcasting
                raise ValueError(
                    f"Static dimensions of BatchedDot don't match, got {x.type.shape} and {y.type.shape}"
                )
            elif not dims:
                return None
            else:
                return dims.pop()

        x_batch_dim, x_row_dim, x_sum_dim = x.type.shape
        y_batch_dim, y_sum_dim, y_col_dim = y.type.shape
        batch_dim = extract_static_dim(x_batch_dim, y_batch_dim)
        # Raise if static sum dimensions do not match
        _ = extract_static_dim(x_sum_dim, y_sum_dim)
        out_shape = (batch_dim, x_row_dim, y_col_dim)

        # Change dtype if needed
        dtype = pytensor.scalar.upcast(x.type.dtype, y.type.dtype)
        x, y = cast(x, dtype), cast(y, dtype)
        out = tensor(dtype=dtype, shape=out_shape)
        return Apply(self, [x, y], [out])

    def perform(self, node, inp, out):
        x, y = inp
        (z,) = out

        if x.shape[0] != y.shape[0]:
            raise TypeError(
                f"Inputs [{', '.join(map(str, inp))}] must have the"
                f" same size in axis 0, but have sizes [{', '.join(str(i.shape[0]) for i in inp)}]."
            )

        z[0] = np.matmul(x, y)

    def c_support_code(self, **kwargs):
        return blas_header_text() + BATCH_GEMM

    def c_libraries(self, **kwargs):
        return ldflags()

    def c_compile_args(self, **kwargs):
        return ldflags(libs=False, flags=True)

    def c_lib_dirs(self, **kwargs):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self, **kwargs):
        return ldflags(libs=False, include_dir=True)

    def c_code(self, node, name, inp, out, sub):
        # Can only compile if linked to blas libraries
        if len(self.c_libraries()) <= 0:
            raise NotImplementedError()

        _x, _y = inp
        (_z,) = out
        fail = sub["fail"]

        # generate contiguity condition
        def contiguous(var, ndim):
            strides = f"PyArray_STRIDES({var})"
            if ndim == 1:
                return f"{strides}[0] == type_size"
            ands = " && ".join(
                f"{strides}[{i}] > 0 && {strides}[{i}] % type_size == 0"
                for i in range(1, ndim)
            )
            ors = " || ".join(f"{strides}[{i}] == type_size" for i in range(1, ndim))
            return f"{ands} && ({ors})"

        x_ndim, y_ndim, z_ndim = (
            node.inputs[0].ndim,
            node.inputs[1].ndim,
            node.outputs[0].ndim,
        )

        # generate code to allocate output based on runtime input shapes
        z_dims = [
            f"PyArray_DIMS({_x})[0]",
            f"PyArray_DIMS({_x})[1]",
            f"PyArray_DIMS({_y})[2]",
        ]

        z_shape_correct = " && ".join(
            f"PyArray_DIMS({_z})[{i}] == {dim}" for i, dim in enumerate(z_dims)
        )
        z_shape = ", ".join(z_dims)
        z_contiguous = contiguous(_z, z_ndim)
        allocate = f"""
            if (NULL == {_z} || !({z_shape_correct})  || !({z_contiguous}))
            {{
                npy_intp dims[{z_ndim}] = {{{z_shape}}};
                Py_XDECREF({_z});
                {_z} = (PyArrayObject*)PyArray_SimpleNew(
                    {z_ndim}, dims, PyArray_TYPE({_x}));
                if(!{_z}) {{
                    PyErr_SetString(PyExc_MemoryError,
                                    "failed to alloc BatchedDot output");
                    {fail}
                }}
            }}
        """

        # code to reallocate inputs contiguously if necessary
        contiguate = []
        for var, ndim in [(_x, x_ndim), (_y, y_ndim)]:
            _contiguous = contiguous(var, ndim)
            contiguate.append(
                f"""
                if (!({_contiguous})) {{
                    PyArrayObject * _copy = (PyArrayObject *) PyArray_Copy({var});
                    if (!_copy)
                        {fail}
                    Py_XDECREF({var});
                    {var} = _copy;
                }}
            """
            )
        contiguate = "\n".join(contiguate)

        return f"""
        int type_num = PyArray_DESCR({_x})->type_num;
        int type_size = PyArray_ITEMSIZE({_x}); // in bytes

        if (PyArray_NDIM({_x}) != 3) {{
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(x) != 3. rank(x) is %d.",
                         PyArray_NDIM({_x}));
            {fail};
        }}
        if (PyArray_NDIM({_y}) != 3) {{
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(y) != 3. rank(y) is %d.",
                         PyArray_NDIM({_y}));
            {fail};
        }}
        if ({_z} && PyArray_NDIM({_z}) != 3) {{
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(z) != 3. rank(z) is %d.",
                         PyArray_NDIM({_z}));
            {fail};
        }}

        // allocate output
        {allocate}
        // reallocate any noncontiguous arrays or arrays with invalid strides
        {contiguate}

        if ((PyArray_DESCR({_x})->type_num != NPY_DOUBLE)
            && (PyArray_DESCR({_x})->type_num != NPY_FLOAT))
        {{PyErr_SetString(PyExc_NotImplementedError, "type(x) is not double or float"); {fail};}}

        if ((PyArray_DESCR({_y})->type_num != NPY_DOUBLE)
            && (PyArray_DESCR({_y})->type_num != NPY_FLOAT))
        {{PyErr_SetString(PyExc_NotImplementedError, "type(y) is not double or float"); {fail};}}

        if ((PyArray_DESCR({_z})->type_num != NPY_DOUBLE)
            && (PyArray_DESCR({_z})->type_num != NPY_FLOAT))
        {{PyErr_SetString(PyExc_NotImplementedError, "type(z) is not double or float"); {fail};}}

        if ((PyArray_DESCR({_x})->type_num != PyArray_DESCR({_y})->type_num)
            ||(PyArray_DESCR({_x})->type_num != PyArray_DESCR({_z})->type_num))
        {{ PyErr_SetString(PyExc_NotImplementedError, "type(x), type(y), type(z) are not all the same"); {fail}; }}

        switch (type_num)
        {{
            case NPY_FLOAT:
            if (batch_gemm<float>(sgemm_, type_size, {_x}, {_y}, {_z})) {{
                {fail};
            }}
            break;
            case NPY_DOUBLE:
            if (batch_gemm<double>(dgemm_, type_size, {_x}, {_y}, {_z})) {{
                {fail};
            }}
            break;
        }}
        """

    def c_code_cache_version(self):
        return (6, blas_header_version())

    def pullback(self, inp, outputs, grads):
        x, y = inp
        (gz,) = grads

        xgrad = _batched_dot(gz, y.dimshuffle(0, 2, 1))
        ygrad = _batched_dot(x.dimshuffle(0, 2, 1), gz)

        # If x or y contain broadcastable dimensions but only one of
        # them know that a matching dimensions is broadcastable, the
        # above code don't always return the right broadcast pattern.
        # This cause problem down the road. See gh-1461.
        if xgrad.broadcastable != x.broadcastable:
            xgrad = specify_broadcastable(
                xgrad, *(ax for (ax, b) in enumerate(x.type.broadcastable) if b)
            )
        if ygrad.broadcastable != y.broadcastable:
            ygrad = specify_broadcastable(
                ygrad, *(ax for (ax, b) in enumerate(y.type.broadcastable) if b)
            )

        return xgrad, ygrad

    def pushforward(self, inputs, outputs, eval_points):
        assert len(inputs) == 2
        assert len(eval_points) == 2
        if isinstance(eval_points[0].type, DisconnectedType) and isinstance(
            eval_points[1].type, DisconnectedType
        ):
            return [disconnected_type()]

        if not isinstance(eval_points[0].type, DisconnectedType):
            t1 = self(eval_points[0], inputs[1])
        if not isinstance(eval_points[1].type, DisconnectedType):
            t2 = self(inputs[0], eval_points[1])

        if not isinstance(eval_points[0].type, DisconnectedType) and not isinstance(
            eval_points[1].type, DisconnectedType
        ):
            return [t1 + t2]
        elif not isinstance(eval_points[0].type, DisconnectedType):
            return [t1]
        else:
            return [t2]

    def infer_shape(self, node, shapes):
        xshp, yshp = shapes
        return [xshp[:-1] + yshp[2:]]


_batched_dot = BatchedDot()
