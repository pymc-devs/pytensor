"""BatchedDot Op and user-facing batched_dot/batched_tensordot functions.

BatchedDot computes: batched_dot(a, b)[i] = dot(a[i], b[i])
"""

import warnings

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

import pytensor.scalar
from pytensor.gradient import DisconnectedType, disconnected_type
from pytensor.graph import vectorize_graph
from pytensor.graph.basic import Apply
from pytensor.link.c.op import COp
from pytensor.tensor.basic import as_tensor_variable, cast
from pytensor.tensor.blas._core import blas_header_text, ldflags
from pytensor.tensor.blas_headers import blas_header_version
from pytensor.tensor.math import dot, tensordot
from pytensor.tensor.shape import specify_broadcastable
from pytensor.tensor.type import DenseTensorType, tensor


class BatchedDot(COp):
    """
    Computes a batch matrix-matrix dot with tensor3 variables

        batched_dot(a, b)[i] = dot(a[i], b[i])
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
                "Consider calling batched_dot instead."
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
        batch_gemm_defn = """
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
        return blas_header_text() + batch_gemm_defn

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

    def infer_shape(self, fgraph, node, shapes):
        xshp, yshp = shapes
        return [xshp[:-1] + yshp[2:]]


_batched_dot = BatchedDot()


def batched_dot(a, b):
    """Compute the batched dot product of two variables.

    I.e.:

        batched_dot(a, b)[i] = dot(a[i], b[i])

    Note that this batched_dot function does one of three things, in the
    following sequence:

        1.  If either a or b is a vector, it returns the batched elementwise
            product without calling the PyTensor BatchedDot op.

        2.  If both a and b have either 2 or 3 dimensions, it calls PyTensor's
            BatchedDot op on a and b.

        3.  If either a or b has more than 3 dimensions, it calls PyTensor's
            batched_tensordot function with appropriate axes. The
            batched_tensordot function expresses high-dimensional batched
            dot products in terms of batched matrix-matrix dot products, so
            it may be possible to further optimize for performance.
    """
    warnings.warn(
        "batched_dot is deprecated. "
        "Use `dot` in conjunction with `tensor.vectorize` or `graph.replace.vectorize_graph`",
        FutureWarning,
    )
    a, b = as_tensor_variable(a), as_tensor_variable(b)

    if a.ndim == 0:
        raise TypeError("a must have at least one (batch) axis")
    elif b.ndim == 0:
        raise TypeError("b must have at least one (batch) axis")

    core_a = a[0].type()
    core_b = b[0].type()
    core_dot = dot(core_a, core_b)
    return vectorize_graph(core_dot, replace={core_a: a, core_b: b})


def batched_tensordot(x, y, axes=2):
    """Compute a batched tensordot product.

    A hybrid of batched_dot and tensordot, this function computes the
    tensordot product between the two tensors, by iterating over the
    first dimension to perform a sequence of tensordots.

    Parameters
    ----------
    x: TensorVariable
        A tensor with sizes e.g.: for 3D (dim1, dim3, dim2)
    y: TensorVariable
        A tensor with sizes e.g.: for 3D (dim1, dim2, dim4)
    axes: int or array-like of length 2
        If an integer, the number of axes to sum over.
        If an array, it must have two array elements containing the axes to sum
        over in each tensor.

        If an integer i, it is converted to an array containing
        the last i dimensions of the first tensor and the first
        i dimensions of the second tensor (excluding the first
        (batch) dimension):
            axes = [list(range(a.ndim - i, b.ndim)), list(range(1,i+1))]

        If an array, its two elements must contain compatible axes
        of the two tensors. For example, [[1, 2], [2, 4]] means sum
        over the 2nd and 3rd axes of a and the 3rd and 5th axes of b.
        (Remember axes are zero-indexed!) The 2nd axis of a and the
        3rd axis of b must have the same shape; the same is true for
        the 3rd axis of a and the 5th axis of b.

    Like tensordot, this function uses a series of dimshuffles and
    reshapes to reduce the tensor dot product to a matrix or vector
    dot product.  Finally, it calls batched_dot to compute the result.
    """
    warnings.warn(
        "batched_tensordot is deprecated. "
        "Use `tensordot` in conjuction with `tensor.vectorize` or `graph.replace.vectorize_graph`",
        FutureWarning,
    )

    if isinstance(axes, int):
        core_axes = axes
    else:
        # Convert batched axes to core axes
        core_axes_a = [a - 1 for a in normalize_axis_tuple(axes[0], x.type.ndim)]
        core_axes = [a - 1 for a in normalize_axis_tuple(axes[1], y.type.ndim)]
        core_axes = [core_axes_a, core_axes]

    core_x = x[0].type()
    core_y = y[0].type()
    core_tensordot = tensordot(core_x, core_y, axes=core_axes)

    return vectorize_graph(core_tensordot, replace={core_x: x, core_y: y})
