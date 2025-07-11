from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy import convolve as numpy_convolve

from pytensor.gradient import DisconnectedType
from pytensor.graph import Apply, Constant
from pytensor.link.c.op import COp
from pytensor.scalar import as_scalar
from pytensor.scalar.basic import upcast
from pytensor.tensor.basic import as_tensor_variable, join, zeros
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.math import maximum, minimum, switch
from pytensor.tensor.type import vector
from pytensor.tensor.variable import TensorVariable


if TYPE_CHECKING:
    from pytensor.tensor import TensorLike


class Convolve1d(COp):
    __props__ = ()
    gufunc_signature = "(n),(k),()->(o)"

    def make_node(self, in1, in2, full_mode):
        in1 = as_tensor_variable(in1)
        in2 = as_tensor_variable(in2)
        full_mode = as_scalar(full_mode)

        if not (in1.ndim == 1 and in2.ndim == 1):
            raise ValueError("Convolution inputs must be vector (ndim=1)")
        if not full_mode.dtype == "bool":
            raise ValueError("Convolution mode must be a boolean type")

        dtype = upcast(in1.dtype, in2.dtype)
        n = in1.type.shape[0]
        k = in2.type.shape[0]
        match full_mode:
            case Constant():
                static_mode = "full" if full_mode.data else "valid"
            case _:
                static_mode = None

        if n is None or k is None or static_mode is None:
            out_shape = (None,)
        elif static_mode == "full":
            out_shape = (n + k - 1,)
        else:  # mode == "valid":
            out_shape = (max(n, k) - min(n, k) + 1,)

        out = vector(dtype=dtype, shape=out_shape)
        return Apply(self, [in1, in2, full_mode], [out])

    def perform(self, node, inputs, outputs):
        # We use numpy_convolve as that's what scipy would use if method="direct" was passed.
        # And mode != "same", which this Op doesn't cover anyway.
        in1, in2, full_mode = inputs
        outputs[0][0] = numpy_convolve(in1, in2, mode="full" if full_mode else "valid")

    def infer_shape(self, fgraph, node, shapes):
        _, _, full_mode = node.inputs
        in1_shape, in2_shape, _ = shapes
        n = in1_shape[0]
        k = in2_shape[0]
        shape_valid = maximum(n, k) - minimum(n, k) + 1
        shape_full = n + k - 1
        shape = switch(full_mode, shape_full, shape_valid)
        return [[shape]]

    def connection_pattern(self, node):
        return [[True], [True], [False]]

    def L_op(self, inputs, outputs, output_grads):
        in1, in2, full_mode = inputs
        [grad] = output_grads

        n = in1.shape[0]
        k = in2.shape[0]

        # If mode is "full", or mode is "valid" and k >= n, then in1_bar mode should use "valid" convolve
        # The expression below is equivalent to ~(full_mode | (k >= n))
        full_mode_in1_bar = ~full_mode & (k < n)
        # If mode is "full", or mode is "valid" and n >= k, then in2_bar mode should use "valid" convolve
        # The expression below is equivalent to ~(full_mode | (n >= k))
        full_mode_in2_bar = ~full_mode & (n < k)

        return [
            self(grad, in2[::-1], full_mode_in1_bar),
            self(grad, in1[::-1], full_mode_in2_bar),
            DisconnectedType()(),
        ]

    def c_code_cache_version(self):
        return (2,)

    def c_code(self, node, name, inputs, outputs, sub):
        in1, in2, full_mode = inputs
        [out] = outputs

        code = f"""
        {{
            PyArrayObject* in2_flipped_view = NULL;

            if (PyArray_NDIM({in1}) != 1 || PyArray_NDIM({in2}) != 1) {{
                PyErr_SetString(PyExc_ValueError, "Convolve1d C code expects 1D arrays.");
                {sub['fail']};
            }}

            npy_intp n_in2 = PyArray_DIM({in2}, 0);

            // Create a reversed view of in2
            if (n_in2 == 0) {{
                PyErr_SetString(PyExc_ValueError, "Convolve1d: second input (kernel) cannot be empty.");
                {sub['fail']};
            }} else {{
                npy_intp view_dims[1];
                view_dims[0] = n_in2;

                npy_intp view_strides[1];
                view_strides[0] = -PyArray_STRIDES({in2})[0];

                void* view_data = (char*)PyArray_DATA({in2}) + (n_in2 - 1) * PyArray_STRIDES({in2})[0];

                Py_INCREF(PyArray_DESCR({in2}));
                in2_flipped_view = (PyArrayObject*)PyArray_NewFromDescr(
                    Py_TYPE({in2}),
                    PyArray_DESCR({in2}),
                    1,  // ndim
                    view_dims,
                    view_strides,
                    view_data,
                    (PyArray_FLAGS({in2}) & ~NPY_ARRAY_WRITEABLE),
                    NULL
                );

                if (!in2_flipped_view) {{
                    PyErr_SetString(PyExc_RuntimeError, "Failed to create flipped kernel view for Convolve1d.");
                    {sub['fail']};
                }}

                Py_INCREF({in2});
                if (PyArray_SetBaseObject(in2_flipped_view, (PyObject*){in2}) < 0) {{
                    Py_DECREF({in2}); // SetBaseObject failed, release the extra INCREF
                    Py_DECREF(in2_flipped_view);
                    in2_flipped_view = NULL;
                    PyErr_SetString(PyExc_RuntimeError, "Failed to set base object for flipped kernel view in Convolve1d.");
                    {sub['fail']};
                }}
                PyArray_UpdateFlags(in2_flipped_view, (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS));
            }}

            // TODO: Use lower level implementation that allows reusing the output buffer
            Py_XDECREF({out});
            {out} = (PyArrayObject*) PyArray_Correlate2((PyObject*){in1}, (PyObject*)in2_flipped_view, {full_mode} ? 2 : 0);
            Py_XDECREF(in2_flipped_view); // Clean up the view if correlate fails
            if (!{out}) {{
                // PyArray_Correlate already set an error
                {sub['fail']};
            }}
        }}
        """
        return code


blockwise_convolve_1d = Blockwise(Convolve1d())


def convolve1d(
    in1: "TensorLike",
    in2: "TensorLike",
    mode: Literal["full", "valid", "same"] = "full",
) -> TensorVariable:
    """Convolve two one-dimensional arrays.

    Convolve in1 and in2, with the output size determined by the mode argument.

    Parameters
    ----------
    in1 : (..., N,) tensor_like
        First input.
    in2 : (..., M,) tensor_like
        Second input.
    mode : {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        - 'full': The output is the full discrete linear convolution of the inputs, with shape (..., N+M-1,).
        - 'valid': The output consists only of elements that do not rely on zero-padding, with shape (..., max(N, M) - min(N, M) + 1,).
        - 'same': The output is the same size as in1, centered with respect to the 'full' output.

    Returns
    -------
    out: tensor_variable
        The discrete linear convolution of in1 with in2.

    """
    in1 = as_tensor_variable(in1)
    in2 = as_tensor_variable(in2)

    if mode == "same":
        # We implement "same" as "valid" with padded `in1`.
        in1_batch_shape = tuple(in1.shape)[:-1]
        zeros_left = in2.shape[-1] // 2
        zeros_right = (in2.shape[-1] - 1) // 2
        in1 = join(
            -1,
            zeros((*in1_batch_shape, zeros_left), dtype=in2.dtype),
            in1,
            zeros((*in1_batch_shape, zeros_right), dtype=in2.dtype),
        )
        mode = "valid"

    full_mode = as_scalar(np.bool_(mode == "full"))
    return cast(TensorVariable, blockwise_convolve_1d(in1, in2, full_mode))
