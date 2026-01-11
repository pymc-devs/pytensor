from typing import TYPE_CHECKING, Literal
from typing import cast as type_cast

import numpy as np
from numpy import convolve as numpy_convolve
from scipy.signal import convolve as scipy_convolve

from pytensor.gradient import disconnected_type
from pytensor.graph import Apply, Constant
from pytensor.graph.op import Op
from pytensor.link.c.op import COp
from pytensor.scalar import as_scalar
from pytensor.scalar.basic import upcast
from pytensor.tensor.basic import as_tensor_variable, join, zeros
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.math import maximum, minimum, switch
from pytensor.tensor.pad import pad
from pytensor.tensor.subtensor import flip
from pytensor.tensor.type import tensor
from pytensor.tensor.variable import TensorVariable


if TYPE_CHECKING:
    from pytensor.tensor import TensorLike


class AbstractConvolveNd:
    __props__ = ()
    ndim: int

    @property
    def gufunc_signature(self):
        data_signature = ",".join([f"n{i}" for i in range(self.ndim)])
        kernel_signature = ",".join([f"k{i}" for i in range(self.ndim)])
        output_signature = ",".join([f"o{i}" for i in range(self.ndim)])

        return f"({data_signature}),({kernel_signature}),()->({output_signature})"

    def make_node(self, in1, in2, full_mode):
        in1 = as_tensor_variable(in1)
        in2 = as_tensor_variable(in2)
        full_mode = as_scalar(full_mode)

        ndim = self.ndim
        if not (in1.ndim == ndim and in2.ndim == self.ndim):
            raise ValueError(
                f"Convolution inputs must have ndim={ndim}, got: in1={in1.ndim}, in2={in2.ndim}"
            )
        if not full_mode.dtype == "bool":
            raise ValueError("Convolution full_mode flag must be a boolean type")

        match full_mode:
            case Constant():
                static_mode = "full" if full_mode.data else "valid"
            case _:
                static_mode = None

        if static_mode is None:
            out_shape = (None,) * ndim
        else:
            out_shape = []
            # TODO: Raise if static shapes are not valid (one input size doesn't dominate the other)
            for n, k in zip(in1.type.shape, in2.type.shape):
                if n is None or k is None:
                    out_shape.append(None)
                elif static_mode == "full":
                    out_shape.append(
                        n + k - 1,
                    )
                else:  # mode == "valid":
                    out_shape.append(
                        max(n, k) - min(n, k) + 1,
                    )
            out_shape = tuple(out_shape)

        dtype = upcast(in1.dtype, in2.dtype)

        out = tensor(dtype=dtype, shape=out_shape)
        return Apply(self, [in1, in2, full_mode], [out])

    def infer_shape(self, fgraph, node, shapes):
        _, _, full_mode = node.inputs
        in1_shape, in2_shape, _ = shapes
        out_shape = [
            switch(full_mode, n + k - 1, maximum(n, k) - minimum(n, k) + 1)
            for n, k in zip(in1_shape, in2_shape)
        ]

        return [out_shape]

    def connection_pattern(self, node):
        return [[True], [True], [False]]

    def L_op(self, inputs, outputs, output_grads):
        in1, in2, full_mode = inputs
        [grad] = output_grads

        n = in1.shape
        k = in2.shape
        # Note: this assumes the shape of one input dominates the other over all dimensions (which is required for a valid forward)

        # If mode is "full", or mode is "valid" and k >= n, then in1_bar mode should use "valid" convolve
        # The expression below is equivalent to ~(full_mode | (k >= n))
        full_mode_in1_bar = ~full_mode & (k < n).any()
        # If mode is "full", or mode is "valid" and n >= k, then in2_bar mode should use "valid" convolve
        # The expression below is equivalent to ~(full_mode | (n >= k))
        full_mode_in2_bar = ~full_mode & (n < k).any()

        return [
            self(grad, flip(in2), full_mode_in1_bar),
            self(grad, flip(in1), full_mode_in2_bar),
            disconnected_type(),
        ]


class Convolve1d(AbstractConvolveNd, COp):  # type: ignore[misc]
    __props__ = ()
    ndim = 1

    def perform(self, node, inputs, outputs):
        # We use numpy_convolve as that's what scipy would use if method="direct" was passed.
        # And mode != "same", which this Op doesn't cover anyway.
        in1, in2, full_mode = inputs
        outputs[0][0] = numpy_convolve(in1, in2, mode="full" if full_mode else "valid")

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
                {sub["fail"]};
            }}

            npy_intp n_in2 = PyArray_DIM({in2}, 0);

            // Create a reversed view of in2
            if (n_in2 == 0) {{
                PyErr_SetString(PyExc_ValueError, "Convolve1d: second input (kernel) cannot be empty.");
                {sub["fail"]};
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
                    {sub["fail"]};
                }}

                Py_INCREF({in2});
                if (PyArray_SetBaseObject(in2_flipped_view, (PyObject*){in2}) < 0) {{
                    Py_DECREF({in2}); // SetBaseObject failed, release the extra INCREF
                    Py_DECREF(in2_flipped_view);
                    in2_flipped_view = NULL;
                    PyErr_SetString(PyExc_RuntimeError, "Failed to set base object for flipped kernel view in Convolve1d.");
                    {sub["fail"]};
                }}
                PyArray_UpdateFlags(in2_flipped_view, (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS));
            }}

            // TODO: Use lower level implementation that allows reusing the output buffer
            Py_XDECREF({out});
            {out} = (PyArrayObject*) PyArray_Correlate2((PyObject*){in1}, (PyObject*)in2_flipped_view, {full_mode} ? 2 : 0);
            Py_XDECREF(in2_flipped_view); // Clean up the view if correlate fails
            if (!{out}) {{
                // PyArray_Correlate already set an error
                {sub["fail"]};
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
    return type_cast(TensorVariable, blockwise_convolve_1d(in1, in2, full_mode))


class Convolve2d(AbstractConvolveNd, Op):  # type: ignore[misc]
    __props__ = ("method",)  # type: ignore[assignment]
    ndim = 2

    def __init__(self, method: Literal["direct", "fft", "auto"] = "auto"):
        self.method = method

    def perform(self, node, inputs, outputs):
        in1, in2, full_mode = inputs
        mode = "full" if full_mode else "valid"
        outputs[0][0] = scipy_convolve(in1, in2, mode=mode, method=self.method)


def convolve2d(
    in1: "TensorLike",
    in2: "TensorLike",
    mode: Literal["full", "valid", "same"] = "full",
    boundary: Literal["fill", "wrap", "symm"] = "fill",
    fillvalue: float | int = 0,
    method: Literal["direct", "fft", "auto"] = "auto",
) -> TensorVariable:
    """Convolve two two-dimensional arrays.

    Convolve in1 and in2, with the output size determined by the mode argument.

    Parameters
    ----------
    in1 : (..., N, M) tensor_like
        First input.
    in2 : (..., K, L) tensor_like
        Second input.
    mode : {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        - 'full': The output is the full discrete linear convolution of the inputs, with shape (..., N+K-1, M+L-1).
        - 'valid': The output consists only of elements that do not rely on zero-padding, with shape (..., max(N, K) - min(N, K) + 1, max(M, L) - min(M, L) + 1).
        - 'same': The output is the same size as in1, centered with respect to the 'full' output.
    boundary : {'fill', 'wrap', 'symm'}, optional
        A string indicating how to handle boundaries:
        - 'fill': Pads the input arrays with fillvalue.
        - 'wrap': Circularly wraps the input arrays.
        - 'symm': Symmetrically reflects the input arrays.
    fillvalue : float or int, optional
        The value to use for padding when boundary is 'fill'. Default is 0.
    method : str, one of 'direct', 'fft', or 'auto'
        Computation method to use. 'direct' uses direct convolution, 'fft' uses FFT-based convolution,
        and 'auto' lets the implementation choose the best method at runtime.

    Returns
    -------
    out: tensor_variable
        The discrete linear convolution of in1 with in2.

    """
    in1 = as_tensor_variable(in1)
    in2 = as_tensor_variable(in2)
    ndim = max(in1.type.ndim, in2.type.ndim)

    def _pad_input(input_tensor, pad_width):
        if boundary == "fill":
            return pad(
                input_tensor,
                pad_width=pad_width,
                mode="constant",
                constant_values=fillvalue,
            )
        if boundary == "wrap":
            return pad(input_tensor, pad_width=pad_width, mode="wrap")
        if boundary == "symm":
            return pad(input_tensor, pad_width=pad_width, mode="symmetric")
        raise ValueError(f"Unsupported boundary mode: {boundary}")

    if mode == "same":
        # Same mode is implemented as "valid" with a padded input.
        pad_width = zeros((ndim, 2), dtype="int64")
        pad_width = pad_width[-2, 0].set(in2.shape[-2] // 2)
        pad_width = pad_width[-2, 1].set((in2.shape[-2] - 1) // 2)
        pad_width = pad_width[-1, 0].set(in2.shape[-1] // 2)
        pad_width = pad_width[-1, 1].set((in2.shape[-1] - 1) // 2)
        in1 = _pad_input(in1, pad_width)
        mode = "valid"

    if mode != "valid" and (boundary != "fill" or fillvalue != 0):
        # We use a valid convolution on an appropriately padded kernel
        *_, k, l = in2.shape

        pad_width = zeros((ndim, 2), dtype="int64")
        pad_width = pad_width[-2, :].set(k - 1)
        pad_width = pad_width[-1, :].set(l - 1)
        in1 = _pad_input(in1, pad_width)

        mode = "valid"

    full_mode = as_scalar(np.bool_(mode == "full"))
    return type_cast(
        TensorVariable, Blockwise(Convolve2d(method=method))(in1, in2, full_mode)
    )
