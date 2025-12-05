import builtins
import warnings
from collections.abc import Sequence
from textwrap import dedent
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

from pytensor import config, printing
from pytensor import scalar as ps
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op
from pytensor.graph.replace import _vectorize_node
from pytensor.link.c.op import COp
from pytensor.link.c.params_type import ParamsType
from pytensor.printing import pprint
from pytensor.raise_op import Assert
from pytensor.scalar.basic import BinaryScalarOp
from pytensor.tensor import TensorLike
from pytensor.tensor.basic import (
    alloc,
    arange,
    as_tensor_variable,
    cast,
    concatenate,
    constant,
    expand_dims,
    stack,
    switch,
)
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import (
    CAReduce,
    Elemwise,
    get_normalized_batch_axes,
    scalar_elemwise,
)
from pytensor.tensor.shape import shape, specify_shape
from pytensor.tensor.type import (
    DenseTensorType,
    complex_dtypes,
    continuous_dtypes,
    discrete_dtypes,
    float_dtypes,
    int_dtypes,
    tensor,
    uint_dtypes,
)
from pytensor.tensor.utils import normalize_reduce_axis
from pytensor.tensor.variable import (
    TensorVariable,
    _tensor_py_operators,
)


if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike

# We capture the builtins that we are going to replace to follow the numpy API
_abs = builtins.abs


if int(config.tensor__cmp_sloppy) > 1:
    # This config variable is a quick-and-dirty way to get low-precision
    # comparisons.  For a more precise setting of these tolerances set
    # them explicitly in your user code by assigning, for example,
    # "pytensor.tensor.math.float32_atol = ..."

    # When config.tensor__cmp_sloppy>1 we are even more sloppy. This is
    # useful to test the GPU as they don't use extended precision and
    # this cause some difference bigger then the normal sloppy.
    float16_atol = 1e-2
    float16_rtol = 5e-2

    float32_atol = 5e-4
    float32_rtol = 1e-3

    float64_rtol = 1e-4
    float64_atol = 1e-3
elif int(config.tensor__cmp_sloppy):
    float16_atol = 5e-3
    float16_rtol = 1e-2

    float32_atol = 1e-4
    float32_rtol = 1e-3

    float64_rtol = 1e-4
    float64_atol = 1e-3
else:
    # If you change those value in test don't forget to put them back
    # when the test end.  Don't forget the case when the test fail.
    float16_atol = 1e-3
    float16_rtol = 1e-3

    float32_atol = 1e-5
    float32_rtol = 1e-5

    # defaults in numpy.allclose
    # Don't be more strict then numpy rtol
    # It cause useless error.
    float64_rtol = 1.0000000000000001e-05
    float64_atol = 1e-8


def __getattr__(name):
    if name == "MaxAndArgmax":
        raise AttributeError(
            "The class `MaxandArgmax` has been deprecated. Call `Max` and `Argmax` seperately as an alternative."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_atol_rtol(a, b):
    tiny = ("float16",)
    narrow = ("float32", "complex64")
    if (str(a.dtype) in tiny) or (str(b.dtype) in tiny):
        atol = float16_atol
        rtol = float16_rtol
    elif (str(a.dtype) in narrow) or (str(b.dtype) in narrow):
        atol = float32_atol
        rtol = float32_rtol
    else:
        atol = float64_atol
        rtol = float64_rtol
    return atol, rtol


def _allclose(a, b, rtol=None, atol=None):
    a = np.asarray(a)
    b = np.asarray(b)
    atol_, rtol_ = _get_atol_rtol(a, b)
    if rtol is not None:
        rtol_ = rtol
    if atol is not None:
        atol_ = atol

    return np.allclose(a, b, atol=atol_, rtol=rtol_)


class Argmax(COp):
    """
    Calculate the argmax over a given axis or over all axes.
    """

    nin = 2  # tensor, axis
    nout = 1
    E_axis = "invalid axis"
    __props__ = ("axis",)
    _f16_ok = True

    params_type = ParamsType(c_axis=ps.int64)

    def __init__(self, axis):
        if axis is not None:
            axis = tuple(sorted(axis))
        self.axis = axis

    def get_params(self, node):
        if self.axis is not None and len(self.axis) == 1:
            c_axis = np.int64(self.axis[0])
        else:
            # The value here doesn't matter, it won't be used
            c_axis = 0
        return self.params_type.get_params(c_axis=c_axis)

    def make_node(self, x):
        x = as_tensor_variable(x)
        if self.axis is None:
            all_axes = list(range(x.ndim))
        else:
            all_axes = self.axis
        inputs = [x]

        # We keep the original broadcastable flags for dimensions on which
        # we do not perform the argmax.
        out_shape = tuple(s for i, s in enumerate(x.type.shape) if i not in all_axes)
        outputs = [tensor(dtype="int64", shape=out_shape, name="argmax")]
        return Apply(self, inputs, outputs)

    def prepare_node(self, node, storage_map, compute_map, impl):
        if len(node.inputs) == 2:
            raise ValueError(
                "You are trying to compile a graph with an old Argmax node.  Either reoptimize your graph or rebuild it to get the new node format."
            )

    def perform(self, node, inp, outs):
        (x,) = inp
        axes = self.axis
        (max_idx,) = outs
        if axes is None:
            axes = tuple(range(x.ndim))
        # Numpy does not support multiple axes for argmax
        # Work around
        keep_axes = np.array([i for i in range(x.ndim) if i not in axes], dtype="int64")
        # Not-reduced axes in front
        transposed_x = np.transpose(
            x, np.concatenate((keep_axes, np.asarray(axes, dtype="int64")))
        )
        kept_shape = transposed_x.shape[: len(keep_axes)]
        reduced_shape = transposed_x.shape[len(keep_axes) :]
        new_shape = (*kept_shape, np.prod(reduced_shape, dtype="int64"))
        reshaped_x = transposed_x.reshape(new_shape)

        max_idx[0] = np.asarray(np.argmax(reshaped_x, axis=-1), dtype="int64")

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (argmax,) = out
        fail = sub["fail"]
        params = sub["params"]
        if self.axis is None:
            axis_code = "axis = NPY_RAVEL_AXIS;"
        else:
            if len(self.axis) != 1:
                raise NotImplementedError()
            # params is only used here for now
            axis_code = f"""
            axis = {params}->c_axis;
            if(axis > PyArray_NDIM({x})-1 || axis < -PyArray_NDIM({x})){{
                PyErr_SetString(PyExc_ValueError,
                "Argmax, bad axis argument");
                {fail}
            }}
            """
        return f"""
        int axis;

        Py_CLEAR({argmax});//todo pass them as out parameter.
        {axis_code}

        {argmax} = (PyArrayObject*)PyArray_ArgMax({x}, axis, NULL);
        if({argmax} == NULL){{
            {fail};
        }}
        if(!PyArray_CheckExact({argmax})){{
            {argmax} = (PyArrayObject*)PyArray_FromAny((PyObject*){argmax}, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
            if({argmax} == NULL){{
                {fail};
            }}
        }}
        if(PyArray_TYPE({argmax}) != NPY_INT64){{
            PyObject * tmp = PyArray_Cast({argmax}, NPY_INT64);
            if (NULL == tmp){{
                {fail};
            }}
            Py_DECREF({argmax});
            {argmax} = (PyArrayObject*)tmp;
        }}
        """

    def c_code_cache_version(self):
        return (3,)

    def infer_shape(self, fgraph, node, shapes):
        (ishape,) = shapes
        if self.axis is None:
            return [()]
        rval = tuple(
            ishape[i]
            for (i, b) in enumerate(node.inputs[0].type.broadcastable)
            if i not in self.axis
        )
        return [rval]

    def R_op(self, inputs, eval_points):
        raise ValueError("Argmax is non-diifferentiable")

    def grad(self, inp, grads):
        (x,) = inp

        return [x.zeros_like()]


def argmax(x: TensorLike, axis=None, keepdims: bool = False):
    """
    Returns indices of maximum elements obtained by iterating over given axis.

    When axis is None (the default value), the argmax is performed
    over the flattened tensor.

    Parameters
    ----------
    x: TensorLike
        Array on which to compute argmax
    axis:
        Axis along which to compute argmax. Unlike numpy multiple partial axis are supported.
    keepdims : bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    Returns
    -------
    TensorVariable
        TensorVariable representing the argmax operation

    """
    x = as_tensor_variable(x)
    axis = normalize_reduce_axis(axis, ndim=x.type.ndim)
    out = Argmax(axis)(x)

    if keepdims:
        out = makeKeepDims(x, out, axis)

    return out


@_vectorize_node.register(Argmax)
def vectorize_argmax_node(op, node, batch_x):
    core_ndim = node.inputs[0].type.ndim
    batch_ndim = batch_x.type.ndim - core_ndim

    if not batch_ndim:
        return node.op.make_node(batch_x)

    batch_axes = get_normalized_batch_axes(op.axis, core_ndim, batch_ndim)
    return type(op)(axis=batch_axes).make_node(batch_x)


def makeKeepDims(x, y, axis):
    """
    Reintroduces in y with length one the axes of x which have been left out
    in a prior reduction of x. With this option, the resulting tensor will
    broadcast correctly against the original tensor x.

    """
    x = as_tensor_variable(x)
    if axis is None:
        axis = list(range(x.type.ndim))
    return expand_dims(y, axis)


def max_and_argmax(a, axis=None, keepdims=False):
    """
    Returns maximum elements and their indices obtained by iterating over
    given axis.

    When axis is None (the default value), the max is performed
    over the flattened tensor.

    Parameters
    ----------
    keepdims : bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """
    # Check axis and convert it to a Python list of integers.
    # Axis will be used as an op param of Max and Argmax.
    return [
        max(a, axis=axis, keepdims=keepdims),
        argmax(a, axis=axis, keepdims=keepdims),
    ]


class FixedOpCAReduce(CAReduce):
    def __str__(self):
        if self.dtype != self.acc_dtype:
            return f"{type(self).__name__}{{{self._axis_str()}, acc={self.acc_dtype}}}"
        else:
            return f"{type(self).__name__}{{{self._axis_str()}}}"


class NonZeroDimsCAReduce(FixedOpCAReduce):
    def _c_all(self, node, name, input_names, output_names, sub):
        setup, alloc, loop, cast = super()._c_all(
            node, name, input_names, output_names, sub
        )

        # We add an additional check for zero-sized dimensions (This seems like
        # something that could enabled in `elemwise_cgen.make_checks`.)
        [iname] = input_names

        axis = self.axis
        if axis is None:
            axis = list(range(len(node.inputs[0].type.broadcastable)))

        pattern = [0] * len(node.inputs[0].broadcastable)
        for i in axis:
            pattern[i] = 1

        pattern_ = str(pattern)[1:-1]

        setup = f"int tosum[]={{{pattern_}}};" + setup
        alloc += dedent(
            f"""
            for(int i=0;i<PyArray_NDIM({iname});i++){{
                if(PyArray_DIMS({iname})[i]==0 && tosum[i]){{
                    PyErr_Format(PyExc_ValueError,
                        "Input of CAReduce{{{node.op.scalar_op}}} has zero-size on axis %%d",i);
                    {sub["fail"]};
                }}
            }}
            """
        )
        return setup, alloc, loop, cast


class Max(NonZeroDimsCAReduce):
    nfunc_spec = ("max", 1, 1)

    def __init__(self, axis):
        super().__init__(ps.maximum, axis)

    def clone(self, **kwargs):
        axis = kwargs.get("axis", self.axis)
        return type(self)(axis=axis)

    def L_op(self, inputs, outputs, grads):
        # The strict sense mathematical gradient of the maximum function is
        # not calculated here for it is not defined at every point where some
        # coordinates are identical. However, since the latter set has null
        # Lebesgue measure, the result may be interpreted as weak gradient.

        # @note: This function should work correctly for L{vector}s.
        # (x, y), (gz, gw)
        # gz*dz/dx + gw*dw/dx, gz*dz/dy + gw*dw/dy
        # gMax * dMax/dx + gArgMax * dArgMax/dx,
        # gMax * dMax/daxis + gArgMax * dArgMax/daxis
        # g_max has one less dimension than x, so you need to complete
        # g_max to x's shape when axis=0 the broadcasting mechanism
        # does it automatically
        [x] = inputs
        [out] = outputs
        [g_out] = grads

        axis = tuple(range(x.ndim)) if self.axis is None else self.axis
        out_pad = expand_dims(out, axis)
        g_out_pad = expand_dims(g_out, axis)

        # Set the grad to the correct position.
        g_x = eq(out_pad, x) * g_out_pad
        return (g_x,)

    def R_op(self, inputs, eval_points):
        [x] = inputs
        if eval_points[0] is None:
            return [None]
        axis = tuple(range(x.ndim) if self.axis is None else self.axis)
        if isinstance(axis, int):
            axis = [axis]
        if len(axis) != 1:
            raise NotImplementedError("R_op supported for max only for one axis!")
        if axis[0] > 1:
            raise NotImplementedError("R_op supported for max only when axis is 0 or 1")
        if inputs[0].ndim != 2:
            raise NotImplementedError(
                "R_op supported for max only when input is a matrix"
            )
        max_pos = Argmax(self.axis)(*inputs)
        if self.axis[0] == 0:
            return [eval_points[0][max_pos, arange(eval_points[0].shape[1])]]
        else:
            return [eval_points[0][arange(eval_points[0].shape[0]), max_pos]]


class Min(NonZeroDimsCAReduce):
    nfunc_spec = ("min", 1, 1)

    def __init__(self, axis):
        super().__init__(ps.minimum, axis)

    def clone(self, **kwargs):
        axis = kwargs.get("axis", self.axis)
        return type(self)(axis=axis)


def max(x, axis=None, keepdims=False):
    """
    Returns maximum elements obtained by iterating over given axis.

    When axis is None (the default value), the max is performed
    over the flattened tensor.

    Parameters
    ----------
    keepdims: bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    Notes
    -----
    We return an error as numpy when we reduce a dim with a shape of 0.

    """
    out = Max(axis=axis)(x)

    if keepdims:
        out = makeKeepDims(x, out, axis)
    return out


def min(x, axis=None, keepdims=False):
    """
    Returns minimum elements obtained by iterating over given axis.

    When axis is None (the default value), the min is performed
    over the flattened tensor.

    Parameters
    ----------
    keepdims: bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """
    x = as_tensor_variable(x)
    str_x_type = str(x.dtype)
    if str_x_type.startswith("float") or str_x_type in int_dtypes:
        return -max(-x, axis=axis, keepdims=keepdims)
    elif str_x_type in uint_dtypes:
        itype = np.iinfo(x.dtype)
        max_val = np.array(itype.max, dtype=itype.dtype)
        return max_val - max(max_val - x, axis=axis, keepdims=keepdims)
    elif str_x_type == "bool":
        return ~max(~x, axis=axis, keepdims=keepdims)
    else:
        # Be careful about unsigned integers, complex
        raise NotImplementedError()


def argmin(x, axis=None, keepdims=False):
    """
    Returns indices of minimum elements obtained by iterating over given axis.

    When axis is None (the default value), the argmin is performed
    over the flattened tensor.

    Parameters
    ----------
    keepdims: bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """
    x = as_tensor_variable(x)
    str_x_type = str(x.dtype)
    if str_x_type.startswith("float") or str_x_type in int_dtypes:
        return argmax(-x, axis=axis, keepdims=keepdims)
    elif str_x_type in uint_dtypes:
        itype = np.iinfo(x.dtype)
        return argmax(itype.max - x, axis=axis, keepdims=keepdims)
    elif str_x_type == "bool":
        return argmax(~x, axis=axis, keepdims=keepdims)
    else:
        # Be careful about unsigned integers, complex
        raise NotImplementedError()


def smallest(*args):
    """
    Return the [elementwise] smallest of a variable number of arguments.

    Like python's min.

    """
    if len(args) == 2:
        a, b = args
        return switch(a < b, a, b)
    else:
        return min(stack(args), axis=0)


def largest(*args):
    """
    Return the [elementwise] largest of a variable number of arguments.

    Like python's max.

    """
    if len(args) == 2:
        a, b = args
        return switch(a > b, a, b)
    else:
        return max(stack(args), axis=0)


def isposinf(x):
    """
    Return if the input variable has positive infinity element

    """
    return eq(x, np.inf)


def isneginf(x):
    """
    Return if the input variable has negative infinity element

    """
    return eq(x, -np.inf)


@scalar_elemwise
def lt(a, b):
    """a < b

    Computes element-wise less than comparison between two tensors.

    Parameters
    ----------
    a : TensorLike
        First input tensor
    b : TensorLike
        Second input tensor

    Returns
    -------
    TensorVariable
        Output tensor of type bool, with 1 (True) where a < b,
        and 0 (False) elsewhere.

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> y = pt.vector("y")
    >>> f = pytensor.function([x, y], pt.lt(x, y))
    >>> f([1, 2, 3], [2, 2, 2])
    array([ True, False, False])
    """


@scalar_elemwise
def gt(a, b):
    """a > b

    Computes element-wise greater than comparison between two tensors.

    Parameters
    ----------
    a : TensorLike
        First input tensor
    b : TensorLike
        Second input tensor

    Returns
    -------
    TensorVariable
        Output tensor of type bool, with 1 (True) where a > b,
        and 0 (False) elsewhere.

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> y = pt.vector("y")
    >>> f = pytensor.function([x, y], pt.gt(x, y))
    >>> f([1, 2, 3], [0, 2, 4])
    array([ True, False, False])
    """


@scalar_elemwise
def le(a, b):
    """a <= b

    Computes element-wise less than or equal comparison between two tensors.

    Parameters
    ----------
    a : TensorLike
        First input tensor
    b : TensorLike
        Second input tensor

    Returns
    -------
    TensorVariable
        Output tensor of type bool, with 1 (True) where a <= b,
        and 0 (False) elsewhere.

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> y = pt.vector("y")
    >>> f = pytensor.function([x, y], pt.le(x, y))
    >>> f([1, 2, 3], [2, 2, 2])
    array([ True,  True, False])
    """


@scalar_elemwise
def ge(a, b):
    """a >= b

    Computes element-wise greater than or equal comparison between two tensors.

    Parameters
    ----------
    a : TensorLike
        First input tensor
    b : TensorLike
        Second input tensor

    Returns
    -------
    TensorVariable
        Output tensor of type bool, with 1 (True) where a >= b,
        and 0 (False) elsewhere.

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> y = pt.vector("y")
    >>> f = pytensor.function([x, y], pt.ge(x, y))
    >>> f([1, 2, 3], [0, 2, 4])
    array([ True,  True, False])
    """


@scalar_elemwise
def eq(a, b):
    """a == b

    Computes element-wise equality between two tensors.

    Parameters
    ----------
    a : TensorLike
        First input tensor
    b : TensorLike
        Second input tensor

    Returns
    -------
    TensorVariable
        Output tensor of type bool, with 1 (True) where elements are equal,
        and 0 (False) elsewhere.

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> import numpy as np
    >>> x = pt.vector("x")
    >>> y = pt.vector("y")
    >>> f = pytensor.function([x, y], pt.eq(x, y))
    >>> f([1, 2, 3], [1, 4, 3])
    array([ True, False,  True])

    Notes
    -----
    Due to Python rules, it is not possible to overload the equality symbol `==` for hashable objects and have it return something other than a boolean,
    so `eq` must always be used to compute the Elemwise equality of TensorVariables (which are hashable).
    """


@scalar_elemwise
def neq(a, b):
    """a != b

    Computes element-wise inequality comparison between two tensors.

    Parameters
    ----------
    a : TensorLike
        First input tensor
    b : TensorLike
        Second input tensor

    Returns
    -------
    TensorVariable
        Output tensor of type bool, with 1 (True) where a != b,
        and 0 (False) elsewhere.

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> y = pt.vector("y")
    >>> f = pytensor.function([x, y], pt.neq(x, y))
    >>> f([1, 2, 3], [1, 4, 3])
    array([False,  True, False])

    Notes
    -----
    Due to Python rules, it is not possible to overload the inequality symbol `!=` for hashable objects and have it return something other than a boolean,
    so `neq` must always be used to compute the Elemwise inequality of TensorVariables (which are hashable).
    """


@scalar_elemwise
def isnan(a):
    """isnan(a)

    Computes element-wise detection of NaN values.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor of type bool, with 1 (True) where elements are NaN,
        and 0 (False) elsewhere.

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> import numpy as np
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.isnan(x))
    >>> f([1, np.nan, 3])
    array([False,  True, False])
    """


# Rename isnan to isnan_ to allow to bypass it when not needed.
# glibc 2.23 don't allow isnan on int, so we remove it from the graph.
isnan_ = isnan


def isnan(a):
    """isnan(a)"""
    a = as_tensor_variable(a)
    if a.dtype in discrete_dtypes:
        return alloc(
            np.asarray(False, dtype="bool"), *[a.shape[i] for i in range(a.ndim)]
        )
    return isnan_(a)


@scalar_elemwise
def isinf(a):
    """isinf(a)

    Computes element-wise detection of infinite values.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor of type bool, with 1 (True) where elements are infinite,
        and 0 (False) elsewhere.

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> import numpy as np
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.isinf(x))
    >>> f([1, np.inf, -np.inf, 3])
    array([False,  True,  True, False])
    """


# Rename isnan to isnan_ to allow to bypass it when not needed.
# glibc 2.23 don't allow isnan on int, so we remove it from the graph.
isinf_ = isinf


def isinf(a):
    """isinf(a)"""
    a = as_tensor_variable(a)
    if a.dtype in discrete_dtypes:
        return alloc(
            np.asarray(False, dtype="bool"), *[a.shape[i] for i in range(a.ndim)]
        )
    return isinf_(a)


def allclose(a, b, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    """
    Implement Numpy's ``allclose`` on tensors.

    ``absolute(a - b) <= (atol + rtol * absolute(b))``

    Parameters
    ----------
    a : TensorLike
        Input to compare.
    b : TensorLike
        Input to compare.
    rtol : float
        The relative tolerance parameter.
    atol : float
        The absolute tolerance parameter.
    equal_nan: bool
        Whether to consider nan's in the same place to be close.

    Returns
    -------
    bool
        A boolean value (of type int8 returned by the tensor elementwise `all`
        function) whether all elements in a and b are in the tolerance range
        defined above.

    Notes
    -----
    Not a symmetric equation. See Numpy's documentation.

    """
    return all(isclose(a, b, rtol, atol, equal_nan))


def isclose(a, b, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    """
    Implements Numpy's ``isclose`` on tensors.

    The tolerance values are positive, typically very small numbers. The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    ``absolute(a - b) <= (atol + rtol * absolute(b))``

    Parameters
    ----------
    a : TensorLike
        Input to compare.
    b : TensorLike
        Input to compare.
    rtol : float
        The relative tolerance parameter.
    atol : float
        The absolute tolerance parameter.
    equal_nan : bool
        Whether to consider nan's in the same place to be close

    Returns
    -------
    int8
        A boolean (int8) array where two arrays are element-wise equal
        within a tolerance.

    Notes
    -----
    Not a symmetric equation. See Numpy's documentation.

    Examples
    --------
    >>> import pytensor
    >>> import numpy as np
    >>> a = np.array([1e10, 1e-7], dtype="float64")
    >>> b = np.array([1.00001e10, 1e-8], dtype="float64")
    >>> pytensor.tensor.isclose(a, b).eval()
    array([ True, False])
    >>> a = np.array([1e10, 1e-8], dtype="float64")
    >>> b = np.array([1.00001e10, 1e-9], dtype="float64")
    >>> pytensor.tensor.isclose(a, b).eval()
    array([ True,  True])
    >>> a = np.array([1e10, 1e-8], dtype="float64")
    >>> b = np.array([1.0001e10, 1e-9], dtype="float64")
    >>> pytensor.tensor.isclose(a, b).eval()
    array([False,  True])
    >>> a = np.array([1.0, np.nan], dtype="float64")
    >>> b = np.array([1.0, np.nan], dtype="float64")
    >>> pytensor.tensor.isclose(a, b).eval()
    array([ True, False])
    >>> a = np.array([1.0, np.nan], dtype="float64")
    >>> b = np.array([1.0, np.nan], dtype="float64")
    >>> pytensor.tensor.isclose(a, b, equal_nan=True).eval()
    array([ True,  True])
    >>> a = np.array([1.0, np.inf], dtype="float64")
    >>> b = np.array([1.0, -np.inf], dtype="float64")
    >>> pytensor.tensor.isclose(a, b).eval()
    array([ True, False])
    >>> a = np.array([1.0, np.inf], dtype="float64")
    >>> b = np.array([1.0, np.inf], dtype="float64")
    >>> pytensor.tensor.isclose(a, b).eval()
    array([ True,  True])

    """
    # close will be an int8 array of 1 where within tolerance
    # and 0 where not within tolerance or there was a nan or inf value.
    diff = _abs(a - b)
    tolerance = atol + rtol * _abs(b)
    close_prelim = le(diff, tolerance)

    a_nan = isnan(a)
    b_nan = isnan(b)
    nans = bitwise_or(a_nan, b_nan)

    a_inf = isinf(a)
    b_inf = isinf(b)
    infs = bitwise_or(a_inf, b_inf)

    nans_or_infs = bitwise_or(nans, infs)

    # close is now an array of 0's except where elements are not nan or inf
    # and are within the tolerance.
    close = bitwise_and(close_prelim, bitwise_not(nans_or_infs))

    # deal with signed inf values. this will make an array inf_eq of 0's
    # except where inf values have the same sign.
    both_infs = bitwise_and(a_inf, b_inf)
    inf_signs_eq = eq(a_inf * sign(a), b_inf * sign(b))
    inf_eq = bitwise_and(both_infs, inf_signs_eq)

    # now create the potential result combining close and inf_eq
    close_with_infs = bitwise_or(close, inf_eq)

    # deal with comparing nan's.
    if equal_nan:
        both_nans = bitwise_and(a_nan, b_nan)
        return bitwise_or(close_with_infs, both_nans)
    # otherwise nan's aren't considered close.
    else:
        return close_with_infs


##########################
# Bit-wise
##########################


@scalar_elemwise
def and_(a, b):
    """bitwise a & b

    Computes element-wise bitwise AND operation between two tensors.

    Parameters
    ----------
    a : TensorLike
        First input tensor
    b : TensorLike
        Second input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the bitwise AND of corresponding elements in a and b.

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x", dtype="int32")
    >>> y = pt.vector("y", dtype="int32")
    >>> f = pytensor.function([x, y], pt.and_(x, y))
    >>> f([1, 2, 3], [4, 2, 1])
    array([0, 2, 1], dtype=int32)

    Notes
    -----
    This function can also be used for logical AND operations
    on boolean tensors.
    """


@scalar_elemwise
def or_(a, b):
    """bitwise a | b

    Computes element-wise bitwise OR operation between two tensors.

    Parameters
    ----------
    a : TensorLike
        First input tensor
    b : TensorLike
        Second input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the bitwise OR of corresponding elements in a and b.

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x", dtype="int32")
    >>> y = pt.vector("y", dtype="int32")
    >>> f = pytensor.function([x, y], pt.or_(x, y))
    >>> f([1, 2, 3], [4, 2, 1])
    array([5, 2, 3], dtype=int32)

    Notes
    -----
    This function can also be used for logical OR operations
    on boolean tensors.
    """


@scalar_elemwise
def xor(a, b):
    """bitwise a ^ b

    Computes element-wise bitwise XOR (exclusive OR) operation between two tensors.

    Parameters
    ----------
    a : TensorLike
        First input tensor
    b : TensorLike
        Second input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the bitwise XOR of corresponding elements in a and b.

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x", dtype="int32")
    >>> y = pt.vector("y", dtype="int32")
    >>> f = pytensor.function([x, y], pt.xor(x, y))
    >>> f([1, 2, 3], [4, 2, 1])
    array([5, 0, 2], dtype=int32)

    Notes
    -----
    For boolean tensors, it computes the logical XOR
    (true when exactly one input is true).
    """


@scalar_elemwise
def invert(a):
    """bitwise ~a

    Computes element-wise bitwise inversion (NOT) of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the bitwise negation of each element in a.

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x", dtype="int8")
    >>> f = pytensor.function([x], pt.invert(x))
    >>> f([0, 1, 2, 3])
    array([-1, -2, -3, -4], dtype=int8)

    Notes
    -----
    For boolean tensors, this function computes the logical NOT.

    For integers, this inverts the bits in the binary representation.
    """


##########################
# Math
##########################


@scalar_elemwise
def abs(a):
    """|`a`|"""


pprint.assign(abs, printing.PatternPrinter(("|%(0)s|", -1000)))


@scalar_elemwise
def exp(a):
    """e^`a`

    Computes the element-wise exponential of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the exponential of each element in `a`

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> import numpy as np
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.exp(x))
    >>> f([0, 1, 2])
    array([1., 2.71828183, 7.3890561 ])

    """


@scalar_elemwise
def exp2(a):
    """2^`a`

    Computes element-wise base-2 exponential of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with 2 raised to the power of each element in `a`

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.exp2(x))
    >>> f([0, 1, 2, 3])
    array([1., 2., 4., 8.])

    Notes
    -----
    This operation is equivalent to `2**a` but may be more numerically stable
    for some values. It corresponds to NumPy's `np.exp2` function.
    """


@scalar_elemwise
def expm1(a):
    """e^`a` - 1

    Computes element-wise exponential of a tensor minus 1: exp(a) - 1.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with exp(x) - 1 computed for each element in `a`

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.expm1(x))
    >>> f([-1, 0, 1])
    array([-0.63212056,  0.        ,  1.71828183])

    Notes
    -----
    This function is more accurate than the naive computation of exp(x) - 1
    for small values of x (where exp(x) is close to 1). It corresponds to
    NumPy's `np.expm1` function.
    """


@scalar_elemwise
def neg(a):
    """-a

    Computes element-wise negation of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the negative of each element in `a`

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.neg(x))
    >>> f([1, -2, 3])
    array([-1,  2, -3])

    Notes
    -----
    This is equivalent to the arithmetic operation `-a` but works within
    the PyTensor computational graph. For complex numbers, this computes
    the complex negative.
    """


@scalar_elemwise
def reciprocal(a):
    """1.0/a

    Computes element-wise reciprocal (1/x) of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the reciprocal of each element in `a`

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.reciprocal(x))
    >>> f([1, 2, 4])
    array([1.  , 0.5 , 0.25])

    Notes
    -----
    This is equivalent to 1/a but is often more numerically stable.
    Division by zero will result in the appropriate IEEE floating point values
    (inf or -inf) or in an error depending on the backend.
    """


@scalar_elemwise
def log(a):
    """base e logarithm of a

    Computes the element-wise natural logarithm of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the natural logarithm of each element in `a`

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> import numpy as np
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.log(x))
    >>> f([1, 2.7, 10])
    array([0., 0.99325178, 2.30258509])

    """


@scalar_elemwise
def log2(a):
    """base 2 logarithm of a

    Computes element-wise base-2 logarithm of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the base-2 logarithm of each element in `a`

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.log2(x))
    >>> f([1, 2, 4, 8])
    array([0., 1., 2., 3.])

    Notes
    -----
    This function computes log(x)/log(2) but may be more numerically accurate
    than the naive computation.
    """


@scalar_elemwise
def log10(a):
    """base 10 logarithm of a

    Computes element-wise base-10 logarithm of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the base-10 logarithm of each element in `a`

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.log10(x))
    >>> f([1, 10, 100, 1000])
    array([0., 1., 2., 3.])

    Notes
    -----
    This function computes log(x)/log(10) but may be more numerically accurate
    than the naive computation.
    """


@scalar_elemwise
def log1p(a):
    """log(1+a)

    Computes element-wise natural logarithm of 1 plus a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the natural logarithm of (1 + a) for each element

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.log1p(x))
    >>> f([0, 1e-7, 1, 3])
    array([0.0000000e+00, 1.0000050e-07, 6.9314718e-01, 1.3862944e+00])

    Notes
    -----
    This function is more accurate than the naive computation of log(1+x)
    for small values of x (close to zero).
    """


@scalar_elemwise
def sign(a):
    """sign of a

    Computes element-wise sign of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the sign of each element in `a`: -1 for negative values,
        0 for zero, and 1 for positive values.

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.sign(x))
    >>> f([-2, 0, 3])
    array([-1.,  0.,  1.])

    Notes
    -----
    For complex inputs, this function
    returns the sign of the magnitude.
    """


def sgn(a):
    """sign of a"""

    warnings.warn(
        "sgn is deprecated and will stop working in the future, use sign instead.",
        FutureWarning,
    )
    return sign(a)


@scalar_elemwise
def ceil(a):
    """ceiling of a

    Computes element-wise ceiling (smallest integer greater than or equal to x) of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the ceiling of each element in `a`

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.ceil(x))
    >>> f([1.5, 2.0, -3.7])
    array([ 2.,  2., -3.])
    """


@scalar_elemwise
def floor(a):
    """floor of a

    Computes element-wise floor (largest integer less than or equal to x) of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the floor of each element in `a`

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.floor(x))
    >>> f([1.5, 2.0, -3.7])
    array([ 1.,  2., -4.])
    """


@scalar_elemwise
def trunc(a):
    """trunc of a

    Computes element-wise truncation (the integer part) of a tensor, effectively rounding downward.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the truncated value (integer part) of each element in `a`

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.trunc(x))
    >>> f([1.5, 2.0, -3.7])
    array([ 1.,  2., -3.])
    """


def iround(a, mode=None):
    """cast(round(a,mode),'int64')"""
    return cast(round(a, mode), "int64")


def round(a, mode=None):
    """round_mode(a) with mode in [half_away_from_zero, half_to_even].
    Default to half_to_even."""
    if mode is None:
        mode = "half_to_even"
        if config.warn__round:
            warnings.warn(
                "pytensor.tensor.round() changed its default from"
                " `half_away_from_zero` to `half_to_even` to have"
                " the same default as NumPy. Use the PyTensor flag"
                " `warn__round=False` to disable this warning."
            )
    if mode == "half_away_from_zero":
        return round_half_away_from_zero(a)
    elif mode == "half_to_even":
        return round_half_to_even(a)
    else:
        raise Exception(f"round mode {mode} is not implemented.")


@scalar_elemwise
def round_half_to_even(a):
    """round_half_to_even(a)"""


@scalar_elemwise
def round_half_away_from_zero(a):
    """round_half_away_from_zero(a)"""


@scalar_elemwise
def sqr(a):
    """square of a

    Computes element-wise square (x²) of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the square of each element in `a`

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.sqr(x))
    >>> f([-2, 0, 3])
    array([4, 0, 9])

    Notes
    -----
    This is equivalent to a**2 or a*a, but may be computed more efficiently.
    """


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
    """Calculate the covariance matrix.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`m = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`. Code and docstring ported from numpy.

    Parameters
    ==========
    m : array_like
        A 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column is
        observations of all those variables.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same form
        as that of `m`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : bool, optional
        Default normalization (False) is by ``(N - 1)``, where ``N`` is the
        number of observations given (unbiased estimate). If `bias` is True, then
        normalization is by ``N``. These values can be overridden by using the
        keyword ``ddof``.
    ddof : int, optional
        If not ``None`` the default value implied by `bias` is overridden.
        The default value is ``None``.

    Returns
    =======
    out : The covariance matrix of the variables.

    """

    if fweights is not None:
        raise NotImplementedError("fweights are not implemented")
    if aweights is not None:
        raise NotImplementedError("aweights are not implemented")

    if not rowvar and m.shape[0] != 1:
        m = m.T

    if y is not None:
        if not rowvar and y.shape[0] != 1:
            y = y.T
        m = concatenate((m, y), axis=0)

    if ddof is None:
        if not bias:
            ddof = 1
        else:
            ddof = 0

    # Determine the normalization
    fact = m.shape[1] - ddof

    m -= m.mean(axis=1, keepdims=1)
    c = m.dot(m.T)
    c *= constant(1) / fact
    return c.squeeze()


@scalar_elemwise
def sqrt(a):
    """square root of a

    Computes element-wise square root of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor (should contain non-negative values)

    Returns
    -------
    TensorVariable
        Output tensor with the square root of each element in `a`

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.sqrt(x))
    >>> f([0, 1, 4, 9])
    array([0., 1., 2., 3.])

    Notes
    -----
    For negative inputs, the behavior depends on the backend, typically
    resulting in NaN values.
    """


@scalar_elemwise
def deg2rad(a):
    """convert degree a to radian

    Computes element-wise conversion from degrees to radians.

    Parameters
    ----------
    a : TensorLike
        Input tensor in degrees

    Returns
    -------
    TensorVariable
        Output tensor with values converted to radians

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.deg2rad(x))
    >>> f([0, 90, 180, 270, 360])
    array([0.        , 1.57079633, 3.14159265, 4.71238898, 6.28318531])

    Notes
    -----
    This function corresponds to NumPy's `np.deg2rad` function.
    The conversion formula is: radians = degrees * (π / 180)
    """


@scalar_elemwise
def rad2deg(a):
    """convert radian a to degree

    Computes element-wise conversion from radians to degrees.

    Parameters
    ----------
    a : TensorLike
        Input tensor in radians

    Returns
    -------
    TensorVariable
        Output tensor with values converted to degrees

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> import numpy as np
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.rad2deg(x))
    >>> f([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    array([  0.,  90., 180., 270., 360.])

    Notes
    -----
    This function corresponds to NumPy's `np.rad2deg` function.
    The conversion formula is: degrees = radians * (180 / π)
    """


@scalar_elemwise
def cos(a):
    """cosine of a

    Computes element-wise cosine of a tensor in radians.

    Parameters
    ----------
    a : TensorLike
        Input tensor in radians

    Returns
    -------
    TensorVariable
        Output tensor with the cosine of each element

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> import numpy as np
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.cos(x))
    >>> f([0, np.pi / 2, np.pi])
    array([ 1.000000e+00,  6.123234e-17, -1.000000e+00])

    Notes
    -----
    This function corresponds to NumPy's `np.cos` function.
    """


@scalar_elemwise
def arccos(a):
    """arccosine of a

    Computes element-wise inverse cosine (arc cosine) of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor (values should be in the range [-1, 1])

    Returns
    -------
    TensorVariable
        Output tensor with the arc cosine of each element in radians,
        in the range [0, π]

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.arccos(x))
    >>> f([1, 0, -1])
    array([0.        , 1.57079633, 3.14159265])

    Notes
    -----
    This function corresponds to NumPy's `np.arccos` function.
    The values returned are in the range [0, π]. Input values outside
    the domain [-1, 1] will produce NaN outputs.
    """


@scalar_elemwise
def sin(a):
    """sine of a

    Computes element-wise sine of a tensor in radians.

    Parameters
    ----------
    a : TensorLike
        Input tensor in radians

    Returns
    -------
    TensorVariable
        Output tensor with the sine of each element

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> import numpy as np
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.sin(x))
    >>> f([0, np.pi / 2, np.pi])
    array([ 0.00000000e+00,  1.00000000e+00,  1.22464680e-16])

    Notes
    -----
    This function corresponds to NumPy's `np.sin` function.
    """


@scalar_elemwise
def arcsin(a):
    """arcsine of a

    Computes element-wise inverse sine (arc sine) of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor (values should be in the range [-1, 1])

    Returns
    -------
    TensorVariable
        Output tensor with the arc sine of each element in radians,
        in the range [-π/2, π/2]

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.arcsin(x))
    >>> f([-1, 0, 1])
    array([-1.57079633,  0.        ,  1.57079633])

    Notes
    -----
    This function corresponds to NumPy's `np.arcsin` function.
    The values returned are in the range [-π/2, π/2]. Input values outside
    the domain [-1, 1] will produce NaN outputs.
    """


@scalar_elemwise
def tan(a):
    """tangent of a

    Computes element-wise tangent of a tensor in radians.

    Parameters
    ----------
    a : TensorLike
        Input tensor in radians

    Returns
    -------
    TensorVariable
        Output tensor with the tangent of each element

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> import numpy as np
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.tan(x))
    >>> f([0, np.pi / 4, np.pi / 2 - 1e-10])  # Avoiding exact π/2 which is undefined
    array([0.00000000e+00, 1.00000000e+00, 1.25655683e+10])

    Notes
    -----
    This function corresponds to NumPy's `np.tan` function.
    Tangent is undefined at π/2 + nπ where n is an integer.
    """


@scalar_elemwise
def arctan(a):
    """arctangent of a

    Computes element-wise inverse tangent (arc tangent) of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the arc tangent of each element in radians,
        in the range [-π/2, π/2]

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.arctan(x))
    >>> f([-1, 0, 1])
    array([-0.78539816,  0.        ,  0.78539816])

    Notes
    -----
    This function corresponds to NumPy's `np.arctan` function.
    The values returned are in the range [-π/2, π/2].
    For the two-argument inverse tangent function, see `arctan2`.
    """


@scalar_elemwise
def arctan2(a, b):
    """arctangent of a / b

    Computes element-wise arc tangent of two values, taking into account
    the quadrant based on the signs of the inputs.

    Parameters
    ----------
    a : TensorLike
        First input tensor, representing the numerator (y-coordinates)
    b : TensorLike
        Second input tensor, representing the denominator (x-coordinates)

    Returns
    -------
    TensorVariable
        Output tensor with the arc tangent of a/b in radians, in the range [-π, π]

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> y = pt.vector("y")
    >>> x = pt.vector("x")
    >>> f = pytensor.function([y, x], pt.arctan2(y, x))
    >>> f([1, -1, 0, 0], [1, -1, 1, -1])
    array([ 0.78539816, -2.35619449,  0.        ,  3.14159265])

    Notes
    -----
    This function corresponds to NumPy's `np.arctan2` function.
    The returned values are in the range [-π, π].

    This function is similar to calculating the arc tangent of a/b, except
    that the signs of both arguments are used to determine the quadrant of
    the result.
    """


@scalar_elemwise
def cosh(a):
    """hyperbolic cosine of a

    Computes element-wise hyperbolic cosine of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the hyperbolic cosine of each element

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.cosh(x))
    >>> f([0, 1, 2])
    array([1.        , 1.54308063, 3.76219569])

    Notes
    -----
    This function corresponds to NumPy's `np.cosh` function.
    The hyperbolic cosine is defined as: cosh(x) = (exp(x) + exp(-x))/2
    """


@scalar_elemwise
def arccosh(a):
    """hyperbolic arc cosine of a

    Computes element-wise inverse hyperbolic cosine of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor (values should be ≥ 1)

    Returns
    -------
    TensorVariable
        Output tensor with the hyperbolic arc cosine of each element

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.arccosh(x))
    >>> f([1, 2, 10])
    array([0.        , 1.31695789, 2.99322285])

    Notes
    -----
    This function corresponds to NumPy's `np.arccosh` function.
    The domain is [1, inf]; values outside this range will produce NaN outputs.
    """


@scalar_elemwise
def sinh(a):
    """hyperbolic sine of a

    Computes element-wise hyperbolic sine of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the hyperbolic sine of each element

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.sinh(x))
    >>> f([0, 1, 2])
    array([0.        , 1.17520119, 3.62686041])

    Notes
    -----
    This function corresponds to NumPy's `np.sinh` function.
    The hyperbolic sine is defined as: sinh(x) = (exp(x) - exp(-x))/2
    """


@scalar_elemwise
def arcsinh(a):
    """hyperbolic arc sine of a

    Computes element-wise inverse hyperbolic sine of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the hyperbolic arc sine of each element

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.arcsinh(x))
    >>> f([-1, 0, 1])
    array([-0.88137359,  0.        ,  0.88137359])

    Notes
    -----
    This function corresponds to NumPy's `np.arcsinh` function.
    The inverse hyperbolic sine is defined for all real numbers.
    """


@scalar_elemwise
def tanh(a):
    """hyperbolic tangent of a

    Computes element-wise hyperbolic tangent of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the hyperbolic tangent of each element,
        with values in the range [-1, 1]

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.tanh(x))
    >>> f([-1, 0, 1])
    array([-0.76159416,  0.        ,  0.76159416])

    Notes
    -----
    This function corresponds to NumPy's `np.tanh` function.
    The hyperbolic tangent is defined as: tanh(x) = sinh(x)/cosh(x)
    """


@scalar_elemwise
def arctanh(a):
    """hyperbolic arc tangent of a

    Computes element-wise inverse hyperbolic tangent of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor (values should be in the range [-1, 1])

    Returns
    -------
    TensorVariable
        Output tensor with the hyperbolic arc tangent of each element

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.arctanh(x))
    >>> f([-0.5, 0, 0.5])
    array([-0.54930614,  0.        ,  0.54930614])

    Notes
    -----
    This function corresponds to NumPy's `np.arctanh` function.
    The domain of arctanh is [-1, 1]; values outside this range
    will produce NaN outputs.
    """


@scalar_elemwise
def erf(a):
    """error function

    Computes the element-wise error function of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the error function evaluated at each element,
        with values in the range [-1, 1]

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.erf(x))
    >>> f([-1, 0, 1])
    array([-0.84270079,  0.        ,  0.84270079])

    Notes
    -----
    This function corresponds to SciPy's `scipy.special.erf` function.
    The error function is defined as:
    erf(x) = (2/√π) * ∫(0 to x) exp(-t²) dt
    """


@scalar_elemwise
def erfc(a):
    """complementary error function

    Computes the element-wise complementary error function of a tensor.

    Parameters
    ----------
    a : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the complementary error function evaluated at each element

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.erfc(x))
    >>> f([-1, 0, 1])
    array([1.84270079, 1.        , 0.15729921])

    Notes
    -----
    This function corresponds to SciPy's `scipy.special.erfc` function.
    The complementary error function is defined as:
    erfc(x) = 1 - erf(x) = (2/√π) * ∫(x to ∞) exp(-t²) dt
    """


@scalar_elemwise
def erfcx(a):
    """scaled complementary error function"""


@scalar_elemwise
def erfinv(a):
    """inverse error function"""


@scalar_elemwise
def erfcinv(a):
    """inverse complementary error function"""


@scalar_elemwise
def owens_t(h, a):
    """owens t function"""


@scalar_elemwise
def gamma(a):
    """gamma function"""


@scalar_elemwise
def gammaln(a):
    """log gamma function"""


@scalar_elemwise
def psi(a):
    """derivative of log gamma function"""


digamma = psi


@scalar_elemwise
def tri_gamma(a):
    """second derivative of the log gamma function"""


@scalar_elemwise
def polygamma(n, x):
    """Polygamma function of order n evaluated at x"""


def chi2sf(x, k):
    """chi squared survival function"""
    warnings.warn("chi2sf is deprecated. Use `gammaincc(k / 2, x / 2)` instead")
    return gammaincc(k / 2, x / 2)


@scalar_elemwise
def gammainc(k, x):
    """Regularized lower gamma function"""


@scalar_elemwise
def gammaincc(k, x):
    """Regularized upper gamma function"""


@scalar_elemwise
def gammau(k, x):
    """Upper incomplete gamma function."""


@scalar_elemwise
def gammal(k, x):
    """Lower incomplete gamma function."""


@scalar_elemwise
def gammaincinv(k, x):
    """Inverse to the regularized lower incomplete gamma function"""


@scalar_elemwise
def gammainccinv(k, x):
    """Inverse of the regularized upper incomplete gamma function"""


@scalar_elemwise
def hyp2f1(a, b, c, z):
    """Gaussian hypergeometric function."""


@scalar_elemwise
def j0(x):
    """Bessel function of the first kind of order 0."""


@scalar_elemwise
def j1(x):
    """Bessel function of the first kind of order 1."""


@scalar_elemwise
def jv(v, x):
    """Bessel function of the first kind of order v (real)."""


@scalar_elemwise
def i0(x):
    """Modified Bessel function of the first kind of order 0."""


@scalar_elemwise
def i1(x):
    """Modified Bessel function of the first kind of order 1."""


@scalar_elemwise
def iv(v, x):
    """Modified Bessel function of the first kind of order v (real)."""


@scalar_elemwise
def ive(v, x):
    """Exponentially scaled modified Bessel function of the first kind of order v (real)."""


@scalar_elemwise
def kve(v, x):
    """Exponentially scaled modified Bessel function of the second kind of real order v."""


def kv(v, x):
    """Modified Bessel function of the second kind of real order v."""
    return kve(v, x) * exp(-x)


def kn(n, x):
    """Modified Bessel function of the second kind of integer order v."""
    return kv(n, x)


@scalar_elemwise
def sigmoid(x):
    """Logistic sigmoid function (1 / (1 + exp(-x)), also known as expit or inverse logit"""


expit = sigmoid


@scalar_elemwise
def softplus(x):
    """Compute log(1 + exp(x)), also known as softplus or log1pexp"""


log1pexp = softplus


@scalar_elemwise
def log1mexp(x):
    """Compute log(1 - exp(x)), also known as log1mexp"""


@scalar_elemwise
def betainc(a, b, x):
    """Regularized incomplete beta function"""


@scalar_elemwise
def betaincinv(a, b, x):
    """Inverse of the regularized incomplete beta function"""


@scalar_elemwise
def real(z):
    """Return real component of complex-valued tensor `z`."""


_tensor_py_operators.real = property(real, doc=real.__doc__)


@scalar_elemwise
def imag(z):
    """Return imaginary component of complex-valued tensor `z`."""


_tensor_py_operators.imag = property(imag, doc=imag.__doc__)


@scalar_elemwise
def angle(z):
    """Return polar-coordinate angle of complex-valued tensor `z`"""


@scalar_elemwise  # numpy.complex cannot build tensors
def complex(real, imag):
    """Return complex-valued tensor with `real` and `imag` components"""


@scalar_elemwise(symbolname="conj")
def _conj(z):
    """Return the complex conjugate of `z`."""


def conjugate(x):
    _x = as_tensor_variable(x)
    if _x.type.dtype not in complex_dtypes:
        return _x
    return _conj(_x)


conj = conjugate


@scalar_elemwise
def complex_from_polar(abs, angle):
    """Return complex-valued tensor from polar coordinate specification."""


def mean(input, axis=None, dtype=None, keepdims=False, acc_dtype=None):
    """
    Computes the mean value along the given axis(es) of a tensor `input`.

    Parameters
    ----------
    axis : None or int or (list of int) (see `Sum`)
        Compute the mean along this axis of the tensor.
        None means all axes (like numpy).
    dtype: None or string
        Dtype to cast the result of the inner summation into.
        For instance, by default, a sum of a float32 tensor will be
        done in float64 (acc_dtype would be float64 by default),
        but that result will be casted back in float32.
    keepdims: bool
        If this is set to True, the axes which are reduced are
        left in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original tensor.
    acc_dtype: None or string
        Dtype to use for the inner summation. This will not
        necessarily be the dtype of the output (in particular
        if it is a discrete (int/uint) dtype, the output will
        be in a float type). If None, then we use the same rules as `sum()`.
    """
    input = as_tensor_variable(input)

    if dtype is not None:
        # The summation will be done with the specified dtype.
        # sum() will complain if it is not suitable.
        sum_dtype = dtype
    else:
        sum_dtype = None
        # float16 overflows on the cast way too often
        if input.dtype == "float16":
            sum_dtype = "float32"

    s = sum(input, axis=axis, dtype=sum_dtype, keepdims=keepdims, acc_dtype=acc_dtype)
    shp = shape(input)

    # Cast shp into a float type
    # TODO Once we have a consistent casting policy, we could simply
    # use true_div.
    if s.dtype in ("float16", "float32", "complex64"):
        shp = cast(shp, "float32")
    else:
        shp = cast(shp, "float64")

    reduced_dims = (
        shp
        if axis is None
        else [shp[i] for i in normalize_axis_tuple(axis, input.type.ndim)]
    )
    s /= variadic_mul(*reduced_dims).astype(shp.dtype)

    # This can happen when axis is an empty list/tuple
    if s.dtype != shp.dtype and s.dtype in discrete_dtypes:
        s = cast(s, shp.dtype)

    if dtype == "float16" or (dtype is None and input.dtype == "float16"):
        s = cast(s, "float16")
    s.name = "mean"
    return s


def var(input, axis=None, ddof=0, keepdims=False, corrected=False):
    """
    Computes the variance along the given axis(es) of a tensor `input`.

    Parameters
    ----------
    axis: None or int or (list of int) (see `Sum`)
        Compute the variance along this axis of the tensor.
        None means all axes (like numpy).
    ddof: Degrees of freedom; 0 would compute the ML estimate, 1 would compute
        the unbiased estimate.
    keepdims : bool
        If this is set to True, the axes which are reduced are
        left in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original tensor.
    corrected : bool
        If this is set to True, the 'corrected_two_pass' algorithm is
        used to compute the variance.
        Refer : http://www.cs.yale.edu/publications/techreports/tr222.pdf

    Notes
    -----
    Default uses the two-pass algorithm (reference below).
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass_algorithm
    Also supports 'corrected_two_pass' algorithm (using the 'corrected' flag)
    which is numerically more stable. There exist other implementations that
    offer better stability, but probably slower.

    """

    if isinstance(ddof, (bool)):
        raise ValueError(
            "Parameter keepdims is now at index 3: (input, \
                          axis=None, ddof=0, keepdims=False, corrected=False)"
        )

    input_ndim = input.type.ndim
    if axis is None:
        axis = list(range(input_ndim))
    elif isinstance(axis, int | np.integer):
        axis = [axis]
    elif isinstance(axis, np.ndarray) and axis.ndim == 0:
        axis = [int(axis)]
    else:
        axis = [int(a) for a in axis]

    # compute the axis-wise mean
    mean_input = mean(input, axis, keepdims=True)

    # center the input
    centered_input = input - mean_input

    # return the mean sqr
    two = constant(2, dtype=centered_input.dtype)
    if ddof == 0:
        v = mean((centered_input**two), axis, keepdims=keepdims)
    else:
        shp = shape(input) - ddof
        v = sum((centered_input**two), axis=axis, keepdims=keepdims)
        for i in axis:
            v = true_div(v, shp[i])

    # use 'corrected_two_pass' algorithm
    if corrected:
        if ddof == 0:
            error = mean(centered_input, axis, keepdims=keepdims) ** 2
        else:
            shp = shape(input) - ddof
            shp_inp = shape(input)
            error = sum(centered_input, axis=axis, keepdims=keepdims) ** 2
            for i in axis:
                error = true_div(error, shp[i] * shp_inp[i])
        v = v - error

    v.name = "var"
    return v


def std(input, axis=None, ddof=0, keepdims=False, corrected=False):
    """
    Computes the standard deviation along the given axis(es) of a tensor `input`.

    Parameters
    ----------
    axis: None or int or (list of int) (see `Sum`)
        Compute the variance along this axis of the tensor.
        None means all axes (like numpy).
    ddof: Degrees of freedom; 0 would compute the ML estimate, 1 would compute
        the unbiased estimate.
    keepdims : bool
        If this is set to True, the axes which are reduced are
        left in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original tensor.
    corrected : bool
        If this is set to True, the 'corrected_two_pass' algorithm is
        used to compute the variance.
        Refer : http://www.cs.yale.edu/publications/techreports/tr222.pdf

    Notes
    -----
    It calls 'var()' and 'var()' uses the two-pass algorithm (reference below).
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass_algorithm
    Function 'var()' also supports 'corrected_two_pass' algorithm (using the
    'corrected' flag) which is numerically more stable. There exist other
    implementations that offer better stability, but probably slower.

    """

    if isinstance(ddof, (bool)):
        raise ValueError(
            "Parameter keepdims is now at index 3: (input, \
                          axis=None, ddof=0, keepdims=False, corrected=False)"
        )

    ret = sqrt(
        var(input=input, axis=axis, ddof=ddof, keepdims=keepdims, corrected=corrected)
    )
    ret.name = "std"
    return ret


def median(x: TensorLike, axis=None) -> TensorVariable:
    """
    Computes the median along the given axis(es) of a tensor `input`.

    Parameters
    ----------
    x: TensorLike
        The input tensor.
    axis: None or int or (list of int) (see `Sum`)
        Compute the median along this axis of the tensor.
        None means all axes (like numpy).
    """
    from pytensor.ifelse import ifelse

    x = as_tensor_variable(x)
    x_ndim = x.type.ndim
    if axis is None:
        axis = list(range(x_ndim))
    else:
        axis = list(normalize_axis_tuple(axis, x_ndim))

    non_axis = [i for i in range(x_ndim) if i not in axis]
    non_axis_shape = [x.shape[i] for i in non_axis]

    # Put axis at the end and unravel them
    x_raveled = x.transpose(*non_axis, *axis)
    if len(axis) > 1:
        x_raveled = x_raveled.reshape((*non_axis_shape, -1))
    raveled_size = x_raveled.shape[-1]
    k = raveled_size // 2

    # Sort the input tensor along the specified axis and pick median value
    x_sorted = x_raveled.sort(axis=-1)
    k_values = x_sorted[..., k]
    km1_values = x_sorted[..., k - 1]

    even_median = (k_values + km1_values) / 2.0
    odd_median = k_values.astype(even_median.type.dtype)
    even_k = eq(mod(raveled_size, 2), 0)
    return ifelse(even_k, even_median, odd_median, name="median")


@scalar_elemwise
def maximum(x, y):
    """elemwise maximum. See max for the maximum in one tensor

    Computes element-wise maximum of two tensors.

    Parameters
    ----------
    x : TensorLike
        First input tensor
    y : TensorLike
        Second input tensor

    Returns
    -------
    TensorLike
        Output tensor with the maximum of corresponding elements in x and y

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> a = pt.vector("a")
    >>> b = pt.vector("b")
    >>> f = pytensor.function([a, b], pt.maximum(a, b))
    >>> f([1, 3, 5], [2, 3, 4])
    array([2, 3, 5])

    Notes
    -----
    This computes the element-wise maximum, while `max(x)` computes the
    maximum value over all elements in a single tensor.
    """
    # see decorator for function body


@scalar_elemwise
def minimum(x, y):
    """elemwise minimum. See min for the minimum in one tensor

    Computes element-wise minimum of two tensors.

    Parameters
    ----------
    x : TensorLike
        First input tensor
    y : TensorLike
        Second input tensor

    Returns
    -------
    TensorLike
        Output tensor with the minimum of corresponding elements in x and y

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> a = pt.vector("a")
    >>> b = pt.vector("b")
    >>> f = pytensor.function([a, b], pt.minimum(a, b))
    >>> f([1, 3, 5], [2, 3, 4])
    array([1, 3, 4])
    """
    # see decorator for function body


def divmod(x, y):
    """elementvise divmod, using floor_div and mod_check"""
    return floor_div(x, y), mod_check(x, y)


@scalar_elemwise
def add(a, *other_terms):
    """elementwise addition

    Computes element-wise addition of tensors.

    Parameters
    ----------
    a : TensorLike
        First input tensor
    *other_terms : tensors
        Other tensors to add

    Returns
    -------
    TensorLike
        Output tensor with the elementwise sum of all inputs

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> y = pt.vector("y")
    >>> z = pt.vector("z")
    >>> f = pytensor.function([x, y, z], pt.add(x, y, z))
    >>> f([1, 2], [3, 4], [5, 6])
    array([ 9, 12])
    """
    # see decorator for function body


def variadic_add(*args):
    """Add that accepts arbitrary number of inputs, including zero or one."""
    if not args:
        return constant(0)
    if len(args) == 1:
        return args[0]
    return add(*args)


@scalar_elemwise
def sub(a, b):
    """elementwise subtraction"""
    # see decorator for function body


@scalar_elemwise
def mul(a, *other_terms):
    """elementwise multiplication"""
    # see decorator for function body


def variadic_mul(*args):
    """Mul that accepts arbitrary number of inputs, including zero or one."""
    if not args:
        return constant(1)
    if len(args) == 1:
        return args[0]
    return mul(*args)


@scalar_elemwise
def true_div(a, b):
    """elementwise [true] division (inverse of multiplication)"""
    # see decorator for function body


@scalar_elemwise
def int_div(a, b):
    """elementwise [floor] division (inverse of multiplication)"""
    # see decorator for function body


# floor_div and int_div are the same thing
floor_div = int_div


def ceil_intdiv(a, b):
    """Safely compute ``ceil(float_division(a, b))``.

    Works for all dtypes, but mostly useful when `a` and `b` are ints.

    """
    # If a and b are int with not many significant bits, we could
    # cast them to float to avoid doing the modulo. We do not know if this
    # is faster or not. But this is not safe for int64, because the cast will
    # lose precision. For example:
    #     cast(cast(a, scalar.upcast(a.type.dtype, 'float32')) / b,
    #          ps.upcast(a.type.dtype, b.type.dtype))

    # We cast for the case when a and b are uint*; otherwise, neq will
    # force their upcast to int.
    div = int_div(a, b)
    ret = cast(neq(a % b, 0), div.dtype) + div
    assert ret.dtype == ps.upcast(
        div.owner.inputs[0].type.dtype, div.owner.inputs[1].type.dtype
    )
    return ret


def mod_check(x, y):
    """Make sure we do not try to use complex numbers."""
    if (
        as_tensor_variable(x).dtype in complex_dtypes
        or as_tensor_variable(y).dtype in complex_dtypes
    ):
        # Currently forbidden.
        raise ps.Mod.complex_error
    else:
        return mod(x, y)


@scalar_elemwise
def mod(a, b):
    """elementwise modulo"""
    # see decorator for function body


@scalar_elemwise
def pow(a, b):
    """elementwise power"""
    # see decorator for function body


@scalar_elemwise
def clip(x, min, max):
    """
    Clip x to be between min and max.

    Note that when `x` is equal to the boundaries, the output is considered
    to be `x`, so at these points, the gradient of the cost wrt the output
    will be propagated to `x`, not to `min` nor `max`. In other words,
    on these points, the gradient wrt `x` will be equal to the gradient wrt
    the output, and the gradient wrt `min` and `max` will be zero.

    """
    # see decorator for function body
    # for grep: clamp, bound


pprint.assign(add, printing.OperatorPrinter("+", -2, "either"))
pprint.assign(mul, printing.OperatorPrinter("*", -1, "either"))
pprint.assign(sub, printing.OperatorPrinter("-", -2, "left"))
pprint.assign(neg, printing.OperatorPrinter("-", 0, "either"))
pprint.assign(true_div, printing.OperatorPrinter("/", -1, "left"))
pprint.assign(int_div, printing.OperatorPrinter("//", -1, "left"))
pprint.assign(pow, printing.OperatorPrinter("**", 1, "right"))


class Dot(Op):
    """
    Computes the dot product of two matrices variables

    Notes
    -----
    Matrix-matrix products are sometimes optimized to Dot22 or Gemm ops
    (see tensor.blas).
    Vector-vector products are sometimes optimized to Ger or CGer (see
    tensor.blas).
    Matrix-vector products are sometimes optimized to Gemv, CGemv (see
    tensor.blas).

    """

    gufunc_signature = "(m,n),(n,p)->(m,p)"
    gufunc_spec = ("matmul", 2, 1)
    __props__ = ()

    def make_node(self, x, y):
        x = as_tensor_variable(x)
        y = as_tensor_variable(y)

        if x.type.ndim != 2:
            raise TypeError(
                f"Dot Op expects a 2D tensor as input 0, got {x} with {x.type.ndim} dimensions"
            )
        if y.type.ndim != 2:
            raise TypeError(
                f"Dot Op expects a 2D tensor as input 1, got {y} with {y.type.ndim} dimensions"
            )

        sx, sy = x.type.shape, y.type.shape
        if sx[-1] is not None and sy[0] is not None and sx[-1] != sy[0]:
            raise ValueError(
                f"Incompatible shared dimension for dot product: {sx}, {sy}"
            )
        out_shape = (sx[0], sy[1])
        out_dtype = ps.upcast(x.type.dtype, y.type.dtype)
        outputs = [tensor(dtype=out_dtype, shape=out_shape)]
        return Apply(self, [x, y], outputs)

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = np.matmul(*inputs)

    def grad(self, inp, grads):
        x, y = inp
        (gz,) = grads

        xgrad = self(gz, y.T)
        ygrad = self(x.T, gz)

        # If x or y contain broadcastable dimensions but only one of
        # them know that a matching dimensions is broadcastable, the
        # above code don't always return the right broadcast pattern.
        # This cause problem down the road. See gh-1461.
        if xgrad.type.shape != x.type.shape:
            xgrad = specify_shape(xgrad, x.type.shape)
        if ygrad.type.shape != y.type.shape:
            ygrad = specify_shape(ygrad, y.type.shape)

        if xgrad.type.dtype not in float_dtypes:
            raise TypeError("Dot grad x output must be a float type")
        if ygrad.type.dtype not in float_dtypes:
            raise TypeError("Dot grad y output must be a float type")

        return xgrad, ygrad

    def R_op(self, inputs, eval_points):
        # R_op for a \dot b evaluated at c for a and d for b is
        # simply c \dot b + a \dot d

        assert len(inputs) == 2
        assert len(eval_points) == 2
        if eval_points[0] is None and eval_points[1] is None:
            return [None]

        if eval_points[0] is not None:
            t1 = self(eval_points[0], inputs[1])
        if eval_points[1] is not None:
            t2 = self(inputs[0], eval_points[1])

        if eval_points[0] is not None and eval_points[1] is not None:
            return [t1 + t2]
        elif eval_points[0] is not None:
            return [t1]
        else:
            return [t2]

    def infer_shape(self, fgraph, node, shapes):
        xshp, yshp = shapes
        return [[xshp[0], yshp[1]]]


_dot = Dot()
pprint.assign(
    _dot, printing.OperatorPrinter(printing.special["middle_dot"], -1, "left")
)


def dot(l, r):
    """Return a symbolic dot product.

    This is designed to work with both sparse and dense tensors types.
    """

    if not isinstance(l, Variable):
        l = as_tensor_variable(l)

    if not isinstance(r, Variable):
        r = as_tensor_variable(r)

    try:
        res = l.__dot__(r)
        if res is NotImplemented:
            raise NotImplementedError
    except (NotImplementedError, AttributeError, TypeError):
        res = r.__rdot__(l)
        if res is NotImplemented:
            raise NotImplementedError()

    return res


def dense_dot(a, b):
    """
    Computes the dot product of two variables.

    For two matrices, this is equivalent to matrix multiplication.
    For two vectors, this is the inner product.
    When one variable is a scalar, this is like elementwise multiplication.
    For N dimensions, this is a sum product over the last axis
    of the first array and the second-to-last axis of the second array:

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Note that this dot function does one of three things, in the following
    sequence:

        1.  If either a or b is scalar, it returns the elementwise product
            without calling the PyTensor Dot op.

        2.  If either a or b has more than 2 dimensions, it calls PyTensor's
            tensordot function with appropriate axes. The tensordot function
            expresses high-dimensional dot products in terms of 2D matrix
            multiplications, so it may be possible to further optimize for
            performance.

        3.  If both a and b have either 1 or 2 dimensions, it calls PyTensor's
            Dot op on a and b.

    Notes
    -----
    Matrix-matrix products are sometimes optimized to Dot22 or Gemm ops
    (see tensor.blas).
    Vector-vector products are sometimes optimized to Ger or CGer (see
    tensor.blas).
    Matrix-vector products are sometimes optimized to Gemv, CGemv (see
    tensor.blas).

    """
    a, b = as_tensor_variable(a), as_tensor_variable(b)

    if not (
        isinstance(a.type, DenseTensorType) and isinstance(b.type, DenseTensorType)
    ):
        raise TypeError("The dense dot product is only supported for dense types")

    if a.ndim == 0 or b.ndim == 0:
        return a * b
    elif a.ndim > 2 or b.ndim > 2:
        return tensordot(a, b, [[a.ndim - 1], [np.maximum(0, b.ndim - 2)]])
    else:
        row_vector = a.ndim == 1
        if row_vector:
            # Promote to row matrix
            a = a[None]

        col_vector = b.ndim == 1
        if col_vector:
            # Promote to column matrix
            b = b[:, None]

        out = _dot(a, b)
        if row_vector:
            # If we promoted a to a row matrix, we need to squeeze the first dimension
            out = out.squeeze(0)
        if col_vector:
            # If we promoted b to a column matrix, we need to squeeze the last dimension
            out = out.squeeze(-1)
        return out


def tensordot(
    a: TensorLike, b: TensorLike, axes: int | Sequence[Sequence[int]] = 2
) -> TensorVariable:
    """
    Compute tensor dot product along specified axes.

    Implementation is mostly taken from numpy version 1.26.0

    Given two tensors, `a` and `b`, and a sequence object containing
    two sequence objects, ``(a_axes, b_axes)``, sum the products of
    `a`'s and `b`'s elements (components) over the axes specified by
    ``a_axes`` and ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N`` dimensions
    of `a` and the first ``N`` dimensions of `b` are summed over.

    Parameters
    ----------
    a, b : TensorLike
        Tensors to "dot".

    axes : int or (2,) array_like
        * integer_like
          If an int N, sum over the last N axes of `a` and the first N axes
          of `b` in order. The sizes of the corresponding axes must match.
        * (2,) array_like
          Or, a list of axes to be summed over, first sequence applying to `a`,
          second to `b`. Both elements array_like must be of the same length.

    Returns
    -------
    output : TensorLike
        The tensor dot product of the input.
        Its shape will be equal to the concatenation of `a` and `b` shapes
        (ignoring the dimensions that were summed over given in ``a_axes``
        and ``b_axes``)

    Examples
    --------
    It may be helpful to consider an example to see what tensordot does.
    PyTensor's implementation is identical to NumPy's. Here ``a`` has shape (2, 3, 4)
    and ``b`` has shape (5, 6, 4, 3). The axes to sum over are [[1, 2], [3, 2]] --
    note that a.shape[1] == b.shape[3] and a.shape[2] == b.shape[2]; these axes
    are compatible. The resulting tensor will have shape (2, 5, 6) -- the
    dimensions that are not being summed:

    >>> a = np.random.random((2, 3, 4))
    >>> b = np.random.random((5, 6, 4, 3))

    #tensordot
    >>> c = np.tensordot(a, b, [[1, 2], [3, 2]])

    #loop replicating tensordot
    >>> a0, a1, a2 = a.shape
    >>> b0, b1, _, _ = b.shape
    >>> cloop = np.zeros((a0, b0, b1))

    #loop over non-summed indices -- these exist
    #in the tensor product.
    >>> for i in range(a0):
    ...     for j in range(b0):
    ...         for k in range(b1):
    ...             # loop over summed indices -- these don't exist
    ...             # in the tensor product.
    ...             for l in range(a1):
    ...                 for m in range(a2):
    ...                     cloop[i, j, k] += a[i, l, m] * b[j, k, m, l]

    >>> np.allclose(c, cloop)
    True

    This specific implementation avoids a loop by transposing a and b such that
    the summed axes of ``a`` are last and the summed axes of ``b`` are first. The
    resulting arrays are reshaped to 2 dimensions and a matrix dot product is taken.
    The result is reshaped back to the required output dimensions.

    In an extreme case, no axes may be specified. The resulting tensor
    will have shape equal to the concatenation of the shapes of a and b:

    >>> c = np.tensordot(a, b, 0)
    >>> print(a.shape)
    (2, 3, 4)
    >>> print(b.shape)
    (5, 6, 4, 3)
    >>> print(c.shape)
    (2, 3, 4, 5, 6, 4, 3)

    See the documentation of numpy.tensordot for more examples.

    """
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    a = as_tensor_variable(a)
    b = as_tensor_variable(b)
    runtime_shape_a = a.shape
    static_shape_a = a.type.shape
    ndim_a = a.type.ndim
    runtime_shape_b = b.shape
    static_shape_b = b.type.shape
    ndim_b = b.type.ndim
    if na != nb:
        raise ValueError(
            "The number of axes supplied for tensordot must be equal for each tensor. "
            f"Got {na} and {nb} respectively."
        )
    axes_a = list(normalize_axis_tuple(axes_a, ndim_a))
    axes_b = list(normalize_axis_tuple(axes_b, ndim_b))

    # The operation is only valid if the original dimensions match in length
    # The ravelling of the dimensions to coerce the operation into a single dot
    # could mask such errors, so we add an Assert if needed.
    must_assert_runtime = False
    for ax_a, ax_b in zip(axes_a, axes_b, strict=True):
        if (
            static_shape_a[ax_a] is not None
            and static_shape_b[ax_b] is not None
            and static_shape_a[ax_a] != static_shape_b[ax_b]
        ):
            raise ValueError(
                "Input arrays have inconsistent type shape along the axes "
                "that are to be reduced with tensordot."
            )
        elif static_shape_a[ax_a] is None or static_shape_b[ax_b] is None:
            if must_assert_runtime:
                a = Assert(
                    "Input array shape along reduced axes of tensordot are not equal"
                )(a, eq(runtime_shape_a[ax_a], runtime_shape_b[ax_b]))
            must_assert_runtime = True

    # Convert tensordot into a stacked dot product.
    # We stack the summed axes and the non-summed axes of each tensor separately,
    # and place the summed axes at the end of a and the beginning of b
    non_summed_axes_a = [k for k in range(ndim_a) if k not in axes_a]
    non_summed_dims_a = [runtime_shape_a[axis] for axis in non_summed_axes_a]
    transpose_axes_a = non_summed_axes_a + axes_a
    # We only need a reshape when we need to combine summed or non-summed dims
    # or introduce a new dimension (expand_dims) when doing a non-scalar outer product (len(axes) = 0)
    a_needs_reshape = (ndim_a != 0) and (
        (len(non_summed_axes_a) > 1) or (len(axes_a) != 1)
    )

    non_summed_axes_b = [k for k in range(ndim_b) if k not in axes_b]
    non_summed_dims_b = [runtime_shape_b[axis] for axis in non_summed_axes_b]
    transpose_axes_b = axes_b + non_summed_axes_b
    b_needs_reshape = (ndim_b != 0) and (
        (len(non_summed_axes_b) > 1) or (len(axes_b) != 1)
    )

    # summed_size_a and summed_size_b must be the same,
    # but to facilitate reasoning about useless reshapes we compute both from their shapes
    at = a.transpose(transpose_axes_a)
    if a_needs_reshape:
        non_summed_size_a = variadic_mul(*non_summed_dims_a)
        summed_size_a = variadic_mul(*[runtime_shape_a[axis] for axis in axes_a])
        at = at.reshape((non_summed_size_a, summed_size_a))

    bt = b.transpose(transpose_axes_b)
    if b_needs_reshape:
        non_summed_size_b = variadic_mul(*non_summed_dims_b)
        summed_size_b = variadic_mul(*[runtime_shape_b[axis] for axis in axes_b])
        bt = bt.reshape((summed_size_b, non_summed_size_b))

    res = dot(at, bt)

    if a_needs_reshape or b_needs_reshape:
        res = res.reshape(non_summed_dims_a + non_summed_dims_b)

    return res


def outer(x, y):
    """Return vector-vector outer product.

    If an input isn't a vector, we flatten it first.

    """
    if x.ndim != 1:
        x = x.flatten()
    if y.ndim != 1:
        y = y.flatten()
    return dot(x.dimshuffle(0, "x"), y.dimshuffle("x", 0))


class All(FixedOpCAReduce):
    """Applies `logical and` to all the values of a tensor along the
    specified axis(es).

    """

    nfunc_spec = ("all", 1, 1)

    def __init__(self, axis=None):
        super().__init__(ps.and_, axis)

    def _output_dtype(self, idtype):
        return "bool"

    def make_node(self, input):
        input = as_tensor_variable(input)
        if input.dtype != "bool":
            input = neq(input, 0)
        ret = super().make_node(input)
        return ret

    def grad(self, inp, grads):
        (x,) = inp
        return [x.zeros_like(config.floatX)]

    def clone(self, **kwargs):
        axis = kwargs.get("axis", self.axis)
        return type(self)(axis=axis)


class Any(FixedOpCAReduce):
    """Applies `bitwise or` to all the values of a tensor along the
    specified axis(es).

    """

    nfunc_spec = ("any", 1, 1)

    def __init__(self, axis=None):
        super().__init__(ps.or_, axis)

    def _output_dtype(self, idtype):
        return "bool"

    def make_node(self, input):
        input = as_tensor_variable(input)
        if input.dtype != "bool":
            input = neq(input, 0)
        ret = super().make_node(input)
        return ret

    def grad(self, inp, grads):
        (x,) = inp
        return [x.zeros_like(config.floatX)]

    def clone(self, **kwargs):
        axis = kwargs.get("axis", self.axis)
        return type(self)(axis=axis)


class Sum(FixedOpCAReduce):
    """
    Sums all the values of a tensor along the specified axis(es).

    Equivalent to `CAReduce(scalar.add, axis=axis, dtype=dtype)`,
    with the difference that this defines the gradient of sum wrt its
    tensor input.

    """

    nfunc_spec = ("sum", 1, 1)

    def __init__(self, axis=None, dtype=None, acc_dtype=None):
        super().__init__(
            ps.add,
            axis=axis,
            dtype=dtype,
            acc_dtype=acc_dtype,
            upcast_discrete_output=True,
        )

    def L_op(self, inp, out, grads):
        (x,) = inp

        if out[0].dtype not in continuous_dtypes:
            return [x.zeros_like(dtype=config.floatX)]

        (gz,) = grads
        gz = as_tensor_variable(gz)
        axis = self.axis
        if axis is None:
            axis = list(range(x.type.ndim))
        if axis == ():
            return (gz,)
        new_dims = []
        i = 0
        for j, _ in enumerate(x.type.broadcastable):
            if j in axis:
                new_dims.append("x")
            else:
                new_dims.append(i)
                i += 1
        gx = Elemwise(ps.second)(x, gz.dimshuffle(new_dims))
        return [gx]

    def R_op(self, inputs, eval_points):
        # There is just one element in inputs and eval_points, the axis are
        # part of self
        if None in eval_points:
            return [None]
        return self(*eval_points, return_list=True)

    def clone(self, **kwargs):
        axis = kwargs.get("axis", self.axis)
        dtype = kwargs.get("dtype", self.dtype)
        acc_dtype = kwargs.get("acc_dtype", self.acc_dtype)
        return type(self)(axis=axis, dtype=dtype, acc_dtype=acc_dtype)


def sum(input, axis=None, dtype=None, keepdims=False, acc_dtype=None):
    """
    Computes the sum along the given axis(es) of a tensor `input`.

    When axis is None (the default value), the sum is performed
    over the flattened tensor.

    For full documentation see `Sum`.
    In particular please pay attention to the important warning when using
    a custom acc_dtype.

    Parameters
    ----------
    keepdims: bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """

    out = Sum(axis=axis, dtype=dtype, acc_dtype=acc_dtype)(input)

    if keepdims:
        out = makeKeepDims(input, out, axis)
    return out


pprint.assign(Sum, printing.FunctionPrinter(["sum"], ["axis"]))


class Prod(FixedOpCAReduce):
    """
    Multiplies all the values of a tensor along the specified axis(es).

    Equivalent to `CAReduce(scalar.mul, axis = axis)`, with the
    difference that this defines the gradient of prod wrt its tensor
    input.

    """

    __props__ = ("scalar_op", "axis", "dtype", "acc_dtype", "no_zeros_in_input")
    nfunc_spec = ("prod", 1, 1)

    def __init__(self, axis=None, dtype=None, acc_dtype=None, no_zeros_in_input=False):
        super().__init__(
            ps.mul,
            axis=axis,
            dtype=dtype,
            acc_dtype=acc_dtype,
            upcast_discrete_output=True,
        )
        self.no_zeros_in_input = no_zeros_in_input

    def L_op(self, inp, out, grads):
        """
        The grad of this Op could be very easy, if it is was not for the case
        where zeros are present in a given "group" (ie. elements reduced
        together to form the product).

        If no zeros are found in the elements of the product, then the
        partial derivative of the product relative to one of the elements
        (one of the inputs) is simply the product of the other elements.
        That's easy to see from the chain rule.

        Now the trick (with no zeros) is to take the overall product, then
        for every original element, the partial derivative is given by
        this product divided by the element itself (which equals the product
        of the other terms). This is easy to do by broadcasting the original
        product.

        (Note that we also need to broadcast-multiply by the
        "incoming gradient", ie. the gradient of the cost relative to the
        output/product).

        With zeros, things get more complicated. For a given group, we have 3
        cases:

        * No zeros in the group. Use previous trick.
        * If only one zero is present, then the gradient for that element is
            non-zero, but is zero for all others.
        * If more than one zero is present, then all the derivatives are zero.

        For the last two cases (with 1 or more zeros), we can't use the
        division trick, as this gives divisions by 0.

        Implementing that case-by-case logic is not as trivial, so a bunch of
        hacks are piled down here to do it. Notably, for the "only one zero"
        case, there's a special Op that computes the product of the elements
        in the group, minus the zero (see `ProdWithoutZeros`). The trick is then
        to use the division trick for groups with no zero, to use the
        `ProdWithoutZeros` op where there's only one zero, and to output a
        derivative of zero for any element part of a group with more than
        one zero.

        I do this by first counting the number of zeros in each group (see the
        `at.eq` bits), then taking this or that behavior (see `at.switch`)
        based on the result of this count.

        """
        (prod_in,) = inp
        (gz,) = grads

        if out[0].dtype in discrete_dtypes or self.acc_dtype in discrete_dtypes:
            # There is an int conversion in the way
            return [prod_in.zeros_like(dtype=config.floatX)]

        # Prepare the broadcasting that is used everywhere to broadcast
        # over the original groups (ie. broadcast over the elements of a given
        # product)
        gz = as_tensor_variable(gz)
        axis = self.axis
        if axis is None:
            axis = list(range(prod_in.type.ndim))
        if axis == ():
            return (gz,)
        new_dims = []
        i = 0
        for j, _ in enumerate(prod_in.type.broadcastable):
            if j in axis:
                new_dims.append("x")
            else:
                new_dims.append(i)
                i += 1

        # result of the product, broadcastable over groups
        prod_out = self(prod_in).dimshuffle(new_dims)
        # incoming gradient, broadcastable over groups
        gz = gz.dimshuffle(new_dims)

        # division trick if we don't have zeros. This will contain
        # NaNs to be eliminated in the `at.switch` if we do have zeros.
        grad_case_without_zeros = gz * prod_out / prod_in

        if self.no_zeros_in_input:
            # this handles inputs with zeros, but only certain input shapes
            return [grad_case_without_zeros]
        else:
            where_zeros = eq(prod_in, 0.0)
            sum_where_zeros = sum(where_zeros, axis=self.axis)
            groups_with_single_zero = eq(sum_where_zeros, 1).dimshuffle(new_dims)
            # tensor with 0 everywhere except for those places where
            # a 0 part of a group with a single zero was to be found
            where_single_zero = groups_with_single_zero * where_zeros
            # further optimization to avoid computing ProdWithoutZeros
            # if the incoming gradient is 0
            where_gz_not_zero = neq(gz, 0.0)
            # only take ProdWithoutZeros for the groups with single zeros
            # with non-null incoming gradient
            where_to_take_prod_without_zeros = (
                groups_with_single_zero * where_gz_not_zero
            )
            # preprocess the original input so that we set 0 everywhere
            # except for groups that contain a single zero, to avoid computing
            # multiplications on other groups
            prod_without_zeros_in = where_to_take_prod_without_zeros * prod_in
            # TODO: put lazy switch here, if it'd work
            # this is pretty efficient already (no multiplication if 0), but
            # it'd be even better if we had a lazy if per element
            prod_without_zeros = ProdWithoutZeros(axis=self.axis)(prod_without_zeros_in)
            prod_without_zeros = prod_without_zeros.dimshuffle(new_dims)

            groups_without_zeros = eq(sum_where_zeros, 0).dimshuffle(new_dims)

            final_grad = switch(
                groups_without_zeros,
                grad_case_without_zeros,
                switch(where_single_zero, prod_without_zeros, 0.0) * gz,
            )

            return [final_grad]

    def c_code_cache_version(self):
        return (1,)

    def clone(self, **kwargs):
        axis = kwargs.get("axis", self.axis)
        dtype = kwargs.get("dtype", self.dtype)
        acc_dtype = kwargs.get("acc_dtype", self.acc_dtype)
        no_zeros_in_input = kwargs.get("no_zeros_in_input", self.no_zeros_in_input)
        return type(self)(
            axis=axis,
            dtype=dtype,
            acc_dtype=acc_dtype,
            no_zeros_in_input=no_zeros_in_input,
        )

    def __str__(self):
        if self.no_zeros_in_input:
            return f"{super().__str__()[:-1]}, no_zeros_in_input}})"
        return super().__str__()

    def __repr__(self):
        return f"{super().__repr__()[:-1]}, no_zeros_in_input={self.no_zeros_in_input})"


def prod(
    input,
    axis=None,
    dtype=None,
    keepdims=False,
    acc_dtype=None,
    no_zeros_in_input=False,
):
    """
    Computes the product along the given axis(es) of a tensor `input`.

    When axis is None (the default value), the product is performed
    over the flattened tensor.

    For full documentation see ``tensor.elemwise.Prod``.

    Parameters
    ----------
    keepdims: bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """

    out = Prod(
        axis, dtype=dtype, acc_dtype=acc_dtype, no_zeros_in_input=no_zeros_in_input
    )(input)

    if keepdims:
        out = makeKeepDims(input, out, axis)
    return out


class MulWithoutZeros(BinaryScalarOp):
    # "identity" here is zero, as in Reduce we don't want to start
    # with reducing (1, something_else): this leads to the erroneous
    # case where a vector of zeros is reduced by binary reductions
    # of (1, 0), which always ends up as 1 (ie. the result for
    # the c version, for the product of [0,0,0], is 1.0)

    identity = 0.0
    commutative = True
    associative = True

    def impl(self, x, y):
        if x == 0:
            return y
        if y == 0:
            return x
        return x * y

    def c_code(self, node, name, inp, out, sub):
        x, y = inp
        (z,) = out
        return f"{z} = (({x} == 0) ? ({y}) : (({y} == 0) ? ({x}) : (({y})*({x}))) );"

    def c_code_cache_version(self):
        return (1,)


mul_without_zeros = MulWithoutZeros(ps.upcast_out, name="mul_without_zeros")


class ProdWithoutZeros(FixedOpCAReduce):
    def __init__(self, axis=None, dtype=None, acc_dtype=None):
        super().__init__(
            mul_without_zeros,
            axis=axis,
            dtype=dtype,
            acc_dtype=acc_dtype,
            upcast_discrete_output=True,
        )

    def grad(self, inp, grads):
        from pytensor.gradient import grad_not_implemented

        (a,) = inp
        a_grad = grad_not_implemented(
            self,
            0,
            a,
            "2nd derivatives of `product(a)` is not currently supported."
            "If `a` is guaranteed to contains no zeros, use "
            "`product(a, no_zeros_in_input=True)`.",
        )
        return [a_grad]

    def clone(self, **kwargs):
        axis = kwargs.get("axis", self.axis)
        dtype = kwargs.get("dtype", self.dtype)
        acc_dtype = kwargs.get("acc_dtype", self.acc_dtype)
        return type(self)(axis=axis, dtype=dtype, acc_dtype=acc_dtype)


def any(x, axis=None, keepdims=False):
    out = Any(axis)(x)

    if keepdims:
        out = makeKeepDims(x, out, axis)
    return out


def all(x, axis=None, keepdims=False):
    out = All(axis)(x)

    if keepdims:
        out = makeKeepDims(x, out, axis)
    return out


def ptp(a, axis=None):
    """
    Range of values (maximum - minimum) along an axis.

    The name of the function comes from the acronym for peak to peak.

    Parameters
    ----------
    a
        Input tensor.
    axis
        Axis along which to find the peaks. By default, flatten the array.

    Returns
    -------
    array
        A new array holding the result.

    """

    a = as_tensor_variable(a)

    out = max(a, axis) - min(a, axis)

    return out


def power(x, y):
    return x**y


def logaddexp(*xs):
    """Logarithm of the sum of exponentiations of the inputs.

    See ``numpy.logaddexp``.

    Parameters
    ----------
    xs : symbolic tensors
        Input

    Returns
    -------
    TensorVariable

    """

    return log(add(*[exp(x) for x in xs]))


def logsumexp(x, axis=None, keepdims=False):
    """Compute the log of the sum of exponentials of input elements.

    See ``scipy.special.logsumexp``.

    Parameters
    ----------
    x : symbolic tensor
        Input

    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default axis is None,
        and all elements are summed.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the original array.

    Returns
    -------
    TensorVariable

    """

    return log(sum(exp(x), axis=axis, keepdims=keepdims))


_matmul = Blockwise(_dot, name="Matmul")


def matmul(x1: "ArrayLike", x2: "ArrayLike", dtype: Optional["DTypeLike"] = None):
    """Compute the matrix product of two tensor variables.

    Parameters
    ----------
    x1, x2
        Input arrays, scalars not allowed.
    dtype
        The desired data-type for the array. If not given, then the type will
        be determined as the minimum type required to hold the objects in the
        sequence.

    Returns
    -------
    out : ndarray
        The matrix product of the inputs. This is a scalar only when both
        `x1`, `x2` are 1-d vectors.

    Raises
    ------
    ValueError
        If the last dimension of `x1` is not the same size as the second-to-last
        dimension of `x2`. If a scalar value is passed in.

    Notes
    -----
    The behavior depends on the arguments in the following way.

    - If both arguments are 2-D they are multiplied like conventional matrices.
    - If either argument is N-D, N > 2, it is treated as a stack of matrices
        residing in the last two indexes and broadcast accordingly.
    - If the first argument is 1-D, it is promoted to a matrix by prepending a
        1 to its dimensions. After matrix multiplication the prepended 1 is removed.
    - If the second argument is 1-D, it is promoted to a matrix by appending a
        1 to its dimensions. After matrix multiplication the appended 1 is removed.

    `matmul` differs from `dot` in two important ways:

    - Multiplication by scalars is not allowed, use `mul` instead.
    - Stacks of matrices are broadcast together as if the matrices were elements,
        respecting the signature ``(n, k), (k, m) -> (n, m)``:
    """
    x1 = as_tensor_variable(x1)
    x2 = as_tensor_variable(x2)
    if x1.type.ndim == 0 or x2.type.ndim == 0:
        raise ValueError("matmul operand cannot be scalar")
    if x1.type.ndim == 1 and x2.type.ndim == 1:
        out = vecdot(x1, x2)
    elif x1.type.ndim == 1:
        out = vecmat(x1, x2)
    elif x2.type.ndim == 1:
        out = matvec(x1, x2)
    else:
        out = _matmul(x1, x2)

    if dtype is not None:
        out = out.astype(dtype)

    return out


def vecdot(
    x1: TensorLike,
    x2: TensorLike,
    dtype: Optional["DTypeLike"] = None,
) -> TensorVariable:
    """Compute the vector dot product of two arrays.

    Parameters
    ----------
    x1, x2
        Input arrays with the same shape.
    dtype
        The desired data-type for the result. If not given, then the type will
        be determined as the minimum type required to hold the objects in the
        sequence.

    Returns
    -------
    TensorVariable
        The vector dot product of the inputs.

    Notes
    -----
    This is equivalent to `numpy.vecdot` and computes the dot product of
    vectors along the last axis of both inputs. Broadcasting is supported
    across all other dimensions.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> # Vector dot product with shape (5,) inputs
    >>> x = pt.vector("x", shape=(5,))  # shape (5,)
    >>> y = pt.vector("y", shape=(5,))  # shape (5,)
    >>> z = pt.vecdot(x, y)  # scalar output
    >>> # Equivalent to numpy.vecdot(x, y)
    >>>
    >>> # With batched inputs of shape (3, 5)
    >>> x_batch = pt.matrix("x", shape=(3, 5))  # shape (3, 5)
    >>> y_batch = pt.matrix("y", shape=(3, 5))  # shape (3, 5)
    >>> z_batch = pt.vecdot(x_batch, y_batch)  # shape (3,)
    >>> # Equivalent to numpy.vecdot(x_batch, y_batch)
    """
    out = matmul(x1[..., None, :], x2[..., :, None]).squeeze((-2, -1))

    if dtype is not None:
        out = out.astype(dtype)

    return out


def matvec(
    x1: TensorLike, x2: TensorLike, dtype: Optional["DTypeLike"] = None
) -> TensorVariable:
    """Compute the matrix-vector product.

    Parameters
    ----------
    x1
        Input array for the matrix with shape (..., M, K).
    x2
        Input array for the vector with shape (..., K).
    dtype
        The desired data-type for the result. If not given, then the type will
        be determined as the minimum type required to hold the objects in the
        sequence.

    Returns
    -------
    TensorVariable
        The matrix-vector product with shape (..., M).

    Notes
    -----
    This is equivalent to `numpy.matvec` and computes the matrix-vector product
    with broadcasting over batch dimensions.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> # Matrix-vector product
    >>> A = pt.matrix("A", shape=(3, 4))  # shape (3, 4)
    >>> v = pt.vector("v", shape=(4,))  # shape (4,)
    >>> result = pt.matvec(A, v)  # shape (3,)
    >>> # Equivalent to numpy.matvec(A, v)
    >>>
    >>> # Batched matrix-vector product
    >>> batched_A = pt.tensor3("A", shape=(2, 3, 4))  # shape (2, 3, 4)
    >>> batched_v = pt.matrix("v", shape=(2, 4))  # shape (2, 4)
    >>> result = pt.matvec(batched_A, batched_v)  # shape (2, 3)
    >>> # Equivalent to numpy.matvec(batched_A, batched_v)
    """
    out = matmul(x1, x2[..., None]).squeeze(-1)

    if dtype is not None:
        out = out.astype(dtype)

    return out


def vecmat(
    x1: TensorLike, x2: TensorLike, dtype: Optional["DTypeLike"] = None
) -> TensorVariable:
    """Compute the vector-matrix product.

    Parameters
    ----------
    x1
        Input array for the vector with shape (..., K).
    x2
        Input array for the matrix with shape (..., K, N).
    dtype
        The desired data-type for the result. If not given, then the type will
        be determined as the minimum type required to hold the objects in the
        sequence.

    Returns
    -------
    TensorVariable
        The vector-matrix product with shape (..., N).

    Notes
    -----
    This is equivalent to `numpy.vecmat` and computes the vector-matrix product
    with broadcasting over batch dimensions.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> # Vector-matrix product
    >>> v = pt.vector("v", shape=(3,))
    >>> A = pt.matrix("A", shape=(3, 4))
    >>> result = pt.vecmat(v, A)  # shape (4,)
    >>> # Equivalent to numpy.vecmat(v, A)
    >>>
    >>> # Batched vector-matrix product
    >>> batched_v = pt.matrix("v", shape=(2, 3))
    >>> batched_A = pt.tensor3("A", shape=(2, 3, 4))
    >>> result = pt.vecmat(batched_v, batched_A)  # shape (2, 4)
    >>> # Equivalent to numpy.vecmat(batched_v, batched_A)
    """
    out = matmul(x2.mT, x1[..., None]).squeeze(-1)

    if dtype is not None:
        out = out.astype(dtype)

    return out


@_vectorize_node.register(Dot)
def vectorize_node_dot(op, node, batched_x, batched_y):
    return matmul(batched_x, batched_y).owner


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    """
    Replace NaN with zero and infinity with large finite numbers (default
    behaviour) or with the numbers defined by the user using the `nan`,
    `posinf` and/or `neginf` keywords.

    NaN is replaced by zero or by the user defined value in
    `nan` keyword, infinity is replaced by the largest finite floating point
    values representable by ``x.dtype`` or by the user defined value in
    `posinf` keyword and -infinity is replaced by the most negative finite
    floating point values representable by ``x.dtype`` or by the user defined
    value in `neginf` keyword.

    Parameters
    ----------
    x : symbolic tensor
        Input array.
    nan
        The value to replace NaN's with in the tensor (default = 0).
    posinf
        The value to replace +INF with in the tensor (default max
        in range representable by ``x.dtype``).
    neginf
        The value to replace -INF with in the tensor (default min
        in range representable by ``x.dtype``).

    Returns
    -------
    out
        The tensor with NaN's, +INF, and -INF replaced with the
        specified and/or default substitutions.
    """
    # Replace NaN's with nan keyword
    is_nan = isnan(x)
    is_pos_inf = isposinf(x)
    is_neg_inf = isneginf(x)

    x = switch(is_nan, nan, x)

    # Get max and min values representable by x.dtype
    maxf = posinf
    minf = neginf

    # Specify the value to replace +INF and -INF with
    if maxf is None:
        maxf = np.finfo(x.real.dtype).max
    if minf is None:
        minf = np.finfo(x.real.dtype).min

    # Replace +INF and -INF values
    x = switch(is_pos_inf, maxf, x)
    x = switch(is_neg_inf, minf, x)

    return x


# NumPy logical aliases
square = sqr

bitwise_and = and_
bitwise_or = or_
bitwise_xor = xor
bitwise_not = invert

greater = gt
greater_equal = ge
less = lt
less_equal = le
equal = eq
not_equal = neq

__all__ = [
    "abs",
    "add",
    "all",
    "allclose",
    "and_",
    "angle",
    "any",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "argmax",
    "argmin",
    "betainc",
    "betaincinv",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "ceil",
    "ceil_intdiv",
    "chi2sf",
    "clip",
    "complex",
    "complex_from_polar",
    "conj",
    "conjugate",
    "cos",
    "cosh",
    "cov",
    "deg2rad",
    "dense_dot",
    "digamma",
    "divmod",
    "dot",
    "eq",
    "equal",
    "erf",
    "erfc",
    "erfcinv",
    "erfcx",
    "erfinv",
    "exp",
    "exp2",
    "expit",
    "expm1",
    "floor",
    "floor_div",
    "gamma",
    "gammainc",
    "gammaincc",
    "gammainccinv",
    "gammaincinv",
    "gammal",
    "gammaln",
    "gammau",
    "ge",
    "greater",
    "greater_equal",
    "gt",
    "hyp2f1",
    "i0",
    "i1",
    "imag",
    "int_div",
    "invert",
    "iround",
    "isclose",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "iv",
    "ive",
    "j0",
    "j1",
    "jv",
    "kn",
    "kv",
    "kve",
    "largest",
    "le",
    "less",
    "less_equal",
    "log",
    "log1mexp",
    "log1p",
    "log1pexp",
    "log2",
    "log10",
    "logaddexp",
    "logsumexp",
    "lt",
    "matmul",
    "matvec",
    "max",
    "max_and_argmax",
    "maximum",
    "mean",
    "median",
    "min",
    "minimum",
    "mod",
    "mul",
    "nan_to_num",
    "neg",
    "neq",
    "not_equal",
    "or_",
    "outer",
    "owens_t",
    "polygamma",
    "pow",
    "power",
    "prod",
    "psi",
    "ptp",
    "rad2deg",
    "real",
    "reciprocal",
    "round",
    "round_half_away_from_zero",
    "round_half_to_even",
    "sgn",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "smallest",
    "softplus",
    "sqr",
    "sqrt",
    "square",
    "std",
    "std",
    "sub",
    "sum",
    "tan",
    "tanh",
    "tensordot",
    "tri_gamma",
    "true_div",
    "trunc",
    "var",
    "vecdot",
    "vecmat",
    "xor",
]
