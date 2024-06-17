from collections.abc import Collection, Iterable

import numpy as np
from numpy.core.multiarray import normalize_axis_index

import pytensor
import pytensor.scalar.basic as ps
from pytensor.gradient import (
    DisconnectedType,
    _float_zeros_like,
    disconnected_type,
    grad_undefined,
)
from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.op import Op
from pytensor.link.c.op import COp
from pytensor.link.c.params_type import ParamsType
from pytensor.link.c.type import EnumList, Generic
from pytensor.misc.safe_asarray import _asarray
from pytensor.raise_op import Assert
from pytensor.scalar import int32 as int_t
from pytensor.scalar import upcast
from pytensor.tensor import as_tensor_variable
from pytensor.tensor import basic as ptb
from pytensor.tensor.basic import alloc, second
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.math import abs as pt_abs
from pytensor.tensor.math import all as pt_all
from pytensor.tensor.math import eq as pt_eq
from pytensor.tensor.math import ge, lt, maximum, minimum, prod, switch
from pytensor.tensor.math import max as pt_max
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.shape import specify_broadcastable
from pytensor.tensor.subtensor import advanced_inc_subtensor1, set_subtensor
from pytensor.tensor.type import TensorType, dvector, int_dtypes, integer_dtypes, vector
from pytensor.tensor.variable import TensorVariable
from pytensor.utils import LOCAL_BITWIDTH, PYTHON_INT_BITWIDTH


class CpuContiguous(COp):
    """
    Check to see if the input is c-contiguous.

    If it is, do nothing, else return a contiguous array.
    """

    __props__ = ()
    view_map = {0: [0]}
    check_input = False

    def make_node(self, x):
        x_ = ptb.as_tensor_variable(x)
        return Apply(self, [x_], [x_.type()])

    def perform(self, node, inputs, output_storage):
        (x,) = inputs
        y = output_storage[0]
        # if the output is contiguous do nothing, else copy
        # the input
        if not x.flags["C_CONTIGUOUS"]:
            x = x.copy()
        assert x.flags["C_CONTIGUOUS"]
        y[0] = x

    def grad(self, inputs, dout):
        return [ptb.as_tensor_variable(dout[0])]

    def c_code(self, node, name, inames, onames, sub):
        (x,) = inames
        (y,) = onames
        code = """
            if (!PyArray_CHKFLAGS({x}, NPY_ARRAY_C_CONTIGUOUS)){{
                // check to see if output is contiguous first
                if ({y} != NULL &&
                    PyArray_CompareLists(PyArray_DIMS({y}), PyArray_DIMS({x}), PyArray_NDIM({x})) &&
                    PyArray_CHKFLAGS({y}, NPY_ARRAY_C_CONTIGUOUS)){{
                    PyArray_CopyInto({y}, {x});
                }}
                else{{
                    Py_XDECREF({y});
                    {y} = PyArray_GETCONTIGUOUS({x});
                }}
            }}
            else{{
                Py_XINCREF({x});
                Py_XDECREF({y});
                {y} = {x};
            }}
            """.format(**locals())
        return code

    def c_code_cache_version(self):
        return (1,)


cpu_contiguous = CpuContiguous()


class SearchsortedOp(COp):
    """Wrapper for ``numpy.searchsorted``.

    For full documentation, see :func:`searchsorted`.

    See Also
    --------
    searchsorted : numpy-like function that uses `SearchsortedOp`

    """

    params_type = Generic()
    __props__ = ("side",)
    check_input = False

    def __init__(self, side="left"):
        if side == "left" or side == "right":
            self.side = side
        else:
            raise ValueError(f"'{side}' is an invalid value for keyword 'side'")

    def get_params(self, node):
        return self.side

    def make_node(self, x, v, sorter=None):
        x = ptb.as_tensor(x, ndim=1)
        v = ptb.as_tensor(v)
        out_type = v.type.clone(dtype="int64")
        if sorter is None:
            return Apply(self, [x, v], [out_type()])
        else:
            sorter = ptb.as_tensor(sorter, ndim=1)
            if PYTHON_INT_BITWIDTH == 32 and sorter.dtype == "int64":
                raise TypeError(
                    "numpy.searchsorted with Python 32bit do not support a"
                    " sorter of int64."
                )
            if sorter.type.ndim == 1 and sorter.type.dtype not in int_dtypes:
                raise TypeError("sorter must be an integer vector", sorter.type)
            return Apply(self, [x, v, sorter], [out_type()])

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[1]]

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        v = inputs[1]
        if len(node.inputs) == 3:
            sorter = inputs[2]
        else:
            sorter = None
        z = output_storage[0]

        z[0] = np.searchsorted(x, v, side=self.side, sorter=sorter).astype(
            node.outputs[0].dtype
        )

    def c_support_code_struct(self, node, name):
        return f"""
            int right_{name};
        """

    def c_init_code_struct(self, node, name, sub):
        side = sub["params"]
        fail = sub["fail"]
        return """
            PyObject* tmp_{name} = PyUnicode_FromString("right");
            if (tmp_{name} == NULL)
                {fail};
            right_{name} = PyUnicode_Compare({side}, tmp_{name});
            Py_DECREF(tmp_{name});
        """.format(**locals())

    def c_code(self, node, name, inames, onames, sub):
        sorter = None
        if len(node.inputs) == 3:
            x, v, sorter = inames
        else:
            x, v = inames
        if not sorter:
            sorter = "NULL"
        (z,) = onames
        fail = sub["fail"]

        return """
            Py_XDECREF({z});
            {z} = (PyArrayObject*) PyArray_SearchSorted({x}, (PyObject*) {v},
                                                          right_{name} ? NPY_SEARCHLEFT : NPY_SEARCHRIGHT, (PyObject*) {sorter});
            if (!{z})
                {fail};
            if (PyArray_TYPE({z}) != NPY_INT64){{
                PyObject * tmp = PyArray_Cast({z}, NPY_INT64);
                Py_XDECREF({z});
                {z} = (PyArrayObject*) tmp;
            }}
        """.format(**locals())

    def c_code_cache_version(self):
        return (2,)

    def grad(self, inputs, output_gradients):
        num_ins = len(inputs)
        if num_ins == 3:
            x, v, sorter = inputs
        else:
            x, v = inputs

        x_grad = _float_zeros_like(x)
        v_grad = _float_zeros_like(v)
        if num_ins == 3:
            return [x_grad, v_grad, disconnected_type()]
        else:
            return [x_grad, v_grad]


def searchsorted(x, v, side="left", sorter=None):
    """Find indices where elements should be inserted to maintain order.

    This wraps ``numpy.searchsorted``. Find the indices into a sorted array
    `x` such that, if the corresponding elements in `v` were inserted
    before the indices, the order of `x` would be preserved.

    Parameters
    ----------
    x : 1-D tensor (array-like)
        Input array. If `sorter` is ``None``, then it must be sorted in
        ascending order, otherwise `sorter` must be an array of indices
        which sorts it.
    v : tensor (array-like)
        Contains the values to be inserted into `x`.
    side : {'left', 'right'}, optional.
        If ``'left'`` (default), the index of the first suitable
        location found is given. If ``'right'``, return the last such index. If
        there is no suitable index, return either 0 or N (where N is the length
        of `x`).
    sorter : 1-D tensor of integers (array-like), optional
        Contains indices that sort array `x` into ascending order.
        They are typically the result of argsort.

    Returns
    -------
    indices : tensor of integers (int64)
        Array of insertion points with the same shape as `v`.

    See Also
    --------
    `numpy.searchsorted <https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.searchsorted.html>`_

    Notes
    -----

        * Binary search is used to find the required insertion points.
        * This Op is working **only on CPU** currently.

    Examples
    --------
    >>> from pytensor import tensor as pt
    >>> from pytensor.tensor import extra_ops
    >>> x = ptb.dvector()
    >>> idx = x.searchsorted(3)
    >>> idx.eval({x: [1,2,3,4,5]})
    array(2)
    >>> extra_ops.searchsorted([1,2,3,4,5], 3).eval()
    array(2)
    >>> extra_ops.searchsorted([1,2,3,4,5], 3, side='right').eval()
    array(3)
    >>> extra_ops.searchsorted([1,2,3,4,5], [-10, 10, 2, 3]).eval()
    array([0, 5, 1, 2])

    .. versionadded:: 0.9

    """
    return SearchsortedOp(side=side)(x, v, sorter)


class CumOp(COp):
    # See function cumsum/cumprod for docstring

    __props__ = ("axis", "mode")
    check_input = False
    params_type = ParamsType(
        c_axis=int_t, mode=EnumList(("MODE_ADD", "add"), ("MODE_MUL", "mul"))
    )

    def __init__(self, axis: int | None = None, mode="add"):
        if mode not in ("add", "mul"):
            raise ValueError(f'{type(self).__name__}: Unknown mode "{mode}"')
        self.axis = axis
        self.mode = mode

    c_axis = property(lambda self: np.MAXDIMS if self.axis is None else self.axis)

    def make_node(self, x):
        x = ptb.as_tensor_variable(x)
        out_type = x.type()

        if self.axis is None:
            out_type = vector(dtype=x.dtype)  # Flatten
        elif self.axis >= x.ndim or self.axis < -x.ndim:
            raise ValueError(f"axis(={self.axis}) out of bounds")

        return Apply(self, [x], [out_type])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        if self.mode == "add":
            z[0] = np.cumsum(x, axis=self.axis)
        else:
            z[0] = np.cumprod(x, axis=self.axis)

    def grad(self, inputs, output_gradients):
        (x,) = inputs
        (gi,) = output_gradients

        if self.axis is None:
            if self.mode == "add":
                return [cumsum(gi[::-1])[::-1].reshape(x.shape)]
            elif self.mode == "mul":
                fx = cumprod(x, axis=self.axis)
                return [cumsum((fx * gi)[::-1])[::-1].reshape(x.shape) / x]
            else:
                raise NotImplementedError(
                    f'{type(self).__name__}: unknown gradient for mode "{self.mode}"'
                )

        reverse_slicing = [slice(None, None, None)] * gi.ndim
        reverse_slicing[self.axis] = slice(None, None, -1)
        reverse_slicing = tuple(reverse_slicing)
        # We need to reverse the gradients along ``self.axis``,
        #  compute cumsum, then reverse again
        if self.mode == "add":
            return [cumsum(gi[reverse_slicing], self.axis)[reverse_slicing]]
        elif self.mode == "mul":
            fx = cumprod(x, axis=self.axis)
            return [cumsum((fx * gi)[reverse_slicing], self.axis)[reverse_slicing] / x]
        else:
            raise NotImplementedError(
                f'{type(self).__name__}: unknown gradient for mode "{self.mode}"'
            )

    def infer_shape(self, fgraph, node, shapes):
        if self.axis is None:
            return [(prod(shapes[0]),)]  # Flatten

        return shapes

    def c_code(self, node, name, inames, onames, sub):
        (x,) = inames
        (z,) = onames
        axis = self.axis
        fail = sub["fail"]
        params = sub["params"]

        code = """
                int axis = {params}->c_axis;
                if (axis == 0 && PyArray_NDIM({x}) == 1)
                    axis = NPY_MAXDIMS;
                npy_intp shape[1] = {{ PyArray_SIZE({x}) }};
                if(axis == NPY_MAXDIMS && !({z} && PyArray_DIMS({z})[0] == shape[0]))
                {{
                    Py_XDECREF({z});
                    {z} = (PyArrayObject*) PyArray_SimpleNew(1, shape, PyArray_TYPE((PyArrayObject*) py_{x}));
                }}

                else if(axis != NPY_MAXDIMS && !({z} && PyArray_CompareLists(PyArray_DIMS({z}), PyArray_DIMS({x}), PyArray_NDIM({x}))))
                {{
                    Py_XDECREF({z});
                    {z} = (PyArrayObject*) PyArray_SimpleNew(PyArray_NDIM({x}), PyArray_DIMS({x}), PyArray_TYPE({x}));
                }}

                if (!{z})
                    {fail};
                {{

                    PyObject * t = NULL;
                    if({params}->mode == MODE_ADD)
                        t = PyArray_CumSum(
                            {x}, axis,
                            PyArray_TYPE({x}), {z});
                    else if({params}->mode == MODE_MUL)
                        t = PyArray_CumProd(
                            {x}, axis,
                            PyArray_TYPE({x}), {z});

                    if (!t){{
                       {fail};
                    }}
                    // Because PyArray_CumSum/CumProd returns a newly created reference on t.
                    Py_XDECREF(t);
                }}
            """.format(**locals())

        return code

    def c_code_cache_version(self):
        return (8,)

    def __str__(self):
        return f"{self.__class__.__name__}{{{self.axis}, {self.mode}}}"


def cumsum(x, axis=None):
    """Return the cumulative sum of the elements along a given `axis`.

    This wraps ``numpy.cumsum``.

    Parameters
    ----------
    x
        Input tensor variable.
    axis
        The axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    # noqa W293
    Example
    -------
    Usage in PyMC:

    .. code-block:: python

    with pm.Model() as model:
        x0 = pm.Normal('x0')
        x = pm.Normal('x', mu=0, sd=1, shape=10)
        # Gaussian random walk
        grw = pm.Deterministic('grw', x0 + pm.math.cumsum(x))

    """
    return CumOp(axis=axis, mode="add")(x)


def cumprod(x, axis=None):
    """Return the cumulative product of the elements along a given `axis`.

    This wraps ``numpy.cumprod``.

    Parameters
    ----------
    x
        Input tensor variable.
    axis
        The axis along which the cumulative product is computed.
        The default (None) is to compute the `cumprod` over the flattened array.


    .. versionadded:: 0.7

    """
    return CumOp(axis=axis, mode="mul")(x)


class CumsumOp(Op):
    __props__ = ("axis",)

    def __new__(typ, *args, **kwargs):
        obj = object.__new__(CumOp, *args, **kwargs)
        obj.mode = "add"
        return obj


class CumprodOp(Op):
    __props__ = ("axis",)

    def __new__(typ, *args, **kwargs):
        obj = object.__new__(CumOp, *args, **kwargs)
        obj.mode = "mul"
        return obj


def diff(x, n=1, axis=-1):
    """Calculate the `n`-th order discrete difference along the given `axis`.

    The first order difference is given by ``out[i] = a[i + 1] - a[i]``
    along the given `axis`, higher order differences are calculated by
    using `diff` recursively. This is heavily inspired by ``numpy.diff``.

    Parameters
    ----------
    x
        Input tensor variable.
    n
        The number of times values are differenced, default is 1.
    axis
        The axis along which the difference is taken, default is the last axis.


    .. versionadded:: 0.6

    """
    ndim = x.ndim
    axis = normalize_axis_index(axis, ndim)

    slice1 = [slice(None)] * ndim
    slice2 = [slice(None)] * ndim
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    for _ in range(n):
        x = x[slice1] - x[slice2]

    return x


def bincount(x, weights=None, minlength=None, assert_nonneg=False):
    """Count number of occurrences of each value in an array of integers.

    The number of bins (of size 1) is one larger than the largest
    value in `x`. If minlength is specified, there will be at least
    this number of bins in the output array (though it will be longer
    if necessary, depending on the contents of `x`). Each bin gives the
    number of occurrences of its index value in `x`. If `weights` is
    specified the input array is weighted by it, i.e. if a value ``n`` is found
    at position ``i``, ``out[n] += weight[i]`` instead of ``out[n] += 1``.

    Parameters
    ----------
    x
        A one dimensional array of non-negative integers
    weights
        An array of the same shape as `x` with corresponding weights.
        Optional.
    minlength
        A minimum number of bins for the output array.  Optional.
    assert_nonneg
        A flag that inserts an ``assert_op`` to check if
        every input `x` is non-negative.  Optional.

    .. versionadded:: 0.6

    """
    if x.ndim != 1:
        raise TypeError("Inputs must be of dimension 1.")

    if assert_nonneg:
        assert_op = Assert("Input to bincount has negative values!")
        x = assert_op(x, pt_all(x >= 0))

    max_value = ptb.cast(x.max() + 1, "int64")

    if minlength is not None:
        max_value = maximum(max_value, minlength)

    # Note: we do not use inc_subtensor(out[x], ...) in the following lines,
    # since out[x] raises an exception if the indices (x) are int8.
    if weights is None:
        out = ptb.zeros([max_value], dtype=x.dtype)
        out = advanced_inc_subtensor1(out, 1, x)
    else:
        out = ptb.zeros([max_value], dtype=weights.dtype)
        out = advanced_inc_subtensor1(out, weights, x)
    return out


def squeeze(x, axis=None):
    """
    Remove broadcastable (length 1) dimensions from the shape of an array.

    It returns the input array, but with the broadcastable dimensions
    removed. This is always `x` itself or a view into `x`.

    .. versionadded:: 0.6

    Parameters
    ----------
    x :
        Input data, tensor variable.
    axis : None or int or tuple of ints, optional
        Selects a subset of broadcastable dimensions to be removed.
        If a non broadcastable dimension is selected, an error is raised.
        If `axis` is ``None``, all broadcastable dimensions will be removed.

    Notes
    -----
    The behavior can differ from that of NumPy in two ways:
        1. If an axis is chosen for a dimension that is not known to be broadcastable
        an error is raised, even if this dimension would be broadcastable when the
        variable is evaluated.
        2. Similarly, if `axis` is ``None``, only dimensions known to be broadcastable will be
        removed, even if there are more dimensions that happen to be broadcastable when
        the variable is evaluated.

    Returns
    -------
    `x` without `axis` dimensions.

    """
    _x = ptb.as_tensor_variable(x)

    if axis is None:
        # By default exclude all broadcastable (length=1) axes
        axis = (i for i in range(_x.ndim) if _x.broadcastable[i])
    elif not isinstance(axis, Collection):
        axis = (axis,)

    # scalar inputs are treated as 1D regarding axis in this `Op`
    try:
        axis = np.core.numeric.normalize_axis_tuple(axis, ndim=max(1, _x.ndim))
    except np.AxisError:
        raise np.AxisError(axis, ndim=_x.ndim)

    if not axis:
        # Nothing to do
        return _x

    if _x.ndim == 0:
        # Nothing could be squeezed
        return _x

    # `Dimshuffle` raises when we try to drop an axis that is not statically broadcastable.
    # We add a `specify_broadcastable` instead of raising.
    non_broadcastable_axis = [i for i in axis if not _x.broadcastable[i]]
    _x = specify_broadcastable(_x, *non_broadcastable_axis)

    return _x.dimshuffle([i for i in range(_x.ndim) if i not in axis])


def compress(condition, x, axis=None):
    """
    Return selected slices of an array along given axis.

    It returns the input tensor, but with selected slices along a given `axis`
    retained. If no `axis` is provided, the tensor is flattened.
    Corresponds to ``numpy.compress``

    .. versionadded:: 0.7

    Parameters
    ----------
    condition
        One dimensional array of non-zero and zero values
        corresponding to indices of slices along a selected axis.
    x
        Input data, tensor variable.
    axis
        The axis along which to slice.

    Returns
    -------
    `x` with selected slices.

    """
    _x = ptb.as_tensor_variable(x)
    indices = ptb.flatnonzero(condition)
    return _x.take(indices, axis=axis)


class Repeat(Op):
    # See the repeat function for docstring

    __props__ = ("axis",)

    def __init__(self, axis=None):
        self.axis = axis

    def make_node(self, x, repeats):
        x = ptb.as_tensor_variable(x)
        repeats = ptb.as_tensor_variable(repeats)

        if repeats.dtype not in integer_dtypes:
            raise TypeError("repeats.dtype must be an integer.")

        # Some dtypes are not supported by numpy's implementation of repeat.
        # Until another one is available, we should fail at graph construction
        # time, not wait for execution.
        ptr_bitwidth = LOCAL_BITWIDTH
        if ptr_bitwidth == 64:
            numpy_unsupported_dtypes = ("uint64",)
        if ptr_bitwidth == 32:
            numpy_unsupported_dtypes = ("uint32", "int64", "uint64")

        if repeats.dtype in numpy_unsupported_dtypes:
            raise TypeError(
                (
                    f"dtypes {numpy_unsupported_dtypes!s} are not supported by numpy.repeat "
                    "for the 'repeats' parameter, "
                ),
                repeats.dtype,
            )

        if self.axis is None:
            out_shape = [None]
        else:
            try:
                const_reps = ptb.get_underlying_scalar_constant_value(repeats)
            except NotScalarConstantError:
                const_reps = None
            if const_reps == 1:
                out_shape = x.type.shape
            else:
                out_shape = list(x.type.shape)
                out_shape[self.axis] = None

        out_type = TensorType(
            x.dtype, shape=tuple(1 if s == 1 else None for s in out_shape)
        )

        return Apply(self, [x, repeats], [out_type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        repeats = inputs[1]
        z = output_storage[0]
        z[0] = np.repeat(x, repeats=repeats, axis=self.axis)

    def connection_pattern(self, node):
        return [[True], [False]]

    def grad(self, inputs, gout):
        (x, repeats) = inputs
        (gz,) = gout
        if repeats.ndim == 0:
            if self.axis is None:
                axis = x.ndim
            else:
                if self.axis >= 0:
                    axis = self.axis + 1
                else:
                    axis = self.axis + x.ndim + 1

            shape = [x.shape[k] for k in range(x.ndim)]
            shape.insert(axis, repeats)

            return [
                gz.reshape(shape, ndim=x.ndim + 1).sum(axis=axis),
                DisconnectedType()(),
            ]
        elif repeats.ndim == 1:
            # For this implementation, we would need to specify the length
            # of repeats in order to split gz in the right way to sum
            # the good part.
            raise NotImplementedError()
        else:
            raise ValueError()

    def infer_shape(self, fgraph, node, ins_shapes):
        i0_shapes = ins_shapes[0]
        repeats = node.inputs[1]
        out_shape = list(i0_shapes)

        # uint64 shape are not supported.
        dtype = None
        if repeats.dtype in ("uint8", "uint16", "uint32"):
            dtype = "int64"
        if self.axis is None:
            if repeats.ndim == 0:
                if len(i0_shapes) == 0:
                    out_shape = [repeats]
                else:
                    res = 1
                    for d in i0_shapes:
                        res = res * d
                    out_shape = (res * repeats,)
            else:
                out_shape = [pt_sum(repeats, dtype=dtype)]
        else:
            if repeats.ndim == 0:
                out_shape[self.axis] = out_shape[self.axis] * repeats
            else:
                out_shape[self.axis] = pt_sum(repeats, dtype=dtype)
        return [out_shape]


def repeat(x, repeats, axis=None):
    """Repeat elements of an array.

    It returns an array which has the same shape as `x`, except along the given
    `axis`. The `axis` parameter is used to specify the axis along which values
    are repeated. By default, a flattened version of `x` is used.

    The number of repetitions for each element is `repeats`.  `repeats` is
    broadcasted to fit the length of the given `axis`.

    Parameters
    ----------
    x
        Input data, tensor variable.
    repeats
        int, scalar or tensor variable
    axis : int, optional

    See Also
    --------
    tensor.tile

    .. versionadded:: 0.6

    """
    repeats = ptb.as_tensor_variable(repeats, dtype=np.int64)

    if repeats.ndim > 1:
        raise ValueError("The dimension of repeats should not exceed 1.")

    if repeats.ndim == 1 and not repeats.broadcastable[0]:
        return Repeat(axis=axis)(x, repeats)
    else:
        if repeats.ndim == 1:
            repeats = repeats[0]

        if x.dtype == "uint64":
            raise TypeError("repeat doesn't support dtype uint64")

        if axis is None:
            axis = 0
            x = x.flatten()
        else:
            if axis >= x.ndim:
                raise ValueError("Axis should not exceed x.ndim-1.")
            if axis < 0:
                axis = x.ndim + axis

        shape = [x.shape[i] for i in range(x.ndim)]

        # shape_ is the shape of the intermediate tensor which has
        # an additional dimension comparing to x. We use alloc to
        # allocate space for this intermediate tensor to replicate x
        # along that additional dimension.
        shape_ = shape[:]
        shape_.insert(axis + 1, repeats)

        # shape is now the shape of output, where shape[axis] becomes
        # shape[axis]*repeats.
        shape[axis] = shape[axis] * repeats

        # dims_ is the dimension of that intermediate tensor.
        dims_ = list(np.arange(x.ndim))
        dims_.insert(axis + 1, "x")

        # After the original tensor is duplicated along the additional
        # dimension, we reshape it to the expected output shape, and
        # return the output z.
        z = ptb.alloc(x.dimshuffle(*dims_), *shape_).reshape(shape)
        return z


class Bartlett(Op):
    # See function bartlett for docstring
    __props__ = ()

    def make_node(self, M):
        M = ptb.as_tensor_variable(M)
        if M.ndim != 0:
            raise TypeError(f"{self.__class__.__name__} only works on scalar input")
        elif M.dtype not in integer_dtypes:
            # dtype is an PyTensor attribute here
            raise TypeError(f"{self.__class__.__name__} only works on integer input")
        return Apply(self, [M], [dvector()])

    def perform(self, node, inputs, out_):
        M = inputs[0]
        (out,) = out_
        out[0] = np.bartlett(M)

    def infer_shape(self, fgraph, node, in_shapes):
        temp = node.inputs[0]
        M = ptb.switch(lt(temp, 0), ptb.cast(0, temp.dtype), temp)
        return [[M]]

    def grad(self, inputs, output_grads):
        return [None for i in inputs]


bartlett_ = Bartlett()


def bartlett(M):
    """
    An instance of this class returns the Bartlett spectral window in the
    time-domain. The Bartlett window is very similar to a triangular window,
    except that the end points are at zero. It is often used in signal
    processing for tapering a signal, without generating too much ripple in
    the frequency domain.

    .. versionadded:: 0.6

    Parameters
    ----------
    M : integer scalar
        Number of points in the output window. If zero or less,
        an empty vector is returned.

    Returns
    -------
    vector of doubles
        The triangular window, with the maximum value normalized to one
        (the value one appears only if the number of samples is odd), with
        the first and last samples equal to zero.

    """
    return bartlett_(M)


class FillDiagonal(Op):
    # See function fill_diagonal for docstring
    __props__ = ()

    def infer_shape(self, fgraph, node, in_shapes):
        return [in_shapes[0]]

    def make_node(self, a, val):
        a = ptb.as_tensor_variable(a)
        val = ptb.as_tensor_variable(val)
        if a.ndim < 2:
            raise TypeError(
                f"{self.__class__.__name__}: first parameter must have at least"
                " two dimensions"
            )
        elif val.ndim != 0:
            raise TypeError(
                f"{self.__class__.__name__}: second parameter must be a scalar"
            )
        val = ptb.cast(val, dtype=upcast(a.dtype, val.dtype))
        if val.dtype != a.dtype:
            raise TypeError(
                f"{self.__class__.__name__}: type of second parameter must be the same as"
                " the first's"
            )
        return Apply(self, [a, val], [a.type()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0].copy()
        val = inputs[1]
        if a.ndim == 2:
            # numpy.fill_diagonal up to date(including 1.6.2) have a
            # bug for tall matrix.
            # For 2-d arrays, we accept rectangular ones.
            step = a.shape[1] + 1
            end = a.shape[1] * a.shape[1]
            # Write the value out into the diagonal.
            a.flat[:end:step] = val
        else:
            np.fill_diagonal(a, val)

        output_storage[0][0] = a

    def grad(self, inp, cost_grad):
        """
        Notes
        -----
        The gradient is currently implemented for matrices only.

        """
        a, val = inp
        grad = cost_grad[0]
        if a.dtype.startswith("complex"):
            return [None, None]
        elif a.ndim > 2:
            raise NotImplementedError(
                f"{self.__class__.__name__}: gradient is currently implemented"
                " for matrices only"
            )
        wr_a = fill_diagonal(grad, 0)  # valid for any number of dimensions
        # diag is only valid for matrices
        wr_val = ptb.diag(grad).sum()
        return [wr_a, wr_val]


fill_diagonal_ = FillDiagonal()


def fill_diagonal(a, val):
    """
    Returns a copy of an array with all elements of the main diagonal set to a
    specified scalar value.

    .. versionadded:: 0.6

    Parameters
    ----------
    a
        Rectangular array of at least two dimensions.
    val
        Scalar value to fill the diagonal whose type must be
        compatible with that of array `a` (i.e. `val` cannot be viewed
        as an upcast of `a`).

    Returns
    -------
    array
        An array identical to `a` except that its main diagonal
        is filled with scalar `val`. (For an array `a` with ``a.ndim >=
        2``, the main diagonal is the list of locations ``a[i, i, ..., i]``
        (i.e. with indices all identical).)

    Support rectangular matrix and tensor with more than two dimensions
    if the later have all dimensions are equals.



    """
    return fill_diagonal_(a, val)


class FillDiagonalOffset(Op):
    # See function fill_diagonal_offset for docstring
    __props__ = ()

    def infer_shape(self, fgraph, node, in_shapes):
        return [in_shapes[0]]

    def make_node(self, a, val, offset):
        a = ptb.as_tensor_variable(a)
        val = ptb.as_tensor_variable(val)
        offset = ptb.as_tensor_variable(offset)
        if a.ndim != 2:
            raise TypeError(
                f"{self.__class__.__name__}: first parameter must have exactly"
                " two dimensions"
            )
        elif val.ndim != 0:
            raise TypeError(
                f"{self.__class__.__name__}: second parameter must be a scalar"
            )
        elif offset.ndim != 0:
            raise TypeError(
                f"{self.__class__.__name__}: third parameter must be a scalar"
            )
        val = ptb.cast(val, dtype=upcast(a.dtype, val.dtype))
        if val.dtype != a.dtype:
            raise TypeError(
                f"{self.__class__.__name__}: type of second parameter must be the same"
                " as the first's"
            )
        elif offset.dtype not in integer_dtypes:
            raise TypeError(
                f"{self.__class__.__name__}: type of third parameter must be as integer"
                " use pytensor.tensor.cast( input, 'int32/int64')"
            )

        return Apply(self, [a, val, offset], [a.type()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0].copy()
        val = inputs[1]
        offset = inputs[2]
        height, width = a.shape

        """
        Notes
        -----
        The fill_diagonal only support rectangular matrix. The output
        of tall matrix is "wrapped", which is an option in numpy 1.9.0
        but was regarded as a bug in numpy 1.6.2. Here I implement the
        fill_diagonal_offset with unwrapped output, so fill_diagonal_offset
        supports tall matrix.(This make a little difference between the output
        of fill_diagonal and fill_diagonal_offset only in the case of tall
        matrix)

        """
        if offset >= 0:
            start = offset
            num_of_step = min(min(width, height), width - offset)
        else:
            start = -offset * a.shape[1]
            num_of_step = min(min(width, height), height + offset)
        step = a.shape[1] + 1
        end = start + step * num_of_step
        # Write the value out into the diagonal.
        a.flat[start:end:step] = val

        output_storage[0][0] = a

    def grad(self, inp, cost_grad):
        """
        Notes
        -----
        The gradient is currently implemented for matrices only.
        """
        a, val, offset = inp
        grad = cost_grad[0]
        height, width = grad.shape

        if a.dtype.startswith("complex"):
            return [None, None]

        # only valid for matrices
        wr_a = fill_diagonal_offset(grad, 0, offset)

        offset_abs = pt_abs(offset)
        pos_offset_flag = ge(offset, 0)
        neg_offset_flag = lt(offset, 0)
        min_wh = minimum(width, height)

        start = offset * pos_offset_flag + offset_abs * width * neg_offset_flag
        num_of_step = minimum(
            min_wh, width * pos_offset_flag + height * neg_offset_flag - offset_abs
        )

        step = a.shape[1] + 1
        end = start + step * num_of_step

        # input of slice should be integer
        start = ptb.cast(start, "int32")
        step = ptb.cast(step, "int32")
        end = ptb.cast(end, "int32")

        wr_val = grad.flatten()[start:end:step].sum()

        wr_offset = grad_undefined(
            self,
            2,
            offset,
            "offset is not defined for non-integer offset so"
            " fill_diagonal_offset(a,val,offset+eps) is undefined",
        )

        return [wr_a, wr_val, wr_offset]


fill_diagonal_offset_ = FillDiagonalOffset()


def fill_diagonal_offset(a, val, offset):
    """
    Returns a copy of an array with all
    elements of the main diagonal set to a specified scalar value.

    Parameters
    ----------
    a
        Rectangular array of two dimensions.
    val
        Scalar value to fill the diagonal whose type must be
        compatible with that of array `a` (i.e. `val` cannot be viewed
        as an upcast of `a`).
    offset
        Scalar value Offset of the diagonal from the main
        diagonal. Can be positive or negative integer.

    Returns
    -------
    array
        An array identical to `a` except that its offset diagonal
        is filled with scalar `val`. The output is unwrapped.

    """
    return fill_diagonal_offset_(a, val, offset)


def to_one_hot(y, nb_class, dtype=None):
    """
    Return a matrix where each row correspond to the one hot
    encoding of each element in `y`.

    Parameters
    ----------
    y
        A vector of integer value between ``0`` and ``nb_class - 1``.
    nb_class : int
        The number of class in `y`.
    dtype : data-type
        The dtype of the returned matrix. Default ``pytensor.config.floatX``.

    Returns
    -------
    object
        A matrix of shape ``(y.shape[0], nb_class)``, where each row ``i`` is
        the one hot encoding of the corresponding ``y[i]`` value.

    """
    ret = ptb.zeros((y.shape[0], nb_class), dtype=dtype)
    ret = set_subtensor(ret[ptb.arange(y.shape[0]), y], 1)
    return ret


class Unique(Op):
    """
    Wraps `numpy.unique`.

    Examples
    --------
    >>> import numpy as np
    >>> import pytensor

    >>> x = pytensor.tensor.vector()
    >>> f = pytensor.function([x], Unique(True, True, False)(x))
    >>> f([1, 2., 3, 4, 3, 2, 1.])
    [array([ 1.,  2.,  3.,  4.]), array([0, 1, 2, 3]), array([0, 1, 2, 3, 2, 1, 0])]

    >>> y = pytensor.tensor.matrix()
    >>> g = pytensor.function([y], Unique(True, True, False)(y))
    >>> g([[1, 1, 1.0], (2, 3, 3.0)])
    [array([ 1.,  2.,  3.]), array([0, 3, 4]), array([0, 0, 0, 1, 2, 2])]

    """

    __props__ = ("return_index", "return_inverse", "return_counts", "axis")

    def __init__(
        self, return_index=False, return_inverse=False, return_counts=False, axis=None
    ):
        self.return_index = return_index
        self.return_inverse = return_inverse
        self.return_counts = return_counts
        self.axis = axis

    def make_node(self, x):
        x = ptb.as_tensor_variable(x)
        self_axis = self.axis
        if self_axis is None:
            out_shape = (None,)
        else:
            if self_axis < 0:
                self_axis += x.type.ndim
            if self_axis < 0 or self_axis >= x.type.ndim:
                raise ValueError(
                    f"Unique axis {self.axis} is outside of input ndim = {x.type.ndim}"
                )
            out_shape = tuple(
                s if s == 1 and axis != self_axis else None
                for axis, s in enumerate(x.type.shape)
            )

        outputs = [TensorType(dtype=x.dtype, shape=out_shape)()]
        typ = TensorType(dtype="int64", shape=(None,))
        if self.return_index:
            outputs.append(typ())
        if self.return_inverse:
            outputs.append(typ())
        if self.return_counts:
            outputs.append(typ())
        return Apply(self, [x], outputs)

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage
        param = {}
        if self.return_index:
            param["return_index"] = True
        if self.return_inverse:
            param["return_inverse"] = True
        if self.return_counts:
            param["return_counts"] = True
        if self.axis is not None:
            param["axis"] = self.axis
        outs = np.unique(x, **param)
        if (
            (not self.return_inverse)
            and (not self.return_index)
            and (not self.return_counts)
        ):
            z[0][0] = outs
        else:
            for i in range(len(outs)):
                z[i][0] = outs[i]

    def infer_shape(self, fgraph, node, i0_shapes):
        ret = fgraph.shape_feature.default_infer_shape(fgraph, node, i0_shapes)
        if self.axis is not None:
            self_axis = self.axis
            ndim = len(i0_shapes[0])
            if self_axis < 0:
                self_axis += ndim
            if self_axis < 0 or self_axis >= ndim:
                raise RuntimeError(
                    f"Unique axis `{self.axis}` is outside of input ndim = {ndim}."
                )
            ret[0] = tuple(
                [fgraph.shape_feature.shape_ir(i, node.outputs[0]) for i in range(ndim)]
            )
        if self.return_inverse:
            if self.axis is None:
                shape = (prod(i0_shapes[0]),)
            else:
                shape = (i0_shapes[0][self_axis],)
            if self.return_index:
                ret[2] = shape
                return ret
            ret[1] = shape
            return ret
        return ret

    def __setstate__(self, state):
        self.__dict__.update(state)
        # For backwards compatibility with pickled instances of Unique that
        # did not have the axis parameter specified
        if "axis" not in state:
            self.axis = None


def unique(
    ar, return_index=False, return_inverse=False, return_counts=False, axis=None
):
    """Find the unique elements of an array.

    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:

        * the indices of the input array that give the unique values
        * the indices of the unique array that reconstruct the input array
        * the number of times each unique value comes up in the input array

    """
    return Unique(return_index, return_inverse, return_counts, axis)(ar)


class UnravelIndex(Op):
    __props__ = ("order",)

    def __init__(self, order="C"):
        assert order in ("C", "F")
        self.order = order

    def make_node(self, indices, dims):
        indices = ptb.as_tensor_variable(indices)
        dims = ptb.as_tensor_variable(dims)

        if indices.dtype not in int_dtypes:
            raise TypeError(
                f"'{indices.dtype}' object cannot be interpreted as an index"
            )
        if dims.dtype not in int_dtypes:
            raise TypeError(f"'{dims.dtype}' object cannot be interpreted as an index")
        if dims.ndim != 1:
            raise TypeError("dims must be a 1D array")

        return Apply(
            self,
            [indices, dims],
            [
                TensorType(dtype="int64", shape=(None,) * indices.type.ndim)()
                for i in range(ptb.get_vector_length(dims))
            ],
        )

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]] * len(node.outputs)

    def perform(self, node, inp, out):
        indices, dims = inp
        res = np.unravel_index(indices, dims, order=self.order)
        assert len(res) == len(out)
        for i in range(len(out)):
            ret = _asarray(res[i], node.outputs[0].dtype)
            if ret.base is not None:
                # NumPy will return a view when it can.
                # But we don't want that.
                ret = ret.copy()
            out[i][0] = ret


def unravel_index(indices, dims, order="C"):
    """
    Converts a flat index or array of flat indices into a tuple
    of coordinate arrays.

    Parameters
    ----------
    indices : PyTensor or NumPy array
        An integer array whose elements are indices into the flattened
        version of an array of dimensions `dims`.
    dims : tuple of ints
        The shape of the array to use for unraveling `indices`.
    order : {'C', 'F'}, optional
        Determines whether the indices should be viewed as indexing in
        row-major (C-style) or column-major (Fortran-style) order.

    Returns
    -------
    unraveled_coords : tuple of ndarray
        Each array in the tuple has the same shape as the `indices`
        array.

    See Also
    --------
    ravel_multi_index

    """
    res = UnravelIndex(order=order)(indices, dims)
    if not isinstance(res, list | tuple):
        return (res,)
    else:
        return tuple(res)


class RavelMultiIndex(Op):
    __props__ = ("mode", "order")

    def __init__(self, mode="raise", order="C"):
        assert mode in ("raise", "wrap", "clip")
        assert order in ("C", "F")
        self.mode = mode
        self.order = order

    def make_node(self, *inp):
        multi_index = [ptb.as_tensor_variable(i) for i in inp[:-1]]
        dims = ptb.as_tensor_variable(inp[-1])

        for i in multi_index:
            if i.dtype not in int_dtypes:
                raise TypeError(f"'{i.dtype}' object cannot be interpreted as an index")
        if dims.dtype not in int_dtypes:
            raise TypeError(f"'{dims.dtype}' object cannot be interpreted as an index")
        if dims.ndim != 1:
            raise TypeError("dims must be a 1D array")

        return Apply(
            self,
            [*multi_index, dims],
            [TensorType(dtype="int64", shape=(None,) * multi_index[0].type.ndim)()],
        )

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]

    def perform(self, node, inp, out):
        multi_index, dims = inp[:-1], inp[-1]
        res = np.ravel_multi_index(multi_index, dims, mode=self.mode, order=self.order)
        out[0][0] = _asarray(res, node.outputs[0].dtype)


def ravel_multi_index(multi_index, dims, mode="raise", order="C"):
    """
    Converts a tuple of index arrays into an array of flat
    indices, applying boundary modes to the multi-index.

    Parameters
    ----------
    multi_index : tuple of PyTensor or NumPy arrays
        A tuple of integer arrays, one array for each dimension.
    dims : tuple of ints
        The shape of array into which the indices from ``multi_index`` apply.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices are handled.  Can specify
        either one mode or a tuple of modes, one mode per index.
        * 'raise' -- raise an error (default)
        * 'wrap' -- wrap around
        * 'clip' -- clip to the range
        In 'clip' mode, a negative index which would normally
        wrap will clip to 0 instead.
    order : {'C', 'F'}, optional
        Determines whether the multi-index should be viewed as
        indexing in row-major (C-style) or column-major
        (Fortran-style) order.

    Returns
    -------
    raveled_indices : TensorVariable
        An array of indices into the flattened version of an array
        of dimensions ``dims``.

    See Also
    --------
    unravel_index

    """
    if not isinstance(multi_index, tuple | list):
        raise TypeError("multi_index must be a tuple or a list.")
    args = (*multi_index, dims)
    return RavelMultiIndex(mode=mode, order=order)(*args)


_broadcast_assert = Assert(
    "Could not broadcast dimensions. Broadcasting is only allowed along "
    "axes that have a statically known length 1. Use `specify_broadcastable` to "
    "inform PyTensor of a known shape."
)
_runtime_broadcast_assert = Assert("Could not broadcast dimensions.")


def broadcast_shape(*arrays, **kwargs) -> tuple[ps.ScalarVariable, ...]:
    """Compute the shape resulting from broadcasting arrays.

    Parameters
    ----------
    *arrays: TensorVariable
        The tensor variables, or their shapes (as tuples),
        for which the broadcast shape is computed.
    arrays_are_shapes: bool (Optional)
        Indicates whether or not the `arrays` contains shape tuples.
        If you use this approach, make sure that the broadcastable dimensions
        are (scalar) constants with the value ``1``--or simply the integer
        ``1``.

    """
    return broadcast_shape_iter(arrays, **kwargs)


def broadcast_shape_iter(
    arrays: Iterable[TensorVariable | tuple[TensorVariable, ...]],
    arrays_are_shapes: bool = False,
    allow_runtime_broadcast: bool = False,
) -> tuple[ps.ScalarVariable, ...]:
    r"""Compute the shape resulting from broadcasting arrays.


    .. warning::

        This function will not make copies, so be careful when calling it with
        a generator/iterator!


    Parameters
    ----------
    arrays
        An iterable of tensors, or a tuple of shapes (as tuples),
        for which the broadcast shape is computed.
    arrays_are_shapes: bool, default False
        Indicates whether or not the `arrays` contains shape tuples.
        If you use this approach, make sure that the broadcastable dimensions
        are (scalar) constants with the value ``1``--or simply the integer
        ``1``. This is not revelant if `allow_runtime_broadcast` is True.
    allow_runtime_broadcast: bool, default False
        Whether to allow non-statically known broadcast on the shape computation.

    """
    one = pytensor.scalar.ScalarConstant(pytensor.scalar.int64, 1)

    if arrays_are_shapes:
        max_dims = max(len(a) for a in arrays)

        array_shapes = [
            (one,) * (max_dims - len(a))
            + tuple(
                one
                if sh == 1 or isinstance(sh, Constant) and sh.value == 1
                else (ps.as_scalar(sh) if not isinstance(sh, Variable) else sh)
                for sh in a
            )
            for a in arrays
        ]
    else:
        max_dims = max(a.ndim for a in arrays)

        _arrays = tuple(ptb.as_tensor_variable(a) for a in arrays)

        array_shapes = [
            (one,) * (max_dims - a.ndim)
            + tuple(one if t_sh == 1 else sh for sh, t_sh in zip(a.shape, a.type.shape))
            for a in _arrays
        ]

    result_dims = []

    for dim_shapes in zip(*array_shapes):
        # Get the shapes in this dimension that are not broadcastable
        # (i.e. not symbolically known to be broadcastable)
        non_bcast_shapes = [shape for shape in dim_shapes if shape != one]

        if len(non_bcast_shapes) == 0:
            # Every shape was broadcastable in this dimension
            result_dims.append(one)
        elif len(non_bcast_shapes) == 1:
            # Only one shape might not be broadcastable in this dimension
            result_dims.extend(non_bcast_shapes)
        else:
            # More than one shape might not be broadcastable in this dimension
            nonconst_nb_shapes: set[int] = set()
            const_nb_shapes: set[Variable] = set()
            for shape in non_bcast_shapes:
                if isinstance(shape, Constant):
                    const_nb_shapes.add(shape.value.item())
                else:
                    nonconst_nb_shapes.add(shape)

            if len(const_nb_shapes) > 1:
                raise ValueError(
                    f"Could not broadcast dimensions. Incompatible shapes were {array_shapes}."
                )

            if len(const_nb_shapes) == 1:
                (first_length,) = const_nb_shapes
                other_lengths = nonconst_nb_shapes
                first_length = ps.as_scalar(first_length)
            else:
                first_length, *other_lengths = nonconst_nb_shapes

            if len(other_lengths) == 0:
                result_dims.append(first_length)
                continue

            if not allow_runtime_broadcast:
                # Add assert that all remaining shapes are equal
                condition = pt_all(
                    [pt_eq(first_length, other) for other in other_lengths]
                )
                result_dims.append(_broadcast_assert(first_length, condition))
            else:
                lengths = as_tensor_variable((first_length, *other_lengths))
                runtime_broadcastable = pt_eq(lengths, one)
                result_dim = pt_abs(
                    pt_max(switch(runtime_broadcastable, -one, lengths))
                )
                condition = pt_all(
                    switch(
                        ~runtime_broadcastable,
                        pt_eq(lengths, result_dim),
                        np.array(True),
                    )
                )
                result_dims.append(_runtime_broadcast_assert(result_dim, condition))

    return tuple(result_dims)


def geomspace(start, end, steps, base=10.0):
    from pytensor.tensor.math import log

    start = ptb.as_tensor_variable(start)
    end = ptb.as_tensor_variable(end)
    return base ** linspace(log(start) / log(base), log(end) / log(base), steps)


def logspace(start, end, steps, base=10.0):
    start = ptb.as_tensor_variable(start)
    end = ptb.as_tensor_variable(end)
    return base ** linspace(start, end, steps)


def linspace(start, end, steps):
    start = ptb.as_tensor_variable(start)
    end = ptb.as_tensor_variable(end)
    arr = ptb.arange(steps)
    arr = ptb.shape_padright(arr, max(start.ndim, end.ndim))
    multiplier = (end - start) / (steps - 1)
    return start + arr * multiplier


def broadcast_to(
    x: TensorVariable, shape: TensorVariable | tuple[Variable, ...]
) -> TensorVariable:
    """Broadcast an array to a new shape.

    Parameters
    ----------
    array
        The array to broadcast.
    shape
        The shape of the desired array.

    Returns
    -------
    broadcast
        A readonly view on the original array with the given shape. It is
        typically not contiguous. Furthermore, more than one element of a
        broadcasted array may refer to a single memory location.

    """
    return alloc(x, *shape)


def broadcast_arrays(*args: TensorVariable) -> tuple[TensorVariable, ...]:
    """Broadcast any number of arrays against each other.

    Parameters
    ----------
    *args
        The arrays to broadcast.

    """

    def broadcast_with_others(a, others):
        for other in others:
            a = second(other, a)
        return a

    brodacasted_vars = []
    for i, a in enumerate(args):
        # We use indexing and not identity in case there are duplicated variables
        others = [a for j, a in enumerate(args) if j != i]
        brodacasted_vars.append(broadcast_with_others(a, others))

    return brodacasted_vars


__all__ = [
    "searchsorted",
    "cumsum",
    "cumprod",
    "diff",
    "bincount",
    "squeeze",
    "compress",
    "repeat",
    "bartlett",
    "fill_diagonal",
    "fill_diagonal_offset",
    "unique",
    "unravel_index",
    "ravel_multi_index",
    "broadcast_shape",
    "broadcast_to",
    "geomspace",
    "logspace",
    "linspace",
    "broadcast_arrays",
]
