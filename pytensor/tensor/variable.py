import copy
import traceback as tb
import warnings
from collections.abc import Iterable
from numbers import Number
from typing import TypeVar

import numpy as np

from pytensor import tensor as pt
from pytensor.configdefaults import config
from pytensor.graph.basic import Constant, OptionalApplyType, Variable
from pytensor.graph.utils import MetaType
from pytensor.scalar import (
    ComplexError,
)
from pytensor.tensor import _get_vector_length
from pytensor.tensor.exceptions import AdvancedIndexingError
from pytensor.tensor.type import TensorType
from pytensor.tensor.type_other import NoneConst
from pytensor.tensor.utils import hash_from_ndarray


_TensorTypeType = TypeVar("_TensorTypeType", bound=TensorType)


class _tensor_py_operators:
    # These can't work because Python requires native output types
    def __bool__(self):
        raise TypeError(
            "TensorVariable cannot be converted to Python boolean. "
            "Call `.astype(bool)` for the symbolic equivalent."
        )

    def __index__(self):
        raise TypeError(
            "TensorVariable cannot be converted to Python integer. "
            "Call `.astype(int)` for the symbolic equivalent."
        )

    def __int__(self):
        raise TypeError(
            "TensorVariable cannot be converted to Python integer. "
            "Call `.astype(int)` for the symbolic equivalent."
        )

    def __float__(self):
        raise TypeError(
            "TensorVariables cannot be converted to Python float. "
            "Call `.astype(float)` for the symbolic equivalent."
        )

    def __complex__(self):
        raise TypeError(
            "TensorVariables cannot be converted to Python complex number. "
            "Call `.astype(complex)` for the symbolic equivalent."
        )

    def __abs__(self):
        return pt.math.abs(self)

    def __neg__(self):
        return pt.math.neg(self)

    def __lt__(self, other):
        return pt.math.lt(self, other)

    def __le__(self, other):
        return pt.math.le(self, other)

    def __gt__(self, other):
        return pt.math.gt(self, other)

    def __ge__(self, other):
        return pt.math.ge(self, other)

    def __invert__(self):
        return pt.math.invert(self)

    def __and__(self, other):
        return pt.math.and_(self, other)

    def __or__(self, other):
        return pt.math.or_(self, other)

    def __xor__(self, other):
        return pt.math.xor(self, other)

    def __rand__(self, other):
        return pt.math.and_(other, self)

    def __ror__(self, other):
        return pt.math.or_(other, self)

    def __rxor__(self, other):
        return pt.math.xor(other, self)

    # def __iand__(self, other):
    #    return _and_inplace(self, other)
    #
    # def __ior__(self, other):
    #    return _or_inplace(self, other)
    #
    # def __ixor__(self, other):
    #    return _xor_inplace(self, other)

    def __add__(self, other):
        try:
            return pt.math.add(self, other)
        # We should catch the minimum number of exception here.
        # Otherwise this will convert error when PyTensor flags
        # compute_test_value is used
        # Evidently, we need to catch NotImplementedError
        # TypeError from as_tensor_variable are caught in Elemwise.make_node
        # Otherwise TensorVariable * SparseVariable won't work!
        except (NotImplementedError, TypeError):
            # We must return NotImplemented and not an
            # NotImplementedError or raise an NotImplementedError.
            # That way python will give a good error message like this
            # `TypeError: unsupported operand type(s) for +:
            # 'TensorVariable' and 'TensorVariable'`
            return NotImplemented

    def __sub__(self, other):
        # See explanation in __add__ for the error caught
        # and the return value in that case
        try:
            return pt.math.sub(self, other)
        except (NotImplementedError, TypeError):
            return NotImplemented

    def __mul__(self, other):
        # See explanation in __add__ for the error caught
        # and the return value in that case
        try:
            return pt.math.mul(self, other)
        except (NotImplementedError, TypeError):
            return NotImplemented

    def __pow__(self, other):
        # See explanation in __add__ for the error caught
        # and the return value in that case
        try:
            return pt.math.pow(self, other)
        except (NotImplementedError, TypeError):
            return NotImplemented

    def __mod__(self, other):
        # See explanation in __add__ for the error caught
        # and the return value in that case
        try:
            return pt.math.mod_check(self, other)
        except ComplexError:
            # This is to raise the exception that occurs when trying to compute
            # x % y with either x or y a complex number.
            raise
        except (NotImplementedError, TypeError):
            return NotImplemented

    def __divmod__(self, other):
        return pt.math.divmod(self, other)

    def __truediv__(self, other):
        return pt.math.true_div(self, other)

    def __floordiv__(self, other):
        return pt.math.floor_div(self, other)

    def __rtruediv__(self, other):
        return pt.math.true_div(other, self)

    def __rfloordiv__(self, other):
        return pt.math.floor_div(other, self)

    # Do not use these; in-place `Op`s should be inserted by optimizations
    # only!
    # def __iadd__(self, other):
    #    return _add_inplace(self, other)
    # def __isub__(self, other):
    #    return _sub_inplace(self, other)
    #
    # def __imul__(self, other):
    #    return _mul_inplace(self, other)
    #
    # def __idiv__(self, other):
    #    return _div_inplace(self, other)
    #
    # def __ipow__(self, other):
    #    return _pow_inplace(self, other)

    def __radd__(self, other):
        return pt.math.add(other, self)

    def __rsub__(self, other):
        return pt.math.sub(other, self)

    def __rmul__(self, other):
        return pt.math.mul(other, self)

    def __rmod__(self, other):
        return pt.math.mod(other, self)

    def __rdivmod__(self, other):
        return pt.math.divmod(other, self)

    def __rpow__(self, other):
        return pt.math.pow(other, self)

    def __ceil__(self):
        return pt.math.ceil(self)

    def __floor__(self):
        return pt.math.floor(self)

    def __trunc__(self):
        return pt.math.trunc(self)

    # NumPy-like transpose property
    @property
    def T(self):
        return pt.basic.transpose(self)

    @property
    def mT(self):
        return pt.basic.matrix_transpose(self)

    def transpose(self, *axes):
        """Transpose this array.

        Returns
        -------
        object
            `tensor.transpose(self, axes)` or `tensor.transpose(self, axes[0])`.

        If only one `axes` argument is provided and it is iterable, then it is
        assumed to be the entire axes tuple, and passed intact to
        tensor.transpose.

        """
        if len(axes) == 0:
            return pt.basic.transpose(self)
        try:
            iter(axes[0])
            iterable = True
        except TypeError:
            iterable = False
        if len(axes) == 1 and iterable:
            return pt.basic.transpose(self, axes[0])
        else:
            return pt.basic.transpose(self, axes)

    @property
    def shape(self):
        return pt.shape(self)

    @property
    def size(self):
        if self.ndim == 1:
            return self.shape[0]
        else:
            return pt.math.prod(self.shape)

    def any(self, axis=None, keepdims=False):
        return pt.math.any(self, axis=axis, keepdims=keepdims)

    def all(self, axis=None, keepdims=False):
        return pt.math.all(self, axis=axis, keepdims=keepdims)

    # Old note: "We can't implement this because Python requests that this
    # function returns an integer."
    # TODO: We could use `get_vector_length` and let it raise an exception just like
    # `__iter__` does
    # def __len__(self):
    #     raise Exception("PyTensor Variables can't work with len(PyTensor "
    #                     "Variable) due to Python restriction. You can use "
    #                     "PyTensorVariable.shape[0] instead.")

    def reshape(self, shape, *, ndim=None):
        """Return a reshaped view/copy of this variable.

        Parameters
        ----------
        shape
            Something that can be converted to a symbolic vector of integers.
        ndim
            The length of the shape. Passing None here means for
            PyTensor to try and guess the length of `shape`.


        .. warning:: This has a different signature than numpy's
                     ndarray.reshape!
                     In numpy you do not need to wrap the shape arguments
                     in a tuple, in pytensor you do need to.

        """
        if ndim is not None:
            if not isinstance(ndim, int):
                raise ValueError(
                    "Expected ndim to be an integer, is " + str(type(ndim))
                )

        return pt.reshape(self, shape, ndim=ndim)

    def dimshuffle(self, *pattern):
        """
        Reorder the dimensions of this variable, optionally inserting
        broadcasted dimensions.

        Parameters
        ----------
        pattern
            List/tuple of int mixed with 'x' for broadcastable dimensions.

        Examples
        --------
        For example, to create a 3D view of a [2D] matrix, call
        ``dimshuffle([0,'x',1])``.  This will create a 3D view such that the
        middle dimension is an implicit broadcasted dimension.  To do the same
        thing on the transpose of that matrix, call ``dimshuffle([1, 'x', 0])``.

        Notes
        -----
        This function supports the pattern passed as a tuple, or as a
        variable-length argument (e.g. ``a.dimshuffle(pattern)`` is equivalent
        to ``a.dimshuffle(*pattern)`` where ``pattern`` is a list/tuple of ints
        mixed with 'x' characters).

        See Also
        --------
        DimShuffle

        """
        if (len(pattern) == 1) and (isinstance(pattern[0], list | tuple | np.ndarray)):
            pattern = pattern[0]
        ds_op = pt.elemwise.DimShuffle(input_ndim=self.type.ndim, new_order=pattern)
        if ds_op.new_order == tuple(range(self.type.ndim)):
            # No-op
            return self
        return ds_op(self)

    def flatten(self, ndim=1):
        return pt.basic.flatten(self, ndim)

    def ravel(self):
        return pt.basic.flatten(self)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        return pt.basic.diagonal(self, offset, axis1, axis2)

    def transfer(self, target):
        """Transfer this this array's data to another device.

        If `target` is `'cpu'` this will transfer to a TensorType (if
        not already one).  Other types may define additional targets.

        Parameters
        ----------
        target : str
            The desired location of the output variable
        """
        return pt.basic.transfer(self, target)

    def arccos(self):
        return pt.math.arccos(self)

    def arccosh(self):
        return pt.math.arccosh(self)

    def arcsin(self):
        return pt.math.arcsin(self)

    def arcsinh(self):
        return pt.math.arcsinh(self)

    def arctan(self):
        return pt.math.arctan(self)

    def arctanh(self):
        return pt.math.arctanh(self)

    def ceil(self):
        return pt.math.ceil(self)

    def cos(self):
        return pt.math.cos(self)

    def cosh(self):
        return pt.math.cosh(self)

    def deg2rad(self):
        return pt.math.deg2rad(self)

    def exp(self):
        return pt.math.exp(self)

    def exp2(self):
        return pt.math.exp2(self)

    def expm1(self):
        return pt.math.expm1(self)

    def floor(self):
        return pt.math.floor(self)

    def log(self):
        return pt.math.log(self)

    def log10(self):
        return pt.math.log10(self)

    def log1p(self):
        return pt.math.log1p(self)

    def log2(self):
        return pt.math.log2(self)

    def rad2deg(self):
        return pt.math.rad2deg(self)

    def sin(self):
        return pt.math.sin(self)

    def sinh(self):
        return pt.math.sinh(self)

    def sqrt(self):
        return pt.math.sqrt(self)

    def tan(self):
        return pt.math.tan(self)

    def tanh(self):
        return pt.math.tanh(self)

    def trunc(self):
        return pt.math.trunc(self)

    def astype(self, dtype):
        return pt.basic.cast(self, dtype)

    def __getitem__(self, args):
        def includes_bool(args_el):
            if isinstance(args_el, np.bool_ | bool) or (
                hasattr(args_el, "dtype") and args_el.dtype == "bool"
            ):
                return True
            if not isinstance(args_el, Variable) and isinstance(args_el, Iterable):
                for el in args_el:
                    if includes_bool(el):
                        return True
            return False

        if isinstance(args, list) and any(isinstance(a, slice) for a in args):
            pass
        elif not isinstance(args, tuple):
            args = (args,)

        # Count the dimensions, check for bools and find ellipses.
        ellipses = []
        index_dim_count = 0
        for i, arg in enumerate(args):
            if arg is np.newaxis or arg is NoneConst:
                # no increase in index_dim_count
                pass
            elif arg is Ellipsis:
                # no increase in index_dim_count
                ellipses.append(i)
            elif (
                isinstance(arg, np.ndarray | Variable)
                and hasattr(arg, "dtype")
                and arg.dtype == "bool"
            ):
                index_dim_count += arg.ndim
            else:
                # Python arrays can contain a mixture of bools and integers,
                # which requires complex rules to handle all special cases.
                # These rules differ slightly between NumPy versions.
                # Since earlier versions of PyTensor did not support any boolean
                # indexing, it is safe to throw an error if we encounter
                # any of these difficult cases.
                if includes_bool(arg):
                    raise TypeError(
                        "TensorType does not support Python bools "
                        "for indexing, such as tensor[[True, False]]. "
                        "To use a boolean mask, convert the mask to "
                        "a NumPy array first, e.g., "
                        "tensor[numpy.array([True, False])]."
                    )
                index_dim_count += 1

        # Check if the number of dimensions isn't too large.
        if self.ndim < index_dim_count:
            raise IndexError(
                f"too many indices for tensor: tensor is {self.ndim}-dimensional, but {index_dim_count} were indexed"
            )

        # Convert an Ellipsis if provided into an appropriate number of
        # slice(None).
        if len(ellipses) > 1:
            raise IndexError("an index can only have a single Ellipsis (`...`)")
        elif len(ellipses) == 1:
            ellipsis_pt = ellipses[0]
            args = list(args)
            args[ellipsis_pt : ellipsis_pt + 1] = [slice(None)] * (
                self.ndim - index_dim_count
            )

        def is_empty_array(val):
            return (isinstance(val, tuple | list) and len(val) == 0) or (
                isinstance(val, np.ndarray) and val.size == 0
            )

        # Force input to be an int datatype if input is an empty list or tuple
        # Else leave it as is if it is a real number
        # Convert python literals to pytensor constants
        args = tuple(
            pt.subtensor.as_index_constant(
                np.array(inp, dtype=np.uint8) if is_empty_array(inp) else inp
            )
            for inp in args
        )

        # Determine if advanced indexing is needed or not.  The logic is
        # already in `index_vars_to_types`: if it succeeds, standard indexing is
        # used; if it fails with `AdvancedIndexingError`, advanced indexing is
        # used
        advanced = False
        for i, arg in enumerate(args):
            if includes_bool(arg):
                advanced = True
                break

            if arg is not np.newaxis and arg is not NoneConst:
                try:
                    pt.subtensor.index_vars_to_types(arg)
                except AdvancedIndexingError:
                    if advanced:
                        break
                    else:
                        advanced = True

        if advanced:
            return pt.subtensor.advanced_subtensor(self, *args)
        else:
            if np.newaxis in args or NoneConst in args:
                # `np.newaxis` (i.e. `None`) in NumPy indexing mean "add a new
                # broadcastable dimension at this location".  Since PyTensor adds
                # new broadcastable dimensions via the `DimShuffle` `Op`, the
                # following code uses said `Op` to add one of the new axes and
                # then uses recursion to apply any other indices and add any
                # remaining new axes.

                counter = 0
                pattern = []
                new_args = []
                for arg in args:
                    if arg is np.newaxis or arg is NoneConst:
                        pattern.append("x")
                        new_args.append(slice(None, None, None))
                    else:
                        pattern.append(counter)
                        counter += 1
                        new_args.append(arg)

                pattern.extend(list(range(counter, self.ndim)))

                view = self.dimshuffle(pattern)
                full_slices = True
                for arg in new_args:
                    # We can't do arg == slice(None, None, None) as in
                    # Python 2.7, this call __lt__ if we have a slice
                    # with some symbolic variable.
                    if not (
                        isinstance(arg, slice)
                        and (arg.start is None or arg.start is NoneConst)
                        and (arg.stop is None or arg.stop is NoneConst)
                        and (arg.step is None or arg.step is NoneConst)
                    ):
                        full_slices = False
                if full_slices:
                    return view
                else:
                    return view.__getitem__(tuple(new_args))
            else:
                return pt.subtensor.Subtensor(args)(
                    self,
                    *pt.subtensor.get_slice_elements(
                        args, lambda entry: isinstance(entry, Variable)
                    ),
                )

    def __setitem__(self, key, value):
        raise TypeError(
            "TensorVariable does not support item assignment. Use the output of `x[idx].set` or `x[idx].inc` instead."
        )

    def take(self, indices, axis=None, mode="raise"):
        return pt.subtensor.take(self, indices, axis, mode)

    def copy(self, name=None):
        """Return a symbolic copy and optionally assign a name.

        Does not copy the tags.
        """
        copied_variable = pt.basic.tensor_copy(self)
        copied_variable.name = name
        return copied_variable

    def __iter__(self):
        try:
            for i in range(pt.basic.get_vector_length(self)):
                yield self[i]
        except TypeError:
            # This prevents accidental iteration via sum(self)
            raise TypeError(
                "TensorType does not support iteration.\n"
                "\tDid you pass a PyTensor variable to a function that expects a list?\n"
                "\tMaybe you are using builtins.sum instead of pytensor.tensor.sum?"
            )

    @property
    def ndim(self) -> int:
        """The rank of this tensor."""
        return self.type.ndim

    @property
    def broadcastable(self):
        """
        The broadcastable signature of this tensor.

        See Also
        --------
        broadcasting

        """
        return self.type.broadcastable

    @property
    def dtype(self):
        """The dtype of this tensor."""
        return self.type.dtype

    def __dot__(left, right):
        return pt.math.dense_dot(left, right)

    def __rdot__(right, left):
        return pt.math.dense_dot(left, right)

    dot = __dot__

    def __matmul__(left, right):
        return pt.math.matmul(left, right)

    def __rmatmul__(right, left):
        return pt.math.matmul(left, right)

    def sum(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        """See :func:`pytensor.tensor.math.sum`."""
        return pt.math.sum(
            self, axis=axis, dtype=dtype, keepdims=keepdims, acc_dtype=acc_dtype
        )

    def prod(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        """See :func:`pytensor.tensor.math.prod`."""
        return pt.math.prod(
            self, axis=axis, dtype=dtype, keepdims=keepdims, acc_dtype=acc_dtype
        )

    def norm(self, L, axis=None, keepdims=False):
        if L == 0:
            raise NotImplementedError()
        if np.isinf(L):
            raise NotImplementedError()
        # optimizations will/should catch cases like L=1, L=2
        y = pt.math.pow(
            pt.math.pow(pt.math.abs(self), L).sum(axis=axis),
            1.0 / L,
        )
        if keepdims:
            return pt.math.makeKeepDims(self, y, axis)
        else:
            return y

    def mean(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        """See :func:`pytensor.tensor.math.mean`."""
        return pt.math.mean(
            self, axis=axis, dtype=dtype, keepdims=keepdims, acc_dtype=acc_dtype
        )

    def var(self, axis=None, ddof=0, keepdims=False, corrected=False):
        """See :func:`pytensor.tensor.math.var`."""
        return pt.math.var(
            self, axis=axis, ddof=ddof, keepdims=keepdims, corrected=corrected
        )

    def std(self, axis=None, ddof=0, keepdims=False, corrected=False):
        """See :func:`pytensor.tensor.math.std`."""
        return pt.math.std(
            self, axis=axis, ddof=ddof, keepdims=keepdims, corrected=corrected
        )

    def min(self, axis=None, keepdims=False):
        """See :func:`pytensor.tensor.math.min`."""
        return pt.math.min(self, axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        """See :func:`pytensor.tensor.math.max`."""
        return pt.math.max(self, axis, keepdims=keepdims)

    def argmin(self, axis=None, keepdims=False):
        """See :func:`pytensor.tensor.math.argmin`."""
        return pt.math.argmin(self, axis, keepdims=keepdims)

    def argmax(self, axis=None, keepdims=False):
        """See :func:`pytensor.tensor.math.argmax`."""
        return pt.math.argmax(self, axis, keepdims=keepdims)

    def nonzero(self, return_matrix=False):
        """See :func:`pytensor.tensor.basic.nonzero`."""
        return pt.nonzero(self, return_matrix=return_matrix)

    def nonzero_values(self):
        """See :func:`pytensor.tensor.basic.nonzero_values`."""
        return pt.nonzero_values(self)

    def sort(self, axis=-1, kind="quicksort", order=None):
        """See :func:`pytensor.tensor.sort.sort`."""
        return pt.sort(self, axis, kind, order)

    def argsort(self, axis=-1, kind="quicksort", order=None):
        """See :func:`pytensor.tensor.sort.argsort`."""
        from pytensor.tensor.sort import argsort

        return argsort(self, axis, kind, order)

    def clip(self, a_min, a_max):
        "See :func:`pytensor.tensor.math.clip`."
        return pt.math.clip(self, a_min, a_max)

    def conj(self):
        """See :func:`pytensor.tensor.math.conj`."""
        return pt.math.conj(self)

    conjugate = conj

    def repeat(self, repeats, axis=None):
        """See :func:`pytensor.tensor.basic.repeat`."""
        return pt.extra_ops.repeat(self, repeats, axis)

    def round(self, mode=None):
        """See :func:`pytensor.tensor.math.round`."""
        return pt.math.round(self, mode)

    def trace(self):
        return pt.linalg.trace(self)

    # This value is set so that PyTensor arrays will trump NumPy operators.
    __array_priority__ = 1000

    def get_underlying_scalar_constant(self):
        return pt.basic.get_underlying_scalar_constant_value(self)

    def zeros_like(model, dtype=None):
        return pt.basic.zeros_like(model, dtype=dtype)

    def ones_like(model, dtype=None):
        return pt.basic.ones_like(model, dtype=dtype)

    def cumsum(self, axis=None):
        return pt.extra_ops.cumsum(self, axis)

    def cumprod(self, axis=None):
        return pt.extra_ops.cumprod(self, axis)

    def searchsorted(self, v, side="left", sorter=None):
        return pt.extra_ops.searchsorted(self, v, side, sorter)

    def ptp(self, axis=None):
        """See :func:`pytensor.tensor.math.ptp`."""

        return pt.math.ptp(self, axis)

    def swapaxes(self, axis1, axis2):
        """See :func:`pytensor.tensor.basic.swapaxes`.

        If a matrix is provided with the right axes, its transpose
        will be returned.

        """
        return pt.basic.swapaxes(self, axis1, axis2)

    def fill(self, value):
        """Fill inputted tensor with the assigned value."""
        return pt.basic.fill(self, value)

    def choose(self, choices, mode="raise"):
        """
        Construct an array from an index array and a set of arrays to choose
        from.

        """
        return pt.basic.choose(self, choices, mode="raise")

    def squeeze(self, axis=None):
        """
        Remove broadcastable dimensions from the shape of an array.

        It returns the input array, but with the broadcastable dimensions
        removed. This is always `x` itself or a view into `x`.

        """
        return pt.extra_ops.squeeze(self, axis=axis)

    def compress(self, a, axis=None):
        """Return selected slices only."""
        return pt.extra_ops.compress(self, a, axis=axis)

    def set(self, y, **kwargs):
        """Return a copy of the variable indexed by self with the indexed values set to y.

        Equivalent to set_subtensor(self, y). See docstrings for kwargs.

        Raises
        ------
        TypeError:
            If self is not the result of a subtensor operation

        Examples
        --------
        >>> import pytensor.tensor as pt
        >>>
        >>> x = pt.ones((3,))
        >>> out = x[1].set(2)
        >>> out.eval()
        array([1., 2., 1.])
        """
        return pt.subtensor.set_subtensor(self, y, **kwargs)

    def inc(self, y, **kwargs):
        """Return a copy of the variable indexed by self with the indexed values incremented by y.

        Equivalent to inc_subtensor(self, y). See docstrings for kwargs.

        Raises
        ------
        TypeError:
            If self is not the result of a subtensor operation

        Examples
        --------

        >>> import pytensor.tensor as pt
        >>>
        >>> x = pt.ones((3,))
        >>> out = x[1].inc(2)
        >>> out.eval()
        array([1., 3., 1.])
        """
        return pt.inc_subtensor(self, y, **kwargs)


class TensorVariable(
    _tensor_py_operators, Variable[_TensorTypeType, OptionalApplyType]
):
    """
    Subclass to add the tensor operators to the basic `Variable` class.

    """

    def __init__(
        self,
        type: _TensorTypeType,
        owner: OptionalApplyType,
        index=None,
        name=None,
    ):
        super().__init__(type, owner, index=index, name=name)
        if config.warn_float64 != "ignore" and type.dtype == "float64":
            msg = (
                "You are creating a TensorVariable "
                "with float64 dtype. You requested an action via "
                "the PyTensor flag warn_float64={ignore,warn,raise,pdb}."
            )
            if config.warn_float64 == "warn":
                # Get the user stack. We don't want function inside the
                # tensor and graph directory to be shown to the user.
                x = tb.extract_stack()
                nb_rm = 0
                while x:
                    file_path = x[-1][0]
                    rm = False
                    for p in [
                        "pytensor/tensor/",
                        "pytensor\\tensor\\",
                        "pytensor/graph/",
                        "pytensor\\tensor\\",
                    ]:
                        if p in file_path:
                            x = x[:-1]
                            nb_rm += 1
                            rm = True
                            break
                    if not rm:
                        break
                warnings.warn(msg, stacklevel=1 + nb_rm)
            elif config.warn_float64 == "raise":
                raise Exception(msg)
            elif config.warn_float64 == "pdb":
                import pdb

                pdb.set_trace()


@_get_vector_length.register(TensorVariable)
def _get_vector_length_TensorVariable(op_or_var, var):
    if var.type.shape[0] is None:
        raise ValueError(f"Length of {var} cannot be determined")
    return var.type.shape[0]


TensorType.variable_type = TensorVariable


class TensorConstantSignature(tuple):
    r"""A signature object for comparing `TensorConstant` instances.

    An instance is a pair with the type ``(Type, ndarray)``.

    TODO FIXME: Subclassing `tuple` is unnecessary, and it appears to be
    preventing the use of a much more convenient `__init__` that removes the
    need for all these lazy computations and their safety checks.

    Also, why do we even need this signature stuff?  We could simply implement
    good `Constant.__eq__` and `Constant.__hash__` implementations.

    We could also produce plain `tuple`\s with hashable values.

    """

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        try:
            (t0, d0), (t1, d1) = self, other
        except Exception:
            return False

        # N.B. compare shape to ensure no broadcasting in ==
        if t0 != t1 or d0.shape != d1.shape:
            return False

        self.no_nan  # Ensure has_nan is computed.
        # Note that in the comparisons below, the elementwise comparisons
        # come last because they are the most expensive checks.
        if self.has_nan:
            other.no_nan  # Ensure has_nan is computed.
            return (
                other.has_nan
                and self.sum == other.sum
                and (self.no_nan.mask == other.no_nan.mask).all()
                and
                # Note that the second test below (==) may crash e.g. for
                # a single scalar NaN value, so we do not run it when all
                # values are missing.
                (self.no_nan.mask.all() or (self.no_nan == other.no_nan).all())
            )
        else:
            # Simple case where we do not need to worry about NaN values.
            # (note that if there are NaN values in d1, this will return
            # False, which is why we do not bother with testing `other.has_nan`
            # here).
            return (self.sum == other.sum) and np.all(d0 == d1)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        t, d = self
        return hash((type(self), t, d.shape, self.sum))

    def pytensor_hash(self):
        _, d = self
        return hash_from_ndarray(d)

    @property
    def sum(self):
        """Compute sum of non NaN / Inf values in the array."""
        try:
            return self._sum
        except AttributeError:
            # Prevent warnings when there are `inf`s and `-inf`s present
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self._sum = self.no_nan.sum()

            # The following 2 lines are needed as in Python 3.3 with NumPy
            # 1.7.1, numpy.ndarray and numpy.memmap aren't hashable.
            if isinstance(self._sum, np.memmap):
                self._sum = np.asarray(self._sum).item()

            if self.has_nan and self.no_nan.mask.all():
                # In this case the sum is not properly computed by numpy.
                self._sum = 0

            if np.isinf(self._sum) or np.isnan(self._sum):
                # NaN may happen when there are both -inf and +inf values.
                if self.has_nan:
                    # Filter both NaN and Inf values.
                    mask = self.no_nan.mask + np.isinf(self[1])
                else:
                    # Filter only Inf values.
                    mask = np.isinf(self[1])
                if mask.all():
                    self._sum = 0
                else:
                    self._sum = np.ma.masked_array(self[1], mask).sum()
                # At this point there should be no more NaN.
                assert not np.isnan(self._sum)

            if isinstance(self._sum, np.ma.core.MaskedConstant):
                self._sum = 0

        return self._sum

    @property
    def no_nan(self):
        try:
            return self._no_nan
        except AttributeError:
            nans = np.isnan(self[1])
            self._no_nan = np.ma.masked_array(self[1], nans)
            self.has_nan = np.any(nans)
        return self._no_nan


def get_unique_constant_value(x: TensorVariable) -> Number | None:
    """Return the unique value of a tensor, if there is one"""
    warnings.warn("get_unique_constant_value is deprecated.", FutureWarning)
    if isinstance(x, TensorConstant):
        return x.unique_value
    return None


class TensorConstant(TensorVariable, Constant[_TensorTypeType]):
    """Subclass to add the tensor operators to the basic `Constant` class."""

    def __init__(self, type: _TensorTypeType, data, name=None):
        data_shape = np.shape(data)

        if len(data_shape) != type.ndim or any(
            ds != ts
            for ds, ts in zip(np.shape(data), type.shape, strict=True)
            if ts is not None
        ):
            raise ValueError(
                f"Shape of data ({data_shape}) does not match shape of type ({type.shape})"
            )

        # We want all the shape information from `data`
        new_type = type.clone(shape=data_shape)

        assert not any(s is None for s in new_type.shape)

        Constant.__init__(self, new_type, data, name)

    def signature(self):
        return TensorConstantSignature((self.type, self.data))

    @property
    def unique_value(self) -> Number | None:
        """Return the unique value of a tensor, if there is one"""
        try:
            return self._unique_value
        except AttributeError:
            data = self.data
            unique_value = None
            if data.size > 0:
                if data.size == 1:
                    unique_value = data.squeeze()
                else:
                    flat_data = data.ravel()
                    if (flat_data == flat_data[0]).all():
                        unique_value = flat_data[0]

                if unique_value is not None:
                    # Don't allow the unique value to be changed
                    unique_value.setflags(write=False)

            self._unique_value = unique_value

        return self._unique_value

    def equals(self, other):
        # Override Constant.equals to allow to compare with
        # numpy.ndarray, and python type.
        if isinstance(other, np.ndarray | int | float):
            # Make a TensorConstant to be able to compare
            other = pt.basic.constant(other)
        return (
            isinstance(other, TensorConstant) and self.signature() == other.signature()
        )

    def __copy__(self):
        # We need to do this to remove the cached attribute
        return type(self)(self.type, self.data, self.name)

    def __deepcopy__(self, memo):
        # We need to do this to remove the cached attribute
        return type(self)(
            copy.deepcopy(self.type, memo),
            copy.deepcopy(self.data, memo),
            copy.deepcopy(self.name, memo),
        )


TensorType.constant_type = TensorConstant


class DenseVariableMeta(MetaType):
    def __instancecheck__(self, o):
        if type(o) is TensorVariable or isinstance(o, DenseVariableMeta):
            return True
        return False


class DenseTensorVariable(TensorType, metaclass=DenseVariableMeta):
    r"""A `Variable` for dense tensors.

    Instances of this class and `TensorVariable`\s are considered dense
    `Variable`\s.
    """


class DenseConstantMeta(MetaType):
    def __instancecheck__(self, o):
        if type(o) is TensorConstant or isinstance(o, DenseConstantMeta):
            return True
        return False


class DenseTensorConstant(TensorType, metaclass=DenseConstantMeta):
    r"""A `Constant` for dense tensors.

    Instances of this class and `TensorConstant`\s are considered dense
    `Constant`\s.
    """
