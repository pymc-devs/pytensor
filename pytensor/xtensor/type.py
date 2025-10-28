import typing
import warnings
from types import EllipsisType

from pytensor.compile import (
    DeepCopyOp,
    ViewOp,
    register_deep_copy_op_c_code,
    register_view_op_c_code,
)
from pytensor.tensor import (
    TensorType,
    _as_tensor_variable,
    as_tensor_variable,
    specify_shape,
)
from pytensor.tensor.math import variadic_mul


try:
    import xarray as xr

    XARRAY_AVAILABLE = True
except ModuleNotFoundError:
    XARRAY_AVAILABLE = False

from collections.abc import Sequence
from typing import Any, Literal, TypeVar

import numpy as np

import pytensor.xtensor as px
from pytensor import _as_symbolic, config
from pytensor.graph import Apply, Constant
from pytensor.graph.basic import OptionalApplyType, Variable
from pytensor.graph.type import HasDataType, HasShape, Type
from pytensor.tensor.basic import constant as tensor_constant
from pytensor.tensor.variable import TensorConstantSignature, TensorVariable


class XTensorType(Type, HasDataType, HasShape):
    """A `Type` for Xtensors (Xarray-like tensors with dims)."""

    __props__ = ("dtype", "shape", "dims")

    def __init__(
        self,
        dtype: str | np.dtype,
        *,
        dims: Sequence[str],
        shape: Sequence[int | None] | None = None,
        name: str | None = None,
    ):
        if dtype == "floatX":
            self.dtype = config.floatX
        else:
            self.dtype = np.dtype(dtype).name

        self.dims = tuple(dims)
        if len(set(dims)) < len(dims):
            raise ValueError(f"Dimensions must be unique. Found duplicates in {dims}: ")
        if shape is None:
            self.shape = (None,) * len(self.dims)
        else:
            self.shape = tuple(shape)
            if len(self.shape) != len(self.dims):
                raise ValueError(
                    f"Shape {self.shape} must have the same length as dims {self.dims}"
                )
        self.ndim = len(self.dims)
        self.name = name
        self.numpy_dtype = np.dtype(self.dtype)
        self.filter_checks_isfinite = False
        # broadcastable is here just for code that would work fine with XTensorType but checks for it
        self.broadcastable = (False,) * self.ndim

    def clone(
        self,
        dtype=None,
        dims=None,
        shape=None,
        **kwargs,
    ):
        if dtype is None:
            dtype = self.dtype
        if dims is None:
            dims = self.dims
        if shape is None:
            shape = self.shape
        return type(self)(dtype=dtype, shape=shape, dims=dims, **kwargs)

    def filter(self, value, strict=False, allow_downcast=None):
        # XTensorType behaves like TensorType at runtime, so we filter the same way.
        return TensorType.filter(
            self, value, strict=strict, allow_downcast=allow_downcast
        )

    @staticmethod
    def may_share_memory(a, b):
        return TensorType.may_share_memory(a, b)

    def filter_variable(self, other, allow_convert=True):
        if not isinstance(other, Variable):
            # The value is not a Variable: we cast it into
            # a Constant of the appropriate Type.
            other = xtensor_constant(other)

        if self.is_super(other.type):
            return other

        if allow_convert:
            other2 = self.convert_variable(other)
            if other2 is not None:
                return other2

        raise TypeError(
            f"Cannot convert Type {other.type} (of Variable {other}) into Type {self}."
            f"You can try to manually convert {other} into a {self}. "
        )

    def convert_variable(self, var):
        var_type = var.type
        if self.is_super(var_type):
            return var
        if isinstance(var_type, XTensorType):
            if (
                self.ndim != var_type.ndim
                or self.dtype != var_type.dtype
                or set(self.dims) != set(var_type.dims)
            ):
                return None

            if self.dims != var_type.dims:
                var = var.transpose(*self.dims)
                var_type = var.type
                if self.is_super(var_type):
                    return var

            if any(
                s_length is not None
                and var_length is not None
                and s_length != var_length
                for s_length, var_length in zip(self.shape, var_type.shape)
            ):
                # Incompatible static shapes
                return None

            # Needs a specify_shape
            return as_xtensor(specify_shape(var.values, self.shape), dims=self.dims)

        if isinstance(var_type, TensorType):
            if (
                self.ndim != var_type.ndim
                or self.dtype != var_type.dtype
                or any(
                    s_length is not None
                    and var_length is not None
                    and s_length != var_length
                    for s_length, var_length in zip(self.shape, var_type.shape)
                )
            ):
                return None
            else:
                return as_xtensor(specify_shape(var, self.shape), dims=self.dims)

        return None

    def __repr__(self):
        return f"XTensorType({self.dtype}, shape={self.shape}, dims={self.dims})"

    def __hash__(self):
        return hash((type(self), self.dtype, self.shape, self.dims))

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.dtype == other.dtype
            and self.dims == other.dims
            and self.shape == other.shape
        )

    def is_super(self, otype):
        if type(self) is not type(otype):
            return False
        if self.dtype != otype.dtype:
            return False
        if self.dims != otype.dims:
            return False
        if any(
            s_dim_length is not None and s_dim_length != o_dim_length
            for s_dim_length, o_dim_length in zip(self.shape, otype.shape)
        ):
            return False
        return True


def xtensor(
    name: str | None = None,
    *,
    dims: Sequence[str],
    shape: Sequence[int | None] | None = None,
    dtype: str | np.dtype = "floatX",
):
    """Create an XTensorVariable.

    Parameters
    ----------
    name : str or None, optional
        The name of the variable
    dims : Sequence[str]
        The names of the dimensions of the tensor
    shape : Sequence[int | None] or None, optional
        The shape of the tensor. If None, defaults to a shape with None for each dimension.
    dtype : str or np.dtype, optional
        The data type of the tensor. Defaults to 'floatX' (config.floatX).

    Returns
    -------
    XTensorVariable
        A new XTensorVariable with the specified name, dims, shape, and dtype.
    """
    return XTensorType(dtype=dtype, dims=dims, shape=shape)(name=name)


_XTensorTypeType = TypeVar("_XTensorTypeType", bound=XTensorType)


class XTensorVariable(Variable[_XTensorTypeType, OptionalApplyType]):
    """Variable of XTensorType."""

    # These can't work because Python requires native output types
    def __bool__(self):
        raise TypeError(
            "XTensorVariable cannot be converted to Python boolean. "
            "Call `.astype(bool)` for the symbolic equivalent."
        )

    def __index__(self):
        raise TypeError(
            "XTensorVariable cannot be converted to Python integer. "
            "Call `.astype(int)` for the symbolic equivalent."
        )

    def __int__(self):
        raise TypeError(
            "XTensorVariable cannot be converted to Python integer. "
            "Call `.astype(int)` for the symbolic equivalent."
        )

    def __float__(self):
        raise TypeError(
            "XTensorVariables cannot be converted to Python float. "
            "Call `.astype(float)` for the symbolic equivalent."
        )

    def __complex__(self):
        raise TypeError(
            "XTensorVariables cannot be converted to Python complex number. "
            "Call `.astype(complex)` for the symbolic equivalent."
        )

    # Python valid overloads
    def __abs__(self):
        return px.math.abs(self)

    def __neg__(self):
        return px.math.neg(self)

    def __lt__(self, other):
        return px.math.lt(self, other)

    def __le__(self, other):
        return px.math.le(self, other)

    def __gt__(self, other):
        return px.math.gt(self, other)

    def __ge__(self, other):
        return px.math.ge(self, other)

    def __invert__(self):
        return px.math.invert(self)

    def __and__(self, other):
        return px.math.and_(self, other)

    def __or__(self, other):
        return px.math.or_(self, other)

    def __xor__(self, other):
        return px.math.xor(self, other)

    def __rand__(self, other):
        return px.math.and_(other, self)

    def __ror__(self, other):
        return px.math.or_(other, self)

    def __rxor__(self, other):
        return px.math.xor(other, self)

    def __add__(self, other):
        return px.math.add(self, other)

    def __sub__(self, other):
        return px.math.sub(self, other)

    def __mul__(self, other):
        return px.math.mul(self, other)

    def __pow__(self, other):
        return px.math.pow(self, other)

    def __mod__(self, other):
        return px.math.mod(self, other)

    def __divmod__(self, other):
        return px.math.divmod(self, other)

    def __truediv__(self, other):
        return px.math.true_div(self, other)

    def __floordiv__(self, other):
        return px.math.floor_div(self, other)

    def __rtruediv__(self, other):
        return px.math.true_div(other, self)

    def __rfloordiv__(self, other):
        return px.math.floor_div(other, self)

    def __radd__(self, other):
        return px.math.add(other, self)

    def __rsub__(self, other):
        return px.math.sub(other, self)

    def __rmul__(self, other):
        return px.math.mul(other, self)

    def __rmod__(self, other):
        return px.math.mod(other, self)

    def __rdivmod__(self, other):
        return px.math.divmod(other, self)

    def __rpow__(self, other):
        return px.math.pow(other, self)

    def __ceil__(self):
        return px.math.ceil(self)

    def __floor__(self):
        return px.math.floor(self)

    def __trunc__(self):
        return px.math.trunc(self)

    # DataArray-like attributes
    # https://docs.xarray.dev/en/latest/api.html#id1
    @property
    def values(self) -> TensorVariable:
        """Convert to a TensorVariable with the same data."""
        return typing.cast(TensorVariable, px.basic.tensor_from_xtensor(self))

    # Can't provide property data because that's already taken by Constants!
    # data = values

    @property
    def coords(self):
        """Not implemented."""
        raise NotImplementedError("coords not implemented for XTensorVariable")

    @property
    def dims(self) -> tuple[str, ...]:
        """The names of the dimensions of the variable."""
        return self.type.dims

    @property
    def sizes(self) -> dict[str, TensorVariable]:
        """The sizes of the dimensions of the variable."""
        return dict(zip(self.dims, self.shape))

    @property
    def as_numpy(self):
        # No-op, since the underlying data is always a numpy array
        return self

    # ndarray attributes
    # https://docs.xarray.dev/en/latest/api.html#ndarray-attributes
    @property
    def ndim(self) -> int:
        """The number of dimensions of the variable."""
        return self.type.ndim

    @property
    def shape(self) -> tuple[TensorVariable, ...]:
        """The shape of the variable."""
        return tuple(px.basic.tensor_from_xtensor(self).shape)  # type: ignore

    @property
    def size(self) -> TensorVariable:
        """The total number of elements in the variable."""
        return typing.cast(TensorVariable, variadic_mul(*self.shape))

    @property
    def dtype(self) -> str:
        """The data type of the variable."""
        return self.type.dtype

    @property
    def broadcastable(self):
        # The concept of broadcastable is not revelant for XTensorVariables, but part of the codebase may request it
        return self.type.broadcastable

    # DataArray contents
    # https://docs.xarray.dev/en/latest/api.html#dataarray-contents
    def rename(self, new_name_or_name_dict=None, **names):
        """Rename the variable or its dimension(s)."""
        if isinstance(new_name_or_name_dict, str):
            new_name = new_name_or_name_dict
            name_dict = None
        else:
            new_name = None
            name_dict = new_name_or_name_dict
        new_out = px.basic.rename(self, name_dict, **names)
        new_out.name = new_name
        return new_out

    def copy(self, name: str | None = None):
        """Create a copy of the variable.

        This is just an identity operation, as XTensorVariables are immutable.
        """
        out = px.math.identity(self)
        out.name = name
        return out

    def astype(self, dtype):
        """Convert the variable to a different data type."""
        return px.math.cast(self, dtype)

    def item(self):
        """Not implemented."""
        raise NotImplementedError("item not implemented for XTensorVariable")

    # Indexing
    # https://docs.xarray.dev/en/latest/api.html#id2
    def __setitem__(self, idx, value):
        """Not implemented. Use `x[idx].set(value)` or `x[idx].inc(value)` instead."""
        raise TypeError(
            "XTensorVariable does not support item assignment. Use the output of `x[idx].set` or `x[idx].inc` instead."
        )

    @property
    def loc(self):
        """Not implemented."""
        raise NotImplementedError("loc not implemented for XTensorVariable")

    def sel(self, *args, **kwargs):
        """Not implemented."""
        raise NotImplementedError("sel not implemented for XTensorVariable")

    def __getitem__(self, idx):
        """Index the variable positionally."""
        if isinstance(idx, dict):
            return self.isel(idx)

        if not isinstance(idx, tuple):
            idx = (idx,)

        return px.indexing.index(self, *idx)

    def isel(
        self,
        indexers: dict[str, Any] | None = None,
        drop: bool = False,  # Unused by PyTensor
        missing_dims: Literal["raise", "warn", "ignore"] = "raise",
        **indexers_kwargs,
    ):
        """Index the variable along the specified dimension(s)."""
        if indexers_kwargs:
            if indexers is not None:
                raise ValueError(
                    "Cannot pass both indexers and indexers_kwargs to isel"
                )
            indexers = indexers_kwargs

        if not indexers:
            # No-op
            return self

        if missing_dims not in {"raise", "warn", "ignore"}:
            raise ValueError(
                f"Unrecognized options {missing_dims} for missing_dims argument"
            )

        # Sort indices and pass them to index
        dims = self.type.dims
        indices = [slice(None)] * self.type.ndim
        for key, idx in indexers.items():
            if idx is Ellipsis:
                # Xarray raises a less informative error, suggesting indices must be integer
                # But slices are also fine
                raise TypeError("Ellipsis (...) is an invalid labeled index")
            try:
                indices[dims.index(key)] = idx
            except IndexError:
                if missing_dims == "raise":
                    raise ValueError(
                        f"Dimension {key} does not exist. Expected one of {dims}"
                    )
                elif missing_dims == "warn":
                    warnings.warn(
                        f"Dimension {key} does not exist. Expected one of {dims}",
                        UserWarning,
                    )

        return px.indexing.index(self, *indices)

    def set(self, value):
        """Return a copy of the variable indexed by self with the indexed values set to y.

        The original variable is not modified.

        Raises
        ------
        ValueError
            If self is not the result of an index operation

        Examples
        --------

        .. testcode::

            import pytensor.xtensor as ptx

            x = ptx.as_xtensor([[0, 0], [0, 0]], dims=("a", "b"))
            idx = ptx.as_xtensor([0, 1], dims=("a",))
            out = x[:, idx].set(1)
            print(out.eval())

        .. testoutput::

            [[1 0]
             [0 1]]


        .. testcode::

            import pytensor.xtensor as ptx

            x = ptx.as_xtensor([[0, 0], [0, 0]], dims=("a", "b"))
            idx = ptx.as_xtensor([0, 1], dims=("a",))
            out = x.isel({"b": idx}).set(-1)
            print(out.eval())

        .. testoutput::

            [[-1  0]
             [ 0 -1]]

        """
        if not (
            self.owner is not None and isinstance(self.owner.op, px.indexing.Index)
        ):
            raise ValueError(
                f"set can only be called on the output of an index (or isel) operation. Self is the result of {self.owner}"
            )

        x, *idxs = self.owner.inputs
        return px.indexing.index_assignment(x, value, *idxs)

    def inc(self, value):
        """Return a copy of the variable indexed by self with the indexed values incremented by value.

        The original variable is not modified.

        Raises
        ------
        ValueError
            If self is not the result of an index operation

        Examples
        --------

        .. testcode::

            import pytensor.xtensor as ptx

            x = ptx.as_xtensor([[1, 1], [1, 1]], dims=("a", "b"))
            idx = ptx.as_xtensor([0, 1], dims=("a",))
            out = x[:, idx].inc(1)
            print(out.eval())

        .. testoutput::

            [[2 1]
             [1 2]]


        .. testcode::

            import pytensor.xtensor as ptx

            x = ptx.as_xtensor([[1, 1], [1, 1]], dims=("a", "b"))
            idx = ptx.as_xtensor([0, 1], dims=("a",))
            out = x.isel({"b": idx}).inc(-1)
            print(out.eval())

        .. testoutput::

            [[0 1]
             [1 0]]

        """
        if not (
            self.owner is not None and isinstance(self.owner.op, px.indexing.Index)
        ):
            raise ValueError(
                f"inc can only be called on the output of an index (or isel) operation. Self is the result of {self.owner}"
            )

        x, *idxs = self.owner.inputs
        return px.indexing.index_increment(x, value, *idxs)

    def _head_tail_or_thin(
        self,
        indexers: dict[str, Any] | int | None,
        indexers_kwargs: dict[str, Any],
        *,
        kind: Literal["head", "tail", "thin"],
    ):
        if indexers_kwargs:
            if indexers is not None:
                raise ValueError(
                    "Cannot pass both indexers and indexers_kwargs to head"
                )
            indexers = indexers_kwargs

        if indexers is None:
            if kind == "thin":
                raise TypeError(
                    "thin() indexers must be either dict-like or a single integer"
                )
            else:
                # Default to 5 for head and tail
                indexers = dict.fromkeys(self.type.dims, 5)

        elif not isinstance(indexers, dict):
            indexers = dict.fromkeys(self.type.dims, indexers)

        if kind == "head":
            indices = {dim: slice(None, value) for dim, value in indexers.items()}
        elif kind == "tail":
            sizes = self.sizes
            # Can't use slice(-value, None), in case value is zero
            indices = {
                dim: slice(sizes[dim] - value, None) for dim, value in indexers.items()
            }
        elif kind == "thin":
            indices = {dim: slice(None, None, value) for dim, value in indexers.items()}
        return self.isel(indices)

    def head(self, indexers: dict[str, Any] | int | None = None, **indexers_kwargs):
        return self._head_tail_or_thin(indexers, indexers_kwargs, kind="head")

    def tail(self, indexers: dict[str, Any] | int | None = None, **indexers_kwargs):
        return self._head_tail_or_thin(indexers, indexers_kwargs, kind="tail")

    def thin(self, indexers: dict[str, Any] | int | None = None, **indexers_kwargs):
        return self._head_tail_or_thin(indexers, indexers_kwargs, kind="thin")

    def squeeze(
        self,
        dim: Sequence[str] | str | None = None,
        drop=None,
        axis: int | Sequence[int] | None = None,
    ):
        """Remove dimensions of size 1.

        Parameters
        ----------
        x : XTensorVariable
            The input tensor
        dim : str or None or iterable of str, optional
            The name(s) of the dimension(s) to remove. If None, all dimensions of size 1
            (known statically) will be removed. Dimensions with unknown static shape will be retained, even if they have size 1 at runtime.
        drop : bool, optional
            If drop=True, drop squeezed coordinates instead of making them scalar.
        axis : int or iterable of int, optional
            The axis(es) to remove. If None, all dimensions of size 1 will be removed.
        Returns
        -------
        XTensorVariable
            A new tensor with the specified dimension(s) removed.
        """
        return px.shape.squeeze(self, dim, drop, axis)

    def expand_dims(
        self,
        dim: str | Sequence[str] | dict[str, int | Sequence] | None = None,
        create_index_for_new_dim: bool = True,
        axis: int | Sequence[int] | None = None,
        **dim_kwargs,
    ):
        """Add one or more new dimensions to the variable.

        Parameters
        ----------
        dim : str | Sequence[str] | dict[str, int | Sequence] | None
            If str or sequence of str, new dimensions with size 1.
            If dict, keys are dimension names and values are either:

            - int: the new size
            - sequence: coordinates (length determines size)
        create_index_for_new_dim : bool, default: True
            Ignored by PyTensor
        axis : int | Sequence[int] | None, default: None
            Not implemented yet. In xarray, specifies where to insert the new dimension(s).
            By default (None), new dimensions are inserted at the beginning (axis=0).
        **dim_kwargs : int | Sequence
            Alternative to `dim` dict. Only used if `dim` is None.

        Returns
        -------
        XTensorVariable
            A tensor with additional dimensions inserted at the front.
        """
        return px.shape.expand_dims(
            self,
            dim,
            create_index_for_new_dim=create_index_for_new_dim,
            axis=axis,
            **dim_kwargs,
        )

    # ndarray methods
    # https://docs.xarray.dev/en/latest/api.html#id7
    def clip(self, min, max):
        """Clip the values of the variable to a specified range."""
        return px.math.clip(self, min, max)

    def conj(self):
        """Return the complex conjugate of the variable."""
        return px.math.conj(self)

    @property
    def imag(self):
        """Return the imaginary part of the variable."""
        return px.math.imag(self)

    @property
    def real(self):
        """Return the real part of the variable."""
        return px.math.real(self)

    @property
    def T(self):
        """Return the full transpose of the variable.

        This is equivalent to calling transpose() with no arguments.
        """
        return self.transpose()

    # Aggregation
    # https://docs.xarray.dev/en/latest/api.html#id6
    def all(self, dim=None):
        """Reduce the variable by applying `all` along some dimension(s)."""
        return px.reduction.all(self, dim)

    def any(self, dim=None):
        """Reduce the variable by applying `any` along some dimension(s)."""
        return px.reduction.any(self, dim)

    def max(self, dim=None):
        """Compute the maximum along the given dimension(s)."""
        return px.reduction.max(self, dim)

    def min(self, dim=None):
        """Compute the minimum along the given dimension(s)."""
        return px.reduction.min(self, dim)

    def mean(self, dim=None):
        """Compute the mean along the given dimension(s)."""
        return px.reduction.mean(self, dim)

    def prod(self, dim=None):
        """Compute the product along the given dimension(s)."""
        return px.reduction.prod(self, dim)

    def sum(self, dim=None):
        """Compute the sum along the given dimension(s)."""
        return px.reduction.sum(self, dim)

    def std(self, dim=None, ddof=0):
        """Compute the standard deviation along the given dimension(s)."""
        return px.reduction.std(self, dim, ddof=ddof)

    def var(self, dim=None, ddof=0):
        """Compute the variance along the given dimension(s)."""
        return px.reduction.var(self, dim, ddof=ddof)

    def cumsum(self, dim=None):
        """Compute the cumulative sum along the given dimension(s)."""
        return px.reduction.cumsum(self, dim)

    def cumprod(self, dim=None):
        """Compute the cumulative product along the given dimension(s)."""
        return px.reduction.cumprod(self, dim)

    def diff(self, dim, n=1):
        """Compute the n-th discrete difference along the given dimension."""
        slice1 = {dim: slice(1, None)}
        slice2 = {dim: slice(None, -1)}
        x = self
        for _ in range(n):
            x = x[slice1] - x[slice2]
        return x

    # Reshaping and reorganizing
    # https://docs.xarray.dev/en/latest/api.html#id8
    def transpose(
        self,
        *dim: str | EllipsisType,
        missing_dims: Literal["raise", "warn", "ignore"] = "raise",
    ):
        """Transpose the dimensions of the variable.

        Parameters
        ----------
        *dim : str | Ellipsis
            Dimensions to transpose. If empty, performs a full transpose.
            Can use ellipsis (...) to represent remaining dimensions.
        missing_dims : {"raise", "warn", "ignore"}, default="raise"
            How to handle dimensions that don't exist in the tensor:

            - "raise": Raise an error if any dimensions don't exist
            - "warn": Warn if any dimensions don't exist
            - "ignore": Silently ignore any dimensions that don't exist

        Returns
        -------
        XTensorVariable
            Transposed tensor with reordered dimensions.

        Raises
        ------
        ValueError
            If missing_dims="raise" and any dimensions don't exist.
            If multiple ellipsis are provided.
        """
        return px.shape.transpose(self, *dim, missing_dims=missing_dims)

    def stack(self, dim, **dims):
        """Stack existing dimensions into a single new dimension."""
        return px.shape.stack(self, dim, **dims)

    def unstack(self, dim, **dims):
        """Unstack a dimension into multiple dimensions of a given size.

        Because XTensorVariables don't have coords, this operation requires the sizes of each unstacked dimension to be specified.
        Also, unstacked dims will follow a C-style order, regardless of the order of the original dimensions.

        .. testcode::

            import pytensor.xtensor as ptx

            x = ptx.as_xtensor([[1, 2], [3, 4]], dims=("a", "b"))
            stacked_cumsum = x.stack({"c": ["a", "b"]}).cumsum("c")
            unstacked_cumsum = stacked_cumsum.unstack({"c": x.sizes})
            print(unstacked_cumsum.eval())

        .. testoutput::

            [[ 1  3]
             [ 6 10]]

        """
        return px.shape.unstack(self, dim, **dims)

    def dot(self, other, dim=None):
        """Generalized dot product with another XTensorVariable."""
        return px.math.dot(self, other, dim=dim)

    def broadcast_like(self, other, exclude=None):
        """Broadcast against another XTensorVariable."""
        _, self_bcast = px.shape.broadcast(other, self, exclude=exclude)
        return self_bcast


class XTensorConstantSignature(TensorConstantSignature):
    pass


class XTensorConstant(XTensorVariable, Constant[_XTensorTypeType]):
    """Constant of XtensorType."""

    def __init__(self, type: _XTensorTypeType, data, name=None):
        data_shape = np.shape(data)

        if len(data_shape) != type.ndim or any(
            ds != ts for ds, ts in zip(np.shape(data), type.shape) if ts is not None
        ):
            raise ValueError(
                f"Shape of data ({data_shape}) does not match shape of type ({type.shape})"
            )

        # We want all the shape information from `data`
        if any(s is None for s in type.shape):
            type = type.clone(shape=data_shape)

        Constant.__init__(self, type, data, name)

    def signature(self):
        return XTensorConstantSignature((self.type, self.data))


XTensorType.variable_type = XTensorVariable  # type: ignore
XTensorType.constant_type = XTensorConstant  # type: ignore


def xtensor_constant(x, name=None, dims: None | Sequence[str] = None):
    """Convert a constant value to an XTensorConstant."""

    x_dims: tuple[str, ...]
    if XARRAY_AVAILABLE and isinstance(x, xr.DataArray):
        xarray_dims = x.dims
        if not all(isinstance(dim, str) for dim in xarray_dims):
            raise NotImplementedError(
                "DataArray can only be converted to xtensor_constant if all dims are of string type"
            )
        x_dims = tuple(typing.cast(typing.Iterable[str], xarray_dims))
        x_data = x.values

        if dims is not None and dims != x_dims:
            raise ValueError(
                f"xr.DataArray dims {x_dims} don't match requested specified {dims}. "
                "Use transpose or rename"
            )
    else:
        x_data = tensor_constant(x).data
        if dims is not None:
            x_dims = tuple(dims)
        else:
            if x_data.ndim == 0:
                x_dims = ()
            else:
                raise TypeError(
                    "Cannot convert TensorLike constant to XTensorConstant without specifying dims."
                )
    try:
        return XTensorConstant(
            XTensorType(dtype=x_data.dtype, dims=x_dims, shape=x_data.shape),
            x_data,
            name=name,
        )
    except TypeError:
        raise TypeError(f"Could not convert {x} to XTensorType")


if XARRAY_AVAILABLE:

    @_as_symbolic.register(xr.DataArray)
    def as_symbolic_xarray(x, **kwargs):
        return xtensor_constant(x, **kwargs)


def as_xtensor(x, dims: Sequence[str] | None = None, *, name: str | None = None):
    """Convert a variable or data to an XTensorVariable.

    Parameters
    ----------
    x : Variable or data
    dims: Sequence[str] or None, optional
        If dims are provided, TensorVariable (or data) will be converted to an XTensorVariable with those dims.
        XTensorVariables will be returned as is, if the dims match. Otherwise, a ValueError is raised.
        If dims are not provided, and the data is not a scalar, an XTensorVariable or xarray.DataArray, an error is raised.
    name: str or None, optional
        Name of the resulting XTensorVariable.
    """

    if isinstance(x, Apply):
        if len(x.outputs) != 1:
            raise ValueError(
                "It is ambiguous which output of a multi-output Op has to be fetched.",
                x,
            )
        else:
            x = x.outputs[0]

    if isinstance(x, Variable):
        if isinstance(x.type, XTensorType):
            if (dims is None) or (x.type.dims == dims):
                return x
            else:
                raise ValueError(
                    f"Variable {x} has dims {x.type.dims}, but requested dims are {dims}."
                )
        if isinstance(x.type, TensorType):
            if dims is None:
                if x.type.ndim == 0:
                    dims = ()
                else:
                    raise TypeError(
                        "non-scalar TensorVariable cannot be converted to XTensorVariable without dims."
                    )
            return px.basic.xtensor_from_tensor(x, dims=dims, name=name)
        else:
            raise TypeError(
                "Variable with type {x.type} cannot be converted to XTensorVariable."
            )
    try:
        return xtensor_constant(x, dims=dims, name=name)
    except TypeError as err:
        raise TypeError(f"Cannot convert {x} to XTensorType {type(x)}") from err


register_view_op_c_code(
    XTensorType,
    # XTensorType is just TensorType under the hood
    *ViewOp.c_code_and_version[TensorType],
)

register_deep_copy_op_c_code(
    XTensorType,
    # XTensorType is just TensorType under the hood
    *DeepCopyOp.c_code_and_version[TensorType],
)


@_as_tensor_variable.register(XTensorVariable)
def _xtensor_as_tensor_variable(
    x: XTensorVariable, *args, allow_xtensor_conversion: bool = False, **kwargs
) -> TensorVariable:
    if not allow_xtensor_conversion:
        raise TypeError(
            "To avoid subtle bugs, PyTensor forbids automatic conversion of XTensorVariable to TensorVariable.\n"
            "You can convert explicitly using `x.values` or pass `allow_xtensor_conversion=True`."
        )
    return as_tensor_variable(x.values, *args, **kwargs)
