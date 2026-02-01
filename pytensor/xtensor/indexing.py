# HERE LIE DRAGONS
# Useful links to make sense of all the numpy/xarray complexity
# https://numpy.org/devdocs//user/basics.indexing.html
# https://numpy.org/neps/nep-0021-advanced-indexing.html
# https://docs.xarray.dev/en/latest/user-guide/indexing.html
# https://tutorial.xarray.dev/intermediate/indexing/advanced-indexing.html
from typing import Literal

from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.scalar.basic import discrete_dtypes
from pytensor.tensor.basic import as_tensor
from pytensor.tensor.subtensor import get_slice_elements, index_vars_to_positions
from pytensor.tensor.type_other import NoneTypeT
from pytensor.xtensor.basic import XOp, xtensor_from_tensor
from pytensor.xtensor.type import XTensorType, as_xtensor, xtensor


def as_idx_variable(idx, indexed_dim: str):
    """Convert an index to either a Python slice or a Variable.

    Parameters
    ----------
    idx : slice | Variable | array-like
        The index to convert
    indexed_dim : str
        The dimension being indexed

    Returns
    -------
    slice | Variable
        Either a Python slice object (for slice indexing) or a Variable (for scalar/array indexing)
    """
    if idx is None or (isinstance(idx, Variable) and isinstance(idx.type, NoneTypeT)):
        raise TypeError(
            "XTensors do not support indexing with None (np.newaxis), use expand_dims instead"
        )
    # Python slices pass through directly (will be converted to positions in idx_list)
    if isinstance(idx, slice):
        # Convert slice components to Variables if needed
        start, stop, step = idx.start, idx.stop, idx.step

        def convert_slice_component(comp):
            if comp is None:
                return None
            if isinstance(comp, Variable):
                return comp
            # Convert literals to tensors
            return as_tensor(comp)

        return slice(
            convert_slice_component(start),
            convert_slice_component(stop),
            convert_slice_component(step),
        )
    elif (
        isinstance(idx, tuple)
        and len(idx) == 2
        and (
            isinstance(idx[0], str)
            or (
                isinstance(idx[0], tuple | list)
                and all(isinstance(d, str) for d in idx[0])
            )
        )
    ):
        # Special case for ("x", array) that xarray supports
        dim, idx = idx
        if isinstance(idx, Variable) and isinstance(idx.type, XTensorType):
            raise IndexError(
                f"Giving a dimension name to an XTensorVariable indexer is not supported: {(dim, idx)}. "
                "Use .rename() instead."
            )
        if isinstance(dim, str):
            dims = (dim,)
        else:
            dims = tuple(dim)
        idx = as_xtensor(as_tensor(idx), dims=dims)
    else:
        # Must be integer / boolean indices, we already counted for None and slices
        try:
            idx = as_xtensor(idx)
        except TypeError:
            idx = as_tensor(idx)
            if idx.type.ndim > 1:
                # Same error that xarray raises
                raise IndexError(
                    "Unlabeled multi-dimensional array cannot be used for indexing"
                )
            # This is implicitly an XTensorVariable with dim matching the indexed one
            idx = xtensor_from_tensor(idx, dims=(indexed_dim,)[: idx.type.ndim])

        if idx.type.dtype == "bool":
            if idx.type.ndim != 1:
                # xarray allaws `x[True]`, but I think it is a bug: https://github.com/pydata/xarray/issues/10379
                # Otherwise, it is always restricted to 1d boolean indexing arrays
                raise NotImplementedError(
                    "Only 1d boolean indexing arrays are supported"
                )
            if idx.type.dims != (indexed_dim,):
                raise IndexError(
                    "Boolean indexer should be unlabeled or on the same dimension to the indexed array. "
                    f"Indexer is on {idx.type.dims} but the target dimension is {indexed_dim}."
                )

            # Convert to nonzero indices
            idx = as_xtensor(idx.values.nonzero()[0], dims=idx.type.dims)

        elif idx.type.dtype not in discrete_dtypes:
            raise TypeError("Numerical indices must be integers or boolean")
    return idx


def xtensor_index_vars_to_positions(entry, counter):
    """Convert Variables to positions for xtensor indexing.

    This is a wrapper around tensor.subtensor.index_vars_to_positions that
    handles XTensorVariable by extracting the underlying TensorVariable.

    Parameters
    ----------
    entry : slice | Variable
        An index entry - either a Python slice or a Variable
    counter : list[int]
        Mutable counter for position tracking

    Returns
    -------
    slice | int
        Slice with position integers for Variables, or position integer
    """
    # Convert XTensorVariable to TensorVariable for processing
    if isinstance(entry, Variable) and isinstance(entry.type, XTensorType):
        # Extract the underlying tensor
        entry = entry.values
    elif isinstance(entry, slice):
        # Process slice components
        start, stop, step = entry.start, entry.stop, entry.step

        def convert_component(comp):
            if comp is None:
                return None
            if isinstance(comp, Variable) and isinstance(comp.type, XTensorType):
                return comp.values
            return comp

        entry = slice(
            convert_component(start), convert_component(stop), convert_component(step)
        )

    # Now use the standard function (which handles TensorVariable)
    return index_vars_to_positions(entry, counter, allow_advanced=True)


def get_static_slice_length(slc: slice, dim_length: None | int) -> int | None:
    """Get the static length of a slice if possible.

    Parameters
    ----------
    slc : slice
        Python slice object with Variable or None components
    dim_length : None | int
        The length of the dimension being sliced

    Returns
    -------
    int | None
        The static length of the slice if it can be determined, otherwise None
    """
    if dim_length is None:
        return None

    # Extract slice components
    start, stop, step = slc.start, slc.stop, slc.step

    # Try to extract constants from Variables
    def get_const_value(x):
        if x is None:
            return None
        if isinstance(x, Constant):
            return x.data
        # If it's not a constant, we can't determine static length
        return ...  # Sentinel for non-constant

    start_val = get_const_value(start)
    stop_val = get_const_value(stop)
    step_val = get_const_value(step)

    # If any component is non-constant (represented by ...), can't determine length
    if start_val is ... or stop_val is ... or step_val is ...:
        return None

    return len(range(*slice(start_val, stop_val, step_val).indices(dim_length)))


class Index(XOp):
    __props__ = ("idx_list",)

    def __init__(self, idx_list):
        """Initialize Index with index list.

        Parameters
        ----------
        idx_list : tuple
            Tuple of indices where slices are stored with Variable/None components,
            and scalar/array indices are Variables. This will be converted to positions.
        """
        counter = [0]
        self.idx_list = tuple(
            xtensor_index_vars_to_positions(entry, counter) for entry in idx_list
        )

    def __hash__(self):
        """Hash using idx_list. Slices are not hashable in Python < 3.12."""
        return hash((type(self), self._hashable_idx_list()))

    def _hashable_idx_list(self):
        """Return a hashable version of idx_list (slices converted to tuples)."""
        return tuple(
            (slice, entry.start, entry.stop, entry.step)
            if isinstance(entry, slice)
            else entry
            for entry in self.idx_list
        )

    def make_node(self, x, *inputs):
        """This should not be called directly. Use the index() factory function instead."""
        raise NotImplementedError(
            "Index.make_node should not be called directly. Use index(x, *idxs) instead."
        )


def index(x, *idxs):
    """Create an indexed xtensor (subtensor).

    Parameters
    ----------
    x : XTensorVariable
        The xtensor to index
    *idxs : slice | Variable | array-like
        The indices to apply

    Returns
    -------
    XTensorVariable
        The indexed xtensor
    """
    x = as_xtensor(x)

    # Handle Ellipsis
    if any(idx is Ellipsis for idx in idxs):
        if idxs.count(Ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        # Convert intermediate Ellipsis to slice(None)
        ellipsis_loc = idxs.index(Ellipsis)
        n_implied_none_slices = x.type.ndim - (len(idxs) - 1)
        idxs = (
            *idxs[:ellipsis_loc],
            *((slice(None),) * n_implied_none_slices),
            *idxs[ellipsis_loc + 1 :],
        )

    x_ndim = x.type.ndim
    x_dims = x.type.dims
    x_shape = x.type.shape
    out_dims = []
    out_shape = []

    def combine_dim_info(idx_dim, idx_dim_shape):
        if idx_dim not in out_dims:
            # First information about the dimension length
            out_dims.append(idx_dim)
            out_shape.append(idx_dim_shape)
        else:
            out_dim_pos = out_dims.index(idx_dim)
            out_dim_shape = out_shape[out_dim_pos]
            if out_dim_shape is None:
                out_shape[out_dim_pos] = idx_dim_shape
            elif idx_dim_shape is not None and idx_dim_shape != out_dim_shape:
                raise IndexError(f"Dimension of indexers mismatch for dim {idx_dim}")

    if len(idxs) > x_ndim:
        raise IndexError("Too many indices")

    processed_idxs = [
        as_idx_variable(idx, dim) for idx, dim in zip(idxs, x_dims, strict=False)
    ]

    for i, idx in enumerate(processed_idxs):
        if isinstance(idx, slice):
            idx_dim = x_dims[i]
            idx_dim_shape = get_static_slice_length(idx, x_shape[i])
            combine_dim_info(idx_dim, idx_dim_shape)
        else:
            if idx.type.ndim == 0:
                # Scalar index, dimension is dropped
                continue

            assert isinstance(idx.type, XTensorType)

            idx_dims = idx.type.dims
            for idx_dim in idx_dims:
                idx_dim_shape = idx.type.shape[idx_dims.index(idx_dim)]
                combine_dim_info(idx_dim, idx_dim_shape)

    for dim_i, shape_i in zip(x_dims[i + 1 :], x_shape[i + 1 :]):
        # Add back any unindexed dimensions
        if dim_i not in out_dims:
            # If the dimension was not indexed, we keep it as is
            combine_dim_info(dim_i, shape_i)

    op = Index(processed_idxs)
    inputs = get_slice_elements(
        processed_idxs, lambda entry: isinstance(entry, Variable)
    )
    output = xtensor(dtype=x.type.dtype, shape=out_shape, dims=out_dims)

    return Apply(op, [x, *inputs], [output]).outputs[0]


class IndexUpdate(XOp):
    __props__ = ("mode", "idx_list")

    def __init__(self, mode: Literal["set", "inc"], idx_list):
        if mode not in ("set", "inc"):
            raise ValueError("mode must be 'set' or 'inc'")
        self.mode = mode
        self.idx_list = idx_list

    def __hash__(self):
        """Hash using mode and idx_list. Slices are not hashable in Python < 3.12."""
        return hash((type(self), self.mode, self._hashable_idx_list()))

    def _hashable_idx_list(self):
        """Return a hashable version of idx_list (slices converted to tuples)."""
        return tuple(
            (slice, entry.start, entry.stop, entry.step)
            if isinstance(entry, slice)
            else entry
            for entry in self.idx_list
        )

    def make_node(self, x, y, x_view, *index_inputs):
        try:
            y = as_xtensor(y)
        except TypeError:
            y = as_xtensor(as_tensor(y), dims=x_view.type.dims)

        if not set(y.type.dims).issubset(x_view.type.dims):
            raise ValueError(
                f"Value dimensions {y.type.dims} must be a subset of the indexed dimensions {x_view.type.dims}"
            )

        out = x.type()
        return Apply(self, [x, y, *index_inputs], [out])


def _advanced_update_index(x, y, *idxs, mode):
    x_indexed = index(x, *idxs)
    index_op = x_indexed.owner.op
    assert isinstance(index_op, Index)

    x_orig, *index_variables = x_indexed.owner.inputs
    op = IndexUpdate(mode, index_op.idx_list)
    return op.make_node(x_orig, y, x_indexed, *index_variables).outputs[0]


def advanced_inc_index(x, y, *idxs):
    return _advanced_update_index(x, y, *idxs, mode="inc")


def advanced_set_index(x, y, *idxs):
    return _advanced_update_index(x, y, *idxs, mode="set")
