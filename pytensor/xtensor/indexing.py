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
from pytensor.tensor.type_other import NoneTypeT, SliceType, make_slice
from pytensor.xtensor.basic import XOp, xtensor_from_tensor
from pytensor.xtensor.type import XTensorType, as_xtensor, xtensor


def as_idx_variable(idx, indexed_dim: str):
    if idx is None or (isinstance(idx, Variable) and isinstance(idx.type, NoneTypeT)):
        raise TypeError(
            "XTensors do not support indexing with None (np.newaxis), use expand_dims instead"
        )
    if isinstance(idx, slice):
        idx = make_slice(idx)
    elif isinstance(idx, Variable) and isinstance(idx.type, SliceType):
        pass
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


def get_static_slice_length(slc: Variable, dim_length: None | int) -> int | None:
    if dim_length is None:
        return None
    if isinstance(slc, Constant):
        d = slc.data
        start, stop, step = d.start, d.stop, d.step
    elif slc.owner is None:
        # It's a root variable no way of knowing what we're getting
        return None
    else:
        # It's a MakeSliceOp
        start, stop, step = slc.owner.inputs
        if isinstance(start, Constant):
            start = start.data
        else:
            return None
        if isinstance(stop, Constant):
            stop = stop.data
        else:
            return None
        if isinstance(step, Constant):
            step = step.data
        else:
            return None
    return len(range(*slice(start, stop, step).indices(dim_length)))


class Index(XOp):
    __props__ = ()

    def make_node(self, x, *idxs):
        x = as_xtensor(x)

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
                # Dim already introduced in output by a previous index
                # Update static shape or raise if incompatible
                out_dim_pos = out_dims.index(idx_dim)
                out_dim_shape = out_shape[out_dim_pos]
                if out_dim_shape is None:
                    # We don't know the size of the dimension yet
                    out_shape[out_dim_pos] = idx_dim_shape
                elif idx_dim_shape is not None and idx_dim_shape != out_dim_shape:
                    raise IndexError(
                        f"Dimension of indexers mismatch for dim {idx_dim}"
                    )

        if len(idxs) > x_ndim:
            raise IndexError("Too many indices")

        idxs = [
            as_idx_variable(idx, dim) for idx, dim in zip(idxs, x_dims, strict=False)
        ]

        for i, idx in enumerate(idxs):
            if isinstance(idx.type, SliceType):
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

        output = xtensor(dtype=x.type.dtype, shape=out_shape, dims=out_dims)
        return Apply(self, [x, *idxs], [output])


index = Index()


class IndexUpdate(XOp):
    __props__ = ("mode",)

    def __init__(self, mode: Literal["set", "inc"]):
        if mode not in ("set", "inc"):
            raise ValueError("mode must be 'set' or 'inc'")
        self.mode = mode

    def make_node(self, x, y, *idxs):
        # Call Index on (x, *idxs) to process inputs and infer output type
        x_view_node = index.make_node(x, *idxs)
        x, *idxs = x_view_node.inputs
        [x_view] = x_view_node.outputs

        try:
            y = as_xtensor(y)
        except TypeError:
            y = as_xtensor(as_tensor(y), dims=x_view.type.dims)

        if not set(y.type.dims).issubset(x_view.type.dims):
            raise ValueError(
                f"Value dimensions {y.type.dims} must be a subset of the indexed dimensions {x_view.type.dims}"
            )

        out = x.type()
        return Apply(self, [x, y, *idxs], [out])


index_assignment = IndexUpdate("set")
index_increment = IndexUpdate("inc")
