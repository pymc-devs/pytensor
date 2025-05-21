# HERE LIE DRAGONS
# Uselful links to make sense of all the numpy/xarray complexity
# https://numpy.org/devdocs//user/basics.indexing.html
# https://numpy.org/neps/nep-0021-advanced-indexing.html
# https://docs.xarray.dev/en/latest/user-guide/indexing.html
# https://tutorial.xarray.dev/intermediate/indexing/advanced-indexing.html

from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.scalar.basic import discrete_dtypes
from pytensor.tensor.basic import as_tensor
from pytensor.tensor.type_other import NoneTypeT, SliceType, make_slice
from pytensor.xtensor.basic import XOp
from pytensor.xtensor.type import XTensorType, as_xtensor, xtensor


def as_idx_variable(idx):
    if idx is None or (isinstance(idx, Variable) and isinstance(idx.type, NoneTypeT)):
        raise TypeError(
            "XTensors do not support indexing with None (np.newaxis), use expand_dims instead"
        )
    if isinstance(idx, slice):
        idx = make_slice(idx)
    elif isinstance(idx, Variable) and isinstance(idx.type, SliceType):
        pass
    else:
        # Must be integer indices, we already counted for None and slices
        try:
            idx = as_tensor(idx)
        except TypeError:
            idx = as_xtensor(idx)
        if idx.type.dtype == "bool":
            raise NotImplementedError("Boolean indexing not yet supported")
        if idx.type.dtype not in discrete_dtypes:
            raise TypeError("Numerical indices must be integers or boolean")
        if idx.type.dtype == "bool" and idx.type.ndim == 0:
            # This can't be triggered right now, but will once we lift the boolean restriction
            raise NotImplementedError("Scalar boolean indices not supported")
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
        idxs = [as_idx_variable(idx) for idx in idxs]

        x_ndim = x.type.ndim
        x_dims = x.type.dims
        x_shape = x.type.shape
        out_dims = []
        out_shape = []
        has_unlabeled_vector_idx = False
        has_labeled_vector_idx = False
        for i, idx in enumerate(idxs):
            if i == x_ndim:
                raise IndexError("Too many indices")
            if isinstance(idx.type, SliceType):
                out_dims.append(x_dims[i])
                out_shape.append(get_static_slice_length(idx, x_shape[i]))
            elif isinstance(idx.type, XTensorType):
                if has_unlabeled_vector_idx:
                    raise NotImplementedError(
                        "Mixing of labeled and unlabeled vector indexing not implemented"
                    )
                has_labeled_vector_idx = True
                idx_dims = idx.type.dims
                for dim in idx_dims:
                    idx_dim_shape = idx.type.shape[idx_dims.index(dim)]
                    if dim in out_dims:
                        # Dim already introduced in output by a previous index
                        # Update static shape or raise if incompatible
                        out_dim_pos = out_dims.index(dim)
                        out_dim_shape = out_shape[out_dim_pos]
                        if out_dim_shape is None:
                            # We don't know the size of the dimension yet
                            out_shape[out_dim_pos] = idx_dim_shape
                        elif (
                            idx_dim_shape is not None and idx_dim_shape != out_dim_shape
                        ):
                            raise IndexError(
                                f"Dimension of indexers mismatch for dim {dim}"
                            )
                    else:
                        # New dimension
                        out_dims.append(dim)
                        out_shape.append(idx_dim_shape)

            else:  # TensorType
                if idx.type.ndim == 0:
                    # Scalar, dimension is dropped
                    pass
                elif idx.type.ndim == 1:
                    if has_labeled_vector_idx:
                        raise NotImplementedError(
                            "Mixing of labeled and unlabeled vector indexing not implemented"
                        )
                    has_unlabeled_vector_idx = True
                    out_dims.append(x_dims[i])
                    out_shape.append(idx.type.shape[0])
                else:
                    # Same error that xarray raises
                    raise IndexError(
                        "Unlabeled multi-dimensional array cannot be used for indexing"
                    )
        for j in range(i + 1, x_ndim):
            # Add any unindexed dimensions
            out_dims.append(x_dims[j])
            out_shape.append(x_shape[j])

        output = xtensor(dtype=x.type.dtype, shape=out_shape, dims=out_dims)
        return Apply(self, [x, *idxs], [output])


index = Index()
