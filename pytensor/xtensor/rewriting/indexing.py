from itertools import zip_longest

from pytensor import as_symbolic
from pytensor.graph import Constant, node_rewriter
from pytensor.tensor import arange, specify_shape
from pytensor.tensor.type_other import NoneTypeT, SliceType
from pytensor.xtensor.basic import tensor_from_xtensor, xtensor_from_tensor
from pytensor.xtensor.indexing import Index
from pytensor.xtensor.rewriting.utils import register_xcanonicalize
from pytensor.xtensor.type import XTensorType


def to_basic_idx(idx):
    if isinstance(idx.type, SliceType):
        if isinstance(idx, Constant):
            return idx.data
        elif idx.owner:
            # MakeSlice Op
            # We transform NoneConsts to regular None so that basic Subtensor can be used if possible
            return slice(
                *[
                    None if isinstance(i.type, NoneTypeT) else i
                    for i in idx.owner.inputs
                ]
            )
        else:
            return idx
    if (
        isinstance(idx.type, XTensorType)
        and idx.type.ndim == 0
        and idx.type.dtype != bool
    ):
        return idx.values
    raise TypeError("Cannot convert idx to basic idx")


@register_xcanonicalize
@node_rewriter(tracks=[Index])
def lower_index(fgraph, node):
    x, *idxs = node.inputs
    [out] = node.outputs
    x_tensor = tensor_from_xtensor(x)

    if all(
        (
            isinstance(idx.type, SliceType)
            or (isinstance(idx.type, XTensorType) and idx.type.ndim == 0)
        )
        for idx in idxs
    ):
        # Special case just basic indexing
        x_tensor_indexed = x_tensor[tuple(to_basic_idx(idx) for idx in idxs)]

    else:
        # General case, we have to align the indices positionally to achieve vectorized or orthogonal indexing
        # May need to convert basic indexing to advanced indexing if it acts on a dimension
        # that is also indexed by an advanced index
        x_dims = x.type.dims
        x_shape = tuple(x.shape)
        out_ndim = out.type.ndim
        out_xdims = out.type.dims
        aligned_idxs = []
        # zip_longest adds the implicit slice(None)
        for i, (idx, x_dim) in enumerate(
            zip_longest(idxs, x_dims, fillvalue=as_symbolic(slice(None)))
        ):
            if isinstance(idx.type, SliceType):
                if not any(
                    (
                        isinstance(other_idx.type, XTensorType)
                        and x_dim in other_idx.dims
                    )
                    for j, other_idx in enumerate(idxs)
                    if j != i
                ):
                    # We can use basic indexing directly if no other index acts on this dimension
                    aligned_idxs.append(idx)
                else:
                    # Otherwise we need to convert the basic index into an equivalent advanced indexing
                    # And align it so it interacts correctly with the other advanced indices
                    adv_idx_equivalent = arange(x_shape[i])[idx]
                    ds_order = ["x"] * out_ndim
                    ds_order[out_xdims.index(x_dim)] = 0
                    aligned_idxs.append(adv_idx_equivalent.dimshuffle(ds_order))
            else:
                assert isinstance(idx.type, XTensorType)
                if idx.type.ndim == 0:
                    # Scalar index, we can use it directly
                    aligned_idxs.append(idx.values)
                else:
                    # Vector index, we need to align the indexing dimensions with the base_dims
                    ds_order = ["x"] * out_ndim
                    for j, idx_dim in enumerate(idx.dims):
                        ds_order[out_xdims.index(idx_dim)] = j
                    aligned_idxs.append(idx.values.dimshuffle(ds_order))
        x_tensor_indexed = x_tensor[tuple(aligned_idxs)]
        # TODO: Align output dimensions if necessary

    # Add lost shape if any
    x_tensor_indexed = specify_shape(x_tensor_indexed, out.type.shape)
    new_out = xtensor_from_tensor(x_tensor_indexed, dims=out.type.dims)
    return [new_out]
