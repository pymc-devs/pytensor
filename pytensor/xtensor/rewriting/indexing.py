from itertools import zip_longest

from pytensor import as_symbolic
from pytensor.graph import Constant, node_rewriter
from pytensor.tensor import TensorType, arange, specify_shape
from pytensor.tensor.subtensor import _non_consecutive_adv_indexing, inc_subtensor
from pytensor.tensor.type_other import NoneTypeT, SliceType
from pytensor.xtensor.basic import tensor_from_xtensor, xtensor_from_tensor
from pytensor.xtensor.indexing import Index, IndexUpdate, index
from pytensor.xtensor.rewriting.utils import register_lower_xtensor
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


def _lower_index(node):
    """Lower XTensorVariable indexing to regular TensorVariable indexing.

    xarray-like indexing has two modes:
    1. Orthogonal indexing: Indices of different output labeled dimensions are combined to produce all combinations of indices.
    2. Vectorized indexing: Indices of the same output labeled dimension are combined point-wise like in regular numpy advanced indexing.

    An Index Op can combine both modes.
    To achieve orthogonal indexing using numpy semantics we must use multidimensional advanced indexing.
    We expand the dims of each index so they are as large as the number of output dimensions, place the indices that
    belong to the same output dimension in the same axis, and those that belong to different output dimensions in different axes.

    For instance to do an outer 2x2 indexing we can select x[arange(x.shape[0])[:, None], arange(x.shape[1])[None, :]],
    This is a generalization of `np.ix_` that allows combining some dimensions, and not others, as well as have
    indices that have more than one dimension at the start.

    In addition, xarray basic index (slices), can be vectorized with other advanced indices (if they act on the same output dimension).
    However, in numpy, basic indices are always orthogonal to advanced indices. To make them behave like vectorized indices
    we have to convert the slices to equivalent advanced indices.
    We do this by creating an `arange` tensor that matches the shape of the dimension being indexed,
    and then indexing it with the original slice. This index is then handled as a regular advanced index.

    Finally, the location of views resulting from advanced indices follows two distinct behaviors in numpy.
    When all advanced indices are consecutive, the respective view is located in the "original" location.
    However, if advanced indices are separated by basic indices (slices in our case), the output views
    always show up at the front of the array. This information is returned as the second output of this function,
    which labels the final position of the indexed dimensions under this rule.
    """

    assert isinstance(node.op, Index)

    x, *idxs = node.inputs
    [out] = node.outputs
    x_tensor_indexed_dims = out.type.dims
    x_tensor = tensor_from_xtensor(x)

    if all(
        (
            isinstance(idx.type, SliceType)
            or (isinstance(idx.type, XTensorType) and idx.type.ndim == 0)
        )
        for idx in idxs
    ):
        # Special case having just basic indexing
        x_tensor_indexed = x_tensor[tuple(to_basic_idx(idx) for idx in idxs)]

    else:
        # General case, we have to align the indices positionally to achieve vectorized or orthogonal indexing
        # May need to convert basic indexing to advanced indexing if it acts on a dimension that is also indexed by an advanced index
        x_dims = x.type.dims
        x_shape = tuple(x.shape)
        out_ndim = out.type.ndim
        out_dims = out.type.dims
        aligned_idxs = []
        basic_idx_axis = []
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
                    # This is an optimization that avoids creating an unnecessary arange tensor
                    # and facilitates the use of the specialized AdvancedSubtensor1 when possible
                    aligned_idxs.append(idx)
                    basic_idx_axis.append(out_dims.index(x_dim))
                else:
                    # Otherwise we need to convert the basic index into an equivalent advanced indexing
                    # And align it so it interacts correctly with the other advanced indices
                    adv_idx_equivalent = arange(x_shape[i])[to_basic_idx(idx)]
                    ds_order = ["x"] * out_ndim
                    ds_order[out_dims.index(x_dim)] = 0
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
                        ds_order[out_dims.index(idx_dim)] = j
                    aligned_idxs.append(idx.values.dimshuffle(ds_order))

        # Squeeze indexing dimensions that were not used because we kept basic indexing slices
        if basic_idx_axis:
            aligned_idxs = [
                idx.squeeze(axis=basic_idx_axis)
                if (isinstance(idx.type, TensorType) and idx.type.ndim > 0)
                else idx
                for idx in aligned_idxs
            ]

        x_tensor_indexed = x_tensor[tuple(aligned_idxs)]

        if basic_idx_axis and _non_consecutive_adv_indexing(aligned_idxs):
            # Numpy moves advanced indexing dimensions to the front when they are not consecutive
            # We need to transpose them back to the expected output order
            x_tensor_indexed_basic_dims = [out_dims[axis] for axis in basic_idx_axis]
            x_tensor_indexed_dims = [
                dim for dim in out_dims if dim not in x_tensor_indexed_basic_dims
            ] + x_tensor_indexed_basic_dims

    return x_tensor_indexed, x_tensor_indexed_dims


@register_lower_xtensor
@node_rewriter(tracks=[Index])
def lower_index(fgraph, node):
    """Lower XTensorVariable indexing to regular TensorVariable indexing.

    The bulk of the work is done by `_lower_index`, except for special logic to control the
    location of non-consecutive advanced indices, and to preserve static shape information.
    """

    [out] = node.outputs
    out_dims = out.type.dims

    x_tensor_indexed, x_tensor_indexed_dims = _lower_index(node)
    if x_tensor_indexed_dims != out_dims:
        # Numpy moves advanced indexing dimensions to the front when they are not consecutive
        # We need to transpose them back to the expected output order
        transpose_order = [x_tensor_indexed_dims.index(dim) for dim in out_dims]
        x_tensor_indexed = x_tensor_indexed.transpose(transpose_order)

    # Add lost shape information
    x_tensor_indexed = specify_shape(x_tensor_indexed, out.type.shape)

    new_out = xtensor_from_tensor(x_tensor_indexed, dims=out.dims)
    return [new_out]


@register_lower_xtensor
@node_rewriter(tracks=[IndexUpdate])
def lower_index_update(fgraph, node):
    """Lower XTensorVariable index update to regular TensorVariable indexing update.

    This rewrite requires converting the index view to a tensor-based equivalent expression,
    just like `lower_index`. It then requires aligning the dimensions of y with the
    dimensions of the index view, with special care for non-consecutive dimensions being
    pulled to the front axis according to numpy rules.
    """
    x, y, *idxs = node.inputs

    # Lower the indexing part first
    indexed_node = index.make_node(x, *idxs)
    x_tensor_indexed, x_tensor_indexed_dims = _lower_index(indexed_node)
    y_tensor = tensor_from_xtensor(y)

    # Align dimensions of y with those of the indexed tensor x
    y_dims = y.type.dims
    y_dims_set = set(y_dims)
    y_order = tuple(
        y_dims.index(x_dim) if x_dim in y_dims_set else "x"
        for x_dim in x_tensor_indexed_dims
    )
    # Remove useless left expand_dims
    while len(y_order) > 0 and y_order[0] == "x":
        y_order = y_order[1:]
    if y_order != tuple(range(y_tensor.type.ndim)):
        y_tensor = y_tensor.dimshuffle(y_order)

    x_tensor_updated = inc_subtensor(
        x_tensor_indexed, y_tensor, set_instead_of_inc=node.op.mode == "set"
    )
    new_out = xtensor_from_tensor(x_tensor_updated, dims=x.type.dims)
    return [new_out]
