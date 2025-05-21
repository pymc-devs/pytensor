from pytensor.graph import Constant, node_rewriter
from pytensor.tensor import TensorType, specify_shape
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
        isinstance(idx.type, XTensorType | TensorType)
        and idx.type.ndim == 0
        and idx.type.dtype != bool
    ):
        return idx
    raise TypeError("Cannot convert idx to basic idx")


def _count_idx_types(idxs):
    basic, vector, xvector = 0, 0, 0
    for idx in idxs:
        if isinstance(idx.type, SliceType):
            basic += 1
        elif idx.type.ndim == 0:
            basic += 1
        elif isinstance(idx.type, TensorType):
            vector += 1
        else:
            xvector += 1
    return basic, vector, xvector


@register_xcanonicalize
@node_rewriter(tracks=[Index])
def lower_index(fgraph, node):
    x, *idxs = node.inputs
    [out] = node.outputs
    x_tensor = tensor_from_xtensor(x)
    n_basic, n_vector, n_xvector = _count_idx_types(idxs)
    if n_xvector == 0 and n_vector == 0:
        x_tensor_indexed = x_tensor[tuple(to_basic_idx(idx) for idx in idxs)]
    elif n_vector == 1 and n_xvector == 0:
        # Special case for single vector index, no orthogonal indexing
        x_tensor_indexed = x_tensor[tuple(idxs)]
    else:
        # Not yet implemented
        return None

    # Add lost shape if any
    x_tensor_indexed = specify_shape(x_tensor_indexed, out.type.shape)
    new_out = xtensor_from_tensor(x_tensor_indexed, dims=out.type.dims)
    return [new_out]
