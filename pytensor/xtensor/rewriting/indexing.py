from pytensor.graph import node_rewriter
from pytensor.tensor import TensorType
from pytensor.tensor.type_other import SliceType
from pytensor.xtensor.basic import tensor_from_xtensor, xtensor_from_tensor
from pytensor.xtensor.indexing import Index
from pytensor.xtensor.rewriting.utils import register_xcanonicalize


def is_basic_idx(idx):
    return (
        isinstance(idx.type, SliceType)
        or isinstance(idx.type, TensorType)
        and idx.type.ndim == 0
        and idx.type.dtype != bool
    )


@register_xcanonicalize
@node_rewriter(tracks=[Index])
def lower_index(fgraph, node):
    x, *idxs = node.inputs
    x_tensor = tensor_from_xtensor(x)
    if all(is_basic_idx(idx) for idx in idxs):
        # Simple case
        x_tensor_indexed = x_tensor[tuple(idxs)]
        new_out = xtensor_from_tensor(x_tensor_indexed, dims=node.outputs[0].type.dims)
        return [new_out]
