from pytensor.graph import node_rewriter
from pytensor.tensor.basic import register_infer_shape
from pytensor.tensor.rewriting.basic import register_canonicalize, register_useless
from pytensor.xtensor.basic import (
    Rename,
    TensorFromXTensor,
    XTensorFromTensor,
    xtensor_from_tensor,
)
from pytensor.xtensor.rewriting.utils import register_lower_xtensor


@register_infer_shape
@register_useless
@register_canonicalize
@register_lower_xtensor
@node_rewriter(tracks=[TensorFromXTensor])
def useless_tensor_from_xtensor(fgraph, node):
    """TensorFromXTensor(XTensorFromTensor(x)) -> x"""
    [x] = node.inputs
    if x.owner and isinstance(x.owner.op, XTensorFromTensor):
        return [x.owner.inputs[0]]


@register_infer_shape
@register_useless
@register_canonicalize
@register_lower_xtensor
@node_rewriter(tracks=[XTensorFromTensor])
def useless_xtensor_from_tensor(fgraph, node):
    """XTensorFromTensor(TensorFromXTensor(x)) -> x"""
    [x] = node.inputs
    if x.owner and isinstance(x.owner.op, TensorFromXTensor):
        return [x.owner.inputs[0]]


@register_lower_xtensor
@node_rewriter(tracks=[TensorFromXTensor])
def useless_tensor_from_xtensor_of_rename(fgraph, node):
    """TensorFromXTensor(Rename(x)) -> TensorFromXTensor(x)"""
    [renamed_x] = node.inputs
    if renamed_x.owner and isinstance(renamed_x.owner.op, Rename):
        [x] = renamed_x.owner.inputs
        return node.op(x, return_list=True)


@register_lower_xtensor
@node_rewriter(tracks=[Rename])
def useless_rename(fgraph, node):
    """

    Rename(Rename(x, inner_dims), outer_dims) -> Rename(x, outer_dims)
    Rename(X, XTensorFromTensor(x, inner_dims), outer_dims) -> XTensorFrom_tensor(x, outer_dims)
    """
    [renamed_x] = node.inputs
    if renamed_x.owner:
        if isinstance(renamed_x.owner.op, Rename):
            [x] = renamed_x.owner.inputs
            return [node.op(x)]
        elif isinstance(renamed_x.owner.op, TensorFromXTensor):
            [x] = renamed_x.owner.inputs
            return [xtensor_from_tensor(x, dims=node.op.new_dims)]
