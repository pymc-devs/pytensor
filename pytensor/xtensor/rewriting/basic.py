from pytensor.graph import node_rewriter
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import Elemwise
from pytensor.xtensor.basic import (
    Rename,
    TensorFromXTensor,
    XBlockwise,
    XElemwise,
    XTensorFromTensor,
    tensor_from_xtensor,
    xtensor_from_tensor,
)
from pytensor.xtensor.rewriting.utils import register_xcanonicalize


@register_xcanonicalize
@node_rewriter(tracks=[TensorFromXTensor])
def useless_tensor_from_xtensor(fgraph, node):
    """TensorFromXTensor(XTensorFromTensor(x)) -> x"""
    [x] = node.inputs
    if x.owner and isinstance(x.owner.op, XTensorFromTensor):
        return [x.owner.inputs[0]]


@register_xcanonicalize
@node_rewriter(tracks=[XTensorFromTensor])
def useless_xtensor_from_tensor(fgraph, node):
    """XTensorFromTensor(TensorFromXTensor(x)) -> x"""
    [x] = node.inputs
    if x.owner and isinstance(x.owner.op, TensorFromXTensor):
        return [x.owner.inputs[0]]


@register_xcanonicalize
@node_rewriter(tracks=[TensorFromXTensor])
def useless_tensor_from_xtensor_of_rename(fgraph, node):
    """TensorFromXTensor(Rename(x)) -> TensorFromXTensor(x)"""
    [renamed_x] = node.inputs
    if renamed_x.owner and isinstance(renamed_x.owner.op, Rename):
        [x] = renamed_x.owner.inputs
        return node.op(x, return_list=True)


@register_xcanonicalize
@node_rewriter(tracks=[Rename])
def useless_rename(fgraph, node):
    """

    Rename(Rename(x, inner_dims), outer_dims) -> Rename(x, outer_dims)
    Rename(X, XTensorFromTensor(x, inner_dims), outer_dims) -> XTensorFrom_tensor(x, outer_dims)
    """
    [renamed_x] = node.inputs
    if renamed_x.owner and isinstance(renamed_x.owner.op, Rename | XTensorFromTensor):
        [x] = renamed_x.owner.inputs
        return node.op(x, return_list=True)


@register_xcanonicalize
@node_rewriter(tracks=[XElemwise])
def lower_elemwise(fgraph, node):
    out_dims = node.outputs[0].type.dims

    # Convert input XTensors to Tensors and align batch dimensions
    tensor_inputs = []
    for inp in node.inputs:
        inp_dims = inp.type.dims
        order = [
            inp_dims.index(out_dim) if out_dim in inp_dims else "x"
            for out_dim in out_dims
        ]
        tensor_inp = tensor_from_xtensor(inp).dimshuffle(order)
        tensor_inputs.append(tensor_inp)

    tensor_outs = Elemwise(scalar_op=node.op.scalar_op)(
        *tensor_inputs, return_list=True
    )

    # Convert output Tensors to XTensors
    new_outs = [
        xtensor_from_tensor(tensor_out, dims=out_dims) for tensor_out in tensor_outs
    ]
    return new_outs


@register_xcanonicalize
@node_rewriter(tracks=[XBlockwise])
def lower_blockwise(fgraph, node):
    op: XBlockwise = node.op
    batch_ndim = node.outputs[0].type.ndim - len(op.outputs_sig[0])
    batch_dims = node.outputs[0].type.dims[:batch_ndim]

    # Convert input Tensors to XTensors, align batch dimensions and place core dimension at the end
    tensor_inputs = []
    for inp, core_dims in zip(node.inputs, op.core_dims[0]):
        inp_dims = inp.type.dims
        # Align the batch dims of the input, and place the core dims on the right
        batch_order = [
            inp_dims.index(batch_dim) if batch_dim in inp_dims else "x"
            for batch_dim in batch_dims
        ]
        core_order = [inp_dims.index(core_dim) for core_dim in core_dims]
        tensor_inp = tensor_from_xtensor(inp).dimshuffle(batch_order + core_order)
        tensor_inputs.append(tensor_inp)

    tensor_op = Blockwise(core_op=node.op.core_op, signature=op.signature)
    tensor_outs = tensor_op(*tensor_inputs, return_list=True)

    # Convert output Tensors to XTensors
    new_outs = [
        xtensor_from_tensor(tensor_out, dims=old_out.type.dims)
        for (tensor_out, old_out) in zip(tensor_outs, node.outputs, strict=True)
    ]
    return new_outs
