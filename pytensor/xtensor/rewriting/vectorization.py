from pytensor.graph import node_rewriter
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.utils import compute_batch_shape
from pytensor.xtensor.basic import tensor_from_xtensor, xtensor_from_tensor
from pytensor.xtensor.rewriting.utils import register_lower_xtensor
from pytensor.xtensor.vectorization import XRV, XBlockwise, XElemwise


@register_lower_xtensor
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


@register_lower_xtensor
@node_rewriter(tracks=[XBlockwise])
def lower_blockwise(fgraph, node):
    op: XBlockwise = node.op
    batch_ndim = node.outputs[0].type.ndim - len(op.core_dims[1][0])
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

    signature = op.signature or getattr(op.core_op, "gufunc_signature", None)
    if signature is None:
        # Build a signature based on the core dimensions
        # The Op signature could be more strict, as core_dims will never be repeated, but no functionality depends greatly on it
        inputs_core_dims, outputs_core_dims = op.core_dims
        inputs_signature = ",".join(
            f"({', '.join(inp_core_dims)})" for inp_core_dims in inputs_core_dims
        )
        outputs_signature = ",".join(
            f"({', '.join(out_core_dims)})" for out_core_dims in outputs_core_dims
        )
        signature = f"{inputs_signature}->{outputs_signature}"
    tensor_op = Blockwise(core_op=op.core_op, signature=signature)
    tensor_outs = tensor_op(*tensor_inputs, return_list=True)

    # Convert output Tensors to XTensors
    new_outs = [
        xtensor_from_tensor(tensor_out, dims=old_out.type.dims)
        for (tensor_out, old_out) in zip(tensor_outs, node.outputs, strict=True)
    ]
    return new_outs


@register_lower_xtensor
@node_rewriter(tracks=[XRV])
def lower_rv(fgraph, node):
    op: XRV = node.op
    core_op = op.core_op

    _, old_out = node.outputs
    rng, *extra_dim_lengths_and_params = node.inputs
    extra_dim_lengths = extra_dim_lengths_and_params[: len(op.extra_dims)]
    params = extra_dim_lengths_and_params[len(op.extra_dims) :]

    batch_ndim = old_out.type.ndim - len(op.core_dims[1])
    param_batch_dims = old_out.type.dims[len(op.extra_dims) : batch_ndim]

    # Convert params Tensors to XTensors, align batch dimensions and place core dimension at the end
    tensor_params = []
    for inp, core_dims in zip(params, op.core_dims[0]):
        inp_dims = inp.type.dims
        # Align the batch dims of the input, and place the core dims on the right
        batch_order = [
            inp_dims.index(batch_dim) if batch_dim in inp_dims else "x"
            for batch_dim in param_batch_dims
        ]
        core_order = [inp_dims.index(core_dim) for core_dim in core_dims]
        tensor_inp = tensor_from_xtensor(inp).dimshuffle(batch_order + core_order)
        tensor_params.append(tensor_inp)

    size = None
    if op.extra_dims:
        # RV size contains the lengths of all batch dimensions, including those coming from the parameters
        if tensor_params:
            param_batch_shape = tuple(
                compute_batch_shape(tensor_params, ndims_params=core_op.ndims_params)
            )
        else:
            param_batch_shape = ()
        size = [*extra_dim_lengths, *param_batch_shape]

    # RVs are their own core Op
    new_next_rng, tensor_out = core_op.make_node(rng, size, *tensor_params).outputs

    # Convert output Tensors to XTensors
    new_out = xtensor_from_tensor(tensor_out, dims=old_out.type.dims)
    return [new_next_rng, new_out]
