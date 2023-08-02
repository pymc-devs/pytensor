from pytensor.graph import node_rewriter
from pytensor.tensor import expand_dims
from pytensor.tensor.elemwise import Elemwise
from pytensor.xtensor.basic import tensor_from_xtensor, XElemwise, xtensor_from_tensor
from pytensor.xtensor.rewriting.utils import register_xcanonicalize


@register_xcanonicalize
@node_rewriter(tracks=[XElemwise])
def xelemwise_to_elemwise(fgraph, node):
    # Convert inputs to TensorVariables and add broadcastable dims
    output_dims = node.outputs[0].type.dims

    tensor_inputs = []
    for inp in node.inputs:
        inp_dims = inp.type.dims
        axis = [i for i, dim in enumerate(output_dims) if dim not in inp_dims]
        tensor_inp = tensor_from_xtensor(inp)
        tensor_inp = expand_dims(tensor_inp, axis)
        tensor_inputs.append(tensor_inp)

    tensor_outs = Elemwise(scalar_op=node.op.scalar_op)(*tensor_inputs, return_list=True)

    # TODO: copy_stack_trace
    new_outs = [xtensor_from_tensor(tensor_out, dims=output_dims) for tensor_out in tensor_outs]
    return new_outs
