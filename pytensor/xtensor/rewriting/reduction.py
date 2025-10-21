from functools import partial

import pytensor.scalar as ps
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.extra_ops import CumOp
from pytensor.tensor.math import All, Any, CAReduce, Max, Min, Prod, Sum
from pytensor.xtensor.basic import tensor_from_xtensor, xtensor_from_tensor
from pytensor.xtensor.reduction import XCumReduce, XReduce
from pytensor.xtensor.rewriting.utils import register_lower_xtensor


@register_lower_xtensor
@node_rewriter(tracks=[XReduce])
def lower_reduce(fgraph, node):
    [x] = node.inputs
    [out] = node.outputs
    x_dims = x.type.dims
    reduce_dims = node.op.dims
    reduce_axis = [x_dims.index(dim) for dim in reduce_dims]

    if not reduce_axis:
        return [x]

    match node.op.binary_op:
        case ps.add:
            tensor_op_class = Sum
        case ps.mul:
            tensor_op_class = Prod
        case ps.and_:
            tensor_op_class = All
        case ps.or_:
            tensor_op_class = Any
        case ps.maximum:
            tensor_op_class = Max
        case ps.minimum:
            tensor_op_class = Min
        case _:
            # Case without known/predefined Ops
            tensor_op_class = partial(CAReduce, scalar_op=node.op.binary_op)

    x_tensor = tensor_from_xtensor(x)
    out_tensor = tensor_op_class(axis=reduce_axis)(x_tensor)
    new_out = xtensor_from_tensor(out_tensor, out.type.dims)
    return [new_out]


@register_lower_xtensor
@node_rewriter(tracks=[XCumReduce])
def lower_cumreduce(fgraph, node):
    [x] = node.inputs
    x_dims = x.type.dims
    reduce_dims = node.op.dims
    reduce_axis = [x_dims.index(dim) for dim in reduce_dims]

    if not reduce_axis:
        return [x]

    match node.op.binary_op:
        case ps.add:
            tensor_op_class = partial(CumOp, mode="add")
        case ps.mul:
            tensor_op_class = partial(CumOp, mode="mul")
        case _:
            # We don't know how to convert an arbitrary binary cum/reduce Op
            return None

    # Each dim corresponds to an application of Cumsum/Cumprod
    out_tensor = tensor_from_xtensor(x)
    for axis in reduce_axis:
        out_tensor = tensor_op_class(axis=axis)(out_tensor)
    out = xtensor_from_tensor(out_tensor, x.type.dims)
    return [out]
