from pytensor.gradient import DisconnectedType, pullback
from pytensor.graph import node_rewriter
from pytensor.graph.basic import clone_get_equiv
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.graph.traversal import ancestors, graph_inputs
from pytensor.tensor.basic import register_infer_shape
from pytensor.tensor.rewriting.basic import register_canonicalize, register_useless
from pytensor.xtensor.basic import (
    LazyGrad,
    Rename,
    TensorFromXTensor,
    XOp,
    XTensorFromTensor,
    grad_connected,
    xtensor_from_tensor,
)
from pytensor.xtensor.random.type import RNGToXRNG, XRNGToRNG
from pytensor.xtensor.rewriting.utils import (
    register_lower_lazy_grad,
    register_lower_xtensor,
)
from pytensor.xtensor.shape import zeros_like


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


@register_infer_shape
@register_useless
@register_canonicalize
@register_lower_xtensor
@node_rewriter(tracks=[RNGToXRNG])
def useless_rng_to_xrng(fgraph, node):
    """RNGToXRNG(XRNGToRNG(x)) -> x"""
    [x] = node.inputs
    if x.owner and isinstance(x.owner.op, XRNGToRNG):
        return [x.owner.inputs[0]]


@register_infer_shape
@register_useless
@register_canonicalize
@register_lower_xtensor
@node_rewriter(tracks=[XRNGToRNG])
def useless_xrng_to_rng(fgraph, node):
    """XRNGToRNG(RNGToXRNG(x)) -> x"""
    [x] = node.inputs
    if x.owner and isinstance(x.owner.op, RNGToXRNG):
        return [x.owner.inputs[0]]


@register_lower_lazy_grad
@node_rewriter(tracks=[LazyGrad])
def expand_lazy_grad(fgraph, node):
    """Differentiate an XOp by lowering it to tensor ops and taking their pullback.

    Runs before lower_xtensor: the differentiated op (``core_op``) is rebuilt on fresh
    stand-ins and lowered to tensor ops in isolation, then differentiated with the
    ordinary tensor pullback. Stand-ins (rather than the real inputs) give a repeated
    input separate per-slot cotangents, and survive the lowering of the conversion ops
    that the real inputs would be folded into.
    """
    op = node.op
    forward_inputs = node.inputs[: -op.n_cotangents]
    cotangents = node.inputs[-op.n_cotangents :]

    dummies = [inp.type() if grad_connected(inp) else inp for inp in forward_inputs]
    lowered = rewrite_graph(
        list(op.core_op.make_node(*dummies).outputs),
        include=("lower_lazy_grad", "lower_xtensor"),
    )
    if any(isinstance(var.owner.op, XOp) for var in ancestors(lowered) if var.owner):
        raise NotImplementedError(f"pullback not implemented for {op.core_op}")

    memo = {d: inp for d, inp in zip(dummies, forward_inputs) if grad_connected(inp)}
    input_grads = pullback(
        lowered,
        list(memo),
        cotangents,
        disconnected_inputs="ignore",
        return_disconnected="disconnected",
    )
    # The lowering and pullback above built nodes inside throwaway FunctionGraphs. Re-clone
    # the grad into fresh nodes so it imports into the main graph through the normal path,
    # grafting the real inputs back in place of the stand-ins. Real variables the grad
    # already shares (the node inputs and any value the gradient reuses) are kept as-is.
    keep = list(node.inputs) + [
        v
        for v in graph_inputs(input_grads, blockers=node.inputs)
        if v not in memo and v not in set(node.inputs)
    ]
    equiv = clone_get_equiv(keep, input_grads, copy_inputs=False, memo=dict(memo))
    # An input the cost doesn't reach through this node contributes a zero (its other
    # paths are summed in by the grad engine); a node output can't be DisconnectedType.
    return [
        zeros_like(inp) if isinstance(grad.type, DisconnectedType) else equiv[grad]
        for grad, inp in zip(input_grads, memo.values())
    ]
