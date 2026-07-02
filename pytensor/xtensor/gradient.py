"""Make xtensor graphs differentiable through ``pytensor.grad``.

xtensor Ops carry no gradient of their own: they are meaningful only once lowered to
tensor Ops. Rather than differentiate them node-by-node, this grabs each xtensor region
-- the subgraph between the tensor<->xtensor conversion boundaries -- as a single
``OpFromGraph`` unit whose inner graph is the region lowered to tensor Ops. ``grad``
then differentiates the unit as a whole through the ordinary tensor rules. The lowering
happens here, in a graph pass registered with ``grad``, never inside an Op's pullback.
"""

from pytensor.compile.builders import OpFromGraph
from pytensor.gradient import register_grad_graph_rewriter
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.replace import graph_replace
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.graph.traversal import ancestors, toposort
from pytensor.tensor.type_other import SliceType
from pytensor.xtensor.basic import (
    TensorFromXTensor,
    tensor_from_xtensor,
    xtensor_from_tensor,
)
from pytensor.xtensor.random.type import (
    XRandomGeneratorType,
    XRNGToRNG,
    rng_to_xrng,
    xrng_to_rng,
)
from pytensor.xtensor.type import XTensorType


_XTENSOR_TYPES = (XTensorType, XRandomGeneratorType)
# Ops that carry an xtensor value back into the tensor world (the region's outputs).
_EXIT_OPS = (TensorFromXTensor, XRNGToRNG)


def _is_xtensor(var: Variable) -> bool:
    return isinstance(var.type, _XTENSOR_TYPES)


def _to_tensor_world(var: Variable) -> Variable:
    """Convert an xtensor-world variable to its tensor-world equivalent."""
    if isinstance(var.type, XTensorType):
        return tensor_from_xtensor(var)
    if isinstance(var.type, XRandomGeneratorType):
        return xrng_to_rng(var)
    return var


def _from_tensor_world(var: Variable, dummy: Variable) -> Variable:
    """Rebuild the xtensor-world equivalent of ``var`` from a tensor-world dummy."""
    if isinstance(var.type, XTensorType):
        return xtensor_from_tensor(dummy, dims=var.type.dims)
    if isinstance(var.type, XRandomGeneratorType):
        return rng_to_xrng(dummy)
    return dummy


def _collapse_region(
    exit_var: Variable, boundaries: set[Variable], memo: dict[Variable, Variable]
) -> Variable:
    """Build the lowered OpFromGraph unit replacing the xtensor region at ``exit_var``.

    The unit's inner graph is built from the original region; its outer inputs are
    resolved through ``memo`` so inner regions already collapsed in this pass are
    consumed through their fresh replacements.
    """
    # Walk the xtensor cone of exit_var up to its boundaries. Non-xtensor inputs (tensors
    # entering via XTensorFromTensor, indices, rngs) and any boundary/leaf become unit
    # inputs; constants stay inside the unit. `boundaries` are the wrt and consider_constant
    # variables, kept as inputs so grad still sees (and can stop at) them after collapse.
    inputs: list[Variable] = []
    seen: set[Variable] = set()

    def visit(var: Variable) -> None:
        # Keep var inside the unit and recurse through its node's inputs.
        if var in seen:
            return
        seen.add(var)
        if var in boundaries or var.owner is None:
            if not isinstance(var, Constant):
                inputs.append(var)
            return
        for inp in var.owner.inputs:
            visit_input(inp)

    def visit_input(inp: Variable) -> None:
        if inp not in boundaries and (
            _is_xtensor(inp)
            # A computed slice (e.g. MakeSlice) must also stay inside the unit: passed
            # in as an opaque SliceType input the lowering could not pattern-match it.
            # Its symbolic components become unit inputs instead; constants stay inside.
            or (isinstance(inp.type, SliceType) and inp.owner is not None)
        ):
            visit(inp)
        elif not isinstance(inp, Constant):
            inputs.append(inp)

    for inp in exit_var.owner.inputs:
        visit_input(inp)
    inputs = list(dict.fromkeys(inputs))

    # Keep the unit's boundaries tensor-typed: convert any xtensor-world input outside
    # the unit and rebuild it inside. This keeps the unit a plain tensor->tensor op that
    # lowers fully, so a repeated grad (e.g. second order wrt an xtensor) sees no residual
    # conversion. Tensor inputs (the common case) are passed through unchanged. Only the
    # outer inputs go through `memo`: boundary/leaf xtensor inputs are never rebuilt, so
    # the inner graph (built from the original identities) is unaffected.
    outer_inputs = [_to_tensor_world(memo.get(v, v)) for v in inputs]
    dummies = [inp.type() for inp in outer_inputs]
    inner = [_from_tensor_world(v, d) for v, d in zip(inputs, dummies)]
    if inputs:
        [region] = graph_replace([exit_var], dict(zip(inputs, inner)), strict=False)
        [lowered] = rewrite_graph([region], include=("lower_xtensor",), clone=False)
    else:
        # Fully constant region: lower it in place, nothing to differentiate through.
        [lowered] = rewrite_graph([exit_var], include=("lower_xtensor",), clone=True)

    if any(
        var.owner is not None
        and isinstance(var.owner.op, _EXIT_OPS)
        and var.owner.inputs[0].owner is not None
        for var in ancestors([lowered])
    ):
        # lower_xtensor could not fully lower the region; wrapping it would leave an
        # un-lowerable xtensor node inside the unit and recurse. Fail loudly instead.
        raise NotImplementedError(
            f"Cannot differentiate through xtensor region ending at {exit_var}: "
            "lower_xtensor left an un-lowered conversion in it."
        )

    if not inputs:
        return lowered
    unit = OpFromGraph(dummies, [lowered], inline=True)
    [new_exit] = unit(*outer_inputs, return_list=True)
    return new_exit


@register_grad_graph_rewriter
def collapse_xtensor_grad_regions(
    outputs: list[Variable], boundaries: list[Variable]
) -> list[Variable]:
    """Collapse every xtensor region in ``outputs`` into a lowered OpFromGraph unit."""
    for out in outputs:
        if _is_xtensor(out):
            raise TypeError(
                "Cannot differentiate an xtensor-typed variable directly: "
                f"{out} (of type {out.type}) was passed as the cost or as a "
                "known_grads key. Convert it to a tensor first, e.g. with "
                "`cost.values`, and differentiate that instead."
            )
    outputs = list(outputs)
    boundary_set = set(boundaries)
    # wrt/consider_constant act by variable identity: an exit that a boundary IS, or
    # that a boundary DEPENDS ON, must survive the collapse unchanged, or the rebuilt
    # graph would no longer contain the boundary and grad could not reach (or stop at)
    # it. Protect the boundaries' whole ancestor cone (`ancestors` is inclusive). The
    # protected set is closed under ancestry, so nothing protected can ever depend on
    # a collapsed exit -- protected variables are never rebuilt below.
    protected = set(ancestors(boundaries))

    def is_exit(var: Variable) -> bool:
        # Only regions with a computed, non-protected xtensor value need collapsing.
        # Converting a bare leaf/constant (e.g. TensorFromXTensor of an XTensorConstant)
        # has nothing to lower away, and collapsing it would just re-wrap the same node
        # and recurse forever. Their trivial boundary pullback handles them instead.
        return (
            var.owner is not None
            and isinstance(var.owner.op, _EXIT_OPS)
            and var.owner.inputs[0].owner is not None
            and var not in protected
            and var.owner.inputs[0] not in protected
        )

    while True:
        # One bottom-up pass over the graph: collapse every exit into its unit and
        # re-clone each downstream node (once) onto the rebuilt variables. Toposort
        # visits dependencies first, so an inner region is collapsed before any region
        # or node consuming it -- `memo` carries the fresh identities forward. This
        # keeps a chain of N regions at one pass instead of one graph rebuild each.
        memo: dict[Variable, Variable] = {}
        for node in toposort(outputs):
            if isinstance(node.op, _EXIT_OPS) and is_exit(node.outputs[0]):
                exit_var = node.outputs[0]
                memo[exit_var] = _collapse_region(exit_var, boundary_set, memo)
            elif any(memo.get(inp, inp) is not inp for inp in node.inputs):
                new_node = node.clone_with_new_inputs(
                    [memo.get(inp, inp) for inp in node.inputs]
                )
                memo.update(zip(node.outputs, new_node.outputs))
        if not memo:
            # `memo` only gains entries through an exit collapse (rebuilds require an
            # already-changed input), so an empty memo means no exits were found.
            return outputs
        outputs = [memo.get(out, out) for out in outputs]
        # Loop as insurance: collapsed units cannot contain further exits (the residual
        # guard raises otherwise), so the next pass normally finds nothing and returns.
        # (Differentiating a unit later re-enters `grad` and hence this hook on its
        # pure-tensor inner graph — those separate, exit-free invocations return here
        # immediately and are not extra passes of this loop.)
