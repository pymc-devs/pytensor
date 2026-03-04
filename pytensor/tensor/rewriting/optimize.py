from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.optimize import ScipyWrapperOp
from pytensor.tensor.rewriting.basic import register_canonicalize


@register_canonicalize
@node_rewriter([ScipyWrapperOp])
def remove_constants_and_duplicate_inputs_scipy(fgraph, node):
    """Inline constants and remove duplicate inputs from ScipyWrapperOp nodes.

    Constants in the outer graph are free symbolic variables in the inner graph.
    Moving them into the inner graph enables constant-folding. Duplicate outer
    inputs can share a single inner variable.

    Only args (inputs[1:]) are candidates — inputs[0] is always the
    optimization variable x.
    """
    op: ScipyWrapperOp = node.op
    inner_x, *inner_args = op.inner_inputs
    outer_x, *outer_args = list(node.inputs)

    givens = {}
    new_inner_args = []
    new_outer_args = []

    for inner_in, outer_in in zip(inner_args, outer_args):
        if isinstance(outer_in, Constant):
            givens[inner_in] = outer_in
        elif outer_in in new_outer_args:
            # De-duplicate outer variable
            idx = new_outer_args.index(outer_in)
            givens[inner_in] = new_inner_args[idx]
        else:
            new_inner_args.append(inner_in)
            new_outer_args.append(outer_in)

    if not givens:
        return None

    new_inner_outputs = clone_replace(op.inner_outputs, replace=givens)
    new_inner_inputs = (inner_x, *new_inner_args)
    new_fgraph = FunctionGraph(new_inner_inputs, new_inner_outputs, clone=False)
    new_op = op.clone_with_new_fgraph(new_fgraph)
    new_outer_inputs = (outer_x, *new_outer_args)
    return new_op.make_node(*new_outer_inputs).outputs
