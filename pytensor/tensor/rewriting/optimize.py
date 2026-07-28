from pytensor.compile import optdb
from pytensor.compile.aliasing import add_supervisor_to_fgraph
from pytensor.compile.io import In
from pytensor.compile.rewriting import rewrite_inner_graph
from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import graph_rewriter, node_rewriter
from pytensor.link.basic import PerformLinker
from pytensor.link.c.basic import CLinker, OpWiseCLinker
from pytensor.link.jax.linker import JAXLinker
from pytensor.link.mlx.linker import MLXLinker
from pytensor.link.numba.linker import NumbaLinker
from pytensor.link.pytorch.linker import PytorchLinker
from pytensor.link.vm import VMLinker
from pytensor.tensor.optimize import ScipyWrapperOp, rewrite_optimize_inner_graph
from pytensor.tensor.rewriting.basic import register_canonicalize


@rewrite_optimize_inner_graph.register(VMLinker)
@rewrite_optimize_inner_graph.register(PerformLinker)
@rewrite_optimize_inner_graph.register(CLinker)
@rewrite_optimize_inner_graph.register(OpWiseCLinker)
def c_rewrite_optimize_inner_graph(linker, op, node, inner, *, mode):
    # Same contract as ``OpFromGraph``: inputs (the optimization variable + args)
    # must not be mutated, so they are protected; inplace may still be baked
    # between purely internal buffers. ``compile_fn`` then only links this graph.
    # The Supervisor is needed even with no mutable inputs: it is the feature that
    # vetoes input-destroying inplace rewrites while ``mode.optimizer`` runs them.
    input_specs = [In(x, borrow=True, mutable=False) for x in inner.inputs]
    add_supervisor_to_fgraph(fgraph=inner, input_specs=input_specs, accept_inplace=True)
    mode.optimizer.rewrite(inner)


@rewrite_optimize_inner_graph.register(NumbaLinker)
@rewrite_optimize_inner_graph.register(JAXLinker)
@rewrite_optimize_inner_graph.register(PytorchLinker)
@rewrite_optimize_inner_graph.register(MLXLinker)
def jit_rewrite_optimize_inner_graph(linker, op, node, inner, *, mode):
    # JIT backends manage memory themselves, so leave the inner graph functional.
    # (Unlike ``OpFromGraph`` under numba, no deepcopies are baked in either: a
    # scipy op is never funcified -- it always perform-links via ``compile_fn``,
    # whose ``FunctionMaker`` pass inserts the boundary deepcopies.)
    mode.excluding("inplace").optimizer.rewrite(inner)


@graph_rewriter
def optimize_inner_graph(fgraph):
    rewrite_inner_graph(
        fgraph, lambda op: isinstance(op, ScipyWrapperOp), rewrite_optimize_inner_graph
    )


optdb.register(
    "optimize_inner_graph",
    optimize_inner_graph,
    "minimum_compile",
    "compile_inner_graph",
    position=49.6,
)


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
    # Thaw the frozen inner graph; substitutions run on the mutable copy.
    unfrozen_fgraph = op.fgraph.unfreeze()
    inner_x, *inner_args = unfrozen_fgraph.inputs
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

    unfrozen_fgraph.replace_all(list(givens.items()), import_missing=True)
    new_inner_inputs = (inner_x, *new_inner_args)
    new_fgraph = FunctionGraph(new_inner_inputs, unfrozen_fgraph.outputs, clone=False)
    new_op = op.clone_with_new_fgraph(new_fgraph)
    new_outer_inputs = (outer_x, *new_outer_args)
    return new_op.make_node(*new_outer_inputs).outputs
