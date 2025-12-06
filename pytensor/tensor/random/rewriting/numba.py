from pytensor.compile import optdb
from pytensor.graph import node_rewriter
from pytensor.graph.rewriting.basic import copy_stack_trace, dfs_rewriter
from pytensor.tensor import as_tensor, constant
from pytensor.tensor.random.op import RandomVariable, RandomVariableWithCoreShape
from pytensor.tensor.rewriting.shape import ShapeFeature


@node_rewriter([RandomVariable])
def introduce_explicit_core_shape_rv(fgraph, node):
    """Introduce the core shape of a RandomVariable.

    We wrap RandomVariable graphs into a RandomVariableWithCoreShape OpFromGraph
    that has an extra "non-functional" input that represents the core shape of the random variable.
    This core_shape is used by the numba backend to pre-allocate the output array.

    If available, the core shape is extracted from the shape feature of the graph,
    which has a higher chance of having been simplified, optimized, constant-folded.
    If missing, we fall back to the op._supp_shape_from_params method.

    This rewrite is required for the numba backend implementation of RandomVariable.

    Example
    -------

    .. code-block:: python

            import pytensor
            import pytensor.tensor as pt

            x = pt.random.dirichlet(alphas=[1, 2, 3], size=(5,))
            pytensor.dprint(x, print_type=True)
            # dirichlet_rv{"(a)->(a)"}.1 [id A] <Matrix(float64, shape=(5, 3))>
            #  ├─ RNG(<Generator(PCG64) at 0x7F09E59C18C0>) [id B] <RandomGeneratorType>
            #  ├─ [5] [id C] <Vector(int64, shape=(1,))>
            #  └─ ExpandDims{axis=0} [id D] <Matrix(int64, shape=(1, 3))>
            #     └─ [1 2 3] [id E] <Vector(int64, shape=(3,))>

            # After the rewrite, note the new core shape input [3] [id B]
            fn = pytensor.function([], x, mode="NUMBA")
            pytensor.dprint(fn.maker.fgraph)
            # [dirichlet_rv{"(a)->(a)"}].1 [id A] 0
            #  ├─ [3] [id B]
            #  ├─ RNG(<Generator(PCG64) at 0x7F15B8E844A0>) [id C]
            #  ├─ [5] [id D]
            #  └─ [[1 2 3]] [id E]
            # Inner graphs:
            # [dirichlet_rv{"(a)->(a)"}] [id A]
            #  ← dirichlet_rv{"(a)->(a)"}.0 [id F]
            #     ├─ *1-<RandomGeneratorType> [id G]
            #     ├─ *2-<Vector(int64, shape=(1,))> [id H]
            #     └─ *3-<Matrix(int64, shape=(1, 3))> [id I]
            #  ← dirichlet_rv{"(a)->(a)"}.1 [id F]
            #     └─ ···
    """
    op: RandomVariable = node.op

    _next_rng, rv = node.outputs
    shape_feature: ShapeFeature | None = getattr(fgraph, "shape_feature", None)
    if shape_feature:
        core_shape = [
            shape_feature.get_shape(rv, -i - 1) for i in reversed(range(op.ndim_supp))
        ]
    else:
        core_shape = op._supp_shape_from_params(op.dist_params(node))

    if len(core_shape) == 0:
        core_shape = constant([], dtype="int64")
    else:
        core_shape = as_tensor(core_shape)

    new_outs = (
        RandomVariableWithCoreShape(
            [core_shape, *node.inputs],
            node.outputs,
            destroy_map={0: [1]} if op.inplace else None,
        )
        .make_node(core_shape, *node.inputs)
        .outputs
    )
    copy_stack_trace(node.outputs, new_outs)
    return new_outs


optdb.register(
    introduce_explicit_core_shape_rv.__name__,
    dfs_rewriter(introduce_explicit_core_shape_rv),
    "numba",
    position=100,
)
