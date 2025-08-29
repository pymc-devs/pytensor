from pytensor.compile import optdb
from pytensor.graph import node_rewriter
from pytensor.graph.rewriting.basic import dfs_rewriter
from pytensor.graph.traversal import applys_between
from pytensor.tensor.basic import as_tensor, constant
from pytensor.tensor.blockwise import Blockwise, BlockwiseWithCoreShape
from pytensor.tensor.rewriting.shape import ShapeFeature


@node_rewriter([Blockwise])
def introduce_explicit_core_shape_blockwise(fgraph, node):
    """Introduce the core shape of a Blockwise.

    We wrap Blockwise graphs into a BlockwiseWithCoreShape OpFromGraph
    that has an extra "non-functional" input that represents the core shape of the Blockwise variable.
    This core_shape is used by the numba backend to pre-allocate the output array.

    If available, the core shape is extracted from the shape feature of the graph,
    which has a higher change of having been simplified, optimized, constant-folded.
    If missing, we fall back to the op._supp_shape_from_params method.

    This rewrite is required for the numba backend implementation of Blockwise.

    Example
    -------

    .. code-block:: python

        import pytensor
        import pytensor.tensor as pt

        x = pt.tensor("x", shape=(5, None, None))
        outs = pt.linalg.svd(x, compute_uv=True)
        pytensor.dprint(outs)
        # Blockwise{SVD{full_matrices=True, compute_uv=True}, (m,n)->(m,m),(k),(n,n)}.0 [id A]
        #  └─ x [id B]
        # Blockwise{SVD{full_matrices=True, compute_uv=True}, (m,n)->(m,m),(k),(n,n)}.1 [id A]
        #  └─ ···
        # Blockwise{SVD{full_matrices=True, compute_uv=True}, (m,n)->(m,m),(k),(n,n)}.2 [id A]
        #  └─ ···

        # After the rewrite, note the new 3 core shape inputs
        fn = pytensor.function([x], outs, mode="NUMBA")
        fn.dprint(print_type=False)
        # [Blockwise{SVD{full_matrices=True, compute_uv=True}, (m,n)->(m,m),(k),(n,n)}].0 [id A] 6
        #  ├─ x [id B]
        #  ├─ MakeVector{dtype='int64'} [id C] 5
        #  │  ├─ Shape_i{1} [id D] 2
        #  │  │  └─ x [id B]
        #  │  └─ Shape_i{1} [id D] 2
        #  │     └─ ···
        #  ├─ MakeVector{dtype='int64'} [id E] 4
        #  │  └─ Minimum [id F] 3
        #  │     ├─ Shape_i{1} [id D] 2
        #  │     │  └─ ···
        #  │     └─ Shape_i{2} [id G] 0
        #  │        └─ x [id B]
        #  └─ MakeVector{dtype='int64'} [id H] 1
        #     ├─ Shape_i{2} [id G] 0
        #     │  └─ ···
        #     └─ Shape_i{2} [id G] 0
        #        └─ ···
        # [Blockwise{SVD{full_matrices=True, compute_uv=True}, (m,n)->(m,m),(k),(n,n)}].1 [id A] 6
        #  └─ ···
        # [Blockwise{SVD{full_matrices=True, compute_uv=True}, (m,n)->(m,m),(k),(n,n)}].2 [id A] 6
        #  └─ ···
    """
    op: Blockwise = node.op
    batch_ndim = op.batch_ndim(node)

    shape_feature: ShapeFeature | None = getattr(fgraph, "shape_feature", None)
    if shape_feature:
        core_shapes = [
            [shape_feature.get_shape(out, i) for i in range(batch_ndim, out.type.ndim)]
            for out in node.outputs
        ]
    else:
        input_shapes = [tuple(inp.shape) for inp in node.inputs]
        core_shapes = [
            out_shape[batch_ndim:]
            for out_shape in op.infer_shape(None, node, input_shapes)
        ]

    core_shapes = [
        as_tensor(core_shape) if len(core_shape) else constant([], dtype="int64")
        for core_shape in core_shapes
    ]

    if any(
        isinstance(node.op, Blockwise)
        for node in applys_between(node.inputs, core_shapes)
    ):
        # If Blockwise shows up in the shape graph we can't introduce the core shape
        return None

    return BlockwiseWithCoreShape(
        [*node.inputs, *core_shapes],
        node.outputs,
        destroy_map=op.destroy_map,
    )(*node.inputs, *core_shapes, return_list=True)


optdb.register(
    introduce_explicit_core_shape_blockwise.__name__,
    dfs_rewriter(introduce_explicit_core_shape_blockwise),
    "numba",
    position=100,
)
