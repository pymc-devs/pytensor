from pytensor.compile import optdb
from pytensor.configdefaults import config
from pytensor.graph import node_rewriter
from pytensor.graph.rewriting.basic import copy_stack_trace, dfs_rewriter
from pytensor.graph.traversal import ancestors
from pytensor.tensor import as_tensor, constant
from pytensor.tensor.basic import cast
from pytensor.tensor.random.basic import (
    InvGammaRV,
    LogNormalRV,
    MvNormalRV,
    NormalRV,
)
from pytensor.tensor.random.op import RandomVariable, RandomVariableWithCoreShape
from pytensor.tensor.rewriting.numba import simplify_core_shape_graphs
from pytensor.tensor.rewriting.shape import ShapeFeature
from pytensor.tensor.variable import TensorConstant


_RVS_TO_CAST_FLOAT_PARAMS_TO_FLOAT64 = frozenset(
    (NormalRV, LogNormalRV, InvGammaRV, MvNormalRV)
)


@node_rewriter([RandomVariable])
def introduce_explicit_core_shape_rv(fgraph, node):
    """Introduce the core shape of a RandomVariable.

    We wrap RandomVariable graphs into a RandomVariableWithCoreShape OpFromGraph
    that has an extra "non-functional" input that represents the core shape of the random variable.
    This core_shape is used by the numba backend to pre-allocate the output array.

    The core shape is built from ``ShapeFeature.get_non_recursive_shape``, whose
    expressions read only the node's own inputs and can therefore be introduced
    after inplacing without conflicting with the destroyers in the surrounding
    graph.

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
    if shape_feature is None:
        shape_feature = ShapeFeature()

    core_shape = [
        shape_feature.get_non_recursive_shape(rv, i)
        for i in range(rv.type.ndim - op.ndim_supp, rv.type.ndim)
    ]

    if len(core_shape) == 0:
        core_shape = constant([], dtype="int64")
    else:
        core_shape = as_tensor(core_shape)

    # ``get_non_recursive_shape`` reads only this node's own inputs, so any
    # RandomVariable in the core-shape graph is itself an input and gets blocked
    # here -- it is computed before the node, so reading its shape needs no extra
    # draw (e.g. a random ``mean`` whose trailing dim is the core shape). Only a
    # RandomVariable reachable *past* the inputs would force a duplicate draw.
    node_inputs = frozenset(node.inputs)
    if any(
        var not in node_inputs and isinstance(var.owner_op, RandomVariable)
        for var in ancestors([core_shape], blockers=node_inputs)
    ):
        return None

    [core_shape] = simplify_core_shape_graphs([core_shape], fgraph)

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


@node_rewriter([RandomVariable])
def cast_rv_float_params_to_float64(fgraph, node):
    """Cast non-float64 floating-point RandomVariable parameters to float64.

    For a few distributions, numba's sampler runs markedly faster -- or only compiles
    at all -- when its float parameters are float64 rather than float32 / integer. The
    declared ``dtype`` (and ``floatX``) is only a final cast on the output, never the
    sampling precision, so upcasting the parameters once here is a performance fix, not
    a semantic one: the draw is still cast to the declared dtype on store, exactly as
    ``RandomVariable.perform`` does (it hands parameters straight to NumPy and narrows
    only the output via ``np.asarray(..., dtype=self.dtype)``).

    Which RVs actually benefit is not predictable from "the generator samples in
    float64" -- it depends on the specific numba implementation and was measured per
    distribution; see ``_RVS_TO_CAST_FLOAT_PARAMS_TO_FLOAT64`` for the opt-in list. For
    ``normal``, the ``int8`` literals of ``normal(0, 1)`` becoming float64 is a ~3x win.

    The target is float64, *not* the output dtype: under ``floatX="float32"`` casting to
    the output dtype would leave the parameter float32 and still mismatched, so it would
    not help. This is why the rewrite is Numba-specific -- JAX can sample natively in
    float32, where forcing float64 parameters would pessimize.
    """
    op = node.op
    if type(op) not in _RVS_TO_CAST_FLOAT_PARAMS_TO_FLOAT64:
        # Opt-in narrowing. The rewrite is registered on the ``RandomVariable`` base
        # class (one cheap isinstance per graph node) and the exact-type set membership
        # here only runs for the few actual RV nodes, bailing on all but the listed ones.
        return None

    if config.warn_float64 != "ignore":
        # The user asked to be warned about / forbidden float64; don't introduce it.
        return None

    dist_params = op.dist_params(node)
    # Every parameter of these (continuous) RVs is real-valued, so upcast any that is not
    # already float64 -- ``normal(0, 1)``'s int8 literals included.
    cast_idxs = [
        i for i, param in enumerate(dist_params) if param.type.dtype != "float64"
    ]
    if not cast_idxs:
        return None

    new_params = list(dist_params)
    for i in cast_idxs:
        param = dist_params[i]
        if isinstance(param, TensorConstant):
            # Fold the cast into the constant so the common ``normal(0, 1)`` case ends
            # up with float64 literals rather than a leftover Cast node.
            new_params[i] = constant(param.data.astype("float64"))
        else:
            new_params[i] = cast(param, "float64")

    new_outputs = op.make_node(
        op.rng_param(node), op.size_param(node), *new_params
    ).outputs
    copy_stack_trace(node.outputs, new_outputs)
    return new_outputs


optdb.register(
    cast_rv_float_params_to_float64.__name__,
    dfs_rewriter(cast_rv_float_params_to_float64),
    "numba",
    # After stabilize (1.5), before specialize (2): a one-shot sweep that inserts the
    # casts, leaving specialize's equilibrium (local_cast_cast + constant folding) to
    # clean them up, and running before the core-shape wrap at position 100.
    position=1.9,
)
