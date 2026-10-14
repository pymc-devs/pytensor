import pytensor.tensor as pt
from pytensor.compile import optdb
from pytensor.graph.rewriting.basic import dfs_rewriter, node_rewriter
from pytensor.tensor.basic import MakeVector
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import Sum
from pytensor.tensor.shape import Reshape
from pytensor.tensor.subtensor import AdvancedIncSubtensor, AdvancedSubtensor
from pytensor.tensor.variable import TensorVariable


@node_rewriter([AdvancedIncSubtensor])
def boolean_indexing_set_or_inc(fgraph, node):
    """Replace `AdvancedIncSubtensor` when using boolean indexing using `Switch`.

    JAX cannot JIT-compile functions that use boolean indexing to set values in
    an array. A workaround is to re-express this logic using `jax.numpy.where`.
    This rewrite allows to improve upon JAX's API.

    """

    # Get out if there is more than one indexing group
    if len(node.inputs) != 3:
        return None

    op = node.op
    [x, y, cond] = node.inputs

    # This rewrite only works when `y` is a scalar, so it can broadcast to the shape of x[cond]
    if y.type.ndim > 0:
        return

    if not isinstance(cond, TensorVariable):
        return

    if not cond.type.dtype == "bool":
        return

    if op.set_instead_of_inc:
        out = pt.where(cond, y, x)
        return out.owner.outputs
    else:
        out = pt.where(cond, x + y, x)
        return out.owner.outputs


optdb.register(
    "jax_boolean_indexing_set_or_inc",
    dfs_rewriter(boolean_indexing_set_or_inc),
    "jax",
    position=100,
)


@node_rewriter([Sum])
def boolean_indexing_sum(fgraph, node):
    """Replace the sum of `AdvancedSubtensor` with exclusively boolean indexing.

    JAX cannot JIT-compile functions that use boolean indexing, but can compile
    those expressions that can be re-expressed using `jax.numpy.where`. This
    rewrite re-rexpressed the model on the behalf of the user and thus allows to
    improve upon JAX's API.

    """
    operand = node.inputs[0]

    if not isinstance(operand, TensorVariable):
        return

    # If it's not a scalar reduction, it couldn't have been a pure boolean mask
    if node.outputs[0].ndim != 0:
        return

    if operand.owner is None:
        return

    if not isinstance(operand.owner.op, AdvancedSubtensor):
        return

    # Get out if AdvancedSubtensor has more than a single indexing operation
    if len(operand.owner.inputs) > 2:
        return

    [x, cond] = operand.owner.inputs

    if not isinstance(cond, TensorVariable):
        return

    if not cond.type.dtype == "bool":
        return

    # Output must be a scalar, since pure boolean indexing returns a vector
    # No need to worry about axis
    out = pt.sum(pt.where(cond, x, 0))
    return out.owner.outputs


optdb.register(
    "jax_boolean_indexing_sum", dfs_rewriter(boolean_indexing_sum), "jax", position=100
)


@node_rewriter([Reshape])
def shape_parameter_as_tuple(fgraph, node):
    """Replace `MakeVector` and `DimShuffle` (when used to transform a scalar
    into a 1d vector) when they are found as the input of a `shape`
    parameter by `JAXShapeTuple` during transpilation.

    The JAX implementations of `MakeVector` and `DimShuffle` always return JAX
    `TracedArrays`, but JAX only accepts concrete values as inputs for the `size`
    or `shape` parameter. When these `Op`s are used to convert scalar or tuple
    inputs, however, we can avoid tracing by making them return a tuple of their
    inputs instead.

    Note that JAX does not accept scalar inputs for the `size` or `shape`
    parameters, and this rewrite also ensures that scalar inputs are turned into
    tuples during transpilation.

    """
    from pytensor.link.jax.dispatch.shape import JAXShapeTuple

    shape_arg = node.inputs[1]
    shape_node = shape_arg.owner

    if shape_node is None:
        return

    if isinstance(shape_node.op, JAXShapeTuple):
        return

    if isinstance(shape_node.op, MakeVector) or (
        isinstance(shape_node.op, DimShuffle)
        and shape_node.op.input_ndim == 0
        and shape_node.op.new_order == ("x",)
    ):
        # Here PyTensor converted a tuple or list to a tensor
        new_shape_args = JAXShapeTuple()(*shape_node.inputs)
        new_inputs = list(node.inputs)
        new_inputs[1] = new_shape_args

        new_node = node.clone_with_new_inputs(new_inputs)
        return new_node.outputs


optdb.register(
    "jax_shape_parameter_as_tuple",
    dfs_rewriter(shape_parameter_as_tuple),
    "jax",
    position=100,
)


@node_rewriter([Blockwise])
def blockwise_unbatched_scalar_inputs(fgraph, node):
    """Keep scalar ``Blockwise`` inputs without batch dims out of ``jnp.vectorize``.

    ``Blockwise.make_node`` pads such inputs into a ``MakeVector`` or an
    ``expand_dims``, whose JAX implementations return traced arrays. Replacing the
    padding by ``JAXUnbatchedScalar`` hands the core function the scalar itself, so
    a static value like ``Scan``'s ``n_steps`` stays static.
    """
    from pytensor.link.jax.dispatch.blockwise import (
        JAXUnbatchedScalar,
        unbatched_scalar_constant,
    )

    new_inputs = list(node.inputs)
    for i, (inp, sig) in enumerate(zip(node.inputs, node.op.inputs_sig, strict=True)):
        if sig or not all(inp.type.broadcastable):
            continue
        # The Blockwise dispatch passes constants as static values itself, and core
        # dispatches like Convolve1d read them from the graph
        if unbatched_scalar_constant(inp, sig) is not None:
            continue
        # Peel the padding down to the scalar. Every layer is all-broadcastable,
        # because a DimShuffle cannot drop a non-broadcastable dim
        scalar = inp
        while scalar.type.ndim:
            match scalar.owner_op_and_inputs:
                case (MakeVector() | DimShuffle(), inner):
                    scalar = inner
                case _:
                    break
        if scalar.type.ndim or scalar is inp:
            continue
        new_inputs[i] = JAXUnbatchedScalar(inp.type.ndim)(scalar)

    if new_inputs == node.inputs:
        return None
    return node.clone_with_new_inputs(new_inputs).outputs


optdb.register(
    "jax_blockwise_unbatched_scalar_inputs",
    dfs_rewriter(blockwise_unbatched_scalar_inputs),
    "jax",
    position=100,
)
