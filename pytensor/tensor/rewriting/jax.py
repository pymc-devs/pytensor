import pytensor.tensor as pt
from pytensor.compile import optdb
from pytensor.graph.rewriting.basic import copy_stack_trace, dfs_rewriter, node_rewriter
from pytensor.tensor.basic import MakeVector, get_underlying_scalar_constant_value
from pytensor.tensor.blockwise import Blockwise, JAXBlockwise, _squeeze_left
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import Sum
from pytensor.tensor.rewriting.elemwise import apply_local_dimshuffle_merge
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
    """Lower Blockwise with explicit scalar inputs that bypass vectorization.

    Removing padding before JAX conversion preserves concrete values such as
    ``Shape_i`` outputs. Unbatched scalars that are already traced stay traced.
    """
    unbatched_inputs = tuple(
        i
        for i, (inp, sig) in enumerate(
            zip(node.inputs, node.op.inputs_sig, strict=True)
        )
        if not sig and all(inp.type.broadcastable)
    )
    if not unbatched_inputs:
        return None

    batch_ndim = node.op.batch_ndim(node)
    inputs = list(node.inputs)
    for i in unbatched_inputs:
        scalar_input = apply_local_dimshuffle_merge(
            fgraph, _squeeze_left(inputs[i], batch_ndim)
        )
        inputs[i] = pt.as_tensor_variable(
            get_underlying_scalar_constant_value(
                scalar_input, raise_not_constant=False
            ),
            dtype=scalar_input.dtype,
        )
    inner_inputs = [inp.type() for inp in inputs]
    padded_inputs = [
        pt.shape_padleft(inp, batch_ndim) if i in unbatched_inputs else inp
        for i, inp in enumerate(inner_inputs)
    ]
    inner_outputs = node.op(*padded_inputs, return_list=True)
    outputs = JAXBlockwise(
        inner_inputs, inner_outputs, destroy_map=node.op.destroy_map
    )(*inputs, return_list=True)
    return copy_stack_trace(node.outputs, outputs)


optdb.register(
    "jax_blockwise_unbatched_scalar_inputs",
    dfs_rewriter(blockwise_unbatched_scalar_inputs),
    "jax",
    position=100,
)
