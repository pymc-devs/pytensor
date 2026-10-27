from collections.abc import Callable, Sequence
from functools import reduce, singledispatch

import numpy as np

from pytensor import tensor as pt
from pytensor.configdefaults import config
from pytensor.gradient import _node_to_pattern
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import graph_replace
from pytensor.graph.rewriting.unify import OpPattern
from pytensor.graph.traversal import graph_inputs
from pytensor.tensor.rewriting.basic import topo_unconditional_constant_folding
from pytensor.tensor.type import TensorType, float_dtypes
from pytensor.tensor.variable import TensorVariable


@singledispatch
def local_sparsity(op, node, zeros):
    """Return the local Jacobian structure of one Apply node.

    ``blocks[o][i]`` is a chain of symbolic factor matrices whose product, of shape
    ``(outputs[o].size, inputs[i].size)``, is the pattern of ``outputs[o]`` in
    ``inputs[i]``. An empty chain is the identity, and None means no entry is set.
    ``zeros`` maps each input to a symbolic flat mask of its elements that are zero for
    every input value, or None when none is known. Inputs with known values are
    ``Constant``s, because the graph is constant-folded first.

    An Op without a rule makes every output element depend on every input element. A
    scalar-typed variable counts as one element.
    """
    input_sizes = [
        variable.size if isinstance(variable.type, TensorType) else 1
        for variable in node.inputs
    ]
    return [
        [
            (
                pt.ones(
                    (
                        output.size if isinstance(output.type, TensorType) else 1,
                        input_size,
                    ),
                    dtype=config.floatX,
                ),
            )
            for input_size in input_sizes
        ]
        for output in node.outputs
    ]


@singledispatch
def local_zeros(op, node, zeros):
    """Return, per output, a symbolic flat mask of the elements that are zero whenever
    the masked input elements are, or None.

    An Op without a rule marks no output element as known zero.
    """
    return [None for _ in node.outputs]


# Rules for Ops that a type alone cannot single out, such as an Elemwise of a particular
# scalar Op, per dispatcher. Keying by pattern makes registering the same pattern again
# replace its rule.
PATTERN_RULES: dict[Callable, dict[OpPattern, Callable]] = {
    local_sparsity: {},
    local_zeros: {},
}


def register_pattern_rule(dispatcher: Callable, pattern: OpPattern) -> Callable:
    """Register a rule of ``dispatcher`` for the Ops that ``pattern`` matches.

    ``dispatcher`` is `local_sparsity` or `local_zeros`. A matching pattern takes
    precedence over the rule registered for the Op's type, and the first registered
    pattern that matches wins.
    """

    def register(rule):
        PATTERN_RULES[dispatcher][pattern] = rule
        return rule

    return register


def rule_for(dispatcher, op):
    """Return the rule of ``dispatcher`` that applies to ``op``."""
    return next(
        (
            rule
            for pattern, rule in PATTERN_RULES[dispatcher].items()
            if pattern.match_op(op)
        ),
        dispatcher,
    )


def node_sparsity(node, zeros):
    """`local_sparsity`, without the blocks that the Op's connection pattern rules out
    or that join an integer input or output.

    A node with no differentiable connection needs no rule.
    """
    connection_pattern = _node_to_pattern(node)
    connected = [
        [
            connection_pattern[input_position][output_position]
            and getattr(variable.type, "dtype", None) in float_dtypes
            and getattr(output.type, "dtype", None) in float_dtypes
            for input_position, variable in enumerate(node.inputs)
        ]
        for output_position, output in enumerate(node.outputs)
    ]
    if not any(any(row) for row in connected):
        return [[None] * len(node.inputs) for _ in node.outputs]

    blocks = rule_for(local_sparsity, node.op)(node.op, node, zeros)
    return [
        [
            block if keep else None
            for block, keep in zip(block_row, connected_row, strict=True)
        ]
        for block_row, connected_row in zip(blocks, connected, strict=True)
    ]


def apply_chain(chain, pattern):
    """Multiply ``pattern`` by the factors of ``chain``, rightmost first."""
    for factor in reversed(chain):
        pattern = pt.dot(factor, pattern)
    return pattern


def jacobian_sparsity(
    outputs: Variable | Sequence[Variable], wrt: Variable | Sequence[Variable]
) -> TensorVariable:
    """Symbolic sparsity pattern of the Jacobian of ``outputs`` with respect to ``wrt``.

    Each Op contributes its local structure through `local_sparsity`, and the blocks
    are composed over the constant-folded graph. The pattern reads the graph's inputs,
    including those held fixed, only through their shapes, so one compiled pattern
    serves every input shape. It holds for every input value, so it may contain entries
    that are zero at a particular point, but never misses one that is nonzero.

    Parameters
    ----------
    outputs : Variable or sequence of Variable
        Tensors whose Jacobian is described. Each is flattened, and they are
        concatenated in the order given.
    wrt : Variable or sequence of Variable
        Tensors to differentiate with respect to, flattened and concatenated in the
        order given. They may be intermediate variables of the graph, ancestors of
        one another, or repeated. Each column is the total derivative, as in
        `jacobian`. Other inputs of the graph are held fixed.

    Returns
    -------
    pattern : TensorVariable of bool
        Shape ``(total output elements, total wrt elements)``. Entry ``(o, i)`` is
        False only if the derivative of output element ``o`` with respect to element
        ``i`` of ``wrt`` is zero for every input value.

    Examples
    --------
    .. code-block:: python

        import pytensor
        import pytensor.tensor as pt

        x = pt.vector("x")
        pattern = pt.jacobian_sparsity(pt.exp(x) * 2, x)
        pytensor.function([x], pattern)([1.0, 2.0, 3.0])
    """
    outputs = [outputs] if isinstance(outputs, Variable) else list(outputs)
    wrt = [wrt] if isinstance(wrt, Variable) else list(wrt)

    inputs = [
        variable
        for variable in graph_inputs([*outputs, *wrt])
        if not isinstance(variable, Constant)
    ]
    # wrt are outputs of the graph, so constant folding keeps their clones reachable
    # without cutting the paths from their ancestors.
    fgraph = FunctionGraph(inputs=inputs, outputs=[*outputs, *wrt], clone=True)
    topo_unconditional_constant_folding.rewrite(fgraph)

    fgraph_wrt = fgraph.outputs[len(outputs) :]
    sizes = [variable.size for variable in fgraph_wrt]  # type: ignore[attr-defined]
    offsets = []
    n_columns = pt.constant(0, dtype="int64")
    for size in sizes:
        offsets.append(n_columns)
        n_columns = n_columns + size

    seeds: dict[Variable, TensorVariable] = {}
    for variable, size, offset in zip(fgraph_wrt, sizes, offsets, strict=True):
        seed = pt.eye(size, n_columns, k=offset, dtype=config.floatX)
        seeds[variable] = seeds[variable] + seed if variable in seeds else seed
    patterns = dict(seeds)
    zeros = dict.fromkeys(fgraph.inputs)
    for variable in fgraph.variables:
        if (
            isinstance(variable, Constant)
            and isinstance(variable.data, np.ndarray)
            and variable not in seeds
        ):
            is_zero = variable.data == 0
            if is_zero.any():
                zeros[variable] = pt.constant(is_zero.ravel())

    for node in fgraph.toposort():
        node_zeros = {variable: zeros.get(variable) for variable in node.inputs}
        zeros.update(
            zip(
                node.outputs,
                rule_for(local_zeros, node.op)(node.op, node, node_zeros),
                strict=True,
            )
        )
        # A wrt variable is perturbed freely, so none of its elements is a known zero.
        zeros.update((output, None) for output in node.outputs if output in seeds)
        if not any(variable in patterns for variable in node.inputs):
            continue

        blocks = node_sparsity(node, node_zeros)
        for output, row in zip(node.outputs, blocks, strict=True):
            contributions = [
                apply_chain(chain, patterns[variable])
                for variable, chain in zip(node.inputs, row, strict=True)
                if chain is not None and variable in patterns
            ]
            if not contributions:
                continue

            # Thresholding keeps the entries 0 or 1, so long chains of products cannot
            # overflow.
            output_pattern = pt.cast(
                pt.gt(reduce(pt.add, contributions), 0), config.floatX
            )
            if (output_zeros := zeros[output]) is not None:
                output_pattern *= pt.cast(~output_zeros, config.floatX)[:, None]
            # A wrt variable is the identity in its own columns.
            if output in seeds:
                output_pattern += seeds[output]
            patterns[output] = output_pattern

    output_patterns = [
        patterns[output]
        if output in patterns
        else pt.zeros((output.size, n_columns), dtype=config.floatX)  # type: ignore[attr-defined]
        for output in fgraph.outputs[: len(outputs)]
    ]
    pattern = pt.gt(pt.concatenate(output_patterns, axis=0), 0)

    # The folded graph holds clones of the inputs, and the caller's graph refers to
    # the originals.
    return graph_replace(  # type: ignore[no-any-return]
        pattern, replace=dict(zip(fgraph.inputs, inputs, strict=True)), strict=False
    )
