from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Number
from types import UnionType
from typing import Any

import numpy as np

from pytensor.graph.basic import Constant, Variable
from pytensor.graph.op import Op


type OpPatternOpTypeType = type[Op] | tuple[type[Op], ...] | UnionType

_MISSING = object()


@dataclass(frozen=True)
class LiteralString:
    """Wrapper for strings that must match literally instead of becoming a `PatternVar`."""

    value: str


class PatternVar:
    """Named wildcard that binds to the first value it is matched against, subject to an optional constraint."""

    __slots__ = ("constraint", "token")

    def __init__(self, token: str, constraint: Callable[[Any], bool] | None = None):
        self.token = token
        self.constraint = constraint

    def __repr__(self) -> str:
        if self.constraint is not None:
            return f"PatternVar({self.token!r}, constraint=...)"
        return f"PatternVar({self.token!r})"


class ConstrainedVar(PatternVar):
    """A logical variable with a constraint."""

    def __new__(cls, constraint, token: str | None = None, prefix: str = ""):
        return object.__new__(cls)

    def __init__(self, constraint, token: str | None = None, prefix: str = ""):
        if token is None:
            token = f"{prefix}_constrained"
        super().__init__(token=token, constraint=constraint)


@dataclass(unsafe_hash=True)
class OpPattern:
    """Class that can be unified with Op instances of a given type (or instance) and parameters.

    Parameters that are not specified in the OpPattern are ignored during unification.

    This is needed because some Ops can be complex to parametrize fully,
    and not all parameters are relevant for a given pattern.


    Examples
    --------

    OpPattern can be used in the `tracks` functionality of `node_rewriter` to more flexible filter out nodes.
    For Ops that are parametrized by other Ops, it's possible to use nested OpPatterns.

    .. test-code::

        from pytensor.graph.rewriting.basic import node_rewriter
        from pytensor.graph.rewriting.unify import OpPattern
        from pytensor.tensor.elemwise import CAReduce
        from pytensor.tensor.blockwise import Blockwise
        from pytensor.tensor.linalg.solvers.general import Solve

        @node_rewriter(tracks=[OpPattern(CAReduce, axis=None)])
        def local_car_reduce_all_rewriter(fgraph, node):
            # This will always be true!
            assert isinstance(node.op, CAReduce) and node.op.axis is None
            ...

        # Any Blockwise whose core_op is a Solve Op (or subclass) instance
        @node_rewriter(tracks=[OpPattern(Blockwise, core_op=OpPattern(Solve))])
        def local_blockwise_solve_triangular_rewriter(fgraph, node):
            # This will always be true!
            assert isinstance(node.op, Blockwise) and isinstance(node.op.core_op, Solve)
            ...

        # Any Blockwise whose core_op is a Solve Op (or subclass) instance with b_ndim==1
        @node_rewriter(tracks=[OpPattern(Blockwise, core_op=OpPattern(Solve, b_ndim=1))])
        def local_blockwise_vector_solve_rewriter(fgraph, node):
            # This will always be true!
            assert (
                isinstance(node.op, Blockwise)
                and isinstance(node.op.core_op, Solve)
                and node.op.core_op.b_ndim == 1
            )
            ...


    OpPattern can be used with `PatternNodeRewriter` to define graph rewrites that match Ops with specific parameters.
    The example below matches two nested CAReduce Ops with the same `scalar_op`,
    the outer with `axis=None` (full reduction) and fuses them into a single CAReduce.
    Note, that because we didn't specify it, the axis of the inner CAReduce can be anything.
    The same goes for other properties of the Op that are not specified in the OpPattern.

    .. testcode::

        from pytensor.graph.rewriting.basic import PatternNodeRewriter
        from pytensor.graph.rewriting.unify import OpPattern
        from pytensor.tensor.basic import Join
        from pytensor.tensor.elemwise import CAReduce, Elemwise

        def output_fn(fgraph, node, s):
            reduce_op = node.op
            reduced_a = reduce_op(s["a"])
            reduced_b = reduce_op(s["b"])
            return Elemwise(s["scalar_op"])(reduced_a, reduced_b)


        PatternNodeRewriter(
            in_pattern=(OpPattern(CAReduce, scalar_op="scalar_op", axis=None),
                (OpPattern(CAReduce, scalar_op="scalar_op",), "x")),
            out_pattern=(OpPattern(CAReduce, scalar_op="scalar_op", axis=None), "x"),
        )

    """

    op_type: OpPatternOpTypeType
    parameters: tuple[tuple[str, Any], ...]

    def __init__(
        self,
        op_type: OpPatternOpTypeType,
        parameters: dict[str, Any] | Sequence[tuple[str, Any]] | None = None,
        **kwargs,
    ):
        if kwargs:
            if parameters is not None:
                raise ValueError(
                    "Cannot provide both parameters dict and keyword arguments"
                )
            parameters = kwargs
        if isinstance(parameters, dict):
            parameters = tuple(sorted(parameters.items()))
        elif isinstance(parameters, list | tuple):
            parameters = tuple(sorted(parameters))
        elif parameters is None:
            parameters = ()
        self.op_type = op_type
        self.parameters = parameters  # type: ignore[assignment]

    def match_op(self, op: Op) -> bool:
        if not isinstance(op, self.op_type):
            return False
        return self.match_parameters(op)

    def match_parameters(self, op: Op) -> bool:
        # This is used by methods that already check the op_type is satisfied
        # Some methods may index on the op_type and know in advance the op is matched
        # Also recursive calls to OpPattern.match_parameters do the op check outside to exit early (see below)
        for key, param in self.parameters:
            if isinstance(param, OpPattern):
                # Parameters can itself be other OpPatterns
                # We check the op_type to avoid a nested call in cases we can reject early
                sub_op = getattr(op, key)
                if not isinstance(sub_op, param.op_type):
                    return False
                # Match the pattern of the inner Op
                # Skip if there are no parameters
                if param.parameters and not param.match_parameters(sub_op):
                    return False
            elif isinstance(param, PatternVar):
                # Wildcard parameter: matches any value (bindings are only tracked
                # in the binding path, see _bind_op_parameters)
                continue
            elif getattr(op, key) != param:
                return False
        return True

    def __str__(self):
        return f"OpPattern({self.op_type}, {', '.join(f'{k}={v}' for k, v in self.parameters)})"


class PatternNode:
    """Pattern for an Apply node: an Op (or OpPattern) head plus patterns for its inputs."""

    __slots__ = ("inputs", "op_match")

    def __init__(self, op_match, inputs: tuple):
        self.op_match = op_match
        self.inputs = inputs

    def __repr__(self) -> str:
        return f"PatternNode({self.op_match!r}, {self.inputs!r})"


type PatternElement = PatternVar | PatternNode | Variable | Any


def convert_strs_to_vars(
    x: tuple | str | dict | OpPattern,
    var_map: dict[str, PatternVar] | None = None,
):
    r"""Convert tuples and strings to pattern trees and logic variables, respectively.

    Constrained logic variables are specified via `dict`s with the keys
    `"pattern"`, which specifies the logic variable as a string, and
    `"constraint"`, which provides the `Callable` constraint.
    """
    if var_map is None:
        var_map = {}

    def _convert(y, op_prop: bool = False):
        if isinstance(y, str):
            v = var_map.get(y)
            if v is None:
                v = PatternVar(token=y)
                var_map[y] = v
            return v
        if isinstance(y, LiteralString):
            return y.value
        if isinstance(y, dict):
            pattern = y["pattern"]
            if not isinstance(pattern, str):
                raise TypeError(
                    "Constraints can only be assigned to logic variables (i.e. strings)"
                )
            constraint = y["constraint"]
            v = var_map.get(pattern)
            if v is None:
                v = PatternVar(token=pattern, constraint=constraint)
                var_map[pattern] = v
            elif v.constraint is None:
                v.constraint = constraint
            elif v.constraint is not constraint:
                raise ValueError(
                    f"Token {pattern!r} is used with two different constraints"
                )
            return v
        if isinstance(y, OpPattern):
            new_params = tuple((k, _convert(v, op_prop=True)) for k, v in y.parameters)
            return OpPattern(y.op_type, new_params)
        if isinstance(y, tuple):
            head, *rest = y
            head_converted = (
                _convert(head, op_prop=True) if isinstance(head, OpPattern) else head
            )
            if not isinstance(head_converted, Op | OpPattern):
                raise TypeError(
                    "Pattern tuples must start with an Op instance or OpPattern; "
                    f"got {head!r} of type {type(head)}"
                )
            children = tuple(_convert(e) for e in rest)
            return PatternNode(head_converted, children)
        if (not op_prop) and isinstance(y, Number | np.ndarray):
            # If we are converting an Op property, we don't want to convert numbers to PyTensor constants
            from pytensor.tensor import as_tensor_variable

            return as_tensor_variable(y)  # type: ignore[arg-type]
        return y

    return _convert(x)


def match_pattern(
    pattern: PatternNode,
    node,
    subs: dict[PatternVar, Any] | None = None,
):
    """Match ``pattern`` against an Apply ``node``.

    Returns a dict mapping pattern variables to the matched values, or None if
    there is no match. A provided ``subs`` mapping seeds the match and is never mutated.
    """
    subs = {} if subs is None else dict(subs)
    if _match_node(pattern, node, subs):
        return subs
    return None


def _match_node(
    pattern: PatternNode,
    node,
    subs: dict[PatternVar, Any],
) -> bool:
    """Match a PatternNode against an Apply node, binding into ``subs`` as it recurses."""
    op_match = pattern.op_match
    node_op = node.op
    if isinstance(op_match, OpPattern):
        if not isinstance(node_op, op_match.op_type):
            return False
        if not _bind_op_parameters(op_match, node_op, subs):
            return False
    elif node_op != op_match:
        return False

    if len(pattern.inputs) != len(node.inputs):
        return False

    for sub_pat, sub_var in zip(pattern.inputs, node.inputs):
        if not _match_element(sub_pat, sub_var, subs):
            return False
    return True


def _match_element(
    pattern,
    var,
    subs: dict[PatternVar, Any],
) -> bool:
    """Match a single pattern element (var, nested node, Variable or raw value) against a variable."""
    if isinstance(pattern, PatternVar):
        return _bind_var(pattern, var, subs)

    if isinstance(pattern, PatternNode):
        if var.owner is None:
            return False
        return _match_node(pattern, var.owner, subs)

    if isinstance(pattern, Variable):
        if isinstance(pattern, Constant) and isinstance(var, Constant):
            return np.array_equiv(pattern.data, var.data)
        return pattern is var

    if isinstance(var, Constant):
        try:
            return bool(np.array_equiv(pattern, var.data))
        except Exception:
            return False
    # A raw (non-Variable) pattern value can never equal a non-Constant Variable
    return False


def _bind_var(pat_var: PatternVar, value, subs: dict[PatternVar, Any]) -> bool:
    """Bind a PatternVar, checking its constraint on first bind and value agreement afterwards."""
    existing = subs.get(pat_var, _MISSING)
    if existing is not _MISSING:
        return _values_equal(existing, value)
    if pat_var.constraint is not None and not pat_var.constraint(value):
        return False
    subs[pat_var] = value
    return True


def _values_equal(a, b) -> bool:
    """Whether two values bound to the same pattern token agree.

    Variables compare by identity, except Constants which compare by data;
    other values (Op parameters) compare by ``==``.
    """
    if a is b:
        return True
    if isinstance(a, Variable) or isinstance(b, Variable):
        if isinstance(a, Constant) and isinstance(b, Constant):
            return bool(np.array_equiv(a.data, b.data))
        # Non-constant Variables compare by identity, which already failed above
        return False
    # Non-symbolic values (Op parameters) compare by equality
    return bool(a == b)


def _bind_op_parameters(op_pat: OpPattern, op: Op, subs: dict[PatternVar, Any]) -> bool:
    """Match OpPattern parameters against an Op, binding PatternVar parameters into ``subs``."""
    for key, param in op_pat.parameters:
        op_val = getattr(op, key)
        if isinstance(param, PatternVar):
            if not _bind_var(param, op_val, subs):
                return False
        elif isinstance(param, OpPattern):
            if not isinstance(op_val, param.op_type):
                return False
            if not _bind_op_parameters(param, op_val, subs):
                return False
        else:
            if op_val != param:
                return False
    return True


def reify_pattern(pattern, subs: Mapping[PatternVar, Any]):
    """Build the replacement Variable described by ``pattern`` using the bindings of a match.

    OpPatterns reify to concrete Op instances by instantiating their op_type with the
    (recursively reified) parameters.
    """
    if isinstance(pattern, PatternVar):
        try:
            return subs[pattern]
        except KeyError:
            raise ValueError(
                f"Output pattern references unbound variable {pattern.token!r}"
            )

    if isinstance(pattern, PatternNode):
        op_match = pattern.op_match
        op = (
            reify_pattern(op_match, subs)
            if isinstance(op_match, OpPattern)
            else op_match
        )
        inputs = [reify_pattern(p, subs) for p in pattern.inputs]
        return op.make_node(*inputs).default_output()

    if isinstance(pattern, OpPattern):
        op_type = pattern.op_type
        if not isinstance(op_type, type):
            raise ValueError(
                f"Cannot instantiate OpPattern with non-type op_type {op_type!r}"
            )
        params = {}
        for key, val in pattern.parameters:
            if isinstance(val, PatternVar):
                params[key] = subs[val]
            elif isinstance(val, OpPattern):
                params[key] = reify_pattern(val, subs)
            else:
                params[key] = val
        return op_type(**params)

    return pattern
