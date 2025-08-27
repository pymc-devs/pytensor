"""
If you have two expressions containing unification variables, these expressions
can be "unified" if there exists an assignment to all unification variables
such that the two expressions are equal.

For instance, [5, A, B] and [A, C, 9] can be unified if A=C=5 and B=9,
yielding [5, 5, 9].
[5, [A, B]] and [A, [1, 2]] cannot be unified because there is no value for A
that satisfies the constraints. That's useful for pattern matching.

"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Any

import numpy as np
from cons.core import ConsError, _car, _cdr
from etuples import apply, etuple, etuplize
from etuples.core import ExpressionTuple
from unification.core import _unify, assoc
from unification.utils import transitive_get as walk
from unification.variable import Var, isvar, var

from pytensor.graph.basic import Constant, Variable
from pytensor.graph.op import Op
from pytensor.graph.type import Type


def eval_if_etuple(x):
    if isinstance(x, ExpressionTuple):
        return x.evaled_obj
    return x


class ConstrainedVar(Var):
    """A logical variable with a constraint.

    These will unify with other `Var`s regardless of the constraints.
    """

    __slots__ = ("constraint",)

    def __new__(cls, constraint, token=None, prefix=""):
        if token is None:
            token = f"{prefix}_{Var._id}"
            Var._id += 1

        key = (token, constraint)
        obj = cls._refs.get(key, None)

        if obj is None:
            obj = object.__new__(cls)
            obj.token = token
            obj.constraint = constraint
            cls._refs[key] = obj

        return obj

    def __eq__(self, other):
        if type(self) is type(other):
            return self.token is other.token and self.constraint == other.constraint
        return NotImplemented

    def __hash__(self):
        return hash((type(self), self.token, self.constraint))

    def __str__(self):
        return f"~{self.token} [{self.constraint}]"

    def __repr__(self):
        return f"{type(self).__name__}({self.constraint!r}, {self.token})"


def car_Variable(x):
    if x.owner:
        return x.owner.op
    else:
        raise ConsError("Not a cons pair.")


_car.add((Variable,), car_Variable)


def cdr_Variable(x):
    if x.owner:
        x_e = etuple(_car(x), *x.owner.inputs, evaled_obj=x)
    else:
        raise ConsError("Not a cons pair.")

    return x_e[1:]


_cdr.add((Variable,), cdr_Variable)


def car_Op(x):
    if hasattr(x, "__props__"):
        return type(x)

    raise ConsError("Not a cons pair.")


_car.add((Op,), car_Op)


def cdr_Op(x):
    if not hasattr(x, "__props__"):
        raise ConsError("Not a cons pair.")

    x_e = etuple(
        _car(x),
        *[getattr(x, p) for p in getattr(x, "__props__", ())],
        evaled_obj=x,
    )
    return x_e[1:]


_cdr.add((Op,), cdr_Op)


def car_Type(x):
    return type(x)


_car.add((Type,), car_Type)


def cdr_Type(x):
    x_e = etuple(
        _car(x), *[getattr(x, p) for p in getattr(x, "__props__", ())], evaled_obj=x
    )
    return x_e[1:]


_cdr.add((Type,), cdr_Type)


def apply_Op_ExpressionTuple(op, etuple_arg):
    res = op.make_node(*etuple_arg)

    try:
        return res.default_output()
    except ValueError:
        return res.outputs


apply.add((Op, ExpressionTuple), apply_Op_ExpressionTuple)


def _unify_etuplize_first_arg(u, v, s):
    try:
        u_et = etuplize(u, shallow=True)
        yield _unify(u_et, v, s)
    except TypeError:
        yield False
        return


_unify.add((Op, ExpressionTuple, Mapping), _unify_etuplize_first_arg)
_unify.add(
    (ExpressionTuple, Op, Mapping), lambda u, v, s: _unify_etuplize_first_arg(v, u, s)
)

_unify.add((Type, ExpressionTuple, Mapping), _unify_etuplize_first_arg)
_unify.add(
    (ExpressionTuple, Type, Mapping), lambda u, v, s: _unify_etuplize_first_arg(v, u, s)
)


def _unify_Variable_Variable(u, v, s):
    # Avoid converting to `etuple`s, when possible
    if u == v:
        yield s
        return

    if u.owner is None and v.owner is None:
        yield False
        return

    yield _unify(
        etuplize(u, shallow=True) if u.owner else u,
        etuplize(v, shallow=True) if v.owner else v,
        s,
    )


_unify.add((Variable, Variable, Mapping), _unify_Variable_Variable)


def _unify_Constant_Constant(u, v, s):
    # XXX: This ignores shape and type differences.  It's only implemented this
    # way for backward compatibility
    if np.array_equiv(u.data, v.data):
        yield s
    else:
        yield False


_unify.add((Constant, Constant, Mapping), _unify_Constant_Constant)


def _unify_Variable_ExpressionTuple(u, v, s):
    # `Constant`s are "atomic"
    if u.owner is None:
        yield False
        return

    yield _unify(etuplize(u, shallow=True), v, s)


_unify.add(
    (Variable, ExpressionTuple, Mapping),
    _unify_Variable_ExpressionTuple,
)
_unify.add(
    (ExpressionTuple, Variable, Mapping),
    lambda u, v, s: _unify_Variable_ExpressionTuple(v, u, s),
)


@_unify.register(ConstrainedVar, (ConstrainedVar, Var, object), Mapping)
def _unify_ConstrainedVar_object(u, v, s):
    u_w = walk(u, s)

    if isvar(v):
        v_w = walk(v, s)
    else:
        v_w = v

    if u_w == v_w:
        yield s
    elif isvar(u_w):
        if (
            not isvar(v_w)
            and isinstance(u_w, ConstrainedVar)
            and not u_w.constraint(eval_if_etuple(v_w))
        ):
            yield False
            return
        yield assoc(s, u_w, v_w)
    elif isvar(v_w):
        if (
            not isvar(u_w)
            and isinstance(v_w, ConstrainedVar)
            and not v_w.constraint(eval_if_etuple(u_w))
        ):
            yield False
            return
        yield assoc(s, v_w, u_w)
    else:
        yield _unify(u_w, v_w, s)


_unify.add((object, ConstrainedVar, Mapping), _unify_ConstrainedVar_object)


@dataclass(frozen=True)
class LiteralString:
    value: str


class OpInstance:
    """Class that can be unified with Op instances of a given type and parameters.

    An op instance is unified as long as the parameters specified in the OpInstance can be unified as well.
    Parameters that are not specified in the OpInstance are ignored during unification.

    This is needed because some Ops can be complex to parametrize fully,
    and not all parameters are relevant for a given pattern.

    Examples
    --------

    .. testcode::

        from unification import var, unify
        from etuples import etuple

        import pytensor.tensor as pt
        from pytensor.graph.rewriting.unify import OpInstance
        from pytensor.tensor.blockwise import Blockwise
        from pytensor.tensor.slinalg import Solve

        A = var("A")
        b = var("b")
        pattern = etuple(
            OpInstance(Blockwise, core_op=OpInstance(Solve, assume_a="gen")), A, b
        )

        A_pt = pt.tensor3("A")
        b_pt = pt.tensor3("b")
        out1 = pt.linalg.solve(A_pt, b_pt)
        out2 = pt.linalg.solve(A_pt, b_pt, assume_a="pos")

        assert unify(pattern, out1) == {A: A_pt, b: b_pt}
        assert unify(pattern, out2) is False

        assume_a = var("assume_a")
        pattern = etuple(
            OpInstance(Blockwise, core_op=OpInstance(Solve, assume_a=assume_a)),
            A,
            b,
        )
        assert unify(pattern, out1) == {A: A_pt, b: b_pt, assume_a: "gen"}
        assert unify(pattern, out2) == {A: A_pt, b: b_pt, assume_a: "pos"}


    """

    def __init__(
        self,
        op_type: type[Op],
        parameters: dict[str, Any] | Sequence[tuple[str, Any]] | None = None,
        **kwargs,
    ):
        if not (isinstance(op_type, type) and issubclass(op_type, Op)):
            raise TypeError(f"Invalid op_type {op_type}. Expected type(Op)")

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
        self.parameters = parameters

    def __str__(self):
        return f"{self.op_type.__name__}({self.op_type}, {', '.join(f'{k}={v}' for k, v in self.parameters)})"


def _unify_parametrized_op(v: Op, u: OpInstance, s: Mapping):
    if not isinstance(v, u.op_type):
        yield False
        return
    for parameter_key, parameter_pattern in u.parameters:
        parameter_value = getattr(v, parameter_key)
        s = yield _unify(parameter_value, parameter_pattern, s)
        if s is False:
            yield False
            return
    yield s


_unify.add((Op, OpInstance, Mapping), _unify_parametrized_op)


def convert_strs_to_vars(
    x: tuple | str | dict, var_map: dict[str, Var] | None = None
) -> ExpressionTuple | Var:
    r"""Convert tuples and strings to `etuple`\s and logic variables, respectively.

    Constrained logic variables are specified via `dict`s with the keys
    `"pattern"`, which specifies the logic variable as a string, and
    `"constraint"`, which provides the `Callable` constraint.
    """
    if var_map is None:
        var_map = {}

    def _convert(y, op_prop=False):
        if isinstance(y, str):
            v = var_map.get(y, var(y))
            var_map[y] = v
            return v
        if isinstance(y, LiteralString):
            return y.value
        elif isinstance(y, dict):
            pattern = y["pattern"]
            if not isinstance(pattern, str):
                raise TypeError(
                    "Constraints can only be assigned to logic variables (i.e. strings)"
                )
            constraint = y["constraint"]
            v = var_map.get(pattern, ConstrainedVar(constraint, pattern))
            var_map[pattern] = v
            return v
        elif isinstance(y, tuple):
            return etuple(*(_convert(e, op_prop=op_prop) for e in y))
        elif isinstance(y, OpInstance):
            return OpInstance(
                y.op_type,
                {k: _convert(v, op_prop=True) for k, v in y.parameters},
            )
        elif (not op_prop) and isinstance(y, Number | np.ndarray):
            # If we are converting an Op property, we don't want to convert numbers to PyTensor constants
            from pytensor.tensor import as_tensor_variable

            return as_tensor_variable(y)
        return y

    return _convert(x)
