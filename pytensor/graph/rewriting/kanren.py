from collections.abc import Callable, Iterator, Mapping


try:
    import numpy as np
    from cons.core import ConsError, _car, _cdr
    from etuples import apply, etuple, etuplize
    from etuples.core import ExpressionTuple
    from kanren import run
    from unification import var
    from unification.core import _unify, assoc
    from unification.utils import transitive_get as walk
    from unification.variable import Var, isvar
except ImportError as _err:
    raise ImportError(
        "pytensor.graph.rewriting.kanren requires the optional packages "
        "'logical-unification', 'kanren', 'etuples' and 'cons'. "
        "Install them with `pip install logical-unification kanren etuples cons`."
    ) from _err

from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.op import Op
from pytensor.graph.rewriting.basic import NodeRewriter
from pytensor.graph.rewriting.unify import (
    ConstrainedVar,
    OpPattern,
    PatternVar,
)
from pytensor.graph.type import Type


Var.register(PatternVar)


def eval_if_etuple(x):
    if isinstance(x, ExpressionTuple):
        return x.evaled_obj
    return x


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
            and u_w.constraint is not None
            and not u_w.constraint(eval_if_etuple(v_w))
        ):
            yield False
            return
        yield assoc(s, u_w, v_w)
    elif isvar(v_w):
        if (
            not isvar(u_w)
            and isinstance(v_w, ConstrainedVar)
            and v_w.constraint is not None
            and not v_w.constraint(eval_if_etuple(u_w))
        ):
            yield False
            return
        yield assoc(s, v_w, u_w)
    else:
        yield _unify(u_w, v_w, s)


_unify.add((object, ConstrainedVar, Mapping), _unify_ConstrainedVar_object)


def _unify_parametrized_op(v, u, s):
    if not isinstance(v, u.op_type):
        yield False
        return
    for parameter_key, parameter_pattern in u.parameters:
        parameter_value = getattr(v, parameter_key)
        new_s = yield _unify(parameter_value, parameter_pattern, s)
        if new_s is False:
            yield False
            return
        s = new_s
    yield s


_unify.add((Op, OpPattern, Mapping), _unify_parametrized_op)


class KanrenRelationSub(NodeRewriter):
    r"""A rewriter that uses `kanren` to match and replace terms.

    See `kanren <https://github.com/pythological/kanren>`__ for more information
    miniKanren and the API for constructing `kanren` goals.

    Example
    -------

    ..code-block:: python

        from kanren import eq, conso, var

        import pytensor.tensor as pt
        from pytensor.graph.rewriting.kanren import KanrenRelationSub


        def relation(in_lv, out_lv):
            # A `kanren` goal that changes `pt.log` terms to `pt.exp`
            cdr_lv = var()
            return eq(conso(pt.log, cdr_lv, in_lv),
                    conso(pt.exp, cdr_lv, out_lv))


        kanren_sub_opt = KanrenRelationSub(relation)

    """

    reentrant = True

    def __init__(
        self,
        kanren_relation: Callable[[Variable, Var], Callable],
        results_filter: Callable[[Iterator], list[ExpressionTuple | Variable] | None]
        | None = None,
        node_filter: Callable[[Apply], bool] = lambda x: True,
    ):
        r"""Create a `KanrenRelationSub`.

        Parameters
        ----------
        kanren_relation
            A function that takes an input graph and an output logic variable and
            returns a `kanren` goal.
        results_filter
            A function that takes the direct output of ``kanren.run(None, ...)``
            and returns a single result.  The default implementation returns
            the first result.
        node_filter
            A function taking a single node and returns ``True`` when the node
            should be processed.
        """
        if results_filter is None:

            def results_filter(
                x: Iterator,
            ) -> list[ExpressionTuple | Variable] | None:
                return next(x, None)

        self.kanren_relation = kanren_relation
        self.results_filter = results_filter
        self.node_filter = node_filter
        super().__init__()

    def transform(self, fgraph, node, enforce_tracks: bool = True):
        if self.node_filter(node) is False:
            return False

        try:
            input_expr = node.default_output()
        except ValueError:
            input_expr = node.outputs

        q = var()
        kanren_results = run(None, q, self.kanren_relation(input_expr, q))

        chosen_res = self.results_filter(kanren_results)  # type: ignore[arg-type]

        if chosen_res:
            if isinstance(chosen_res, list):
                new_outputs = [eval_if_etuple(v) for v in chosen_res]
            else:
                new_outputs = [eval_if_etuple(chosen_res)]  # type: ignore[unreachable]

            return new_outputs
        else:
            return False
