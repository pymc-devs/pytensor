"""Tests for the specialized pattern matcher in :mod:`pytensor.graph.rewriting.unify`."""

import numpy as np

import pytensor.tensor as pt
from pytensor.graph.basic import Apply, Constant
from pytensor.graph.op import Op
from pytensor.graph.rewriting.unify import (
    Asterisk,
    ConstrainedVar,
    LiteralString,
    OpPattern,
    PatternNode,
    PatternVar,
    convert_strs_to_vars,
    match_pattern,
    reify_pattern,
)


class CustomOp(Op):
    __props__ = ("a",)

    def __init__(self, a):
        self.a = a

    def make_node(self, *inputs):
        return Apply(self, list(inputs), [pt.vector()])

    def perform(self, node, inputs, outputs):
        raise NotImplementedError()


class CustomOpNoProps(Op):
    def __init__(self, a):
        self.a = a

    def __eq__(self, other):
        return type(self) is type(other) and self.a == other.a

    def __hash__(self):
        return hash((type(self), self.a))

    def make_node(self, *inputs):
        return Apply(self, list(inputs), [pt.vector()])

    def perform(self, node, inputs, outputs):
        raise NotImplementedError()


def test_convert_strs_to_vars_strings():
    res = convert_strs_to_vars("a")
    assert isinstance(res, PatternVar)
    assert res.token == "a"


def test_convert_strs_to_vars_shared_var_map():
    var_map = {}
    res = convert_strs_to_vars((pt.add, "x", "x"), var_map=var_map)
    assert isinstance(res, PatternNode)
    assert res.inputs[0] is res.inputs[1]
    assert res.inputs[0].token == "x"


def test_convert_strs_to_vars_constraint():
    def is_int(v):
        return isinstance(v, int)

    var_map = {}
    res = convert_strs_to_vars(
        (pt.add, {"pattern": "x", "constraint": is_int}, "x"),
        var_map=var_map,
    )
    # Both references to "x" are the same PatternVar
    assert res.inputs[0] is res.inputs[1]
    assert res.inputs[0].constraint is is_int


def test_convert_strs_to_vars_literal_string():
    res = convert_strs_to_vars(LiteralString("hello"))
    assert res == "hello"


def test_convert_strs_to_vars_numeric_constant():
    val = np.r_[1, 2]
    res = convert_strs_to_vars((pt.add, val, "x"))
    assert isinstance(res.inputs[0], Constant)
    assert np.array_equal(res.inputs[0].data, val)


def test_match_simple_chain():
    x = pt.vector("x")
    out = pt.log(pt.exp(x))

    pat = convert_strs_to_vars((pt.log, (pt.exp, "x")))
    subs = match_pattern(pat, out.owner)
    assert subs is not None
    [(_, bound)] = subs.items()
    assert bound is x


def test_match_repeated_var_same_value():
    x = pt.vector("x")
    out = pt.mul(x, x)
    pat = convert_strs_to_vars((pt.mul, "z", "z"))
    subs = match_pattern(pat, out.owner)
    assert subs is not None
    [(pat_var, bound)] = subs.items()
    assert pat_var.token == "z"
    assert bound is x


def test_match_repeated_var_different_value_fails():
    x = pt.vector("x")
    y = pt.vector("y")
    out = pt.mul(x, y)
    pat = convert_strs_to_vars((pt.mul, "z", "z"))
    assert match_pattern(pat, out.owner) is None


def test_match_with_constraint():
    def is_scalar(v):
        return all(s == 1 for s in v.type.shape)

    x = pt.vector("x")
    s = pt.scalar("s")
    out = pt.sub(s, x)
    pat = convert_strs_to_vars((pt.sub, {"pattern": "a", "constraint": is_scalar}, "x"))
    assert match_pattern(pat, out.owner) is not None

    y = pt.vector("y")
    out2 = pt.sub(y, x)
    assert match_pattern(pat, out2.owner) is None


def test_match_op_pattern_isinstance():
    x = pt.matrix("x")
    out = pt.sum(x)

    from pytensor.tensor.elemwise import CAReduce

    pat = convert_strs_to_vars((OpPattern(CAReduce, axis=None), "x"))
    subs = match_pattern(pat, out.owner)
    assert subs is not None
    [(_, bound)] = subs.items()
    assert bound is x


def test_match_op_pattern_with_var_param():
    from pytensor.tensor.elemwise import CAReduce

    x = pt.matrix("x")
    out = pt.sum(x)
    pat = convert_strs_to_vars((OpPattern(CAReduce, scalar_op="sop", axis=None), "x"))
    subs = match_pattern(pat, out.owner)
    assert subs is not None
    by_name = {p.token: v for p, v in subs.items()}
    assert by_name["x"] is x
    assert by_name["sop"] is out.owner.op.scalar_op


def test_match_op_pattern_param_mismatch():
    from pytensor.tensor.elemwise import CAReduce

    x = pt.matrix("x")
    out = pt.sum(x, axis=0)  # axis=(0,), not None
    pat = convert_strs_to_vars((OpPattern(CAReduce, axis=None), "x"))
    assert match_pattern(pat, out.owner) is None


def test_match_nested_op_pattern():
    from pytensor.tensor.blockwise import Blockwise

    A = pt.tensor3("A")
    b = pt.tensor3("b")
    out = pt.linalg.solve(A, b)

    from pytensor.tensor.slinalg import Solve

    pat = convert_strs_to_vars(
        (
            OpPattern(
                Blockwise,
                core_op=OpPattern(Solve, assume_a=LiteralString("gen")),
            ),
            "A",
            "b",
        )
    )
    res = match_pattern(pat, out.owner)
    assert res is not None


def test_match_literal_int_constant():
    x = pt.scalar("x")
    out = pt.add(x, pt.as_tensor_variable(2.0))
    pat = convert_strs_to_vars((pt.add, "x", 2.0))
    assert match_pattern(pat, out.owner) is not None


def test_match_commutative_add_swapped():
    x = pt.scalar("x")
    out_left = pt.add(pt.as_tensor_variable(1.0), x)
    out_right = pt.add(x, pt.as_tensor_variable(1.0))
    pat = convert_strs_to_vars((pt.add, 1.0, "x"))

    s1 = match_pattern(pat, out_left.owner)
    s2 = match_pattern(pat, out_right.owner)
    assert s1 is not None
    assert s2 is not None
    [(_, b1)] = s1.items()
    [(_, b2)] = s2.items()
    assert b1 is x
    assert b2 is x


def test_match_commutative_does_not_apply_to_sub():
    x = pt.scalar("x")
    out = pt.sub(pt.as_tensor_variable(1.0), x)
    pat_swapped = convert_strs_to_vars((pt.sub, "x", 1.0))
    assert match_pattern(pat_swapped, out.owner) is None


def test_match_variadic_asterisk():
    a = pt.vector("a")
    b = pt.vector("b")
    c = pt.vector("c")
    out = pt.add(a, b, c)
    pat = convert_strs_to_vars((pt.add, "first", Asterisk("rest")))

    subs = match_pattern(pat, out.owner)
    assert subs is not None
    by_name = {
        (p.token if isinstance(p, (PatternVar, Asterisk)) else p): v
        for p, v in subs.items()
    }
    assert by_name["first"] is a
    assert by_name["rest"] == (b, c)


def test_match_variadic_too_few_inputs():
    a = pt.vector("a")
    out = pt.exp(a)
    pat = convert_strs_to_vars((pt.exp, "x", "y", Asterisk("rest")))
    assert match_pattern(pat, out.owner) is None


def test_match_variadic_preserves_input_order_under_commutative_backtrack():
    a = pt.vector("a", dtype="float32")
    b = pt.vector("b", dtype="float64")
    c = pt.vector("c", dtype="float32")

    pat = convert_strs_to_vars(
        (
            pt.add,
            {"pattern": "first", "constraint": lambda v: v.dtype == "float64"},
            Asterisk("rest"),
        )
    )

    s1 = match_pattern(pat, pt.add(a, b, c).owner)
    n1 = {k.token: v for k, v in s1.items()}
    assert n1["first"] is b
    assert n1["rest"] == (a, c)

    s2 = match_pattern(pat, pt.add(c, a, b).owner)
    n2 = {k.token: v for k, v in s2.items()}
    assert n2["first"] is b
    assert n2["rest"] == (c, a)


def test_reify_variadic():
    a = pt.vector("a")
    b = pt.vector("b")
    c = pt.vector("c")
    rest = Asterisk("rest")
    var_map = {}
    in_pat = convert_strs_to_vars((pt.add, "first", rest), var_map=var_map)
    subs = match_pattern(in_pat, pt.add(a, b, c).owner)
    assert subs is not None

    out_pat = convert_strs_to_vars((pt.mul, "first", rest), var_map=var_map)
    result = reify_pattern(out_pat, subs)
    assert result.owner.op == pt.mul
    assert result.owner.inputs == [a, b, c]


def test_reify_simple():
    x = pt.vector("x")
    pv = PatternVar("x")
    pat = PatternNode(pt.log, (pv,))
    out = reify_pattern(pat, {pv: x})
    assert out.owner is not None
    assert out.owner.op == pt.log


def test_reify_op_pattern_to_op():
    import pytensor.scalar as ps
    from pytensor.tensor.elemwise import CAReduce

    pv_x = PatternVar("x")
    pv_scalar = PatternVar("sop")
    pat = PatternNode(
        OpPattern(CAReduce, parameters=(("axis", None), ("scalar_op", pv_scalar))),
        (pv_x,),
    )
    x = pt.matrix("x")
    out = reify_pattern(pat, {pv_x: x, pv_scalar: ps.add})
    assert out.owner is not None
    assert out.owner.op.axis is None


def test_constrained_var_back_compat():
    cvar = ConstrainedVar(lambda v: isinstance(v, int), "tok")
    assert isinstance(cvar, PatternVar)
    assert cvar.constraint(1)
    assert not cvar.constraint("x")
