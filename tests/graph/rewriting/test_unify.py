import numpy as np
import pytest
from cons import car, cdr
from cons.core import ConsError
from etuples import apply, etuple, etuplize
from etuples.core import ExpressionTuple
from unification import reify, unify, var
from unification.variable import Var

import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor.graph.basic import Apply, Constant, equal_computations
from pytensor.graph.op import Op
from pytensor.graph.rewriting.unify import ConstrainedVar, convert_strs_to_vars
from pytensor.tensor.type import TensorType
from tests.graph.utils import MyType


class CustomOp(Op):
    __props__ = ("a",)

    def __init__(self, a):
        self.a = a

    def make_node(self, *inputs):
        return Apply(self, list(inputs), [pt.vector()])

    def perform(self, node, inputs, outputs):
        raise NotImplementedError()


class CustomOpNoPropsNoEq(Op):
    def __init__(self, a):
        self.a = a

    def make_node(self, *inputs):
        return Apply(self, list(inputs), [pt.vector()])

    def perform(self, node, inputs, outputs):
        raise NotImplementedError()


class CustomOpNoProps(CustomOpNoPropsNoEq):
    def __eq__(self, other):
        return type(self) is type(other) and self.a == other.a

    def __hash__(self):
        return hash((type(self), self.a))


def test_cons():
    x_pt = pt.vector("x")
    y_pt = pt.vector("y")

    z_pt = x_pt + y_pt

    res = car(z_pt)
    assert res == z_pt.owner.op

    res = cdr(z_pt)
    assert res == [x_pt, y_pt]

    with pytest.raises(ConsError):
        car(x_pt)

    with pytest.raises(ConsError):
        cdr(x_pt)

    op1 = CustomOp(1)

    assert car(op1) == CustomOp
    assert cdr(op1) == (1,)

    tt1 = TensorType("float32", shape=(1, None))

    assert car(tt1) == TensorType
    assert cdr(tt1) == ("float32", (1, None))

    op1_np = CustomOpNoProps(1)

    with pytest.raises(ConsError):
        car(op1_np)

    with pytest.raises(ConsError):
        cdr(op1_np)

    atype_pt = ps.float64
    car_res = car(atype_pt)
    cdr_res = cdr(atype_pt)
    assert car_res is type(atype_pt)
    assert cdr_res == [atype_pt.dtype]

    atype_pt = pt.lvector
    car_res = car(atype_pt)
    cdr_res = cdr(atype_pt)
    assert car_res is type(atype_pt)
    assert cdr_res == [atype_pt.dtype, atype_pt.shape]


def test_etuples():
    x_pt = pt.vector("x")
    y_pt = pt.vector("y")

    z_pt = etuple(x_pt, y_pt)

    res = apply(pt.add, z_pt)

    assert res.owner.op == pt.add
    assert res.owner.inputs == [x_pt, y_pt]

    w_pt = etuple(pt.add, x_pt, y_pt)

    res = w_pt.evaled_obj
    assert res.owner.op == pt.add
    assert res.owner.inputs == [x_pt, y_pt]

    # This `Op` doesn't expand into an `etuple` (i.e. it's "atomic")
    op1_np = CustomOpNoProps(1)

    res = apply(op1_np, z_pt)
    assert res.owner.op == op1_np

    q_pt = op1_np(x_pt, y_pt)
    res = etuplize(q_pt)
    assert res[0] == op1_np

    with pytest.raises(TypeError):
        etuplize(op1_np)

    class MyMultiOutOp(Op):
        def make_node(self, *inputs):
            outputs = [MyType()(), MyType()()]
            return Apply(self, list(inputs), outputs)

        def perform(self, node, inputs, outputs):
            outputs[0] = np.array(inputs[0])
            outputs[1] = np.array(inputs[0])

    x_pt = pt.vector("x")
    op1_np = MyMultiOutOp()
    res = apply(op1_np, etuple(x_pt))
    assert len(res) == 2
    assert res[0].owner.op == op1_np
    assert res[1].owner.op == op1_np


def test_unify_Variable():
    x_pt = pt.vector("x")
    y_pt = pt.vector("y")

    z_pt = x_pt + y_pt

    # `Variable`, `Variable`
    s = unify(z_pt, z_pt)
    assert s == {}

    # These `Variable`s have no owners
    v1 = MyType()()
    v2 = MyType()()

    assert v1 != v2

    s = unify(v1, v2)
    assert s is False

    op_lv = var()
    z_ppt_et = etuple(op_lv, x_pt, y_pt)

    # `Variable`, `ExpressionTuple`
    s = unify(z_pt, z_ppt_et, {})

    assert op_lv in s
    assert s[op_lv] == z_pt.owner.op

    res = reify(z_ppt_et, s)

    assert isinstance(res, ExpressionTuple)
    assert equal_computations([res.evaled_obj], [z_pt])

    z_et = etuple(pt.add, x_pt, y_pt)

    # `ExpressionTuple`, `ExpressionTuple`
    s = unify(z_et, z_ppt_et, {})

    assert op_lv in s
    assert s[op_lv] == z_et[0]

    res = reify(z_ppt_et, s)

    assert isinstance(res, ExpressionTuple)
    assert equal_computations([res.evaled_obj], [z_et.evaled_obj])

    # `ExpressionTuple`, `Variable`
    s = unify(z_et, x_pt, {})
    assert s is False

    # This `Op` doesn't expand into an `ExpressionTuple`
    op1_np = CustomOpNoProps(1)

    q_pt = op1_np(x_pt, y_pt)

    a_lv = var()
    b_lv = var()
    # `Variable`, `ExpressionTuple`
    s = unify(q_pt, etuple(op1_np, a_lv, b_lv))

    assert s[a_lv] == x_pt
    assert s[b_lv] == y_pt


def test_unify_Op():
    # These `Op`s expand into `ExpressionTuple`s
    op1 = CustomOp(1)
    op2 = CustomOp(1)

    # `Op`, `Op`
    s = unify(op1, op2)
    assert s == {}

    # `ExpressionTuple`, `Op`
    s = unify(etuplize(op1), op2)
    assert s == {}

    # These `Op`s don't expand into `ExpressionTuple`s
    op1_np = CustomOpNoProps(1)
    op2_np = CustomOpNoProps(1)

    s = unify(op1_np, op2_np)
    assert s == {}

    # Same, but this one also doesn't implement `__eq__`
    op1_np_neq = CustomOpNoPropsNoEq(1)

    s = unify(op1_np_neq, etuplize(op1))
    assert s is False


def test_unify_Constant():
    # Make sure `Constant` unification works
    c1_pt = pt.as_tensor(np.r_[1, 2])
    c2_pt = pt.as_tensor(np.r_[1, 2])

    # `Constant`, `Constant`
    s = unify(c1_pt, c2_pt)
    assert s == {}


def test_unify_Type():
    t1 = TensorType(np.float64, shape=(1, None))
    t2 = TensorType(np.float64, shape=(1, None))

    # `Type`, `Type`
    s = unify(t1, t2)
    assert s == {}

    # `Type`, `ExpressionTuple`
    s = unify(t1, etuple(TensorType, "float64", (1, None)))
    assert s == {}

    from pytensor.scalar.basic import ScalarType

    st1 = ScalarType(np.float64)
    st2 = ScalarType(np.float64)

    s = unify(st1, st2)
    assert s == {}


def test_ConstrainedVar():
    cvar = ConstrainedVar(lambda x: isinstance(x, str))

    assert repr(cvar).startswith("ConstrainedVar(")
    assert repr(cvar).endswith(f", {cvar.token})")

    s = unify(cvar, 1)
    assert s is False

    s = unify(1, cvar)
    assert s is False

    s = unify(cvar, "hi")
    assert s[cvar] == "hi"

    s = unify("hi", cvar)
    assert s[cvar] == "hi"

    x_lv = var()
    s = unify(cvar, x_lv)
    assert s == {cvar: x_lv}

    s = unify(cvar, x_lv, {x_lv: "hi"})
    assert s[cvar] == "hi"

    s = unify(x_lv, cvar, {x_lv: "hi"})
    assert s[cvar] == "hi"

    s_orig = {cvar: "hi", x_lv: "hi"}
    s = unify(x_lv, cvar, s_orig)
    assert s == s_orig

    s_orig = {cvar: "hi", x_lv: "bye"}
    s = unify(x_lv, cvar, s_orig)
    assert s is False

    x_pt = pt.vector("x")
    y_pt = pt.vector("y")
    op1_np = CustomOpNoProps(1)
    r_pt = etuple(op1_np, x_pt, y_pt)

    def constraint(x):
        return isinstance(x, tuple)

    a_lv = ConstrainedVar(constraint)
    res = reify(etuple(op1_np, a_lv), {a_lv: r_pt})

    assert res[1] == r_pt


def test_convert_strs_to_vars():
    res = convert_strs_to_vars("a")
    assert isinstance(res, Var)
    assert res.token == "a"

    x_pt = pt.vector()
    y_pt = pt.vector()
    res = convert_strs_to_vars((("a", x_pt), y_pt))
    assert res == etuple(etuple(var("a"), x_pt), y_pt)

    def constraint(x):
        return isinstance(x, str)

    res = convert_strs_to_vars(
        (({"pattern": "a", "constraint": constraint}, x_pt), y_pt)
    )
    assert res == etuple(etuple(ConstrainedVar(constraint, "a"), x_pt), y_pt)

    # Make sure constrained logic variables are the same across distinct uses
    # of their string names
    res = convert_strs_to_vars(({"pattern": "a", "constraint": constraint}, "a"))
    assert res[0] is res[1]

    var_map = {"a": var("a")}
    res = convert_strs_to_vars(("a",), var_map=var_map)
    assert res[0] is var_map["a"]

    # Make sure numbers and NumPy arrays are converted
    val = np.r_[1, 2]
    res = convert_strs_to_vars((val,))
    assert isinstance(res[0], Constant)
    assert np.array_equal(res[0].data, val)
