from pathlib import Path

import numpy as np
import pytest

import pytensor
from pytensor import scalar as ps
from pytensor.graph.basic import Apply
from pytensor.link.c.op import COp
from pytensor.link.c.type import CDataType, CEnumType, EnumList, EnumType
from pytensor.tensor.type import TensorType, continuous_dtypes


class ProdOp(COp):
    __props__ = ()

    def make_node(self, i):
        return Apply(self, [i], [CDataType("void *", "py_decref")()])

    def c_support_code(self, **kwargs):
        return """
void py_decref(void *p) {
Py_XDECREF((PyObject *)p);
}
"""

    def c_code(self, node, name, inps, outs, sub):
        return f"""
Py_XDECREF({outs[0]});
{outs[0]} = (void *){inps[0]};
Py_INCREF({inps[0]});
"""
        # FIXME: should it not be outs[0]?

    def c_code_cache_version(self):
        return (0,)

    def perform(self, *args, **kwargs):
        raise NotImplementedError()


class GetOp(COp):
    __props__ = ()

    def make_node(self, c):
        return Apply(self, [c], [TensorType("float32", shape=(None,))()])

    def c_support_code(self, **kwargs):
        return """
void py_decref(void *p) {
Py_XDECREF((PyObject *)p);
}
"""

    def c_code(self, node, name, inps, outs, sub):
        return f"""
Py_XDECREF({outs[0]});
{outs[0]} = (PyArrayObject *){inps[0]};
Py_INCREF({outs[0]});
"""

    def c_code_cache_version(self):
        return (0,)

    def perform(self, *args, **kwargs):
        raise NotImplementedError()


@pytest.mark.skipif(
    not pytensor.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_cdata():
    i = TensorType("float32", shape=(None,))()
    c = ProdOp()(i)
    i2 = GetOp()(c)
    mode = None
    if pytensor.config.mode == "FAST_COMPILE":
        mode = "FAST_RUN"

    # This should be a passthrough function for vectors
    f = pytensor.function([i], i2, mode=mode)

    v = np.random.standard_normal((9,)).astype("float32")

    v2 = f(v)
    assert (v2 == v).all()


class MyOpEnumList(COp):
    __props__ = ("op_chosen",)
    params_type = EnumList(
        ("ADD", "+"),
        ("SUB", "-"),
        ("MULTIPLY", "*"),
        ("DIVIDE", "/"),
        ctype="unsigned long long",
    )

    def __init__(self, choose_op):
        assert self.params_type.ADD == 0
        assert self.params_type.SUB == 1
        assert self.params_type.MULTIPLY == 2
        assert self.params_type.DIVIDE == 3
        assert self.params_type.fromalias("+") == self.params_type.ADD
        assert self.params_type.fromalias("-") == self.params_type.SUB
        assert self.params_type.fromalias("*") == self.params_type.MULTIPLY
        assert self.params_type.fromalias("/") == self.params_type.DIVIDE
        assert self.params_type.has_alias(choose_op)
        self.op_chosen = choose_op

    def get_params(self, node):
        return self.op_chosen

    def make_node(self, a, b):
        return Apply(self, [ps.as_scalar(a), ps.as_scalar(b)], [ps.float64()])

    def perform(self, node, inputs, outputs):
        op = self.params_type.filter(self.get_params(node))
        a, b = inputs
        (o,) = outputs
        if op == self.params_type.ADD:
            o[0] = a + b
        elif op == self.params_type.SUB:
            o[0] = a - b
        elif op == self.params_type.MULTIPLY:
            o[0] = a * b
        elif op == self.params_type.DIVIDE:
            if any(dtype in continuous_dtypes for dtype in (a.dtype, b.dtype)):
                o[0] = a / b
            else:
                o[0] = a // b
        else:
            raise NotImplementedError("Unknown op id " + str(op))
        o[0] = np.float64(o[0])

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, inputs, outputs, sub):
        op = sub["params"]
        o = outputs[0]
        a = inputs[0]
        b = inputs[1]
        fail = sub["fail"]
        return f"""
        switch({op}) {{
            case ADD:
                {o} = {a} + {b};
                break;
            case SUB:
                {o} = {a} - {b};
                break;
            case MULTIPLY:
                {o} = {a} * {b};
                break;
            case DIVIDE:
                {o} = {a} / {b};
                break;
            default:
                {{{fail}}}
                break;
        }}
        """


class MyOpCEnumType(COp):
    __props__ = ("python_value",)
    params_type = CEnumType(
        ("MILLION", "million"),
        ("BILLION", "billion"),
        ("TWO_BILLIONS", "two_billions"),
        ctype="size_t",
    )

    def c_header_dirs(self, **kwargs):
        return [Path(__file__).parent / "c_code"]

    def c_headers(self, **kwargs):
        return ["test_cenum.h"]

    def __init__(self, value_name):
        # As we see, Python values of constants are not related to real C values.
        assert self.params_type.MILLION == 0
        assert self.params_type.BILLION == 1
        assert self.params_type.TWO_BILLIONS == 2
        assert self.params_type.has_alias(value_name)
        self.python_value = self.params_type.fromalias(value_name)

    def get_params(self, node):
        return self.python_value

    def make_node(self):
        return Apply(self, [], [ps.uint32()])

    def perform(self, *args, **kwargs):
        raise NotImplementedError()

    def c_code_cache_version(self):
        return (3,)

    def c_code(self, node, name, inputs, outputs, sub):
        # params in C code will already contains expected C constant value.
        return f"""
        {outputs[0]} = {sub['params']};
        """


class TestEnumTypes:
    def test_enum_class(self):
        # Check that invalid enum name raises exception.
        for invalid_name in ("a", "_A", "0"):
            try:
                EnumList(invalid_name)
            except AttributeError:
                pass
            else:
                raise Exception("EnumList with invalid name should fail.")

            try:
                EnumType(**{invalid_name: 0})
            except AttributeError:
                pass
            else:
                raise Exception("EnumType with invalid name should fail.")

        # Check that invalid enum value raises exception.
        try:
            EnumType(INVALID_VALUE="string is not allowed.")
        except TypeError:
            pass
        else:
            raise Exception("EnumType with invalid value should fail.")

        # Check EnumType.
        e1 = EnumType(C1=True, C2=12, C3=True, C4=-1, C5=False, C6=0.0)
        e2 = EnumType(C1=1, C2=12, C3=1, C4=-1.0, C5=0.0, C6=0)
        assert e1 == e2
        assert not (e1 != e2)
        assert hash(e1) == hash(e2)
        # Check access to attributes.
        assert len((e1.ctype, e1.C1, e1.C2, e1.C3, e1.C4, e1.C5, e1.C6)) == 7

        # Check enum with aliases.
        e1 = EnumType(A=("alpha", 0), B=("beta", 1), C=2)
        e2 = EnumType(A=("alpha", 0), B=("beta", 1), C=2)
        e3 = EnumType(A=("a", 0), B=("beta", 1), C=2)
        assert e1 == e2
        assert e1 != e3
        assert e1.filter("beta") == e1.fromalias("beta") == e1.B == 1
        assert e1.filter("C") == e1.fromalias("C") == e1.C == 2

        # Check that invalid alias (same as a constant) raises exception.
        try:
            EnumList(("A", "a"), ("B", "B"))
        except TypeError:
            EnumList(("A", "a"), ("B", "b"))
        else:
            raise Exception(
                "Enum with an alias name equal to a constant name should fail."
            )

    def test_op_with_enumlist(self):
        a = ps.int32()
        b = ps.int32()
        c_add = MyOpEnumList("+")(a, b)
        c_sub = MyOpEnumList("-")(a, b)
        c_multiply = MyOpEnumList("*")(a, b)
        c_divide = MyOpEnumList("/")(a, b)
        f = pytensor.function([a, b], [c_add, c_sub, c_multiply, c_divide])
        va = 12
        vb = 15
        ref = [va + vb, va - vb, va * vb, va // vb]
        out = f(va, vb)
        assert ref == out, (ref, out)

    @pytest.mark.skipif(
        not pytensor.config.cxx,
        reason="G++ not available, so we need to skip this test.",
    )
    def test_op_with_cenumtype(self):
        million = MyOpCEnumType("million")()
        billion = MyOpCEnumType("billion")()
        two_billions = MyOpCEnumType("two_billions")()
        f = pytensor.function([], [million, billion, two_billions])
        val_million, val_billion, val_two_billions = f()
        assert val_million == 1000000
        assert val_billion == val_million * 1000
        assert val_two_billions == val_billion * 2

    @pytensor.config.change_flags(**{"cmodule__debug": True})
    def test_op_with_cenumtype_debug(self):
        self.test_op_with_cenumtype()
