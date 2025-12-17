import numpy as np
import pytest

from pytensor import Out
from pytensor.compile import shared
from pytensor.compile.function import function
from pytensor.compile.mode import Mode
from pytensor.configdefaults import config
from pytensor.graph.basic import Apply, Constant, NominalVariable, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.link.basic import PerformLinker
from pytensor.link.c.basic import CLinker, DualLinker, OpWiseCLinker
from pytensor.link.c.op import COp
from pytensor.link.c.type import CType
from pytensor.link.vm import VMLinker
from pytensor.tensor.type import iscalar, matrix, vector
from tests.link.test_link import make_function


def as_variable(x):
    assert isinstance(x, Variable)
    return x


class TDouble(CType):
    def filter(self, data, strict=False, allow_downcast=False):
        return float(data)

    def c_declare(self, name, sub, check_input=True):
        return f"double {name}; void* {name}_bad_thing;"

    def c_init(self, name, sub):
        return f"""
        {name} = 0;
        {name}_bad_thing = malloc(100000);
        //printf("Initializing {name}\
");
        """

    def c_literal(self, data):
        return str(data)

    def c_extract(self, name, sub, check_input=True, **kwargs):
        fail = sub["fail"]
        return f"""
        if (!PyFloat_Check(py_{name})) {{
            PyErr_SetString(PyExc_TypeError, "not a double!");
            {fail}
        }}
        {name} = PyFloat_AsDouble(py_{name});
        {name}_bad_thing = NULL;
        //printf("Extracting {name}\\n");
        """

    def c_sync(self, name, sub):
        return f"""
        Py_XDECREF(py_{name});
        py_{name} = PyFloat_FromDouble({name});
        if (!py_{name})
            py_{name} = Py_None;
        //printf("Syncing {name}\
");
        """

    def c_cleanup(self, name, sub):
        return f"""
        //printf("Cleaning up {name}\
");
        if ({name}_bad_thing)
            free({name}_bad_thing);
        """

    def c_code_cache_version(self):
        return (1,)

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))


tdouble = TDouble()


def double(name):
    return Variable(tdouble, None, None, name=name)


class MyOp(COp):
    __props__ = ("nin", "name")

    def __init__(self, nin, name):
        self.nin = nin
        self.name = name

    def make_node(self, *inputs):
        assert len(inputs) == self.nin
        inputs = list(map(as_variable, inputs))
        for input in inputs:
            if input.type is not tdouble:
                raise Exception("Error 1")
        outputs = [double(self.name + "_R")]
        return Apply(self, inputs, outputs)

    def __str__(self):
        return self.name

    def perform(self, node, inputs, output_storage):
        (out,) = output_storage
        out[0] = self.impl(*inputs)

    def c_code_cache_version(self):
        return (1,)


# class Unary(MyOp):
#    def __init__(self):
#        MyOp.__init__(self, 1, self.__class__.__name__)


class Binary(MyOp):
    def __init__(self):
        MyOp.__init__(self, 2, self.__class__.__name__)


class Add(Binary):
    def c_code(self, node, name, inp, out, sub):
        x, y = inp
        (z,) = out
        return f"{z} = {x} + {y};"

    def impl(self, x, y):
        return x + y


add = Add()


class Sub(Binary):
    def c_code(self, node, name, inp, out, sub):
        x, y = inp
        (z,) = out
        return f"{z} = {x} - {y};"

    def impl(self, x, y):
        return x - y


sub = Sub()


class BadSub(Binary):
    def c_code(self, node, name, inp, out, sub):
        x, y = inp
        (z,) = out
        return f"{z} = {x} - {y};"

    def impl(self, x, y):
        return -10  # erroneous (most of the time)


bad_sub = BadSub()


class Mul(Binary):
    def c_code(self, node, name, inp, out, sub):
        x, y = inp
        (z,) = out
        return f"{z} = {x} * {y};"

    def impl(self, x, y):
        return x * y


mul = Mul()


class Div(Binary):
    def c_code(self, node, name, inp, out, sub):
        x, y = inp
        (z,) = out
        return f"{z} = {x} / {y};"

    def impl(self, x, y):
        return x / y


div = Div()


def inputs():
    x = double("x")
    y = double("y")
    z = double("z")
    return x, y, z


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_clinker_straightforward():
    x, y, z = inputs()
    e = add(mul(add(x, y), div(x, y)), bad_sub(bad_sub(x, y), z))
    lnk = CLinker().accept(FunctionGraph([x, y, z], [e]))
    fn = make_function(lnk)
    assert fn(2.0, 2.0, 2.0) == 2.0


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_cthunk_str():
    x = double("x")
    y = double("y")
    e = add(x, y)
    lnk = CLinker().accept(FunctionGraph([x, y], [e]))
    cthunk, _input_storage, _output_storage = lnk.make_thunk()
    assert str(cthunk).startswith("_CThunk")
    assert "module" in str(cthunk)


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_clinker_literal_inlining():
    x, y, z = inputs()
    z = Constant(tdouble, 4.12345678)
    e = add(mul(add(x, y), div(x, y)), bad_sub(bad_sub(x, y), z))
    lnk = CLinker().accept(FunctionGraph([x, y], [e]))
    fn = make_function(lnk)
    assert abs(fn(2.0, 2.0) + 0.12345678) < 1e-9
    code = lnk.code_gen()
    # print "=== Code generated ==="
    # print code
    assert "4.12345678" in code  # we expect the number to be inlined


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_clinker_literal_cache():
    mode = Mode(linker="c")

    A = matrix()
    input1 = vector()

    normal_svd = np.array(
        [
            [5.936276e01, -4.664007e-07, -2.56265e-06],
            [-4.664007e-07, 9.468691e-01, -3.18862e-02],
            [-2.562651e-06, -3.188625e-02, 1.05226e00],
        ],
        dtype=config.floatX,
    )

    orientationi = np.array([59.36276866, 1.06116353, 0.93797339], dtype=config.floatX)

    for out1 in [A - input1[0] * np.identity(3), input1[0] * np.identity(3)]:
        benchmark = function(
            inputs=[A, input1], outputs=[out1], on_unused_input="ignore", mode=mode
        )

        out1 = benchmark(normal_svd, orientationi)


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_clinker_single_node():
    x, y, _z = inputs()
    node = add.make_node(x, y)
    lnk = CLinker().accept(FunctionGraph(node.inputs, node.outputs))
    fn = make_function(lnk)
    assert fn(2.0, 7.0) == 9


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
@pytest.mark.parametrize(
    "linker", [CLinker(), VMLinker(use_cloop=True)], ids=["C", "CVM"]
)
@pytest.mark.parametrize("atomic_type", ["constant", "nominal"])
def test_clinker_atomic_inputs(linker, atomic_type):
    """Test that compiling variants of the same graph with different order of atomic inputs works correctly

    Indirect regression test for https://github.com/pymc-devs/pytensor/issues/1670
    """

    def call(thunk_out, args):
        thunk, input_storage, output_storage = thunk_out
        assert len(input_storage) == len(args)
        for i, arg in zip(input_storage, args):
            i.data = arg
        thunk()
        assert len(output_storage) == 1, "Helper function assumes one output"
        return output_storage[0].data

    if atomic_type == "constant":
        # Put large value to make sure we don't forget to specify it
        x = Constant(tdouble, 999, name="x")
        one = Constant(tdouble, 1.0)
        two = Constant(tdouble, 2.0)
    else:
        x = NominalVariable(0, tdouble, name="x")
        one = NominalVariable(1, tdouble, name="one")
        two = NominalVariable(1, tdouble, name="two")

    sub_one = sub(x, one)
    sub_two = sub(x, two)

    # It may seem strange to have a constant as an input,
    # but that's exactly how C_Ops define a single node FunctionGraph
    # to be compiled by the CLinker.
    # FunctionGraph(node.inputs, node.outputs)
    fg1 = FunctionGraph(inputs=[x, one], outputs=[sub_one])
    thunk1 = linker.accept(fg1).make_thunk()
    assert call(thunk1, [10, 1]) == 9
    # Technically, passing a wrong constant is undefined behavior,
    # Just checking the current behavior, NOT ENFORCING IT
    assert call(thunk1, [10, 0]) == 10

    # The old code didn't use to handle a swap of atomic inputs correctly
    # Because it didn't expect Atomic variables to be in the inputs list
    # This reordering doesn't usually happen, because C_Ops pass the inputs in the order of the node.
    # What can happen is that we compile the same FunctionGraph with CLinker and CVMLinker,
    # The CLinker takes the whole FunctionGraph as is, with the required inputs specified by the user
    # While the CVMLinker will call the CLinker on its one Op with all inputs (required and constants)
    # This difference in input signature used to be ignored by the cache key,
    # but the generated code cared about the number of explicit inputs.
    # Changing the order of inputs is a smoke test to make sure we pay attention to the input signature.
    # The fg4 below tests the actual number of inputs changing.
    fg2 = FunctionGraph(inputs=[one, x], outputs=[sub_one])
    thunk2 = linker.accept(fg2).make_thunk()
    assert call(thunk2, [1, 10]) == 9
    # Again, technically undefined behavior
    assert call(thunk2, [0, 10]) == 10

    fg3 = FunctionGraph(inputs=[x, two], outputs=[sub_two])
    thunk3 = linker.accept(fg3).make_thunk()
    assert call(thunk3, [10, 2]) == 8

    # For completeness, confirm the CLinker cmodule_key are all different
    key1 = CLinker().accept(fg1).cmodule_key()
    key2 = CLinker().accept(fg2).cmodule_key()
    key3 = CLinker().accept(fg3).cmodule_key()

    if atomic_type == "constant":
        # Case that only make sense for constant atomic inputs

        # This used to complain that an extra imaginary argument didn't have the right dtype
        # Because it used to reuse the codegen from the previous examples incorrectly
        fg4 = FunctionGraph(inputs=[x], outputs=[sub_one])
        thunk4 = linker.accept(fg4).make_thunk()
        assert call(thunk4, [10]) == 9

        # Note that fg1 and fg3 are structurally identical, but have distinct constants
        # Therefore they have distinct module keys.
        # This behavior could change in the future, to enable more caching reuse:
        # https://github.com/pymc-devs/pytensor/issues/1672
        key4 = CLinker().accept(fg4).cmodule_key()
        assert len({key1, key2, key3, key4}) == 4
    else:
        # With nominal inputs, fg1 and fg3 are identical
        assert key1 != key2
        assert key1 == key3


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_clinker_cvm_same_function():
    # Direct regression test for
    # https://github.com/pymc-devs/pytensor/issues/1670
    x1 = NominalVariable(0, vector("x", shape=(10,), dtype="float64").type)
    y1 = NominalVariable(1, vector("y", shape=(10,), dtype="float64").type)
    const1 = np.arange(10)
    out = x1 + const1 * y1

    # Without borrow the C / CVM code is different
    fn = function(
        [x1, y1], [Out(out, borrow=True)], mode=Mode(linker="c", optimizer="fast_run")
    )
    fn(np.zeros(10), np.zeros(10))

    fn = function(
        [x1, y1],
        [Out(out, borrow=True)],
        mode=Mode(linker="cvm", optimizer="fast_run"),
    )
    fn(
        np.zeros(10), np.zeros(10)
    )  # Used to raise ValueError: expected an ndarray, not None


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_clinker_dups():
    # Testing that duplicate inputs are allowed.
    x, _y, _z = inputs()
    e = add(x, x)
    lnk = CLinker().accept(FunctionGraph([x, x], [e]))
    fn = make_function(lnk)
    assert fn(2.0, 2.0) == 4
    # note: for now the behavior of fn(2.0, 7.0) is undefined


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_clinker_not_used_inputs():
    # Testing that unused inputs are allowed.
    x, y, z = inputs()
    e = add(x, y)
    lnk = CLinker().accept(FunctionGraph([x, y, z], [e]))
    fn = make_function(lnk)
    assert fn(2.0, 1.5, 1.0) == 3.5


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_clinker_dups_inner():
    # Testing that duplicates are allowed inside the graph
    x, y, z = inputs()
    e = add(mul(y, y), add(x, z))
    lnk = CLinker().accept(FunctionGraph([x, y, z], [e]))
    fn = make_function(lnk)
    assert fn(1.0, 2.0, 3.0) == 8.0


# slow on linux, but near sole test and very central
def test_opwiseclinker_straightforward():
    x, y, z = inputs()
    e = add(mul(add(x, y), div(x, y)), bad_sub(bad_sub(x, y), z))
    lnk = OpWiseCLinker().accept(FunctionGraph([x, y, z], [e]))
    fn = make_function(lnk)
    if config.cxx:
        assert fn(2.0, 2.0, 2.0) == 2.0
    else:
        # The python version of bad_sub always return -10.
        assert fn(2.0, 2.0, 2.0) == -6


def test_opwiseclinker_constant():
    x, y, z = inputs()
    x = Constant(tdouble, 7.2, name="x")
    e = add(mul(x, y), mul(y, z))
    lnk = OpWiseCLinker().accept(FunctionGraph([y, z], [e]))
    fn = make_function(lnk)
    res = fn(1.5, 3.0)
    assert res == 15.3


class MyExc(Exception):
    pass


def _my_checker(x, y):
    if x[0] != y[0]:
        raise MyExc("Output mismatch.", {"performlinker": x[0], "clinker": y[0]})


def test_duallinker_straightforward():
    x, y, z = inputs()
    e = add(mul(x, y), mul(y, z))  # add and mul are correct in C and in Python
    lnk = DualLinker(checker=_my_checker).accept(FunctionGraph([x, y, z], [e]))
    fn = make_function(lnk)
    res = fn(7.2, 1.5, 3.0)
    assert res == 15.3


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_duallinker_mismatch():
    x, y, z = inputs()
    # bad_sub is correct in C but erroneous in Python
    e = bad_sub(mul(x, y), mul(y, z))
    g = FunctionGraph([x, y, z], [e])
    lnk = DualLinker(checker=_my_checker).accept(g)
    fn = make_function(lnk)

    # good
    assert make_function(CLinker().accept(g))(1.0, 2.0, 3.0) == -4.0
    # good
    assert make_function(OpWiseCLinker().accept(g))(1.0, 2.0, 3.0) == -4.0

    # (purposely) wrong
    assert make_function(PerformLinker().accept(g))(1.0, 2.0, 3.0) == -10.0

    with pytest.raises(MyExc):
        # this runs OpWiseCLinker and PerformLinker in parallel and feeds
        # variables of matching operations to _my_checker to verify that they
        # are the same.
        fn(1.0, 2.0, 3.0)


class AddFail(Binary):
    def c_code(self, node, name, inp, out, sub):
        x, y = inp
        (z,) = out
        fail = sub["fail"]
        return f"""{z} = {x} + {y};
            PyErr_SetString(PyExc_RuntimeError, "failing here");
            {fail};"""

    def impl(self, x, y):
        return x + y


add_fail = AddFail()


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_c_fail_error():
    x, y, z = inputs()
    x = Constant(tdouble, 7.2, name="x")
    e = add_fail(mul(x, y), mul(y, z))
    lnk = OpWiseCLinker().accept(FunctionGraph([y, z], [e]))
    fn = make_function(lnk)
    with pytest.raises(RuntimeError):
        fn(1.5, 3.0)


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_shared_input_output():
    # Test bug reported on the mailing list by Alberto Orlandi
    # https://groups.google.com/d/topic/theano-users/6dLaEqc2R6g/discussion
    # The shared variable is both an input and an output of the function.
    inc = iscalar("inc")
    state = shared(0)
    state.name = "state"
    linker = CLinker()
    mode = Mode(linker=linker)
    f = function([inc], state, updates=[(state, state + inc)], mode=mode)
    g = function([inc], state, updates=[(state, state + inc)])

    # Initial value
    f0 = f(0)
    g0 = g(0)
    assert f0 == g0 == 0, (f0, g0)

    # Increment state via f, returns the previous value.
    f2 = f(2)
    assert f2 == f0, (f2, f0)
    f0 = f(0)
    g0 = g(0)
    assert f0 == g0 == 2, (f0, g0)

    # Increment state via g, returns the previous value
    g3 = g(3)
    assert g3 == g0, (g3, g0)
    f0 = f(0)
    g0 = g(0)
    assert f0 == g0 == 5, (f0, g0)

    vstate = shared(np.zeros(3, dtype="int32"))
    vstate.name = "vstate"
    fv = function([inc], vstate, updates=[(vstate, vstate + inc)], mode=mode)
    gv = function([inc], vstate, updates=[(vstate, vstate + inc)])

    # Initial value
    fv0 = fv(0)
    gv0 = gv(0)
    assert np.all(fv0 == 0), fv0
    assert np.all(gv0 == 0), gv0

    # Increment state via f, returns the previous value.
    fv2 = fv(2)
    assert np.all(fv2 == fv0), (fv2, fv0)
    fv0 = fv(0)
    gv0 = gv(0)
    assert np.all(fv0 == 2), fv0
    assert np.all(gv0 == 2), gv0

    # Increment state via g, returns the previous value
    gv3 = gv(3)
    assert np.all(gv3 == gv0), (gv3, gv0)
    fv0 = fv(0)
    gv0 = gv(0)
    assert np.all(fv0 == 5), fv0
    assert np.all(gv0 == 5), gv0


def test_cmodule_key_empty_props():
    """Make sure `CLinker.cmodule_key_` is correct when `COp.__props__` is empty."""

    class MyAdd(COp):
        __props__ = ()

        def make_node(self, *inputs):
            inputs = list(map(as_variable, inputs))
            outputs = [tdouble()]
            return Apply(self, inputs, outputs)

        def __str__(self):
            return self.name

        def perform(self, node, inputs, output_storage):
            (out,) = output_storage
            out[0] = sum(*inputs)

        def c_code_cache_version(self):
            return (1,)

        def c_code(self, node, name, inp, out, sub):
            x, y = inp
            (z,) = out
            return f"{z} = {x} + {y};"

    x = tdouble("x")
    y = tdouble("y")

    z = MyAdd()(x, y)

    fg = FunctionGraph(outputs=[z])

    linker = CLinker()
    linker.accept(fg)
    key = linker.cmodule_key()
    # None of the C version values should be empty
    assert all(kv for kv in key[0])
