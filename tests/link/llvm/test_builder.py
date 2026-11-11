"""The IR builder's conveniences: literal operands, simplification, else, names."""

import ctypes

import pytest


ir = pytest.importorskip("pytensor_llvm.ir")


def _function(params, ret, name="f"):
    m = ir.Module("t")
    f = m.function(name, params, ret, exported=True)
    return m, f, f.builder()


def _compile(m, restype, *argtypes, name="f"):
    c = m.compile(cache=False)
    fn = ctypes.CFUNCTYPE(restype, *argtypes)(c.address(name))
    fn._keep = c
    return fn


def test_literals_take_the_type_of_the_other_operand():
    m, f, b = _function([ir.i64, ir.f64], ir.f64)
    i, x = f.params
    i = b.add(b.mul(2, i), 1)  # i64 literals
    y = b.fadd(b.fmul(x, 2), 0.5)  # an int literal is fine for a float
    y = b.select(b.sgt(i, 10), y, 0.0)
    b.ret(b.fadd(y, b.sitofp(i, ir.f64)))
    fn = _compile(m, ctypes.c_double, ctypes.c_int64, ctypes.c_double)
    assert fn(7, 1.5) == (1.5 * 2 + 0.5) + 15
    assert fn(2, 1.5) == 5.0


def test_literals_that_dont_fit_raise():
    m, f, b = _function([ir.i64, ir.i8], ir.i64)
    i, small = f.params
    with pytest.raises(ValueError, match="can't be a i64"):
        b.add(i, 1.5)
    with pytest.raises(ValueError, match="one operand must be a value"):
        b.add(1, 2)
    with pytest.raises(ValueError, match="doesn't fit"):
        b.add(small, 1000)
    with pytest.raises(ValueError, match="operand types differ"):
        b.add(i, small)


def test_constants_and_typed_counts():
    m, f, b = _function([ir.ptr], ir.i64)
    assert b.i64(3).type == ir.i64 and b.i32(3).type == ir.i32 and b.i8(3).type == ir.i8
    assert b.f64(1.5).const_float == 1.5 and b.f32(1.5).type == ir.f32
    (p,) = f.params
    with b.loop(4) as i:  # the count is an i64
        b.store(i, b.gep(ir.i64, p, i))
    b.ret(b.load(ir.i64, b.gep(ir.i64, p, 3)))
    buf = (ctypes.c_int64 * 4)()
    assert _compile(m, ctypes.c_int64, ctypes.c_void_p)(buf) == 3
    assert list(buf) == [0, 1, 2, 3]


def test_integer_arithmetic_simplifies():
    m, f, b = _function([ir.i64, ir.i64, ir.i1], ir.i64)
    x, y, flag = f.params
    same = repr(x)
    # identities return the operand
    for v in [
        b.add(x, 0),
        b.add(0, x),
        b.sub(x, 0),
        b.mul(x, 1),
        b.mul(1, x, nsw=True),
        b.sdiv(x, 1),
    ]:
        assert repr(v) == same
    for v in [
        b.shl(x, 0),
        b.or_(x, 0),
        b.xor(0, x),
        b.and_(x, -1),
        b.and_(x, x),
        b.smin(x, x),
    ]:
        assert repr(v) == same
    assert repr(b.and_(flag, True)) == repr(flag) and repr(b.or_(False, flag)) == repr(
        flag
    )
    # absorbing elements and constants fold to constants
    assert b.mul(x, 0).const_int == 0 and b.and_(0, x).const_int == 0
    assert b.or_(flag, True).const_int == 1 and b.and_(flag, False).const_int == 0
    assert (
        b.add(b.i64(2), 3).const_int == 5 and b.mul(b.i64(4), b.i64(5)).const_int == 20
    )
    assert b.slt(b.i64(1), 2).const_int == 1 and b.eq(b.i64(1), 2).const_int == 0
    assert b.smin(b.i64(3), 5).const_int == 3 and b.smax(b.i64(3), 5).const_int == 5
    assert b.umin(b.i64(-1), 5).const_int == 5 and b.umax(b.i64(-1), 5).const_int == -1
    # a select on a constant condition is its operand
    assert repr(b.select(b.const_bool(True), x, y)) == same
    assert repr(b.select(b.slt(b.i64(1), 0), y, x)) == same
    assert repr(b.select(flag, x, x)) == same
    # what can't be simplified is an instruction
    assert b.add(x, y).const_int is None
    b.ret(b.smax(b.smin(x, y), 0))
    fn = _compile(m, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_bool)
    assert [fn(3, 5, 0), fn(5, 3, 0), fn(-2, 9, 0)] == [3, 3, 0]


@pytest.mark.parametrize(
    "name", ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]
)
def test_comparison_shortcuts(name):
    m, f, b = _function([ir.i64, ir.i64], ir.i32)
    x, y = f.params
    short = getattr(b, name)(x, y)
    assert (
        repr(short).split("=")[1]
        == repr(b.icmp(getattr(ir.ICmp, name.upper()), x, y)).split("=")[1]
    )
    b.ret(b.zext(short, ir.i32))
    fn = _compile(m, ctypes.c_int32, ctypes.c_int64, ctypes.c_int64)
    py = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}[
        name[-2:]
    ]
    for xv, yv in [(1, 2), (2, 1), (3, 3), (-1, 2)]:
        ux, uy = (xv % 2**64, yv % 2**64) if name[0] == "u" else (xv, yv)
        assert fn(xv, yv) == int(eval(f"{ux} {py} {uy}"))


def test_else_follows_an_if():
    m, f, b = _function([ir.i64], ir.i64)
    (x,) = f.params
    slot = b.alloca(ir.i64)
    with b.if_(b.slt(x, 0)):
        b.store(b.neg(x), slot)
    with b.else_():
        with b.if_(b.sgt(x, 100)):
            b.store(b.i64(100), slot)
        with b.else_():
            b.store(x, slot)
    b.ret(b.load(ir.i64, slot))
    fn = _compile(m, ctypes.c_int64, ctypes.c_int64)
    assert [fn(-4), fn(7), fn(1000)] == [4, 7, 100]


def test_else_needs_its_if():
    m, f, b = _function([ir.i64], ir.i64)
    (x,) = f.params
    with pytest.raises(ValueError, match="must directly follow"):
        b.else_()
    with b.if_(b.slt(x, 0)):
        pass
    y = b.add(x, 1)  # something came between
    with pytest.raises(ValueError, match="must directly follow"):
        b.else_()
    with b.if_(b.slt(x, 0)):
        pass
    with b.else_():
        pass
    with pytest.raises(ValueError, match="must directly follow"):
        b.else_()  # an if has one else
    b.ret(y)
    m.verify()


def test_bodies_may_return_one_value_or_none():
    m, f, b = _function([ir.i64], ir.i64)
    (n,) = f.params
    (total,) = b.fold(n, [b.i64(0)], lambda i, acc: b.add(acc, i))  # one value, no list
    assert b.if_else(b.sgt(total, 5), lambda: None, lambda: None) == []
    (r,) = b.if_else(b.sgt(total, 5), lambda: total, lambda: b.i64(-1))
    b.ret(r)
    fn = _compile(m, ctypes.c_int64, ctypes.c_int64)
    assert [fn(5), fn(3)] == [10, -1]


def test_named_values_show_in_the_ir():
    m, f, b = _function([ir.i64, ir.i64], ir.i64)
    x, y = f.params
    x.named("x")
    total = b.add(x, y).named("total")
    assert b.i64(1).named("one").const_int == 1  # constants have no names
    b.ret(total)
    text = m.ir()
    assert "%total = add i64 %x" in text and "one" not in text


def test_stubs_list_the_builders_methods():
    """The type stubs shipped with the extension know every builder method."""
    import ast
    import pathlib

    import pytensor_llvm

    stub = (
        pathlib.Path(pytensor_llvm.__file__).parent.parent
        / "pytensor_llvm-stubs"
        / "ir.pyi"
    )
    if not stub.exists():
        pytest.skip("pytensor_llvm is installed without its stubs")
    classes = {
        n.name: n
        for n in ast.parse(stub.read_text()).body
        if isinstance(n, ast.ClassDef)
    }
    for cls in [
        "Builder",
        "Value",
        "Module",
        "Function",
        "Type",
        "CompiledModule",
        "Parallel",
    ]:
        typed = {n.name for n in classes[cls].body if isinstance(n, ast.FunctionDef)}
        typed |= {
            n.target.id for n in classes[cls].body if isinstance(n, ast.AnnAssign)
        }
        real = {n for n in dir(getattr(ir, cls)) if not n.startswith("_")}
        assert real <= typed, f"{cls}: the stubs miss {sorted(real - typed)}"
