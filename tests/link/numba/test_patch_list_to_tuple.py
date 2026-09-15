import operator

import pytest


numba = pytest.importorskip("numba")

import pytensor.link.numba.dispatch  # noqa: F401  (installs the patch)


def _interpret(fn):
    from numba.core import bytecode, interpreter

    func_id = bytecode.FunctionIdentity.from_function(fn)
    interp = interpreter.Interpreter(func_id)
    return interp.interpret(bytecode.ByteCode(func_id))


def _make_wide_caller(n_args):
    args_sig = ", ".join(f"a{i}" for i in range(n_args))
    src = f"def caller(callee, {args_sig}):\n    return callee({args_sig})\n"
    glb: dict = {}
    exec(src, glb)
    return glb["caller"]


def test_wide_call_collapses_to_single_build_tuple():
    """A >30-argument call leaves one build_tuple, not a chain of prefix tuples."""
    from numba.core import ir

    func_ir = _interpret(_make_wide_caller(40))
    tuple_adds = [
        stmt
        for blk in func_ir.blocks.values()
        for stmt in blk.body
        if isinstance(stmt, ir.Assign)
        and isinstance(stmt.value, ir.Expr)
        and stmt.value.op == "binop"
        and stmt.value.fn is operator.add
    ]
    assert not tuple_adds
    # the empty list the peephole starts from may survive as dead code
    widths = [
        len(stmt.value.items)
        for blk in func_ir.blocks.values()
        for stmt in blk.body
        if isinstance(stmt, ir.Assign)
        and isinstance(stmt.value, ir.Expr)
        and stmt.value.op == "build_tuple"
        and stmt.value.items
    ]
    assert widths == [40]


def test_wide_call_computes_correctly():
    @numba.njit
    def callee(*args):
        total = 0.0
        for a in args:
            total += a
        return total

    caller = numba.njit(_make_wide_caller(35))
    vals = [float(i) for i in range(35)]
    assert caller(callee, *vals) == sum(vals)


def test_wide_call_with_kwarg():
    """>30 positional args plus a keyword argument: the collapsed tuple must
    reach the call's vararg directly or numba's kwargs peephole rejects it."""
    n_args = 40
    params = ", ".join(f"a{i}" for i in range(n_args))
    glb: dict = {}
    exec(
        f"def callee({params}, k):\n"
        f"    return a0 + k\n"
        f"def caller({params}):\n"
        f"    return jitted_callee({params}, k=1.0)\n",
        glb,
    )
    glb["jitted_callee"] = numba.njit(glb["callee"])
    caller = numba.njit(glb["caller"])
    assert caller(*[float(i) for i in range(n_args)]) == 1.0


def test_tuple_unpacking_still_works():
    """Chains interleaved with genuine unpacking still compute correctly."""

    @numba.njit
    def spread(a, b):
        t = (*a, 1.0, *b, 2.0)
        return len(t), t[len(a)], t[-1]

    n, mid, last = spread((3.0, 4.0), (5.0,))
    assert (n, mid, last) == (5, 1.0, 2.0)


def test_wide_call_with_starred_unpacking():
    """Runs of appends around a starred unpack each coalesce, so the chain is
    bounded by the number of extends rather than the argument count."""
    from numba.core import ir

    n = 35
    params = ", ".join(f"a{i}" for i in range(n))
    glb: dict = {}
    exec(f"def caller(callee, t, {params}):\n    return callee({params}, *t)\n", glb)
    func_ir = _interpret(glb["caller"])
    adds = [
        stmt
        for blk in func_ir.blocks.values()
        for stmt in blk.body
        if isinstance(stmt, ir.Assign)
        and isinstance(stmt.value, ir.Expr)
        and stmt.value.op == "binop"
        and stmt.value.fn is operator.add
    ]
    assert len(adds) <= 2

    @numba.njit
    def callee(*args):
        return args[0] + args[-1]

    cfunc = numba.njit(glb["caller"])
    vals = [float(i) for i in range(n)]
    assert cfunc(callee, (99.0, 7.0), *vals) == 7.0
