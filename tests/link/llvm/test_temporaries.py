"""Temporary memory of generated code: the stack for small static sizes, else scratch."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.builders import Execution, OpFromGraph
from pytensor.compile.mode import get_mode


ir = pytest.importorskip("pytensor_llvm.ir")

import pytensor.scalar as ps  # noqa: E402
from pytensor.link.llvm.codegen.emit import (  # noqa: E402
    STACK_BUDGET_BYTES,
    STACK_TEMPORARY_BYTES,
    Codegen,
)


STRICT = get_mode("LLVM").clone(link_kwargs={"numba_fallback": False})


def codegen():
    m = ir.Module("temporaries")
    f = m.function("f", [ir.ptr, ir.ptr, ir.i64], ir.i32)
    b = f.builder()
    ctx, config, n = f.params
    return m, b, Codegen(b, ctx, config), n


def counts(m):
    text = m.ir()
    return text.count(" = alloca "), text.count("call ptr @ptl_rt_scratch(")


def test_small_static_temporaries_are_on_the_stack():
    m, b, cg, _n = codegen()
    cg.temp_memory(b.i64(8))
    a = cg.temporary(ps.float64, [4, b.i64(8)])  # Python ints are fine in shapes
    assert a.layout.is_c((4, 8)) and [s.const_int for s in a.strides] == [64, 8]
    cg.temp_memory(b.i64(STACK_TEMPORARY_BYTES))
    assert counts(m) == (3, 0)
    assert "align 64" in m.ir()


def test_large_dynamic_and_escaping_temporaries_are_scratch():
    """Scratch memory is freed when its scope ends, last first; memory that escapes is
    the caller's to free."""
    m, b, cg, n = codegen()
    with cg.scratch_scope(function=True):
        cg.temp_memory(b.i64(STACK_TEMPORARY_BYTES + 1))  # too large
        cg.temp_memory(b.mul(n, 8))  # size only known when running
        cg.temp_memory(b.i64(8), escapes=True)  # outlives the function
        cg.temporary(ps.float64, [n])
        assert "@ptl_rt_scratch_free" not in m.ir()
    assert counts(m) == (0, 4)
    lines = m.ir().splitlines()
    taken = [ln.split()[0] for ln in lines if "call ptr @ptl_rt_scratch(" in ln]
    freed = [
        ln.split(", ptr ")[1].rstrip(")")
        for ln in lines
        if "call void @ptl_rt_scratch_free" in ln
    ]
    assert freed == [taken[3], taken[1], taken[0]]
    assert "declare noalias ptr @ptl_rt_scratch" in m.ir()
    with pytest.raises(ValueError, match="outside of a scratch_scope"):
        cg.temp_memory(b.mul(n, 8))


def test_the_stack_has_a_budget_per_function():
    m, b, cg, _n = codegen()
    fit = STACK_BUDGET_BYTES // STACK_TEMPORARY_BYTES
    with cg.scratch_scope(function=True):
        for _ in range(fit + 3):
            cg.temp_memory(b.i64(STACK_TEMPORARY_BYTES))
    assert counts(m) == (fit, 3)


@pytest.mark.parametrize("n", [4, 1000])
def test_units_with_static_temporaries(n):
    """The intermediate values of a unit are temporaries: on the stack for small
    arrays, scratch for large ones. Calls don't see each other's."""
    x = pt.vector("x", shape=(n,))
    y = pt.cumsum(pt.exp(x))
    z = pt.cumsum(y[::-1] * 2.0)
    out = pt.concatenate([y, z]).sum() + pt.cumprod(z / (1 + abs(z)))[-1]
    xo = pt.vector("xo", shape=(n,))
    unit = OpFromGraph([x], [out], execution=Execution.COMPILED, inline=False)
    f = pytensor.function([xo], unit(xo), mode=STRICT)
    assert len(f.maker.linker.program.steps) == 1

    def reference(v):
        y = np.cumsum(np.exp(v))
        z = np.cumsum(y[::-1] * 2.0)
        return np.concatenate([y, z]).sum() + np.cumprod(z / (1 + abs(z)))[-1]

    rng = np.random.default_rng(0)
    for _ in range(3):
        v = rng.normal(size=n) * 0.01
        np.testing.assert_allclose(f(v), reference(v), rtol=1e-10)


def test_temporaries_inside_parallel_loops_belong_to_their_chunk():
    """A reduction that accumulates in another dtype uses a temporary per output row;
    with parallel loops every chunk must have its own."""
    x = pt.matrix("x", dtype="float32")
    f = pytensor.function([x], pt.sum(x * 2, axis=1, acc_dtype="float64"), mode=STRICT)
    v = np.random.default_rng(1).random((2000, 300)).astype("float32")
    np.testing.assert_allclose(f(v), (v.astype("float64") * 2).sum(axis=1), rtol=1e-6)


def test_units_free_their_temporaries_of_dynamic_shape():
    """Intermediate values of a unit with lengths known only when running are scratch
    memory their node takes; the unit frees every piece."""
    x = pt.vector("x")
    y = pt.cumsum(pt.exp(x))
    out = pt.cumsum(y[::-1] * 2.0).sum()
    xo = pt.vector("xo")
    unit = OpFromGraph([x], [out], execution=Execution.COMPILED, inline=False)
    f = pytensor.function([xo], unit(xo), mode=STRICT)
    # (a view's slot has no memory to free: after optimization no call is left for it)
    ir = f.maker.linker.ir(optimized=True)
    taken = ir.count("call ptr @ptl_rt_scratch(")
    assert taken > 0
    assert ir.count("call void @ptl_rt_scratch_free(") == taken
    v = np.random.default_rng(2).normal(size=7) * 0.1
    np.testing.assert_allclose(
        f(v), np.cumsum(np.cumsum(np.exp(v))[::-1] * 2.0).sum(), rtol=1e-10
    )


def test_batch_loops_without_scratch_call_no_scratch_functions():
    """A Blockwise loop around a core that takes no scratch memory has no scratch calls
    (they would keep LLVM from optimizing the loop)."""
    from pytensor.tensor.blockwise import Blockwise
    from pytensor.tensor.extra_ops import CumOp

    x = pt.tensor3("x", shape=(2, 3, 4))
    cum = Blockwise(CumOp(axis=0, mode="add"), signature="(n)->(n)")
    f = pytensor.function([x], cum(x), mode=STRICT)
    (loop,) = [
        m for m in f.maker.linker.stats.modules if m.description.startswith("Blockwise")
    ]
    assert "@ptl_rt_scratch" not in loop.optimized_ir()
    v = np.random.default_rng(3).random((2, 3, 4))
    np.testing.assert_allclose(f(v), np.cumsum(v, axis=-1))


def test_large_static_temporaries_of_a_batch_loop_are_taken_once():
    """Scratch memory of static size is taken when the function starts: a Blockwise
    loop reuses it in every iteration and frees it after the loop."""
    from pytensor.tensor.blockwise import Blockwise

    v = pt.vector("v", shape=(200,))  # intermediates of 1600 bytes: scratch
    core = OpFromGraph([v], [pt.cumsum(pt.cumsum(v) * 2.0)])
    x = pt.matrix("x", shape=(5, 200))
    f = pytensor.function([x], Blockwise(core, signature="(n)->(n)")(x), mode=STRICT)
    ir = f.maker.linker.ir(optimized=True)
    kernel = next(fn for fn in ir.split("\ndefine")[1:] if "@ptl_rt_scratch(" in fn)
    entry, rest = kernel.split("\n\n", 1)
    assert entry.count("@ptl_rt_scratch(") > 0
    assert "@ptl_rt_scratch(" not in rest
    assert rest.count("@ptl_rt_scratch_free(") == entry.count("@ptl_rt_scratch(")
    xv = np.random.default_rng(4).random((5, 200))
    np.testing.assert_allclose(f(xv), np.cumsum(np.cumsum(xv, -1) * 2, -1))
