"""Temporary memory of generated code: the stack for small static sizes, else scratch."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.builders import Execution, OpFromGraph
from pytensor.compile.mode import get_mode


ir = pytest.importorskip("pytensor_llvm.ir")

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
    return text.count(" = alloca "), text.count("call ptr @ptl_rt_scratch")


def test_small_static_temporaries_are_on_the_stack():
    m, b, cg, n = codegen()
    cg.temp_memory(b.i64(8))
    a = cg.temporary("float64", [4, b.i64(8)])  # Python ints are fine in shapes
    assert a.layout.is_c((4, 8)) and [s.const_int for s in a.strides] == [64, 8]
    cg.temp_memory(b.i64(STACK_TEMPORARY_BYTES))
    assert counts(m) == (3, 0)
    assert "align 64" in m.ir()


def test_large_dynamic_and_escaping_temporaries_are_scratch():
    m, b, cg, n = codegen()
    cg.temp_memory(b.i64(STACK_TEMPORARY_BYTES + 1))  # too large
    cg.temp_memory(b.mul(n, 8))  # size only known when running
    cg.temp_memory(b.i64(8), escapes=True)  # outlives the function
    cg.temporary("float64", [n])
    assert counts(m) == (0, 4)
    assert "declare noalias ptr @ptl_rt_scratch" in m.ir()


def test_the_stack_has_a_budget_per_function():
    m, b, cg, n = codegen()
    fit = STACK_BUDGET_BYTES // STACK_TEMPORARY_BYTES
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
    assert len(f.maker.linker.program.tasks) == 1

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
