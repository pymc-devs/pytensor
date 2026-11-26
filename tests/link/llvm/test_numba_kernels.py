"""Ops without an LLVM implementation run as numba kernels."""

import ctypes
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import pytensor
import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor.compile.mode import get_mode
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.link.numba.dispatch.random import (
    numba_core_rv_default,
    numba_core_rv_funcify,
)
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.random.op import RandomVariable


pytensor_llvm = pytest.importorskip("pytensor_llvm")
pytest.importorskip("numba")

LLVM = get_mode("LLVM")
rng = np.random.default_rng(3)


def compiled(f):
    return f.maker.linker.program


def numba_steps(f):
    return [d for d in compiled(f).steps if "numba:" in d]


def test_numba_ops_mixed_with_rust_kernels():
    x = pt.matrix("x")
    s = pt.lvector("s")
    outs = [
        pt.repeat(x[:, 0], s[:4]) * 2,  # numba -> generated Elemwise
        pt.repeat(x[0], s[:5]).sum(),  # numba -> generated reduction
        pt.argmax(x, axis=1),
        pt.argsort(pt.exp(x), axis=1),  # generated Elemwise feeding a numba kernel
        pt.linalg.det(x[:4, :4]),
    ]
    f = pytensor.function([x, s], outs, mode=LLVM)
    assert len(numba_steps(f)) >= 3
    v = rng.normal(size=(4, 5))
    sv = np.array([1, 0, 3, 2, 1])
    res = f(v, sv)
    ref = pytensor.function([x, s], outs, mode="FAST_COMPILE")(v, sv)
    for r, e in zip(res, ref, strict=True):
        np.testing.assert_allclose(r, e)
    # strided input into a numba kernel
    vt = rng.normal(size=(5, 4)).T
    np.testing.assert_allclose(f(vt, sv)[1], np.repeat(vt[0], sv).sum())


class _Total(Op):
    """The sum of a vector as a ScalarType (no LLVM implementation)."""

    def make_node(self, x):
        return Apply(self, [x], [ps.float64()])

    def perform(self, node, inputs, outputs):
        outputs[0][0] = np.float64(inputs[0].sum())


def test_numba_scalar_inputs_and_outputs():
    x = pt.vector("x")
    k = pt.lscalar("k")
    m = pt.matrix("m")
    outs = [
        x[k],
        x[k:] * x[k],
        # a ScalarType output, inside a unit and as a Blockwise core's
        pt.tensor_from_scalar(_Total()(x) * 2),
        Blockwise(_Total(), signature="(n)->()")(m) * 2,
    ]
    f = pytensor.function([x, k, m], outs, mode=LLVM)
    v, w = rng.normal(size=6), rng.normal(size=(2, 3))
    a, b, total, totals = f(v, 2, w)
    assert a == pytest.approx(v[2])
    np.testing.assert_allclose(b, v[2:] * v[2])
    assert total == pytest.approx(2 * v.sum())
    np.testing.assert_allclose(totals, 2 * w.sum(axis=1))


def test_numba_generator_model():
    # Random variables without a sampler draw through numba with a Generator built
    # from the call context (`ctx_generator`), which fills numba's own struct for it:
    # if numba changes that, this says so before the draws go wrong
    from numba import types
    from numba.core.datamodel import default_manager

    from pytensor.link.llvm.kernels.numba import BIT_GENERATOR_MEMBERS

    generator = default_manager[types.npy_rng]
    assert generator._fields == ("bit_generator", "meminfo", "parent")
    bit_generator = default_manager[generator.get_member_fe_type("bit_generator")]
    assert bit_generator._fields == BIT_GENERATOR_MEMBERS
    assert bit_generator.get_member_fe_type("parent") == types.pyobject
    for member in BIT_GENERATOR_MEMBERS[1:]:
        assert bit_generator.get_member_fe_type(member) == types.uintp


class _NumbaUniformRV(RandomVariable):
    """A uniform random variable the backend has no sampler for: numba draws it, as
    `low + (high - low) * next_double`."""

    name = "uniform"
    signature = "(),()->()"
    dtype = "float64"

    def rng_fn(self, rng, low, high, size):
        return rng.uniform(low, high, size=size)


numba_core_rv_funcify.register(_NumbaUniformRV)(numba_core_rv_default)


def test_numba_random_variables_draw_from_the_call_state():
    # no sampler: numba's core function draws (one per element of the batch loop),
    # through the session's state, the same numbers as the native uniform sampler
    # (`rng_uniform`) from the same seed
    raw = pytensor.function([], _NumbaUniformRV()(0.0, 1.0, size=(5,)), mode=LLVM)
    native = pytensor.function([], pt.random.uniform(0.0, 1.0, size=(5,)), mode=LLVM)
    assert raw.maker.linker.stats.numba and not native.maker.linker.stats.numba
    compiled(raw).seed(7)
    compiled(native).seed(7)
    first = raw()
    np.testing.assert_array_equal(first, native())
    assert not np.array_equal(first, raw())
    # bounded integers (`next_uint64`, `next_uint32`)
    permutation = pytensor.function([], pt.random.permutation(20), mode=LLVM)
    assert permutation.maker.linker.stats.numba
    p, q = permutation(), permutation()
    assert sorted(p) == list(range(20)) and not np.array_equal(p, q)


class _PythonUniformRV(RandomVariable):
    """A uniform random variable numba has no core function for: its `perform` runs."""

    name = "uniform"
    signature = "(),()->()"
    dtype = "float64"

    def rng_fn(self, rng, low, high, size):
        return rng.uniform(low, high, size=size)


def test_random_variables_without_numba_core_draw_from_a_seeded_generator():
    x = _PythonUniformRV()(2.0, 3.0, size=(4,))
    with pytest.warns(UserWarning, match="Generator seeded from the call's random"):
        f = pytensor.function([], x, mode=LLVM)
    compiled(f).seed(7)
    first, second = f(), f()
    assert np.all((first >= 2) & (first < 3)) and not np.array_equal(first, second)
    compiled(f).seed(7)
    np.testing.assert_array_equal(f(), first)


def test_numba_outputs_have_the_layout_of_their_value():
    # i0 has no LLVM implementation; of a transposed matrix, its result is packed in
    # the transposed order
    x = pt.vector("x")
    f = pytensor.function([x], pt.i0(pt.exp(x).reshape((2, 3)).T), mode=LLVM)
    assert f.maker.linker.stats.numba == ["I0"]
    v = rng.normal(size=6)
    np.testing.assert_allclose(f(v), np.i0(np.exp(v).reshape(2, 3).T))


def test_numba_exception_is_reported():
    x = pt.vector("x")
    s = pt.lvector("s")
    f = pytensor.function([x, s], pt.repeat(x, s) * 2, mode=LLVM)
    assert numba_steps(f)
    np.testing.assert_allclose(f(np.arange(3.0), np.array([1, 0, 2])), [0.0, 4.0, 4.0])
    with pytest.raises(Exception, match="numba: Repeat"):
        f(np.arange(3.0), np.array([1, 2]))
    np.testing.assert_allclose(f(np.arange(2.0), np.array([0, 2])), [2.0, 2.0])


class _Picky(Op):
    """No numba implementation: numba runs `perform` in object mode."""

    __props__ = ()

    def make_node(self, x):
        x = pt.as_tensor(x)
        return Apply(self, [x], [pt.matrix(dtype="float64")])

    def perform(self, node, inputs, output_storage):
        (x,) = inputs
        if x.sum() < 0:
            raise ValueError("negative")
        output_storage[0][0] = x * 2


def test_numba_object_mode_exception_is_reported():
    x = pt.matrix("x")
    f = pytensor.function([x], pt.dot(_Picky()(x), x.T), mode=LLVM)
    assert numba_steps(f)
    np.testing.assert_allclose(f(np.ones((2, 2))), np.full((2, 2), 4.0))
    with pytest.raises(Exception, match=r"numba: _Picky: .* raised an exception"):
        f(-np.ones((2, 2)))


def test_numba_object_mode_from_the_cache(tmp_path):
    # numba runs a set with a 2-d mask in object mode; the second process loads the
    # kernel from numba's cache
    script = textwrap.dedent(
        """
        import numpy as np, pytensor, pytensor.tensor as pt
        from pytensor.compile.mode import get_mode

        mask = np.array([[True, False], [False, True], [True, True]])
        x, v = pt.matrix("x", shape=(3, 2)), pt.vector("v", shape=(4,))
        mode = get_mode("LLVM").excluding("bool_idx_to_nonzero")
        f = pytensor.function([x, v], pt.set_subtensor(x[mask], v) * 2, mode=mode)
        assert any("numba:" in d for d in f.maker.linker.program.steps)
        print(f(np.zeros((3, 2)), np.arange(1.0, 5.0)).tolist())
        """
    )
    env = dict(
        os.environ,
        PYTENSOR_FLAGS=f"compiledir={tmp_path / 'compiledir'}",
    )
    for _ in range(2):
        out = subprocess.run(
            [sys.executable, "-W", "ignore", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert out.strip() == "[[2.0, 0.0], [0.0, 4.0], [6.0, 8.0]]"


def test_numba_can_be_disabled():
    x = pt.matrix("x")
    mode = get_mode("LLVM").clone(link_kwargs={"numba_fallback": False})
    with pytest.raises(NotImplementedError, match="No LLVM"):
        pytensor.function([x], pt.argsort(x, axis=0), mode=mode)


def test_numba_through_c_api_without_gil():
    from tests.link.llvm.test_c_api import Api, PtArray, as_ptarrays

    x = pt.matrix("x")
    out = (pt.argsort(x, axis=1) * 2).sum(axis=0)
    f = pytensor.function([x], out, mode=LLVM)
    assert numba_steps(f)
    v = rng.normal(size=(30, 7))
    fn = compiled(f).c_function()
    a = Api.from_address(compiled(f).c_api_table())
    call_t = ctypes.CFUNCTYPE(
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.POINTER(PtArray),
        ctypes.POINTER(PtArray),
    )
    new_t = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)
    free_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    # ctypes.CFUNCTYPE callables release the GIL while the native call runs
    st = new_t(a.session_new)(fn)
    ins, _keep = as_ptarrays([v])
    res = np.empty(7, dtype="int64")
    outs, _keep2 = as_ptarrays([res])
    assert call_t(a.call)(st, ins, outs) == 0
    np.testing.assert_allclose(res, (np.argsort(v, axis=1) * 2).sum(axis=0))
    free_t(a.session_free)(st)
    free_t(a.function_decref)(fn)
