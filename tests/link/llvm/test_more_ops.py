"""Split, CumOp, ExtractDiag and multi-index advanced indexing (all native, no fallback)."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.link.llvm.linker import LLVM_REWRITES, LLVMLinker


pytest.importorskip("pytensor_llvm")

LLVM = Mode(linker=LLVMLinker(numba_fallback=False), optimizer=RewriteDatabaseQuery(include=["fast_run", *LLVM_REWRITES]))
REF = Mode(linker="py", optimizer="fast_run")
rng = np.random.default_rng(13)
x, v, y = pt.matrix("x"), pt.vector("v"), pt.vector("y")
t = pt.tensor3("t")
i, j = pt.lvector("i"), pt.lvector("j")
rows, cols = np.tril_indices(4)

CASES = {
    "tril_gather": ([x], [x[rows, cols]], [rng.normal(size=(4, 4))]),
    "two_index_arrays": ([x, i, j], [x[i, j] * 2], [rng.normal(size=(4, 5)), np.array([0, -1, 3]), np.array([4, 0, -2])]),
    "index_array_and_scalar": ([x, i], [x[i, 0] + x[i, 1]], [rng.normal(size=(4, 5)), np.array([0, 3, 3, -1])]),
    "set_tril": ([x, y], [pt.set_subtensor(x[rows, cols], y)], [rng.normal(size=(4, 4)), rng.normal(size=10)]),
    "inc_duplicates": ([x, i, j, y], [pt.inc_subtensor(x[i, j], y)], [np.zeros((4, 5)), np.array([0, 0, 3]), np.array([1, 1, 2]), np.array([1.0, 2.0, 3.0])]),
    "inc_broadcast_scalar": ([x, i, j], [pt.inc_subtensor(x[i, j], 1.5)], [np.zeros((4, 5)), np.array([0, 0, 3]), np.array([1, 1, 2])]),
    "grad_of_tril_gather": ([x, y], [pytensor.grad((x[rows, cols] * y).sum(), x)], [rng.normal(size=(4, 4)), rng.normal(size=10)]),
    "split_constant": ([v], pt.split(v, [2, 3, 1], n_splits=3), [rng.normal(size=6)]),
    "split_variable": ([v, i], pt.split(v, i, n_splits=3), [rng.normal(size=6), np.array([1, 0, 5])]),
    "cumsum_axis0": ([x], [pt.cumsum(x, axis=0)], [rng.normal(size=(4, 3))]),
    "cumsum_strided": ([x], [pt.cumsum(x.T, axis=1)], [rng.normal(size=(4, 3))]),
    "cumprod": ([x], [pt.cumprod(x, axis=1)], [rng.normal(size=(4, 3))]),
    "cumsum_flat": ([x], [pt.cumsum(x)], [rng.normal(size=(4, 3))]),
    "cumsum_int": ([i], [pt.cumsum(i)], [np.array([1, 2, 3, -4])]),
    "cumsum_grad": ([x], [pytensor.grad((pt.cumsum(x, axis=0) ** 2).sum(), x)], [rng.normal(size=(4, 3))]),
    "diagonal": ([x], [pt.diagonal(x) * 2], [rng.normal(size=(4, 3))]),
    "diagonal_offsets": ([x], [pt.diagonal(x, 1), pt.diagonal(x, -2)], [rng.normal(size=(4, 3))]),
    "diagonal_3d": ([t], [pt.diagonal(t, 0, 0, 2) + 0], [rng.normal(size=(3, 2, 4))]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_matches_python(name):
    ins, outs, vals = CASES[name]
    f = pytensor.function(ins, outs, mode=LLVM)
    ref = pytensor.function(ins, outs, mode=REF)(*vals)
    for r, e in zip(f(*vals), ref, strict=True):
        assert r.shape == e.shape and r.dtype == e.dtype
        np.testing.assert_allclose(r, e, rtol=1e-12)


def test_multi_index_out_of_bounds_raises():
    f = pytensor.function([x, i, j], x[i, j], mode=LLVM)
    with pytest.raises(IndexError):
        f(np.zeros((2, 2)), np.array([0]), np.array([2]))


def test_scalar_and_array_index_become_view_plus_take():
    # x[g, 0] -> x[:, 0][g]: no flat-index lowering, a single native take
    g = pt.lvector("g")
    outs = [x[g, 0] * 2, pt.inc_subtensor(x[1, g], 1.0), pytensor.grad((x[g, 1] ** 2).sum(), x)]
    f = pytensor.function([x, g], outs, mode=LLVM)
    vals = [rng.normal(size=(4, 5)), np.array([0, 3, -1, 3])]
    ref = pytensor.function([x, g], outs, mode=REF)(*vals)
    for r, e in zip(f(*vals), ref, strict=True):
        np.testing.assert_allclose(r, e, rtol=1e-12)
    sched = f.maker.linker.compiled.info()["schedule"]
    assert not [d for d in sched if "reshape" in d], sched
