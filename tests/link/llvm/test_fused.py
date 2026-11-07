"""FusedElemwise: indexed reads/writes and reductions fused into one native loop."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.link.llvm.linker import LLVM_REWRITES, LLVMLinker


pytest.importorskip("pytensor_llvm")

# numba_fallback=False: everything must be native
LLVM = Mode(
    linker=LLVMLinker(numba_fallback=False),
    optimizer=RewriteDatabaseQuery(include=["fast_run", *LLVM_REWRITES]),
)
REF = Mode(linker="py", optimizer="fast_run")
rng = np.random.default_rng(11)


def compare(inputs, outputs, values):
    f = pytensor.function(inputs, outputs, mode=LLVM, on_unused_input="ignore")
    res = f(*values)
    ref = pytensor.function(inputs, outputs, mode=REF, on_unused_input="ignore")(*values)
    if not isinstance(res, list):
        res, ref = [res], [ref]
    for r, e in zip(res, ref, strict=True):
        assert r.dtype == e.dtype and r.shape == e.shape, (r.dtype, e.dtype, r.shape, e.shape)
        np.testing.assert_allclose(r, e, rtol=1e-12)
    return f


def fused(f):
    return [d for d in f.maker.linker.compiled.info()["schedule"] if d.startswith("FusedElemwise")]


def test_reduced_gather():
    x = pt.vector("x")
    idx = pt.lvector("idx")
    vals = [rng.normal(size=10), rng.integers(-10, 10, 100)]
    for out in [x[idx].sum(), (pt.exp(x[idx]) * 2).max()]:  # separately: single-client reads
        f = compare([x, idx], out, vals)
        assert fused(f)


def test_shared_and_separate_indices():
    x, y, z = pt.vectors("x", "y", "z")
    a = pt.lvector("a")
    b = pt.lvector("b")
    outs = [(x[a] + y[a] * z).sum(), (x[a] * y[b]).sum(), x[a] - y[b] + z]
    vals = [rng.normal(size=7), rng.normal(size=9), rng.normal(size=50), rng.integers(0, 7, 50), rng.integers(-9, 9, 50)]
    # one function per expression: PyTensor only fuses indexed reads with a single client
    for out in outs:
        f = compare([x, y, z, a, b], out, vals)
        assert fused(f)


@pytest.mark.parametrize("set_", [False, True])
def test_indexed_writes(set_):
    v = pt.vector("v")
    idx = pt.lvector("idx")
    n = pt.lscalar("n")
    fn = pt.set_subtensor if set_ else pt.inc_subtensor
    out = fn(pt.zeros(n)[idx], pt.exp(v) * 3)
    idx_val = rng.permutation(20)[:15] if set_ else rng.integers(-20, 20, 60)  # duplicates accumulate
    f = compare([v, idx, n], out, [rng.normal(size=len(idx_val)), idx_val, 20])
    assert fused(f)


def test_two_dim_loop_with_partial_reductions():
    x = pt.matrix("x")  # (rows, groups)
    w = pt.matrix("w")  # (rows, obs)
    idx = pt.lvector("idx")
    g = x[:, idx] * w  # read along axis 1
    outs = [g.sum(axis=0), g.sum(axis=1), (g**2).sum()]
    f = compare([x, w, idx], outs, [rng.normal(size=(4, 6)), rng.normal(size=(4, 30)), rng.integers(0, 6, 30)])
    assert fused(f)


def test_hierarchical_logp_and_grad_is_fused():
    n_groups, n_obs = 12, 300
    mu, log_tau = pt.dscalars("mu", "log_tau")
    theta = pt.vector("theta")
    group = pt.lvector("group")
    y = pt.vector("y")
    eta = mu + pt.exp(log_tau) * theta
    logp = (-0.5 * (y - eta[group]) ** 2).sum() + (-0.5 * theta**2).sum() - 0.5 * mu**2 + log_tau
    grads = pytensor.grad(logp, [mu, log_tau, theta])
    vals = [np.array(0.2), np.array(-0.3), rng.normal(size=n_groups), rng.integers(0, n_groups, n_obs), rng.normal(size=n_obs)]
    f = compare([mu, log_tau, theta, group, y], [logp, *grads], vals)
    sched = fused(f)
    assert any("reads=1" in d and "inc" in d and "reduce" in d for d in sched), sched


def test_out_of_bounds_index_raises():
    x = pt.vector("x")
    idx = pt.lvector("idx")
    f = pytensor.function([x, idx], (x[idx] * 2).sum(), mode=LLVM)
    assert fused(f)
    assert f(np.arange(3.0), np.array([0, -3, 2])) == pytest.approx(2 * (0 + 0 + 2))
    with pytest.raises(Exception, match="out of bounds"):
        f(np.arange(3.0), np.array([0, 3]))


def test_reduction_only_fusion():
    x = pt.vector("x")
    m = pt.matrix("m")
    f = compare([x], (pt.exp(x) * 2).sum(), [rng.normal(size=1001)])
    assert fused(f)
    # inner dim reduced (register accumulator) and outer dim reduced, C and F inputs
    outs = [(pt.log1p(m**2)).sum(axis=1), (m * 3).max(axis=0), pt.exp(m).sum()]
    for v in [rng.normal(size=(37, 53)), np.asfortranarray(rng.normal(size=(37, 53))), rng.normal(size=(74, 106))[::2, ::2]]:
        compare([m], outs, [v])


def test_unfused_patterns_are_inlined_natively():
    x = pt.vector("x")
    m = pt.lmatrix("m")  # 2-d index array: not fused natively, inlined instead
    out = (pt.exp(x[m]) * 2).sum()
    f = compare([x, m], out, [rng.normal(size=5), rng.integers(0, 5, (3, 4))])
    assert not [d for d in f.maker.linker.compiled.info()["schedule"] if d.startswith("foreign")]
