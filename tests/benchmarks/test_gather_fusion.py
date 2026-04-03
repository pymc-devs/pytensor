"""Micro-benchmarks for Elemwise fusion with indexed reads and updates.

Tests the benefit of fusing AdvancedSubtensor1 (indexed reads) and
AdvancedIncSubtensor1 (indexed updates) into Elemwise loops, avoiding
materialization of intermediate arrays.
"""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import config
from pytensor.compile.mode import get_mode
from pytensor.tensor.rewriting.indexed_elemwise import IndexedElemwise
from pytensor.tensor.subtensor import AdvancedIncSubtensor1, advanced_subtensor1


@pytest.fixture(
    params=[
        (85, 919, 2, 6),  # radon-like: small
        (1000, 100_000, 2, 4),  # medium
    ],
    ids=["small-85x919", "medium-1Kx100K"],
)
def gather_benchmark_setup(request):
    n_bins, n_data, n_gathered, n_direct = request.param

    rng = np.random.default_rng(42)
    idx = rng.integers(n_bins, size=n_data).astype(np.int64)
    idx.sort()

    sources = [pt.vector(f"src_{i}", shape=(n_bins,)) for i in range(n_gathered)]
    directs = [pt.vector(f"dir_{i}", shape=(n_data,)) for i in range(n_direct)]

    terms = [advanced_subtensor1(s, idx) for s in sources] + directs
    out = terms[0]
    for t in terms[1:]:
        out = out + t

    inputs = sources + directs
    numba_mode = get_mode("NUMBA")

    fn_fused = pytensor.function(inputs, out, mode=numba_mode, trust_input=True)
    fn_unfused = pytensor.function(
        inputs,
        out,
        mode=numba_mode.excluding("fuse_indexed_into_elemwise"),
        trust_input=True,
    )

    assert any(
        isinstance(n.op, IndexedElemwise) for n in fn_fused.maker.fgraph.toposort()
    ), "IndexedElemwise not found in fused graph"
    assert not any(
        isinstance(n.op, IndexedElemwise) for n in fn_unfused.maker.fgraph.toposort()
    ), "IndexedElemwise found in unfused graph"

    rng = np.random.default_rng(1)
    vals = [rng.normal(size=inp.type.shape).astype(config.floatX) for inp in inputs]

    np.testing.assert_allclose(fn_fused(*vals), fn_unfused(*vals), rtol=1e-10)

    return fn_fused, fn_unfused, vals


def test_gather_fusion_fused(gather_benchmark_setup, benchmark):
    fn_fused, _, vals = gather_benchmark_setup
    fn_fused(*vals)  # warmup
    benchmark(fn_fused, *vals)


def test_gather_fusion_unfused(gather_benchmark_setup, benchmark):
    _, fn_unfused, vals = gather_benchmark_setup
    fn_unfused(*vals)  # warmup
    benchmark(fn_unfused, *vals)


# ---------------------------------------------------------------------------
# Scatter output fusion benchmarks
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        (85, 919, 2, 6, "inc"),  # radon-like: small, inc mode
        (85, 919, 2, 6, "set"),  # radon-like: small, set mode
        (1000, 100_000, 2, 4, "inc"),  # medium, inc mode
    ],
    ids=["small-85x919-inc", "small-85x919-set", "medium-1Kx100K-inc"],
)
def scatter_benchmark_setup(request):
    n_bins, n_data, n_gathered, n_direct, mode = request.param

    rng = np.random.default_rng(42)
    idx = rng.integers(n_bins, size=n_data).astype(np.int64)
    idx.sort()

    sources = [pt.vector(f"src_{i}", shape=(n_bins,)) for i in range(n_gathered)]
    directs = [pt.vector(f"dir_{i}", shape=(n_data,)) for i in range(n_direct)]
    target = pt.vector("target", shape=(n_bins,))

    terms = [advanced_subtensor1(s, idx) for s in sources] + directs
    elemwise_out = terms[0]
    for t in terms[1:]:
        elemwise_out = elemwise_out + t

    if mode == "inc":
        out = pt.inc_subtensor(target[idx], elemwise_out)
    else:
        out = pt.set_subtensor(target[idx], elemwise_out)

    inputs = sources + directs + [target]
    numba_mode = get_mode("NUMBA")

    fn_fused = pytensor.function(inputs, out, mode=numba_mode, trust_input=True)
    fn_unfused = pytensor.function(
        inputs,
        out,
        mode=numba_mode.excluding("fuse_indexed_into_elemwise"),
        trust_input=True,
    )

    assert any(
        isinstance(n.op, IndexedElemwise) for n in fn_fused.maker.fgraph.toposort()
    ), "IndexedElemwise not found in fused graph"
    assert not any(
        isinstance(n.op, AdvancedIncSubtensor1)
        for n in fn_fused.maker.fgraph.toposort()
    ), "AdvancedIncSubtensor1 still present in fused graph"
    assert not any(
        isinstance(n.op, IndexedElemwise) for n in fn_unfused.maker.fgraph.toposort()
    ), "IndexedElemwise found in unfused graph"

    rng = np.random.default_rng(1)
    vals = [rng.normal(size=inp.type.shape).astype(config.floatX) for inp in inputs]

    np.testing.assert_allclose(fn_fused(*vals), fn_unfused(*vals), rtol=1e-10)

    return fn_fused, fn_unfused, vals


def test_scatter_fusion_fused(scatter_benchmark_setup, benchmark):
    fn_fused, _, vals = scatter_benchmark_setup
    fn_fused(*vals)  # warmup
    benchmark(fn_fused, *vals)


def test_scatter_fusion_unfused(scatter_benchmark_setup, benchmark):
    _, fn_unfused, vals = scatter_benchmark_setup
    fn_unfused(*vals)  # warmup
    benchmark(fn_unfused, *vals)
