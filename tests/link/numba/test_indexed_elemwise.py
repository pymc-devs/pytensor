"""Tests for IndexedElemwise fusion (indexed reads in Elemwise loops)."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import get_mode
from pytensor.tensor.rewriting.indexed_elemwise import IndexedElemwise
from pytensor.tensor.subtensor import advanced_subtensor1


numba = pytest.importorskip("numba")

NUMBA_MODE = get_mode("NUMBA")
NUMBA_NO_FUSION = NUMBA_MODE.excluding("fuse_indexed_into_elemwise")


def fused_and_unfused(inputs, output):
    """Compile fused and unfused versions of a graph."""
    fn = pytensor.function(inputs, output, mode=NUMBA_MODE, trust_input=True)
    fn_u = pytensor.function(inputs, output, mode=NUMBA_NO_FUSION, trust_input=True)
    return fn, fn_u


def assert_fused(fn):
    """Assert that the compiled graph contains an IndexedElemwise node."""
    assert any(isinstance(n.op, IndexedElemwise) for n in fn.maker.fgraph.toposort()), (
        "IndexedElemwise not found in fused graph"
    )


class TestIndexedReadFusion:
    """Test indexed reads (AdvancedSubtensor1) fused into Elemwise."""

    def test_single_index_axis0(self):
        rng = np.random.default_rng(42)
        idx = rng.integers(85, size=919).astype(np.int64)
        x = pt.vector("x", shape=(85,))
        y = pt.vector("y", shape=(919,))
        fn, fn_u = fused_and_unfused([x, y], advanced_subtensor1(x, idx) + y)
        assert_fused(fn)
        xv, yv = rng.normal(size=(85,)), rng.normal(size=(919,))
        np.testing.assert_allclose(fn(xv, yv), fn_u(xv, yv), rtol=1e-10)

    def test_multiple_gathered_sources(self):
        rng = np.random.default_rng(42)
        idx = rng.integers(85, size=919).astype(np.int64)
        x1 = pt.vector("x1", shape=(85,))
        x2 = pt.vector("x2", shape=(85,))
        y = pt.vector("y", shape=(919,))
        fn, fn_u = fused_and_unfused(
            [x1, x2, y], advanced_subtensor1(x1, idx) + advanced_subtensor1(x2, idx) + y
        )
        assert_fused(fn)
        xv1, xv2, yv = (
            rng.normal(size=(85,)),
            rng.normal(size=(85,)),
            rng.normal(size=(919,)),
        )
        np.testing.assert_allclose(fn(xv1, xv2, yv), fn_u(xv1, xv2, yv), rtol=1e-10)

    def test_broadcast_index_axis0(self):
        """Static shape=(1,) index on axis 0 broadcasts against larger direct input."""
        x = pt.vector("x", shape=(100,))
        y = pt.vector("y", shape=(50,))
        idx = np.array([5], dtype=np.int64)  # shape (1,), broadcastable
        out = advanced_subtensor1(x, idx) + y
        fn, fn_u = fused_and_unfused([x, y], out)
        assert_fused(fn)
        xv = np.arange(100, dtype="float64")
        yv = np.ones(50)
        np.testing.assert_allclose(fn(xv, yv), fn_u(xv, yv), rtol=1e-10)

    def test_negative_indices(self):
        """Negative indices must be handled correctly (sign-extended, not zero-extended)."""
        rng = np.random.default_rng(42)
        # Use unknown static shape so negative indices can't be canonicalized away
        x = pt.vector("x")
        y = pt.vector("y")
        idx = pt.vector("idx", dtype="int64")
        fn, fn_u = fused_and_unfused([x, idx, y], x[idx] + y)
        assert_fused(fn)
        xv = rng.normal(size=100)
        # Negative indices: -1 means last element, -2 second to last, etc.
        idxv = np.array([-1, -2, -3, 0, 1], dtype=np.int64)
        yv = rng.normal(size=5)
        np.testing.assert_allclose(fn(xv, idxv, yv), fn_u(xv, idxv, yv), rtol=1e-10)


class TestShapeValidation:
    """Test that mismatched index/input shapes raise runtime errors.

    All inputs use ``shape=(None,)`` so shapes are unknown at compile time.
    The fused loop's ``compute_itershape`` must catch mismatches at runtime.
    """

    def test_mismatched_index_and_direct_input(self):
        """Index length doesn't match direct input on the same loop dim."""
        x = pt.vector("x", shape=(None,))
        y = pt.vector("y", shape=(None,))
        idx = pt.vector("idx", dtype="int64", shape=(None,))
        out = x[idx] + y
        fn = pytensor.function([x, idx, y], out, mode=NUMBA_MODE, trust_input=True)
        assert_fused(fn)
        # Matching: idx=50, y=50 — should work
        fn(np.zeros(100), np.zeros(50, dtype=np.int64), np.zeros(50))
        # Mismatched: idx=50, y=49 — should error
        with pytest.raises(Exception):
            fn(np.zeros(100), np.zeros(50, dtype=np.int64), np.zeros(49))

    def test_runtime_broadcast_on_index_dim(self):
        """Symbolic shapes that happen to be 1 at runtime — broadcast check."""
        x = pt.vector("x", shape=(None,))
        y = pt.vector("y", shape=(None,))
        idx = pt.vector("idx", dtype="int64", shape=(None,))
        out = x[idx] + y
        fn = pytensor.function([x, idx, y], out, mode=NUMBA_MODE, trust_input=True)
        assert_fused(fn)
        # Both idx and y have length 1 — should work (both agree on dim 0)
        result = fn(np.zeros(100), np.zeros(1, dtype=np.int64), np.zeros(1))
        assert result.shape == (1,)
        # idx=1, y=5 — should error (shape mismatch, no static broadcast info)
        with pytest.raises(Exception):
            fn(np.zeros(100), np.zeros(1, dtype=np.int64), np.zeros(5))
