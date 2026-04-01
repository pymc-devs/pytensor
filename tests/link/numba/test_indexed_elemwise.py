"""Tests for IndexedElemwise fusion (indexed reads and updates in Elemwise loops)."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import config
from pytensor.compile.mode import get_mode
from pytensor.tensor.rewriting.indexed_elemwise import IndexedElemwise
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    advanced_subtensor1,
)


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


# ============================================================
# Correctness tests — indexed reads
# ============================================================


class TestIndexedReadFusion:
    """Test indexed reads (AdvancedSubtensor1 / AdvancedSubtensor) fused into Elemwise."""

    def test_single_index_axis0(self):
        rng = np.random.default_rng(42)
        idx = rng.integers(85, size=919).astype(np.int64)
        x = pt.vector("x", shape=(85,))
        y = pt.vector("y", shape=(919,))
        fn, fn_u = fused_and_unfused([x, y], advanced_subtensor1(x, idx) + y)
        assert_fused(fn)
        xv, yv = rng.normal(size=(85,)), rng.normal(size=(919,))
        np.testing.assert_allclose(fn(xv, yv), fn_u(xv, yv), rtol=1e-10)

    def test_single_index_axis1(self):
        rng = np.random.default_rng(42)
        idx = rng.integers(85, size=919).astype(np.int64)
        x = pt.matrix("x", shape=(3, 85))
        y = pt.matrix("y", shape=(3, 919))
        fn, fn_u = fused_and_unfused([x, y], x[:, idx] + y)
        assert_fused(fn)
        xv, yv = rng.normal(size=(3, 85)), rng.normal(size=(3, 919))
        np.testing.assert_allclose(fn(xv, yv), fn_u(xv, yv), rtol=1e-10)

    def test_multi_index_2d(self):
        rng = np.random.default_rng(42)
        x = pt.matrix("x", shape=(100, 200))
        ir = pt.vector("ir", dtype="int64", shape=(50,))
        ic = pt.vector("ic", dtype="int64", shape=(50,))
        y = pt.vector("y", shape=(50,))
        fn, fn_u = fused_and_unfused([x, ir, ic, y], x[ir, ic] + y)
        assert_fused(fn)
        xv = rng.normal(size=(100, 200))
        rv, cv = rng.integers(100, size=50), rng.integers(200, size=50)
        yv = rng.normal(size=(50,))
        np.testing.assert_allclose(fn(xv, rv, cv, yv), fn_u(xv, rv, cv, yv), rtol=1e-10)

    def test_multi_index_3d_trailing_dim(self):
        rng = np.random.default_rng(42)
        x3 = pt.tensor3("x3", shape=(100, 200, 5))
        ir = pt.vector("ir", dtype="int64", shape=(50,))
        ic = pt.vector("ic", dtype="int64", shape=(50,))
        z = pt.matrix("z", shape=(50, 5))
        fn, fn_u = fused_and_unfused([x3, ir, ic, z], x3[ir, ic] + z)
        assert_fused(fn)
        xv = rng.normal(size=(100, 200, 5))
        rv, cv = rng.integers(100, size=50), rng.integers(200, size=50)
        zv = rng.normal(size=(50, 5))
        np.testing.assert_allclose(fn(xv, rv, cv, zv), fn_u(xv, rv, cv, zv), rtol=1e-10)

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

    def test_broadcast_index_axis1(self):
        """Static shape=(1,) index on axis 1 broadcasts against larger direct input."""
        x = pt.matrix("x", shape=(3, 100))
        y = pt.matrix("y", shape=(3, 50))
        idx = np.array([5], dtype=np.int64)
        out = x[:, idx] + y  # x[:, idx] has shape (3, 1), broadcasts to (3, 50)
        fn, fn_u = fused_and_unfused([x, y], out)
        assert_fused(fn)
        xv = np.arange(300.0).reshape(3, 100)
        yv = np.ones((3, 50))
        np.testing.assert_allclose(fn(xv, yv), fn_u(xv, yv), rtol=1e-10)

    def test_nd_index_axis0(self):
        """2D matrix index on axis 0."""
        rng = np.random.default_rng(42)
        x = pt.vector("x", shape=(100,))
        mat_idx = pt.matrix("mat_idx", dtype="int64", shape=(10, 5))
        y = pt.matrix("y", shape=(10, 5))
        fn, fn_u = fused_and_unfused([x, mat_idx, y], pt.exp(x[mat_idx]) + y)
        assert_fused(fn)
        xv = rng.normal(size=(100,))
        iv = rng.integers(100, size=(10, 5)).astype(np.int64)
        yv = rng.normal(size=(10, 5))
        np.testing.assert_allclose(fn(xv, iv, yv), fn_u(xv, iv, yv), rtol=1e-10)

    def test_nd_index_axis1(self):
        """2D matrix index on axis 1 (via undo_take_reshape_for_fusion)."""
        rng = np.random.default_rng(42)
        x = pt.matrix("x", shape=(3, 100))
        mat_idx = pt.matrix("mat_idx", dtype="int64", shape=(10, 5))
        fn, fn_u = fused_and_unfused([x, mat_idx], pt.exp(x[:, mat_idx]))
        assert_fused(fn)
        xv = rng.normal(size=(3, 100))
        iv = rng.integers(100, size=(10, 5)).astype(np.int64)
        np.testing.assert_allclose(fn(xv, iv), fn_u(xv, iv), rtol=1e-10)

    def test_nd_index_with_trailing_dims(self):
        """2D index on axis 0 with trailing source dims."""
        rng = np.random.default_rng(42)
        x = pt.matrix("x", shape=(100, 7))
        mat_idx = pt.matrix("mat_idx", dtype="int64", shape=(10, 5))
        fn, fn_u = fused_and_unfused([x, mat_idx], pt.exp(x[mat_idx]))
        assert_fused(fn)
        xv = rng.normal(size=(100, 7))
        iv = rng.integers(100, size=(10, 5)).astype(np.int64)
        np.testing.assert_allclose(fn(xv, iv), fn_u(xv, iv), rtol=1e-10)

    def test_nd_index_broadcast(self):
        """Broadcastable 2D index (shape (1, C)) broadcasts against direct input."""
        rng = np.random.default_rng(42)
        x = pt.vector("x", shape=(100,))
        bc_idx = pt.matrix("bc_idx", dtype="int64", shape=(1, 5))
        y = pt.matrix("y", shape=(10, 5))
        fn, fn_u = fused_and_unfused([x, bc_idx, y], x[bc_idx] + y)
        assert_fused(fn)
        xv = rng.normal(size=(100,))
        bcv = rng.integers(100, size=(1, 5)).astype(np.int64)
        yv = rng.normal(size=(10, 5))
        np.testing.assert_allclose(fn(xv, bcv, yv), fn_u(xv, bcv, yv), rtol=1e-10)

    def test_scalar_and_vector_index(self):
        """x[scalar_idx, vector_idx] — 0-d index broadcasts with 1-d index."""
        rng = np.random.default_rng(42)
        x = pt.matrix("x", shape=(100, 200))
        scalar_idx = pt.scalar("scalar_idx", dtype="int64")
        vec_idx = pt.vector("vec_idx", dtype="int64", shape=(50,))
        y = pt.vector("y", shape=(50,))
        fn, fn_u = fused_and_unfused(
            [x, scalar_idx, vec_idx, y], x[scalar_idx, vec_idx] + y
        )
        assert_fused(fn)
        xv = rng.normal(size=(100, 200))
        sv = np.array(rng.integers(100), dtype=np.int64)
        vv = rng.integers(200, size=50).astype(np.int64)
        yv = rng.normal(size=50)
        np.testing.assert_allclose(fn(xv, sv, vv, yv), fn_u(xv, sv, vv, yv), rtol=1e-10)

    def test_vector_and_scalar_index(self):
        """x[vector_idx, scalar_idx] — reversed order."""
        rng = np.random.default_rng(42)
        x = pt.matrix("x", shape=(100, 200))
        vec_idx = pt.vector("vec_idx", dtype="int64", shape=(50,))
        scalar_idx = pt.scalar("scalar_idx", dtype="int64")
        y = pt.vector("y", shape=(50,))
        fn, fn_u = fused_and_unfused(
            [x, vec_idx, scalar_idx, y], x[vec_idx, scalar_idx] + y
        )
        assert_fused(fn)
        xv = rng.normal(size=(100, 200))
        vv = rng.integers(100, size=50).astype(np.int64)
        sv = np.array(rng.integers(200), dtype=np.int64)
        yv = rng.normal(size=50)
        np.testing.assert_allclose(fn(xv, vv, sv, yv), fn_u(xv, vv, sv, yv), rtol=1e-10)

    def test_scalar_and_vector_index_with_trailing_dim(self):
        """x[scalar_idx, vector_idx] on a 3-d tensor with trailing dim."""
        rng = np.random.default_rng(42)
        x = pt.tensor3("x", shape=(100, 200, 7))
        scalar_idx = pt.scalar("scalar_idx", dtype="int64")
        vec_idx = pt.vector("vec_idx", dtype="int64", shape=(50,))
        z = pt.matrix("z", shape=(50, 7))
        fn, fn_u = fused_and_unfused(
            [x, scalar_idx, vec_idx, z], x[scalar_idx, vec_idx] + z
        )
        assert_fused(fn)
        xv = rng.normal(size=(100, 200, 7))
        sv = np.array(rng.integers(100), dtype=np.int64)
        vv = rng.integers(200, size=50).astype(np.int64)
        zv = rng.normal(size=(50, 7))
        np.testing.assert_allclose(fn(xv, sv, vv, zv), fn_u(xv, sv, vv, zv), rtol=1e-10)

    def test_all_scalar_indices(self):
        """AdvancedSubtensor with all 0-d indices (degenerate case)."""
        rng = np.random.default_rng(42)
        x = pt.matrix("x", shape=(100, 200))
        si0 = pt.scalar("si0", dtype="int64")
        si1 = pt.scalar("si1", dtype="int64")
        y = pt.scalar("y", dtype="float64")
        adv_sub = AdvancedSubtensor(idx_list=(0, 1))(x, si0, si1)
        fn, fn_u = fused_and_unfused([x, si0, si1, y], adv_sub + y)
        assert_fused(fn)
        xv = rng.normal(size=(100, 200))
        s0 = np.array(rng.integers(100), dtype=np.int64)
        s1 = np.array(rng.integers(200), dtype=np.int64)
        yv = np.array(rng.normal())
        np.testing.assert_allclose(fn(xv, s0, s1, yv), fn_u(xv, s0, s1, yv), rtol=1e-10)

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


# ============================================================
# Correctness tests — indexed updates
# ============================================================


class TestIndexedUpdateFusion:
    """Test indexed updates (AdvancedIncSubtensor1) fused into Elemwise."""

    def test_no_fusion_when_val_broadcasts_against_target(self):
        """Don't fuse (yet) if elemwise output broadcasts against target's trailing axes.

        TODO: support this by making the update output's core_ndim > 0.
        """
        rng = np.random.default_rng(42)
        idx = rng.integers(5, size=10).astype(np.int64)
        target = pt.matrix("target", shape=(5, 10))
        x = pt.vector("x", shape=(85,))
        y = pt.vector("y", shape=(10,))
        elemwise_out = advanced_subtensor1(x, idx) + y  # shape (10,)
        out = pt.inc_subtensor(target[idx], elemwise_out)
        fn, fn_u = fused_and_unfused([x, y, target], out)
        # Write not fused — val (10,) would broadcast to (10, 10) in the target.
        # Read fusion still creates an IndexedElemwise, but the
        # AdvancedIncSubtensor1 must remain outside it.
        assert any(
            isinstance(n.op, AdvancedIncSubtensor1) for n in fn.maker.fgraph.toposort()
        )
        xv = rng.normal(size=(85,))
        yv = rng.normal(size=(10,))
        tv = rng.normal(size=(5, 10))
        np.testing.assert_allclose(fn(xv, yv, tv), fn_u(xv, yv, tv), rtol=1e-10)

    def test_no_fusion_when_val_broadcasts_along_index_dim(self):
        """Don't fuse if val is broadcastable on the index loop dim.

        When val is constant across the index (e.g. shape (1,) with idx
        of size 10), fusing would recompute the same Elemwise result at
        every index position.  Better to compute once and scatter.
        """
        rng = np.random.default_rng(42)
        idx = rng.integers(5, size=10).astype(np.int64)
        target = pt.vector("target", shape=(5,))
        x = pt.tensor("x", shape=(1,))
        out = pt.inc_subtensor(target[idx], pt.exp(x))
        fn, fn_u = fused_and_unfused([x, target], out)
        # Write not fused — val broadcasts along the index loop dim.
        assert any(
            isinstance(n.op, AdvancedIncSubtensor1) for n in fn.maker.fgraph.toposort()
        )
        xv = rng.normal(size=(1,))
        tv = rng.normal(size=(5,))
        np.testing.assert_allclose(fn(xv, tv), fn_u(xv, tv), rtol=1e-10)

    def test_inc_subtensor(self):
        rng = np.random.default_rng(42)
        idx = rng.integers(85, size=919).astype(np.int64)
        x = pt.vector("x", shape=(85,))
        y = pt.vector("y", shape=(919,))
        t = pt.vector("t", shape=(85,))
        out = pt.inc_subtensor(t[idx], advanced_subtensor1(x, idx) + y)
        fn, fn_u = fused_and_unfused([x, y, t], out)
        assert_fused(fn)
        xv, yv, tv = (
            rng.normal(size=(85,)),
            rng.normal(size=(919,)),
            rng.normal(size=(85,)),
        )
        np.testing.assert_allclose(fn(xv, yv, tv), fn_u(xv, yv, tv), rtol=1e-10)

    def test_set_subtensor(self):
        rng = np.random.default_rng(42)
        idx = rng.integers(85, size=919).astype(np.int64)
        x = pt.vector("x", shape=(85,))
        y = pt.vector("y", shape=(919,))
        t = pt.vector("t", shape=(85,))
        out = pt.set_subtensor(t[idx], advanced_subtensor1(x, idx) + y)
        fn, fn_u = fused_and_unfused([x, y, t], out)
        assert_fused(fn)
        xv, yv, tv = (
            rng.normal(size=(85,)),
            rng.normal(size=(919,)),
            rng.normal(size=(85,)),
        )
        np.testing.assert_allclose(fn(xv, yv, tv), fn_u(xv, yv, tv), rtol=1e-10)

    def test_target_not_modified_when_non_inplace(self):
        """Non-inplace scatter should not modify the original target."""
        rng = np.random.default_rng(42)
        idx = rng.integers(85, size=919).astype(np.int64)
        x = pt.vector("x", shape=(85,))
        y = pt.vector("y", shape=(919,))
        t = pt.vector("t", shape=(85,))
        out = pt.inc_subtensor(t[idx], advanced_subtensor1(x, idx) + y)
        fn = pytensor.function([x, y, t], out, mode=NUMBA_MODE, trust_input=True)
        xv, yv = rng.normal(size=(85,)), rng.normal(size=(919,))
        tv = rng.normal(size=(85,))
        tv_copy = tv.copy()
        fn(xv, yv, tv)
        np.testing.assert_array_equal(tv, tv_copy)

    def test_multi_index_inc_subtensor(self):
        rng = np.random.default_rng(42)
        target = pt.matrix("target", shape=(100, 200))
        ir = pt.vector("ir", dtype="int64", shape=(50,))
        ic = pt.vector("ic", dtype="int64", shape=(50,))
        x = pt.vector("x", shape=(50,))
        out = pt.inc_subtensor(target[ir, ic], pt.exp(x))
        fn, fn_u = fused_and_unfused([target, ir, ic, x], out)
        assert_fused(fn)
        tv = rng.normal(size=(100, 200))
        rv = rng.integers(100, size=50).astype(np.int64)
        cv = rng.integers(200, size=50).astype(np.int64)
        xv = rng.normal(size=50)
        np.testing.assert_allclose(fn(tv, rv, cv, xv), fn_u(tv, rv, cv, xv), rtol=1e-10)

    def test_multi_index_set_subtensor(self):
        rng = np.random.default_rng(42)
        target = pt.matrix("target", shape=(100, 200))
        ir = pt.vector("ir", dtype="int64", shape=(50,))
        ic = pt.vector("ic", dtype="int64", shape=(50,))
        x = pt.vector("x", shape=(50,))
        out = pt.set_subtensor(target[ir, ic], pt.exp(x))
        fn, fn_u = fused_and_unfused([target, ir, ic, x], out)
        assert_fused(fn)
        tv = rng.normal(size=(100, 200))
        rv = rng.integers(100, size=50).astype(np.int64)
        cv = rng.integers(200, size=50).astype(np.int64)
        xv = rng.normal(size=50)
        np.testing.assert_allclose(fn(tv, rv, cv, xv), fn_u(tv, rv, cv, xv), rtol=1e-10)

    def test_multi_index_write_with_trailing_dims(self):
        rng = np.random.default_rng(42)
        target = pt.tensor3("target", shape=(100, 200, 5))
        ir = pt.vector("ir", dtype="int64", shape=(50,))
        ic = pt.vector("ic", dtype="int64", shape=(50,))
        x = pt.matrix("x", shape=(50, 5))
        out = pt.inc_subtensor(target[ir, ic], pt.exp(x))
        fn, fn_u = fused_and_unfused([target, ir, ic, x], out)
        assert_fused(fn)
        tv = rng.normal(size=(100, 200, 5))
        rv = rng.integers(100, size=50).astype(np.int64)
        cv = rng.integers(200, size=50).astype(np.int64)
        xv = rng.normal(size=(50, 5))
        np.testing.assert_allclose(fn(tv, rv, cv, xv), fn_u(tv, rv, cv, xv), rtol=1e-10)

    def test_combined_multi_index_read_write(self):
        """Read and write share the same multi-index arrays."""
        rng = np.random.default_rng(42)
        target = pt.matrix("target", shape=(100, 200))
        source = pt.matrix("source", shape=(100, 200))
        ir = pt.vector("ir", dtype="int64", shape=(50,))
        ic = pt.vector("ic", dtype="int64", shape=(50,))
        x = pt.vector("x", shape=(50,))
        out = pt.inc_subtensor(target[ir, ic], source[ir, ic] + x)
        fn, fn_u = fused_and_unfused([target, source, ir, ic, x], out)
        assert_fused(fn)
        tv = rng.normal(size=(100, 200))
        sv = rng.normal(size=(100, 200))
        rv = rng.integers(100, size=50).astype(np.int64)
        cv = rng.integers(200, size=50).astype(np.int64)
        xv = rng.normal(size=50)
        np.testing.assert_allclose(
            fn(tv, sv, rv, cv, xv), fn_u(tv, sv, rv, cv, xv), rtol=1e-10
        )

    def test_scalar_and_vector_index_inc(self):
        """inc_subtensor with x[scalar_idx, vector_idx] — 0-d and 1-d indices."""
        rng = np.random.default_rng(42)
        target = pt.matrix("target", shape=(100, 200))
        scalar_idx = pt.scalar("scalar_idx", dtype="int64")
        vec_idx = pt.vector("vec_idx", dtype="int64", shape=(50,))
        x = pt.vector("x", shape=(50,))
        out = pt.inc_subtensor(target[scalar_idx, vec_idx], pt.exp(x))
        fn, fn_u = fused_and_unfused([target, scalar_idx, vec_idx, x], out)
        assert_fused(fn)
        tv = rng.normal(size=(100, 200))
        sv = np.array(rng.integers(100), dtype=np.int64)
        vv = rng.integers(200, size=50).astype(np.int64)
        xv = rng.normal(size=50)
        np.testing.assert_allclose(fn(tv, sv, vv, xv), fn_u(tv, sv, vv, xv), rtol=1e-10)


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

    def test_mismatched_multi_index_lengths(self):
        """Two index arrays in a multi-index have different lengths."""
        x = pt.matrix("x", shape=(None, None))
        ir = pt.vector("ir", dtype="int64", shape=(None,))
        ic = pt.vector("ic", dtype="int64", shape=(None,))
        y = pt.vector("y", shape=(None,))
        out = x[ir, ic] + y
        fn = pytensor.function([x, ir, ic, y], out, mode=NUMBA_MODE, trust_input=True)
        assert_fused(fn)
        # Matching: all 50 — should work
        fn(
            np.zeros((100, 200)),
            np.zeros(50, dtype=np.int64),
            np.zeros(50, dtype=np.int64),
            np.zeros(50),
        )
        # Mismatched: ir=50, ic=49 — should error
        with pytest.raises(Exception):
            fn(
                np.zeros((100, 200)),
                np.zeros(50, dtype=np.int64),
                np.zeros(49, dtype=np.int64),
                np.zeros(50),
            )

    def test_mismatched_index_vs_direct_on_non_indexed_axis(self):
        """Index and direct input disagree on a non-indexed (trailing) axis."""
        x = pt.tensor3("x", shape=(None, None, None))
        ir = pt.vector("ir", dtype="int64", shape=(None,))
        ic = pt.vector("ic", dtype="int64", shape=(None,))
        z = pt.matrix("z", shape=(None, None))
        out = x[ir, ic] + z  # result shape (N, trailing)
        fn = pytensor.function([x, ir, ic, z], out, mode=NUMBA_MODE, trust_input=True)
        assert_fused(fn)
        # Matching: trailing dim = 5 for both
        fn(
            np.zeros((10, 20, 5)),
            np.zeros(3, dtype=np.int64),
            np.zeros(3, dtype=np.int64),
            np.zeros((3, 5)),
        )
        # Mismatched: x trailing=5, z trailing=4
        with pytest.raises(Exception):
            fn(
                np.zeros((10, 20, 5)),
                np.zeros(3, dtype=np.int64),
                np.zeros(3, dtype=np.int64),
                np.zeros((3, 4)),
            )

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

    def test_shared_read_write_index_no_broadcast(self):
        """A shared read+write index must not broadcast at runtime."""
        x = pt.vector("x", shape=(None,))
        y = pt.vector("y", shape=(None,))
        t = pt.vector("t", shape=(None,))
        idx = pt.vector("idx", dtype="int64", shape=(None,))
        out = pt.inc_subtensor(t[idx], x[idx] + y)
        fn = pytensor.function([x, y, t, idx], out, mode=NUMBA_MODE, trust_input=True)
        fn(np.zeros(100), np.zeros(50), np.zeros(100), np.arange(50, dtype=np.int64))
        with pytest.raises(Exception):
            fn(
                np.zeros(100),
                np.zeros(50),
                np.zeros(100),
                np.array([0], dtype=np.int64),
            )

    def test_mismatched_nd_index_dims(self):
        """ND index shape doesn't match direct input shape."""
        x = pt.vector("x", shape=(None,))
        mat_idx = pt.matrix("mat_idx", dtype="int64", shape=(None, None))
        y = pt.matrix("y", shape=(None, None))
        out = x[mat_idx] + y
        fn = pytensor.function([x, mat_idx, y], out, mode=NUMBA_MODE, trust_input=True)
        assert_fused(fn)
        # Matching: mat_idx=(3,4), y=(3,4) — should work
        fn(np.zeros(100), np.zeros((3, 4), dtype=np.int64), np.zeros((3, 4)))
        # Mismatched: mat_idx=(3,4), y=(3,5) — should error
        with pytest.raises(Exception):
            fn(np.zeros(100), np.zeros((3, 4), dtype=np.int64), np.zeros((3, 5)))

    # ============================================================
    # Radon model integration test
    # ============================================================

    def test_write_only_index_no_broadcast(self):
        """A write-only index must not broadcast against the loop.

        target[idx] += exp(y) where idx is write-only (not used for reads).
        If idx has shape (1,) at runtime but y has shape (50,), the loop
        runs 50 iterations.  The index must not silently broadcast — it
        should error because the shapes don't match.
        """
        rng = np.random.default_rng(42)
        y = pt.vector("y", shape=(None,))
        t = pt.vector("t", shape=(None,))
        idx = pt.vector("idx", dtype="int64", shape=(None,))
        out = pt.inc_subtensor(t[idx], pt.exp(y))
        fn = pytensor.function([y, t, idx], out, mode=NUMBA_MODE, trust_input=True)
        fn_u = pytensor.function(
            [y, t, idx], out, mode=NUMBA_NO_FUSION, trust_input=True
        )
        assert_fused(fn)
        # Matching shapes — should work
        yv = rng.normal(size=50)
        tv = rng.normal(size=100)
        iv = rng.integers(100, size=50).astype(np.int64)
        np.testing.assert_allclose(fn(yv, tv, iv), fn_u(yv, tv, iv), rtol=1e-10)
        # Mismatched: idx=1, y=50 — should error
        with pytest.raises(Exception):
            fn(np.zeros(50), np.zeros(100), np.array([0], dtype=np.int64))


class TestRepeatedAccumulationIndices:
    """Test inc_subtensor with repeated indices (same position accumulated multiple times)."""

    def test_repeated_inc_subtensor(self):
        """inc_subtensor with repeated indices should accumulate correctly."""
        rng = np.random.default_rng(42)
        target = pt.vector("target", shape=(10,))
        x = pt.vector("x", shape=(20,))
        # Indices with repeats: multiple values go to the same target position
        idx = np.array(
            [0, 1, 2, 0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
            dtype=np.int64,
        )
        out = pt.inc_subtensor(target[idx], pt.exp(x))
        fn, fn_u = fused_and_unfused([target, x], out)
        assert_fused(fn)
        tv = rng.normal(size=10)
        xv = rng.normal(size=20)
        np.testing.assert_allclose(fn(tv, xv), fn_u(tv, xv), rtol=1e-10)

    def test_repeated_set_subtensor(self):
        """set_subtensor with repeated indices -- last write wins."""
        rng = np.random.default_rng(42)
        target = pt.vector("target", shape=(10,))
        x = pt.vector("x", shape=(5,))
        idx = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        out = pt.set_subtensor(target[idx], pt.exp(x))
        fn, fn_u = fused_and_unfused([target, x], out)
        assert_fused(fn)
        tv = rng.normal(size=10)
        xv = rng.normal(size=5)
        np.testing.assert_allclose(fn(tv, xv), fn_u(tv, xv), rtol=1e-10)

    def test_repeated_inc_with_read(self):
        """Combined read + inc_subtensor with repeated indices."""
        rng = np.random.default_rng(42)
        source = pt.vector("source", shape=(10,))
        target = pt.vector("target", shape=(10,))
        idx = np.array([0, 1, 0, 1, 2, 2, 3, 3], dtype=np.int64)
        out = pt.inc_subtensor(target[idx], source[idx] * 2.0)
        fn, fn_u = fused_and_unfused([source, target], out)
        assert_fused(fn)
        sv = rng.normal(size=10)
        tv = rng.normal(size=10)
        np.testing.assert_allclose(fn(sv, tv), fn_u(sv, tv), rtol=1e-10)


class TestRadonModel:
    """Test fusion on the radon hierarchical model (logp + gradient)."""

    def test_radon_model_correctness(self):
        import sys

        sys.path.insert(0, "tests/benchmarks")
        from test_compilation import create_radon_model

        joined_inputs, [model_logp, model_dlogp] = create_radon_model()
        fn, fn_u = fused_and_unfused([joined_inputs], [model_logp, model_dlogp])
        rng = np.random.default_rng(1)
        x = rng.normal(size=joined_inputs.type.shape).astype(config.floatX)
        results_fused = fn(x)
        results_unfused = fn_u(x)
        for i, (rf, ru) in enumerate(zip(results_fused, results_unfused)):
            np.testing.assert_allclose(rf, ru, rtol=1e-6, err_msg=f"Output {i}")

    def test_radon_model_no_unfused_indexing(self):
        """After fusion, no AdvancedSubtensor1 or AdvancedIncSubtensor1 should remain."""
        import sys

        sys.path.insert(0, "tests/benchmarks")
        from test_compilation import create_radon_model

        from pytensor.tensor.subtensor import AdvancedSubtensor1

        joined_inputs, [model_logp, model_dlogp] = create_radon_model()
        fn = pytensor.function(
            [joined_inputs],
            [model_logp, model_dlogp],
            mode=NUMBA_MODE,
            trust_input=True,
        )
        nodes = fn.maker.fgraph.toposort()
        assert not any(isinstance(n.op, AdvancedSubtensor1) for n in nodes)
        assert not any(isinstance(n.op, AdvancedIncSubtensor1) for n in nodes)
