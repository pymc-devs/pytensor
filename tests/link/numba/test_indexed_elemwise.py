"""Tests for IndexedElemwise fusion (indexed reads and updates in Elemwise loops)."""

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import Mode, function, get_mode
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.rewriting.indexed_elemwise import IndexedElemwise
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedSubtensor,
)


numba = pytest.importorskip("numba")

NUMBA_MODE = get_mode("NUMBA")
NUMBA_NO_FUSION = NUMBA_MODE.excluding("fuse_indexed_into_elemwise")


def fused_and_unfused(inputs, output):
    """Compile fused and unfused versions of a graph."""
    fn = function(inputs, output, mode=NUMBA_MODE, trust_input=True)
    fn_u = function(inputs, output, mode=NUMBA_NO_FUSION, trust_input=True)
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
        fn, fn_u = fused_and_unfused([x, y], x[idx] + y)
        assert_fused(fn)
        xv, yv = rng.normal(size=(85,)), rng.normal(size=(919,))
        np.testing.assert_allclose(fn(xv, yv), fn_u(xv, yv), rtol=1e-10)

    def test_multiple_read_sources(self):
        rng = np.random.default_rng(42)
        idx = rng.integers(85, size=919).astype(np.int64)
        x1 = pt.vector("x1", shape=(85,))
        x2 = pt.vector("x2", shape=(85,))
        y = pt.vector("y", shape=(919,))
        fn, fn_u = fused_and_unfused([x1, x2, y], x1[idx] + x2[idx] + y)
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
        idx = pt.vector("idx", dtype="int64", shape=(1,))
        out = x[idx] + y
        fn, fn_u = fused_and_unfused([x, y, idx], out)
        assert_fused(fn)
        xv = np.arange(100, dtype="float64")
        yv = np.ones(50)
        idxv = np.array([5], dtype=np.int64)
        np.testing.assert_allclose(fn(xv, yv, idxv), fn_u(xv, yv, idxv), rtol=1e-10)

    def test_broadcast_index_axis1(self):
        """Static shape=(1,) index on axis 1 broadcasts against larger direct input."""
        x = pt.matrix("x", shape=(3, 100))
        y = pt.matrix("y", shape=(3, 50))
        idx = pt.vector("idx", dtype="int64", shape=(1,))
        out = x[:, idx] + y  # x[:, idx] has shape (3, 1), broadcasts to (3, 50)
        fn, fn_u = fused_and_unfused([x, y, idx], out)
        assert_fused(fn)
        xv = np.arange(300.0).reshape(3, 100)
        yv = np.ones((3, 50))
        idxv = np.array([5], dtype=np.int64)
        np.testing.assert_allclose(fn(xv, yv, idxv), fn_u(xv, yv, idxv), rtol=1e-10)

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

    def test_regrouped_gather(self):
        """Gather over a (3, 4) index, regrouped to runtime shape (s0, s1).

        A detection that assumed the reshape restores the index shape would
        rewrite this to x[mat] — shape (3, 4, 5) instead of the requested
        (2, 6, 5). _unwrap_reshaped_take builds the (s0, s1) index from the
        reshape target instead, so the regrouping fuses and stays correct.
        """
        rng = np.random.default_rng(15)
        x = pt.matrix("x", shape=(8, 5))
        mat = pt.matrix("mat", dtype="int64", shape=(3, 4))
        s0, s1 = pt.lscalar("s0"), pt.lscalar("s1")
        out = pt.exp(x[mat.reshape((-1,))].reshape((s0, s1, x.shape[1])))
        fn, fn_u = fused_and_unfused([x, mat, s0, s1], out)
        assert_fused(fn)
        xv = rng.normal(size=(8, 5))
        mv = rng.integers(0, 8, size=(3, 4))
        res = fn(xv, mv, np.int64(2), np.int64(6))
        res_u = fn_u(xv, mv, np.int64(2), np.int64(6))
        assert res.shape == res_u.shape == (2, 6, 5)
        np.testing.assert_allclose(res, res_u, rtol=1e-10)

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

    def test_same_index_both_axes(self):
        """x[idx, idx] — same 1-D index on both axes (diagonal read)."""
        rng = np.random.default_rng(42)
        x = pt.matrix("x", shape=(10, 10))
        idx = pt.vector("idx", dtype="int64", shape=(5,))
        y = pt.vector("y", shape=(5,))
        fn, fn_u = fused_and_unfused([x, idx, y], pt.exp(x[idx, idx]) + y)
        assert_fused(fn)
        xv = rng.normal(size=(10, 10))
        iv = rng.integers(10, size=5).astype(np.int64)
        yv = rng.normal(size=5)
        np.testing.assert_allclose(fn(xv, iv, yv), fn_u(xv, iv, yv), rtol=1e-10)


class TestIndexedWriteFusion:
    """Test indexed updates (AdvancedIncSubtensor1) fused into Elemwise."""

    def test_no_fusion_when_idx_axes_outside_elemwise_loop(self):
        """Don't fuse if the indexed axes are not within the Elemwise loop.

        Here the index is on axis 0 of target(5, 10), but the Elemwise
        output (10,) corresponds to axis 1 (the non-indexed trailing axis).
        The indexed axis doesn't overlap with the Elemwise computation, so
        fusing would misalign which input dims map to which target dims.
        """
        rng = np.random.default_rng(42)
        idx = rng.integers(5, size=10).astype(np.int64)
        target = pt.matrix("target", shape=(5, 10))
        x = pt.vector("x", shape=(85,))
        y = pt.vector("y", shape=(10,))
        elemwise_out = x[idx] + y  # shape (10,)
        out = target[idx].inc(elemwise_out)
        fn, fn_u = fused_and_unfused([x, y, target], out)
        # Write not fused — the Elemwise loop dim is the non-indexed axis,
        # not the indexed axis. Read fusion may still create an
        # IndexedElemwise, but the AdvancedIncSubtensor1 must remain outside.
        assert any(
            isinstance(n.op, AdvancedIncSubtensor) for n in fn.maker.fgraph.toposort()
        )
        xv = rng.normal(size=(85,))
        yv = rng.normal(size=(10,))
        tv = rng.normal(size=(5, 10))
        np.testing.assert_allclose(fn(xv, yv, tv), fn_u(xv, yv, tv), rtol=1e-10)

    def test_no_fusion_when_val_broadcasts_along_index_dim(self):
        """Don't fuse if val is broadcastable on the index loop dim."""
        rng = np.random.default_rng(42)
        idx = rng.integers(5, size=10).astype(np.int64)
        target = pt.vector("target", shape=(5,))
        x = pt.tensor("x", shape=(1,))
        out = target[idx].inc(pt.exp(x))
        fn, fn_u = fused_and_unfused([x, target], out)
        assert any(
            isinstance(n.op, AdvancedIncSubtensor) for n in fn.maker.fgraph.toposort()
        )
        xv = rng.normal(size=(1,))
        tv = rng.normal(size=(5,))
        np.testing.assert_allclose(fn(xv, tv), fn_u(xv, tv), rtol=1e-10)

    @pytest.mark.parametrize(
        "target_trailing, should_fuse",
        [(3, False), (1, True)],
        ids=["target_not_bcast", "target_bcast"],
    )
    def test_val_broadcasts_along_trailing_dim(self, target_trailing, should_fuse):
        """Reject when val broadcasts in a trailing dim where target doesn't.

        target(5, T)[idx] += exp(col_vec) with col_vec shape (10, 1).
        When T>1, the codegen write index uses constant 0 for broadcastable
        dims, so only target[idx[i], 0] would be written.
        When T=1, both broadcast so a single write is correct.
        """
        rng = np.random.default_rng(42)
        idx = rng.integers(5, size=10).astype(np.int64)
        target = pt.matrix("target", shape=(5, target_trailing))
        x = pt.tensor("x", shape=(10, 1))
        out = target[idx].inc(pt.exp(x))
        fn, fn_u = fused_and_unfused([x, target], out)
        if should_fuse:
            assert_fused(fn)
        else:
            assert any(
                isinstance(n.op, AdvancedIncSubtensor)
                for n in fn.maker.fgraph.toposort()
            )
        xv = rng.normal(size=(10, 1))
        tv = rng.normal(size=(5, target_trailing))
        np.testing.assert_allclose(fn(xv, tv), fn_u(xv, tv), rtol=1e-10)

    @pytest.mark.parametrize(
        "target_shape, val_shape",
        [
            ((5, 3), (5, 10)),
            ((5, 3), (10,)),
            ((2, 5, 3), (5, 10)),
        ],
        ids=["no_excess", "full_excess", "partial_excess"],
    )
    def test_write_with_non_indexed_leading_dims(self, target_shape, val_shape):
        """Fuse writes when indexed axis has non-indexed dims to its left.

        The Elemwise output may cover all non-indexed leading dims (no_excess),
        none of them (full_excess), or only some (partial_excess).
        Excess dims are moved right via transpose_non_indexed_write_axes.
        """
        rng = np.random.default_rng(42)
        idx = rng.integers(target_shape[-1], size=10).astype(np.int64)
        n_slices = len(target_shape) - 1
        target = pt.tensor("target", shape=target_shape)
        val = pt.tensor("val", shape=val_shape)
        slices = (slice(None),) * n_slices
        out = target[(*slices, idx)].inc(pt.exp(val))

        fn = function([val, target], out, mode=NUMBA_MODE, trust_input=True)
        fn_u = function([val, target], out, mode=NUMBA_NO_FUSION, trust_input=True)
        assert_fused(fn)
        valv = rng.normal(size=val_shape)
        tv = rng.normal(size=target_shape)
        np.testing.assert_allclose(fn(valv, tv), fn_u(valv, tv), rtol=1e-10)

    def test_inc_subtensor(self):
        rng = np.random.default_rng(42)
        idx = rng.integers(85, size=919).astype(np.int64)
        x = pt.vector("x", shape=(85,))
        y = pt.vector("y", shape=(919,))
        t = pt.vector("t", shape=(85,))
        out = t[idx].inc(x[idx] + y)
        fn, fn_u = fused_and_unfused([x, y, t], out)
        assert_fused(fn)
        xv, yv, tv = (
            rng.normal(size=(85,)),
            rng.normal(size=(919,)),
            rng.normal(size=(85,)),
        )
        np.testing.assert_allclose(
            fn(xv, yv, tv.copy()), fn_u(xv, yv, tv.copy()), rtol=1e-10
        )

    def test_write_of_inplace_elemwise(self):
        """Inplace must not survive on a write-target output.

        The loop writes the elementwise result to the write buffer, never to the
        inplaced input, so a surviving inner inplace entry would destroy the dot
        intermediate undeclared in the Python-mode fallback.
        """
        rng = np.random.default_rng(9)
        idx = rng.integers(8, size=30).astype(np.int64)
        x = pt.matrix("x", shape=(30, 4))
        y = pt.matrix("y", shape=(4, 4))
        z = pt.matrix("z", shape=(30, 4))
        t = pt.matrix("t", shape=(8, 4))
        out = t[idx].inc((x @ y) * z)
        fn, fn_u = fused_and_unfused([x, y, z, t], out)
        assert_fused(fn)
        [node] = [
            n for n in fn.maker.fgraph.toposort() if isinstance(n.op, IndexedElemwise)
        ]
        assert not any(
            n.op.inplace_pattern
            for n in node.op.fgraph.toposort()
            if isinstance(n.op, Elemwise)
        )
        xv, yv, zv, tv = (
            rng.normal(size=(30, 4)),
            rng.normal(size=(4, 4)),
            rng.normal(size=(30, 4)),
            rng.normal(size=(8, 4)),
        )
        np.testing.assert_allclose(
            fn(xv, yv, zv, tv.copy()), fn_u(xv, yv, zv, tv.copy()), rtol=1e-10
        )

    def test_write_with_direct_use_keeps_inplace(self):
        """Inplace survives on an output that is both written and used directly.

        The write-and-direct duplication keeps the original output materialized
        (the write consumes a duplicate), so the inplace claimed on the dot
        intermediate stays valid and must not be stripped.
        """
        rng = np.random.default_rng(10)
        idx = rng.integers(8, size=30).astype(np.int64)
        x = pt.matrix("x", shape=(30, 4))
        y = pt.matrix("y", shape=(4, 4))
        z = pt.matrix("z", shape=(30, 4))
        t = pt.matrix("t", shape=(8, 4))
        w = (x @ y) * z
        fn, fn_u = fused_and_unfused([x, y, z, t], [t[idx].inc(w), w])
        assert_fused(fn)
        [node] = [
            n for n in fn.maker.fgraph.toposort() if isinstance(n.op, IndexedElemwise)
        ]
        # Two destroy entries: the write buffer, and the dot intermediate kept
        # inplace by the materialized output
        assert len(node.op.destroy_map) == 2
        xv, yv, zv, tv = (
            rng.normal(size=(30, 4)),
            rng.normal(size=(4, 4)),
            rng.normal(size=(30, 4)),
            rng.normal(size=(8, 4)),
        )
        for res, res_u in zip(fn(xv, yv, zv, tv.copy()), fn_u(xv, yv, zv, tv.copy())):
            np.testing.assert_allclose(res, res_u, rtol=1e-10)

    def test_set_subtensor(self):
        rng = np.random.default_rng(42)
        idx = rng.integers(85, size=919).astype(np.int64)
        x = pt.vector("x", shape=(85,))
        y = pt.vector("y", shape=(919,))
        t = pt.vector("t", shape=(85,))
        out = t[idx].set(x[idx] + y)
        fn, fn_u = fused_and_unfused([x, y, t], out)
        assert_fused(fn)
        xv, yv, tv = (
            rng.normal(size=(85,)),
            rng.normal(size=(919,)),
            rng.normal(size=(85,)),
        )
        np.testing.assert_allclose(
            fn(xv, yv, tv.copy()), fn_u(xv, yv, tv.copy()), rtol=1e-10
        )

    def test_set_subtensor_same_index_both_axes(self):
        """target[idx, idx].set(vals) — same 1-D index on both axes (diagonal write)."""
        rng = np.random.default_rng(42)
        x = pt.vector("x", shape=(5,))
        target = pt.matrix("target", shape=(10, 10))
        idx = pt.vector("idx", dtype="int64", shape=(5,))
        out = target[idx, idx].set(pt.sqrt(x))
        fn, fn_u = fused_and_unfused([x, target, idx], out)
        assert_fused(fn)
        xv = np.abs(rng.normal(size=5))
        tv = rng.normal(size=(10, 10))
        iv = np.arange(5, dtype=np.int64)
        np.testing.assert_allclose(
            fn(xv, tv.copy(), iv), fn_u(xv, tv.copy(), iv), rtol=1e-10
        )

    def test_target_not_modified_when_non_inplace(self):
        """Non-inplace indexed write should not modify the original target."""
        rng = np.random.default_rng(42)
        idx = rng.integers(85, size=919).astype(np.int64)
        x = pt.vector("x", shape=(85,))
        y = pt.vector("y", shape=(919,))
        t = pt.vector("t", shape=(85,))
        out = t[idx].inc(x[idx] + y)
        fn = function([x, y, t], out, mode=NUMBA_MODE, trust_input=True)
        xv, yv = rng.normal(size=(85,)), rng.normal(size=(919,))
        tv = rng.normal(size=(85,))
        tv_copy = tv.copy()
        fn(xv, yv, tv)
        np.testing.assert_array_equal(tv, tv_copy)

    def test_multiple_write_targets_different_lengths(self):
        """Sibling writes into different-length targets keyed by different indices.

        Write-target buffers are matched to outputs by ascending output index; when
        the index-group order disagrees with that (here the length-3 target is
        written before the length-2 one), each output must still allocate its own
        target buffer instead of a sibling's.
        """
        idx_s = np.array([0, 1, 0], dtype=np.int64)
        idx_b = np.array([2, 0, 1], dtype=np.int64)
        ts = pt.vector("ts", shape=(2,))
        tb = pt.vector("tb", shape=(3,))

        out_b = tb[idx_b].inc(pt.exp(ts[idx_s] + tb[idx_b]))
        out_s = ts[idx_s].inc(pt.log1p(ts[idx_s] * tb[idx_b]))
        fn, fn_u = fused_and_unfused([ts, tb], [out_b, out_s])
        assert_fused(fn)

        rng = np.random.default_rng(42)
        tsv, tbv = rng.normal(size=2), rng.normal(size=3)
        for fused_o, unfused_o in zip(fn(tsv, tbv), fn_u(tsv, tbv)):
            np.testing.assert_allclose(fused_o, unfused_o, rtol=1e-10)

    @pytest.mark.parametrize(
        "read_idx, write_idx",
        # Non-contiguous so the indices stay Advanced(Inc)Subtensor1 rather than
        # being canonicalised into basic slices.
        [
            ([0, 2, 5], [1, 3, 7]),
            ([0, 2, 5], [0, 2, 5]),
            ([0, 2, 5], [5, 0, 2]),
        ],
        ids=["write_out_of_read_range", "write_equals_read", "write_permutes_read"],
    )
    def test_write_target_aliases_read_source(self, read_idx, write_idx):
        """Indexed write into a buffer that is also read through the same Elemwise.

        ``set_subtensor(b[write_idx], b[read_idx] * 2)`` reads and writes the same
        buffer ``b``. Fusing the in-place write would alias the destroyed write
        target with the live read input, so the write must stay external while the
        read still fuses -- without raising an aliasing error or aborting the pass.

        The aliasing is only genuinely unsafe when read and write indices overlap
        in a *different order* (``write_permutes_read``): then an in-loop write
        could clobber a position another iteration still has to read. When the
        indices don't overlap (``write_out_of_read_range``) or overlap in the same
        order (``write_equals_read``) the alias is harmless and could be fused in
        the future via a ``tolerated_aliased`` flag. For now we conservatively skip
        the write in all cases; this test pins the correctness of that behaviour.
        """
        rng = np.random.default_rng(42)
        x = pt.vector("x", shape=(9,))
        b = x + 1.0
        read_idx = np.array(read_idx, dtype=np.int64)
        write_idx = np.array(write_idx, dtype=np.int64)
        out = b[write_idx].set(b[read_idx] * 2.0)
        fn, fn_u = fused_and_unfused([x], out)
        # The read fuses into an IndexedElemwise; the aliasing write stays external.
        assert_fused(fn)
        assert any(
            isinstance(n.op, AdvancedIncSubtensor) for n in fn.maker.fgraph.toposort()
        )
        xv = rng.normal(size=9)
        np.testing.assert_allclose(fn(xv), fn_u(xv), rtol=1e-10)

    def test_non_inplace_aliasing_write_preserves_input(self):
        """A non-inplace write whose target is also a read source fuses with a copy.

        Here ``t`` is read and written, and the write is non-inplace (``t`` is a
        protected input), so it fuses by binding a *copy* of ``t`` to the write
        slot. That slot must be a distinct inner input from the read source;
        otherwise construct_nominal_fgraph dedupes them, the copy goes dead inside,
        and the inner write destroys the read buffer -- silently mutating ``t``.

        Checked through ``OpFromGraph.perform``, which runs the inner graph
        literally; the numba backend uses the indexed-output spec and would not
        expose the corruption.
        """
        t = pt.vector("t", shape=(None,))
        idx = np.array([0, 2, 5], dtype=np.int64)
        out = t[idx].set(t[idx] * 2.0)
        fn = function([t], out, mode=NUMBA_MODE, trust_input=True)

        # The non-inplace write fuses fully -- no external AdvancedIncSubtensor1.
        assert_fused(fn)
        assert not any(
            isinstance(n.op, AdvancedIncSubtensor) for n in fn.maker.fgraph.toposort()
        )

        # The inner write must destroy exactly the input the op's destroy_map names.
        [node] = [
            n for n in fn.maker.fgraph.toposort() if isinstance(n.op, IndexedElemwise)
        ]
        [(_out_idx, [destroyed_pos])] = node.op.destroy_map.items()
        [inner_write] = [
            n for n in node.op.fgraph.apply_nodes if getattr(n.op, "destroy_map", None)
        ]
        assert inner_write.inputs[0] is node.op.fgraph.inputs[destroyed_pos]

        # Run the inner graph via perform; the input buffer must be left untouched.
        perform_outs = node.op(*node.inputs, return_list=True)
        f_perform = function(
            [t],
            perform_outs,
            mode=Mode(linker="py", optimizer=None),
            accept_inplace=True,
            trust_input=True,
        )
        tv = np.arange(1, 10, dtype="float64")
        expected = tv.copy()
        expected[idx] = tv[idx] * 2.0
        tv_in = tv.copy()
        np.testing.assert_allclose(f_perform(tv_in)[0], expected, rtol=1e-10)
        np.testing.assert_array_equal(tv_in, tv)


class TestRepeatedAccumulationIndices:
    """Test inc_subtensor with repeated indices (same position accumulated multiple times)."""

    def test_repeated_inc_subtensor(self):
        """inc_subtensor with repeated indices should accumulate correctly."""
        rng = np.random.default_rng(42)
        target = pt.vector("target", shape=(10,))
        x = pt.vector("x", shape=(20,))
        idx = np.array(
            [0, 1, 2, 0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
            dtype=np.int64,
        )
        out = target[idx].inc(pt.exp(x))
        fn, fn_u = fused_and_unfused([target, x], out)
        assert_fused(fn)
        tv = rng.normal(size=10)
        xv = rng.normal(size=20)
        np.testing.assert_allclose(fn(tv.copy(), xv), fn_u(tv.copy(), xv), rtol=1e-10)

    def test_repeated_set_subtensor(self):
        """set_subtensor with repeated indices -- last write wins."""
        rng = np.random.default_rng(42)
        target = pt.vector("target", shape=(10,))
        x = pt.vector("x", shape=(5,))
        idx = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        out = target[idx].set(pt.exp(x))
        fn, fn_u = fused_and_unfused([target, x], out)
        assert_fused(fn)
        tv = rng.normal(size=10)
        xv = rng.normal(size=5)
        np.testing.assert_allclose(fn(tv.copy(), xv), fn_u(tv.copy(), xv), rtol=1e-10)

    def test_repeated_inc_with_read(self):
        """Combined read + inc_subtensor with repeated indices."""
        rng = np.random.default_rng(42)
        source = pt.vector("source", shape=(10,))
        target = pt.vector("target", shape=(10,))
        idx = np.array([0, 1, 0, 1, 2, 2, 3, 3], dtype=np.int64)
        out = target[idx].inc(source[idx] * 2.0)
        fn, fn_u = fused_and_unfused([source, target], out)
        assert_fused(fn)
        sv = rng.normal(size=10)
        tv = rng.normal(size=10)
        np.testing.assert_allclose(fn(sv, tv.copy()), fn_u(sv, tv.copy()), rtol=1e-10)


class TestShapeValidation:
    """Test that mismatched index/input shapes raise runtime errors."""

    def test_mismatched_index_and_direct_input(self):
        """Index length doesn't match direct input on the same loop dim."""
        x = pt.vector("x", shape=(None,))
        y = pt.vector("y", shape=(None,))
        idx = pt.vector("idx", dtype="int64", shape=(None,))
        out = x[idx] + y
        fn = function([x, idx, y], out, mode=NUMBA_MODE, trust_input=True)
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
        fn = function([x, idx, y], out, mode=NUMBA_MODE, trust_input=True)
        assert_fused(fn)
        # Both idx and y have length 1 — should work (both agree on dim 0)
        result = fn(np.zeros(100), np.zeros(1, dtype=np.int64), np.zeros(1))
        assert result.shape == (1,)
        # idx=1, y=5 — should error (shape mismatch, no static broadcast info)
        with pytest.raises(Exception):
            fn(np.zeros(100), np.zeros(1, dtype=np.int64), np.zeros(5))

    def test_no_fusion_with_bounded_basic_slice_read(self):
        """Regression: a bounded basic slice on a non-indexed axis can't be
        carried by the fused loop, which substitutes the full source array and
        iterates non-indexed axes wholesale. Fusing ``x[1:4, idx]`` dropped the
        ``1:4`` offset and iterated x's full axis 0 against y's, raising at
        runtime. The slice must block fusion and results stay correct."""
        rng = np.random.default_rng(2202)
        x = pt.matrix("x", shape=(6, 6))
        y = pt.matrix("y", shape=(3, 3))
        idx = pt.constant(np.array([0, 2, 4]))

        out = x[1:4, idx] + y
        fn = function([x, y], out, mode=NUMBA_MODE, trust_input=True)
        assert not any(
            isinstance(n.op, IndexedElemwise) for n in fn.maker.fgraph.toposort()
        )

        ref = function([x, y], out, mode=NUMBA_NO_FUSION, trust_input=True)
        xv, yv = rng.normal(size=(6, 6)), rng.normal(size=(3, 3))
        np.testing.assert_allclose(fn(xv, yv), ref(xv, yv), rtol=1e-10)

    def test_no_fusion_with_bounded_basic_slice_write(self):
        """As above for an indexed write: a bounded basic slice on the write
        target's non-indexed axis must block fusion."""
        rng = np.random.default_rng(2203)
        t = pt.matrix("t", shape=(6, 6))
        y = pt.matrix("y", shape=(3, 3))
        idx = pt.constant(np.array([0, 2, 4]))

        out = t[1:4, idx].set(pt.exp(y))
        fn = function([t, y], out, mode=NUMBA_MODE, trust_input=True)
        assert not any(
            isinstance(n.op, IndexedElemwise) for n in fn.maker.fgraph.toposort()
        )

        ref = function([t, y], out, mode=NUMBA_NO_FUSION, trust_input=True)
        tv, yv = rng.normal(size=(6, 6)), rng.normal(size=(3, 3))
        np.testing.assert_allclose(fn(tv.copy(), yv), ref(tv.copy(), yv), rtol=1e-10)

    def test_loop_shape_regression(self):
        """
        Regression test for https://github.com/pymc-devs/pytensor/issues/2201

        Stale loop shape branch would only look at the first dimension of indices.
        In this test example it would think we are iterating over a [24, 24] loop instead of a [24, 3]
        """
        rng = np.random.default_rng(2201)

        resp_idx = rng.integers(0, 6, size=24).astype("int32")
        item_idx = rng.integers(0, 5, size=(24, 3)).astype("int32")
        mask = pt.dmatrix("mask", shape=(24, 3))
        beta = pt.dmatrix("beta", shape=(6, 5))

        u = beta[resp_idx[:, None], item_idx]
        u_masked = pt.where(mask, u, -1e10)
        out = u_masked.sum()

        f = function([beta, mask], out, mode=NUMBA_MODE)
        assert_fused(f)

        ref_f = function([beta, mask], out, mode=Mode(linker="py", optimizer=None))
        test_beta = np.zeros((6, 5))
        test_mask = np.ones((24, 3), dtype="bool")
        res = f(beta=test_beta, mask=test_mask)
        ref_res = ref_f(beta=test_beta, mask=test_mask)
        np.testing.assert_allclose(res, ref_res, strict=True)
