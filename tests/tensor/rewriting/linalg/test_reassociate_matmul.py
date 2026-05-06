import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.blas import BatchedDot, Dot22, Gemm
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import Dot
from pytensor.tensor.rewriting.linalg.reassociate_matmul import (
    CostExpr,
    _provably_less,
    reassociate_matmul_chain,
)


requires_fast_run = pytest.mark.skipif(
    config.mode == "FAST_COMPILE",
    reason="reassociate_matmul_chain is only registered for FAST_RUN",
)


def _matmul_nodes(fgraph):
    """Apply nodes that perform a matmul-like contraction."""
    nodes = []
    for n in fgraph.toposort():
        op = n.op
        if isinstance(op, Dot | Dot22 | Gemm | BatchedDot):
            nodes.append(n)
        elif isinstance(op, Blockwise) and isinstance(op.core_op, Dot):
            nodes.append(n)
    return nodes


def _matmul_output_shapes(fgraph):
    """Static shapes of every matmul output (last entry is the final output)."""
    return [n.outputs[0].type.shape for n in _matmul_nodes(fgraph)]


def _dimshuffle_count(fgraph):
    return sum(1 for n in fgraph.toposort() if isinstance(n.op, DimShuffle))


class TestProvablyLess:
    def test_int_dominance(self):
        a = CostExpr.from_dim_product([10, 20, 5])  # 1000
        b = CostExpr.from_dim_product([10, 20, 5]) + CostExpr.from_dim_product(
            [1, 2, 3]
        )  # 1006
        assert _provably_less(a, b)
        assert not _provably_less(b, a)
        assert not _provably_less(a, a)

    def test_zero_vs_positive(self):
        a = CostExpr.zero()
        b = CostExpr.from_dim_product([7])
        assert _provably_less(a, b)
        assert not _provably_less(b, a)
        assert not _provably_less(a, a)

    def test_symbolic_no_proof(self):
        # m*k*n vs k*n*p: different monomials, no dominance either way.
        m = pt.scalar("m")
        k = pt.scalar("k")
        n = pt.scalar("n")
        p = pt.scalar("p")
        a = CostExpr.from_dim_product([m, k, n])
        b = CostExpr.from_dim_product([k, n, p])
        assert not _provably_less(a, b)
        assert not _provably_less(b, a)

    def test_extra_unmatched_b_monomial(self):
        # a = m*k, b = m*k + p: b strictly larger under p >= 1.
        m = pt.scalar("m")
        k = pt.scalar("k")
        p = pt.scalar("p")
        a = CostExpr.from_dim_product([m, k])
        b = CostExpr.from_dim_product([m, k]) + CostExpr.from_dim_product([p])
        assert _provably_less(a, b)
        assert not _provably_less(b, a)

    def test_b_dominates_with_extra_symbol(self):
        # a = m*k, b = m*k*p: b is sound but not strict (p == 1 makes them equal).
        m = pt.scalar("m")
        k = pt.scalar("k")
        p = pt.scalar("p")
        a = CostExpr.from_dim_product([m, k])
        b = CostExpr.from_dim_product([m, k, p])
        assert not _provably_less(a, b)

    def test_greedy_fails_kuhn_succeeds(self):
        # Greedy first-fit pairs a[0]=x with b's first dominator (x*y), leaving
        # a[1]=x*y with only x*z, which doesn't dominate. The valid matching is
        # x <-> x*z and x*y <-> x*y; the unmatched `w` makes the strict comparison
        # succeed.
        x, y, z, w = pt.scalar("x"), pt.scalar("y"), pt.scalar("z"), pt.scalar("w")
        a = CostExpr.from_dim_product([x]) + CostExpr.from_dim_product([x, y])
        b = (
            CostExpr.from_dim_product([x, y])
            + CostExpr.from_dim_product([x, z])
            + CostExpr.from_dim_product([w])
        )
        assert _provably_less(a, b)


class TestReassociateChain:
    @requires_fast_run
    def test_optimal_static_ordering(self):
        # (100x2) @ (2x100) @ (100x100): optimal is A @ (B @ C) with B@C producing
        # a (2, 100) intermediate, far cheaper than (A @ B) @ C.
        A = pt.matrix("A", shape=(100, 2))
        B = pt.matrix("B", shape=(2, 100))
        C = pt.matrix("C", shape=(100, 100))

        f = function([A, B, C], A @ B @ C)
        shapes = _matmul_output_shapes(f.maker.fgraph)
        assert (2, 100) in shapes, (
            f"Expected the optimal split to produce a (2, 100) intermediate, "
            f"got {shapes}"
        )

    def test_value_equivalence_4_matrices(self):
        rng = np.random.default_rng(0)
        shapes = [(10, 20), (20, 5), (5, 30), (30, 3)]
        np_arrays = [rng.normal(size=s).astype(config.floatX) for s in shapes]
        pt_inputs = [pt.matrix(f"M{i}", shape=s) for i, s in enumerate(shapes)]
        out = pt_inputs[0] @ pt_inputs[1] @ pt_inputs[2] @ pt_inputs[3]
        f = function(pt_inputs, out)
        np.testing.assert_allclose(
            f(*np_arrays), np.linalg.multi_dot(np_arrays), rtol=1e-5
        )
        if config.mode != "FAST_COMPILE":
            # Optimal split for these shapes is A @ (B @ (C @ D)) with cost 1350
            # vs. naive 3400. Pin the smallest internal intermediate as a
            # sentinel that the rewriter chose this tree.
            shapes_seen = _matmul_output_shapes(f.maker.fgraph)
            assert (5, 3) in shapes_seen, shapes_seen

    def test_symbolic_shapes_preserve_user_order(self):
        # When all dims are symbolic, no parenthesization is provably cheaper, so
        # the rewriter must leave the user's order intact.
        A = pt.matrix("A")
        B = pt.matrix("B")
        C = pt.matrix("C")
        out = A @ (B @ C)
        f = function([A, B, C], out)

        nodes = _matmul_nodes(f.maker.fgraph)
        assert len(nodes) == 2
        # The first matmul (in topo order) should be the BC contraction -- its
        # inputs are exactly B and C, not the result of A @ anything.
        first = nodes[0]
        assert set(first.inputs) == {B, C}, (
            f"Expected first matmul to be B @ C, got inputs {first.inputs}"
        )

    def test_already_optimal_no_rewrite(self):
        # User-written order is already optimal: (A @ B) -> (2, 100) @ C -> (2, 5)
        # at cost 20_000 vs. the alternative A @ (B @ C) at 50_000. The rewriter
        # must not disturb this.
        A = pt.matrix("A", shape=(2, 100))
        B = pt.matrix("B", shape=(100, 100))
        C = pt.matrix("C", shape=(100, 5))
        f = function([A, B, C], (A @ B) @ C)
        shapes = _matmul_output_shapes(f.maker.fgraph)
        assert shapes == [(2, 100), (2, 5)], (
            f"Expected the user's two-step order to be preserved exactly; got {shapes}"
        )

    def test_multi_client_breakage(self):
        # If an intermediate matmul output has more than one client, the chain
        # extender must stop there -- absorbing past it would duplicate the
        # intermediate's computation when emitting the new tree.
        A = pt.matrix("A", shape=(100, 2))
        B = pt.matrix("B", shape=(2, 100))
        C = pt.matrix("C", shape=(100, 100))
        D = pt.matrix("D", shape=(100, 5))
        ab = A @ B  # (100, 100), used twice
        out1 = ab @ C @ D
        out2 = ab.sum()
        f = function([A, B, C, D], [out1, out2])
        rng = np.random.default_rng(1)
        a_v = rng.normal(size=(100, 2)).astype(config.floatX)
        b_v = rng.normal(size=(2, 100)).astype(config.floatX)
        c_v = rng.normal(size=(100, 100)).astype(config.floatX)
        d_v = rng.normal(size=(100, 5)).astype(config.floatX)
        r1, r2 = f(a_v, b_v, c_v, d_v)
        ab_v = a_v @ b_v
        np.testing.assert_allclose(r1, ab_v @ c_v @ d_v, rtol=1e-5)
        np.testing.assert_allclose(r2, ab_v.sum(), rtol=1e-5)

    def test_quadratic_form_static(self):
        # A @ B @ A.T with A: (50, 100), B: (100, 100). The transpose makes the
        # third operand opaquely (100, 50). Both parenthesizations cost 750_000:
        #   (A @ B) @ A.T: 50*100*100 + 50*100*50
        #   A @ (B @ A.T): 100*100*50 + 50*100*50
        # The rewriter has no provable win and must leave the user's order alone.
        A = pt.matrix("A", shape=(50, 100))
        B = pt.matrix("B", shape=(100, 100))
        f = function([A, B], A @ B @ A.T)
        shapes = _matmul_output_shapes(f.maker.fgraph)
        assert shapes == [(50, 100), (50, 50)], (
            f"Expected user's symmetric chain to be preserved; got {shapes}"
        )

    def test_batched_blockwise_dot_reorders(self):
        # Batched chain shaped to force a reorder: A(7,100,2) B(7,2,100) C(7,100,5).
        # Naive (A@B)@C: 7*100*2*100 + 7*100*100*5 = 490k.
        # Optimal A@(B@C): 7*2*100*5 + 7*100*2*5    = 14k.
        rng = np.random.default_rng(2)
        shapes = [(7, 100, 2), (7, 2, 100), (7, 100, 5)]
        np_arrays = [rng.normal(size=s).astype(config.floatX) for s in shapes]
        pt_inputs = [pt.tensor3(f"B{i}", shape=s) for i, s in enumerate(shapes)]
        out = pt_inputs[0] @ pt_inputs[1] @ pt_inputs[2]
        f = function(pt_inputs, out)
        np.testing.assert_allclose(
            f(*np_arrays),
            np_arrays[0] @ np_arrays[1] @ np_arrays[2],
            rtol=1e-5,
        )
        if config.mode != "FAST_COMPILE":
            shapes_seen = _matmul_output_shapes(f.maker.fgraph)
            assert (7, 2, 5) in shapes_seen, shapes_seen

    @requires_fast_run
    def test_dot22_chain_post_blas(self):
        # By the time the rewriter runs (post-BLAS), 2-D dots may have been
        # promoted to Dot22. The chain extender must treat Dot22 as a chain link
        # so the chain spans the full sequence AND so the rebuilt tree retains the
        # BLAS-promoted form.
        A = pt.matrix("A", shape=(100, 2))
        B = pt.matrix("B", shape=(2, 100))
        C = pt.matrix("C", shape=(100, 100))
        f = function([A, B, C], A @ B @ C)
        nodes = _matmul_nodes(f.maker.fgraph)
        ops_seen = {type(n.op).__name__ for n in nodes}
        assert ops_seen & {"Dot22", "Gemm"}, (
            f"Expected BLAS-promoted nodes to survive reassociation; got {ops_seen}"
        )
        shapes = _matmul_output_shapes(f.maker.fgraph)
        assert (2, 100) in shapes, shapes

    @pytest.mark.parametrize(
        "chain_shapes,must_contain",
        [
            # Classic textbook 3-matrix: A(10x30) B(30x5) C(5x60). Optimal is
            # (A@B)@C with intermediate (10, 5).
            ([(10, 30), (30, 5), (5, 60)], {(10, 5)}),
            # 5-matrix CLRS-style chain. Optimal is (A(BC))(DE) -- intermediates
            # are (35, 5), (30, 5), (5, 20), (30, 20). Two of those are the
            # required-present sentinels.
            ([(30, 35), (35, 15), (15, 5), (5, 10), (10, 20)], {(35, 5), (5, 20)}),
        ],
    )
    def test_textbook_examples(self, chain_shapes, must_contain):
        rng = np.random.default_rng(3)
        np_arrays = [rng.normal(size=s).astype(config.floatX) for s in chain_shapes]
        pt_inputs = [pt.matrix(f"M{i}", shape=s) for i, s in enumerate(chain_shapes)]
        out = pt_inputs[0]
        for x in pt_inputs[1:]:
            out = out @ x
        f = function(pt_inputs, out)
        np.testing.assert_allclose(
            f(*np_arrays), np.linalg.multi_dot(np_arrays), rtol=1e-5
        )
        if config.mode != "FAST_COMPILE":
            shapes = set(_matmul_output_shapes(f.maker.fgraph))
            missing = must_contain - shapes
            assert not missing, (
                f"Missing expected intermediates {missing}; got {shapes}"
            )

    def test_balanced_tree_decomposition(self):
        # User's explicit `(A @ B) @ (C @ D)` is a non-linear tree. The recursive
        # decomposer must still see the full 4-operand chain through both subtrees.
        # Shapes are chosen so the user's order has three (100, 100) contractions
        # costing ~1M FLOPs and the optimal `A @ (B @ (C @ D))` avoids them
        # entirely (~21k FLOPs). The unavoidable (100, 100) is the final output;
        # any *internal* (100, 100) means the rewriter missed the cross-subtree
        # chain.
        A = pt.matrix("A", shape=(100, 2))
        B = pt.matrix("B", shape=(2, 100))
        C = pt.matrix("C", shape=(100, 3))
        D = pt.matrix("D", shape=(3, 100))
        f = function([A, B, C, D], (A @ B) @ (C @ D))

        rng = np.random.default_rng(7)
        a_v = rng.normal(size=(100, 2)).astype(config.floatX)
        b_v = rng.normal(size=(2, 100)).astype(config.floatX)
        c_v = rng.normal(size=(100, 3)).astype(config.floatX)
        d_v = rng.normal(size=(3, 100)).astype(config.floatX)
        np.testing.assert_allclose(
            f(a_v, b_v, c_v, d_v), (a_v @ b_v) @ (c_v @ d_v), rtol=1e-5
        )

        if config.mode != "FAST_COMPILE":
            shapes = _matmul_output_shapes(f.maker.fgraph)
            n_big = sum(1 for s in shapes if s == (100, 100))
            assert n_big <= 1, (
                f"Expected at most one (100,100) (the final output); got {shapes}"
            )


class TestLifting:
    def test_lift_expand_dims_recovers_chain(self):
        # `expand_dims(A @ B, 0) @ C @ D` only sees three operands without lifting,
        # and the (A @ B) intermediate is forced to (100, 100). After lifting the
        # expand_dims past A @ B, the chain becomes 4-element and the DP can avoid
        # the large intermediate by routing through the small m=2/k=3 dims.
        A = pt.matrix("A", shape=(100, 2))
        B = pt.matrix("B", shape=(2, 100))
        C = pt.tensor3("C", shape=(1, 100, 3))
        D = pt.tensor3("D", shape=(1, 3, 100))
        ab3 = pt.expand_dims(A @ B, 0)
        f = function([A, B, C, D], ab3 @ C @ D)

        rng = np.random.default_rng(11)
        a_v = rng.normal(size=(100, 2)).astype(config.floatX)
        b_v = rng.normal(size=(2, 100)).astype(config.floatX)
        c_v = rng.normal(size=(1, 100, 3)).astype(config.floatX)
        d_v = rng.normal(size=(1, 3, 100)).astype(config.floatX)
        np.testing.assert_allclose(
            f(a_v, b_v, c_v, d_v), (a_v @ b_v)[None] @ c_v @ d_v, rtol=1e-5
        )

        if config.mode != "FAST_COMPILE":
            # The unavoidable (100, 100) (or (1, 100, 100)) is the final output.
            # Any internal contraction with that shape means the lift didn't fire.
            shapes = _matmul_output_shapes(f.maker.fgraph)
            non_final = shapes[:-1] if shapes else []
            assert (1, 100, 100) not in non_final and (100, 100) not in non_final, (
                f"Lift+reorder should avoid the (100,100) intermediate; got {shapes}"
            )

    def test_lift_matrix_transpose(self):
        # `(L @ R).T @ C` lifts to `R.T @ L.T @ C` (operand order swapped). Shapes
        # are chosen so the lifted form's `R.T @ (L.T @ C)` is dramatically cheaper
        # than the user's `(L @ R).T @ C` -- verifies that matrix-transpose lift
        # exposes a real reorder, not just a no-op restructuring.
        # L (100, 2), R (2, 50), C (100, 100):
        #   user:   100*2*50 + 50*100*100        = 510_000
        #   lifted [R.T (50,2), L.T (2,100), C (100,100)] with R.T @ (L.T @ C):
        #     L.T @ C = 2*100*100 = 20_000; R.T @ that = 50*2*100 = 10_000.
        L = pt.matrix("L", shape=(100, 2))
        R = pt.matrix("R", shape=(2, 50))
        C = pt.matrix("C", shape=(100, 100))
        f = function([L, R, C], (L @ R).T @ C)

        rng = np.random.default_rng(13)
        l_v = rng.normal(size=(100, 2)).astype(config.floatX)
        r_v = rng.normal(size=(2, 50)).astype(config.floatX)
        c_v = rng.normal(size=(100, 100)).astype(config.floatX)
        np.testing.assert_allclose(f(l_v, r_v, c_v), (l_v @ r_v).T @ c_v, rtol=1e-5)

        if config.mode != "FAST_COMPILE":
            # Optimal lifted tree's intermediate is L.T @ C = (2, 100).
            shapes = _matmul_output_shapes(f.maker.fgraph)
            assert (2, 100) in shapes, (
                f"Expected matrix-transpose lift to enable a (2, 100) "
                f"intermediate; got {shapes}"
            )

    def test_lift_atomic_gating_no_extra_nodes(self):
        # All-symbolic shapes -> no provable win -> no rewrite. The lift must NOT
        # add extra DimShuffle nodes when the rewrite isn't committed. Test the
        # rewriter standalone on a fresh fgraph (no other passes) so the assertion
        # isolates its behavior.
        L = pt.matrix("L")
        R = pt.matrix("R")
        C = pt.matrix("C")
        D = pt.matrix("D")
        out = pt.expand_dims(L @ R, 0) @ pt.expand_dims(C, 0) @ pt.expand_dims(D, 0)

        fgraph = FunctionGraph([L, R, C, D], [out], clone=False)
        before = _dimshuffle_count(fgraph)
        reassociate_matmul_chain.apply(fgraph)
        after = _dimshuffle_count(fgraph)
        assert after == before, (
            f"Speculative lift leaked DimShuffle nodes: before={before} after={after}"
        )

    def test_lift_squeeze(self):
        # `squeeze(A @ B, 0) @ C @ D` where (A@B) has a leading-1 batch dim. After
        # the squeeze lift the chain spans 4 operands and the DP can route through
        # the small k=2/k=3 dims rather than the (100, 100) intermediate the
        # original chain forced.
        A = pt.tensor3("A", shape=(1, 100, 2))
        B = pt.tensor3("B", shape=(1, 2, 100))
        C = pt.matrix("C", shape=(100, 3))
        D = pt.matrix("D", shape=(3, 100))
        ab2 = pt.squeeze(A @ B, axis=0)
        f = function([A, B, C, D], ab2 @ C @ D)

        rng = np.random.default_rng(23)
        a_v = rng.normal(size=(1, 100, 2)).astype(config.floatX)
        b_v = rng.normal(size=(1, 2, 100)).astype(config.floatX)
        c_v = rng.normal(size=(100, 3)).astype(config.floatX)
        d_v = rng.normal(size=(3, 100)).astype(config.floatX)
        np.testing.assert_allclose(
            f(a_v, b_v, c_v, d_v),
            (a_v @ b_v).squeeze(axis=0) @ c_v @ d_v,
            rtol=1e-5,
        )
        if config.mode != "FAST_COMPILE":
            # Final output is (100, 100); any internal one means the lift didn't fire.
            shapes = _matmul_output_shapes(f.maker.fgraph)
            non_final = shapes[:-1] if shapes else []
            assert (100, 100) not in non_final, (
                f"Squeeze lift should expose a chain that avoids the (100,100) "
                f"intermediate; got {shapes}"
            )

    def test_lift_through_heterogeneous_ndim_does_not_crash(self):
        # A DimShuffle wrapping a Blockwise(Dot) whose operands have different
        # ndims (one 2-D, one 3-D) cannot have its lift propagated to both inner
        # operands -- the lift's `new_order` references indices the 2-D operand
        # doesn't have. The rewriter must bail and treat the wrapper as opaque
        # rather than crashing or producing a malformed graph.
        L = pt.matrix("L", shape=(4, 5))
        R = pt.tensor3("R", shape=(7, 5, 6))  # broadcasts L to (7, 4, 6)
        C = pt.tensor3("C", shape=(7, 6, 8))
        ds_inner = pt.expand_dims(L @ R, 0)
        f = function([L, R, C], ds_inner @ pt.expand_dims(C, 0))

        rng = np.random.default_rng(29)
        l_v = rng.normal(size=(4, 5)).astype(config.floatX)
        r_v = rng.normal(size=(7, 5, 6)).astype(config.floatX)
        c_v = rng.normal(size=(7, 6, 8)).astype(config.floatX)
        np.testing.assert_allclose(
            f(l_v, r_v, c_v),
            (l_v @ r_v)[None] @ c_v[None],
            rtol=1e-5,
        )

    def test_no_batched_dot_when_batch_dims_could_broadcast(self):
        # BatchedDot does not broadcast -- its perform/C path errors at runtime
        # when batch dims differ. If the chain has one operand with static batch
        # dim 1 and another with non-1 batch dim, the rewriter must NOT emit
        # BatchedDot for those pairs; matmul/Blockwise(Dot) is the only safe
        # choice.
        A = pt.tensor3("A", shape=(1, 4, 5))
        B = pt.tensor3("B", shape=(7, 5, 6))
        C = pt.tensor3("C", shape=(7, 6, 4))
        f = function([A, B, C], A @ B @ C)

        # Any BatchedDot in the rewritten graph must have both operands' static
        # batch dim known and equal. Anything else is a latent runtime crash.
        for node in _matmul_nodes(f.maker.fgraph):
            if isinstance(node.op, BatchedDot):
                lb = node.inputs[0].type.shape[0]
                rb = node.inputs[1].type.shape[0]
                assert lb is not None and rb is not None and lb == rb, (
                    f"BatchedDot with broadcasting-incompatible batch dims: "
                    f"left={node.inputs[0].type.shape}, "
                    f"right={node.inputs[1].type.shape}"
                )

        rng = np.random.default_rng(31)
        a_v = rng.normal(size=(1, 4, 5)).astype(config.floatX)
        b_v = rng.normal(size=(7, 5, 6)).astype(config.floatX)
        c_v = rng.normal(size=(7, 6, 4)).astype(config.floatX)
        np.testing.assert_allclose(f(a_v, b_v, c_v), a_v @ b_v @ c_v, rtol=1e-5)

    def test_stacked_dimshuffle_lift_does_not_crash(self):
        # `DimShuffle(DimShuffle(matmul))`: only the outer DimShuffle satisfies the
        # "single-client wrapping a chain-link matmul" pattern -- the inner
        # DimShuffle is not a chain link itself. The lift attempt on the outer
        # must bail (treat the whole stack as opaque) and produce a correct
        # graph.
        A = pt.matrix("A", shape=(2, 3))
        B = pt.matrix("B", shape=(3, 4))
        C = pt.tensor4("C", shape=(1, 1, 4, 5))
        D = pt.tensor4("D", shape=(1, 1, 5, 6))
        ab_stacked = pt.expand_dims(pt.expand_dims(A @ B, 0), 0)
        f = function([A, B, C, D], ab_stacked @ C @ D)

        rng = np.random.default_rng(37)
        a_v = rng.normal(size=(2, 3)).astype(config.floatX)
        b_v = rng.normal(size=(3, 4)).astype(config.floatX)
        c_v = rng.normal(size=(1, 1, 4, 5)).astype(config.floatX)
        d_v = rng.normal(size=(1, 1, 5, 6)).astype(config.floatX)
        ab_stacked_v = (a_v @ b_v)[None, None]
        np.testing.assert_allclose(
            f(a_v, b_v, c_v, d_v), ab_stacked_v @ c_v @ d_v, rtol=1e-5
        )

    def test_lift_blocked_by_multi_client_dimshuffle(self):
        # If the DimShuffle has multiple clients, lifting would duplicate work for
        # the OTHER clients. The rewriter must NOT lift in this case.
        A = pt.matrix("A", shape=(100, 2))
        B = pt.matrix("B", shape=(2, 100))
        C = pt.tensor3("C", shape=(1, 100, 3))
        D = pt.tensor3("D", shape=(1, 3, 100))
        ab3 = pt.expand_dims(A @ B, 0)
        out1 = ab3 @ C @ D
        out2 = ab3
        f = function([A, B, C, D], [out1, out2])
        rng = np.random.default_rng(19)
        a_v = rng.normal(size=(100, 2)).astype(config.floatX)
        b_v = rng.normal(size=(2, 100)).astype(config.floatX)
        c_v = rng.normal(size=(1, 100, 3)).astype(config.floatX)
        d_v = rng.normal(size=(1, 3, 100)).astype(config.floatX)
        ab3_v = (a_v @ b_v)[None]
        r1, r2 = f(a_v, b_v, c_v, d_v)
        np.testing.assert_allclose(r1, ab3_v @ c_v @ d_v, rtol=1e-5)
        np.testing.assert_allclose(r2, ab3_v, rtol=1e-5)
        # The original (A @ B) must still be in the graph because ab3 is also
        # consumed by out2; if we wrongly lifted the expand_dims, A @ B would be
        # gone.
        nodes = _matmul_nodes(f.maker.fgraph)
        assert any(n.outputs[0].type.shape == (100, 100) for n in nodes), (
            f"Expected A@B (100,100) to remain because ab3 has multiple clients; "
            f"got {[n.outputs[0].type.shape for n in nodes]}"
        )
