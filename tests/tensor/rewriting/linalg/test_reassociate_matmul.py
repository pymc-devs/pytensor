import functools
import operator

import numpy as np
import pytest

import pytensor.tensor as pt
import pytensor.tensor.rewriting.linalg.reassociate_matmul as reassociate
from pytensor import function
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.tensor.blas import Dot22, Dot22Scalar, Gemm, Gemv
from pytensor.tensor.rewriting.linalg.reassociate_matmul import (
    CostExpr,
    _provably_cheaper,
    reassociate_matmul_chain,
)
from tests.unittest_tools import RewriteTester


def _reassociate(inputs, outputs):
    return RewriteTester(
        inputs,
        outputs,
        include=("canonicalize",),
        custom_rewrite=reassociate_matmul_chain,
    )


def _canonical(*outputs):
    return rewrite_graph(list(outputs), include=("canonicalize",))


class TestProvablyCheaper:
    def test_int_dominance(self):
        a = CostExpr.from_dim_product([10, 20, 5])  # 1000
        b = CostExpr.from_dim_product([10, 20, 5]) + CostExpr.from_dim_product(
            [1, 2, 3]
        )  # 1006
        assert _provably_cheaper(a, b)
        assert not _provably_cheaper(b, a)
        assert not _provably_cheaper(a, a)

    def test_zero_vs_positive(self):
        a = CostExpr.zero()
        b = CostExpr.from_dim_product([7])
        assert _provably_cheaper(a, b)
        assert not _provably_cheaper(b, a)
        assert not _provably_cheaper(a, a)

    def test_symbolic_no_proof(self):
        # m*k*n vs k*n*p: different monomials, no dominance either way.
        m, k, n, p = (pt.scalar(name) for name in "mknp")
        a = CostExpr.from_dim_product([m, k, n])
        b = CostExpr.from_dim_product([k, n, p])
        assert not _provably_cheaper(a, b)
        assert not _provably_cheaper(b, a)

    def test_extra_unmatched_b_monomial(self):
        # a = m*k, b = m*k + p: b strictly larger under p >= 1.
        m, k, p = (pt.scalar(name) for name in "mkp")
        a = CostExpr.from_dim_product([m, k])
        b = CostExpr.from_dim_product([m, k]) + CostExpr.from_dim_product([p])
        assert _provably_cheaper(a, b)
        assert not _provably_cheaper(b, a)

    def test_tie_at_ones_is_cheaper(self):
        # a = m*k, b = m*k*p: equal at p == 1, so a is never worse and better for p > 1
        m, k, p = (pt.scalar(name) for name in "mkp")
        a = CostExpr.from_dim_product([m, k])
        b = CostExpr.from_dim_product([m, k, p])
        assert _provably_cheaper(a, b)
        assert not _provably_cheaper(b, a)

    def test_split_coefficient(self):
        # 2*n^2 <= n^3 + n^2 needs one unit of a's coefficient on each b monomial
        n = pt.scalar("n")
        a = CostExpr.from_dim_product([2, n, n])
        b = CostExpr.from_dim_product([n, n, n]) + CostExpr.from_dim_product([n, n])
        assert _provably_cheaper(a, b)
        assert not _provably_cheaper(b, a)

    @pytest.mark.parametrize(
        "build",
        [
            # 2*n*k > k*n^2 + n^2 at n = 1, k = 10
            lambda n, k: (
                CostExpr.from_dim_product([2, n, k]),
                CostExpr.from_dim_product([k, n, n])
                + CostExpr.from_dim_product([n, n]),
            ),
            # 3*n^2 > n^3 + n^2 at n = 1
            lambda n, k: (
                CostExpr.from_dim_product([3, n, n]),
                CostExpr.from_dim_product([n, n, n])
                + CostExpr.from_dim_product([n, n]),
            ),
        ],
        ids=["low_rank", "excess_coefficient"],
    )
    def test_false_for_some_sizes(self, build):
        a, b = build(pt.scalar("n"), pt.scalar("k"))
        assert not _provably_cheaper(a, b)

    def test_assignment_needs_rerouting(self):
        # Assigning a's x to b's first dominator (x*y) leaves a's x*y uncovered. The
        # valid assignment is x <-> x*z and x*y <-> x*y.
        x, y, z, w = (pt.scalar(name) for name in "xyzw")
        a = CostExpr.from_dim_product([x]) + CostExpr.from_dim_product([x, y])
        b = (
            CostExpr.from_dim_product([x, y])
            + CostExpr.from_dim_product([x, z])
            + CostExpr.from_dim_product([w])
        )
        assert _provably_cheaper(a, b)


class TestReassociateChain:
    def test_optimal_static_ordering(self):
        A = pt.matrix("A", shape=(100, 2))
        B = pt.matrix("B", shape=(2, 100))
        C = pt.matrix("C", shape=(100, 100))

        result = _reassociate([A, B, C], [A @ B @ C])
        result.assert_graph(*_canonical(A @ (B @ C)))
        rng = np.random.default_rng(0)
        result.assert_eval(
            rng.normal(size=(100, 2)),
            rng.normal(size=(2, 100)),
            rng.normal(size=(100, 100)),
        )

    @pytest.mark.parametrize(
        "shapes, expected",
        [
            # Cost 1350 against 3400 for the user's left-to-right order
            (
                [(10, 20), (20, 5), (5, 30), (30, 3)],
                lambda A, B, C, D: A @ (B @ (C @ D)),
            ),
            # CLRS 15.2
            (
                [(30, 35), (35, 15), (15, 5), (5, 10), (10, 20)],
                lambda A, B, C, D, E: (A @ (B @ C)) @ (D @ E),
            ),
        ],
        ids=["four_matrices", "clrs"],
    )
    def test_textbook_chains(self, shapes, expected):
        matrices = [pt.matrix(f"M{i}", shape=s) for i, s in enumerate(shapes)]
        out = functools.reduce(operator.matmul, matrices)

        result = _reassociate(matrices, [out])
        result.assert_graph(*_canonical(expected(*matrices)))
        rng = np.random.default_rng(0)
        result.assert_eval(*(rng.normal(size=shape) for shape in shapes))

    def test_balanced_tree_is_one_chain(self):
        A = pt.matrix("A", shape=(100, 2))
        B = pt.matrix("B", shape=(2, 100))
        C = pt.matrix("C", shape=(100, 3))
        D = pt.matrix("D", shape=(3, 100))

        result = _reassociate([A, B, C, D], [(A @ B) @ (C @ D)])
        result.assert_graph(*_canonical(A @ ((B @ C) @ D)))

    def test_right_associated_chain(self):
        A = pt.matrix("A", shape=(2, 100))
        B = pt.matrix("B", shape=(100, 100))
        C = pt.matrix("C", shape=(100, 5))

        result = _reassociate([A, B, C], [A @ (B @ C)])
        result.assert_graph(*_canonical((A @ B) @ C))

    def test_batched_chain(self):
        A = pt.tensor3("A", shape=(7, 100, 2))
        B = pt.tensor3("B", shape=(7, 2, 100))
        C = pt.tensor3("C", shape=(7, 100, 5))

        result = _reassociate([A, B, C], [A @ B @ C])
        result.assert_graph(*_canonical(A @ (B @ C)))
        rng = np.random.default_rng(0)
        result.assert_eval(
            rng.normal(size=(7, 100, 2)),
            rng.normal(size=(7, 2, 100)),
            rng.normal(size=(7, 100, 5)),
        )

    def test_broadcast_batch_chain(self):
        A = pt.tensor3("A", shape=(1, 4, 5))
        B = pt.tensor3("B", shape=(7, 5, 6))
        C = pt.tensor3("C", shape=(7, 6, 4))

        # A @ (B @ C) costs 1400 against 1512, broadcasting A's batch dim of 1
        result = _reassociate([A, B, C], [(A @ B) @ C])
        result.assert_graph(*_canonical(A @ (B @ C)))
        rng = np.random.default_rng(0)
        result.assert_eval(
            rng.normal(size=(1, 4, 5)),
            rng.normal(size=(7, 5, 6)),
            rng.normal(size=(7, 6, 4)),
        )

    def test_batched_outer_product_link(self):
        A = pt.tensor3("A", shape=(3, 5, 5))
        g = pt.tensor3("g", shape=(3, 5, 1))
        v = pt.matrix("v", shape=(1, 5))

        # A @ (g @ v) costs 450 against 150, with v broadcast over the batch
        result = _reassociate([A, g, v], [A @ (g @ v)])
        result.assert_graph(*_canonical((A @ g) @ v))
        rng = np.random.default_rng(0)
        result.assert_eval(
            rng.normal(size=(3, 5, 5)),
            rng.normal(size=(3, 5, 1)),
            rng.normal(size=(1, 5)),
        )

    def test_symbolic_gp_gradient_chain(self):
        # The Cholesky pullback of L @ v builds L.T @ (g @ v.T) with a square L: n^3 + n^2
        # against 2n^2 for (L.T @ g) @ v.T
        n = pt.lscalar("n")
        L = pt.specify_shape(pt.matrix("L"), (n, n))
        g = pt.matrix("g", shape=(None, 1))
        v = pt.matrix("v", shape=(1, None))

        result = _reassociate([L.owner.inputs[0], n, g, v], [L.T @ (g @ v)])
        result.assert_graph(*_canonical((L.T @ g) @ v))

    def test_row_times_column_product_link(self):
        # Mul commutes, so v * g is still the length-1 contraction g @ v
        n = pt.lscalar("n")
        L = pt.specify_shape(pt.matrix("L"), (n, n))
        g = pt.matrix("g", shape=(None, 1))
        v = pt.matrix("v", shape=(1, None))

        result = _reassociate([L.owner.inputs[0], n, g, v], [L.T @ (v * g)])
        result.assert_graph(*_canonical((L.T @ g) @ v))
        rng = np.random.default_rng(0)
        result.assert_eval(
            rng.normal(size=(4, 4)), 4, rng.normal(size=(4, 1)), rng.normal(size=(1, 4))
        )

    def test_mixed_static_and_symbolic_proof(self):
        # A @ (B @ C) costs 9 + 9n, (A @ B) @ C costs 6n, cheaper for every n >= 1
        A = pt.matrix("A", shape=(None, 3))
        B = pt.matrix("B", shape=(3, 1))
        C = pt.matrix("C", shape=(1, 3))

        result = _reassociate([A, B, C], [A @ (B @ C)])
        result.assert_graph(*_canonical((A @ B) @ C))
        rng = np.random.default_rng(0)
        result.assert_eval(
            rng.normal(size=(4, 3)), rng.normal(size=(3, 1)), rng.normal(size=(1, 3))
        )

    def test_incomparable_candidates_resolved_by_operand_order(self):
        # B @ (C @ D) costs 2pn + 4p and (B @ C) @ D costs 2pn + 4n. Both provably beat
        # the user's order, so the pick must depend on operand order, never on hashes
        n, p = pt.lscalar("n"), pt.lscalar("p")
        A = pt.matrix("A", shape=(3, 2))
        B = pt.specify_shape(pt.matrix("B", shape=(2, None)), (2, p))
        C = pt.specify_shape(pt.matrix("C"), (p, n))
        D = pt.specify_shape(pt.matrix("D", shape=(None, 2)), (n, 2))
        E = pt.matrix("E", shape=(2, 2))

        inputs = [A, B.owner.inputs[0], C.owner.inputs[0], D.owner.inputs[0], E, n, p]
        for _ in range(5):
            result = _reassociate(inputs, [A @ B @ C @ D @ E])
            result.assert_graph(*_canonical(A @ ((B @ (C @ D)) @ E)))

    @pytest.mark.parametrize(
        "shapes, build",
        [
            ([(None, None)] * 3, lambda A, B, C: A @ (B @ C)),
            # Reassociating costs 2nk against kn^2 + n^2, worse at n = 1, k = 10
            ([(None, None), (None, None), (None, 1)], lambda A, B, C: (A @ B) @ C),
            # Already optimal: 21_000 against 51_000 for A @ (B @ C)
            ([(2, 100), (100, 100), (100, 5)], lambda A, B, C: (A @ B) @ C),
            ([(10, 30), (30, 5), (5, 60)], lambda A, B, C: (A @ B) @ C),
        ],
        ids=["symbolic", "symbolic_low_rank", "already_optimal", "textbook_optimal"],
    )
    def test_user_order_kept(self, shapes, build):
        matrices = [pt.matrix(name, shape=s) for name, s in zip("ABC", shapes)]
        out = build(*matrices)

        result = _reassociate(matrices, [out])
        result.assert_graph(*_canonical(out))

    def test_quadratic_form_tie_kept(self):
        # Both orders cost 750_000, so neither is provably cheaper
        A = pt.matrix("A", shape=(50, 100))
        B = pt.matrix("B", shape=(100, 100))
        out = A @ B @ A.T

        result = _reassociate([A, B], [out])
        result.assert_graph(*_canonical(out))

    def test_multi_client_intermediate_is_a_leaf(self):
        A = pt.matrix("A", shape=(100, 2))
        B = pt.matrix("B", shape=(2, 100))
        C = pt.matrix("C", shape=(100, 100))
        D = pt.matrix("D", shape=(100, 5))
        ab = A @ B

        result = _reassociate([A, B, C, D], [ab @ C @ D, ab.sum()])
        result.assert_graph(*_canonical(ab @ (C @ D), ab.sum()))

    @pytest.mark.parametrize("assoc", ["left", "right"])
    def test_failed_chain_solved_once(self, monkeypatch, assoc):
        solve = reassociate._solve_chain
        calls = []
        monkeypatch.setattr(
            reassociate,
            "_solve_chain",
            lambda *args, **kwargs: calls.append(None) or solve(*args, **kwargs),
        )
        matrices = [pt.matrix(f"M{i}") for i in range(10)]
        if assoc == "left":
            out = functools.reduce(operator.matmul, matrices)
        else:
            out = functools.reduce(lambda acc, m: m @ acc, reversed(matrices))

        result = _reassociate(matrices, [out])
        result.assert_graph(*_canonical(out))
        assert len(calls) == 1


class TestLifting:
    def test_expand_dims(self):
        A = pt.matrix("A", shape=(100, 2))
        B = pt.matrix("B", shape=(2, 100))
        C = pt.tensor3("C", shape=(1, 100, 3))
        D = pt.tensor3("D", shape=(1, 3, 100))
        out = pt.expand_dims(A @ B, 0) @ C @ D

        result = _reassociate([A, B, C, D], [out])
        result.assert_graph(
            *_canonical(pt.expand_dims(A, 0) @ ((pt.expand_dims(B, 0) @ C) @ D))
        )
        rng = np.random.default_rng(0)
        result.assert_eval(
            rng.normal(size=(100, 2)),
            rng.normal(size=(2, 100)),
            rng.normal(size=(1, 100, 3)),
            rng.normal(size=(1, 3, 100)),
        )

    def test_squeeze(self):
        A = pt.tensor3("A", shape=(1, 100, 2))
        B = pt.tensor3("B", shape=(1, 2, 100))
        C = pt.matrix("C", shape=(100, 3))
        D = pt.matrix("D", shape=(3, 100))
        out = pt.squeeze(A @ B, axis=0) @ C @ D

        result = _reassociate([A, B, C, D], [out])
        result.assert_graph(
            *_canonical(pt.squeeze(A, axis=0) @ ((pt.squeeze(B, axis=0) @ C) @ D))
        )

    def test_matrix_transpose(self):
        L = pt.matrix("L", shape=(100, 2))
        R = pt.matrix("R", shape=(2, 50))
        C = pt.matrix("C", shape=(100, 100))

        result = _reassociate([L, R, C], [(L @ R).T @ C])
        result.assert_graph(*_canonical(R.T @ (L.T @ C)))
        rng = np.random.default_rng(0)
        result.assert_eval(
            rng.normal(size=(100, 2)),
            rng.normal(size=(2, 50)),
            rng.normal(size=(100, 100)),
        )

    def test_uncommitted_lift_adds_nothing(self):
        # Symbolic shapes: no provable win, so the speculative lift must leave no trace
        L, R, C, D = (pt.matrix(name) for name in "LRCD")
        out = pt.expand_dims(L @ R, 0) @ pt.expand_dims(C, 0) @ pt.expand_dims(D, 0)

        result = _reassociate([L, R, C, D], [out])
        result.assert_graph(*_canonical(out))

    def test_stacked_dimshuffle_not_lifted(self):
        # Only the outer of two stacked DimShuffles wraps the matmul
        A = pt.matrix("A", shape=(2, 3))
        B = pt.matrix("B", shape=(3, 4))
        C = pt.tensor4("C", shape=(1, 1, 4, 5))
        D = pt.tensor4("D", shape=(1, 1, 5, 6))
        out = pt.expand_dims(pt.expand_dims(A @ B, 0), 0) @ C @ D

        result = _reassociate([A, B, C, D], [out])
        result.assert_graph(*_canonical(out))

    def test_multi_client_dimshuffle_not_lifted(self):
        A = pt.matrix("A", shape=(100, 2))
        B = pt.matrix("B", shape=(2, 100))
        C = pt.tensor3("C", shape=(1, 100, 3))
        D = pt.tensor3("D", shape=(1, 3, 100))
        ab3 = pt.expand_dims(A @ B, 0)
        outs = [ab3 @ C @ D, ab3]

        result = _reassociate([A, B, C, D], outs)
        result.assert_graph(*_canonical(*outs))


def _blas_nodes(fn):
    return sorted(
        (type(node.op).__name__, node.outputs[0].type.shape)
        for node in fn.maker.fgraph.apply_nodes
        if isinstance(node.op, Dot22 | Dot22Scalar | Gemm | Gemv)
    )


class TestBackends:
    @pytest.mark.parametrize(
        "build, outer_op",
        [
            (lambda A, B, C, D: A @ B @ C, "Dot22"),
            (lambda A, B, C, D: A @ B @ C + D, "Gemm"),
            (lambda A, B, C, D: 2 * (A @ B @ C), "Dot22Scalar"),
        ],
        ids=["dot22", "gemm", "dot22scalar"],
    )
    def test_cvm_blas_chain(self, build, outer_op):
        inputs = [
            pt.matrix("A", shape=(100, 2)),
            pt.matrix("B", shape=(2, 100)),
            pt.matrix("C", shape=(100, 100)),
            pt.matrix("D", shape=(100, 100)),
        ]

        fn = function(inputs, build(*inputs), mode="CVM", on_unused_input="ignore")
        assert _blas_nodes(fn) == [("Dot22", (2, 100)), (outer_op, (100, 100))]

    def test_cvm_gemv_chain(self):
        C = pt.matrix("C", shape=(100, 100))
        D = pt.matrix("D", shape=(100, 100))
        y = pt.vector("y", shape=(100,))

        fn = function([C, D, y], C @ D @ y, mode="CVM")
        assert _blas_nodes(fn) == [("Gemv", (100,)), ("Gemv", (100,))]
