import numpy as np
import scipy.linalg
from numpy.testing import assert_allclose

from pytensor import function
from pytensor import tensor as pt
from pytensor.configdefaults import config
from pytensor.graph import FunctionGraph, ancestors
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.products import KroneckerProduct, MultiDot
from pytensor.tensor.math import Dot
from tests.unittest_tools import assert_equal_computations


def test_nested_blockdiag_fusion():
    x = pt.tensor("x", shape=(3, 3))
    y = pt.tensor("y", shape=(3, 3))
    z = pt.tensor("z", shape=(3, 3))

    inner = BlockDiagonal(2)(x, y)
    outer = BlockDiagonal(2)(inner, z)

    initial_count = sum(
        1
        for node in ancestors([outer])
        if getattr(node, "owner", None) and isinstance(node.owner.op, BlockDiagonal)
    )
    assert initial_count == 2, "Setup failed: expected 2 nested BlockDiagonal ops"

    fgraph = FunctionGraph(inputs=[x, y, z], outputs=[outer])
    rewrite_graph(fgraph, include=("fast_run", "blockdiag_fusion"))

    fused_nodes = [
        node for node in fgraph.toposort() if isinstance(node.op, BlockDiagonal)
    ]
    assert len(fused_nodes) == 1, "Nested BlockDiagonal ops were not fused"

    fused_op = fused_nodes[0].op
    assert fused_op.n_inputs == 3, f"Expected n_inputs=3, got {fused_op.n_inputs}"

    out_shape = fgraph.outputs[0].type.shape
    assert out_shape == (9, 9), f"Unexpected fused output shape: {out_shape}"


def test_deeply_nested_blockdiag_fusion():
    x = pt.tensor("x", shape=(3, 3))
    y = pt.tensor("y", shape=(3, 3))
    z = pt.tensor("z", shape=(3, 3))
    w = pt.tensor("w", shape=(3, 3))

    inner1 = BlockDiagonal(2)(x, y)
    inner2 = BlockDiagonal(2)(inner1, z)
    outer = BlockDiagonal(2)(inner2, w)

    fgraph = FunctionGraph(inputs=[x, y, z, w], outputs=[outer])
    rewrite_graph(fgraph, include=("fast_run", "blockdiag_fusion"))

    fused_block_diag_nodes = [
        node for node in fgraph.apply_nodes if isinstance(node.op, BlockDiagonal)
    ]
    assert len(fused_block_diag_nodes) == 1, (
        f"Expected 1 fused BlockDiagonal, got {len(fused_block_diag_nodes)}"
    )

    fused_block_diag_op = fused_block_diag_nodes[0].op

    assert fused_block_diag_op.n_inputs == 4, (
        f"Expected n_inputs=4 after fusion, got {fused_block_diag_op.n_inputs}"
    )

    out_shape = fgraph.outputs[0].type.shape
    expected_shape = (12, 12)  # 4 blocks of (3x3)
    assert out_shape == expected_shape, (
        f"Unexpected fused output shape: expected {expected_shape}, got {out_shape}"
    )


def test_diag_of_blockdiag():
    n_matrices = 10
    matrix_size = (5, 5)
    sub_matrices = pt.tensor("sub_matrices", shape=(n_matrices, *matrix_size))
    bd_output = pt.linalg.block_diag(*[sub_matrices[i] for i in range(n_matrices)])
    diag_output = pt.diag(bd_output)
    f_rewritten = function([sub_matrices], diag_output, mode="FAST_RUN")

    # Rewrite Test
    nodes = f_rewritten.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, BlockDiagonal) for node in nodes)

    # Value Test
    sub_matrices_test = np.random.rand(n_matrices, *matrix_size).astype(config.floatX)
    bd_output_test = scipy.linalg.block_diag(
        *[sub_matrices_test[i] for i in range(n_matrices)]
    )
    diag_output_test = np.diag(bd_output_test)
    rewritten_val = f_rewritten(sub_matrices_test)
    assert_allclose(
        diag_output_test,
        rewritten_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_det_of_blockdiag_():
    n_matrices = 100
    matrix_size = (5, 5)
    sub_matrices = pt.tensor("sub_matrices", shape=(n_matrices, *matrix_size))
    bd_output = pt.linalg.block_diag(*[sub_matrices[i] for i in range(n_matrices)])
    det_output = pt.linalg.det(bd_output)
    f_rewritten = function([sub_matrices], det_output, mode="FAST_RUN")

    # Rewrite Test
    nodes = f_rewritten.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, BlockDiagonal) for node in nodes)

    # Value Test
    sub_matrices_test = np.random.rand(n_matrices, *matrix_size).astype(config.floatX)
    bd_output_test = scipy.linalg.block_diag(
        *[sub_matrices_test[i] for i in range(n_matrices)]
    )
    det_output_test = np.linalg.det(bd_output_test)
    rewritten_val = f_rewritten(sub_matrices_test)
    assert_allclose(
        det_output_test,
        rewritten_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_slogdet_of_blockdiag():
    n_matrices = 10
    matrix_size = (5, 5)
    sub_matrices = pt.tensor("sub_matrices", shape=(n_matrices, *matrix_size))
    bd_output = pt.linalg.block_diag(*[sub_matrices[i] for i in range(n_matrices)])
    sign_output, logdet_output = pt.linalg.slogdet(bd_output)
    f_rewritten = function(
        [sub_matrices], [sign_output, logdet_output], mode="FAST_RUN"
    )

    # Rewrite Test
    nodes = f_rewritten.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, BlockDiagonal) for node in nodes)

    # Value Test
    sub_matrices_test = np.random.rand(n_matrices, *matrix_size).astype(config.floatX)
    bd_output_test = scipy.linalg.block_diag(
        *[sub_matrices_test[i] for i in range(n_matrices)]
    )
    sign_output_test, logdet_output_test = np.linalg.slogdet(bd_output_test)
    rewritten_sign_val, rewritten_logdet_val = f_rewritten(sub_matrices_test)
    assert_allclose(
        sign_output_test,
        rewritten_sign_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )
    assert_allclose(
        logdet_output_test,
        rewritten_logdet_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_diag_of_kronecker():
    a, b = pt.dmatrices("a", "b")
    kron_prod = pt.linalg.kron(a, b)
    diag_kron_prod = pt.diag(kron_prod)
    f_rewritten = function([a, b], diag_kron_prod, mode="FAST_RUN")

    # Rewrite Test
    nodes = f_rewritten.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, KroneckerProduct) for node in nodes)

    # Value Test
    a_test, b_test = np.random.rand(2, 20, 20)
    kron_prod_test = np.kron(a_test, b_test)
    diag_kron_prod_test = np.diag(kron_prod_test)
    rewritten_val = f_rewritten(a_test, b_test)
    assert_allclose(
        diag_kron_prod_test,
        rewritten_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_det_of_kronecker():
    a, b = pt.dmatrices("a", "b")
    kron_prod = pt.linalg.kron(a, b)
    det_output = pt.linalg.det(kron_prod)
    f_rewritten = function([a, b], [det_output], mode="FAST_RUN")

    # Rewrite Test
    nodes = f_rewritten.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, KroneckerProduct) for node in nodes)

    # Value Test
    a_test, b_test = np.random.rand(2, 20, 20)
    kron_prod_test = np.kron(a_test, b_test)
    det_output_test = np.linalg.det(kron_prod_test)
    rewritten_det_val = f_rewritten(a_test, b_test)
    assert_allclose(
        det_output_test,
        rewritten_det_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_slogdet_kronecker_rewrite():
    a, b = pt.dmatrices("a", "b")
    kron_prod = pt.linalg.kron(a, b)
    sign_output, logdet_output = pt.linalg.slogdet(kron_prod)
    f_rewritten = function([a, b], [sign_output, logdet_output], mode="FAST_RUN")

    # Rewrite Test
    nodes = f_rewritten.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, KroneckerProduct) for node in nodes)

    # Value Test
    a_test, b_test = np.random.rand(2, 20, 20)
    kron_prod_test = np.kron(a_test, b_test)
    sign_output_test, logdet_output_test = np.linalg.slogdet(kron_prod_test)
    rewritten_sign_val, rewritten_logdet_val = f_rewritten(a_test, b_test)
    assert_allclose(
        sign_output_test,
        rewritten_sign_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )
    assert_allclose(
        logdet_output_test,
        rewritten_logdet_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


class TestMultiDotRewrite:
    def test_dot_chain_absorbed(self):
        """A @ B @ C @ D should become MultiDot after canonicalize."""
        A = pt.matrix("A", shape=(10, 20))
        B = pt.matrix("B", shape=(20, 5))
        C = pt.matrix("C", shape=(5, 30))
        D = pt.matrix("D", shape=(30, 3))
        out = A @ B @ C @ D

        rewritten = rewrite_graph(out, include=("canonicalize",))
        assert rewritten.owner and isinstance(rewritten.owner.op, MultiDot)
        assert len(rewritten.owner.inputs) == 4

    def test_optimal_ordering(self):
        """Chain ordering should minimize FLOPs, not use naive left-to-right."""
        # (100x2) @ (2x100) @ (100x100): optimal is A @ (B @ C)
        A = pt.matrix("A", shape=(100, 2))
        B = pt.matrix("B", shape=(2, 100))
        C = pt.matrix("C", shape=(100, 100))

        f = function([A, B, C], A @ B @ C)
        dot_nodes = [n for n in f.maker.fgraph.toposort() if isinstance(n.op, Dot)]
        # First dot should be B @ C (2x100), not A @ B (100x100)
        assert dot_nodes[0].outputs[0].type.shape == (2, 100)

    def test_fuse_multi_dot_operands(self):
        A = pt.matrix("A", shape=(10, 20))
        B = pt.matrix("B", shape=(20, 5))
        C = pt.matrix("C", shape=(5, 30))
        D = pt.matrix("D", shape=(30, 8))
        E = pt.matrix("E", shape=(8, 3))
        F = pt.matrix("F", shape=(3, 12))

        out = pt.linalg.multi_dot([A, B, C]) @ pt.linalg.multi_dot([D, E, F])
        rewritten = rewrite_graph(out, include=("canonicalize",))
        expected = pt.linalg.multi_dot([A, B, C, D, E, F])
        assert_equal_computations([rewritten], [expected])

        out = A @ B @ pt.linalg.multi_dot([C, D, E, F])
        rewritten = rewrite_graph(out, include=("canonicalize",))
        expected = pt.linalg.multi_dot([A, B, C, D, E, F])
        assert_equal_computations([rewritten], [expected])

        out = pt.linalg.multi_dot([A, B, C, D]) @ E @ F
        rewritten = rewrite_graph(out, include=("canonicalize",))
        expected = pt.linalg.multi_dot([A, B, C, D, E, F])
        assert_equal_computations([rewritten], [expected])
