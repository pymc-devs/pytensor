"""Tests for ONNX backend linear algebra operations (Tier 4)."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from tests.link.onnx.test_basic import compare_onnx_and_py, get_onnx_node_types


# Matrix Multiplication Tests


def test_dot_2d():
    """Test 2D matrix multiplication (Dot op)."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    A = pt.matrix("A", dtype="float32")
    B = pt.matrix("B", dtype="float32")
    C = pt.dot(A, B)

    A_val = np.random.randn(3, 4).astype("float32")
    B_val = np.random.randn(4, 5).astype("float32")

    fn, result = compare_onnx_and_py([A, B], C, [A_val, B_val])

    expected = np.dot(A_val, B_val)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    # Verify ONNX uses MatMul
    node_types = get_onnx_node_types(fn)
    assert "MatMul" in node_types, f"Expected 'MatMul' node, got {node_types}"


def test_dot_1d_2d():
    """Test vector-matrix multiplication."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    v = pt.vector("v", dtype="float32")
    M = pt.matrix("M", dtype="float32")
    result = pt.dot(v, M)

    v_val = np.random.randn(4).astype("float32")
    M_val = np.random.randn(4, 5).astype("float32")

    fn, output = compare_onnx_and_py([v, M], result, [v_val, M_val])

    expected = np.dot(v_val, M_val)
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)

    # Should be 1D output
    assert output.ndim == 1, f"Expected 1D output, got shape {output.shape}"


def test_batched_dot():
    """Test batched matrix multiplication."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    A = pt.tensor3("A", dtype="float32")
    B = pt.tensor3("B", dtype="float32")
    C = pt.batched_dot(A, B)

    A_val = np.random.randn(2, 3, 4).astype("float32")
    B_val = np.random.randn(2, 4, 5).astype("float32")

    fn, result = compare_onnx_and_py([A, B], C, [A_val, B_val])

    expected = np.einsum("bij,bjk->bik", A_val, B_val)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    # ONNX MatMul handles batched operations natively
    node_types = get_onnx_node_types(fn)
    assert "MatMul" in node_types


def test_gemm():
    """Test GEMM: beta*C + alpha*A@B."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from pytensor.tensor.blas import gemm

    A = pt.matrix("A", dtype="float32")
    B = pt.matrix("B", dtype="float32")
    C = pt.matrix("C", dtype="float32")

    # GEMM: gemm(C, alpha, A, B, beta) = beta*C + alpha*dot(A, B)
    # GEMM: 0.5 * C + 2.0 * A @ B
    alpha = np.float32(2.0)
    beta = np.float32(0.5)
    result = gemm(C, alpha, A, B, beta)

    A_val = np.random.randn(3, 4).astype("float32")
    B_val = np.random.randn(4, 5).astype("float32")
    C_val = np.random.randn(3, 5).astype("float32")

    fn, output = compare_onnx_and_py([A, B, C], result, [A_val, B_val, C_val])

    expected = beta * C_val + alpha * np.dot(A_val, B_val)
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)

    # ONNX has Gemm operator
    node_types = get_onnx_node_types(fn)
    assert "Gemm" in node_types, f"Expected 'Gemm' node, got {node_types}"


# Matrix Decomposition Tests (Unsupported)


@pytest.mark.skip(
    reason="SVD not in standard ONNX opset - requires contrib ops or custom implementation"
)
def test_svd_not_supported():
    """Test SVD - expected to be unsupported in standard ONNX.

    SVD decomposes A into U, S, V.T where A = U @ diag(S) @ V.T
    This is NOT available in standard ONNX opset.

    Options:
    1. Use ONNX Runtime contrib op (platform-specific)
    2. Implement as sequence of operations (very complex)
    3. Skip and document as unsupported

    This test documents the expected behavior if we choose to implement.
    """
    from pytensor.link.onnx.linker import ONNXLinker
    from pytensor.tensor.nlinalg import svd

    A = pt.matrix("A", dtype="float32")
    U, s, Vt = svd(A, full_matrices=False)

    # Well-conditioned test matrix
    rng = np.random.default_rng(42)
    A_val = rng.normal(size=(4, 3)).astype("float32")

    # This will raise NotImplementedError
    onnx_mode = Mode(linker=ONNXLinker(), optimizer=None)
    with pytest.raises(NotImplementedError, match="SVD not supported"):
        fn = pytensor.function([A], [U, s, Vt], mode=onnx_mode)


@pytest.mark.skip(reason="Cholesky not in standard ONNX opset")
def test_cholesky_not_supported():
    """Test Cholesky decomposition - not in standard ONNX.

    Cholesky decomposes positive definite A into L @ L.T
    where L is lower triangular.

    Not available in standard ONNX opset. ONNX Runtime may have
    contrib op: com.microsoft.Cholesky
    """
    from pytensor.link.onnx.linker import ONNXLinker
    from pytensor.tensor.slinalg import cholesky

    A = pt.matrix("A", dtype="float32")
    L = cholesky(A)

    # Positive definite matrix
    rng = np.random.default_rng(42)
    X = rng.normal(size=(4, 4)).astype("float32")
    A_val = X @ X.T  # Positive definite

    onnx_mode = Mode(linker=ONNXLinker(), optimizer=None)
    with pytest.raises(NotImplementedError, match="Cholesky not supported"):
        fn = pytensor.function([A], L, mode=onnx_mode)


# Linear System Solving Tests (Unsupported)


@pytest.mark.skip(reason="Solve not in standard ONNX opset")
def test_solve_not_supported():
    """Test Solve operation - not in standard ONNX.

    Solve finds X such that A @ X = B.
    Not available in standard ONNX. Would require:
    - LU decomposition (not in ONNX)
    - Forward/backward substitution
    - Or matrix inverse + matmul
    """
    from pytensor.link.onnx.linker import ONNXLinker
    from pytensor.tensor.slinalg import solve

    A = pt.matrix("A", dtype="float32")
    B = pt.matrix("B", dtype="float32")
    X = solve(A, B)

    rng = np.random.default_rng(42)
    A_val = rng.normal(size=(4, 4)).astype("float32")
    A_val = A_val + 0.5 * np.eye(4, dtype="float32")  # Well-conditioned
    B_val = rng.normal(size=(4, 3)).astype("float32")

    onnx_mode = Mode(linker=ONNXLinker(), optimizer=None)
    with pytest.raises(NotImplementedError, match="Solve not supported"):
        fn = pytensor.function([A, B], X, mode=onnx_mode)


# Matrix Properties Tests (Unsupported)


@pytest.mark.skip(
    reason="Det requires LU decomposition - complex custom implementation needed"
)
def test_det_custom_implementation():
    """Test matrix determinant - requires custom implementation.

    Determinant can be computed via:
    1. LU decomposition + product of diagonal (preferred)
    2. QR decomposition + product of R diagonal
    3. Direct computation for small matrices

    All approaches require operations not in standard ONNX.
    """
    from pytensor.link.onnx.linker import ONNXLinker
    from pytensor.tensor.nlinalg import det

    A = pt.matrix("A", dtype="float32")
    d = det(A)

    rng = np.random.default_rng(42)
    A_val = rng.normal(size=(4, 4)).astype("float32")

    onnx_mode = Mode(linker=ONNXLinker(), optimizer=None)
    with pytest.raises(NotImplementedError, match="Det not supported"):
        fn = pytensor.function([A], d, mode=onnx_mode)


@pytest.mark.skip(reason="Matrix inverse not in standard ONNX opset")
def test_matrix_inverse_not_supported():
    """Test matrix inverse - not in standard ONNX.

    Matrix inverse could be implemented via:
    1. LU decomposition + solving (not available)
    2. Adjugate method (very complex)
    3. Gradient descent (iterative, expensive)

    Not practical for standard ONNX export.
    """
    from pytensor.link.onnx.linker import ONNXLinker
    from pytensor.tensor.nlinalg import matrix_inverse

    A = pt.matrix("A", dtype="float32")
    A_inv = matrix_inverse(A)

    rng = np.random.default_rng(42)
    A_val = rng.normal(size=(4, 4)).astype("float32")
    A_val = A_val + 0.5 * np.eye(4, dtype="float32")

    onnx_mode = Mode(linker=ONNXLinker(), optimizer=None)
    with pytest.raises(NotImplementedError, match="Matrix inverse not supported"):
        fn = pytensor.function([A], A_inv, mode=onnx_mode)


# Extract Diagonal Tests


def test_extract_diag():
    """Test extracting diagonal from matrix.

    This CAN be implemented in ONNX using:
    - Identity matrix of appropriate size
    - Element-wise multiply with input
    - ReduceSum along one axis

    Or using Gather operations.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    A = pt.matrix("A", dtype="float32")
    d = pt.diag(A)  # Extract diagonal

    A_val = np.random.randn(4, 4).astype("float32")

    fn, result = compare_onnx_and_py([A], d, [A_val])

    expected = np.diag(A_val)
    np.testing.assert_array_equal(result, expected)
