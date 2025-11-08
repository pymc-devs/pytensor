"""Integration tests for ONNX backend - complete models and workflows."""

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.tensor.special import softmax
from tests.link.onnx.test_basic import compare_onnx_and_py


def test_simple_mlp():
    """Test simple MLP using matmul, add, and activation.

    This integration test verifies that a complete neural network
    layer can be exported to ONNX.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    # Input
    x = pt.matrix("x", dtype="float32")

    # Weights and biases
    W1 = pt.matrix("W1", dtype="float32")
    b1 = pt.vector("b1", dtype="float32")
    W2 = pt.matrix("W2", dtype="float32")
    b2 = pt.vector("b2", dtype="float32")

    # Layer 1: x @ W1 + b1, then ReLU
    h = pt.maximum(pt.dot(x, W1) + b1, 0)

    # Layer 2: h @ W2 + b2, then softmax (axis=-1 for row-wise probabilities)
    logits = pt.dot(h, W2) + b2
    output = softmax(logits, axis=-1)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(5, 10)).astype("float32")
    W1_val = rng.normal(size=(10, 20)).astype("float32")
    b1_val = rng.normal(size=(20,)).astype("float32")
    W2_val = rng.normal(size=(20, 3)).astype("float32")
    b2_val = rng.normal(size=(3,)).astype("float32")

    fn, result = compare_onnx_and_py(
        [x, W1, b1, W2, b2], output, [x_val, W1_val, b1_val, W2_val, b2_val]
    )

    # Verify output is valid probabilities
    assert result.shape == (5, 3), f"Expected shape (5, 3), got {result.shape}"
    assert np.allclose(result.sum(axis=1), 1.0), "Softmax should sum to 1"
    assert np.all(result >= 0) and np.all(result <= 1), "Probabilities should be in [0, 1]"
