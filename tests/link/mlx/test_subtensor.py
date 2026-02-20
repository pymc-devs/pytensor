import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.tensor import subtensor as pt_subtensor
from pytensor.tensor import tensor
from tests.link.mlx.test_basic import compare_mlx_and_py


mx = pytest.importorskip("mlx.core")


def test_mlx_Subtensor_basic():
    """Test basic subtensor operations with constant indices."""
    shape = (3, 4, 5)
    x_pt = tensor("x", shape=shape, dtype="float32")
    x_np = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    # Basic indexing with single elements
    out_pt = x_pt[1, 2, 0]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    compare_mlx_and_py([x_pt], [out_pt], [x_np])

    # Basic indexing with slices
    out_pt = x_pt[1:, 1, :]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    compare_mlx_and_py([x_pt], [out_pt], [x_np])

    out_pt = x_pt[:2, 1, :]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    compare_mlx_and_py([x_pt], [out_pt], [x_np])

    out_pt = x_pt[1:2, 1, :]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    compare_mlx_and_py([x_pt], [out_pt], [x_np])

    # Negative indexing
    out_pt = x_pt[-1, -1, -1]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    compare_mlx_and_py([x_pt], [out_pt], [x_np])

    # Step slicing
    out_pt = x_pt[::2, ::2, ::2]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    compare_mlx_and_py([x_pt], [out_pt], [x_np])

    # Reverse indexing
    out_pt = x_pt[::-1, ::-1, ::-1]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    compare_mlx_and_py([x_pt], [out_pt], [x_np])


def test_mlx_AdvancedSubtensor():
    """Test advanced subtensor operations."""
    shape = (3, 4, 5)
    x_pt = tensor("x", shape=shape, dtype="float32")
    x_np = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    # Advanced indexing with array indices
    out_pt = pt_subtensor.advanced_subtensor1(x_pt, [1, 2])
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor1)
    compare_mlx_and_py([x_pt], [out_pt], [x_np])

    # Multi-dimensional advanced indexing
    out_pt = x_pt[[1, 2], [2, 3]]
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)
    compare_mlx_and_py([x_pt], [out_pt], [x_np])

    # Mixed advanced and basic indexing
    out_pt = x_pt[[1, 2], :]
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)
    compare_mlx_and_py([x_pt], [out_pt], [x_np])

    out_pt = x_pt[[1, 2], :, [3, 4]]
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)
    compare_mlx_and_py([x_pt], [out_pt], [x_np])


@pytest.mark.xfail(
    raises=ValueError, reason="MLX does not support boolean indexing yet"
)
def test_mlx_AdvancedSubtensor_boolean():
    """Test advanced subtensor operations with boolean indexing."""
    shape = (3, 4, 5)
    x_pt = tensor("x", shape=shape, dtype="float32")
    x_np = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    # Boolean indexing with constant mask
    bool_mask = np.array([True, False, True])
    out_pt = x_pt[bool_mask]
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)
    compare_mlx_and_py([x_pt], [out_pt], [x_np])


def test_mlx_IncSubtensor_set():
    """Test set operations using IncSubtensor (set_instead_of_inc=True)."""
    # Test data
    x_np = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
    x_pt = pt.constant(x_np)

    # Set single element
    st_pt = pt.as_tensor_variable(np.array(-10.0, dtype=np.float32))
    out_pt = pt_subtensor.set_subtensor(x_pt[1, 2, 3], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    assert out_pt.owner.op.set_instead_of_inc
    compare_mlx_and_py([], [out_pt], [])


def test_mlx_IncSubtensor_increment():
    """Test increment operations using IncSubtensor (set_instead_of_inc=False)."""
    # Test data
    x_np = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
    x_pt = pt.constant(x_np)

    # Increment single element
    st_pt = pt.as_tensor_variable(np.array(-10.0, dtype=np.float32))
    out_pt = pt_subtensor.inc_subtensor(x_pt[1, 2, 3], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    assert not out_pt.owner.op.set_instead_of_inc
    compare_mlx_and_py([], [out_pt], [])


def test_mlx_AdvancedIncSubtensor_set():
    """Test advanced set operations using AdvancedIncSubtensor."""
    rng = np.random.default_rng(213234)

    # Test data
    x_np = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
    x_pt = pt.constant(x_np)

    # Set with advanced indexing - this actually works in MLX!
    st_pt = pt.as_tensor_variable(rng.uniform(-1, 1, size=(2, 4, 5)).astype(np.float32))
    out_pt = pt_subtensor.set_subtensor(x_pt[np.r_[0, 2]], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    assert out_pt.owner.op.set_instead_of_inc
    compare_mlx_and_py([], [out_pt], [])


def test_mlx_AdvancedIncSubtensor_increment():
    """Test advanced increment operations using AdvancedIncSubtensor."""
    rng = np.random.default_rng(213234)

    # Test data
    x_np = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
    x_pt = pt.constant(x_np)

    # Increment with advanced indexing - this actually works in MLX!
    st_pt = pt.as_tensor_variable(rng.uniform(-1, 1, size=(2, 4, 5)).astype(np.float32))
    out_pt = pt_subtensor.inc_subtensor(x_pt[np.r_[0, 2]], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    assert not out_pt.owner.op.set_instead_of_inc
    compare_mlx_and_py([], [out_pt], [])


def test_mlx_AdvancedIncSubtensor1_operations():
    """Test AdvancedIncSubtensor1 operations (handled by IncSubtensor dispatcher)."""
    rng = np.random.default_rng(213234)

    # Test data
    x_np = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
    x_pt = pt.constant(x_np)

    # Test set operation - this actually works in MLX!
    st_pt = pt.as_tensor_variable(rng.uniform(-1, 1, size=(2, 4, 5)).astype(np.float32))
    indices = [1, 2]

    # Create AdvancedIncSubtensor1 manually for set operation
    out_pt = pt_subtensor.advanced_set_subtensor1(x_pt, st_pt, indices)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor1)
    assert out_pt.owner.op.set_instead_of_inc
    compare_mlx_and_py([], [out_pt], [])


@pytest.mark.xfail(reason="Inplace operations not yet supported in MLX mode")
def test_mlx_inplace_variants():
    """Test inplace variants of all subtensor operations."""
    # Test data
    x_np = np.arange(12, dtype=np.float32).reshape((3, 4))
    x_pt = pt.constant(x_np)

    # Test inplace IncSubtensor (set)
    st_pt = pt.as_tensor_variable(np.array([-1.0, -2.0], dtype=np.float32))
    out_pt = pt_subtensor.set_subtensor(x_pt[0, :2], st_pt, inplace=True)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    assert out_pt.owner.op.inplace
    assert out_pt.owner.op.set_instead_of_inc
    compare_mlx_and_py([], [out_pt], [])


def test_mlx_subtensor_edge_cases():
    """Test edge cases and boundary conditions."""
    # Empty slices - use constant array
    x_pt = pt.constant(np.arange(10, dtype=np.float32))
    out_pt = x_pt[5:5]  # Empty slice
    compare_mlx_and_py([], [out_pt], [])

    # Single element arrays
    x_pt = pt.tensor(shape=(1,), dtype="float32", name="x")
    x_np = np.array([42.0], dtype=np.float32)
    out_pt = x_pt[0]
    compare_mlx_and_py([x_pt], [out_pt], [x_np])

    # Large step sizes - use constant array
    x_pt = pt.constant(np.arange(20, dtype=np.float32))
    out_pt = x_pt[::5]
    compare_mlx_and_py([], [out_pt], [])

    # Negative steps - use constant array
    x_pt = pt.constant(np.arange(10, dtype=np.float32))
    out_pt = x_pt[::-2]
    compare_mlx_and_py([], [out_pt], [])


@pytest.mark.xfail(reason="MLX indexing with tuples not yet supported")
def test_mlx_subtensor_with_variables():
    """Test subtensor operations with PyTensor variables as inputs."""
    # Test with variable arrays (not constants)
    x_pt = pt.matrix("x", dtype="float32")
    y_pt = pt.vector("y", dtype="float32")

    x_np = np.arange(12, dtype=np.float32).reshape((3, 4))
    y_np = np.array([-1.0, -2.0], dtype=np.float32)

    # Set operation with variables
    out_pt = pt_subtensor.set_subtensor(x_pt[0, :2], y_pt)
    compare_mlx_and_py([x_pt, y_pt], [out_pt], [x_np, y_np])
