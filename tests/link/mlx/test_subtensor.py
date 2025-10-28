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

    # Increment slice
    out_pt = pt_subtensor.inc_subtensor(x_pt[:, :, 2:], st_pt)
    compare_mlx_and_py([], [out_pt], [])

    out_pt = pt_subtensor.inc_subtensor(x_pt[:, :, -3:], st_pt)
    compare_mlx_and_py([], [out_pt], [])

    out_pt = pt_subtensor.inc_subtensor(x_pt[::2, ::2, ::2], st_pt)
    compare_mlx_and_py([], [out_pt], [])

    out_pt = pt_subtensor.inc_subtensor(x_pt[:, :, :], st_pt)
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


@pytest.mark.xfail(
    reason="MLX slice indices must be integers or None, dynamic slices not supported"
)
def test_mlx_MakeSlice():
    """Test MakeSlice operation."""
    # Test slice creation
    start = pt.iscalar("start")
    stop = pt.iscalar("stop")
    step = pt.iscalar("step")

    # Create a slice using MakeSlice
    slice_op = pt_subtensor.MakeSlice()
    slice_pt = slice_op(start, stop, step)

    # Use simple constant array instead of arange
    x_pt = pt.constant(np.arange(10, dtype=np.float32))
    out_pt = x_pt[slice_pt]

    compare_mlx_and_py([start, stop, step], [out_pt], [1, 8, 2])


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


def test_mlx_subtensor_with_variables():
    """Test subtensor operations with PyTensor variables as inputs.
    
    This test now works thanks to the fix for np.int64 indexing, which also
    handles the conversion of MLX scalar arrays in slice components.
    """
    # Test with variable arrays (not constants)
    x_pt = pt.matrix("x", dtype="float32")
    y_pt = pt.vector("y", dtype="float32")

    x_np = np.arange(12, dtype=np.float32).reshape((3, 4))
    y_np = np.array([-1.0, -2.0], dtype=np.float32)

    # Set operation with variables
    out_pt = pt_subtensor.set_subtensor(x_pt[0, :2], y_pt)
    compare_mlx_and_py([x_pt, y_pt], [out_pt], [x_np, y_np])


def test_mlx_subtensor_with_numpy_int64():
    """Test Subtensor operations with np.int64 indices.
    
    This tests the fix for MLX's strict requirement that indices must be
    Python int, not np.int64 or other NumPy integer types.
    """
    # Test data
    x_np = np.arange(12, dtype=np.float32).reshape((3, 4))
    x_pt = pt.constant(x_np)
    
    # Single np.int64 index - this was failing before the fix
    idx = np.int64(1)
    out_pt = x_pt[idx]
    compare_mlx_and_py([], [out_pt], [])
    
    # Multiple np.int64 indices
    out_pt = x_pt[np.int64(1), np.int64(2)]
    compare_mlx_and_py([], [out_pt], [])
    
    # Negative np.int64 index
    out_pt = x_pt[np.int64(-1)]
    compare_mlx_and_py([], [out_pt], [])
    
    # Mixed Python int and np.int64
    out_pt = x_pt[1, np.int64(2)]
    compare_mlx_and_py([], [out_pt], [])


def test_mlx_subtensor_slices_with_numpy_int64():
    """Test Subtensor with slices containing np.int64 components.
    
    This tests that slice start/stop/step values can be np.int64.
    """
    x_np = np.arange(20, dtype=np.float32)
    x_pt = pt.constant(x_np)
    
    # Slice with np.int64 start
    out_pt = x_pt[np.int64(2):]
    compare_mlx_and_py([], [out_pt], [])
    
    # Slice with np.int64 stop
    out_pt = x_pt[:np.int64(5)]
    compare_mlx_and_py([], [out_pt], [])
    
    # Slice with np.int64 start and stop
    out_pt = x_pt[np.int64(2):np.int64(8)]
    compare_mlx_and_py([], [out_pt], [])
    
    # Slice with np.int64 step
    out_pt = x_pt[::np.int64(2)]
    compare_mlx_and_py([], [out_pt], [])
    
    # Slice with all np.int64 components
    out_pt = x_pt[np.int64(1):np.int64(10):np.int64(2)]
    compare_mlx_and_py([], [out_pt], [])
    
    # Negative np.int64 in slice
    out_pt = x_pt[np.int64(-5):]
    compare_mlx_and_py([], [out_pt], [])


def test_mlx_incsubtensor_with_numpy_int64():
    """Test IncSubtensor (set/inc) with np.int64 indices and slices.
    
    This is the main test for the reported issue with inc_subtensor.
    """
    # Test data
    x_np = np.arange(12, dtype=np.float32).reshape((3, 4))
    x_pt = pt.constant(x_np)
    y_pt = pt.as_tensor_variable(np.array(10.0, dtype=np.float32))
    
    # Set with np.int64 index
    out_pt = pt_subtensor.set_subtensor(x_pt[np.int64(1), np.int64(2)], y_pt)
    compare_mlx_and_py([], [out_pt], [])
    
    # Increment with np.int64 index
    out_pt = pt_subtensor.inc_subtensor(x_pt[np.int64(1), np.int64(2)], y_pt)
    compare_mlx_and_py([], [out_pt], [])
    
    # Set with slice containing np.int64 - THE ORIGINAL FAILING CASE
    out_pt = pt_subtensor.set_subtensor(x_pt[:, :np.int64(2)], y_pt)
    compare_mlx_and_py([], [out_pt], [])
    
    # Increment with slice containing np.int64 - THE ORIGINAL FAILING CASE
    out_pt = pt_subtensor.inc_subtensor(x_pt[:, :np.int64(2)], y_pt)
    compare_mlx_and_py([], [out_pt], [])
    
    # Complex slice with np.int64
    y2_pt = pt.as_tensor_variable(np.ones((2, 2), dtype=np.float32))
    out_pt = pt_subtensor.inc_subtensor(
        x_pt[np.int64(0):np.int64(2), np.int64(1):np.int64(3)], y2_pt
    )
    compare_mlx_and_py([], [out_pt], [])


def test_mlx_incsubtensor_original_issue():
    """Test the exact example from the issue report.
    
    This was failing with: ValueError: Slice indices must be integers or None.
    """
    x_np = np.arange(9, dtype=np.float64).reshape((3, 3))
    x_pt = pt.constant(x_np, dtype="float64")
    
    # The exact failing case from the issue
    out_pt = pt_subtensor.inc_subtensor(x_pt[:, :2], 10)
    compare_mlx_and_py([], [out_pt], [])
    
    # Verify it also works with set_subtensor
    out_pt = pt_subtensor.set_subtensor(x_pt[:, :2], 10)
    compare_mlx_and_py([], [out_pt], [])


def test_mlx_advanced_subtensor_with_numpy_int64():
    """Test AdvancedSubtensor with np.int64 in mixed indexing."""
    x_np = np.arange(24, dtype=np.float32).reshape((3, 4, 2))
    x_pt = pt.constant(x_np)
    
    # Advanced indexing with list, but other dimensions use np.int64
    # Note: This creates AdvancedSubtensor, not basic Subtensor
    out_pt = x_pt[[0, 2], np.int64(1)]
    compare_mlx_and_py([], [out_pt], [])
    
    # Mixed advanced and basic indexing with np.int64 in slice
    out_pt = x_pt[[0, 2], np.int64(1):np.int64(3)]
    compare_mlx_and_py([], [out_pt], [])


def test_mlx_advanced_incsubtensor_with_numpy_int64():
    """Test AdvancedIncSubtensor with np.int64."""
    x_np = np.arange(15, dtype=np.float32).reshape((5, 3))
    x_pt = pt.constant(x_np)
    
    # Value to set/increment
    y_pt = pt.as_tensor_variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
    
    # Advanced indexing set with array indices
    indices = [np.int64(0), np.int64(2)]
    out_pt = pt_subtensor.set_subtensor(x_pt[indices], y_pt)
    compare_mlx_and_py([], [out_pt], [])
    
    # Advanced indexing increment
    out_pt = pt_subtensor.inc_subtensor(x_pt[indices], y_pt)
    compare_mlx_and_py([], [out_pt], [])
