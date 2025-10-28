"""
Pure MLX indexing behavior tests.

This module tests MLX's indexing capabilities with different index types
to understand what conversions are needed for PyTensor compatibility.
"""

import numpy as np
import pytest


mx = pytest.importorskip("mlx.core")


def test_mlx_python_int_indexing():
    """Test that MLX accepts Python int for indexing."""
    x = mx.array([1, 2, 3, 4, 5])

    # Single index
    result = x[2]
    assert result == 3

    # Slice with Python int
    result = x[1:4]
    assert list(result) == [2, 3, 4]

    # Slice with step :)
    result = x[0:5:2]
    assert list(result) == [1, 3, 5]


def test_mlx_numpy_int64_single_index():
    """Test MLX behavior with np.int64 single index."""
    x = mx.array([1, 2, 3, 4, 5])

    # This should fail with MLX
    with pytest.raises(ValueError, match="Cannot index mlx array"):
        _ = x[np.int64(2)]


def test_mlx_numpy_int64_in_slice():
    """Test MLX behavior with np.int64 in slice components."""
    x = mx.array([1, 2, 3, 4, 5])

    # Slice with np.int64 start
    with pytest.raises(ValueError, match="Slice indices must be integers or None"):
        _ = x[np.int64(1) : 4]

    # Slice with np.int64 stop
    with pytest.raises(ValueError, match="Slice indices must be integers or None"):
        _ = x[1 : np.int64(4)]

    # Slice with np.int64 step
    with pytest.raises(ValueError, match="Slice indices must be integers or None"):
        _ = x[0 : 5 : np.int64(2)]


def test_mlx_conversion_int64_to_python_int():
    """Test that converting np.int64 to Python int works for MLX indexing."""
    x = mx.array([1, 2, 3, 4, 5])

    # Convert np.int64 to Python int
    idx = int(np.int64(2))
    result = x[idx]
    assert result == 3

    # Convert in slice
    start = int(np.int64(1))
    stop = int(np.int64(4))
    step = int(np.int64(2))
    result = x[start:stop:step]
    assert list(result) == [2, 4]


def test_mlx_slice_with_none():
    """Test that MLX accepts None in slice components."""
    x = mx.array([1, 2, 3, 4, 5])

    # Slice with None
    result = x[None:3]
    assert list(result) == [1, 2, 3]

    result = x[2:None]
    assert list(result) == [3, 4, 5]

    result = x[None:None:2]
    assert list(result) == [1, 3, 5]


def test_mlx_multidimensional_indexing():
    """Test MLX indexing with multidimensional arrays."""
    x = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Python int indexing works
    result = x[1, 2]
    assert result == 6

    # Mixed slice and int
    result = x[1, :]
    assert list(result) == [4, 5, 6]

    result = x[:, 1]
    assert list(result) == [2, 5, 8]

    # Multiple slices
    result = x[0:2, 1:3]
    expected = [[2, 3], [5, 6]]
    assert result.tolist() == expected


def test_mlx_negative_indices():
    """Test MLX with negative indices (both Python int and np.int64)."""
    x = mx.array([1, 2, 3, 4, 5])

    # Negative Python int works
    result = x[-1]
    assert result == 5

    result = x[-3:-1]
    assert list(result) == [3, 4]

    # Negative np.int64 should fail
    with pytest.raises(ValueError, match="Cannot index mlx array"):
        _ = x[np.int64(-1)]

    # But converting to Python int works
    idx = int(np.int64(-1))
    result = x[idx]
    assert result == 5


def test_mlx_array_indexing():
    """Test MLX with array indices (advanced indexing)."""
    x = mx.array([1, 2, 3, 4, 5])

    # Array indexing with MLX array works
    indices = mx.array([0, 2, 4])
    result = x[indices]
    assert list(result) == [1, 3, 5]

    # Array indexing with NumPy array should fail
    indices = np.array([0, 2, 4])
    with pytest.raises(ValueError, match="Cannot index mlx array"):
        _ = x[indices]

    # But converting NumPy array to MLX array works
    indices_mlx = mx.array(indices)
    result = x[indices_mlx]
    assert list(result) == [1, 3, 5]


def test_conversion_helper_behavior():
    """Test the behavior of our proposed int conversion helper."""

    def get_slice_int(element):
        """Helper to convert slice components to Python int."""
        if element is None:
            return None
        try:
            return int(element)
        except Exception:
            return element

    # Test with None
    assert get_slice_int(None) is None

    # Test with Python int
    assert get_slice_int(5) == 5
    assert isinstance(get_slice_int(5), int)

    # Test with np.int64
    assert get_slice_int(np.int64(5)) == 5
    assert isinstance(get_slice_int(np.int64(5)), int)

    # Test with np.int32
    assert get_slice_int(np.int32(5)) == 5
    assert isinstance(get_slice_int(np.int32(5)), int)

    # Test with array (should pass through)
    arr = np.array([1, 2, 3])
    result = get_slice_int(arr)
    assert np.array_equal(result, arr)


def test_mlx_indexing_with_converted_slices():
    """Test that MLX indexing works after converting slice components."""
    x = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def normalize_slice(s):
        """Convert slice components to Python int."""
        if not isinstance(s, slice):
            # For non-slice indices, try to convert to int
            try:
                return int(s)
            except (TypeError, ValueError):
                return s

        return slice(
            int(s.start) if s.start is not None else None,
            int(s.stop) if s.stop is not None else None,
            int(s.step) if s.step is not None else None,
        )

    # Create slices with np.int64
    slice1 = slice(np.int64(0), np.int64(2), None)
    slice2 = slice(None, np.int64(2), None)

    # Convert and use
    normalized = (normalize_slice(slice1), normalize_slice(slice2))
    result = x[normalized]
    expected = [[1, 2], [4, 5]]
    assert result.tolist() == expected
