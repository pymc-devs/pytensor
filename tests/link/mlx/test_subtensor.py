import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function
from pytensor.compile.mode import Mode
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
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)
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
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
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


@pytest.mark.parametrize(
    "func", (pt_subtensor.advanced_inc_subtensor1, pt_subtensor.advanced_set_subtensor1)
)
def test_mlx_AdvancedIncSubtensor1_runtime_broadcast(func):
    """MLX must reject runtime broadcasting of ``y``, matching C/Numba/JAX/PyTorch.

    ``AdvancedIncSubtensor1`` requires ``y`` to already match the indexed shape;
    a statically non-broadcastable dimension that is length 1 at runtime is an
    error, not a silent broadcast.
    """
    y = pt.matrix("y", dtype="float32", shape=(None, None))
    x = pt.zeros((10, 5))
    idxs = np.repeat(np.arange(10), 2)  # 20 indices
    out = func(x, y, idxs)
    assert isinstance(out.owner.op, pt_subtensor.AdvancedIncSubtensor)

    f = function([y], out, mode=Mode(linker="mlx", optimizer=None))
    f(np.ones((20, 5), dtype=np.float32))  # correctly sized y works

    with pytest.raises(ValueError, match="Runtime broadcasting not allowed"):
        f(np.ones((1, 5), dtype=np.float32))  # broadcast along index
    with pytest.raises(ValueError, match="Runtime broadcasting not allowed"):
        f(np.ones((20, 1), dtype=np.float32))  # broadcast along buffer


def test_mlx_IncSubtensor_slice_grad():
    """Gradient of a basic slice lowers to an ``IncSubtensor`` with slice bounds
    passed as (array) inputs; these must be coerced to Python ints for MLX."""
    x_pt = pt.vector("x", dtype="float32")
    x_np = np.arange(6, dtype=np.float32)

    # Contiguous and strided (RoPE-style) slices both exercise the slice path.
    for sl in (x_pt[0:3], x_pt[0::2]):
        g = pt.grad((sl**2).sum(), x_pt)
        assert isinstance(g.owner.op, pt_subtensor.IncSubtensor)
        compare_mlx_and_py([x_pt], [g], [x_np])


def test_mlx_IncSubtensor_negative_step_slice_grad():
    # The wrong result here (previously attributed to ml-explore/mlx#3716) was
    # actually the negative-stride read feeding the elementwise gradient term,
    # now materialized by the Subtensor dispatch.
    x_pt = pt.vector("x", dtype="float32")
    x_np = np.arange(6, dtype=np.float32)
    g = pt.grad((x_pt[::-1] ** 2).sum(), x_pt)
    assert isinstance(g.owner.op, pt_subtensor.IncSubtensor)
    compare_mlx_and_py([x_pt], [g], [x_np], mlx_mode="MLX")


@pytest.mark.parametrize(
    "func",
    (pt_subtensor.advanced_inc_subtensor1, pt_subtensor.advanced_set_subtensor1),
    ids=("inc", "set"),
)
def test_mlx_AdvancedIncSubtensor1_duplicate_indices(func):
    """Duplicate indices must accumulate for inc (``np.add.at`` semantics).

    Gradients of advanced indexing (e.g. embedding lookups with repeated token
    ids) produce inc with duplicate indices; MLX must sum all contributions
    rather than writing each destination once.
    """
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    idxs = np.array([0, 0, 0, 1], dtype=np.int64)
    out = func(x, y, idxs)
    assert isinstance(out.owner.op, pt_subtensor.AdvancedIncSubtensor)

    x_np = np.zeros(3, dtype=np.float32)
    y_np = np.ones(4, dtype=np.float32)
    compare_mlx_and_py([x, y], [out], [x_np, y_np])


def test_mlx_AdvancedIncSubtensor1_duplicate_indices_edge_cases():
    """Duplicate accumulation with negative indices and a scalar (broadcast) ``y``."""
    x = pt.vector("x", dtype="int32")
    y = pt.scalar("y", dtype="int32")
    idxs = np.array([-1, -1, 0, -1], dtype=np.int64)
    out = pt_subtensor.advanced_inc_subtensor1(x, y, idxs)
    assert isinstance(out.owner.op, pt_subtensor.AdvancedIncSubtensor)

    compare_mlx_and_py([x, y], [out], [np.zeros(3, dtype=np.int32), np.int32(2)])


def test_mlx_AdvancedIncSubtensor_duplicate_indices():
    """``AdvancedIncSubtensor`` with duplicate indices accumulates like ``np.add.at``."""
    x = pt.matrix("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    rows = np.array([0, 0, 1], dtype=np.int64)
    cols = np.array([1, 1, 2], dtype=np.int64)
    out = pt_subtensor.inc_subtensor(x[rows, cols], y)
    assert isinstance(out.owner.op, pt_subtensor.AdvancedIncSubtensor)
    assert not out.owner.op.set_instead_of_inc
    assert not out.owner.op.ignore_duplicates

    x_np = np.zeros((3, 3), dtype=np.float32)
    y_np = np.ones(3, dtype=np.float32)
    compare_mlx_and_py([x, y], [out], [x_np, y_np])


def test_mlx_AdvancedIncSubtensor_ignore_duplicates():
    """``ignore_duplicates=True`` requests write-once (numpy ``x[idx] += y``).

    Duplicate indices must NOT be accumulated in this mode, matching the
    reference ``perform`` and the PyTorch/Numba backends.
    """
    x = pt.vector("x", dtype="float32")
    out = pt_subtensor.inc_subtensor(
        x[[0, 1, 0]], np.float32(5.0), ignore_duplicates=True
    )
    assert isinstance(out.owner.op, pt_subtensor.AdvancedIncSubtensor)
    assert out.owner.op.ignore_duplicates

    compare_mlx_and_py([x], [out], [np.zeros(3, dtype=np.float32)])


@pytest.mark.parametrize("axis", [0, 1])
def test_mlx_negative_step_slice_elemwise(axis):
    """A negative-stride slice feeding an elementwise op must materialize.

    Under the full ``mode="MLX"`` (``mx.compile``), an elementwise op fed by a
    negative-stride view used to be miscompiled (trailing entries zeroed). The
    Subtensor dispatch now copies reversed slices into a contiguous array. This
    is what unblocks Scan gradients over sequences, which reverse the trace.
    """
    x = pt.matrix("x", dtype="float32")
    rev = x[::-1] if axis == 0 else x[:, ::-1]
    out = 2.0 * rev
    assert isinstance(rev.owner.op, pt_subtensor.Subtensor)
    x_np = np.arange(15, dtype=np.float32).reshape(5, 3)
    compare_mlx_and_py([x], [out], [x_np], mlx_mode="MLX")
