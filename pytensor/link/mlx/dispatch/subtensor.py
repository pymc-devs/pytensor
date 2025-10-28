from copy import deepcopy

import numpy as np

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    indices_from_subtensor,
)
from pytensor.tensor.type_other import MakeSlice


def normalize_indices_for_mlx(ilist, idx_list):
    """Convert indices to MLX-compatible format.
    
    MLX has strict requirements for indexing:
    - Integer indices must be Python int, not np.int64 or other NumPy integer types
    - Slice components (start, stop, step) must be Python int or None, not np.int64
    - MLX arrays created from scalars need to be converted back to Python int
    - Array indices for advanced indexing are handled separately
    
    This function converts all integer-like indices and slice components to Python int
    while preserving None values and passing through array indices unchanged.
    
    Parameters
    ----------
    ilist : tuple
        Runtime index values to be passed to indices_from_subtensor
    idx_list : tuple
        Static index specification from the Op's idx_list attribute
        
    Returns
    -------
    tuple
        Normalized indices compatible with MLX array indexing
        
    Examples
    --------
    >>> # Single np.int64 index converted to Python int
    >>> normalize_indices_for_mlx((np.int64(1),), (True,))
    (1,)
    
    >>> # Slice with np.int64 components
    >>> indices = indices_from_subtensor((np.int64(0), np.int64(2)), (slice(None, None),))
    >>> # After normalization, slice components are Python int
    
    Notes
    -----
    This conversion is necessary because MLX's C++ indexing implementation
    does not recognize NumPy scalar types, raising ValueError when encountered.
    Additionally, mlx_typify converts NumPy scalars to MLX arrays, which also
    need to be converted back to Python int for use in indexing operations.
    Converting to Python int is zero-cost for Python int inputs and minimal
    overhead for NumPy scalars and MLX scalar arrays.
    """
    import mlx.core as mx
    
    def normalize_element(element):
        """Convert a single index element to MLX-compatible format."""
        if element is None:
            # None is valid in slices (e.g., x[None:5] or x[:None])
            return None
        elif isinstance(element, slice):
            # Recursively normalize slice components
            return slice(
                normalize_element(element.start),
                normalize_element(element.stop),
                normalize_element(element.step),
            )
        elif isinstance(element, mx.array):
            # MLX arrays from mlx_typify need special handling
            # If it's a 0-d array (scalar), convert to Python int/float
            if element.ndim == 0:
                # Extract the scalar value
                item = element.item()
                # Convert to Python int if it's an integer type
                if element.dtype in (mx.int8, mx.int16, mx.int32, mx.int64,
                                     mx.uint8, mx.uint16, mx.uint32, mx.uint64):
                    return int(item)
                else:
                    return float(item)
            else:
                # Multi-dimensional array for advanced indexing - pass through
                return element
        elif isinstance(element, (np.integer, np.floating)):
            # Convert NumPy scalar to Python int/float
            # This handles np.int64, np.int32, np.float64, etc.
            return int(element) if isinstance(element, np.integer) else float(element)
        elif isinstance(element, (int, float)):
            # Python int/float are already compatible
            return element
        else:
            # Pass through other types (arrays for advanced indexing, etc.)
            return element
    
    # Get indices from PyTensor's subtensor utility
    raw_indices = indices_from_subtensor(ilist, idx_list)
    
    # Normalize each index element
    normalized = tuple(normalize_element(idx) for idx in raw_indices)
    
    return normalized


@mlx_funcify.register(Subtensor)
def mlx_funcify_Subtensor(op, node, **kwargs):
    """MLX implementation of Subtensor operation.
    
    Uses normalize_indices_for_mlx to ensure all indices are compatible with MLX.
    """
    idx_list = getattr(op, "idx_list", None)

    def subtensor(x, *ilists):
        # Normalize indices to handle np.int64 and other NumPy types
        indices = normalize_indices_for_mlx(ilists, idx_list)
        if len(indices) == 1:
            indices = indices[0]

        return x.__getitem__(indices)

    return subtensor


@mlx_funcify.register(AdvancedSubtensor)
@mlx_funcify.register(AdvancedSubtensor1)
def mlx_funcify_AdvancedSubtensor(op, node, **kwargs):
    """MLX implementation of AdvancedSubtensor operation.
    
    Uses normalize_indices_for_mlx to ensure all indices are compatible with MLX,
    including handling np.int64 in mixed basic/advanced indexing scenarios.
    """
    idx_list = getattr(op, "idx_list", None)

    def advanced_subtensor(x, *ilists):
        # Normalize indices to handle np.int64 and other NumPy types
        indices = normalize_indices_for_mlx(ilists, idx_list)
        if len(indices) == 1:
            indices = indices[0]

        return x.__getitem__(indices)

    return advanced_subtensor


@mlx_funcify.register(IncSubtensor)
@mlx_funcify.register(AdvancedIncSubtensor1)
def mlx_funcify_IncSubtensor(op, node, **kwargs):
    """MLX implementation of IncSubtensor operation.
    
    Uses normalize_indices_for_mlx to ensure all indices are compatible with MLX.
    Handles both set_instead_of_inc=True (assignment) and False (increment).
    """
    idx_list = getattr(op, "idx_list", None)

    if getattr(op, "set_instead_of_inc", False):

        def mlx_fn(x, indices, y):
            if not op.inplace:
                x = deepcopy(x)
            x[indices] = y
            return x

    else:

        def mlx_fn(x, indices, y):
            if not op.inplace:
                x = deepcopy(x)
            x[indices] += y
            return x

    def incsubtensor(x, y, *ilist, mlx_fn=mlx_fn, idx_list=idx_list):
        # Normalize indices to handle np.int64 and other NumPy types
        indices = normalize_indices_for_mlx(ilist, idx_list)

        if len(indices) == 1:
            indices = indices[0]

        return mlx_fn(x, indices, y)

    return incsubtensor


@mlx_funcify.register(AdvancedIncSubtensor)
def mlx_funcify_AdvancedIncSubtensor(op, node, **kwargs):
    """MLX implementation of AdvancedIncSubtensor operation.
    
    Uses normalize_indices_for_mlx to ensure all indices are compatible with MLX.
    Note: For advanced indexing, ilist contains the actual array indices.
    """
    idx_list = getattr(op, "idx_list", None)
    
    if getattr(op, "set_instead_of_inc", False):

        def mlx_fn(x, indices, y):
            if not op.inplace:
                x = deepcopy(x)
            x[indices] = y
            return x

    else:

        def mlx_fn(x, indices, y):
            if not op.inplace:
                x = deepcopy(x)
            x[indices] += y
            return x

    def advancedincsubtensor(x, y, *ilist, mlx_fn=mlx_fn, idx_list=idx_list):
        # Normalize indices to handle np.int64 and other NumPy types
        indices = normalize_indices_for_mlx(ilist, idx_list)
        
        # For advanced indexing, if we have a single tuple of indices, unwrap it
        if len(indices) == 1:
            indices = indices[0]
        
        return mlx_fn(x, indices, y)

    return advancedincsubtensor


@mlx_funcify.register(MakeSlice)
def mlx_funcify_MakeSlice(op, **kwargs):
    def makeslice(*x):
        return slice(*x)

    return makeslice
