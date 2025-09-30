import numpy as np


# function that replicates np.unique from numpy < 2.0
def old_np_unique(
    arr, return_index=False, return_inverse=False, return_counts=False, axis=None
):
    """Replicate np.unique from numpy versions < 2.0"""
    if not return_inverse:
        return np.unique(arr, return_index, return_inverse, return_counts, axis)

    outs = list(np.unique(arr, return_index, return_inverse, return_counts, axis))

    inv_idx = 2 if return_index else 1

    if axis is None:
        outs[inv_idx] = np.ravel(outs[inv_idx])
    else:
        inv_shape = (arr.shape[axis],)
        outs[inv_idx] = outs[inv_idx].reshape(inv_shape)

    return tuple(outs)
