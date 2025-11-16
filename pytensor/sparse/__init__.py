from pytensor.sparse import rewriting, sharedvar
from pytensor.sparse.basic import *
from pytensor.sparse.math import *
from pytensor.sparse.sharedvar import sparse_constructor as shared
from pytensor.sparse.type import SparseTensorType, _is_sparse


def sparse_grad(var):
    """This function return a new variable whose gradient will be
    stored in a sparse format instead of dense.

    Currently only variable created by AdvancedSubtensor1 is supported.
    i.e. a_tensor_var[an_int_vector].

    .. versionadded:: 0.6rc4
    """
    from pytensor.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1

    if not (
        var.owner and isinstance(var.owner.op, AdvancedSubtensor | AdvancedSubtensor1)
    ):
        raise TypeError(
            "Sparse gradient is only implemented for AdvancedSubtensor and AdvancedSubtensor1"
        )

    x = var.owner.inputs[0]
    indices = var.owner.inputs[1:]

    if len(indices) > 1:
        raise TypeError(
            "Sparse gradient is only implemented for single advanced indexing"
        )

    ret = AdvancedSubtensor1(sparse_grad=True)(x, indices[0])
    return ret
