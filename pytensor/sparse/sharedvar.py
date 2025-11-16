import copy

import scipy.sparse

from pytensor.compile import shared_constructor
from pytensor.sparse.variable import SparseTensorType, SparseVariable
from pytensor.tensor.sharedvar import TensorSharedVariable


class SparseTensorSharedVariable(TensorSharedVariable, SparseVariable):  # type: ignore[misc]
    @property
    def format(self):
        return self.type.format


@shared_constructor.register(scipy.sparse.spmatrix)
def sparse_constructor(
    value, name=None, strict=False, allow_downcast=None, borrow=False, format=None
):
    if format is None:
        format = value.format

    type = SparseTensorType(format=format, dtype=value.dtype)

    if not borrow:
        value = copy.deepcopy(value)

    return SparseTensorSharedVariable(
        type=type, value=value, strict=strict, allow_downcast=allow_downcast, name=name
    )
