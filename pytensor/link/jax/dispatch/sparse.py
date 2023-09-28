import jax.experimental.sparse as jsp
from scipy.sparse import spmatrix

from pytensor.graph.type import HasDataType
from pytensor.link.jax.dispatch import jax_funcify, jax_typify
from pytensor.sparse.basic import Dot, StructuredDot, Transpose
from pytensor.sparse.type import SparseTensorType
from pytensor.tensor import TensorType


@jax_typify.register(spmatrix)
def jax_typify_spmatrix(matrix, dtype=None, **kwargs):
    return jsp.BCOO.from_scipy_sparse(matrix)


class BCOOType(TensorType, HasDataType):
    """JAX-compatible BCOO type.

    This type is not exposed to users directly.

    It is introduced by the JIT linker in place of any SparseTensorType input
    variables used in the original function. Nodes in the function graph will
    still show the original types as inputs and outputs.
    """

    def filter(self, data, strict: bool = False, allow_downcast=None):
        if isinstance(data, jsp.BCOO):
            return data

        if strict:
            raise TypeError()

        return jax_typify(data)


@jax_typify.register(SparseTensorType)
def jax_typify_SparseTensorType(type):
    return BCOOType(
        dtype=type.dtype,
        shape=type.shape,
        name=type.name,
        broadcastable=type.broadcastable,
    )


@jax_funcify.register(Dot)
@jax_funcify.register(StructuredDot)
def jax_funcify_sparse_dot(op, node, **kwargs):
    @jsp.sparsify
    def sparse_dot(x, y):
        out = x @ y
        if isinstance(out, jsp.BCOO) and not isinstance(
            node.outputs[0].type, SparseTensorType
        ):
            out = out.todense()
        return out

    return sparse_dot


@jax_funcify.register(Transpose)
def jax_funcify_sparse_transpose(op, **kwargs):
    def sparse_transpose(x):
        return x.T

    return sparse_transpose
