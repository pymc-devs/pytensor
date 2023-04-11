import jax.experimental.sparse as jsp
from scipy.sparse import spmatrix

from pytensor.graph.basic import Constant
from pytensor.link.jax.dispatch import jax_funcify, jax_typify
from pytensor.sparse.basic import Dot, StructuredDot
from pytensor.sparse.type import SparseTensorType


@jax_typify.register(spmatrix)
def jax_typify_spmatrix(matrix, dtype=None, **kwargs):
    # Note: This changes the type of the constants from CSR/CSC to BCOO
    # We could add BCOO as a PyTensor type but this would only be useful for JAX graphs
    # and it would break the premise of one graph -> multiple backends.
    # The same situation happens with RandomGenerators...
    return jsp.BCOO.from_scipy_sparse(matrix)


@jax_funcify.register(Dot)
@jax_funcify.register(StructuredDot)
def jax_funcify_sparse_dot(op, node, **kwargs):
    for input in node.inputs:
        if isinstance(input.type, SparseTensorType) and not isinstance(input, Constant):
            raise NotImplementedError(
                "JAX sparse dot only implemented for constant sparse inputs"
            )

    if isinstance(node.outputs[0].type, SparseTensorType):
        raise NotImplementedError("JAX sparse dot only implemented for dense outputs")

    @jsp.sparsify
    def sparse_dot(x, y):
        out = x @ y
        if isinstance(out, jsp.BCOO):
            out = out.todense()
        return out

    return sparse_dot
