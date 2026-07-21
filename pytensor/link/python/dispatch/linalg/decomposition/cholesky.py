import numpy as np
from scipy.linalg import get_lapack_funcs

from pytensor.link.python.dispatch.basic import python_funcify
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky


@python_funcify.register(Cholesky)
def python_funcify_Cholesky(op, node=None, **kwargs):
    lower = op.lower
    overwrite_a = op.overwrite_a
    (potrf,) = get_lapack_funcs(("potrf",), dtype=node.inputs[0].type.dtype)

    def cholesky(x):
        if x.size == 0:
            return np.empty_like(x)

        # potrf only honors overwrite_a for F-contiguous input; transpose a
        # C-contiguous array to benefit from it.
        c_contiguous_input = overwrite_a and x.flags["C_CONTIGUOUS"]
        if c_contiguous_input:
            x = x.T
            factor, info = potrf(x, lower=not lower, overwrite_a=True, clean=True)
            factor = factor.T
        else:
            factor, info = potrf(x, lower=lower, overwrite_a=overwrite_a, clean=True)

        if info != 0:
            factor[...] = np.nan
        return factor

    return cholesky
