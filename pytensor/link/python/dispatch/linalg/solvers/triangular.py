import numpy as np
from scipy.linalg import get_lapack_funcs

from pytensor.link.python.dispatch.basic import python_funcify
from pytensor.tensor.linalg.solvers.triangular import SolveTriangular


@python_funcify.register(SolveTriangular)
def python_funcify_SolveTriangular(op, node=None, **kwargs):
    lower = op.lower
    unit_diagonal = op.unit_diagonal
    overwrite_b = op.overwrite_b
    (trtrs,) = get_lapack_funcs(("trtrs",), dtype=node.outputs[0].type.dtype)

    def solve_triangular(A, b):
        if b.size == 0:
            return np.empty_like(b)

        if A.flags["F_CONTIGUOUS"]:
            x, info = trtrs(
                A,
                b,
                overwrite_b=overwrite_b,
                lower=lower,
                trans=0,
                unitdiag=unit_diagonal,
            )
        else:
            # trtrs expects Fortran ordering, so solve the transposed system.
            x, info = trtrs(
                A.T,
                b,
                overwrite_b=overwrite_b,
                lower=not lower,
                trans=1,
                unitdiag=unit_diagonal,
            )

        if info != 0:
            x[...] = np.nan
        return x

    return solve_triangular
