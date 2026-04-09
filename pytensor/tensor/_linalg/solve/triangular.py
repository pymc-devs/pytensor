from typing import cast

import numpy as np
from scipy.linalg import get_lapack_funcs

from pytensor.graph.op import Op
from pytensor.tensor import basic as ptb
from pytensor.tensor._linalg.solve.core import SolveBase, _default_b_ndim
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.variable import TensorVariable


class SolveTriangular(SolveBase):
    """Solve a system of linear equations."""

    __props__ = (
        "unit_diagonal",
        "lower",
        "b_ndim",
        "overwrite_b",
    )

    def __init__(self, *, unit_diagonal=False, **kwargs):
        if kwargs.get("overwrite_a", False):
            raise ValueError("overwrite_a is not supported for SolverTriangulare")

        # There's a naming inconsistency between solve_triangular (trans) and solve (transposed). Internally, we can use
        # transpose everywhere, but expose the same API as scipy.linalg.solve_triangular
        super().__init__(**kwargs)
        self.unit_diagonal = unit_diagonal

    def perform(self, node, inputs, outputs):
        A, b = inputs

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("expected square matrix")

        if A.shape[0] != b.shape[0]:
            raise ValueError(f"shapes of a {A.shape} and b {b.shape} are incompatible")

        (trtrs,) = get_lapack_funcs(("trtrs",), (A, b))

        # Quick return for empty arrays
        if b.size == 0:
            outputs[0][0] = np.empty_like(b, dtype=trtrs.dtype)
            return

        if A.flags["F_CONTIGUOUS"]:
            x, info = trtrs(
                A,
                b,
                overwrite_b=self.overwrite_b,
                lower=self.lower,
                trans=0,
                unitdiag=self.unit_diagonal,
            )
        else:
            # transposed system is solved since trtrs expects Fortran ordering
            x, info = trtrs(
                A.T,
                b,
                overwrite_b=self.overwrite_b,
                lower=not self.lower,
                trans=1,
                unitdiag=self.unit_diagonal,
            )

        if info != 0:
            x[...] = np.nan

        outputs[0][0] = x

    def pullback(self, inputs, outputs, output_gradients):
        res = super().pullback(inputs, outputs, output_gradients)

        if self.lower:
            res[0] = ptb.tril(res[0])
        else:
            res[0] = ptb.triu(res[0])

        return res

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        if 1 in allowed_inplace_inputs:
            new_props = self._props_dict()  # type: ignore
            new_props["overwrite_b"] = True
            return type(self)(**new_props)
        else:
            return self


def solve_triangular(
    a: TensorVariable,
    b: TensorVariable,
    *,
    trans: int | str = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    check_finite: bool = True,
    b_ndim: int | None = None,
) -> TensorVariable:
    """Solve the equation `a x = b` for `x`, assuming `a` is a triangular matrix.

    Parameters
    ----------
    a: TensorVariable
        Square input data
    b: TensorVariable
        Input data for the right hand side.
    lower : bool, optional
        Use only data contained in the lower triangle of `a`. Default is to use upper triangle.
    trans: {0, 1, 2, 'N', 'T', 'C'}, optional
        Type of system to solve:
        trans       system
        0 or 'N'    a x = b
        1 or 'T'    a^T x = b
        2 or 'C'    a^H x = b
    unit_diagonal: bool, optional
        If True, diagonal elements of `a` are assumed to be 1 and will not be referenced.
    check_finite : bool, optional
        Unused by PyTensor. PyTensor will return nan if the operation fails.
    b_ndim : int
        Whether the core case of b is a vector (1) or matrix (2).
        This will influence how batched dimensions are interpreted.
    """
    b_ndim = _default_b_ndim(b, b_ndim)

    if trans in [1, "T", True]:
        a = a.mT
        lower = not lower
    if trans in [2, "C"]:
        a = a.conj().mT
        lower = not lower

    ret = Blockwise(
        SolveTriangular(
            lower=lower,
            unit_diagonal=unit_diagonal,
            b_ndim=b_ndim,
        )
    )(a, b)
    return cast(TensorVariable, ret)
