import typing
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import get_lapack_funcs

from pytensor.graph import Apply, Op
from pytensor.tensor.basic import as_tensor, diagonal
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.type import tensor, vector
from pytensor.tensor.variable import TensorVariable


if TYPE_CHECKING:
    from pytensor.tensor import TensorLike


class LUFactorTridiagonal(Op):
    """Compute LU factorization of a tridiagonal matrix (lapack gttrf)"""

    __props__ = (
        "overwrite_dl",
        "overwrite_d",
        "overwrite_du",
    )
    gufunc_signature = "(dl),(d),(dl)->(dl),(d),(dl),(du2),(d)"

    def __init__(self, overwrite_dl=False, overwrite_d=False, overwrite_du=False):
        self.destroy_map = dm = {}
        if overwrite_dl:
            dm[0] = [0]
        if overwrite_d:
            dm[1] = [1]
        if overwrite_du:
            dm[2] = [2]
        self.overwrite_dl = overwrite_dl
        self.overwrite_d = overwrite_d
        self.overwrite_du = overwrite_du
        super().__init__()

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        return type(self)(
            overwrite_dl=0 in allowed_inplace_inputs,
            overwrite_d=1 in allowed_inplace_inputs,
            overwrite_du=2 in allowed_inplace_inputs,
        )

    def make_node(self, dl, d, du):
        dl, d, du = map(as_tensor, (dl, d, du))

        if not all(inp.type.ndim == 1 for inp in (dl, d, du)):
            raise ValueError("Diagonals must be vectors")

        ndl, nd, ndu = (inp.type.shape[-1] for inp in (dl, d, du))

        match (ndl, nd, ndu):
            case (int(), _, _):
                n = ndl + 1
            case (_, int(), _):
                n = nd + 1
            case (_, _, int()):
                n = ndu + 1
            case _:
                n = None

        dummy_arrays = [np.zeros((), dtype=inp.type.dtype) for inp in (dl, d, du)]
        out_dtype = get_lapack_funcs("gttrf", dummy_arrays).dtype
        outputs = [
            vector(shape=(None if n is None else (n - 1),), dtype=out_dtype),
            vector(shape=(n,), dtype=out_dtype),
            vector(shape=(None if n is None else n - 1,), dtype=out_dtype),
            vector(shape=(None if n is None else n - 2,), dtype=out_dtype),
            vector(shape=(n,), dtype=np.int32),
        ]
        return Apply(self, [dl, d, du], outputs)

    def perform(self, node, inputs, output_storage):
        gttrf = get_lapack_funcs("gttrf", dtype=node.outputs[0].type.dtype)
        dl, d, du, du2, ipiv, _ = gttrf(
            *inputs,
            overwrite_dl=self.overwrite_dl,
            overwrite_d=self.overwrite_d,
            overwrite_du=self.overwrite_du,
        )
        output_storage[0][0] = dl
        output_storage[1][0] = d
        output_storage[2][0] = du
        output_storage[3][0] = du2
        output_storage[4][0] = ipiv


class SolveLUFactorTridiagonal(Op):
    """Solve a system of linear equations with a tridiagonal coefficient matrix (lapack gttrs)."""

    __props__ = ("b_ndim", "overwrite_b", "transposed")

    def __init__(self, b_ndim: int, transposed: bool, overwrite_b=False):
        if b_ndim not in (1, 2):
            raise ValueError("b_ndim must be 1 or 2")
        if b_ndim == 1:
            self.gufunc_signature = "(dl),(d),(dl),(du2),(d),(d)->(d)"
        else:
            self.gufunc_signature = "(dl),(d),(dl),(du2),(d),(d,rhs)->(d,rhs)"
        if overwrite_b:
            self.destroy_map = {0: [5]}
        self.b_ndim = b_ndim
        self.transposed = transposed
        self.overwrite_b = overwrite_b
        super().__init__()

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        # b matrix is the 5th input
        if 5 in allowed_inplace_inputs:
            props = self._props_dict()  # type: ignore
            props["overwrite_b"] = True
            return type(self)(**props)

        return self

    def make_node(self, dl, d, du, du2, ipiv, b):
        dl, d, du, du2, ipiv, b = map(as_tensor, (dl, d, du, du2, ipiv, b))

        if b.type.ndim != self.b_ndim:
            raise ValueError("Wrong number of dimensions for input b.")

        if not all(inp.type.ndim == 1 for inp in (dl, d, du, du2, ipiv)):
            raise ValueError("Inputs must be vectors")

        ndl, nd, ndu, ndu2, nipiv = (
            inp.type.shape[-1] for inp in (dl, d, du, du2, ipiv)
        )
        nb = b.type.shape[0]

        match (ndl, nd, ndu, ndu2, nipiv):
            case (int(), _, _, _, _):
                n = ndl + 1
            case (_, int(), _, _, _):
                n = nd
            case (_, _, int(), _, _):
                n = ndu + 1
            case (_, _, _, int(), _):
                n = ndu2 + 2
            case (_, _, _, _, int()):
                n = nipiv
            case _:
                n = nb

        dummy_arrays = [
            np.zeros((), dtype=inp.type.dtype) for inp in (dl, d, du, du2, b)
        ]
        out_dtype = get_lapack_funcs("gttrs", dummy_arrays).dtype
        if self.b_ndim == 1:
            output_shape = (n,)
        else:
            output_shape = (n, b.type.shape[-1])

        outputs = [tensor(shape=output_shape, dtype=out_dtype)]
        return Apply(self, [dl, d, du, du2, ipiv, b], outputs)

    def perform(self, node, inputs, output_storage):
        gttrs = get_lapack_funcs("gttrs", dtype=node.outputs[0].type.dtype)
        x, _ = gttrs(
            *inputs,
            overwrite_b=self.overwrite_b,
            trans="N" if not self.transposed else "T",
        )
        output_storage[0][0] = x


def tridiagonal_lu_factor(
    a: "TensorLike",
) -> tuple[
    TensorVariable, TensorVariable, TensorVariable, TensorVariable, TensorVariable
]:
    """Return the decomposition of A implied by a solve tridiagonal (LAPACK's gttrf)

    Parameters
    ----------
    a
        The input matrix.

    Returns
    -------
    dl, d, du, du2, ipiv
        The LU factorization of A.
    """
    dl, d, du = (diagonal(a, offset=o, axis1=-2, axis2=-1) for o in (-1, 0, 1))
    dl, d, du, du2, ipiv = typing.cast(
        list[TensorVariable], Blockwise(LUFactorTridiagonal())(dl, d, du)
    )
    return dl, d, du, du2, ipiv


def tridiagonal_lu_solve(
    a_diagonals: tuple[
        "TensorLike", "TensorLike", "TensorLike", "TensorLike", "TensorLike"
    ],
    b: "TensorLike",
    *,
    b_ndim: int,
    transposed: bool = False,
) -> TensorVariable:
    """Solve a tridiagonal system of equations using LU factorized inputs (LAPACK's gttrs).

    Parameters
    ----------
    a_diagonals
        The outputs of tridiagonal_lu_factor(A).
    b
        The right-hand side vector or matrix.
    b_ndim
        The number of dimensions of the right-hand side.
    transposed
        Whether to solve the transposed system.

    Returns
    -------
    TensorVariable
        The solution vector or matrix.
    """
    dl, d, du, du2, ipiv = a_diagonals
    return typing.cast(
        TensorVariable,
        Blockwise(SolveLUFactorTridiagonal(b_ndim=b_ndim, transposed=transposed))(
            dl, d, du, du2, ipiv, b
        ),
    )
