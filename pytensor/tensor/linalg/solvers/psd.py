import numpy as np
import scipy.linalg as scipy_linalg
from scipy.linalg import get_lapack_funcs

from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor import TensorLike
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg.solvers.core import SolveBase, _default_b_ndim
from pytensor.tensor.type import tensor


class CholeskySolve(SolveBase):
    __props__ = (
        "lower",
        "b_ndim",
        "overwrite_b",
    )

    def __init__(self, **kwargs):
        if kwargs.get("overwrite_a", False):
            raise ValueError("overwrite_a is not supported for CholeskySolve")
        super().__init__(**kwargs)

    def make_node(self, *inputs):
        # Allow base class to do input validation
        super_apply = super().make_node(*inputs)
        A, b = super_apply.inputs
        [super_out] = super_apply.outputs
        # The dtype of chol_solve does not match solve, which the base class checks
        dtype = scipy_linalg.cho_solve(
            (np.ones((1, 1), dtype=A.dtype), False),
            np.ones((1,), dtype=b.dtype),
        ).dtype
        out = tensor(dtype=dtype, shape=super_out.type.shape)
        return Apply(self, [A, b], [out])

    def perform(self, node, inputs, output_storage):
        c, b = inputs

        (potrs,) = get_lapack_funcs(("potrs",), (c, b))

        if c.shape[0] != c.shape[1]:
            raise ValueError("The factored matrix c is not square.")
        if c.shape[1] != b.shape[0]:
            raise ValueError(f"incompatible dimensions ({c.shape} and {b.shape})")

        # Quick return for empty arrays
        if b.size == 0:
            output_storage[0][0] = np.empty_like(b, dtype=potrs.dtype)
            return

        x, info = potrs(c, b, lower=self.lower, overwrite_b=self.overwrite_b)
        if info != 0:
            x[...] = np.nan

        output_storage[0][0] = x

    def pullback(self, *args, **kwargs):
        # TODO: Base impl should work, let's try it
        raise NotImplementedError()

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        if 1 in allowed_inplace_inputs:
            new_props = self._props_dict()  # type: ignore
            new_props["overwrite_b"] = True
            return type(self)(**new_props)
        else:
            return self


def cho_solve(
    c_and_lower: tuple[TensorLike, bool],
    b: TensorLike,
    *,
    b_ndim: int | None = None,
):
    """Solve the linear equations A x = b, given the Cholesky factorization of A.

    Parameters
    ----------
    c_and_lower : tuple of (TensorLike, bool)
        Cholesky factorization of a, as given by cho_factor
    b : TensorLike
        Right-hand side
    check_finite : bool
        Unused by PyTensor. PyTensor will return nan if the operation fails.
    b_ndim : int
        Whether the core case of b is a vector (1) or matrix (2).
        This will influence how batched dimensions are interpreted.
    """
    A, lower = c_and_lower
    b_ndim = _default_b_ndim(b, b_ndim)
    return Blockwise(CholeskySolve(lower=lower, b_ndim=b_ndim))(A, b)
