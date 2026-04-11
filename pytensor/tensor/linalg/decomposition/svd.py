from collections.abc import Sequence
from typing import cast

import numpy as np

from pytensor.gradient import DisconnectedType, disconnected_type
from pytensor.graph import Apply, Op, Variable
from pytensor.tensor import basic as ptb
from pytensor.tensor import math as ptm
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.type import matrix, vector


class SVD(Op):
    """
    Computes singular value decomposition of matrix A, into U, S, V such that A = U @ S @ V

    Parameters
    ----------
    full_matrices : bool, optional
        If True (default), u and v have the shapes (M, M) and (N, N),
        respectively.
        Otherwise, the shapes are (M, K) and (K, N), respectively,
        where K = min(M, N).
    compute_uv : bool, optional
        Whether or not to compute u and v in addition to s.
        True by default.

    """

    # See doc in the docstring of the function just after this class.
    __props__ = ("full_matrices", "compute_uv")

    def __init__(self, full_matrices: bool = True, compute_uv: bool = True):
        self.full_matrices = bool(full_matrices)
        self.compute_uv = bool(compute_uv)
        if self.compute_uv:
            if self.full_matrices:
                self.gufunc_signature = "(m,n)->(m,m),(k),(n,n)"
            else:
                self.gufunc_signature = "(m,n)->(m,k),(k),(k,n)"
        else:
            self.gufunc_signature = "(m,n)->(k)"

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2, "The input of svd function should be a matrix."

        in_dtype = x.type.numpy_dtype
        if in_dtype.name.startswith("int"):
            out_dtype = np.dtype(f"f{in_dtype.itemsize}")
        else:
            out_dtype = in_dtype

        s = vector(dtype=out_dtype)

        if self.compute_uv:
            u = matrix(dtype=out_dtype)
            vt = matrix(dtype=out_dtype)
            return Apply(self, [x], [u, s, vt])
        else:
            return Apply(self, [x], [s])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        assert x.ndim == 2, "The input of svd function should be a matrix."
        if self.compute_uv:
            u, s, vt = outputs
            u[0], s[0], vt[0] = np.linalg.svd(x, self.full_matrices, self.compute_uv)
        else:
            (s,) = outputs
            s[0] = np.linalg.svd(x, self.full_matrices, self.compute_uv)

    def infer_shape(self, fgraph, node, shapes):
        (x_shape,) = shapes
        M, N = x_shape
        K = ptm.minimum(M, N)
        s_shape = (K,)
        if self.compute_uv:
            u_shape = (M, M) if self.full_matrices else (M, K)
            vt_shape = (N, N) if self.full_matrices else (K, N)
            return [u_shape, s_shape, vt_shape]
        else:
            return [s_shape]

    def pullback(
        self,
        inputs: Sequence[Variable],
        outputs: Sequence[Variable],
        output_grads: Sequence[Variable],
    ) -> list[Variable]:
        """
        Reverse-mode gradient of the SVD function. Adapted from the autograd implementation here:
        https://github.com/HIPS/autograd/blob/01eacff7a4f12e6f7aebde7c4cb4c1c2633f217d/autograd/numpy/linalg.py#L194

        And the mxnet implementation described in ..[1]

        References
        ----------
        .. [1] Seeger, Matthias, et al. "Auto-differentiating linear algebra." arXiv preprint arXiv:1710.08717 (2017).
        """

        def s_grad_only(
            U: ptb.TensorVariable, VT: ptb.TensorVariable, ds: ptb.TensorVariable
        ) -> list[Variable]:
            A_bar = (U.conj() * ds[..., None, :]) @ VT
            return [A_bar]

        (A,) = (cast(ptb.TensorVariable, x) for x in inputs)

        if not self.compute_uv:
            # We need all the components of the SVD to compute the gradient of A even if we only use the singular values
            # in the cost function.
            U, _, VT = svd(A, full_matrices=False, compute_uv=True)
            ds = cast(ptb.TensorVariable, output_grads[0])
            return s_grad_only(U, VT, ds)

        elif self.full_matrices:
            raise NotImplementedError(
                "Gradient of svd not implemented for full_matrices=True"
            )

        else:
            U, s, VT = (cast(ptb.TensorVariable, x) for x in outputs)

            # Handle disconnected inputs
            # If a user asked for all the matrices but then only used a subset in the cost function, the unused outputs
            # will be DisconnectedType. We replace DisconnectedTypes with zero matrices of the correct shapes.
            new_output_grads = []
            is_disconnected = [
                isinstance(x.type, DisconnectedType) for x in output_grads
            ]
            if all(is_disconnected):
                # This should never actually be reached by Pytensor -- the SVD Op should be pruned from the gradient
                # graph if it's fully disconnected. It is included for completeness.
                return [disconnected_type()]  # pragma: no cover

            elif is_disconnected == [True, False, True]:
                # This is the same as the compute_uv = False, so we can drop back to that simpler computation, without
                # needing to re-compoute U and VT
                ds = cast(ptb.TensorVariable, output_grads[1])
                return s_grad_only(U, VT, ds)

            for disconnected, output_grad, output in zip(
                is_disconnected, output_grads, [U, s, VT], strict=True
            ):
                if disconnected:
                    new_output_grads.append(output.zeros_like())
                else:
                    new_output_grads.append(output_grad)

            (dU, ds, dVT) = (cast(ptb.TensorVariable, x) for x in new_output_grads)

            V = VT.T
            dV = dVT.T

            m, n = A.shape[-2:]

            k = ptm.min((m, n))
            eye = ptb.eye(k)

            def h(t):
                """
                Approximation of s_i ** 2 - s_j ** 2, from .. [1].
                Robust to identical singular values (singular matrix input), although
                gradients are still wrong in this case.
                """
                eps = 1e-8

                # sign(0) = 0 in pytensor, which defeats the whole purpose of this function
                sign_t = ptb.where(ptm.eq(t, 0), 1, ptm.sign(t))
                return ptm.maximum(ptm.abs(t), eps) * sign_t

            numer = ptb.ones((k, k)) - eye
            denom = h(s[None] - s[:, None]) * h(s[None] + s[:, None])
            E = numer / denom

            utgu = U.T @ dU
            vtgv = VT @ dV

            A_bar = (E * (utgu - utgu.conj().T)) * s[..., None, :]
            A_bar = A_bar + eye * ds[..., :, None]
            A_bar = A_bar + s[..., :, None] * (E * (vtgv - vtgv.conj().T))
            A_bar = U.conj() @ A_bar @ VT

            A_bar = ptb.switch(
                ptm.eq(m, n),
                A_bar,
                ptb.switch(
                    ptm.lt(m, n),
                    A_bar
                    + (
                        U / s[..., None, :] @ dVT @ (ptb.eye(n) - V @ V.conj().T)
                    ).conj(),
                    A_bar
                    + (V / s[..., None, :] @ dU.T @ (ptb.eye(m) - U @ U.conj().T)).T,
                ),
            )
            return [A_bar]


def svd(a, full_matrices: bool = True, compute_uv: bool = True):
    """
    This function performs the SVD on CPU.

    Parameters
    ----------
    full_matrices : bool, optional
        If True (default), u and v have the shapes (M, M) and (N, N),
        respectively.
        Otherwise, the shapes are (M, K) and (K, N), respectively,
        where K = min(M, N).
    compute_uv : bool, optional
        Whether or not to compute u and v in addition to s.
        True by default.

    Returns
    -------
    U, V, D : matrices

    """
    return Blockwise(SVD(full_matrices, compute_uv))(a)
