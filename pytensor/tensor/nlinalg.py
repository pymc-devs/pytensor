import warnings
from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal, cast

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

from pytensor import scalar as ps
from pytensor.compile.builders import OpFromGraph
from pytensor.gradient import DisconnectedType, disconnected_type
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor import TensorLike
from pytensor.tensor import basic as ptb
from pytensor.tensor import math as ptm
from pytensor.tensor.basic import as_tensor_variable, diagonal
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.type import (
    Variable,
    dmatrix,
    dvector,
    iscalar,
    matrix,
    scalar,
    tensor,
    vector,
)


class MatrixPinv(Op):
    __props__ = ("hermitian",)
    gufunc_signature = "(m,n)->(n,m)"

    def __init__(self, hermitian):
        self.hermitian = hermitian

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        if x.type.numpy_dtype.kind in "ibu":
            out_dtype = "float64"
        else:
            out_dtype = x.dtype
        return Apply(self, [x], [matrix(shape=x.type.shape, dtype=out_dtype)])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        z[0] = np.linalg.pinv(x, hermitian=self.hermitian)

    def L_op(self, inputs, outputs, g_outputs):
        r"""The gradient function should return

            .. math:: V\frac{\partial X^+}{\partial X},

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. According to `Wikipedia
        <https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse#Derivative>`_,
        this corresponds to

            .. math:: (-X^+ V^T X^+ + X^+ X^{+T} V (I - X X^+) + (I - X^+ X) V X^{+T} X^+)^T.
        """
        (x,) = inputs
        (z,) = outputs
        (gz,) = g_outputs

        x_dot_z = ptm.dot(x, z)
        z_dot_x = ptm.dot(z, x)

        grad = (
            -matrix_dot(z, gz.T, z)
            + matrix_dot(z, z.T, gz, (ptb.identity_like(x_dot_z) - x_dot_z))
            + matrix_dot((ptb.identity_like(z_dot_x) - z_dot_x), gz, z.T, z)
        ).T
        return [grad]

    def infer_shape(self, fgraph, node, shapes):
        return [list(reversed(shapes[0]))]


def pinv(x, hermitian=False):
    """Computes the pseudo-inverse of a matrix :math:`A`.

    The pseudo-inverse of a matrix :math:`A`, denoted :math:`A^+`, is
    defined as: "the matrix that 'solves' [the least-squares problem]
    :math:`Ax = b`," i.e., if :math:`\\bar{x}` is said solution, then
    :math:`A^+` is that matrix such that :math:`\\bar{x} = A^+b`.

    Note that :math:`Ax=AA^+b`, so :math:`AA^+` is close to the identity matrix.
    This method is not faster than `matrix_inverse`. Its strength comes from
    that it works for non-square matrices.
    If you have a square matrix though, `matrix_inverse` can be both more
    exact and faster to compute. Also this op does not get optimized into a
    solve op.

    """
    return Blockwise(MatrixPinv(hermitian=hermitian))(x)


class MatrixInverse(Op):
    r"""Computes the inverse of a matrix :math:`A`.

    Given a square matrix :math:`A`, ``matrix_inverse`` returns a square
    matrix :math:`A_{inv}` such that the dot product :math:`A \cdot A_{inv}`
    and :math:`A_{inv} \cdot A` equals the identity matrix :math:`I`.

    Notes
    -----
    When possible, the call to this op will be optimized to the call
    of ``solve``.

    """

    __props__ = ()
    gufunc_signature = "(m,m)->(m,m)"
    gufunc_spec = ("numpy.linalg.inv", 1, 1)

    def __init__(self):
        pass

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        if x.type.numpy_dtype.kind in "ibu":
            out_dtype = "float64"
        else:
            out_dtype = x.dtype
        return Apply(self, [x], [matrix(shape=x.type.shape, dtype=out_dtype)])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        z[0] = np.linalg.inv(x)

    def grad(self, inputs, g_outputs):
        r"""The gradient function should return

            .. math:: V\frac{\partial X^{-1}}{\partial X},

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to

            .. math:: (X^{-1} \cdot V^{T} \cdot X^{-1})^T.

        """
        (x,) = inputs
        xi = self(x)
        (gz,) = g_outputs
        # ptm.dot(gz.T,xi)
        return [-matrix_dot(xi, gz.T, xi).T]

    def R_op(self, inputs, eval_points):
        r"""The gradient function should return

            .. math:: \frac{\partial X^{-1}}{\partial X}V,

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to

            .. math:: X^{-1} \cdot V \cdot X^{-1}.

        """
        (x,) = inputs
        xi = self(x)
        (ev,) = eval_points
        if ev is None:
            return [None]
        return [-matrix_dot(xi, ev, xi)]

    def infer_shape(self, fgraph, node, shapes):
        return shapes


inv = matrix_inverse = Blockwise(MatrixInverse())


def matrix_dot(*args):
    r"""Shorthand for product between several dots.

    Given :math:`N` matrices :math:`A_0, A_1, .., A_N`, ``matrix_dot`` will
    generate the matrix product between all in the given order, namely
    :math:`A_0 \cdot A_1 \cdot A_2 \cdot .. \cdot A_N`.

    """
    rval = args[0]
    for a in args[1:]:
        rval = ptm.dot(rval, a)
    return rval


def trace(X):
    """
    Returns the sum of diagonal elements of matrix X.
    """
    warnings.warn(
        "pytensor.tensor.linalg.trace is deprecated. Use pytensor.tensor.trace instead.",
        FutureWarning,
    )
    return diagonal(X).sum()


class Det(Op):
    """
    Matrix determinant. Input should be a square matrix.

    """

    __props__ = ()
    gufunc_signature = "(m,m)->()"
    gufunc_spec = ("numpy.linalg.det", 1, 1)

    def make_node(self, x):
        x = as_tensor_variable(x)
        if x.ndim != 2:
            raise ValueError(
                f"Input passed is not a valid 2D matrix. Current ndim {x.ndim} != 2"
            )
        # Check for known shapes and square matrix
        if None not in x.type.shape and (x.type.shape[0] != x.type.shape[1]):
            raise ValueError(
                f"Determinant not defined for non-square matrix inputs. Shape received is {x.type.shape}"
            )
        if x.type.numpy_dtype.kind in "ibu":
            out_dtype = "float64"
        else:
            out_dtype = x.dtype
        o = scalar(dtype=out_dtype)
        return Apply(self, [x], [o])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        try:
            z[0] = np.asarray(np.linalg.det(x))
        except Exception as e:
            raise ValueError("Failed to compute determinant", x) from e

    def grad(self, inputs, g_outputs):
        (gz,) = g_outputs
        (x,) = inputs
        return [gz * self(x) * matrix_inverse(x).T]

    def infer_shape(self, fgraph, node, shapes):
        return [()]

    def __str__(self):
        return "Det"


det = Blockwise(Det())


class SLogDet(Op):
    """
    Compute the log determinant and its sign of the matrix. Input should be a square matrix.
    """

    __props__ = ()
    gufunc_signature = "(m,m)->(),()"
    gufunc_spec = ("numpy.linalg.slogdet", 1, 2)

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        if x.type.numpy_dtype.kind in "ibu":
            out_dtype = "float64"
        else:
            out_dtype = x.dtype
        sign = scalar(dtype=out_dtype)
        det = scalar(dtype=out_dtype)
        return Apply(self, [x], [sign, det])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (sign, det) = outputs
        try:
            sign[0], det[0] = (np.array(z) for z in np.linalg.slogdet(x))
        except Exception as e:
            raise ValueError("Failed to compute determinant", x) from e

    def infer_shape(self, fgraph, node, shapes):
        return [(), ()]

    def __str__(self):
        return "SLogDet"


def slogdet(x: TensorLike) -> tuple[ptb.TensorVariable, ptb.TensorVariable]:
    """
    Compute the sign and (natural) logarithm of the determinant of an array.

    Returns a naive graph which is optimized later using rewrites with the det operation.

    Parameters
    ----------
    x : (..., M, M) tensor or tensor_like
        Input tensor, has to be square.

    Returns
    -------
    A tuple with the following attributes:

    sign : (...) tensor_like
        A number representing the sign of the determinant. For a real matrix,
        this is 1, 0, or -1.
    logabsdet : (...) tensor_like
        The natural log of the absolute value of the determinant.

    If the determinant is zero, then `sign` will be 0 and `logabsdet`
    will be -inf. In all cases, the determinant is equal to
    ``sign * exp(logabsdet)``.
    """
    det_val = det(x)
    return ptm.sign(det_val), ptm.log(ptm.abs(det_val))


class Eig(Op):
    """
    Compute the eigenvalues and right eigenvectors of a square array.
    """

    __props__: tuple[str, ...] = ()
    # Can't use numpy directly in Blockwise, because of the dynamic dtype
    # gufunc_spec = ("numpy.linalg.eig", 1, 2)
    gufunc_signature = "(m,m)->(m),(m,m)"

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2

        M, N = x.type.shape

        if M is not None and N is not None and M != N:
            raise ValueError(
                f"Input to Eig must be a square matrix, got static shape: ({M}, {N})"
            )

        dtype = np.promote_types(x.dtype, np.complex64)

        w = tensor(dtype=dtype, shape=(M,))
        v = tensor(dtype=dtype, shape=(M, N))

        return Apply(self, [x], [w, v])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        dtype = np.promote_types(x.dtype, np.complex64)

        w, v = np.linalg.eig(x)

        # If the imaginary part of the eigenvalues is zero, numpy automatically casts them to real. We require
        # a statically known return dtype, so we have to cast back to complex to avoid dtype mismatch.
        outputs[0][0] = w.astype(dtype, copy=False)
        outputs[1][0] = v.astype(dtype, copy=False)

    def infer_shape(self, fgraph, node, shapes):
        (x_shapes,) = shapes
        n, _ = x_shapes

        return [(n,), (n, n)]

    def L_op(self, inputs, outputs, output_grads):
        raise NotImplementedError(
            "Gradients for Eig is not implemented because it always returns complex values, "
            "for which autodiff is not yet supported in PyTensor (PRs welcome :) ).\n"
            "If you know that your input has strictly real-valued eigenvalues (e.g. it is a "
            "symmetric matrix), use pt.linalg.eigh instead."
        )


def eig(x: TensorLike):
    """
    Return the eigenvalues and right eigenvectors of a square array.

    Note that regardless of the input dtype, the eigenvalues and eigenvectors are returned as complex numbers. As a
    result, the gradient of this operation is not implemented (because PyTensor does not support autodiff for complex
    values yet).

    If you know that your input has strictly real-valued eigenvalues (e.g. it is a symmetric matrix), use
    `pytensor.tensor.linalg.eigh` instead.

    Parameters
    ----------
    x: TensorLike
        Square matrix, or array of such matrices
    """
    return Blockwise(Eig())(x)


class Eigh(Eig):
    """
    Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.
    """

    __props__ = ("UPLO",)

    def __init__(self, UPLO="L"):
        assert UPLO in ("L", "U")
        self.UPLO = UPLO

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        # Numpy's linalg.eigh may return either double or single
        # presision eigenvalues depending on installed version of
        # LAPACK.  Rather than trying to reproduce the (rather
        # involved) logic, we just probe linalg.eigh with a trivial
        # input.
        w_dtype = np.linalg.eigh([[np.dtype(x.dtype).type()]])[0].dtype.name
        w = vector(dtype=w_dtype)
        v = matrix(dtype=w_dtype)
        return Apply(self, [x], [w, v])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (w, v) = outputs
        w[0], v[0] = np.linalg.eigh(x, self.UPLO)

    def L_op(self, inputs, outputs, output_grads):
        r"""The gradient function should return

           .. math:: \sum_n\left(W_n\frac{\partial\,w_n}
                           {\partial a_{ij}} +
                     \sum_k V_{nk}\frac{\partial\,v_{nk}}
                           {\partial a_{ij}}\right),

        where [:math:`W`, :math:`V`] corresponds to ``g_outputs``,
        :math:`a` to ``inputs``, and  :math:`(w, v)=\mbox{eig}(a)`.

        Analytic formulae for eigensystem gradients are well-known in
        perturbation theory:

           .. math:: \frac{\partial\,w_n}
                          {\partial a_{ij}} = v_{in}\,v_{jn}


           .. math:: \frac{\partial\,v_{kn}}
                          {\partial a_{ij}} =
                \sum_{m\ne n}\frac{v_{km}v_{jn}}{w_n-w_m}

        """
        (x,) = inputs
        w, v = outputs
        gw, gv = _zero_disconnected([w, v], output_grads)

        return [EighGrad(self.UPLO)(x, w, v, gw, gv)]


def _zero_disconnected(outputs, grads):
    l = []
    for o, g in zip(outputs, grads, strict=True):
        if isinstance(g.type, DisconnectedType):
            l.append(o.zeros_like())
        else:
            l.append(g)
    return l


class EighGrad(Op):
    """
    Gradient of an eigensystem of a Hermitian matrix.

    """

    __props__ = ("UPLO",)

    def __init__(self, UPLO="L"):
        assert UPLO in ("L", "U")
        self.UPLO = UPLO
        if UPLO == "L":
            self.tri0 = np.tril
            self.tri1 = partial(np.triu, k=1)
        else:
            self.tri0 = np.triu
            self.tri1 = partial(np.tril, k=-1)

    def make_node(self, x, w, v, gw, gv):
        x, w, v, gw, gv = map(as_tensor_variable, (x, w, v, gw, gv))
        assert x.ndim == 2
        assert w.ndim == 1
        assert v.ndim == 2
        assert gw.ndim == 1
        assert gv.ndim == 2
        out_dtype = ps.upcast(x.dtype, w.dtype, v.dtype, gw.dtype, gv.dtype)
        out = matrix(dtype=out_dtype)
        return Apply(self, [x, w, v, gw, gv], [out])

    def perform(self, node, inputs, outputs):
        """
        Implements the "reverse-mode" gradient for the eigensystem of
        a square matrix.

        """
        x, w, v, W, V = inputs
        N = x.shape[0]
        outer = np.outer

        def G(n):
            return sum(
                v[:, m] * V.T[n].dot(v[:, m]) / (w[n] - w[m])
                for m in range(N)
                if m != n
            )

        g = sum(outer(v[:, n], v[:, n] * W[n] + G(n)) for n in range(N))

        # Numpy's eigh(a, 'L') (eigh(a, 'U')) is a function of tril(a)
        # (triu(a)) only.  This means that partial derivative of
        # eigh(a, 'L') (eigh(a, 'U')) with respect to a[i,j] is zero
        # for i < j (i > j).  At the same time, non-zero components of
        # the gradient must account for the fact that variation of the
        # opposite triangle contributes to variation of two elements
        # of Hermitian (symmetric) matrix. The following line
        # implements the necessary logic.
        out = self.tri0(g) + self.tri1(g).T

        # Make sure we return the right dtype even if NumPy performed
        # upcasting in self.tri0.
        outputs[0][0] = np.asarray(out, dtype=node.outputs[0].dtype)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]


def eigh(a, UPLO="L"):
    return Eigh(UPLO)(a)


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

    def L_op(
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


class Lstsq(Op):
    __props__ = ()

    def make_node(self, x, y, rcond):
        x = as_tensor_variable(x)
        y = as_tensor_variable(y)
        rcond = as_tensor_variable(rcond)
        return Apply(
            self,
            [x, y, rcond],
            [
                dmatrix(),
                dvector(),
                iscalar(),
                dvector(),
            ],
        )

    def perform(self, node, inputs, outputs):
        zz = np.linalg.lstsq(inputs[0], inputs[1], inputs[2])
        outputs[0][0] = zz[0]
        outputs[1][0] = zz[1]
        outputs[2][0] = np.asarray(zz[2])
        outputs[3][0] = zz[3]


lstsq = Lstsq()


def matrix_power(M, n):
    r"""Raise a square matrix, ``M``, to the (integer) power ``n``.

    This implementation uses exponentiation by squaring which is
    significantly faster than the naive implementation.
    The time complexity for exponentiation by squaring is
    :math: `\mathcal{O}((n \log M)^k)`

    Parameters
    ----------
    M: TensorVariable
    n: int

    """
    if n < 0:
        M = pinv(M)
        n = abs(n)

    # Shortcuts when 0 < n <= 3
    if n == 0:
        return ptb.eye(M.shape[-2])

    elif n == 1:
        return M

    elif n == 2:
        return ptm.dot(M, M)

    elif n == 3:
        return ptm.dot(ptm.dot(M, M), M)

    result = z = None

    while n > 0:
        z = M if z is None else ptm.dot(z, z)
        n, bit = divmod(n, 2)
        if bit:
            result = z if result is None else ptm.dot(result, z)

    return result


def _multi_svd_norm(
    x: ptb.TensorVariable, row_axis: int, col_axis: int, reduce_op: Callable
):
    """Compute a function of the singular values of the 2-D matrices in `x`.

    This is a private utility function used by `pytensor.tensor.nlinalg.norm()`.

    Copied from `np.linalg._multi_svd_norm`.

    Parameters
    ----------
    x : TensorVariable
        Input tensor.
    row_axis, col_axis : int
        The axes of `x` that hold the 2-D matrices.
    reduce_op : callable
        Reduction op. Should be one of `pt.min`, `pt.max`, or `pt.sum`

    Returns
    -------
    result : float or ndarray
        If `x` is 2-D, the return values is a float.
        Otherwise, it is an array with ``x.ndim - 2`` dimensions.
        The return values are either the minimum or maximum or sum of the
        singular values of the matrices, depending on whether `op`
        is `pt.amin` or `pt.amax` or `pt.sum`.

    """
    y = ptb.moveaxis(x, (row_axis, col_axis), (-2, -1))
    result = reduce_op(svd(y, compute_uv=False), axis=-1)
    return result


VALID_ORD = Literal["fro", "f", "nuc", "inf", "-inf", 0, 1, -1, 2, -2]


def norm(
    x: ptb.TensorVariable,
    ord: float | VALID_ORD | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
):
    """
    Matrix or vector norm.

    Parameters
    ----------
    x: TensorVariable
        Tensor to take norm of.

    ord: float, str or int, optional
        Order of norm. If `ord` is a str, it must be one of the following:
            - 'fro' or 'f' : Frobenius norm
            - 'nuc' : nuclear norm
            - 'inf' : Infinity norm
            - '-inf' : Negative infinity norm
        If an integer, order can be one of -2, -1, 0, 1, or 2.
        Otherwise `ord` must be a float.

        Default is the Frobenius (L2) norm.

    axis: tuple of int, optional
        Axes over which to compute the norm. If None, norm of entire matrix (or vector) is computed. Row or column
        norms can be computed by passing a single integer; this will treat a matrix like a batch of vectors.

    keepdims: bool
        If True, dummy axes will be inserted into the output so that norm.dnim == x.dnim. Default is False.

    Returns
    -------
    TensorVariable
        Norm of `x` along axes specified by `axis`.

    Notes
    -----
    Batched dimensions are supported to the left of the core dimensions. For example, if `x` is a 3D tensor with
    shape (2, 3, 4), then `norm(x)` will compute the norm of each 3x4 matrix in the batch.

    If the input is a 2D tensor and should be treated as a batch of vectors, the `axis` argument must be specified.
    """
    x = ptb.as_tensor_variable(x)

    ndim = x.ndim
    core_ndim = min(2, ndim)
    batch_ndim = ndim - core_ndim

    if axis is None:
        # Handle some common cases first. These can be computed more quickly than the default SVD way, so we always
        # want to check for them.
        if (
            (ord is None)
            or (ord in ("f", "fro") and core_ndim == 2)
            or (ord == 2 and core_ndim == 1)
        ):
            x = x.reshape(tuple(x.shape[:-2]) + (-1,) + (1,) * (core_ndim - 1))
            batch_T_dim_order = tuple(range(batch_ndim)) + tuple(
                range(batch_ndim + core_ndim - 1, batch_ndim - 1, -1)
            )

            if x.dtype.startswith("complex"):
                x_real = x.real  # type: ignore
                x_imag = x.imag  # type: ignore
                sqnorm = (
                    ptb.transpose(x_real, batch_T_dim_order) @ x_real
                    + ptb.transpose(x_imag, batch_T_dim_order) @ x_imag
                )
            else:
                sqnorm = ptb.transpose(x, batch_T_dim_order) @ x
            ret = ptm.sqrt(sqnorm).squeeze()
            if keepdims:
                ret = ptb.shape_padright(ret, core_ndim)
            return ret

        # No special computation to exploit -- set default axis before continuing
        axis = tuple(range(core_ndim))

    elif not isinstance(axis, tuple):
        try:
            axis = int(axis)
        except Exception as e:
            raise TypeError(
                "'axis' must be None, an integer, or a tuple of integers"
            ) from e

        axis = (axis,)

    if len(axis) == 1:
        # Vector norms
        if ord in [None, "fro", "f"] and (core_ndim == 2):
            # This is here to catch the case where X is a 2D tensor but the user wants to treat it as a batch of
            # vectors. Other vector norms will work fine in this case.
            ret = ptm.sqrt(ptm.sum((x.conj() * x).real, axis=axis, keepdims=keepdims))
        elif (ord == "inf") or (ord == np.inf):
            ret = ptm.max(ptm.abs(x), axis=axis, keepdims=keepdims)
        elif (ord == "-inf") or (ord == -np.inf):
            ret = ptm.min(ptm.abs(x), axis=axis, keepdims=keepdims)
        elif ord == 0:
            ret = ptm.neq(x, 0).sum(axis=axis, keepdims=keepdims)
        elif ord == 1:
            ret = ptm.sum(ptm.abs(x), axis=axis, keepdims=keepdims)
        elif isinstance(ord, str):
            raise ValueError(f"Invalid norm order '{ord}' for vectors")
        else:
            ret = ptm.sum(ptm.abs(x) ** ord, axis=axis, keepdims=keepdims)
            ret **= ptm.reciprocal(ord)

        return ret

    elif len(axis) == 2:
        # Matrix norms
        row_axis, col_axis = (
            batch_ndim + x for x in normalize_axis_tuple(axis, core_ndim)
        )
        axis = (row_axis, col_axis)

        if ord in [None, "fro", "f"]:
            ret = ptm.sqrt(ptm.sum((x.conj() * x).real, axis=axis))

        elif (ord == "inf") or (ord == np.inf):
            if row_axis > col_axis:
                row_axis -= 1
            ret = ptm.max(ptm.sum(ptm.abs(x), axis=col_axis), axis=row_axis)

        elif (ord == "-inf") or (ord == -np.inf):
            if row_axis > col_axis:
                row_axis -= 1
            ret = ptm.min(ptm.sum(ptm.abs(x), axis=col_axis), axis=row_axis)

        elif ord == 1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = ptm.max(ptm.sum(ptm.abs(x), axis=row_axis), axis=col_axis)

        elif ord == -1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = ptm.min(ptm.sum(ptm.abs(x), axis=row_axis), axis=col_axis)

        elif ord == 2:
            ret = _multi_svd_norm(x, row_axis, col_axis, ptm.max)

        elif ord == -2:
            ret = _multi_svd_norm(x, row_axis, col_axis, ptm.min)

        elif ord == "nuc":
            ret = _multi_svd_norm(x, row_axis, col_axis, ptm.sum)

        else:
            raise ValueError(f"Invalid norm order for matrices: {ord}")

        if keepdims:
            ret = ptb.expand_dims(ret, axis)

        return ret
    else:
        raise ValueError(
            f"Cannot compute norm when core_dims < 1 or core_dims > 3, found: core_dims = {core_ndim}"
        )


class TensorInv(Op):
    """
    Class wrapper for tensorinv() function;
    PyTensor utilization of numpy.linalg.tensorinv;
    """

    __props__ = ("ind",)

    def __init__(self, ind=2):
        self.ind = ind

    def make_node(self, a):
        a = as_tensor_variable(a)
        out = a.type()
        return Apply(self, [a], [out])

    def perform(self, node, inputs, outputs):
        (a,) = inputs
        (x,) = outputs
        x[0] = np.linalg.tensorinv(a, self.ind)

    def infer_shape(self, fgraph, node, shapes):
        sp = shapes[0][self.ind :] + shapes[0][: self.ind]
        return [sp]


def tensorinv(a, ind=2):
    """
    PyTensor utilization of numpy.linalg.tensorinv;

    Compute the 'inverse' of an N-dimensional array.
    The result is an inverse for `a` relative to the tensordot operation
    ``tensordot(a, b, ind)``, i. e., up to floating-point accuracy,
    ``tensordot(tensorinv(a), a, ind)`` is the "identity" tensor for the
    tensordot operation.

    Parameters
    ----------
    a : array_like
        Tensor to 'invert'. Its shape must be 'square', i. e.,
        ``prod(a.shape[:ind]) == prod(a.shape[ind:])``.
    ind : int, optional
        Number of first indices that are involved in the inverse sum.
        Must be a positive integer, default is 2.

    Returns
    -------
    b : ndarray
        `a`'s tensordot inverse, shape ``a.shape[ind:] + a.shape[:ind]``.

    Raises
    ------
    LinAlgError
        If `a` is singular or not 'square' (in the above sense).
    """
    return TensorInv(ind)(a)


class TensorSolve(Op):
    """
    PyTensor utilization of numpy.linalg.tensorsolve
    Class wrapper for tensorsolve function.

    """

    __props__ = ("axes",)

    def __init__(self, axes=None):
        self.axes = axes

    def make_node(self, a, b):
        a = as_tensor_variable(a)
        b = as_tensor_variable(b)
        out_dtype = ps.upcast(a.dtype, b.dtype)
        x = matrix(dtype=out_dtype)
        return Apply(self, [a, b], [x])

    def perform(self, node, inputs, outputs):
        (
            a,
            b,
        ) = inputs
        (x,) = outputs
        x[0] = np.linalg.tensorsolve(a, b, self.axes)


def tensorsolve(a, b, axes=None):
    """
    PyTensor utilization of numpy.linalg.tensorsolve.

    Solve the tensor equation ``a x = b`` for x.
    It is assumed that all indices of `x` are summed over in the product,
    together with the rightmost indices of `a`, as is done in, for example,
    ``tensordot(a, x, axes=len(b.shape))``.

    Parameters
    ----------
    a : array_like
        Coefficient tensor, of shape ``b.shape + Q``. `Q`, a tuple, equals
        the shape of that sub-tensor of `a` consisting of the appropriate
        number of its rightmost indices, and must be such that
        ``prod(Q) == prod(b.shape)`` (in which sense `a` is said to be
        'square').
    b : array_like
        Right-hand tensor, which can be of any shape.
    axes : tuple of ints, optional
        Axes in `a` to reorder to the right, before inversion.
        If None (default), no reordering is done.
    Returns
    -------
    x : ndarray, shape Q
    Raises
    ------
    LinAlgError
        If `a` is singular or not 'square' (in the above sense).
    """

    return TensorSolve(axes)(a, b)


class KroneckerProduct(OpFromGraph):
    """
    Wrapper Op for Kronecker graphs
    """


def kron(a, b):
    """Kronecker product.

    Same as np.kron(a, b)

    Parameters
    ----------
    a: array_like
    b: array_like

    Returns
    -------
    array_like with a.ndim + b.ndim - 2 dimensions
    """
    a = as_tensor_variable(a)
    b = as_tensor_variable(b)

    if a is b:
        # In case a is the same as b, we need a different variable to build the OFG
        b = a.copy()

    if a.ndim + b.ndim <= 2:
        raise TypeError(
            "kron: inputs dimensions must sum to 3 or more. "
            f"You passed {int(a.ndim)} and {int(b.ndim)}."
        )

    if a.ndim < b.ndim:
        a = ptb.expand_dims(a, tuple(range(b.ndim - a.ndim)))
    elif b.ndim < a.ndim:
        b = ptb.expand_dims(b, tuple(range(a.ndim - b.ndim)))
    a_reshaped = ptb.expand_dims(a, tuple(range(1, 2 * a.ndim, 2)))
    b_reshaped = ptb.expand_dims(b, tuple(range(0, 2 * b.ndim, 2)))
    out_shape = tuple(a.shape * b.shape)
    output_out_of_shape = a_reshaped * b_reshaped
    output_reshaped = output_out_of_shape.reshape(out_shape)

    return KroneckerProduct(inputs=[a, b], outputs=[output_reshaped])(a, b)


__all__ = [
    "det",
    "eig",
    "eigh",
    "inv",
    "kron",
    "lstsq",
    "matrix_dot",
    "matrix_power",
    "norm",
    "pinv",
    "slogdet",
    "svd",
    "tensorinv",
    "tensorsolve",
    "trace",
]
