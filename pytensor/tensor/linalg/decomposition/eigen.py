import warnings
from typing import cast

import numpy as np
import scipy.linalg as scipy_linalg

import pytensor
from pytensor import scalar as ps
from pytensor.gradient import DisconnectedType
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor import TensorLike
from pytensor.tensor.basic import as_tensor_variable, diag, eye, tril, triu
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg.dtype_utils import linalg_real_output_dtype
from pytensor.tensor.type import Variable, matrix, tensor, vector
from pytensor.tensor.math import sub, switch
from pytensor.tensor.type_other import NoneTypeT


def _zero_disconnected(outputs, grads):
    l = []
    for o, g in zip(outputs, grads, strict=True):
        if isinstance(g.type, DisconnectedType):
            l.append(o.zeros_like())
        else:
            l.append(g)
    return l


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

    def pullback(self, inputs, outputs, output_grads):
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


class Eigh(Op):
    """
    Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.

    Optionally solves the generalized eigenvalue problem ``A @ v = w * B @ v``
    when a second matrix *b* is provided (delegated to ``scipy.linalg.eigh``).
    """

    __props__ = ("lower", "overwrite_a", "overwrite_b", "driver")

    def __init__(
        self,
        lower: bool = True,
        UPLO: str | None = None,
        overwrite_a: bool = False,
        overwrite_b: bool = False,
        driver: str = "evr",
    ):
        if UPLO is not None:
            warnings.warn(
                "UPLO is deprecated and will be removed in a future version. Use the ``lower`` argument "
                "instead.",
                stacklevel=2,
                category=DeprecationWarning,
            )
            lower = UPLO == "L"

        if driver not in ("evr", "evd"):
            raise ValueError(
                f"Invalid driver: {driver!r}. Must be one of 'evr', 'evd'."
            )

        if overwrite_a and overwrite_b:
            raise ValueError(
                "overwrite_a and overwrite_b are mutually exclusive: pytensor "
                "tracks at most one destroyed input per output."
            )

        self.lower = lower
        self.overwrite_a = overwrite_a
        self.overwrite_b = overwrite_b
        self.driver = driver

        # Output 1 (eigenvectors) is the one that lands in the destroyed buffer.
        if self.overwrite_a:
            self.destroy_map = {1: [0]}
        elif self.overwrite_b:
            self.destroy_map = {1: [1]}

    def make_node(self, a, b=None):
        a = as_tensor_variable(a)
        assert a.ndim == 2
        M, N = a.type.shape

        if M is not None and N is not None and M != N:
            raise ValueError(
                f"Input to Eigh must be a square matrix, got static shape: ({M}, {N})"
            )

        has_b = b is not None and not (
            isinstance(b, Variable) and isinstance(b.type, NoneTypeT)
        )

        if has_b:
            b = as_tensor_variable(b)
            inputs = [a, b]
        else:
            inputs = [a]

        w_dtype = linalg_real_output_dtype(*[x.type.dtype for x in inputs])

        w = tensor(dtype=w_dtype, shape=(M,))
        v = tensor(dtype=w_dtype, shape=(M, N))

        return Apply(self, inputs, [w, v])

    def perform(self, node, inputs, outputs):
        (w, v) = outputs
        if len(inputs) == 2:
            # Generalized eigenproblem: scipy doesn't accept driver= with b
            w[0], v[0] = scipy_linalg.eigh(
                inputs[0],
                b=inputs[1],
                lower=self.lower,
                overwrite_a=self.overwrite_a,
                overwrite_b=self.overwrite_b,
            )
        else:
            w[0], v[0] = scipy_linalg.eigh(
                inputs[0],
                lower=self.lower,
                overwrite_a=self.overwrite_a,
                driver=self.driver,
            )

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        # overwrite_a and overwrite_b are mutually exclusive; prefer overwrite_a
        # arbitrarily (memory savings are identical)
        new_props = self._props_dict()  # type: ignore
        if 0 in allowed_inplace_inputs:
            new_props["overwrite_a"] = True
        elif 1 in allowed_inplace_inputs:
            new_props["overwrite_b"] = True
        else:
            return self
        return type(self)(**new_props)

    def infer_shape(self, fgraph, node, shapes):
        n = shapes[0][0]
        return [(n,), (n, n)]

    def pullback(self, inputs, outputs, output_grads):
        r"""Symbolic gradient of ``eigh``.

        For the standard symmetric problem,

        .. math::

            A V = V \operatorname{diag}(w), \qquad V^T V = I,

        define

        .. math::

            F_{ij} =
            \begin{cases}
                \frac{1}{w_j - w_i}, & i \ne j, \\
                0, & i = j .
            \end{cases}

        Then the pullback is

        .. math::

            C = V^T g_V,
            \qquad
            M = \operatorname{diag}(g_w) + F \odot C,
            \qquad
            g_A = V M V^T.

        For the generalized symmetric-definite problem,

        .. math::

            A V = B V \operatorname{diag}(w), \qquad V^T B V = I,

        the pullback is

        .. math::

            C = V^T g_V,
            \qquad
            M = \operatorname{diag}(g_w) + F \odot C,

        .. math::

            g_A = V M V^T,

        .. math::

            g_B =
            -V \left(M \operatorname{diag}(w)\right) V^T
            - \frac12 V \operatorname{diag}(\operatorname{diag}(C)) V^T.

        The gradients are symmetrized on return to match the triangular storage
        specified by ``UPLO``.

        These formulas assume distinct eigenvalues. When eigenvalues are repeated,
        the factors ``1 / (w_j - w_i)`` are singular and the eigenvector gradient is
        not uniquely defined.
        """
        w, v = outputs
        gw, gv = _zero_disconnected([w, v], output_grads)

        # F_ij = 1/(w_j - w_i) for i != j, 0 on diagonal
        w_diff = sub.outer(w, w).T
        F = switch(eye(w.shape[0], dtype="bool"), 0.0, 1.0 / w_diff)

        if len(inputs) == 1:
            inner = diag(gw) + F * (v.T @ gv)
            g = v @ inner @ v.T

            if self.lower:
                out = tril(g) + triu(g, k=1).T
            else:
                out = triu(g) + tril(g, k=-1).T
            return [out]
        else:
            C = v.T @ gv
            inner = diag(gw) + F * C

            ga = v @ inner @ v.T
            gb = -v @ (inner * w[None, :]) @ v.T
            gb = gb - 0.5 * v @ diag(diag(C)) @ v.T

            if self.lower:
                ga_sym = tril(ga) + triu(ga, k=1).T
                gb_sym = tril(gb) + triu(gb, k=1).T
            else:
                ga_sym = triu(ga) + tril(ga, k=-1).T
                gb_sym = triu(gb) + tril(gb, k=-1).T
            return [ga_sym, gb_sym]


def eigh(
    a: TensorLike,
    b: TensorLike | None = None,
    lower: bool = True,
    UPLO: str | None = None,
    driver: str = "evr",
) -> list[Variable]:
    """
    Return the eigenvalues and eigenvectors of a symmetric/Hermitian matrix.

    Parameters
    ----------
    a : TensorLike
        Symmetric/Hermitian matrix (or batch thereof).
    b : TensorLike, optional
        Second matrix for the generalized eigenvalue problem ``A v = w B v``.
        Must be positive-definite. If ``None``, the standard eigenvalue
        problem is solved.
    lower : bool
        Whether to use the lower or upper triangle of a (and b, if provided). Default is True
    UPLO : {'L', 'U'}, optional
        Whether to use the lower or upper triangle of a (and b, if provided). Default is 'L' (lower).
        UPLO is deprecated and will be removed in a future version. Use the ``lower`` argument instead.
    driver : {'evr', 'evd'}, optional
        LAPACK driver to use. ``'evr'`` (default) uses the MRRR algorithm, the fastest general-purpose driver.
        This is the default used by Scipy. ``'evd'`` uses divide-and-conquer, matching NumPy, JAX, and MLX.

    Returns
    -------
    w : Variable
        Eigenvalues of the system, in ascending order.
    v : Variable
        Eigenvectors of the system, in ascending order.
    """
    if UPLO is not None:
        warnings.warn(
            "UPLO is deprecated and will be removed in a future version. ",
            stacklevel=2,
            category=DeprecationWarning,
        )
        lower = UPLO == "L"

    if b is None:
        signature = "(m,m)->(m),(m,m)"
        return cast(
            list[Variable],
            Blockwise(Eigh(lower=lower, driver=driver), signature=signature)(a),
        )

    # Generalized eigenproblem always uses divide-and-conquer
    signature = "(m,m),(m,m)->(m),(m,m)"
    return cast(
        list[Variable],
        Blockwise(Eigh(lower=lower, driver="evd"), signature=signature)(a, b),
    )


class Eigvalsh(Op):
    """
    Generalized eigenvalues of a Hermitian positive definite eigensystem.

    """

    __props__ = ("lower",)

    def __init__(self, lower=True):
        assert lower in [True, False]
        self.lower = lower

    def make_node(self, a, b=None):
        a = as_tensor_variable(a)
        assert a.ndim == 2

        if b is None or (isinstance(b, Variable) and isinstance(b.type, NoneTypeT)):
            w = vector(dtype=a.dtype)
            return Apply(self, [a], [w])
        else:
            b = as_tensor_variable(b)
            assert a.ndim == 2
            assert b.ndim == 2

            out_dtype = pytensor.scalar.upcast(a.dtype, b.dtype)
            w = vector(dtype=out_dtype)
            return Apply(self, [a, b], [w])

    def perform(self, node, inputs, outputs):
        (w,) = outputs
        if len(inputs) == 2:
            w[0] = scipy_linalg.eigvalsh(a=inputs[0], b=inputs[1], lower=self.lower)
        else:
            w[0] = scipy_linalg.eigvalsh(a=inputs[0], b=None, lower=self.lower)

    def pullback(self, inputs, outputs, g_outputs):
        a, b = inputs
        (gw,) = g_outputs
        return EigvalshGrad(self.lower)(a, b, gw)

    def infer_shape(self, fgraph, node, shapes):
        n = shapes[0][0]
        return [(n,)]


class EigvalshGrad(Op):
    """
    Gradient of generalized eigenvalues of a Hermitian positive definite
    eigensystem.

    """

    # Note: This Op (EigvalshGrad), should be removed and replaced with a graph
    # of pytensor ops that is constructed directly in Eigvalsh.grad.
    # But this can only be done once scipy.linalg.eigh is available as an Op
    # (currently the Eigh uses numpy.linalg.eigh, which doesn't let you
    # pass the right-hand-side matrix for a generalized eigenproblem.) See the
    # discussion on GitHub at
    # https://github.com/Theano/Theano/pull/1846#discussion-diff-12486764

    __props__ = ("lower",)

    def __init__(self, lower=True):
        assert lower in [True, False]
        self.lower = lower
        if lower:
            self.tri0 = np.tril
            self.tri1 = lambda a: np.triu(a, 1)
        else:
            self.tri0 = np.triu
            self.tri1 = lambda a: np.tril(a, -1)

    def make_node(self, a, b, gw):
        a = as_tensor_variable(a)
        b = as_tensor_variable(b)
        gw = as_tensor_variable(gw)
        assert a.ndim == 2
        assert b.ndim == 2
        assert gw.ndim == 1

        out_dtype = pytensor.scalar.upcast(a.dtype, b.dtype, gw.dtype)
        out1 = matrix(dtype=out_dtype)
        out2 = matrix(dtype=out_dtype)
        return Apply(self, [a, b, gw], [out1, out2])

    def perform(self, node, inputs, outputs):
        (a, b, gw) = inputs
        w, v = scipy_linalg.eigh(a, b, lower=self.lower)
        gA = v.dot(np.diag(gw).dot(v.T))
        gB = -v.dot(np.diag(gw * w).dot(v.T))

        # See EighGrad comments for an explanation of these lines
        out1 = self.tri0(gA) + self.tri1(gA).T
        out2 = self.tri0(gB) + self.tri1(gB).T
        outputs[0][0] = np.asarray(out1, dtype=node.outputs[0].dtype)
        outputs[1][0] = np.asarray(out2, dtype=node.outputs[1].dtype)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0], shapes[1]]


def eigvalsh(a, b, lower=True):
    return Eigvalsh(lower)(a, b)
