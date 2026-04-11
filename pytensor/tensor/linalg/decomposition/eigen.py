from functools import partial

import numpy as np
import scipy.linalg as scipy_linalg

import pytensor
from pytensor import scalar as ps
from pytensor.gradient import DisconnectedType
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor import TensorLike
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.type import Variable, matrix, tensor, vector
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

    def pullback(self, inputs, outputs, output_grads):
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
