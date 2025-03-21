import logging
import warnings
from collections.abc import Sequence
from functools import reduce
from typing import Literal, cast

import numpy as np
import scipy.linalg as scipy_linalg
from numpy.exceptions import ComplexWarning

import pytensor
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor import TensorLike, as_tensor_variable
from pytensor.tensor import basic as ptb
from pytensor.tensor import math as ptm
from pytensor.tensor.basic import diagonal
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.nlinalg import kron, matrix_dot
from pytensor.tensor.shape import reshape
from pytensor.tensor.type import matrix, tensor, vector
from pytensor.tensor.variable import TensorVariable


logger = logging.getLogger(__name__)


class Cholesky(Op):
    # TODO: LAPACK wrapper with in-place behavior, for solve also

    __props__ = ("lower", "check_finite", "on_error", "overwrite_a")
    gufunc_signature = "(m,m)->(m,m)"

    def __init__(
        self,
        *,
        lower: bool = True,
        check_finite: bool = True,
        on_error: Literal["raise", "nan"] = "raise",
        overwrite_a: bool = False,
    ):
        self.lower = lower
        self.check_finite = check_finite
        if on_error not in ("raise", "nan"):
            raise ValueError('on_error must be one of "raise" or ""nan"')
        self.on_error = on_error
        self.overwrite_a = overwrite_a

        if self.overwrite_a:
            self.destroy_map = {0: [0]}

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]

    def make_node(self, x):
        x = as_tensor_variable(x)
        if x.type.ndim != 2:
            raise TypeError(
                f"Cholesky only allowed on matrix (2-D) inputs, got {x.type.ndim}-D input"
            )
        # Call scipy to find output dtype
        dtype = scipy_linalg.cholesky(np.eye(1, dtype=x.type.dtype)).dtype
        return Apply(self, [x], [tensor(shape=x.type.shape, dtype=dtype)])

    def perform(self, node, inputs, outputs):
        [x] = inputs
        [out] = outputs
        try:
            # Scipy cholesky only makes use of overwrite_a when it is F_CONTIGUOUS
            # If we have a `C_CONTIGUOUS` array we transpose to benefit from it
            if self.overwrite_a and x.flags["C_CONTIGUOUS"]:
                out[0] = scipy_linalg.cholesky(
                    x.T,
                    lower=not self.lower,
                    check_finite=self.check_finite,
                    overwrite_a=True,
                ).T
            else:
                out[0] = scipy_linalg.cholesky(
                    x,
                    lower=self.lower,
                    check_finite=self.check_finite,
                    overwrite_a=self.overwrite_a,
                )

        except scipy_linalg.LinAlgError:
            if self.on_error == "raise":
                raise
            else:
                out[0] = np.full(x.shape, np.nan, dtype=node.outputs[0].type.dtype)

    def L_op(self, inputs, outputs, gradients):
        """
        Cholesky decomposition reverse-mode gradient update.

        Symbolic expression for reverse-mode Cholesky gradient taken from [#]_

        References
        ----------
        .. [#] I. Murray, "Differentiation of the Cholesky decomposition",
           http://arxiv.org/abs/1602.07527

        """

        dz = gradients[0]
        chol_x = outputs[0]

        # Replace the cholesky decomposition with 1 if there are nans
        # or solve_upper_triangular will throw a ValueError.
        if self.on_error == "nan":
            ok = ~ptm.any(ptm.isnan(chol_x))
            chol_x = ptb.switch(ok, chol_x, 1)
            dz = ptb.switch(ok, dz, 1)

        # deal with upper triangular by converting to lower triangular
        if not self.lower:
            chol_x = chol_x.T
            dz = dz.T

        def tril_and_halve_diagonal(mtx):
            """Extracts lower triangle of square matrix and halves diagonal."""
            return ptb.tril(mtx) - ptb.diag(ptb.diagonal(mtx) / 2.0)

        def conjugate_solve_triangular(outer, inner):
            """Computes L^{-T} P L^{-1} for lower-triangular L."""
            solve_upper = SolveTriangular(lower=False, b_ndim=2)
            return solve_upper(outer.T, solve_upper(outer.T, inner.T).T)

        s = conjugate_solve_triangular(
            chol_x, tril_and_halve_diagonal(chol_x.T.dot(dz))
        )

        if self.lower:
            grad = ptb.tril(s + s.T) - ptb.diag(ptb.diagonal(s))
        else:
            grad = ptb.triu(s + s.T) - ptb.diag(ptb.diagonal(s))

        if self.on_error == "nan":
            return [ptb.switch(ok, grad, np.nan)]
        else:
            return [grad]

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        if not allowed_inplace_inputs:
            return self
        new_props = self._props_dict()  # type: ignore
        new_props["overwrite_a"] = True
        return type(self)(**new_props)


def cholesky(
    x: "TensorLike",
    lower: bool = True,
    *,
    check_finite: bool = False,
    overwrite_a: bool = False,
    on_error: Literal["raise", "nan"] = "raise",
):
    """
    Return a triangular matrix square root of positive semi-definite `x`.

    L = cholesky(X, lower=True) implies dot(L, L.T) == X.

    Parameters
    ----------
    x: tensor_like
    lower : bool, default=True
        Whether to return the lower or upper cholesky factor
    check_finite : bool, default=False
        Whether to check that the input matrix contains only finite numbers.
    overwrite_a: bool, ignored
        Whether to use the same memory for the output as `a`. This argument is ignored, and is present here only
        for consistency with scipy.linalg.cholesky.
    on_error : ['raise', 'nan']
        If on_error is set to 'raise', this Op will raise a `scipy.linalg.LinAlgError` if the matrix is not positive definite.
        If on_error is set to 'nan', it will return a matrix containing nans instead.

    Returns
    -------
    TensorVariable
        Lower or upper triangular Cholesky factor of `x`

    Example
    -------
    .. testcode::

        import pytensor
        import pytensor.tensor as pt
        import numpy as np

        x = pt.tensor('x', shape=(5, 5), dtype='float64')
        L = pt.linalg.cholesky(x)

        f = pytensor.function([x], L)
        x_value = np.random.normal(size=(5, 5))
        x_value = x_value @ x_value.T # Ensures x is positive definite
        L_value = f(x_value)
        assert np.allclose(L_value @ L_value.T, x_value)

    """

    return Blockwise(Cholesky(lower=lower, on_error=on_error))(x)


class SolveBase(Op):
    """Base class for `scipy.linalg` matrix equation solvers."""

    __props__: tuple[str, ...] = (
        "lower",
        "check_finite",
        "b_ndim",
        "overwrite_a",
        "overwrite_b",
    )

    def __init__(
        self,
        *,
        lower=False,
        check_finite=True,
        b_ndim,
        overwrite_a=False,
        overwrite_b=False,
    ):
        self.lower = lower
        self.check_finite = check_finite
        assert b_ndim in (1, 2)
        self.b_ndim = b_ndim
        if b_ndim == 1:
            self.gufunc_signature = "(m,m),(m)->(m)"
        else:
            self.gufunc_signature = "(m,m),(m,n)->(m,n)"
        self.overwrite_a = overwrite_a
        self.overwrite_b = overwrite_b
        destroy_map = {}
        if self.overwrite_a and self.overwrite_b:
            # An output destroying two inputs is not yet supported
            # destroy_map[0] = [0, 1]
            raise NotImplementedError(
                "It's not yet possible to overwrite_a and overwrite_b simultaneously"
            )
        elif self.overwrite_a:
            destroy_map[0] = [0]
        elif self.overwrite_b:
            destroy_map[0] = [1]
        self.destroy_map = destroy_map

    def perform(self, node, inputs, outputs):
        raise NotImplementedError(
            "SolveBase should be subclassed with an perform method"
        )

    def make_node(self, A, b):
        A = as_tensor_variable(A)
        b = as_tensor_variable(b)

        if A.ndim != 2:
            raise ValueError(f"`A` must be a matrix; got {A.type} instead.")
        if b.ndim != self.b_ndim:
            raise ValueError(f"`b` must have {self.b_ndim} dims; got {b.type} instead.")

        # Infer dtype by solving the most simple case with 1x1 matrices
        o_dtype = scipy_linalg.solve(
            np.ones((1, 1), dtype=A.dtype),
            np.ones((1,), dtype=b.dtype),
        ).dtype
        x = tensor(dtype=o_dtype, shape=b.type.shape)
        return Apply(self, [A, b], [x])

    def infer_shape(self, fgraph, node, shapes):
        Ashape, Bshape = shapes
        rows = Ashape[1]
        if len(Bshape) == 1:
            return [(rows,)]
        else:
            cols = Bshape[1]
            return [(rows, cols)]

    def L_op(self, inputs, outputs, output_gradients):
        r"""Reverse-mode gradient updates for matrix solve operation :math:`c = A^{-1} b`.

        Symbolic expression for updates taken from [#]_.

        References
        ----------
        .. [#] M. B. Giles, "An extended collection of matrix derivative results
          for forward and reverse mode automatic differentiation",
          http://eprints.maths.ox.ac.uk/1079/

        """
        A, b = inputs

        c = outputs[0]
        # C is a scalar representing the entire graph
        # `output_gradients` is (dC/dc,)
        # We need to return (dC/d[inv(A)], dC/db)
        c_bar = output_gradients[0]

        props_dict = self._props_dict()
        props_dict["lower"] = not self.lower

        solve_op = type(self)(**props_dict)

        b_bar = solve_op(A.T, c_bar)
        # force outer product if vector second input
        A_bar = -ptm.outer(b_bar, c) if c.ndim == 1 else -b_bar.dot(c.T)

        return [A_bar, b_bar]


def _default_b_ndim(b, b_ndim):
    if b_ndim is not None:
        assert b_ndim in (1, 2)
        return b_ndim

    b = as_tensor_variable(b)
    if b_ndim is None:
        return min(b.ndim, 2)  # By default, assume the core case is a matrix


class CholeskySolve(SolveBase):
    __props__ = (
        "lower",
        "check_finite",
        "b_ndim",
        "overwrite_b",
    )

    def __init__(self, **kwargs):
        if kwargs.get("overwrite_a", False):
            raise ValueError("overwrite_a is not supported for CholeskySolve")
        kwargs.setdefault("lower", True)
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
        C, b = inputs
        rval = scipy_linalg.cho_solve(
            (C, self.lower),
            b,
            check_finite=self.check_finite,
            overwrite_b=self.overwrite_b,
        )

        output_storage[0][0] = rval

    def L_op(self, *args, **kwargs):
        # TODO: Base impl should work, let's try it
        raise NotImplementedError()

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        if 1 in allowed_inplace_inputs:
            new_props = self._props_dict()  # type: ignore
            new_props["overwrite_b"] = True
            return type(self)(**new_props)
        else:
            return self


def cho_solve(c_and_lower, b, *, check_finite=True, b_ndim: int | None = None):
    """Solve the linear equations A x = b, given the Cholesky factorization of A.

    Parameters
    ----------
    (c, lower) : tuple, (array, bool)
        Cholesky factorization of a, as given by cho_factor
    b : array
        Right-hand side
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    b_ndim : int
        Whether the core case of b is a vector (1) or matrix (2).
        This will influence how batched dimensions are interpreted.
    """
    A, lower = c_and_lower
    b_ndim = _default_b_ndim(b, b_ndim)
    return Blockwise(
        CholeskySolve(lower=lower, check_finite=check_finite, b_ndim=b_ndim)
    )(A, b)


class SolveTriangular(SolveBase):
    """Solve a system of linear equations."""

    __props__ = (
        "unit_diagonal",
        "lower",
        "check_finite",
        "b_ndim",
        "overwrite_b",
    )

    def __init__(self, *, unit_diagonal=False, **kwargs):
        if kwargs.get("overwrite_a", False):
            raise ValueError("overwrite_a is not supported for SolverTriangulare")
        super().__init__(**kwargs)
        self.unit_diagonal = unit_diagonal

    def perform(self, node, inputs, outputs):
        A, b = inputs
        outputs[0][0] = scipy_linalg.solve_triangular(
            A,
            b,
            lower=self.lower,
            trans=0,
            unit_diagonal=self.unit_diagonal,
            check_finite=self.check_finite,
            overwrite_b=self.overwrite_b,
        )

    def L_op(self, inputs, outputs, output_gradients):
        res = super().L_op(inputs, outputs, output_gradients)

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
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
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
            check_finite=check_finite,
            b_ndim=b_ndim,
        )
    )(a, b)
    return cast(TensorVariable, ret)


class Solve(SolveBase):
    """
    Solve a system of linear equations.
    """

    __props__ = (
        "assume_a",
        "lower",
        "check_finite",
        "b_ndim",
        "overwrite_a",
        "overwrite_b",
    )

    def __init__(self, *, assume_a="gen", **kwargs):
        # Triangular and diagonal are handled outside of Solve
        valid_options = ["gen", "sym", "her", "pos", "tridiagonal", "banded"]

        assume_a = assume_a.lower()
        # We use the old names as the different dispatches are more likely to support them
        long_to_short = {
            "general": "gen",
            "symmetric": "sym",
            "hermitian": "her",
            "positive definite": "pos",
        }
        assume_a = long_to_short.get(assume_a, assume_a)

        if assume_a not in valid_options:
            raise ValueError(
                f"Invalid assume_a: {assume_a}. It must be one of {valid_options} or {list(long_to_short.keys())}"
            )

        if assume_a in ("tridiagonal", "banded"):
            from scipy import __version__ as sp_version

            if tuple(map(int, sp_version.split(".")[:-1])) < (1, 15):
                warnings.warn(
                    f"assume_a={assume_a} requires scipy>=1.5.0. Defaulting to assume_a='gen'.",
                    UserWarning,
                )
                assume_a = "gen"

        super().__init__(**kwargs)
        self.assume_a = assume_a

    def perform(self, node, inputs, outputs):
        a, b = inputs
        outputs[0][0] = scipy_linalg.solve(
            a=a,
            b=b,
            lower=self.lower,
            check_finite=self.check_finite,
            assume_a=self.assume_a,
            overwrite_a=self.overwrite_a,
            overwrite_b=self.overwrite_b,
        )

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        if not allowed_inplace_inputs:
            return self
        new_props = self._props_dict()  # type: ignore
        # PyTensor doesn't allow an output to destroy two inputs yet
        # new_props["overwrite_a"] = 0 in allowed_inplace_inputs
        # new_props["overwrite_b"] = 1 in allowed_inplace_inputs
        if 1 in allowed_inplace_inputs:
            # Give preference to overwrite_b
            new_props["overwrite_b"] = True
        # We can't overwrite_a if we're assuming tridiagonal
        elif not self.assume_a == "tridiagonal":  # allowed inputs == [0]
            new_props["overwrite_a"] = True
        return type(self)(**new_props)


def solve(
    a,
    b,
    *,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: str = "gen",
    transposed: bool = False,
    b_ndim: int | None = None,
):
    """Solves the linear equation set ``a * x = b`` for the unknown ``x`` for square ``a`` matrix.

    If the data matrix is known to be a particular type then supplying the
    corresponding string to ``assume_a`` key chooses the dedicated solver.
    The available options are

    ===================  ================================
     diagonal             'diagonal'
     tridiagonal          'tridiagonal'
     banded               'banded'
     upper triangular     'upper triangular'
     lower triangular     'lower triangular'
     symmetric            'symmetric' (or 'sym')
     hermitian            'hermitian' (or 'her')
     positive definite    'positive definite' (or 'pos')
     general              'general' (or 'gen')
    ===================  ================================

    If omitted, ``'general'`` is the default structure.

    The datatype of the arrays define which solver is called regardless
    of the values. In other words, even when the complex array entries have
    precisely zero imaginary parts, the complex solver will be called based
    on the data type of the array.

    Parameters
    ----------
    a : (..., N, N) array_like
        Square input data
    b : (..., N, NRHS) array_like
        Input data for the right hand side.
    lower : bool, default False
        Ignored unless ``assume_a`` is one of ``'sym'``, ``'her'``, or ``'pos'``.
        If True, the calculation uses only the data in the lower triangle of `a`;
        entries above the diagonal are ignored. If False (default), the
        calculation uses only the data in the upper triangle of `a`; entries
        below the diagonal are ignored.
    overwrite_a : bool
        Unused by PyTensor. PyTensor will always perform the operation in-place if possible.
    overwrite_b : bool
        Unused by PyTensor. PyTensor will always perform the operation in-place if possible.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    assume_a : str, optional
        Valid entries are explained above.
    transposed: bool, default False
        If True, solves the system A^T x = b. Default is False.
    b_ndim : int
        Whether the core case of b is a vector (1) or matrix (2).
        This will influence how batched dimensions are interpreted.
        By default, we assume b_ndim = b.ndim is 2 if b.ndim > 1, else 1.
    """
    assume_a = assume_a.lower()

    if assume_a in ("lower triangular", "upper triangular"):
        lower = "lower" in assume_a
        return solve_triangular(
            a,
            b,
            lower=lower,
            trans=transposed,
            check_finite=check_finite,
            b_ndim=b_ndim,
        )

    b_ndim = _default_b_ndim(b, b_ndim)

    if assume_a == "diagonal":
        a_diagonal = diagonal(a, axis1=-2, axis2=-1)
        b_transposed = b[None, :] if b_ndim == 1 else b.mT
        x = (b_transposed / pt.expand_dims(a_diagonal, -2)).mT
        if b_ndim == 1:
            x = x.squeeze(-1)
        return x

    if transposed:
        a = a.mT
        lower = not lower

    return Blockwise(
        Solve(
            lower=lower,
            check_finite=check_finite,
            assume_a=assume_a,
            b_ndim=b_ndim,
        )
    )(a, b)


class Eigvalsh(Op):
    """
    Generalized eigenvalues of a Hermitian positive definite eigensystem.

    """

    __props__ = ("lower",)

    def __init__(self, lower=True):
        assert lower in [True, False]
        self.lower = lower

    def make_node(self, a, b):
        if b == pytensor.tensor.type_other.NoneConst:
            a = as_tensor_variable(a)
            assert a.ndim == 2

            out_dtype = pytensor.scalar.upcast(a.dtype)
            w = vector(dtype=out_dtype)
            return Apply(self, [a], [w])
        else:
            a = as_tensor_variable(a)
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

    def grad(self, inputs, g_outputs):
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


class Expm(Op):
    """
    Compute the matrix exponential of a square array.

    """

    __props__ = ()

    def make_node(self, A):
        A = as_tensor_variable(A)
        assert A.ndim == 2
        expm = matrix(dtype=A.dtype)
        return Apply(
            self,
            [
                A,
            ],
            [
                expm,
            ],
        )

    def perform(self, node, inputs, outputs):
        (A,) = inputs
        (expm,) = outputs
        expm[0] = scipy_linalg.expm(A)

    def grad(self, inputs, outputs):
        (A,) = inputs
        (g_out,) = outputs
        return [ExpmGrad()(A, g_out)]

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]


class ExpmGrad(Op):
    """
    Gradient of the matrix exponential of a square array.

    """

    __props__ = ()

    def make_node(self, A, gw):
        A = as_tensor_variable(A)
        assert A.ndim == 2
        out = matrix(dtype=A.dtype)
        return Apply(
            self,
            [A, gw],
            [
                out,
            ],
        )

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]

    def perform(self, node, inputs, outputs):
        # Kalbfleisch and Lawless, J. Am. Stat. Assoc. 80 (1985) Equation 3.4
        # Kind of... You need to do some algebra from there to arrive at
        # this expression.
        (A, gA) = inputs
        (out,) = outputs
        w, V = scipy_linalg.eig(A, right=True)
        U = scipy_linalg.inv(V).T

        exp_w = np.exp(w)
        X = np.subtract.outer(exp_w, exp_w) / np.subtract.outer(w, w)
        np.fill_diagonal(X, exp_w)
        Y = U.dot(V.T.dot(gA).dot(U) * X).dot(V.T)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ComplexWarning)
            out[0] = Y.astype(A.dtype)


expm = Expm()


class SolveContinuousLyapunov(Op):
    """
    Solves a continuous Lyapunov equation, :math:`AX + XA^H = B`, for :math:`X.

    Continuous time Lyapunov equations are special cases of Sylvester equations, :math:`AX + XB = C`, and can be solved
    efficiently using the Bartels-Stewart algorithm. For more details, see the docstring for
    scipy.linalg.solve_continuous_lyapunov
    """

    __props__ = ()
    gufunc_signature = "(m,m),(m,m)->(m,m)"

    def make_node(self, A, B):
        A = as_tensor_variable(A)
        B = as_tensor_variable(B)

        out_dtype = pytensor.scalar.upcast(A.dtype, B.dtype)
        X = pytensor.tensor.matrix(dtype=out_dtype)

        return pytensor.graph.basic.Apply(self, [A, B], [X])

    def perform(self, node, inputs, output_storage):
        (A, B) = inputs
        X = output_storage[0]

        out_dtype = node.outputs[0].type.dtype
        X[0] = scipy_linalg.solve_continuous_lyapunov(A, B).astype(out_dtype)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]

    def grad(self, inputs, output_grads):
        # Gradient computations come from Kao and Hennequin (2020), https://arxiv.org/pdf/2011.11430.pdf
        # Note that they write the equation as AX + XA.H + Q = 0, while scipy uses AX + XA^H = Q,
        # so minor adjustments need to be made.
        A, Q = inputs
        (dX,) = output_grads

        X = self(A, Q)
        S = self(A.conj().T, -dX)  # Eq 31, adjusted

        A_bar = S.dot(X.conj().T) + S.conj().T.dot(X)
        Q_bar = -S  # Eq 29, adjusted

        return [A_bar, Q_bar]


_solve_continuous_lyapunov = Blockwise(SolveContinuousLyapunov())


def solve_continuous_lyapunov(A: TensorLike, Q: TensorLike) -> TensorVariable:
    """
    Solve the continuous Lyapunov equation :math:`A X + X A^H + Q = 0`.

    Parameters
    ----------
    A: TensorLike
        Square matrix of shape ``N x N``.
    Q: TensorLike
        Square matrix of shape ``N x N``.

    Returns
    -------
    X: TensorVariable
        Square matrix of shape ``N x N``

    """

    return cast(TensorVariable, _solve_continuous_lyapunov(A, Q))


class BilinearSolveDiscreteLyapunov(Op):
    """
    Solves a discrete lyapunov equation, :math:`AXA^H - X = Q`, for :math:`X.

    The solution is computed by first transforming the discrete-time problem into a continuous-time form. The continuous
    time lyapunov is a special case of a Sylvester equation, and can be efficiently solved. For more details, see the
    docstring for scipy.linalg.solve_discrete_lyapunov
    """

    gufunc_signature = "(m,m),(m,m)->(m,m)"

    def make_node(self, A, B):
        A = as_tensor_variable(A)
        B = as_tensor_variable(B)

        out_dtype = pytensor.scalar.upcast(A.dtype, B.dtype)
        X = pytensor.tensor.matrix(dtype=out_dtype)

        return pytensor.graph.basic.Apply(self, [A, B], [X])

    def perform(self, node, inputs, output_storage):
        (A, B) = inputs
        X = output_storage[0]

        out_dtype = node.outputs[0].type.dtype
        X[0] = scipy_linalg.solve_discrete_lyapunov(A, B, method="bilinear").astype(
            out_dtype
        )

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]

    def grad(self, inputs, output_grads):
        # Gradient computations come from Kao and Hennequin (2020), https://arxiv.org/pdf/2011.11430.pdf
        A, Q = inputs
        (dX,) = output_grads

        X = self(A, Q)

        # Eq 41, note that it is not written as a proper Lyapunov equation
        S = self(A.conj().T, dX)

        A_bar = pytensor.tensor.linalg.matrix_dot(
            S, A, X.conj().T
        ) + pytensor.tensor.linalg.matrix_dot(S.conj().T, A, X)
        Q_bar = S
        return [A_bar, Q_bar]


_bilinear_solve_discrete_lyapunov = Blockwise(BilinearSolveDiscreteLyapunov())


def _direct_solve_discrete_lyapunov(
    A: TensorVariable, Q: TensorVariable
) -> TensorVariable:
    r"""
    Directly solve the discrete Lyapunov equation :math:`A X A^H - X = Q` using the kronecker method of Magnus and
    Neudecker.

    This involves constructing and inverting an intermediate matrix :math:`A \otimes A`, with shape :math:`N^2 x N^2`.
    As a result, this method scales poorly with the size of :math:`N`, and should be avoided for large :math:`N`.
    """

    if A.type.dtype.startswith("complex"):
        AxA = kron(A, A.conj())
    else:
        AxA = kron(A, A)

    eye = pt.eye(AxA.shape[-1])

    vec_Q = Q.ravel()
    vec_X = solve(eye - AxA, vec_Q, b_ndim=1)

    return reshape(vec_X, A.shape)


def solve_discrete_lyapunov(
    A: TensorLike,
    Q: TensorLike,
    method: Literal["direct", "bilinear"] = "bilinear",
) -> TensorVariable:
    """Solve the discrete Lyapunov equation :math:`A X A^H - X = Q`.

    Parameters
    ----------
    A: TensorLike
        Square matrix of shape N x N
    Q: TensorLike
        Square matrix of shape N x N
    method: str, one of ``"direct"`` or ``"bilinear"``
        Solver method used, . ``"direct"`` solves the problem directly via matrix inversion.  This has a pure
        PyTensor implementation and can thus be cross-compiled to supported backends, and should be preferred when
         ``N`` is not large. The direct method scales poorly with the size of ``N``, and the bilinear can be
        used in these cases.

    Returns
    -------
    X: TensorVariable
        Square matrix of shape ``N x N``. Solution to the Lyapunov equation

    """
    if method not in ["direct", "bilinear"]:
        raise ValueError(
            f'Parameter "method" must be one of "direct" or "bilinear", found {method}'
        )

    A = as_tensor_variable(A)
    Q = as_tensor_variable(Q)

    if method == "direct":
        signature = BilinearSolveDiscreteLyapunov.gufunc_signature
        X = pt.vectorize(_direct_solve_discrete_lyapunov, signature=signature)(A, Q)
        return cast(TensorVariable, X)

    elif method == "bilinear":
        return cast(TensorVariable, _bilinear_solve_discrete_lyapunov(A, Q))

    else:
        raise ValueError(f"Unknown method {method}")


class SolveDiscreteARE(Op):
    __props__ = ("enforce_Q_symmetric",)
    gufunc_signature = "(m,m),(m,n),(m,m),(n,n)->(m,m)"

    def __init__(self, enforce_Q_symmetric: bool = False):
        self.enforce_Q_symmetric = enforce_Q_symmetric

    def make_node(self, A, B, Q, R):
        A = as_tensor_variable(A)
        B = as_tensor_variable(B)
        Q = as_tensor_variable(Q)
        R = as_tensor_variable(R)

        out_dtype = pytensor.scalar.upcast(A.dtype, B.dtype, Q.dtype, R.dtype)
        X = pytensor.tensor.matrix(dtype=out_dtype)

        return pytensor.graph.basic.Apply(self, [A, B, Q, R], [X])

    def perform(self, node, inputs, output_storage):
        A, B, Q, R = inputs
        X = output_storage[0]

        if self.enforce_Q_symmetric:
            Q = 0.5 * (Q + Q.T)

        out_dtype = node.outputs[0].type.dtype
        X[0] = scipy_linalg.solve_discrete_are(A, B, Q, R).astype(out_dtype)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]

    def grad(self, inputs, output_grads):
        # Gradient computations come from Kao and Hennequin (2020), https://arxiv.org/pdf/2011.11430.pdf
        A, B, Q, R = inputs

        (dX,) = output_grads
        X = self(A, B, Q, R)

        K_inner = R + matrix_dot(B.T, X, B)

        # K_inner is guaranteed to be symmetric, because X and R are symmetric
        K_inner_inv_BT = solve(K_inner, B.T, assume_a="sym")
        K = matrix_dot(K_inner_inv_BT, X, A)

        A_tilde = A - B.dot(K)

        dX_symm = 0.5 * (dX + dX.T)
        S = solve_discrete_lyapunov(A_tilde, dX_symm)

        A_bar = 2 * matrix_dot(X, A_tilde, S)
        B_bar = -2 * matrix_dot(X, A_tilde, S, K.T)
        Q_bar = S
        R_bar = matrix_dot(K, S, K.T)

        return [A_bar, B_bar, Q_bar, R_bar]


def solve_discrete_are(
    A: TensorLike,
    B: TensorLike,
    Q: TensorLike,
    R: TensorLike,
    enforce_Q_symmetric: bool = False,
) -> TensorVariable:
    """
    Solve the discrete Algebraic Riccati equation :math:`A^TXA - X - (A^TXB)(R + B^TXB)^{-1}(B^TXA) + Q = 0`.

    Discrete-time Algebraic Riccati equations arise in the context of optimal control and filtering problems, as the
    solution to Linear-Quadratic Regulators (LQR), Linear-Quadratic-Guassian (LQG) control problems, and as the
    steady-state covariance of the Kalman Filter.

    Such problems typically have many solutions, but we are generally only interested in the unique *stabilizing*
    solution. This stable solution, if it exists, will be returned by this function.

    Parameters
    ----------
    A: TensorLike
        Square matrix of shape M x M
    B: TensorLike
        Square matrix of shape M x M
    Q: TensorLike
        Symmetric square matrix of shape M x M
    R: TensorLike
        Square matrix of shape N x N
    enforce_Q_symmetric: bool
        If True, the provided Q matrix is transformed to 0.5 * (Q + Q.T) to ensure symmetry

    Returns
    -------
    X: TensorVariable
        Square matrix of shape M x M, representing the solution to the DARE
    """

    return cast(
        TensorVariable, Blockwise(SolveDiscreteARE(enforce_Q_symmetric))(A, B, Q, R)
    )


def _largest_common_dtype(tensors: Sequence[TensorVariable]) -> np.dtype:
    return reduce(lambda l, r: np.promote_types(l, r), [x.dtype for x in tensors])


class BaseBlockDiagonal(Op):
    __props__ = ("n_inputs",)

    def __init__(self, n_inputs):
        input_sig = ",".join(f"(m{i},n{i})" for i in range(n_inputs))
        self.gufunc_signature = f"{input_sig}->(m,n)"

        if n_inputs == 0:
            raise ValueError("n_inputs must be greater than 0")
        self.n_inputs = n_inputs

    def grad(self, inputs, gout):
        shapes = pt.stack([i.shape for i in inputs])
        index_end = shapes.cumsum(0)
        index_begin = index_end - shapes
        slices = [
            ptb.ix_(
                pt.arange(index_begin[i, 0], index_end[i, 0]),
                pt.arange(index_begin[i, 1], index_end[i, 1]),
            )
            for i in range(len(inputs))
        ]
        return [gout[0][slc] for slc in slices]

    def infer_shape(self, fgraph, nodes, shapes):
        first, second = zip(*shapes, strict=True)
        return [(pt.add(*first), pt.add(*second))]

    def _validate_and_prepare_inputs(self, matrices, as_tensor_func):
        if len(matrices) != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} matri{'ces' if self.n_inputs > 1 else 'x'}, got {len(matrices)}"
            )
        matrices = list(map(as_tensor_func, matrices))
        if any(mat.type.ndim != 2 for mat in matrices):
            raise TypeError("All inputs must have dimension 2")
        return matrices


class BlockDiagonal(BaseBlockDiagonal):
    __props__ = ("n_inputs",)

    def make_node(self, *matrices):
        matrices = self._validate_and_prepare_inputs(matrices, pt.as_tensor)
        dtype = _largest_common_dtype(matrices)
        out_type = pytensor.tensor.matrix(dtype=dtype)
        return Apply(self, matrices, [out_type])

    def perform(self, node, inputs, output_storage, params=None):
        dtype = node.outputs[0].type.dtype
        output_storage[0][0] = scipy_linalg.block_diag(*inputs).astype(dtype)


def block_diag(*matrices: TensorVariable):
    """
    Construct a block diagonal matrix from a sequence of input tensors.

    Given the inputs `A`, `B` and `C`, the output will have these arrays arranged on the diagonal:

    [[A, 0, 0],
     [0, B, 0],
     [0, 0, C]]

    Parameters
    ----------
    A, B, C ... : tensors
        Input tensors to form the block diagonal matrix. last two dimensions of the inputs will be used, and all
        inputs should have at least 2 dimensins.

    Returns
    -------
    out: tensor
        The block diagonal matrix formed from the input matrices.

    Examples
    --------
    Create a block diagonal matrix from two 2x2 matrices:

    ..code-block:: python

        import numpy as np
        from pytensor.tensor.linalg import block_diag

        A = pt.as_tensor_variable(np.array([[1, 2], [3, 4]]))
        B = pt.as_tensor_variable(np.array([[5, 6], [7, 8]]))

        result = block_diagonal(A, B, name='X')
        print(result.eval())
        Out: array([[1, 2, 0, 0],
                     [3, 4, 0, 0],
                     [0, 0, 5, 6],
                     [0, 0, 7, 8]])
    """
    _block_diagonal_matrix = Blockwise(BlockDiagonal(n_inputs=len(matrices)))
    return _block_diagonal_matrix(*matrices)


__all__ = [
    "cholesky",
    "solve",
    "eigvalsh",
    "expm",
    "solve_discrete_lyapunov",
    "solve_continuous_lyapunov",
    "solve_discrete_are",
    "solve_triangular",
    "block_diag",
    "cho_solve",
]
