import logging
import warnings
from collections.abc import Sequence
from functools import partial, reduce
from typing import cast

import numpy as np
import scipy.linalg as scipy_linalg
from scipy.linalg import get_lapack_funcs

import pytensor
from pytensor import tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor import TensorLike
from pytensor.tensor import basic as ptb
from pytensor.tensor import math as ptm
from pytensor.tensor._linalg.decomposition.lu import pivot_to_permutation
from pytensor.tensor.basic import as_tensor_variable, diagonal
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.type import matrix, tensor
from pytensor.tensor.variable import TensorVariable
from pytensor.utils import unzip


logger = logging.getLogger(__name__)


class SolveBase(Op):
    """Base class for `scipy.linalg` matrix equation solvers."""

    __props__: tuple[str, ...] = (
        "lower",
        "b_ndim",
        "overwrite_a",
        "overwrite_b",
    )

    def __init__(
        self,
        *,
        lower=False,
        b_ndim,
        overwrite_a=False,
        overwrite_b=False,
    ):
        self.lower = lower

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

    def pullback(self, inputs, outputs, output_gradients):
        r"""Reverse-mode gradient updates for matrix solve operation :math:`c = A^{-1} b`.

        Symbolic expression for updates taken from [#]_.

        References
        ----------
        .. [#] M. B. Giles, "An extended collection of matrix derivative results
          for forward and reverse mode automatic differentiation",
          http://eprints.maths.ox.ac.uk/1079/

        """
        A, _b = inputs

        c = outputs[0]
        # C is a scalar representing the entire graph
        # `output_gradients` is (dC/dc,)
        # We need to return (dC/d[inv(A)], dC/db)
        c_bar = output_gradients[0]

        props_dict = self._props_dict()
        props_dict["lower"] = not self.lower

        solve_op = type(self)(**props_dict)

        b_bar = solve_op(A.mT, c_bar)
        # force outer product if vector second input
        A_bar = -ptm.outer(b_bar, c) if c.ndim == 1 else -b_bar.dot(c.T)

        if props_dict.get("unit_diagonal", False):
            n = A_bar.shape[-1]
            A_bar = A_bar[pt.arange(n), pt.arange(n)].set(pt.zeros(n))

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


def _lu_solve(
    LU: TensorLike,  # noqa: F811
    pivots: TensorLike,
    b: TensorLike,
    trans: bool = False,
    b_ndim: int | None = None,
):
    b_ndim = _default_b_ndim(b, b_ndim)

    LU, pivots, b = map(pt.as_tensor_variable, [LU, pivots, b])

    inv_permutation = pivot_to_permutation(pivots, inverse=True)
    x = b[inv_permutation] if not trans else b
    # TODO: Use PermuteRows on b
    # x = permute_rows(b, pivots) if not trans else b

    x = solve_triangular(
        LU,
        x,
        lower=not trans,
        unit_diagonal=not trans,
        trans=trans,
        b_ndim=b_ndim,
    )

    x = solve_triangular(
        LU,
        x,
        lower=trans,
        unit_diagonal=trans,
        trans=trans,
        b_ndim=b_ndim,
    )

    # TODO: Use PermuteRows(inverse=True) on x
    # if trans:
    #     x = permute_rows(x, pivots, inverse=True)
    x = x[pt.argsort(inv_permutation)] if trans else x
    return x


def lu_solve(
    LU_and_pivots: tuple[TensorLike, TensorLike],
    b: TensorLike,
    trans: bool = False,
    b_ndim: int | None = None,
    check_finite: bool = True,
    overwrite_b: bool = False,
):
    """
    Solve a system of linear equations given the LU decomposition of the matrix.

    Parameters
    ----------
    LU_and_pivots: tuple[TensorLike, TensorLike]
        LU decomposition of the matrix, as returned by `lu_factor`
    b: TensorLike
        Right-hand side of the equation
    trans: bool
        If True, solve A^T x = b, instead of Ax = b. Default is False
    b_ndim: int, optional
        The number of core dimensions in b. Used to distinguish between a batch of vectors (b_ndim=1) and a matrix
        of vectors (b_ndim=2). Default is None, which will infer the number of core dimensions from the input.
    check_finite: bool
        Unused by PyTensor. PyTensor will return nan if the operation fails.
    overwrite_b: bool
        Ignored by Pytensor. Pytensor will always compute inplace when possible.
    """
    b_ndim = _default_b_ndim(b, b_ndim)
    if b_ndim == 1:
        signature = "(m,m),(m),(m)->(m)"
    else:
        signature = "(m,m),(m),(m,n)->(m,n)"
    partialled_func = partial(_lu_solve, trans=trans, b_ndim=b_ndim)
    return pt.vectorize(partialled_func, signature=signature)(*LU_and_pivots, b)


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


class Solve(SolveBase):
    """
    Solve a system of linear equations.
    """

    __props__ = (
        "assume_a",
        "lower",
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
        try:
            outputs[0][0] = scipy_linalg.solve(
                a=a,
                b=b,
                lower=self.lower,
                check_finite=False,
                assume_a=self.assume_a,
                overwrite_a=self.overwrite_a,
                overwrite_b=self.overwrite_b,
            )
        except np.linalg.LinAlgError:
            outputs[0][0] = np.full(a.shape, np.nan, dtype=a.dtype)

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
    check_finite : bool
        Unused by PyTensor. PyTensor returns nan if the operation fails.
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
            assume_a=assume_a,
            b_ndim=b_ndim,
        )
    )(a, b)


class Expm(Op):
    """
    Compute the matrix exponential of a square array.
    """

    __props__ = ()
    gufunc_signature = "(m,m)->(m,m)"

    def make_node(self, A):
        A = as_tensor_variable(A)
        assert A.ndim == 2

        expm = matrix(dtype=A.dtype, shape=A.type.shape)

        return Apply(self, [A], [expm])

    def perform(self, node, inputs, outputs):
        (A,) = inputs
        (expm,) = outputs
        expm[0] = scipy_linalg.expm(A)

    def pullback(self, inputs, outputs, output_grads):
        # Kalbfleisch and Lawless, J. Am. Stat. Assoc. 80 (1985) Equation 3.4
        # Kind of... You need to do some algebra from there to arrive at
        # this expression.
        (A,) = inputs
        (_,) = outputs  # Outputs not used; included for signature consistency only
        (A_bar,) = output_grads

        w, V = pt.linalg.eig(A)

        exp_w = pt.exp(w)
        numer = pt.sub.outer(exp_w, exp_w)
        denom = pt.sub.outer(w, w)

        # When w_i ≈ w_j, we have a removable singularity in the expression for X, because
        # lim b->a (e^a - e^b) / (a - b) = e^a (derivation left for the motivated reader)
        X = pt.where(pt.abs(denom) < 1e-8, exp_w, numer / denom)

        diag_idx = pt.arange(w.shape[0])
        X = X[..., diag_idx, diag_idx].set(exp_w)

        inner = solve(V, A_bar.T @ V).T
        result = solve(V.T, inner * X) @ V.T

        # At this point, result is always a complex dtype. If the input was real, the output should be
        # real as well (and all the imaginary parts are numerical noise)
        if A.dtype not in ("complex64", "complex128"):
            return [result.real]

        return [result]

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]


expm = Blockwise(Expm())


def _largest_common_dtype(tensors: Sequence[TensorVariable]) -> np.dtype:
    return reduce(lambda l, r: np.promote_types(l, r), [x.dtype for x in tensors])


class BaseBlockDiagonal(Op):
    __props__: tuple[str, ...] = ("n_inputs",)

    def __init__(self, n_inputs):
        input_sig = ",".join(f"(m{i},n{i})" for i in range(n_inputs))
        self.gufunc_signature = f"{input_sig}->(m,n)"

        if n_inputs <= 1:
            raise ValueError("n_inputs must be greater than 1")
        self.n_inputs = n_inputs

    def pullback(self, inputs, outputs, gout):
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
        first, second = unzip(shapes, n=2, strict=True)
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

        shapes_by_dim = tuple(zip(*(m.type.shape for m in matrices)))
        out_shape = tuple(
            [
                sum(dim_shapes)
                if not any(shape is None for shape in dim_shapes)
                else None
                for dim_shapes in shapes_by_dim
            ]
        )

        out_type = pytensor.tensor.matrix(shape=out_shape, dtype=dtype)
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
    if len(matrices) == 1:
        return matrices[0]

    _block_diagonal_matrix = Blockwise(BlockDiagonal(n_inputs=len(matrices)))
    return _block_diagonal_matrix(*matrices)


_deprecated_names = {
    "solve_continuous_lyapunov",
    "solve_discrete_are",
    "solve_discrete_lyapunov",
}


def __getattr__(name):
    if name in _deprecated_names:
        warnings.warn(
            f"{name} has been moved from tensor/slinalg.py as part of a reorganization "
            "of linear algebra routines in Pytensor. Imports from slinalg.py will fail in Pytensor 3.0.\n"
            f"Please use the stable user-facing linalg API: from pytensor.tensor.linalg import {name}",
            DeprecationWarning,
            stacklevel=2,
        )
        from pytensor.tensor._linalg.solve import linear_control

        return getattr(linear_control, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


# Re-exports: decomposition ops that were moved to _linalg/decomposition/
# These are kept here for backwards compatibility.
from pytensor.tensor._linalg.decomposition.cholesky import (  # noqa: E402, F401
    Cholesky,
    cholesky,
)
from pytensor.tensor._linalg.decomposition.eigen import (  # noqa: E402, F401
    Eigvalsh,
    EigvalshGrad,
    eigvalsh,
)
from pytensor.tensor._linalg.decomposition.lu import (  # noqa: E402, F401
    LU,
    LUFactor,
    PivotToPermutations,
    lu,
    lu_factor,
)
from pytensor.tensor._linalg.decomposition.qr import QR, qr  # noqa: E402, F401
from pytensor.tensor._linalg.decomposition.schur import (  # noqa: E402, F401
    QZ,
    Schur,
    ordqz,
    qz,
    schur,
)


__all__ = [
    "block_diag",
    "cho_solve",
    "cholesky",
    "eigvalsh",
    "expm",
    "lu",
    "lu_factor",
    "lu_solve",
    "ordqz",
    "pivot_to_permutation",
    "qr",
    "qz",
    "schur",
    "solve",
    "solve_triangular",
]
