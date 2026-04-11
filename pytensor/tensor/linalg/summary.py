import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor import TensorLike
from pytensor.tensor import basic as ptb
from pytensor.tensor import math as ptm
from pytensor.tensor.basic import as_tensor_variable, diagonal
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg.decomposition.svd import svd
from pytensor.tensor.type import scalar


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

    def pullback(self, inputs, outputs, g_outputs):
        from pytensor.tensor.linalg.inverse import matrix_inverse

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


def _multi_svd_norm(
    x: ptb.TensorVariable, row_axis: int, col_axis: int, reduce_op: Callable
):
    """Compute a function of the singular values of the 2-D matrices in `x`.

    This is a private utility function used by `pytensor.tensor.linalg.norm()`.

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
