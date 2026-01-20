from functools import wraps
from warnings import warn

import numpy as np
import scipy.sparse as scipy_sparse

import pytensor.scalar as ps
import pytensor.sparse.basic as psb
import pytensor.tensor.basic as ptb
import pytensor.tensor.math as ptm
from pytensor import config
from pytensor.gradient import grad_not_implemented
from pytensor.graph import Apply, Op
from pytensor.link.c.op import COp
from pytensor.sparse.type import SparseTensorType
from pytensor.tensor.shape import specify_broadcastable
from pytensor.tensor.type import TensorType, Variable, complex_dtypes, tensor


def structured_elemwise(tensor_op):
    """
    A decorator to create structured element-wise operations on sparse matrices.

    An operation is called "structured" if it operates only on the non-zeros elements of a sparse matrix.
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args):
            x = psb.as_sparse_variable(args[0])
            assert x.format in ("csr", "csc")

            xs = [ps.as_scalar(arg) for arg in args[1:]]
            data, ind, ptr, _shape = psb.csm_properties(x)
            data = tensor_op(data, *xs)

            return psb.CSM(x.format)(data, ind, ptr, _shape)

        return wrapper

    return decorator


@structured_elemwise(ptm.abs)
def abs(x):
    """
    Compute abs(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.sigmoid)
def structured_sigmoid(x):
    """
    Compute sigmoid(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.exp)
def structured_exp(x):
    """
    Compute exp(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.log)
def structured_log(x):
    """
    Compute log(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.pow)
def structured_pow(x, y):
    """
    Compute x**y for all non-zero elements of x.
    """


@structured_elemwise(ptm.minimum)
def structured_minimum(x, y):
    """
    Compute min(x, y) for all non-zero elements of x, where y is a scalar.
    """


@structured_elemwise(ptm.maximum)
def structured_maximum(x, y):
    """
    Compute max(x, y) for all non-zero elements of x, where y is a scalar.
    """


@structured_elemwise(ptm.add)
def structured_add(x, y):
    """
    Compute x + y for all non-zero elements of x, where y is a scalar.
    """


@structured_elemwise(ptm.sin)
def sin(x):
    """
    Compute sin(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.sinh)
def sinh(x):
    """
    Compute sinh(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.arcsin)
def arcsin(x):
    """
    Compute arcsin(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.arcsinh)
def arcsinh(x):
    """
    Compute arcsinh(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.tan)
def tan(x):
    """
    Compute tan(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.tanh)
def tanh(x):
    """
    Compute tanh(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.arctan)
def arctan(x):
    """
    Compute arctan(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.arctanh)
def arctanh(x):
    """
    Compute arctanh(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.round_half_to_even)
def rint(x):
    """
    Compute round_half_to_even(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.sign)
def sign(x):
    """
    Compute sign(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.ceil)
def ceil(x):
    """
    Compute ceil(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.floor)
def floor(x):
    """
    Compute floor(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.log1p)
def log1p(x):
    """
    Compute log(1 + x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.expm1)
def expm1(x):
    """
    Compute exp(x) - 1 for all non-zero elements of x.
    """


@structured_elemwise(ptm.deg2rad)
def deg2rad(x):
    """
    Convert degrees to radians for all non-zero elements of x.
    """


@structured_elemwise(ptm.rad2deg)
def rad2deg(x):
    """
    Convert radians to degrees for all non-zero elements of x.
    """


@structured_elemwise(ptm.trunc)
def trunc(x):
    """
    Truncate the decimal part of x for all non-zero elements of x.
    """


@structured_elemwise(ptm.sqr)
def sqr(x):
    """
    Compute sqr(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.sqrt)
def sqrt(x):
    """
    Compute sqrt(x) for all non-zero elements of x.
    """


@structured_elemwise(ptm.conjugate)
def _conj(x):
    """
    Compute the complex conjugate of x for all non-zero elements of x.
    """


def conjugate(x):
    _x = psb.as_sparse_variable(x)
    if _x.type.dtype not in complex_dtypes:
        return _x
    return _conj(_x)


structured_conjugate = conj = conjugate


class SpSum(Op):
    """

    WARNING: judgement call...
    We are not using the structured in the comparison or hashing
    because it doesn't change the perform method therefore, we
    *do* want Sums with different structured values to be merged
    by the merge optimization and this requires them to compare equal.
    """

    __props__ = ("axis",)

    def __init__(self, axis=None, sparse_grad=True):
        super().__init__()
        self.axis = axis
        self.structured = sparse_grad
        if self.axis not in (None, 0, 1):
            raise ValueError("Illegal value for self.axis.")

    def make_node(self, x):
        x = psb.as_sparse_variable(x)
        assert x.format in ("csr", "csc")

        if self.axis is not None:
            out_shape = (None,)
        else:
            out_shape = ()

        z = TensorType(dtype=x.dtype, shape=out_shape)()
        return Apply(self, [x], [z])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        if self.axis is None:
            z[0] = np.asarray(x.sum())
        else:
            z[0] = np.asarray(x.sum(self.axis)).ravel()

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.dtype not in psb.continuous_dtypes:
            return [x.zeros_like(dtype=config.floatX)]
        if self.structured:
            if self.axis is None:
                r = gz * psb.sp_ones_like(x)
            elif self.axis == 0:
                r = psb.col_scale(psb.sp_ones_like(x), gz)
            elif self.axis == 1:
                r = psb.row_scale(psb.sp_ones_like(x), gz)
            else:
                raise ValueError("Illegal value for self.axis.")
        else:
            o_format = x.format
            x = psb.dense_from_sparse(x)
            if psb._is_sparse_variable(gz):
                gz = psb.dense_from_sparse(gz)
            if self.axis is None:
                r = ptb.second(x, gz)
            else:
                ones = ptb.ones_like(x)
                if self.axis == 0:
                    r = specify_broadcastable(gz.dimshuffle("x", 0), 0) * ones
                elif self.axis == 1:
                    r = specify_broadcastable(gz.dimshuffle(0, "x"), 1) * ones
                else:
                    raise ValueError("Illegal value for self.axis.")
            r = psb.SparseFromDense(o_format)(r)
        return [r]

    def infer_shape(self, fgraph, node, shapes):
        r = None
        if self.axis is None:
            r = [()]
        elif self.axis == 0:
            r = [(shapes[0][1],)]
        else:
            r = [(shapes[0][0],)]
        return r

    def __str__(self):
        return f"{self.__class__.__name__}{{axis={self.axis}}}"


def sp_sum(x, axis=None, sparse_grad=False):
    """
    Calculate the sum of a sparse matrix along the specified axis.

    It operates a reduction along the specified axis. When `axis` is `None`,
    it is applied along all axes.

    Parameters
    ----------
    x
        Sparse matrix.
    axis
        Axis along which the sum is applied. Integer or `None`.
    sparse_grad : bool
        `True` to have a structured grad.

    Returns
    -------
    object
        The sum of `x` in a dense format.

    Notes
    -----
    The grad implementation is controlled with the `sparse_grad` parameter.
    `True` will provide a structured grad and `False` will provide a regular
    grad. For both choices, the grad returns a sparse matrix having the same
    format as `x`.

    This op does not return a sparse matrix, but a dense tensor matrix.

    """

    return SpSum(axis, sparse_grad)(x)


class AddSS(Op):
    # add(sparse, sparse).
    # see the doc of add() for more detail.
    __props__ = ()

    def make_node(self, x, y):
        x, y = map(psb.as_sparse_variable, [x, y])
        assert x.format in ("csr", "csc")
        assert y.format in ("csr", "csc")
        out_dtype = ps.upcast(x.type.dtype, y.type.dtype)
        return Apply(
            self,
            [x, y],
            [SparseTensorType(dtype=out_dtype, format=x.type.format)()],
        )

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert psb._is_sparse(x) and psb._is_sparse(y)
        assert x.shape == y.shape
        out[0] = x + y

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        assert psb._is_sparse_variable(x) and psb._is_sparse_variable(y)
        assert psb._is_sparse_variable(gz)
        return gz, gz

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]


add_s_s = AddSS()


class AddSSData(Op):
    """Add two sparse matrices assuming they have the same sparsity pattern.

    Notes
    -----
    The grad implemented is structured.

    """

    __props__ = ()

    def make_node(self, x, y):
        """

        Parameters
        ----------
        x
            Sparse matrix.
        y
            Sparse matrix.

        Notes
        -----
        `x` and `y` are assumed to have the same sparsity pattern.

        """
        x, y = map(psb.as_sparse_variable, [x, y])
        assert x.format in ("csr", "csc")
        assert y.format in ("csr", "csc")
        if x.type.dtype != y.type.dtype:
            raise NotImplementedError()
        if x.type.format != y.type.format:
            raise NotImplementedError()
        return Apply(
            self,
            [x, y],
            [SparseTensorType(dtype=x.type.dtype, format=x.type.format)()],
        )

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert psb._is_sparse(x) and psb._is_sparse(y)
        assert x.shape == y.shape
        assert x.data.shape == y.data.shape
        out[0] = x.copy()
        out[0].data += y.data

    def grad(self, inputs, gout):
        (gz,) = gout
        is_continuous = [(i.dtype in psb.continuous_dtypes) for i in inputs]
        derivative = {True: gz, False: None}
        return [derivative[b] for b in is_continuous]

    def infer_shape(self, fgraph, node, ins_shapes):
        return [ins_shapes[0]]


add_s_s_data = AddSSData()


class AddSD(Op):
    # add(sparse, sparse).
    # see the doc of add() for more detail.
    __props__ = ()

    def make_node(self, x, y):
        x, y = psb.as_sparse_variable(x), ptb.as_tensor_variable(y)
        assert x.format in ("csr", "csc")
        out_dtype = ps.upcast(x.type.dtype, y.type.dtype)

        # The magic number two here arises because L{scipy.sparse}
        # objects must be matrices (have dimension 2)
        assert y.type.ndim == 2
        return Apply(
            self,
            [x, y],
            [TensorType(dtype=out_dtype, shape=y.type.shape)()],
        )

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert psb._is_dense(y)

        # The asarray is needed as in some case, this return a
        # numpy.matrixlib.defmatrix.matrix object and not an ndarray.
        out[0] = np.asarray(x + y, dtype=node.outputs[0].type.dtype)

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        assert psb._is_sparse_variable(x) and psb._is_dense_variable(y)
        assert psb._is_dense_variable(gz)
        return psb.sp_ones_like(x) * gz, gz

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[1]]


add_s_d = AddSD()


class StructuredAddSV(Op):
    """Structured addition of a sparse matrix and a dense vector.

    The elements of the vector are only added to the corresponding
    non-zero elements of the sparse matrix. Therefore, this operation
    outputs another sparse matrix.

    Notes
    -----
    The grad implemented is structured since the op is structured.

    """

    __props__ = ()

    def make_node(self, x, y):
        """
        Parameters
        ----------
        x
            Sparse matrix.
        y
            Tensor type vector.

        """
        x = psb.as_sparse_variable(x)
        assert x.format in ("csr", "csc")
        y = ptb.as_tensor_variable(y)

        assert y.type.ndim == 1

        if x.type.dtype != y.type.dtype:
            raise NotImplementedError()
        return Apply(
            self,
            [x, y],
            [SparseTensorType(dtype=x.type.dtype, format=x.type.format)()],
        )

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert psb._is_sparse(x) and not psb._is_sparse(y)
        assert x.shape[1] == y.shape[0]
        out[0] = x.__class__(x + (x.toarray() != 0) * y)

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        assert psb._is_sparse_variable(x) and not psb._is_sparse_variable(y)
        assert psb._is_sparse_variable(gz)
        return gz, sp_sum(gz, axis=0, sparse_grad=True)

    def infer_shape(self, fgraph, node, ins_shapes):
        return [ins_shapes[0]]


structured_add_s_v = StructuredAddSV()


def add(x, y):
    """
    Add two matrices, at least one of which is sparse.

    This method will provide the right op according
    to the inputs.

    Parameters
    ----------
    x
        A matrix variable.
    y
        A matrix variable.

    Returns
    -------
    A sparse matrix
        `x` + `y`

    Notes
    -----
    At least one of `x` and `y` must be a sparse matrix.

    The grad will be structured only when one of the variable will be a dense
    matrix.

    """

    if hasattr(x, "getnnz"):
        x = psb.as_sparse_variable(x)
    if hasattr(y, "getnnz"):
        y = psb.as_sparse_variable(y)
    if not isinstance(x, Variable):
        x = ptb.as_tensor_variable(x)
    if not isinstance(y, Variable):
        y = ptb.as_tensor_variable(y)

    x_is_sparse_variable = psb._is_sparse_variable(x)
    y_is_sparse_variable = psb._is_sparse_variable(y)

    assert x_is_sparse_variable or y_is_sparse_variable
    if x_is_sparse_variable and y_is_sparse_variable:
        return add_s_s(x, y)
    elif x_is_sparse_variable and not y_is_sparse_variable:
        return add_s_d(x, y)
    elif y_is_sparse_variable and not x_is_sparse_variable:
        return add_s_d(y, x)
    else:
        raise NotImplementedError()


def subtract(x, y):
    """
    Subtract two matrices, at least one of which is sparse.

    This method will provide the right op according to the inputs.

    Parameters
    ----------
    x : SparseVariable or TensorVariable
        A matrix variable.
    y : SparseVariable or TensorVariable
        A matrix variable.

    Returns
    -------
    result: SparseVariable
        Result of `x - y`, as a sparse matrix.

    Notes
    -----
    At least one of `x` and `y` must be a sparse matrix.

    The grad will be structured only when one of the variable will be a dense matrix.
    """
    return x + (-y)


def sub(x, y):
    warn(
        "pytensor.sparse.sub is deprecated and will be removed in a future version. Use "
        "pytensor.sparse.subtract instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    return subtract(x, y)


sub.__doc__ = subtract.__doc__


class SparseSparseMultiply(Op):
    # mul(sparse, sparse)
    # See the doc of mul() for more detail
    __props__ = ()

    def make_node(self, x, y):
        x, y = psb.as_sparse_variable(x), psb.as_sparse_variable(y)
        assert x.format in ("csr", "csc")
        assert y.format in ("csr", "csc")
        out_dtype = ps.upcast(x.type.dtype, y.type.dtype)
        return Apply(
            self,
            [x, y],
            [SparseTensorType(dtype=out_dtype, format=x.type.format)()],
        )

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert psb._is_sparse(x) and psb._is_sparse(y)
        assert len(x.shape) == 2
        assert y.shape == x.shape
        # This calls the element-wise multiple
        # x * y calls dot...
        out[0] = x.multiply(y)

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        return y * gz, x * gz

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]


mul_s_s = SparseSparseMultiply()


class SparseDenseMultiply(Op):
    # mul(sparse, dense)
    # See the doc of mul() for more detail
    __props__ = ()

    def make_node(self, x, y):
        x, y = psb.as_sparse_variable(x), ptb.as_tensor_variable(y)

        assert x.format in ("csr", "csc")

        # upcast the tensor. Is the cast of sparse done implemented?
        dtype = ps.upcast(x.type.dtype, y.type.dtype)

        # The magic number two here arises because L{scipy.sparse}
        # objects must be matrices (have dimension 2)
        # Broadcasting of the sparse matrix is not supported.
        # We support nd == 0 used by grad of SpSum()
        if y.type.ndim not in (0, 2):
            raise ValueError(f"y {y} must have 0 or 2 dimensions. Got {y.type.ndim}")
        if y.type.ndim == 0:
            out_shape = x.type.shape
        if y.type.ndim == 2:
            # Combine with static shape information from y
            out_shape = []
            for x_st_dim_length, y_st_dim_length in zip(x.type.shape, y.type.shape):
                if x_st_dim_length is None:
                    out_shape.append(y_st_dim_length)
                else:
                    out_shape.append(x_st_dim_length)
                    # If both are known, they must match
                    if (
                        y_st_dim_length is not None
                        and y_st_dim_length != x_st_dim_length
                    ):
                        raise ValueError(
                            f"Incompatible static shapes {x}: {x.type.shape}, {y}: {y.type.shape}"
                        )
            out_shape = tuple(out_shape)
        out = SparseTensorType(dtype=dtype, format=x.type.format, shape=out_shape)()
        return Apply(self, [x, y], [out])

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        out_dtype = node.outputs[0].dtype
        assert psb._is_sparse(x) and psb._is_dense(y)

        if x.dtype == out_dtype:
            z = x.copy()
        else:
            z = x.astype(out_dtype)
        out[0] = z
        z_data = z.data

        if y.ndim == 0:
            z_data *= y
        else:  # y_ndim == 2
            # if we have enough memory to fit y, maybe we can fit x.asarray()
            # too?
            # TODO: change runtime from O(M*N) to O(nonzeros)
            M, N = x.shape
            assert x.shape == y.shape
            indices = x.indices
            indptr = x.indptr
            if x.format == "csc":
                for j in range(0, N):
                    for i_idx in range(indptr[j], indptr[j + 1]):
                        i = indices[i_idx]
                        z_data[i_idx] *= y[i, j]
            elif x.format == "csr":
                for i in range(0, M):
                    for j_idx in range(indptr[i], indptr[i + 1]):
                        j = indices[j_idx]
                        z_data[j_idx] *= y[i, j]

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        assert psb._is_sparse_variable(x) and psb._is_dense_variable(y)
        assert psb._is_sparse_variable(gz)
        return y * gz, psb.dense_from_sparse(x * gz)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]


mul_s_d = SparseDenseMultiply()


class SparseDenseVectorMultiply(Op):
    """Element-wise multiplication of sparse matrix by a broadcasted dense vector element wise.

    Notes
    -----
    The grad implemented is regular, i.e. not structured.

    """

    # TODO: Merge with the SparseDenseMultiply Op

    __props__ = ()

    def make_node(self, x, y):
        """
        Parameters
        ----------
        x
            Sparse matrix to multiply.
        y
            Tensor broadcastable vector.

        """
        x = psb.as_sparse_variable(x)
        assert x.format in ("csr", "csc")
        y = ptb.as_tensor_variable(y)

        if y.type.ndim != 1:
            raise ValueError(f"y {y} must have 1 dimension. Got {y.type.ndim}")

        if x.type.dtype != y.type.dtype:
            raise NotImplementedError(
                f"Differing dtypes not supported. Got {x.type.dtype} and {y.type.dtype}."
            )
        out_shape = [x.type.shape[0]]
        if x.type.shape[-1] is None:
            out_shape.append(y.type.shape[0])
        else:
            out_shape.append(x.type.shape[-1])
            if y.type.shape[-1] is not None and x.type.shape[-1] != y.type.shape[-1]:
                raise ValueError(
                    f"Incompatible static shapes for multiplication {x}: {x.type.shape}, {y}: {y.type.shape}"
                )
        return Apply(
            self,
            [x, y],
            [
                SparseTensorType(
                    dtype=x.type.dtype, format=x.type.format, shape=tuple(out_shape)
                )()
            ],
        )

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert psb._is_sparse(x) and not psb._is_sparse(y)
        assert x.shape[1] == y.shape[0]
        out[0] = x.__class__(x.toarray() * y)

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        assert psb._is_sparse_variable(x) and psb._is_dense_variable(y)
        assert psb._is_sparse_variable(gz)

        # mul_s_v is not implemented if the types vary

        if gz.dtype == "float64" and y.dtype == "float32":
            y = y.astype("float64")

        if gz.dtype == "float32" and y.dtype == "float64":
            gz = gz.astype("float64")

        return mul_s_v(gz, y), sp_sum(x * gz, axis=0, sparse_grad=True)

    def infer_shape(self, fgraph, node, ins_shapes):
        return [ins_shapes[0]]


mul_s_v = SparseDenseVectorMultiply()


def multiply(x, y):
    """
    Multiply elementwise two matrices, at least one of which is sparse.

    This method will provide the right op according to the inputs.

    Parameters
    ----------
    x : SparseTensorType
        A matrix variable.
    y : SparseTensorType
        A matrix variable.

    Returns
    -------
    result: SparseTensorType
        The elementwise multiplication of `x` and `y`.

    Notes
    -----
    At least one of `x` and `y` must be a sparse matrix.

    The gradient is regular, i.e. not structured.
    """

    x = psb.as_sparse_or_tensor_variable(x)
    y = psb.as_sparse_or_tensor_variable(y)

    x_is_sparse_variable = psb._is_sparse_variable(x)
    y_is_sparse_variable = psb._is_sparse_variable(y)

    assert x_is_sparse_variable or y_is_sparse_variable
    if x_is_sparse_variable and y_is_sparse_variable:
        # mul_s_s is not implemented if the types differ
        if y.dtype == "float64" and x.dtype == "float32":
            x = x.astype("float64")
        return mul_s_s(x, y)
    elif x_is_sparse_variable or y_is_sparse_variable:
        if y_is_sparse_variable:
            x, y = y, x
        # mul is unimplemented if the dtypes differ
        if y.dtype == "float64" and x.dtype == "float32":
            x = x.astype("float64")
        if y.ndim == 1:
            return mul_s_v(x, y)
        else:
            return mul_s_d(x, y)
    else:
        raise NotImplementedError()


def mul(x, y):
    warn(
        "pytensor.sparse.mul is deprecated and will be removed in a future version. Use "
        "pytensor.sparse.multiply instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    return multiply(x, y)


mul.__doc__ = multiply.__doc__


class __ComparisonOpSS(Op):
    """
    Used as a superclass for all comparisons between two sparses matrices.

    Parameters
    ----------
    x
        First compared sparse matrix.
    y
        Second compared sparse matrix

    Returns
    -------
    object
        Comparison(x,y)

    """

    __props__ = ()

    # Function to override
    def comparison(self, x, y):
        raise NotImplementedError()

    def make_node(self, x, y):
        x = psb.as_sparse_variable(x)
        y = psb.as_sparse_variable(y)

        if x.type.format != y.type.format:
            raise NotImplementedError()
        return Apply(
            self, [x, y], [SparseTensorType(dtype="uint8", format=x.type.format)()]
        )

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert psb._is_sparse(x) and psb._is_sparse(y)
        assert x.shape == y.shape
        # FIXME: Scipy csc > csc outputs csr format, but make_node assumes it will be the same as inputs
        # Casting to respect make_node, but this is very inefficient
        # TODO: Why not go with default bool?
        out[0] = (
            self.comparison(x, y).astype("uint8").asformat(node.outputs[0].type.format)
        )

    def infer_shape(self, fgraph, node, ins_shapes):
        return [ins_shapes[0]]


class __ComparisonOpSD(Op):
    """
    Used as a superclass for all comparisons between sparse and dense matrix.

    Parameters
    ----------
    x
        Sparse matrix.
    y
        Dense matrix.

    Returns
    -------
    object
        Comparison(x,y)

    """

    __props__ = ()

    # Function to override
    def comparison(self, x, y):
        raise NotImplementedError()

    def make_node(self, x, y):
        x, y = psb.as_sparse_variable(x), ptb.as_tensor_variable(y)

        assert y.type.ndim == 2
        out = TensorType(dtype="uint8", shape=(None, None))()
        return Apply(self, [x, y], [out])

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert psb._is_sparse(x)
        assert x.shape == y.shape
        assert psb._is_dense(y)
        o = self.comparison(x, y).astype("uint8")
        o = np.asarray(o)
        out[0] = o

    def infer_shape(self, fgraph, node, ins_shapes):
        return [ins_shapes[0]]


def __ComparisonSwitch(SS, SD, DS):
    """

    Parameters
    ----------
    SS
        Function to apply between two sparses matrices.
    SD
        Function to apply between a sparse and a dense matrix.
    DS
        Function to apply between a dense and a sparse matrix.

    Returns
    -------
    function
        Switch function taking two matrices as input.

    Notes
    -----
    At least one of `x` and `y` must be a sparse matrix.

    DS swap input as a dense matrix cannot be a left operand.

    """

    def helper(x, y):
        if hasattr(x, "getnnz"):
            x = psb.as_sparse_variable(x)
        if hasattr(y, "getnnz"):
            y = psb.as_sparse_variable(y)
        if not isinstance(x, Variable):
            x = ptb.as_tensor_variable(x)
        if not isinstance(y, Variable):
            y = ptb.as_tensor_variable(y)

        x_is_sparse_variable = psb._is_sparse_variable(x)
        y_is_sparse_variable = psb._is_sparse_variable(y)

        assert x_is_sparse_variable or y_is_sparse_variable
        if x_is_sparse_variable and y_is_sparse_variable:
            return SS(x, y)
        elif x_is_sparse_variable and not y_is_sparse_variable:
            return SD(x, y)
        elif y_is_sparse_variable and not x_is_sparse_variable:
            return DS(y, x)
        else:
            raise NotImplementedError()

    return helper


class EqualSS(__ComparisonOpSS):
    def comparison(self, x, y):
        return x == y


equal_s_s = EqualSS()


class EqualSD(__ComparisonOpSD):
    def comparison(self, x, y):
        return x == y


equal_s_d = EqualSD()


class NotEqualSS(__ComparisonOpSS):
    def comparison(self, x, y):
        return x != y


not_equal_s_s = NotEqualSS()


class NotEqualSD(__ComparisonOpSD):
    def comparison(self, x, y):
        return x != y


not_equal_s_d = NotEqualSD()


class LessThanSS(__ComparisonOpSS):
    def comparison(self, x, y):
        return x < y


less_than_s_s = LessThanSS()


class LessThanSD(__ComparisonOpSD):
    def comparison(self, x, y):
        return x < y


less_than_s_d = LessThanSD()


class GreaterThanSS(__ComparisonOpSS):
    def comparison(self, x, y):
        return x > y


greater_than_s_s = GreaterThanSS()


class GreaterThanSD(__ComparisonOpSD):
    def comparison(self, x, y):
        return x > y


greater_than_s_d = GreaterThanSD()


class LessEqualSS(__ComparisonOpSS):
    def comparison(self, x, y):
        return x <= y


less_equal_s_s = LessEqualSS()


class LessEqualSD(__ComparisonOpSD):
    def comparison(self, x, y):
        return x <= y


less_equal_s_d = LessEqualSD()


class GreaterEqualSS(__ComparisonOpSS):
    def comparison(self, x, y):
        return x >= y


greater_equal_s_s = GreaterEqualSS()


class GreaterEqualSD(__ComparisonOpSD):
    def comparison(self, x, y):
        return x >= y


greater_equal_s_d = GreaterEqualSD()

eq = __ComparisonSwitch(equal_s_s, equal_s_d, equal_s_d)

neq = __ComparisonSwitch(not_equal_s_s, not_equal_s_d, not_equal_s_d)

lt = __ComparisonSwitch(less_than_s_s, less_than_s_d, greater_than_s_d)

gt = __ComparisonSwitch(greater_than_s_s, greater_than_s_d, less_than_s_d)

le = __ComparisonSwitch(less_equal_s_s, less_equal_s_d, greater_equal_s_d)

ge = __ComparisonSwitch(greater_equal_s_s, greater_equal_s_d, less_equal_s_d)


class TrueDot(Op):
    # TODO
    # Simplify code by splitting into DotSS and DotSD.

    __props__ = ()

    # The grad_preserves_dense attribute doesn't change the
    # execution behavior.  To let the optimizer merge nodes with
    # different values of this attribute we shouldn't compare it
    # here.

    def __init__(self, grad_preserves_dense=True):
        self.grad_preserves_dense = grad_preserves_dense

    def make_node(self, x, y):
        # NOTE
        # Because of trickiness of implementing,
        # we assume that the left argument x is a
        # SparseVariable (not dense)

        if x.type.dtype != y.type.dtype:
            raise NotImplementedError()

        if not psb._is_sparse_variable(x):
            raise TypeError(x)

        # These are the conversions performed by scipy.sparse.dot
        if x.type.format == "csc" or x.type.format == "coo":
            myformat = "csc"
        elif x.type.format == "csr":
            myformat = "csr"
        else:
            raise NotImplementedError()

        inputs = [x, y]  # Need to convert? e.g. assparse
        outputs = [SparseTensorType(dtype=x.type.dtype, format=myformat)()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inp, out_):
        # TODO
        # -Verify that output is sufficiently sparse,
        #  and raise a warning if it is not.
        # -Also determine that we are storing the
        #  output in the best storage format?

        x, y = inp
        (out,) = out_
        rval = x.dot(y)
        if not scipy_sparse.issparse(rval):
            rval = getattr(scipy_sparse, x.format + "_matrix")(rval)
        # x.dot call tocsr() that will "upcast" to ['int8', 'uint8', 'short',
        # 'ushort', 'intc', 'uintc', 'longlong', 'ulonglong', 'single',
        # 'double', 'longdouble', 'csingle', 'cdouble', 'clongdouble']
        # But ulonglong is uint64 on x86-64, but with a different typenum!
        if rval.dtype.num != np.dtype(str(rval.dtype)).num:
            assert str(rval.dtype) == node.outputs[0].dtype
            # Create a view with the expected typenum.
            format = node.outputs[0].type.format
            data = rval.data.view(dtype=node.outputs[0].dtype)
            indices = rval.indices
            indptr = rval.indptr
            _shape = rval.shape
            # No need to copy indices and indptr as in CSM.perform(),
            # as there is only one user of them.
            if format == "csc":
                rval = scipy_sparse.csc_matrix(
                    (data, indices, indptr), _shape, copy=False
                )
            else:
                assert format == "csr"
                rval = scipy_sparse.csr_matrix(
                    (data, indices, indptr), _shape, copy=False
                )
        out[0] = rval

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        assert psb._is_sparse_variable(gz)
        assert psb._is_sparse_variable(x)

        rval = [true_dot(gz, y.T), true_dot(x.T, gz)]
        if psb._is_dense_variable(y):
            if self.grad_preserves_dense:
                rval[1] = psb.dense_from_sparse(rval[1])
        return rval

    def infer_shape(self, fgraph, node, shapes):
        return [(shapes[0][0], shapes[1][1])]


def true_dot(x, y, grad_preserves_dense=True):
    """
    Operation for efficiently calculating the dot product when
    one or all operands are sparse. Supported formats are CSC and CSR.
    The output of the operation is sparse.

    Parameters
    ----------
    x
        Sparse matrix.
    y
        Sparse matrix or 2d tensor variable.
    grad_preserves_dense : bool
        If True (default), makes the grad of dense inputs dense.
        Otherwise the grad is always sparse.

    Returns
    -------
    The dot product `x`.`y` in a sparse format.

    Notex
    -----
    The grad implemented is regular, i.e. not structured.

    """
    # TODO
    # Maybe the triple-transposition formulation
    # (when x is dense) is slow. See if there is a
    # direct way to do this.

    if hasattr(x, "getnnz"):
        x = psb.as_sparse_variable(x)
        assert x.format in ("csr", "csc")
    if hasattr(y, "getnnz"):
        y = psb.as_sparse_variable(y)
        assert y.format in ("csr", "csc")

    x_is_sparse_variable = psb._is_sparse_variable(x)
    y_is_sparse_variable = psb._is_sparse_variable(y)

    if not x_is_sparse_variable and not y_is_sparse_variable:
        raise TypeError()
    if x_is_sparse_variable:
        return TrueDot(grad_preserves_dense)(x, y)
    else:
        assert y_is_sparse_variable
        return psb.transpose(TrueDot(grad_preserves_dense)(y.T, x.T))


class StructuredDot(Op):
    __props__ = ()

    def make_node(self, a, b):
        a = psb.as_sparse_variable(a)
        assert a.format in ("csr", "csc", "bsr")

        if not psb._is_sparse_variable(a):
            raise TypeError(
                "First argument must be of type SparseVariable or SparseConstant"
            )
        dtype_out = ps.upcast(a.type.dtype, b.type.dtype)
        if b.type.ndim != 2:
            raise NotImplementedError("non-matrix b")

        if psb._is_sparse_variable(b):
            return Apply(self, [a, b], [SparseTensorType(a.type.format, dtype_out)()])
        else:
            return Apply(
                self,
                [a, b],
                [
                    tensor(
                        dtype=dtype_out,
                        shape=(None, 1 if b.type.shape[1] == 1 else None),
                    )
                ],
            )

    def perform(self, node, inputs, outputs):
        (a, b) = inputs
        (out,) = outputs
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                "shape mismatch in StructuredDot.perform", (a.shape, b.shape)
            )

        # Multiplication of objects of `*_matrix` type means dot product
        # The result can be sparse or dense, depending on the inputs.
        variable = a * b
        if isinstance(node.outputs[0].type, SparseTensorType):
            assert psb._is_sparse(variable)
            out[0] = variable
            return

        assert psb._is_dense(variable)  # scipy 0.7 automatically converts to dense

        # dot of an NxM sparse matrix, with a Mx1 dense matrix, returns vector
        # not matrix
        if variable.ndim == 1:
            variable = np.expand_dims(variable, 1)
        elif variable.ndim != 2:
            raise Exception("Output of structured dot should be a matrix (ndim=2)")

        assert variable.ndim == 2

        if variable.shape != (a.shape[0], b.shape[1]):
            raise Exception(
                f"a.shape={a.shape}, b.shape={b.shape}, variable.shape={variable.shape}?"
            )

        # The cast is needed as otherwise we hit the bug mentioned into
        # _asarray function documentation.
        out[0] = np.asarray(variable, str(variable.dtype))

    def grad(self, inputs, gout):
        # a is sparse, b is dense, g_out is dense
        # ga = g_out x b.T
        # gb = a.T x g_out
        (a, b) = inputs
        (g_out,) = gout
        return [structured_dot_grad(a, b, g_out), structured_dot(a.T, g_out)]

    def infer_shape(self, fgraph, node, shapes):
        return [(shapes[0][0], shapes[1][1])]


_structured_dot = StructuredDot()


def structured_dot(x, y):
    """
    Structured Dot is like dot, except that only the gradient wrt non-zero elements of the sparse matrix
    `a` are calculated and propagated.

    The output is presumed to be a dense matrix, and is represented by a TensorType instance.

    Parameters
    ----------
    a
        A sparse matrix.
    b
        A sparse or dense matrix.

    Returns
    -------
    A sparse matrix
        The dot product of `a` and `b`.

    Notes
    -----
    The grad implemented is structured.

    """

    # @todo: Maybe the triple-transposition formulation (when x is dense)
    # is slow. See if there is a direct way to do this.
    # (JB 20090528: Transposing tensors and sparse matrices is constant-time,
    # inplace, and fast.)

    if hasattr(x, "getnnz"):
        x = psb.as_sparse_variable(x)
        assert x.format in ("csr", "csc")
    if hasattr(y, "getnnz"):
        y = psb.as_sparse_variable(y)
        assert y.format in ("csr", "csc")

    x_is_sparse_variable = psb._is_sparse_variable(x)
    y_is_sparse_variable = psb._is_sparse_variable(y)
    if not x_is_sparse_variable and not y_is_sparse_variable:
        raise TypeError("structured_dot requires at least one sparse argument")

    if x_is_sparse_variable:
        return _structured_dot(x, y)
    else:
        assert y_is_sparse_variable
        return _structured_dot(y.T, x.T).T


class StructuredDotGradCSC(COp):
    # Op that produces the grad of StructuredDot.

    # :param a_indices: Matrix indices
    # :param a_indptr: Matrix indptr
    # :param b: Right operand
    # :param g_ab: Accumulated gradient.

    # :return: The grad of `a`.`b` for `a` accumulated
    #          with g_ab.

    # :note: The grad implemented is structured.
    # :note: a_* are the corresponding properties of a sparse
    #        matrix in csc format.
    __props__ = ()

    def make_node(self, a_indices, a_indptr, b, g_ab):
        return Apply(
            self,
            [a_indices, a_indptr, b, g_ab],
            [tensor(dtype=g_ab.dtype, shape=(None,))],
        )

    def perform(self, node, inputs, outputs):
        (a_indices, a_indptr, b, g_ab) = inputs
        (out,) = outputs
        g_a_data = np.zeros(a_indices.shape, dtype=g_ab.dtype)
        for j in range(len(a_indptr) - 1):
            ind0 = a_indptr[j]
            ind1 = a_indptr[j + 1]
            for i_idx in range(ind0, ind1):
                i = a_indices[i_idx]
                # Depending on the type of g_ab and b (sparse or dense),
                # the following dot product can result in a scalar or
                # a (1, 1) sparse matrix.
                dot_val = np.dot(g_ab[i], b[j].T)
                if isinstance(dot_val, scipy_sparse.spmatrix):
                    dot_val = dot_val[0, 0]
                g_a_data[i_idx] = dot_val
        out[0] = g_a_data

    def c_code_cache_version(self):
        return (2,)

    def c_code(self, node, name, inputs, outputs, sub):
        (_indices, _indptr, _d, _g) = inputs
        (_zout,) = outputs
        if node.inputs[2].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for b")
        if node.inputs[3].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for g_ab")

        fail = sub["fail"]
        return f"""
        if (PyArray_NDIM({_d}) != 2) {{PyErr_SetString(PyExc_NotImplementedError, "rank(d) != 2"); {fail};}}
        if (PyArray_NDIM({_g}) != 2) {{PyErr_SetString(PyExc_NotImplementedError, "rank(g) != 2"); {fail};}}
        if (PyArray_NDIM({_indices}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1"); {fail};}}
        if (PyArray_NDIM({_indptr}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1"); {fail};}}

        if( PyArray_TYPE({_indices}) != NPY_INT32) {{
        PyErr_SetString(PyExc_NotImplementedError, "C"); {fail};}}

        if( PyArray_TYPE({_indptr}) != NPY_INT32)
        {{PyErr_SetString(PyExc_NotImplementedError, "D"); {fail};}}

        if( PyArray_DIMS({_d})[1] != PyArray_DIMS({_g})[1])
        {{PyErr_SetString(PyExc_NotImplementedError, "d and g have different numbers of columns"); {fail};}}

        if (!{_zout}
            || (PyArray_DIMS({_zout})[0] != PyArray_DIMS({_indices})[0]))
        {{
            Py_XDECREF({_zout});
            {_zout} = (PyArrayObject*) PyArray_SimpleNew(1, PyArray_DIMS({_indices}), PyArray_TYPE({_g}));
        }}

        {{   //makes it compile even though labels jump over variable definitions.
            npy_intp nnz = PyArray_DIMS({_indices})[0];
            npy_intp N =  PyArray_DIMS({_indptr})[0]-1; //TODO: error checking with this

            npy_intp Sindices = PyArray_STRIDES({_indices})[0]/PyArray_ITEMSIZE({_indices});
            npy_intp Sindptr = PyArray_STRIDES({_indptr})[0]/PyArray_ITEMSIZE({_indptr});

            const npy_intp Sd1 = PyArray_STRIDES({_d})[1]/PyArray_ITEMSIZE({_d});
            const npy_intp Sg1 = PyArray_STRIDES({_g})[1]/PyArray_ITEMSIZE({_g});

            const npy_intp K = PyArray_DIMS({_d})[1];

            const npy_int32 * __restrict__ indptr = (npy_int32 *)PyArray_DATA({_indptr});
            const npy_int32 * __restrict__ indices = (npy_int32 *)PyArray_DATA({_indices});

            // loop over columns
            for (npy_int32 j = 0; j < N; ++j)
            {{
                // extract j-th row of dense matrix
                const dtype_{_d}* __restrict__ d_row = (dtype_{_d}*)(PyArray_BYTES({_d}) + PyArray_STRIDES({_d})[0] * j);
                if(j >= PyArray_DIMS({_d})[0]) {{PyErr_SetString(PyExc_NotImplementedError, "G"); {fail};}}

                // for each non-null value in the sparse column
                for (npy_int32 i_idx = indptr[j * Sindptr]; i_idx < indptr[(j+1) * Sindptr]; ++i_idx)
                {{
                    // extract row index of non-null value
                    npy_int32 i = indices[i_idx * Sindices];

                    // extract corresponding row in gradient
                    const dtype_{_g}* __restrict__ g_row = (dtype_{_g}*)(PyArray_BYTES({_g}) + PyArray_STRIDES({_g})[0] * i);
                    double ip = 0.0;

                    // make sure that row index is not bigger than actual number of rows
                    // Note: wouldn't the above operation fail if that were the case ?
                    //       when would this ever be true anyway ?
                    if (i >= PyArray_DIMS({_g})[0])
                    {{PyErr_SetString(PyExc_NotImplementedError, "H"); {fail};}}

                    // perform dot product of dense and sparse rows
                    for(int k = 0; k < K; ++k)
                    {{
                        ip += d_row[k * Sd1] * g_row[k*Sg1];
                    }}

                    // write resulting gradient to sparse output
                    ((dtype_{_zout}* __restrict__)(PyArray_BYTES({_zout}) + i_idx * PyArray_STRIDES({_zout})[0]))[0] = ip;
                }}
            }}
        }}

        """

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]


sdg_csc = StructuredDotGradCSC()


class StructuredDotGradCSR(COp):
    # Op that produces the grad of StructuredDot.

    # :param a_indices: Matrix indices
    # :param a_indptr: Matrix indptr
    # :param b: Right operand
    # :param g_ab: Accumulated gradient.

    # :return: The grad of `a`.`b` for `a` accumulated
    #          with g_ab.

    # :note: The grad implemented is structured.
    # :note: a_* are the corresponding properties of a sparse
    #        matrix in csr format.
    __props__ = ()

    def make_node(self, a_indices, a_indptr, b, g_ab):
        return Apply(
            self, [a_indices, a_indptr, b, g_ab], [tensor(dtype=b.dtype, shape=(None,))]
        )

    def perform(self, node, inputs, outputs):
        (a_indices, a_indptr, b, g_ab) = inputs
        (out,) = outputs
        g_a_data = np.zeros(a_indices.shape, dtype=g_ab.dtype)
        for i in range(len(a_indptr) - 1):  # loop over rows
            ind0 = a_indptr[i]
            ind1 = a_indptr[i + 1]
            # loop over values in that row (columns)
            for j_idx in range(ind0, ind1):
                j = a_indices[j_idx]
                # grad is dot product of i-th row of gradient with j-th row of b
                # Depending on the type of g_ab and b (sparse or dense),
                # the following dot product can result in a scalar or
                # a (1, 1) sparse matrix.
                dot_val = np.dot(g_ab[i], b[j].T)
                if isinstance(dot_val, scipy_sparse.spmatrix):
                    dot_val = dot_val[0, 0]
                g_a_data[j_idx] = dot_val
        out[0] = g_a_data

    def c_code_cache_version(self):
        return (2,)

    def c_code(self, node, name, inputs, outputs, sub):
        (_indices, _indptr, _d, _g) = inputs
        (_zout,) = outputs
        if node.inputs[2].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for b")
        if node.inputs[3].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for g_ab")

        fail = sub["fail"]
        return f"""
        if (PyArray_NDIM({_d}) != 2) {{PyErr_SetString(PyExc_NotImplementedError, "rank(d) != 2"); {fail};}}
        if (PyArray_NDIM({_g}) != 2) {{PyErr_SetString(PyExc_NotImplementedError, "rank(g) != 2"); {fail};}}
        if (PyArray_NDIM({_indices}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1"); {fail};}}
        if (PyArray_NDIM({_indptr}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1"); {fail};}}

        if( PyArray_TYPE({_indices}) != NPY_INT32) {{
        PyErr_SetString(PyExc_NotImplementedError, "C"); {fail};}}

        if( PyArray_TYPE({_indptr}) != NPY_INT32)
        {{PyErr_SetString(PyExc_NotImplementedError, "D"); {fail};}}

        if( PyArray_DIMS({_d})[1] != PyArray_DIMS({_g})[1])
        {{PyErr_SetString(PyExc_NotImplementedError, "d and g have different numbers of columns"); {fail};}}

        if (!{_zout}
            || (PyArray_DIMS({_zout})[0] != PyArray_DIMS({_indices})[0]))
        {{
            Py_XDECREF({_zout});
            {_zout} = (PyArrayObject*) PyArray_SimpleNew(1, PyArray_DIMS({_indices}), PyArray_TYPE({_g}));
        }}

        {{   //makes it compile even though labels jump over variable definitions.
            npy_intp nnz = PyArray_DIMS({_indices})[0];
            // extract number of rows
            npy_intp N =  PyArray_DIMS({_indptr})[0]-1; //TODO: error checking with this

            npy_intp Sindices = PyArray_STRIDES({_indices})[0]/PyArray_ITEMSIZE({_indices});
            npy_intp Sindptr = PyArray_STRIDES({_indptr})[0]/PyArray_ITEMSIZE({_indptr});

            const npy_intp Sd1 = PyArray_STRIDES({_d})[1]/PyArray_ITEMSIZE({_d});
            const npy_intp Sg1 = PyArray_STRIDES({_g})[1]/PyArray_ITEMSIZE({_g});

            const npy_intp K = PyArray_DIMS({_d})[1];

            const npy_int32 * __restrict__ indptr = (npy_int32 *)PyArray_DATA({_indptr});
            const npy_int32 * __restrict__ indices = (npy_int32 *)PyArray_DATA({_indices});

            // loop over columns of sparse matrix
            for (npy_int32 i = 0; i < N; ++i)
            {{
                // for each non-null value in the sparse row
                for (npy_int32 j_idx = indptr[i * Sindptr]; j_idx < indptr[(i+1) * Sindptr]; ++j_idx)
                {{
                    // extract column index of non-null value
                    npy_int32 j = indices[j_idx * Sindices];

                    // extract j-th row of dense matrix
                    const dtype_{_d}* __restrict__ d_row = (dtype_{_d}*)(PyArray_BYTES({_d}) + PyArray_STRIDES({_d})[0] * j);
                    if(j >= PyArray_DIMS({_d})[0]) {{PyErr_SetString(PyExc_NotImplementedError, "G"); {fail};}}

                    // extract corresponding row in gradient
                    const dtype_{_g}* __restrict__ g_row = (dtype_{_g}*)(PyArray_BYTES({_g}) + PyArray_STRIDES({_g})[0] * i);
                    double ip = 0.0;

                    // make sure that row index is not bigger than actual number of rows
                    // Note: wouldn't the above operation fail if that were the case ?
                    //       when would this ever be true anyway ?
                    if (i >= PyArray_DIMS({_g})[0])
                    {{PyErr_SetString(PyExc_NotImplementedError, "H"); {fail};}}

                    // perform dot product of dense and sparse rows
                    for(int k = 0; k < K; ++k)
                    {{
                        ip += d_row[k * Sd1] * g_row[k*Sg1];
                    }}

                    // write resulting gradient to sparse output
                    ((dtype_{_zout}* __restrict__)(PyArray_BYTES({_zout}) + j_idx * PyArray_STRIDES({_zout})[0]))[0] = ip;
                }}
            }}
        }}

        """

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]


sdg_csr = StructuredDotGradCSR()


def structured_dot_grad(sparse_A, dense_B, ga):
    if sparse_A.type.format in ("csc", "csr"):
        if sparse_A.type.format == "csc":
            sdgcsx = sdg_csc
            CSx = psb.CSC
        else:
            sdgcsx = sdg_csr
            CSx = psb.CSR

        g_A_data = sdgcsx(
            psb.csm_indices(sparse_A), psb.csm_indptr(sparse_A), dense_B, ga
        )
        return CSx(
            g_A_data,
            psb.csm_indices(sparse_A),
            psb.csm_indptr(sparse_A),
            psb.csm_shape(sparse_A),
        )
    else:
        raise NotImplementedError()


class SamplingDot(Op):
    """Compute the dot product ``dot(x, y.T) = z`` for only a subset of `z`.

    This is equivalent to ``p * (x . y.T)`` where ``*`` is the element-wise
    product, ``x`` and ``y`` operands of the dot product and ``p`` is a matrix that
    contains 1 when the corresponding element of ``z`` should be calculated
    and ``0`` when it shouldn't. Note that `SamplingDot` has a different interface
    than `dot` because it requires ``x`` to be a ``m x k`` matrix while
    ``y`` is a ``n x k`` matrix instead of the usual ``k x n`` matrix.

    Notes
    -----
    It will work if the pattern is not binary value, but if the
    pattern doesn't have a high sparsity proportion it will be slower
    then a more optimized dot followed by a normal elemwise
    multiplication.

    The grad implemented is regular, i.e. not structured.

    """

    __props__ = ()

    def make_node(self, x, y, p):
        """
        Parameters
        ----------
        x
            Tensor matrix.
        y
            Tensor matrix.
        p
            Sparse matrix in csr format.

        """
        x = ptb.as_tensor_variable(x)
        y = ptb.as_tensor_variable(y)
        p = psb.as_sparse_variable(p)
        assert p.format in ("csr", "csc")

        if not psb._is_sparse_variable(p):
            raise TypeError(p)

        # TODO: use it.
        # dtype_out = ps.upcast(x.type.dtype, y.type.dtype, p.type.dtype)

        return Apply(self, [x, y, p], [p.type()])

    def perform(self, node, inputs, outputs):
        (x, y, p) = inputs
        (out,) = outputs
        if psb._is_sparse(x):
            raise TypeError(x)

        if psb._is_sparse(y):
            raise TypeError(y)

        if not psb._is_sparse(p):
            raise TypeError(p)

        out[0] = p.__class__(p.multiply(np.dot(x, y.T)))

    def grad(self, inputs, gout):
        (x, y, p) = inputs
        (gz,) = gout
        rval = [dot(p * gz, y), dot((p * gz).T, x), grad_not_implemented(self, 2, p)]

        return rval

    def infer_shape(self, fgraph, node, ins_shapes):
        return [ins_shapes[2]]


sampling_dot = SamplingDot()


class Dot(Op):
    __props__ = ()

    def __str__(self):
        return "Sparse" + self.__class__.__name__

    def infer_shape(self, fgraph, node, shapes):
        xshp, yshp = shapes
        x, y = node.inputs
        if x.ndim == 2 and y.ndim == 2:
            return [(xshp[0], yshp[1])]
        if x.ndim == 1 and y.ndim == 2:
            return [(yshp[1],)]
        if x.ndim == 2 and y.ndim == 1:
            return [(xshp[0],)]
        if x.ndim == 1 and y.ndim == 1:
            return [()]
        raise NotImplementedError()

    def make_node(self, x, y):
        dtype_out = ps.upcast(x.dtype, y.dtype)

        # Sparse dot product should have at least one sparse variable
        # as input. If the other one is not sparse, it has to be converted
        # into a tensor.
        if isinstance(x, scipy_sparse.spmatrix):
            x = psb.as_sparse_variable(x)
        if isinstance(y, scipy_sparse.spmatrix):
            y = psb.as_sparse_variable(y)

        x_is_sparse_var = psb._is_sparse_variable(x)
        y_is_sparse_var = psb._is_sparse_variable(y)

        if not x_is_sparse_var and not y_is_sparse_var:
            raise TypeError(
                "Sparse dot product should have at least one "
                "sparse variable as inputs, but the inputs are "
                f"{x} ({x.type}) and {y} ({y.type})."
            )

        if x_is_sparse_var:
            shape_x = (None,) * x.type.ndim
        else:
            x = ptb.as_tensor_variable(x)
            shape_x = x.type.shape
            assert y.format in ("csr", "csc")
            if x.ndim not in (1, 2):
                raise TypeError(
                    "Input 0 (0-indexed) must have ndim of "
                    f"1 or 2, {int(x.type.ndim)} given."
                )

        if y_is_sparse_var:
            shape_y = (None,) * y.type.ndim
        else:
            y = ptb.as_tensor_variable(y)
            shape_y = y.type.shape
            assert x.format in ("csr", "csc")
            if y.ndim not in (1, 2):
                raise TypeError(
                    "Input 1 (1-indexed) must have ndim of "
                    f"1 or 2, {int(y.type.ndim)} given."
                )

        if len(shape_y) == 2:
            shape_out = shape_x[:-1] + shape_y[1:]
        elif len(shape_y) == 1:
            shape_out = shape_x[:-1]

        return Apply(self, [x, y], [tensor(dtype=dtype_out, shape=shape_out)])

    def perform(self, node, inputs, out):
        x, y = inputs
        out = out[0]
        x_is_sparse = psb._is_sparse(x)
        y_is_sparse = psb._is_sparse(y)

        if not x_is_sparse and not y_is_sparse:
            raise TypeError(x)

        # Multiplication of objects of `*_matrix` type means dot product
        rval = x * y

        if x_is_sparse and y_is_sparse:
            rval = rval.toarray()

        out[0] = np.asarray(rval, dtype=node.outputs[0].dtype)

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        assert psb._is_sparse_variable(x) or psb._is_sparse_variable(y)
        rval = []

        if psb._is_dense_variable(y):
            rval.append(ptm.dot(gz, y.T))
        else:
            rval.append(dot(gz, y.T))
        if psb._is_dense_variable(x):
            rval.append(ptm.dot(x.T, gz))
        else:
            rval.append(dot(x.T, gz))

        return rval


_dot = Dot()


def dot(x, y):
    """Efficiently compute the dot product when one or all operands are sparse.

    Supported formats are CSC and CSR.  The output of the operation is dense.

    Parameters
    ----------
    x
        Sparse or dense matrix variable.
    y
        Sparse or dense matrix variable.

    Returns
    -------
    The dot product ``x @ y`` in a dense format.

    Notes
    -----
    The grad implemented is regular, i.e. not structured.

    At least one of `x` or `y` must be a sparse matrix.

    When the operation has the form ``dot(csr_matrix, dense)``
    the gradient of this operation can be performed inplace
    by `UsmmCscDense`. This leads to significant speed-ups.

    """

    if hasattr(x, "getnnz"):
        x = psb.as_sparse_variable(x)
    if hasattr(y, "getnnz"):
        y = psb.as_sparse_variable(y)

    x_is_sparse_variable = psb._is_sparse_variable(x)
    y_is_sparse_variable = psb._is_sparse_variable(y)

    if not x_is_sparse_variable and not y_is_sparse_variable:
        raise TypeError()

    return _dot(x, y)


class Usmm(Op):
    """Computes the dense matrix resulting from ``alpha * x @ y + z``.

    Notes
    -----
    At least one of `x` or `y` must be a sparse matrix.

    """

    __props__ = ()

    def __str__(self):
        return "Usmm{no_inplace}"

    def make_node(self, alpha, x, y, z):
        """

        Parameters
        ----------
        alpha
            A scalar.
        x
            Matrix variable.
        y
            Matrix variable.
        z
            Dense matrix.

        """
        if not psb._is_sparse_variable(x) and not psb._is_sparse_variable(y):
            # If x and y are tensor, we don't want to use this class
            # We should use Dot22 and Gemm in that case.
            raise TypeError(x)

        dtype_out = ps.upcast(
            alpha.type.dtype, x.type.dtype, y.type.dtype, z.type.dtype
        )
        alpha = ptb.as_tensor_variable(alpha)
        z = ptb.as_tensor_variable(z)

        assert z.type.ndim == 2
        assert alpha.type.shape == (1,) * alpha.type.ndim
        if not psb._is_sparse_variable(x):
            x = ptb.as_tensor_variable(x)
            assert y.format in ("csr", "csc")
            assert x.type.ndim == 2
        if not psb._is_sparse_variable(y):
            y = ptb.as_tensor_variable(y)
            assert x.format in ("csr", "csc")
            assert y.type.ndim == 2

        return Apply(
            self,
            [alpha, x, y, z],
            [tensor(dtype=dtype_out, shape=(None, None))],
        )

    def perform(self, node, inputs, outputs):
        (alpha, x, y, z) = inputs
        (out,) = outputs
        x_is_sparse = psb._is_sparse(x)
        y_is_sparse = psb._is_sparse(y)

        if not x_is_sparse and not y_is_sparse:
            raise TypeError(x)

        rval = x * y
        if isinstance(rval, scipy_sparse.spmatrix):
            rval = rval.toarray()
        if rval.dtype == alpha.dtype:
            rval *= alpha  # Faster because operation is inplace
        else:
            rval = rval * alpha
        if rval.dtype == z.dtype:
            rval += z  # Faster because operation is inplace
        else:
            rval = rval + z

        out[0] = rval


usmm = Usmm()
