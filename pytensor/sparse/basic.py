"""
Classes for handling sparse matrices.

To read about different sparse formats, see
http://www-users.cs.umn.edu/~saad/software/SPARSKIT/paper.ps

TODO: Automatic methods for determining best sparse format?

"""

from warnings import warn

import numpy as np
import scipy.sparse
from numpy.lib.stride_tricks import as_strided

import pytensor
from pytensor import _as_symbolic, as_symbolic
from pytensor import scalar as ps
from pytensor.configdefaults import config
from pytensor.gradient import DisconnectedType, disconnected_type, grad_undefined
from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.op import Op
from pytensor.link.c.type import generic
from pytensor.sparse.type import SparseTensorType, _is_sparse
from pytensor.tensor import basic as ptb
from pytensor.tensor.basic import Split
from pytensor.tensor.math import minimum
from pytensor.tensor.shape import specify_broadcastable
from pytensor.tensor.type import TensorType, ivector, scalar, tensor, vector
from pytensor.tensor.type import continuous_dtypes as tensor_continuous_dtypes
from pytensor.tensor.type import discrete_dtypes as tensor_discrete_dtypes


sparse_formats = ["csc", "csr"]

"""
Types of sparse matrices to use for testing.

"""
_mtypes = [scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]
# _mtypes = [sparse.csc_matrix, sparse.csr_matrix, sparse.dok_matrix,
# sparse.lil_matrix, sparse.coo_matrix]
# * new class ``dia_matrix`` : the sparse DIAgonal format
# * new class ``bsr_matrix`` : the Block CSR format
_mtype_to_str = {scipy.sparse.csc_matrix: "csc", scipy.sparse.csr_matrix: "csr"}


def _is_sparse_variable(x):
    """

    Returns
    -------
    boolean
        True iff x is a L{SparseVariable} (and not a L{TensorType},
        for instance).

    """
    if not isinstance(x, Variable):
        raise NotImplementedError(
            "this function should only be called on "
            "*variables* (of type sparse.SparseTensorType "
            "or TensorType, for instance), not ",
            x,
        )
    return isinstance(x.type, SparseTensorType)


def _is_dense_variable(x):
    """

    Returns
    -------
    boolean
        True if x is a L{TensorType} (and not a L{SparseVariable},
        for instance).

    """
    if not isinstance(x, Variable):
        raise NotImplementedError(
            "this function should only be called on "
            "*variables* (of type sparse.SparseTensorType or "
            "TensorType, for instance), not ",
            x,
        )
    return isinstance(x.type, TensorType)


def _is_dense(x):
    """

    Returns
    -------
    boolean
        True unless x is a L{scipy.sparse.spmatrix} (and not a
        L{numpy.ndarray}).

    """
    if not isinstance(x, scipy.sparse.spmatrix | np.ndarray):
        raise NotImplementedError(
            "this function should only be called on "
            "sparse.scipy.sparse.spmatrix or "
            "numpy.ndarray, not,",
            x,
        )
    return isinstance(x, np.ndarray)


@_as_symbolic.register(scipy.sparse.spmatrix)
def as_symbolic_sparse(x, **kwargs):
    return as_sparse_variable(x, **kwargs)


def as_sparse_variable(x, name=None, ndim=None, **kwargs):
    """
    Wrapper around SparseVariable constructor to construct
    a Variable with a sparse matrix with the same dtype and
    format.

    Parameters
    ----------
    x
        A sparse matrix.

    Returns
    -------
    object
        SparseVariable version of `x`.

    """

    # TODO
    # Verify that sp is sufficiently sparse, and raise a
    # warning if it is not

    if isinstance(x, Apply):
        if len(x.outputs) != 1:
            raise ValueError(
                "It is ambiguous which output of a multi-output Op has to be fetched.",
                x,
            )
        else:
            x = x.outputs[0]
    if isinstance(x, Variable):
        if not isinstance(x.type, SparseTensorType):
            raise TypeError(
                "Variable type field must be a SparseTensorType.", x, x.type
            )
        return x
    try:
        from pytensor.sparse.variable import constant

        return constant(x, name=name)
    except TypeError:
        raise TypeError(f"Cannot convert {x} to SparseTensorType", type(x))


as_sparse = as_sparse_variable

as_sparse_or_tensor_variable = as_symbolic


def sp_ones_like(x):
    """
    Construct a sparse matrix of ones with the same sparsity pattern.

    Parameters
    ----------
    x
        Sparse matrix to take the sparsity pattern.

    Returns
    -------
    A sparse matrix
        The same as `x` with data changed for ones.

    """
    # TODO: don't restrict to CSM formats
    data, indices, indptr, _shape = csm_properties(x)
    return CSM(format=x.format)(ptb.ones_like(data), indices, indptr, _shape)


def sp_zeros_like(x):
    """
    Construct a sparse matrix of zeros.

    Parameters
    ----------
    x
        Sparse matrix to take the shape.

    Returns
    -------
    A sparse matrix
        The same as `x` with zero entries for all element.

    """

    # TODO: don't restrict to CSM formats
    _, _, indptr, _shape = csm_properties(x)
    return CSM(format=x.format)(
        data=np.array([], dtype=x.type.dtype),
        indices=np.array([], dtype="int32"),
        indptr=ptb.zeros_like(indptr),
        shape=_shape,
    )


# for more dtypes, call SparseTensorType(format, dtype)
def matrix(format, name=None, dtype=None, shape=None):
    if dtype is None:
        dtype = config.floatX
    type = SparseTensorType(format=format, dtype=dtype, shape=shape)
    return type(name)


def csc_matrix(name=None, dtype=None, shape=None):
    return matrix("csc", name=name, dtype=dtype, shape=shape)


def csr_matrix(name=None, dtype=None, shape=None):
    return matrix("csr", name=name, dtype=dtype, shape=shape)


def bsr_matrix(name=None, dtype=None):
    return matrix("bsr", name, dtype)


csc_dmatrix = SparseTensorType(format="csc", dtype="float64")
csr_dmatrix = SparseTensorType(format="csr", dtype="float64")
bsr_dmatrix = SparseTensorType(format="bsr", dtype="float64")
csc_fmatrix = SparseTensorType(format="csc", dtype="float32")
csr_fmatrix = SparseTensorType(format="csr", dtype="float32")
bsr_fmatrix = SparseTensorType(format="bsr", dtype="float32")

all_dtypes = list(SparseTensorType.dtype_specs_map)
complex_dtypes = [t for t in all_dtypes if t[:7] == "complex"]
float_dtypes = [t for t in all_dtypes if t[:5] == "float"]
int_dtypes = [t for t in all_dtypes if t[:3] == "int"]
uint_dtypes = [t for t in all_dtypes if t[:4] == "uint"]
integer_dtypes = int_dtypes + uint_dtypes

continuous_dtypes = complex_dtypes + float_dtypes
discrete_dtypes = int_dtypes + uint_dtypes


class CSMProperties(Op):
    """Create arrays containing all the properties of a given sparse matrix.

    More specifically, this `Op` extracts the ``.data``, ``.indices``,
    ``.indptr`` and ``.shape`` fields.

    For specific field, `csm_data`, `csm_indices`, `csm_indptr`
    and `csm_shape` are provided.

    Notes
    -----
    The grad implemented is regular, i.e. not structured.
    `infer_shape` method is not available for this `Op`.

    We won't implement infer_shape for this op now. This will
    ask that we implement an GetNNZ op, and this op will keep
    the dependence on the input of this op. So this won't help
    to remove computations in the graph. To remove computation,
    we will need to make an infer_sparse_pattern feature to
    remove computations. Doing this is trickier then the
    infer_shape feature. For example, how do we handle the case
    when some op create some 0 values? So there is dependence
    on the values themselves. We could write an infer_shape for
    the last output that is the shape, but I dough this will
    get used.

    We don't return a view of the shape, we create a new ndarray from the shape
    tuple.
    """

    __props__ = ()
    view_map = {0: [0], 1: [0], 2: [0]}

    def __init__(self, kmap=None):
        if kmap is not None:
            raise Exception("Do not use kmap, it is removed")

    def make_node(self, csm):
        """

        The output vectors correspond to the tuple
        ``(data, indices, indptr, shape)``, i.e. the properties of a `csm`
        array.

        Parameters
        ----------
        csm
            Sparse matrix in `CSR` or `CSC` format.

        """

        csm = as_sparse_variable(csm)
        assert csm.format in ("csr", "csc")
        data = vector(dtype=csm.type.dtype)
        return Apply(self, [csm], [data, ivector(), ivector(), ivector()])

    def perform(self, node, inputs, out):
        (csm,) = inputs
        out[0][0] = np.asarray(csm.data)
        out[1][0] = np.asarray(csm.indices, dtype="int32")
        out[2][0] = np.asarray(csm.indptr, dtype="int32")
        out[3][0] = np.asarray(csm.shape, dtype="int32")

    def grad(self, inputs, g):
        # g[1:] is all integers, so their Jacobian in this op
        # is 0. We thus don't need to worry about what their values
        # are.

        # if g[0] is disconnected, then this op doesn't contribute
        # any gradient anywhere. but we know that at least one of
        # g[1:] is connected, or this grad method wouldn't have been
        # called, so we should report zeros
        (csm,) = inputs
        if isinstance(g[0].type, DisconnectedType):
            return [csm.zeros_like()]

        _data, indices, indptr, _shape = csm_properties(csm)
        return [CSM(csm.format)(g[0], indices, indptr, _shape)]


# don't make this a function or it breaks some optimizations below
csm_properties = CSMProperties()


def csm_data(csm):
    """
    Return the data field of the sparse variable.

    """
    return csm_properties(csm)[0]


def csm_indices(csm):
    """
    Return the indices field of the sparse variable.

    """
    return csm_properties(csm)[1]


def csm_indptr(csm):
    """
    Return the indptr field of the sparse variable.

    """
    return csm_properties(csm)[2]


def csm_shape(csm):
    """
    Return the shape field of the sparse variable.

    """
    return csm_properties(csm)[3]


class CSM(Op):
    """Construct a CSM matrix from constituent parts.

    Notes
    -----
    The grad method returns a dense vector, so it provides a regular grad.

    """

    __props__ = ("format",)

    def __init__(self, format, kmap=None):
        if format not in ("csr", "csc"):
            raise ValueError("format must be one of: 'csr', 'csc'", format)
        self.format = format
        if kmap is not None:
            raise Exception("Do not use kmap, it is removed")
        # should view the other inputs too, but viewing multiple
        # inputs is not currently supported by the destroyhandler
        self.view_map = {0: [0]}

    def make_node(self, data, indices, indptr, shape):
        """

        Parameters
        ----------
        data
            One dimensional tensor representing the data of the sparse matrix to
            construct.
        indices
            One dimensional tensor of integers representing the indices of the sparse
            matrix to construct.
        indptr
            One dimensional tensor of integers representing the indice pointer for
            the sparse matrix to construct.
        shape
            One dimensional tensor of integers representing the shape of the sparse
            matrix to construct.

        """
        data = ptb.as_tensor_variable(data)

        if not isinstance(indices, Variable):
            indices_ = np.asarray(indices)
            indices_32 = np.asarray(indices, dtype="int32")
            assert (indices_ == indices_32).all()
            indices = indices_32
        if not isinstance(indptr, Variable):
            indptr_ = np.asarray(indptr)
            indptr_32 = np.asarray(indptr, dtype="int32")
            assert (indptr_ == indptr_32).all()
            indptr = indptr_32
        if not isinstance(shape, Variable):
            shape_ = np.asarray(shape)
            shape_32 = np.asarray(shape, dtype="int32")
            assert (shape_ == shape_32).all()
            shape = shape_32

        indices = ptb.as_tensor_variable(indices)
        indptr = ptb.as_tensor_variable(indptr)
        shape = ptb.as_tensor_variable(shape)

        if data.type.ndim != 1:
            raise TypeError("data argument must be a vector", data.type, data.type.ndim)
        if indices.type.ndim != 1 or indices.type.dtype not in discrete_dtypes:
            raise TypeError("indices must be vector of integers", indices, indices.type)
        if indptr.type.ndim != 1 or indptr.type.dtype not in discrete_dtypes:
            raise TypeError("indices must be vector of integers", indptr, indptr.type)
        if shape.type.ndim != 1 or shape.type.dtype not in discrete_dtypes:
            raise TypeError("n_rows must be integer type", shape, shape.type)

        static_shape = (None, None)
        if (
            shape.owner is not None
            and isinstance(shape.owner.op, CSMProperties)
            and shape.owner.outputs[3] is shape
        ):
            static_shape = shape.owner.inputs[0].type.shape

        return Apply(
            self,
            [data, indices, indptr, shape],
            [
                SparseTensorType(
                    dtype=data.type.dtype, format=self.format, shape=static_shape
                )()
            ],
        )

    def perform(self, node, inputs, outputs):
        # for efficiency, if remap does nothing, then do not apply it
        (data, indices, indptr, _shape) = inputs
        (out,) = outputs

        if len(_shape) != 2:
            raise ValueError("Shape should be an array of length 2")
        if data.shape != indices.shape:
            errmsg = (
                "Data (shape "
                + repr(data.shape)
                + " must have the same number of elements "
                + "as indices (shape"
                + repr(indices.shape)
                + ")"
            )
            raise ValueError(errmsg)
        if self.format == "csc":
            out[0] = scipy.sparse.csc_matrix(
                (data, indices.copy(), indptr.copy()), np.asarray(_shape), copy=False
            )
        else:
            assert self.format == "csr"
            out[0] = scipy.sparse.csr_matrix(
                (data, indices.copy(), indptr.copy()), _shape.copy(), copy=False
            )

    def connection_pattern(self, node):
        return [[True], [False], [False], [False]]

    def grad(self, inputs, gout):
        (x_data, x_indices, x_indptr, x_shape) = inputs
        (g_out,) = gout
        g_data, g_indices, g_indptr, g_shape = csm_properties(g_out)
        # unpack the data vector and wrap it as a 1d TensorType
        g_data = csm_grad()(
            x_data, x_indices, x_indptr, x_shape, g_data, g_indices, g_indptr, g_shape
        )
        return [
            g_data,
            disconnected_type(),
            disconnected_type(),
            disconnected_type(),
        ]

    def infer_shape(self, fgraph, node, shapes):
        # node.inputs[3] is of length as we only support sparse matrix.
        return [(node.inputs[3][0], node.inputs[3][1])]


CSC = CSM("csc")

CSR = CSM("csr")


class CSMGrad(Op):
    """Compute the gradient of a CSM.

    Note
    ----
    CSM creates a matrix from data, indices, and indptr vectors; it's gradient
    is the gradient of the data vector only. There are two complexities to
    calculate this gradient:

    1. The gradient may be sparser than the input matrix defined by (data,
    indices, indptr). In this case, the data vector of the gradient will have
    less elements than the data vector of the input because sparse formats
    remove 0s. Since we are only returning the gradient of the data vector,
    the relevant 0s need to be added back.
    2. The elements in the sparse dimension are not guaranteed to be sorted.
    Therefore, the input data vector may have a different order than the
    gradient data vector.
    """

    __props__ = ()

    def __init__(self, kmap=None):
        if kmap is not None:
            raise Exception("Do not use kmap, it is removed")
        # This class always allocate a new output.
        # I keep this here to help GD understand what this kmap think is.
        # if self.kmap is None:
        #    self.view_map = {0: [1]}

    def make_node(
        self, x_data, x_indices, x_indptr, x_shape, g_data, g_indices, g_indptr, g_shape
    ):
        gout_data = g_data.type()
        return Apply(
            self,
            [
                x_data,
                x_indices,
                x_indptr,
                x_shape,
                g_data,
                g_indices,
                g_indptr,
                g_shape,
            ],
            [gout_data],
        )

    def perform(self, node, inputs, outputs):
        (
            x_data,
            x_indices,
            x_indptr,
            x_shape,
            g_data,
            g_indices,
            g_indptr,
            _g_shape,
        ) = inputs
        (g_out,) = outputs
        if len(x_indptr) - 1 == x_shape[0]:
            sp_dim = x_shape[1]
        else:
            sp_dim = x_shape[0]

        g_row = np.zeros(sp_dim, dtype=g_data.dtype)
        gout_data = np.zeros(x_data.shape, dtype=node.outputs[0].dtype)

        for i in range(len(x_indptr) - 1):
            for j_ptr in range(g_indptr[i], g_indptr[i + 1]):
                g_row[g_indices[j_ptr]] += g_data[j_ptr]

            for j_ptr in range(x_indptr[i], x_indptr[i + 1]):
                gout_data[j_ptr] = g_row[x_indices[j_ptr]]

            for j_ptr in range(g_indptr[i], g_indptr[i + 1]):
                g_row[g_indices[j_ptr]] = 0

        g_out[0] = gout_data

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[1]]


csm_grad = CSMGrad


class Cast(Op):
    __props__ = ("out_type",)

    def __init__(self, out_type):
        self.out_type = out_type

    def make_node(self, x):
        x = as_sparse_variable(x)
        assert x.format in ("csr", "csc")
        return Apply(
            self, [x], [SparseTensorType(dtype=self.out_type, format=x.format)()]
        )

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (out,) = outputs
        assert _is_sparse(x)
        out[0] = x.astype(self.out_type)

    def grad(self, inputs, outputs_gradients):
        gz = outputs_gradients[0]

        if gz.dtype in complex_dtypes:
            raise NotImplementedError("grad not implemented for complex types")
        if inputs[0].dtype in complex_dtypes:
            raise NotImplementedError("grad not implemented for complex types")

        if gz.dtype in discrete_dtypes:
            if inputs[0].dtype in discrete_dtypes:
                return [inputs[0].zeros_like(dtype=config.floatX)]
            else:
                return [inputs[0].zeros_like()]
        else:
            if inputs[0].dtype in discrete_dtypes:
                return [gz]
            else:
                return [Cast(inputs[0].dtype)(gz)]

    def infer_shape(self, fgraph, node, ins_shapes):
        return ins_shapes

    def __str__(self):
        return f"{self.__class__.__name__}({self.out_type})"


bcast = Cast("int8")
wcast = Cast("int16")
icast = Cast("int32")
lcast = Cast("int64")
fcast = Cast("float32")
dcast = Cast("float64")
ccast = Cast("complex64")
zcast = Cast("complex128")


def cast(variable, dtype):
    """
    Cast sparse variable to the desired dtype.

    Parameters
    ----------
    variable
        Sparse matrix.
    dtype
        The dtype wanted.

    Returns
    -------
    Same as `x` but having `dtype` as dtype.

    Notes
    -----
    The grad implemented is regular, i.e. not structured.

    """
    return Cast(dtype)(variable)


class DenseFromSparse(Op):
    """Convert a sparse matrix to a dense one.

    Notes
    -----
    The grad implementation can be controlled through the constructor via the
    `structured` parameter. `True` will provide a structured grad while `False`
    will provide a regular grad. By default, the grad is structured.

    """

    __props__ = ()

    def __init__(self, structured=True):
        self.sparse_grad = structured

    def __str__(self):
        return f"{self.__class__.__name__}{{structured_grad={self.sparse_grad}}}"

    def __call__(self, x):
        if not isinstance(x.type, SparseTensorType):
            return x

        return super().__call__(x)

    def make_node(self, x):
        """

        Parameters
        ----------
        x
            A sparse matrix.

        """
        x = as_sparse_variable(x)
        return Apply(
            self,
            [x],
            [TensorType(dtype=x.type.dtype, shape=x.type.shape)()],
        )

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (out,) = outputs
        if _is_dense(x):
            warn(
                "You just called DenseFromSparse on a dense matrix.",
            )
            out[0] = x
        else:
            out[0] = x.toarray()
        assert _is_dense(out[0])

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if self.sparse_grad:
            left = sp_ones_like(x)
            right = gz

            # Do upcasting if necessary to avoid an unimplemented case
            # of mul

            if right.dtype == "float64" and left.dtype == "float32":
                left = left.astype("float64")

            if right.dtype == "float32" and left.dtype == "float64":
                right = right.astype("float64")

            return [left * right]
        else:
            return [SparseFromDense(x.type.format)(gz)]

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]


dense_from_sparse = DenseFromSparse()


class SparseFromDense(Op):
    """Convert a dense matrix to a sparse matrix."""

    __props__ = ()

    def __init__(self, format):
        self.format = format

    def __str__(self):
        return f"{self.__class__.__name__}{{{self.format}}}"

    def __call__(self, x):
        if isinstance(x.type, SparseTensorType):
            return x

        return super().__call__(x)

    def make_node(self, x):
        """

        Parameters
        ----------
        x
            A dense matrix.

        """
        x = ptb.as_tensor_variable(x)
        if x.ndim > 2:
            raise TypeError(
                "PyTensor does not have sparse tensor types with more "
                f"than 2 dimensions, but {x}.ndim = {x.ndim}"
            )
        elif x.ndim == 1:
            x = x.dimshuffle("x", 0)
        elif x.ndim == 0:
            x = x.dimshuffle("x", "x")
        else:
            assert x.ndim == 2

        return Apply(
            self, [x], [SparseTensorType(dtype=x.type.dtype, format=self.format)()]
        )

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (out,) = outputs
        out[0] = SparseTensorType.format_cls[self.format](x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        gx = dense_from_sparse(gz)
        gx = specify_broadcastable(
            gx, *(ax for (ax, b) in enumerate(x.type.broadcastable) if b)
        )
        return (gx,)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]


csr_from_dense = SparseFromDense("csr")

csc_from_dense = SparseFromDense("csc")


class GetItemList(Op):
    """Select row of sparse matrix, returning them as a new sparse matrix."""

    __props__ = ()

    def infer_shape(self, fgraph, node, shapes):
        return [(shapes[1][0], shapes[0][1])]

    def make_node(self, x, index):
        """

        Parameters
        ----------
        x
            Sparse matrix.
        index
            List of rows.

        """
        x = as_sparse_variable(x)
        assert x.format in ("csr", "csc")

        ind = ptb.as_tensor_variable(index)
        assert ind.ndim == 1
        assert ind.dtype in integer_dtypes

        return Apply(self, [x, ind], [x.type()])

    def perform(self, node, inp, outputs):
        (out,) = outputs
        x = inp[0]
        indices = inp[1]
        assert _is_sparse(x)
        out[0] = x[indices]

    def grad(self, inputs, g_outputs):
        x, indices = inputs
        (gout,) = g_outputs
        return [
            get_item_list_grad(x, indices, gout),
            grad_undefined(self, 1, indices, "No gradient for this input"),
        ]


get_item_list = GetItemList()


class GetItemListGrad(Op):
    __props__ = ()

    def infer_shape(self, fgraph, node, shapes):
        return [(shapes[0])]

    def make_node(self, x, index, gz):
        x = as_sparse_variable(x)
        gz = as_sparse_variable(gz)

        assert x.format in ("csr", "csc")
        assert gz.format in ("csr", "csc")

        ind = ptb.as_tensor_variable(index)
        assert ind.ndim == 1
        assert ind.dtype in integer_dtypes

        scipy_ver = [int(n) for n in scipy.__version__.split(".")[:2]]

        if not scipy_ver >= [0, 13]:
            raise NotImplementedError("Scipy version is to old")

        return Apply(self, [x, ind, gz], [x.type()])

    def perform(self, node, inp, outputs):
        (out,) = outputs
        x = inp[0]
        indices = inp[1]
        gz = inp[2]

        if x.format in ["csr"]:
            y = scipy.sparse.csr_matrix((x.shape[0], x.shape[1]))
        else:
            y = scipy.sparse.csc_matrix((x.shape[0], x.shape[1]))
        for a in range(0, len(indices)):
            y[indices[a]] = gz[a]

        out[0] = y


get_item_list_grad = GetItemListGrad()


class GetItem2Lists(Op):
    """Select elements of sparse matrix, returning them in a vector."""

    __props__ = ()

    def make_node(self, x, ind1, ind2):
        """

        Parameters
        ----------
        x
            Sparse matrix.
        index
            List of two lists, first list indicating the row of each element and second
            list indicating its column.

        """
        x = as_sparse_variable(x)
        assert x.format in ("csr", "csc")
        ind1 = ptb.as_tensor_variable(ind1)
        ind2 = ptb.as_tensor_variable(ind2)
        assert ind1.dtype in integer_dtypes
        assert ind2.dtype in integer_dtypes

        return Apply(self, [x, ind1, ind2], [vector()])

    def perform(self, node, inp, outputs):
        (out,) = outputs
        x = inp[0]
        ind1 = inp[1]
        ind2 = inp[2]
        # SciPy returns the corresponding elements as a `matrix`-type instance,
        # which isn't what we want, so we convert it into an `ndarray`
        out[0] = np.asarray(x[ind1, ind2]).flatten()

    def grad(self, inputs, g_outputs):
        x, ind1, ind2 = inputs
        (gout,) = g_outputs
        return [
            get_item_2lists_grad(x, ind1, ind2, gout),
            grad_undefined(self, 1, ind1, "No gradient for this input"),
            grad_undefined(self, 1, ind2, "No gradient for this input"),
        ]


get_item_2lists = GetItem2Lists()


class GetItem2ListsGrad(Op):
    __props__ = ()

    def infer_shape(self, fgraph, node, shapes):
        return [(shapes[0])]

    def make_node(self, x, ind1, ind2, gz):
        x = as_sparse_variable(x)

        assert x.format in ("csr", "csc")

        ind1 = ptb.as_tensor_variable(ind1)
        ind2 = ptb.as_tensor_variable(ind2)
        assert ind1.ndim == 1
        assert ind2.ndim == 1
        assert ind1.dtype in integer_dtypes
        assert ind2.dtype in integer_dtypes

        return Apply(self, [x, ind1, ind2, gz], [x.type()])

    def perform(self, node, inp, outputs):
        (out,) = outputs
        x = inp[0]
        ind1 = inp[1]
        ind2 = inp[2]
        gz = inp[3]

        if x.format in ["csr"]:
            y = scipy.sparse.csr_matrix((x.shape[0], x.shape[1]))
        else:
            y = scipy.sparse.csc_matrix((x.shape[0], x.shape[1]))
        z = 0
        for z in range(0, len(ind1)):
            y[(ind1[z], ind2[z])] = gz[z]

        out[0] = y


get_item_2lists_grad = GetItem2ListsGrad()


class GetItem2d(Op):
    """Implement a subtensor of sparse variable, returning a sparse matrix.

    If you want to take only one element of a sparse matrix see
    `GetItemScalar` that returns a tensor scalar.

    Notes
    -----
    Subtensor selection always returns a matrix, so indexing with [a:b, c:d]
    is forced. If one index is a scalar, for instance, x[a:b, c] or x[a, b:c],
    an error will be raised. Use instead x[a:b, c:c+1] or x[a:a+1, b:c].

    The above indexing methods are not supported because the return value
    would be a sparse matrix rather than a sparse vector, which is a
    deviation from numpy indexing rule. This decision is made largely
    to preserve consistency between numpy and pytensor. This may be revised
    when sparse vectors are supported.

    The grad is not implemented for this op.

    """

    __props__ = ()

    def make_node(self, x, index):
        """

        Parameters
        ----------
        x
            Sparse matrix.
        index
            Tuple of slice object.

        """
        scipy_ver = [int(n) for n in scipy.__version__.split(".")[:2]]
        x = as_sparse_variable(x)
        assert x.format in ("csr", "csc")
        assert len(index) in (1, 2)

        input_op = [x]
        generic_None = Constant(generic, None)

        for ind in index:
            if isinstance(ind, slice):
                # in case of slice is written in pytensor variable
                start = ind.start
                stop = ind.stop
                step = ind.step
                # If start or stop or step are None, make them a Generic
                # constant. Else, they should be converted to Tensor Variables
                # of dimension 1 and int/uint dtype.
                if scipy_ver < [0, 14] and ind.step is not None:
                    raise ValueError(
                        "Slice with step is not support with current version of Scipy."
                    )
                if ind.step is None or ind.step == 1:
                    step = generic_None
                else:
                    if not isinstance(step, Variable):
                        step = ptb.as_tensor_variable(step)
                    if not (step.ndim == 0 and step.dtype in tensor_discrete_dtypes):
                        raise ValueError(
                            (
                                "Impossible to index into a sparse matrix with "
                                f"slice where step={step}"
                            ),
                            step.ndim,
                            step.dtype,
                        )

                if start is None:
                    start = generic_None
                else:
                    if not isinstance(start, Variable):
                        start = ptb.as_tensor_variable(start)
                    if not (start.ndim == 0 and start.dtype in tensor_discrete_dtypes):
                        raise ValueError(
                            (
                                "Impossible to index into a sparse matrix with "
                                f"slice where start={start}"
                            ),
                            start.ndim,
                            start.dtype,
                        )

                if stop is None:
                    stop = generic_None
                else:
                    if not isinstance(stop, Variable):
                        stop = ptb.as_tensor_variable(stop)
                    if not (stop.ndim == 0 and stop.dtype in tensor_discrete_dtypes):
                        raise ValueError(
                            (
                                "Impossible to index into a sparse matrix with "
                                f"slice where stop={stop}"
                            ),
                            stop.ndim,
                            stop.dtype,
                        )

            elif (
                isinstance(ind, Variable) and getattr(ind, "ndim", -1) == 0
            ) or np.isscalar(ind):
                raise NotImplementedError(
                    "PyTensor has no sparse vector. "
                    "Use X[a:b, c:d], X[a:b, c:c+1] or X[a:b] instead."
                )
            else:
                raise ValueError(
                    "Advanced indexing is not implemented for sparse "
                    f"matrices. Argument not supported: {ind}"
                )
            input_op += [start, stop, step]
        if len(index) == 1:
            input_op += [generic_None, generic_None, generic_None]

        return Apply(self, input_op, [x.type()])

    def perform(self, node, inputs, outputs):
        (x, start1, stop1, step1, start2, stop2, step2) = inputs
        (out,) = outputs
        assert _is_sparse(x)
        out[0] = x[start1:stop1:step1, start2:stop2:step2]


get_item_2d = GetItem2d()


class GetItemScalar(Op):
    """Subtensor of a sparse variable that takes two scalars as index and returns a scalar.

    If you want to take a slice of a sparse matrix see `GetItem2d` that returns a
    sparse matrix.

    Notes
    -----
    The grad is not implemented for this op.

    """

    __props__ = ()

    def infer_shape(self, fgraph, node, shapes):
        return [()]

    def make_node(self, x, index):
        """

        Parameters
        ----------
        x
            Sparse matrix.
        index
            Tuple of scalars.

        """
        x = as_sparse_variable(x)
        assert x.format in ("csr", "csc")
        assert len(index) == 2

        input_op = [x]

        for ind in index:
            if isinstance(ind, slice):
                raise Exception("GetItemScalar called with a slice as index!")

            # in case of indexing using int instead of pytensor variable
            elif isinstance(ind, int):
                ind = ptb.constant(ind)
                input_op += [ind]

            # in case of indexing using pytensor variable
            elif ind.ndim == 0:
                input_op += [ind]
            else:
                raise NotImplementedError

        return Apply(self, input_op, [scalar(dtype=x.dtype)])

    def perform(self, node, inputs, outputs):
        (x, ind1, ind2) = inputs
        (out,) = outputs
        assert _is_sparse(x)
        out[0] = np.asarray(x[ind1, ind2], x.dtype)


get_item_scalar = GetItemScalar()


class Transpose(Op):
    """Transpose of a sparse matrix.

    Notes
    -----
    The returned matrix will not be in the same format. `csc` matrix will be changed
    in `csr` matrix and `csr` matrix in `csc` matrix.

    The grad is regular, i.e. not structured.

    """

    view_map = {0: [0]}

    format_map = {"csr": "csc", "csc": "csr"}
    __props__ = ()

    def __str__(self):
        return "Sparse" + self.__class__.__name__

    def make_node(self, x):
        """

        Parameters
        ----------
        x
            Sparse matrix.

        """
        x = as_sparse_variable(x)
        assert x.format in ("csr", "csc")
        return Apply(
            self,
            [x],
            [
                SparseTensorType(
                    dtype=x.type.dtype, format=self.format_map[x.type.format]
                )()
            ],
        )

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (out,) = outputs
        assert _is_sparse(x)
        out[0] = x.transpose()

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        assert _is_sparse_variable(x) and _is_sparse_variable(gz)
        return (transpose(gz),)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0][::-1]]


transpose = Transpose()


class Neg(Op):
    """Negative of the sparse matrix (i.e. multiply by ``-1``).

    Notes
    -----
    The grad is regular, i.e. not structured.

    """

    __props__ = ()

    def __str__(self):
        return "Sparse" + self.__class__.__name__

    def make_node(self, x):
        """

        Parameters
        ----------
        x
            Sparse matrix.

        """
        x = as_sparse_variable(x)
        assert x.format in ("csr", "csc")
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (out,) = outputs
        assert _is_sparse(x)
        out[0] = -x

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        assert _is_sparse_variable(x) and _is_sparse_variable(gz)
        return (-gz,)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]


neg = Neg()


class ColScaleCSC(Op):
    # Scale each columns of a sparse matrix by the corresponding
    # element of a dense vector

    # :param x: A sparse matrix.
    # :param s: A dense vector with length equal to the number
    #           of columns of `x`.

    # :return: A sparse matrix in the same format as `x` which
    #          each column had been multiply by the corresponding
    #          element of `s`.

    # :note: The grad implemented is structured.

    __props__ = ()

    def make_node(self, x, s):
        if x.format != "csc":
            raise ValueError("x was not a csc matrix")
        return Apply(self, [x, s], [x.type()])

    def perform(self, node, inputs, outputs):
        (x, s) = inputs
        (z,) = outputs
        _M, N = x.shape
        assert x.format == "csc"
        assert s.shape == (N,)

        y = x.copy()

        for j in range(0, N):
            y.data[y.indptr[j] : y.indptr[j + 1]] *= s[j]

        z[0] = y

    def grad(self, inputs, gout):
        from pytensor.sparse.math import sp_sum

        (x, s) = inputs
        (gz,) = gout
        return [col_scale(gz, s), sp_sum(x * gz, axis=0)]

    def infer_shape(self, fgraph, node, ins_shapes):
        return [ins_shapes[0]]


class RowScaleCSC(Op):
    # Scale each row of a sparse matrix by the corresponding element of
    # a dense vector

    # :param x: A sparse matrix.
    # :param s: A dense vector with length equal to the number
    #           of rows of `x`.

    # :return: A sparse matrix in the same format as `x` which
    #          each row had been multiply by the corresponding
    #          element of `s`.

    # :note: The grad implemented is structured.

    view_map = {0: [0]}
    __props__ = ()

    def make_node(self, x, s):
        x = as_sparse_variable(x)
        assert x.format in ("csr", "csc")
        return Apply(self, [x, s], [x.type()])

    def perform(self, node, inputs, outputs):
        (x, s) = inputs
        (z,) = outputs
        M, N = x.shape
        assert x.format == "csc"
        assert s.shape == (M,)

        indices = x.indices
        indptr = x.indptr

        y_data = x.data.copy()

        for j in range(0, N):
            for i_idx in range(indptr[j], indptr[j + 1]):
                y_data[i_idx] *= s[indices[i_idx]]

        z[0] = scipy.sparse.csc_matrix((y_data, indices, indptr), (M, N))

    def grad(self, inputs, gout):
        from pytensor.sparse.math import sp_sum

        (x, s) = inputs
        (gz,) = gout
        return [row_scale(gz, s), sp_sum(x * gz, axis=1)]

    def infer_shape(self, fgraph, node, ins_shapes):
        return [ins_shapes[0]]


def col_scale(x, s):
    """
    Scale each columns of a sparse matrix by the corresponding element of a
    dense vector.

    Parameters
    ----------
    x
        A sparse matrix.
    s
        A dense vector with length equal to the number of columns of `x`.

    Returns
    -------
    A sparse matrix in the same format as `x` which each column had been
    multiply by the corresponding element of `s`.

    Notes
    -----
    The grad implemented is structured.

    """

    if x.format == "csc":
        return ColScaleCSC()(x, s)
    elif x.format == "csr":
        return RowScaleCSC()(x.T, s).T
    else:
        raise NotImplementedError()


def row_scale(x, s):
    """
    Scale each row of a sparse matrix by the corresponding element of
    a dense vector.

    Parameters
    ----------
    x
        A sparse matrix.
    s
        A dense vector with length equal to the number of rows of `x`.

    Returns
    -------
    A sparse matrix
        A sparse matrix in the same format as `x` whose each row has been
        multiplied by the corresponding element of `s`.

    Notes
    -----
    The grad implemented is structured.

    """
    return col_scale(x.T, s).T


class Diag(Op):
    """Extract the diagonal of a square sparse matrix as a dense vector.

    Notes
    -----
    The grad implemented is regular, i.e. not structured, since the output is a
    dense vector.

    """

    __props__ = ()

    def make_node(self, x):
        """

        Parameters
        ----------
        x
            A square sparse matrix in csc format.

        """
        x = as_sparse_variable(x)
        assert x.format in ("csr", "csc")
        return Apply(self, [x], [tensor(dtype=x.dtype, shape=(None,))])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        N, M = x.shape
        if N != M:
            raise ValueError("Diag only apply on square matrix")
        z[0] = x.diagonal()

    def grad(self, inputs, gout):
        (_x,) = inputs
        (gz,) = gout
        return [square_diagonal(gz)]

    def infer_shape(self, fgraph, nodes, shapes):
        return [(minimum(*shapes[0]),)]


diag = Diag()


class SquareDiagonal(Op):
    """Produce a square sparse (csc) matrix with a diagonal given by a dense vector.

    Notes
    -----
    The grad implemented is regular, i.e. not structured.

    """

    __props__ = ()

    def make_node(self, diag):
        """

        Parameters
        ----------
        x
            Dense vector for the diagonal.

        """
        diag = ptb.as_tensor_variable(diag)
        if diag.type.ndim != 1:
            raise TypeError("data argument must be a vector", diag.type)

        return Apply(self, [diag], [SparseTensorType(dtype=diag.dtype, format="csc")()])

    def perform(self, node, inputs, outputs):
        (z,) = outputs
        diag = inputs[0]

        N = len(diag)
        data = diag[:N]
        indices = list(range(N))
        indptr = list(range(N + 1))
        tup = (data, indices, indptr)

        z[0] = scipy.sparse.csc_matrix(tup, copy=True)

    def grad(self, inputs, gout):
        (gz,) = gout
        return [diag(gz)]

    def infer_shape(self, fgraph, nodes, shapes):
        return [(shapes[0][0], shapes[0][0])]


square_diagonal = SquareDiagonal()


class EnsureSortedIndices(Op):
    """Re-sort indices of a sparse matrix.

    CSR column indices are not necessarily sorted. Likewise
    for CSC row indices. Use `ensure_sorted_indices` when sorted
    indices are required (e.g. when passing data to other
    libraries).

    Notes
    -----
    The grad implemented is regular, i.e. not structured.

    """

    __props__ = ("inplace",)

    def __init__(self, inplace):
        self.inplace = inplace
        if self.inplace:
            self.view_map = {0: [0]}

    def make_node(self, x):
        """
        Parameters
        ----------
        x
            A sparse matrix.

        """
        x = as_sparse_variable(x)
        assert x.format in ("csr", "csc")
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        if self.inplace:
            z[0] = x.sort_indices()
        else:
            z[0] = x.sorted_indices()

    def grad(self, inputs, output_grad):
        return [output_grad[0]]

    def infer_shape(self, fgraph, node, i0_shapes):
        return i0_shapes

    def __str__(self):
        if self.inplace:
            return self.__class__.__name__ + "{inplace}"
        else:
            return self.__class__.__name__ + "{no_inplace}"


ensure_sorted_indices = EnsureSortedIndices(inplace=False)


def clean(x):
    """
    Remove explicit zeros from a sparse matrix, and re-sort indices.

    CSR column indices are not necessarily sorted. Likewise
    for CSC row indices. Use `clean` when sorted
    indices are required (e.g. when passing data to other
    libraries) and to ensure there are no zeros in the data.

    Parameters
    ----------
    x
        A sparse matrix.

    Returns
    -------
    A sparse matrix
        The same as `x` with indices sorted and zeros
        removed.

    Notes
    -----
    The grad implemented is regular, i.e. not structured.

    """
    return ensure_sorted_indices(remove0(x))


class Stack(Op):
    __props__ = ("format", "dtype")

    def __init__(self, format=None, dtype=None):
        if format is None:
            self.format = "csc"
        else:
            self.format = format

        if dtype is None:
            raise ValueError("The output dtype must be specified.")
        self.dtype = dtype

    def make_node(self, *mat):
        if not mat:
            raise ValueError("Cannot join an empty list of sparses.")
        var = [as_sparse_variable(x) for x in mat]

        for x in var:
            assert x.format in ("csr", "csc")

        return Apply(
            self, var, [SparseTensorType(dtype=self.dtype, format=self.format)()]
        )

    def __str__(self):
        return f"{self.__class__.__name__}({self.format},{self.dtype})"


class HStack(Stack):
    def perform(self, node, block, outputs):
        (out,) = outputs
        for b in block:
            assert _is_sparse(b)
        out[0] = scipy.sparse.hstack(block, format=self.format, dtype=self.dtype)
        # Some version of scipy (at least 0.14.0.dev-c4314b0)
        # Do not cast to the wanted dtype.
        if out[0].dtype != self.dtype:
            out[0] = out[0].astype(self.dtype)

    def grad(self, inputs, gout):
        (gz,) = gout
        is_continuous = [
            (inputs[i].dtype in tensor_continuous_dtypes) for i in range(len(inputs))
        ]

        if _is_sparse_variable(gz):
            gz = dense_from_sparse(gz)

        split = Split(len(inputs))(gz, 1, ptb.stack([x.shape[1] for x in inputs]))
        if not isinstance(split, list):
            split = [split]

        derivative = [SparseFromDense(self.format)(s) for s in split]

        def choose(continuous, derivative):
            if continuous:
                return derivative
            else:
                return None

        return [choose(c, d) for c, d in zip(is_continuous, derivative, strict=True)]

    def infer_shape(self, fgraph, node, ins_shapes):
        d = sum(shape[1] for shape in ins_shapes)
        return [(ins_shapes[0][0], d)]


def hstack(blocks, format=None, dtype=None):
    """
    Stack sparse matrices horizontally (column wise).

    This wrap the method hstack from scipy.

    Parameters
    ----------
    blocks
        List of sparse array of compatible shape.
    format
        String representing the output format. Default is csc.
    dtype
        Output dtype.

    Returns
    -------
    array
        The concatenation of the sparse array column wise.

    Notes
    -----
    The number of line of the sparse matrix must agree.

    The grad implemented is regular, i.e. not structured.

    """

    blocks = [as_sparse_variable(i) for i in blocks]
    if dtype is None:
        dtype = ps.upcast(*[i.dtype for i in blocks])
    return HStack(format=format, dtype=dtype)(*blocks)


class VStack(Stack):
    def perform(self, node, block, outputs):
        (out,) = outputs
        for b in block:
            assert _is_sparse(b)
        out[0] = scipy.sparse.vstack(block, format=self.format, dtype=self.dtype)
        # Some version of scipy (at least 0.14.0.dev-c4314b0)
        # Do not cast to the wanted dtype.
        if out[0].dtype != self.dtype:
            out[0] = out[0].astype(self.dtype)

    def grad(self, inputs, gout):
        (gz,) = gout
        is_continuous = [
            (inputs[i].dtype in tensor_continuous_dtypes) for i in range(len(inputs))
        ]

        if _is_sparse_variable(gz):
            gz = dense_from_sparse(gz)

        split = Split(len(inputs))(gz, 0, ptb.stack([x.shape[0] for x in inputs]))
        if not isinstance(split, list):
            split = [split]

        derivative = [SparseFromDense(self.format)(s) for s in split]

        def choose(continuous, derivative):
            if continuous:
                return derivative
            else:
                return None

        return [choose(c, d) for c, d in zip(is_continuous, derivative, strict=True)]

    def infer_shape(self, fgraph, node, ins_shapes):
        d = sum(shape[0] for shape in ins_shapes)
        return [(d, ins_shapes[0][1])]


def vstack(blocks, format=None, dtype=None):
    """
    Stack sparse matrices vertically (row wise).

    This wrap the method vstack from scipy.

    Parameters
    ----------
    blocks
        List of sparse array of compatible shape.
    format
        String representing the output format. Default is csc.
    dtype
        Output dtype.

    Returns
    -------
    array
        The concatenation of the sparse array row wise.

    Notes
    -----
    The number of column of the sparse matrix must agree.

    The grad implemented is regular, i.e. not structured.

    """

    blocks = [as_sparse_variable(i) for i in blocks]
    if dtype is None:
        dtype = ps.upcast(*[i.dtype for i in blocks])
    return VStack(format=format, dtype=dtype)(*blocks)


class Remove0(Op):
    """Remove explicit zeros from a sparse matrix.

    Notes
    -----
    The grad implemented is regular, i.e. not structured.

    """

    __props__ = ("inplace",)

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def __str__(self):
        l = []
        if self.inplace:
            l.append("inplace")
        return f"{self.__class__.__name__}{{{', '.join(l)}}}"

    def make_node(self, x):
        """

        Parameters
        ----------
        x
            Sparse matrix.

        """
        x = as_sparse_variable(x)
        assert x.format in ("csr", "csc")
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        if self.inplace:
            c = x
        else:
            c = x.copy()
        c.eliminate_zeros()
        z[0] = c

    def grad(self, inputs, gout):
        (_x,) = inputs
        (gz,) = gout
        return [gz]

    def infer_shape(self, fgraph, node, i0_shapes):
        return i0_shapes


remove0 = Remove0()


class ConstructSparseFromList(Op):
    """Constructs a sparse matrix out of a list of 2-D matrix rows.

    Notes
    -----
    The grad implemented is regular, i.e. not structured.

    """

    __props__ = ()

    def make_node(self, x, values, ilist):
        """

        This creates a sparse matrix with the same shape as `x`. Its
        values are the rows of `values` moved.  It operates similar to
        the following pseudo-code:

        .. code-block:: python

            output = csc_matrix.zeros_like(x, dtype=values.dtype)
            for in_idx, out_idx in enumerate(ilist):
                output[out_idx] = values[in_idx]


        Parameters
        ----------
        x
            A dense matrix that specifies the output shape.
        values
            A dense matrix with the values to use for output.
        ilist
            A dense vector with the same length as the number of rows of values.
            It specifies where in the output to put the corresponding rows.

        """
        x_ = ptb.as_tensor_variable(x)
        values_ = ptb.as_tensor_variable(values)
        ilist_ = ptb.as_tensor_variable(ilist)

        if ilist_.type.dtype not in integer_dtypes:
            raise TypeError("index must be integers")
        if ilist_.type.ndim != 1:
            raise TypeError("index must be vector")
        if x_.type.ndim != 2:
            raise TypeError(
                f"cannot create a sparse matrix with {int(x_.type.ndim)} dimensions"
            )
        if values_.type.ndim != 2:
            raise TypeError(
                f"cannot create a sparse matrix from values with {int(values_.type.ndim)} ndim"
            )

        # We only need the shape of `x` in the perform
        # If we keep in the graph the x variable as input of the Apply node,
        # this can rise the memory usage. That is why the Apply node
        # take `x_.shape` as input and not `x`.
        return Apply(self, [x_.shape, values_, ilist_], [csc_matrix(dtype=x.dtype)])

    def perform(self, node, inp, out_):
        out_shape, values, ilist = inp
        (out,) = out_
        rows, cols = values.shape
        assert rows == len(ilist)
        indptr = np.arange(cols + 1) * rows
        indices = as_strided(
            ilist, strides=(0, ilist.strides[0]), shape=(cols, ilist.shape[0])
        ).flatten()
        data = values.T.flatten()
        out[0] = scipy.sparse.csc_matrix(
            (data, indices, indptr), shape=out_shape, dtype=values.dtype
        )

    def infer_shape(self, fgraph, node, ishapes):
        x = node.inputs[0]
        return [[x[0], x[1]]]

    def R_op(self, inputs, eval_points):
        if None in eval_points[:2]:
            return [None]
        return self.make_node(eval_points[0], eval_points[1], *inputs[2:]).outputs

    def connection_pattern(self, node):
        rval = [[True], [True], [False]]
        return rval

    def grad(self, inputs, grads):
        (g_output,) = grads
        _x, _y = inputs[:2]
        idx_list = inputs[2:]

        gx = g_output
        gy = pytensor.tensor.subtensor.advanced_subtensor1(g_output, *idx_list)

        return [gx, gy, *(disconnected_type() for _ in range(len(idx_list)))]


construct_sparse_from_list = ConstructSparseFromList()
