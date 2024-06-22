r"""`Op` classes for working with ``numpy.ndarrays`` symbolically.

This module primarily defines `Op`\s for the creation, conversion, and
manipulation of tensors.

"""

import builtins
import warnings
from collections.abc import Sequence
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING
from typing import cast as type_cast

import numpy as np
from numpy.core.multiarray import normalize_axis_index
from numpy.core.numeric import normalize_axis_tuple

import pytensor
import pytensor.scalar.sharedvar
from pytensor import compile, config, printing
from pytensor import scalar as ps
from pytensor.gradient import DisconnectedType, grad_undefined
from pytensor.graph import RewriteDatabaseQuery
from pytensor.graph.basic import Apply, Constant, Variable, equal_computations
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.graph.replace import _vectorize_node
from pytensor.graph.rewriting.db import EquilibriumDB
from pytensor.graph.type import HasShape, Type
from pytensor.link.c.op import COp
from pytensor.link.c.params_type import ParamsType
from pytensor.misc.safe_asarray import _asarray
from pytensor.printing import Printer, min_informative_str, pprint, set_precedence
from pytensor.raise_op import CheckAndRaise, assert_op
from pytensor.scalar import int32
from pytensor.scalar.basic import ScalarConstant, ScalarVariable
from pytensor.tensor import (
    _as_tensor_variable,
    _get_vector_length,
    as_tensor_variable,
    get_vector_length,
)
from pytensor.tensor.blockwise import Blockwise, vectorize_node_fallback
from pytensor.tensor.elemwise import (
    DimShuffle,
    Elemwise,
    get_normalized_batch_axes,
    scalar_elemwise,
)
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.shape import (
    Shape,
    Shape_i,
    Unbroadcast,
    shape,
    shape_padaxis,
    shape_padleft,
    shape_padright,
    shape_tuple,
    specify_broadcastable,
)
from pytensor.tensor.type import (
    TensorType,
    discrete_dtypes,
    float_dtypes,
    int_dtypes,
    integer_dtypes,
    tensor,
    uint_dtypes,
    values_eq_approx_always_true,
)
from pytensor.tensor.variable import (
    TensorConstant,
    TensorVariable,
    get_unique_constant_value,
)


if TYPE_CHECKING:
    from pytensor.tensor import TensorLike


def __oplist_tag(thing, tag):
    tags = getattr(thing, "__oplist_tags", [])
    tags.append(tag)
    thing.__oplist_tags = tags


@_as_tensor_variable.register(Apply)
def _as_tensor_Apply(x, name, ndim, **kwargs):
    # use Apply's default output mechanism
    if (x.op.default_output is None) and (len(x.outputs) != 1):
        raise TypeError(
            "Multi-output Op without default_output encountered. "
            "Retry using only one of the outputs directly."
        )

    x = x.default_output()

    return as_tensor_variable(x, name=name, ndim=ndim, **kwargs)


@_as_tensor_variable.register(ScalarVariable)
@_as_tensor_variable.register(ScalarConstant)
def _as_tensor_Scalar(x, name, ndim, **kwargs):
    return as_tensor_variable(tensor_from_scalar(x), name=name, ndim=ndim, **kwargs)


@_as_tensor_variable.register(Variable)
def _as_tensor_Variable(x, name, ndim, **kwargs):
    if not isinstance(x.type, TensorType):
        raise TypeError(
            f"Tensor type field must be a TensorType; found {type(x.type)}."
        )

    if ndim is None:
        return x

    if x.type.ndim > ndim:
        # Strip off leading broadcastable dimensions
        non_broadcastables = tuple(
            idx for idx in range(x.type.ndim) if x.type.shape[idx] != 1
        )

        if non_broadcastables:
            x = x.dimshuffle(list(range(x.type.ndim))[non_broadcastables[0] :])
        else:
            x = x.dimshuffle()

        if x.ndim > ndim:
            raise ValueError(
                f"Tensor of type {x.type} could not be cast to have {ndim} dimensions"
            )
        return x
    elif x.type.ndim < ndim:
        return shape_padleft(x, n_ones=(ndim - x.type.ndim))
    else:
        return x


@_as_tensor_variable.register(list)
@_as_tensor_variable.register(tuple)
def _as_tensor_Sequence(x, name, ndim, dtype=None, **kwargs):
    if len(x) == 0:
        return constant(x, name=name, ndim=ndim, dtype=dtype)

    # If a sequence has `Variable`s in it, then we want
    # to customize the conversion to a tensor type.
    def extract_constants(i):
        if isinstance(i, Variable):
            if isinstance(i, Constant):
                return i.data
            else:
                raise TypeError
        else:
            return i

    try:
        x = type(x)(extract_constants(i) for i in x)
    except TypeError:
        if builtins.all(getattr(i, "ndim", None) == 0 for i in x) and (
            ndim is None or ndim == 1
        ):
            # In this instance, we have a sequence of constants with which we
            # want to construct a vector, so we can use `MakeVector` directly.
            if dtype is None:
                dtype = ps.upcast(*[i.dtype for i in x if hasattr(i, "dtype")])
            return MakeVector(dtype)(*x)

        # In this case, we have at least one non-`Constant` term, so we
        # couldn't get an underlying non-symbolic sequence of objects and we to
        # symbolically join terms.
        return stack(x)

    return constant(x, name=name, ndim=ndim, dtype=dtype)


@_as_tensor_variable.register(np.bool_)
@_as_tensor_variable.register(np.number)
@_as_tensor_variable.register(Number)
@_as_tensor_variable.register(np.ndarray)
def _as_tensor_numbers(x, name, ndim, dtype=None, **kwargs):
    return constant(x, name=name, ndim=ndim, dtype=dtype)


@_as_tensor_variable.register(bool)
def _as_tensor_bool(x, name, ndim, **kwargs):
    raise TypeError(
        "Cannot cast True or False as a tensor variable. Please use "
        "np.array(True) or np.array(False) if you need these constants. "
        "This error might be caused by using the == operator on "
        "Variables. v == w does not do what you think it does, "
        "use pytensor.tensor.eq(v, w) instead."
    )


as_tensor = as_tensor_variable


def constant(x, name=None, ndim=None, dtype=None) -> TensorConstant:
    """Return a `TensorConstant` with value `x`.

    Raises
    ------
    TypeError
        `x` could not be converted to a numpy.ndarray.
    ValueError
        `x` could not be expanded to have ndim dimensions.

    """
    if isinstance(x, TensorConstant):
        if (
            (name is None or x.name == name)
            and (ndim is None or x.ndim == ndim)
            and (dtype is None or x.dtype == dtype)
        ):
            return x
        else:
            x = x.data

    x_ = ps.convert(x, dtype=dtype)

    if ndim is not None:
        if x_.ndim < ndim:
            x_ = np.expand_dims(x_, axis=tuple(range(ndim - x_.ndim)))
        elif x_.ndim > ndim:
            try:
                x_ = np.squeeze(x_, axis=tuple(range(x_.ndim - ndim)))
            except np.AxisError:
                raise ValueError(
                    f"ndarray could not be cast to constant with {int(ndim)} dimensions"
                )

        assert x_.ndim == ndim

    ttype = TensorType(dtype=x_.dtype, shape=x_.shape)

    return TensorConstant(ttype, x_, name=name)


def _obj_is_wrappable_as_tensor(x):
    try:
        constant(x)
        return True
    except TypeError:
        return False


_scalar_constant_value_elemwise_ops = (
    ps.Cast,
    ps.Switch,
    ps.NEQ,
    ps.EQ,
    ps.LT,
    ps.GT,
    ps.LE,
    ps.GE,
    ps.Sub,
    ps.Add,
    ps.Mod,
    ps.Mul,
    ps.IntDiv,
    ps.TrueDiv,
    ps.ScalarMinimum,
    ps.ScalarMaximum,
)


def get_scalar_constant_value(
    v, elemwise=True, only_process_constants=False, max_recur=10
):
    """
    Checks whether 'v' is a scalar (ndim = 0).

    If 'v' is a scalar then this function fetches the underlying constant by calling
    'get_underlying_scalar_constant_value()'.

    If 'v' is not a scalar, it raises a NotScalarConstantError.

    """
    if isinstance(v, Variable | np.ndarray):
        if v.ndim != 0:
            raise NotScalarConstantError()
    return get_underlying_scalar_constant_value(
        v, elemwise, only_process_constants, max_recur
    )


def get_underlying_scalar_constant_value(
    orig_v, elemwise=True, only_process_constants=False, max_recur=10
):
    """Return the constant scalar(0-D) value underlying variable `v`.

    If `v` is the output of dimshuffles, fills, allocs, etc,
    cast, OutputGuard, DeepCopyOp, ScalarFromTensor, ScalarOp, Elemwise
    and some pattern with Subtensor, this function digs through them.

    If `v` is not some view of constant scalar data, then raise a
    NotScalarConstantError.

    Parameters
    ----------
    elemwise : bool
        If False, we won't try to go into elemwise. So this call is faster.
        But we still investigate in Second Elemwise (as this is a substitute
        for Alloc)
    only_process_constants : bool
        If True, we only attempt to obtain the value of `orig_v` if it's
        directly constant and don't try to dig through dimshuffles, fills,
        allocs, and other to figure out its value.
    max_recur : int
        The maximum number of recursion.

    Notes
    -----
        There may be another function similar to this one in the code,
        but I'm not sure where it is.

    """
    v = orig_v
    while True:
        if v is None:
            # None is not a scalar (and many uses of this function seem
            # to depend on passing it None)
            raise NotScalarConstantError()

        if isinstance(v, np.integer | int | float):
            return np.asarray(v)

        if isinstance(v, np.ndarray):
            try:
                return np.array(v.item(), dtype=v.dtype)
            except ValueError:
                raise NotScalarConstantError()

        if isinstance(v, Constant):
            unique_value = get_unique_constant_value(v)
            if unique_value is not None:
                data = unique_value
            else:
                data = v.data

            if isinstance(data, np.ndarray):
                try:
                    return np.array(data.item(), dtype=v.dtype)
                except ValueError:
                    raise NotScalarConstantError()

            from pytensor.sparse.type import SparseTensorType

            if isinstance(v.type, SparseTensorType):
                raise NotScalarConstantError()

            return data

        if not only_process_constants and getattr(v, "owner", None) and max_recur > 0:
            max_recur -= 1
            if isinstance(
                v.owner.op,
                Alloc
                | DimShuffle
                | Unbroadcast
                | compile.ops.OutputGuard
                | compile.DeepCopyOp,
            ):
                # OutputGuard is only used in debugmode but we
                # keep it here to avoid problems with old pickles
                v = v.owner.inputs[0]
                continue
            elif isinstance(v.owner.op, Shape_i):
                i = v.owner.op.i
                inp = v.owner.inputs[0]
                if isinstance(inp, Constant):
                    return np.asarray(np.shape(inp.data)[i])
                # The shape of a broadcastable dimension is 1
                if isinstance(inp.type, HasShape) and inp.type.shape[i] is not None:
                    return np.asarray(inp.type.shape[i])

            # Don't act as the constant_folding optimization here as this
            # fct is used too early in the optimization phase.  This would
            # mess with the stabilization optimization and be too slow.
            # We put all the scalar Ops used by get_canonical_form_slice()
            # to allow it to determine the broadcast pattern correctly.
            elif isinstance(v.owner.op, ScalarFromTensor | TensorFromScalar):
                v = v.owner.inputs[0]
                continue
            elif isinstance(v.owner.op, CheckAndRaise):
                # check if all conditions are constant and true
                conds = [
                    get_underlying_scalar_constant_value(c, max_recur=max_recur)
                    for c in v.owner.inputs[1:]
                ]
                if builtins.all(0 == c.ndim and c != 0 for c in conds):
                    v = v.owner.inputs[0]
                    continue
            elif isinstance(v.owner.op, ps.ScalarOp):
                if isinstance(v.owner.op, ps.Second):
                    # We don't need both input to be constant for second
                    shp, val = v.owner.inputs
                    v = val
                    continue
                if isinstance(v.owner.op, _scalar_constant_value_elemwise_ops):
                    const = [
                        get_underlying_scalar_constant_value(i, max_recur=max_recur)
                        for i in v.owner.inputs
                    ]
                    ret = [[None]]
                    v.owner.op.perform(v.owner, const, ret)
                    return np.asarray(ret[0][0].copy())
            # In fast_compile, we don't enable local_fill_to_alloc, so
            # we need to investigate Second as Alloc. So elemwise
            # don't disable the check for Second.
            elif isinstance(v.owner.op, Elemwise):
                if isinstance(v.owner.op.scalar_op, ps.Second):
                    # We don't need both input to be constant for second
                    shp, val = v.owner.inputs
                    v = val
                    continue
                elif elemwise and isinstance(
                    v.owner.op.scalar_op, _scalar_constant_value_elemwise_ops
                ):
                    const = [
                        get_underlying_scalar_constant_value(i, max_recur=max_recur)
                        for i in v.owner.inputs
                    ]
                    ret = [[None]]
                    v.owner.op.perform(v.owner, const, ret)
                    return np.asarray(ret[0][0].copy())
            elif (
                isinstance(v.owner.op, pytensor.tensor.subtensor.Subtensor)
                and v.ndim == 0
            ):
                if isinstance(v.owner.inputs[0], TensorConstant):
                    from pytensor.tensor.subtensor import get_constant_idx

                    cdata = tuple(get_constant_idx(v.owner.op.idx_list, v.owner.inputs))
                    try:
                        return np.asarray(
                            v.owner.inputs[0].data.__getitem__(cdata).copy()
                        )
                    except IndexError:
                        raise IndexError(
                            str(tuple(v.owner.op.idx_list))
                            + " is not a valid index into "
                            + str(v.owner.inputs[0].data)
                        )

                # The index list 'idx_list' should have length the same
                # shape as the input.
                # TODO: implement the case where we take a scalar in a matrix
                assert len(v.owner.op.idx_list) == v.owner.inputs[0].ndim

                # Needed to make better graph in this test in
                # pytensor/tensor/tests/test_sharedvar.py:
                # test_shared_options.test_specify_shape_partial
                if (
                    v.owner.inputs[0].owner
                    and isinstance(v.owner.inputs[0].owner.op, Join)
                    and len(v.owner.op.idx_list) == 1
                ):
                    # Ensure the Join is joining only (effectively) scalar
                    # variables (so that the constant value can be found at the
                    # same index as the one used in the sub-tensor).
                    if builtins.all(
                        var.ndim == 1 for var in v.owner.inputs[0].owner.inputs[1:]
                    ):
                        idx = v.owner.op.idx_list[0]
                        if isinstance(idx, Type):
                            idx = get_underlying_scalar_constant_value(
                                v.owner.inputs[1], max_recur=max_recur
                            )
                        try:
                            # TODO: assert joined axis is 0.
                            length = 0
                            loop = False
                            for joined in v.owner.inputs[0].owner.inputs[1:]:
                                ll = get_vector_length(joined)
                                if idx < length + ll:
                                    v = joined[idx - length]
                                    loop = True
                                    break
                                length += ll
                            if loop:
                                continue
                        except TypeError:
                            pass
                        except ValueError:
                            pass

                elif (
                    v.owner.inputs[0].owner
                    and isinstance(v.owner.inputs[0].owner.op, MakeVector)
                    and
                    # MakeVector normally accept only scalar as input.
                    # We put this check in case there is change in the future
                    builtins.all(
                        var.ndim == 0 for var in v.owner.inputs[0].owner.inputs
                    )
                    and len(v.owner.op.idx_list) == 1
                ):
                    idx = v.owner.op.idx_list[0]
                    if isinstance(idx, Type):
                        idx = get_underlying_scalar_constant_value(
                            v.owner.inputs[1], max_recur=max_recur
                        )
                    # Python 2.4 does not support indexing with numpy.integer
                    # So we cast it.
                    idx = int(idx)
                    ret = v.owner.inputs[0].owner.inputs[idx]
                    ret = get_underlying_scalar_constant_value(ret, max_recur=max_recur)
                    # MakeVector can cast implicitly its input in some case.
                    return _asarray(ret, dtype=v.type.dtype)

                # This is needed when we take the grad as the Shape op
                # are not already changed into MakeVector
                owner = v.owner
                leftmost_parent = owner.inputs[0]
                if leftmost_parent.owner and isinstance(
                    leftmost_parent.owner.op, Shape
                ):
                    op = owner.op
                    idx_list = op.idx_list
                    idx = idx_list[0]
                    if isinstance(idx, Type):
                        idx = get_underlying_scalar_constant_value(
                            owner.inputs[1], max_recur=max_recur
                        )
                    grandparent = leftmost_parent.owner.inputs[0]
                    gp_shape = grandparent.type.shape
                    ndim = grandparent.type.ndim
                    if grandparent.owner and isinstance(
                        grandparent.owner.op, Unbroadcast
                    ):
                        ggp_shape = grandparent.owner.inputs[0].type.shape
                        l = [get_underlying_scalar_constant_value(s) for s in ggp_shape]
                        gp_shape = tuple(l)

                    if not (idx < ndim):
                        msg = (
                            "get_underlying_scalar_constant_value detected "
                            f"deterministic IndexError: x.shape[{int(idx)}] "
                            f"when x.ndim={int(ndim)}."
                        )
                        if config.exception_verbosity == "high":
                            msg += f" x={min_informative_str(v)}"
                        else:
                            msg += f" x={v}"
                        raise ValueError(msg)

                    gp_shape_val = gp_shape[idx]
                    if gp_shape_val is not None and gp_shape_val > -1:
                        return np.asarray(gp_shape_val)

                    if isinstance(grandparent, Constant):
                        return np.asarray(np.shape(grandparent.data)[idx])

        raise NotScalarConstantError()


class TensorFromScalar(COp):
    __props__ = ()

    def make_node(self, s):
        if not isinstance(s.type, ps.ScalarType):
            raise TypeError("Input must be a `ScalarType` `Type`")

        return Apply(self, [s], [tensor(dtype=s.type.dtype, shape=())])

    def perform(self, node, inp, out_):
        (s,) = inp
        (out,) = out_
        out[0] = np.asarray(s)

    def infer_shape(self, fgraph, node, in_shapes):
        return [()]

    def grad(self, inp, grads):
        (s,) = inp
        (dt,) = grads
        if s.type.dtype in float_dtypes:
            assert dt.type.dtype in float_dtypes
            return [scalar_from_tensor(dt)]

        # If the input dtype is an integer, then so is the output dtype,
        # and the "zero" gradient can be represented in that int dtype.
        # Currently, pytensor.grad insists that the dtype of the returned
        # gradient has a float dtype, so we use floatX.
        if s.type.dtype in discrete_dtypes:
            return [s.zeros_like().astype(config.floatX)]

        raise NotImplementedError("grad not implemented for complex dtypes")

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        fail = sub["fail"]

        return """
            {z} = (PyArrayObject*)PyArray_FromScalar(py_{x}, NULL);
            if({z} == NULL){{
                {fail};
            }}
            """.format(**locals())

    def c_code_cache_version(self):
        return (2,)


tensor_from_scalar = TensorFromScalar()


class ScalarFromTensor(COp):
    __props__ = ()

    def __call__(self, *args, **kwargs) -> ScalarVariable:
        return type_cast(ScalarVariable, super().__call__(*args, **kwargs))

    def make_node(self, t):
        if not isinstance(t.type, TensorType) or t.type.ndim > 0:
            raise TypeError("Input must be a scalar `TensorType`")

        return Apply(
            self, [t], [ps.get_scalar_type(dtype=t.type.dtype).make_variable()]
        )

    def perform(self, node, inp, out_):
        (s,) = inp
        (out,) = out_
        out[0] = s.flatten()[0]

    def infer_shape(self, fgraph, node, in_shapes):
        return [()]

    def grad(self, inp, grads):
        (s,) = inp
        (dt,) = grads
        return [tensor_from_scalar(dt)]

    def R_op(self, inputs, eval_points):
        if None in eval_points:
            return [None]
        return self.make_node(*eval_points).outputs

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        fail = sub["fail"]
        return """
        {z} = ((dtype_{x}*)(PyArray_DATA({x})))[0];
        """.format(**locals())

    def c_code_cache_version(self):
        return (1,)


scalar_from_tensor = ScalarFromTensor()


# to be removed as we get the epydoc routine-documenting thing going
# -JB 20080924
def _conversion(real_value: Op, name: str) -> Op:
    __oplist_tag(real_value, "casting")
    real_value.__module__ = "tensor.basic"
    pprint.assign(real_value, printing.FunctionPrinter([name]))
    return real_value


# These _convert_to_<type> functions have leading underscores to indicate that
# they should not be called directly.  They do not perform sanity checks about
# what types you are casting to what.  That logic is implemented by the
# `cast()` function below.

_convert_to_bool: Elemwise = _conversion(Elemwise(ps.convert_to_bool), "bool")
"""Cast to boolean"""

_convert_to_int8: Elemwise = _conversion(Elemwise(ps.convert_to_int8), "int8")
"""Cast to 8-bit integer"""

_convert_to_int16: Elemwise = _conversion(Elemwise(ps.convert_to_int16), "int16")
"""Cast to 16-bit integer"""

_convert_to_int32: Elemwise = _conversion(Elemwise(ps.convert_to_int32), "int32")
"""Cast to 32-bit integer"""

_convert_to_int64: Elemwise = _conversion(Elemwise(ps.convert_to_int64), "int64")
"""Cast to 64-bit integer"""

_convert_to_uint8: Elemwise = _conversion(Elemwise(ps.convert_to_uint8), "uint8")
"""Cast to unsigned 8-bit integer"""

_convert_to_uint16: Elemwise = _conversion(Elemwise(ps.convert_to_uint16), "uint16")
"""Cast to unsigned 16-bit integer"""

_convert_to_uint32: Elemwise = _conversion(Elemwise(ps.convert_to_uint32), "uint32")
"""Cast to unsigned 32-bit integer"""

_convert_to_uint64: Elemwise = _conversion(Elemwise(ps.convert_to_uint64), "uint64")
"""Cast to unsigned 64-bit integer"""

_convert_to_float16: Elemwise = _conversion(Elemwise(ps.convert_to_float16), "float16")
"""Cast to half-precision floating point"""

_convert_to_float32: Elemwise = _conversion(Elemwise(ps.convert_to_float32), "float32")
"""Cast to single-precision floating point"""

_convert_to_float64: Elemwise = _conversion(Elemwise(ps.convert_to_float64), "float64")
"""Cast to double-precision floating point"""

_convert_to_complex64: Elemwise = _conversion(
    Elemwise(ps.convert_to_complex64), "complex64"
)
"""Cast to single-precision complex"""

_convert_to_complex128: Elemwise = _conversion(
    Elemwise(ps.convert_to_complex128), "complex128"
)
"""Cast to double-precision complex"""

_cast_mapping = {
    "bool": _convert_to_bool,
    "int8": _convert_to_int8,
    "int16": _convert_to_int16,
    "int32": _convert_to_int32,
    "int64": _convert_to_int64,
    "uint8": _convert_to_uint8,
    "uint16": _convert_to_uint16,
    "uint32": _convert_to_uint32,
    "uint64": _convert_to_uint64,
    "float16": _convert_to_float16,
    "float32": _convert_to_float32,
    "float64": _convert_to_float64,
    "complex64": _convert_to_complex64,
    "complex128": _convert_to_complex128,
}


def cast(x, dtype: str | np.dtype) -> TensorVariable:
    """Symbolically cast `x` to a Tensor of type `dtype`."""

    if isinstance(dtype, str) and dtype == "floatX":
        dtype = config.floatX

    dtype_name = np.dtype(dtype).name

    _x = as_tensor_variable(x)
    if _x.type.dtype == dtype_name:
        return _x
    if _x.type.dtype.startswith("complex") and not dtype_name.startswith("complex"):
        raise TypeError(
            "Casting from complex to real is ambiguous: consider real(), "
            "imag(), angle() or abs()"
        )
    return _cast_mapping[dtype_name](x)


@scalar_elemwise
def switch(cond, ift, iff):
    """if cond then ift else iff"""


# where = switch
def where(cond, ift, iff):
    if ift is not None and iff is not None:
        return switch(cond, ift, iff)
    else:
        return cond.nonzero()


@scalar_elemwise
def second(a, b):
    """Create a matrix by filling the broadcasted shapes of a and b with the values of b

    Equivalent to `np.broadcast_arrays(a, b)[1]`
    Equivalent to `np.array(a).fill(b)` when b is a scalar value.

    """


fill = second
pprint.assign(fill, printing.FunctionPrinter(["fill"]))


def ones_like(model, dtype=None, opt=False):
    """equivalent of numpy.ones_like
    Parameters
    ----------
    model : tensor
    dtype : data-type, optional
    opt : If True, we will return a constant instead of a graph when possible.
          Useful for PyTensor optimization, not for user building a graph as this
          have the consequence that model isn't always in the graph.

    Returns
    -------
    tensor
        tensor the shape of model containing ones of the type of dtype.
    """
    _model = as_tensor_variable(model)

    if dtype is None:
        dtype = _model.type.dtype
    ret = constant(1.0, dtype=dtype)
    # TODO: Remove this weird option
    if opt and ret.type == _model.type:
        return ret
    return fill(_model, ret)


def zeros_like(model, dtype=None, opt=False):
    """equivalent of numpy.zeros_like
    Parameters
    ----------
    model : tensor
    dtype : data-type, optional
    opt : If True, we will return a constant instead of a graph when possible.
          Useful for PyTensor optimization, not for user building a graph as this
          have the consequence that model isn't always in the graph.

    Returns
    -------
    tensor
        tensor the shape of model containing zeros of the type of dtype.
    """

    _model = as_tensor_variable(model)

    if dtype is None:
        dtype = _model.type.dtype
    ret = constant(0.0, dtype=dtype)
    # TODO: Remove this weird option
    if opt and ret.type == _model.type:
        return ret
    return fill(_model, ret)


def zeros(shape, dtype=None):
    """Create a `TensorVariable` filled with zeros, closer to NumPy's syntax than ``alloc``."""
    if not (
        isinstance(shape, np.ndarray | Sequence)
        or (isinstance(shape, TensorVariable) and shape.ndim > 0)
    ):
        shape = [shape]
    if dtype is None:
        dtype = config.floatX
    return alloc(np.array(0, dtype=dtype), *shape)


def ones(shape, dtype=None):
    """Create a `TensorVariable` filled with ones, closer to NumPy's syntax than ``alloc``."""
    if not (
        isinstance(shape, np.ndarray | Sequence)
        or (isinstance(shape, TensorVariable) and shape.ndim > 0)
    ):
        shape = [shape]
    if dtype is None:
        dtype = config.floatX
    return alloc(np.array(1, dtype=dtype), *shape)


class Nonzero(Op):
    """
    Return the indices of the elements that are non-zero.

    Parameters
    ----------
    a: array_like
        Input array.

    Returns
    -------
    indices: list
        A list containing the indices of the non-zero elements of `a`.

    See Also
    --------
    nonzero_values : Return the non-zero elements of the input array
    flatnonzero : Return the indices of the non-zero elements of the
        flattened input array.

    """

    __props__ = ()

    def make_node(self, a):
        a = as_tensor_variable(a)
        if a.ndim == 0:
            raise ValueError("Nonzero only supports non-scalar arrays.")
        output = [TensorType(dtype="int64", shape=(None,))() for i in range(a.ndim)]
        return Apply(self, [a], output)

    def perform(self, node, inp, out_):
        a = inp[0]

        result_tuple = np.nonzero(a)
        for i, res in enumerate(result_tuple):
            out_[i][0] = res.astype("int64")

    def grad(self, inp, grads):
        return [grad_undefined(self, 0, inp[0])]


_nonzero = Nonzero()


def nonzero(a, return_matrix=False):
    """
    Returns one of the following:

        If return_matrix is False (default, same as NumPy):
            A tuple of vector arrays such that the ith element of the jth array
            is the index of the ith non-zero element of the input array in the
            jth dimension.

        If return_matrix is True (same as PyTensor Op):
            Returns a matrix of shape (ndim, number of nonzero elements) such
            that element (i,j) is the index in the ith dimension of the jth
            non-zero element.

    Parameters
    ----------
    a : array_like
        Input array.
    return_matrix : bool
        If True, returns a symbolic matrix. If False, returns a tuple of
        arrays. Defaults to False.

    Returns
    -------
    tuple of vectors or matrix

    See Also
    --------
    nonzero_values : Return the non-zero elements of the input array
    flatnonzero : Return the indices of the non-zero elements of the
        flattened input array.

    """
    res = _nonzero(a)
    if isinstance(res, list):
        res = tuple(res)
    else:
        res = (res,)

    if return_matrix:
        if len(res) > 1:
            return stack(res, 0)
        elif len(res) == 1:
            return shape_padleft(res[0])
    else:
        return res


def flatnonzero(a):
    """Return a vector of indices that are non-zero in the flattened version of `a`.

    Parameters
    ----------
    a : tensor
        Input tensor

    Returns
    -------
    vector
        Output vector, containing the indices of the elements of `a.flatten()`
        that are non-zero.

    See Also
    --------
    nonzero : Return the indices of the non-zero elements of the input array.
    nonzero_values : Return the non-zero elements of the input array

    """
    _a = as_tensor_variable(a)
    if _a.ndim == 0:
        raise ValueError("Nonzero only supports non-scalar arrays.")
    return nonzero(_a.flatten(), return_matrix=False)[0]


def nonzero_values(a):
    """Return a vector of non-zero elements contained in the input array.

    Parameters
    ----------
    a : tensor
        Input tensor

    Returns
    -------
    vector
        Output vector, containing the non-zero elements of a.

    See Also
    --------
    nonzero : Return the indices of the non-zero elements of the input array.
    flatnonzero : Return the indices of the non-zero elements of the
        flattened input array.

    """
    _a = as_tensor_variable(a)
    return _a.flatten()[flatnonzero(_a)]


class Tri(Op):
    __props__ = ("dtype",)

    def __init__(self, dtype=None):
        if dtype is None:
            dtype = config.floatX
        self.dtype = dtype

    def make_node(self, N, M, k):
        N = as_tensor_variable(N)
        M = as_tensor_variable(M)
        k = as_tensor_variable(k)
        return Apply(
            self,
            [N, M, k],
            [TensorType(dtype=self.dtype, shape=(None, None))()],
        )

    def perform(self, node, inp, out_):
        N, M, k = inp
        (out,) = out_
        out[0] = np.tri(N, M, k, dtype=self.dtype)

    def infer_shape(self, fgraph, node, in_shapes):
        out_shape = [node.inputs[0], node.inputs[1]]
        return [out_shape]

    def grad(self, inp, grads):
        return [grad_undefined(self, i, inp[i]) for i in range(3)]


def tri(N, M=None, k=0, dtype=None):
    """
    An array with ones at and below the given diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the array.
    M : int, optional
        Number of columns in the array.
        By default, `M` is taken equal to `N`.
    k : int, optional
        The sub-diagonal at and below which the array is filled.
        `k` = 0 is the main diagonal, while `k` < 0 is below it,
        and `k` > 0 is above.  The default is 0.
    dtype : dtype, optional
        Data type of the returned array.  The default is float.

    Returns
    -------
    Array of shape (N, M)
        Array with its lower triangle filled with ones and zero elsewhere;
        in other words ``T[i,j] == 1`` for ``i <= j + k``, 0 otherwise.

    """
    if dtype is None:
        dtype = config.floatX
    if M is None:
        M = N
    op = Tri(dtype)
    return op(N, M, k)


def tril(m, k=0):
    """
    Lower triangle of an array.

    Return a copy of an array with elements above the `k`-th diagonal zeroed.
    For arrays with ``ndim`` exceeding 2, `tril` will apply to the final two
    axes.

    Parameters
    ----------
    m : array_like, shape (..., M, N)
        Input array.
    k : int, optional
        Diagonal above which to zero elements.  `k = 0` (the default) is the
        main diagonal, `k < 0` is below it and `k > 0` is above.

    Returns
    -------
    tril : ndarray, shape (..., M, N)
        Lower triangle of `m`, of same shape and data-type as `m`.

    See Also
    --------
    triu : Same thing, only for the upper triangle.

    Examples
    --------
    >>> at.tril(np.arange(1,13).reshape(4,3), -1).eval()
    array([[ 0,  0,  0],
           [ 4,  0,  0],
           [ 7,  8,  0],
           [10, 11, 12]])

    >>> at.tril(np.arange(3*4*5).reshape(3, 4, 5)).eval()
    array([[[ 0,  0,  0,  0,  0],
            [ 5,  6,  0,  0,  0],
            [10, 11, 12,  0,  0],
            [15, 16, 17, 18,  0]],

           [[20,  0,  0,  0,  0],
            [25, 26,  0,  0,  0],
            [30, 31, 32,  0,  0],
            [35, 36, 37, 38,  0]],

           [[40,  0,  0,  0,  0],
            [45, 46,  0,  0,  0],
            [50, 51, 52,  0,  0],
            [55, 56, 57, 58,  0]]])

    """
    return m * tri(*m.shape[-2:], k=k, dtype=m.dtype)


def triu(m, k=0):
    """
    Upper triangle of an array.

    Return a copy of an array with the elements below the `k`-th diagonal
    zeroed. For arrays with ``ndim`` exceeding 2, `triu` will apply to the
    final two axes.

    Please refer to the documentation for `tril` for further details.

    See Also
    --------
    tril : Lower triangle of an array.

    Examples
    --------
    >>> at.triu(np.arange(1,13).reshape(4,3), -1).eval()
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 0,  8,  9],
           [ 0,  0, 12]])

    >>> at.triu(np.arange(3*4*5).reshape(3, 4, 5)).eval()
    array([[[ 0,  1,  2,  3,  4],
            [ 0,  6,  7,  8,  9],
            [ 0,  0, 12, 13, 14],
            [ 0,  0,  0, 18, 19]],

           [[20, 21, 22, 23, 24],
            [ 0, 26, 27, 28, 29],
            [ 0,  0, 32, 33, 34],
            [ 0,  0,  0, 38, 39]],

           [[40, 41, 42, 43, 44],
            [ 0, 46, 47, 48, 49],
            [ 0,  0, 52, 53, 54],
            [ 0,  0,  0, 58, 59]]])

    """
    return m * (constant(1, dtype=m.dtype) - tri(*m.shape[-2:], k=k - 1, dtype=m.dtype))


def tril_indices(
    n: int | ScalarVariable,
    k: int | ScalarVariable = 0,
    m: int | ScalarVariable | None = None,
) -> tuple[TensorVariable, TensorVariable]:
    """
    Return the indices for the lower-triangle of an (n, m) array.

    Parameters
    ----------
    n : integer scalar
        The row dimension of the arrays for which the returned indices will be valid.
    k : integer scalar, optional
        Diagonal offset to use when forming the indices. `k = 0` (the default)
        is the main diagonal, `k < 0` is below it and `k > 0` is above.
    m : integer scalar, optional
        The column dimension of the arrays for which the returned arrays will
        be valid. By default m is taken equal to n.

    Returns
    -------
    inds : tuple of TensorVariable's
        The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.
    """
    return tri(n, m, k, dtype=bool).nonzero()


def tril_indices_from(
    a: np.ndarray | TensorVariable,
    k: int | ScalarVariable = 0,
) -> tuple[TensorVariable, TensorVariable]:
    """
    Return the indices for the lower-triangle of arr.

    Parameters
    ----------
    arr : {array_like, TensorVariable}, shape(N, N)
        The indices will be valid for square arrays.
    k : integer scalar, optional
        Diagonal offset to use when forming the indices. `k = 0` (the default)
        is the main diagonal, `k < 0` is below it and `k > 0` is above.

    Returns
    -------
    tril_indices_from : tuple, shape(2) of TensorVariable, shape(N)
        Indices for the lower-triangle of arr.

    Raises
    ------
    ValueError
        If the input is not a 2d array.
    """
    if a.ndim != 2:
        raise ValueError("The input array must be two dimensional.")
    return tril_indices(a.shape[0], k=k, m=a.shape[1])


def triu_indices(
    n: int | ScalarVariable,
    k: int | ScalarVariable = 0,
    m: int | ScalarVariable | None = None,
) -> tuple[TensorVariable, TensorVariable]:
    """
    Return the indices for the upper-triangle of an (n, m) array.

    Parameters
    ----------
    n : integer scalar
        The row dimension of the arrays for which the returned indices will be valid.
    k : integer scalar, optional
        Diagonal offset to use when forming the indices. `k = 0` (the default)
        is the main diagonal, `k < 0` is below it and `k > 0` is above.
    m : int scalar, optional
        The column dimension of the arrays for which the returned arrays will
        be valid. By default m is taken equal to n.

    Returns
    -------
    inds : tuple of TensorVariable's
        The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.
    """
    return (constant(1, dtype=int) - tri(n, m, k - 1, dtype=int)).nonzero()


def triu_indices_from(
    a: np.ndarray | TensorVariable,
    k: int | ScalarVariable = 0,
) -> tuple[TensorVariable, TensorVariable]:
    """
    Return the indices for the upper-triangle of arr.

    Parameters
    ----------
    arr : {array_like, TensorVariable}, shape(N, N)
        The indices will be valid for square arrays.
    k : integer scalar, optional
        Diagonal offset to use when forming the indices. `k = 0` (the default)
        is the main diagonal, `k < 0` is below it and `k > 0` is above.

    Returns
    -------
    triu_indices_from : tuple, shape(2) of TensorVariable, shape(N)
        Indices for the upper-triangle of arr.

    Raises
    ------
    ValueError
        If the input is not a 2d array.
    """
    if a.ndim != 2:
        raise ValueError("The input array must be two dimensional.")
    return triu_indices(a.shape[0], k=k, m=a.shape[1])


class Eye(Op):
    _output_type_depends_on_input_value = True
    __props__ = ("dtype",)

    def __init__(self, dtype=None):
        if dtype is None:
            dtype = config.floatX
        self.dtype = dtype

    def make_node(self, n, m, k):
        n = as_tensor_variable(n)
        m = as_tensor_variable(m)
        k = as_tensor_variable(k)
        assert n.ndim == 0
        assert m.ndim == 0
        assert k.ndim == 0

        _, static_shape = infer_static_shape((n, m))

        return Apply(
            self,
            [n, m, k],
            [TensorType(dtype=self.dtype, shape=static_shape)()],
        )

    def perform(self, node, inp, out_):
        n, m, k = inp
        (out,) = out_
        out[0] = np.eye(n, m, k, dtype=self.dtype)

    def infer_shape(self, fgraph, node, in_shapes):
        out_shape = [node.inputs[0], node.inputs[1]]
        return [out_shape]

    def grad(self, inp, grads):
        return [grad_undefined(self, i, inp[i]) for i in range(3)]


def eye(n, m=None, k=0, dtype=None):
    """Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n : int
        Number of rows in the output.
    m : int, optional
        Number of columns in the output. If None, defaults to `N`.
    k : int, optional
        Index of the diagonal: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal, and a negative value
        to a lower diagonal.
    dtype : data-type, optional
        Data-type of the returned array.

    Returns
    -------
    ndarray of shape (N,M)
        An array where all elements are equal to zero, except for the `k`-th
        diagonal, whose values are equal to one.

    """
    if dtype is None:
        dtype = config.floatX
    if m is None:
        m = n
    localop = Eye(dtype)
    return localop(n, m, k)


def identity_like(x, dtype: str | np.generic | np.dtype | None = None):
    """Create a tensor with ones on main diagonal and zeroes elsewhere.

    Parameters
    ----------
    x : tensor
    dtype : data-type, optional

    Returns
    -------
    tensor
        tensor the shape of x with ones on main diagonal and zeroes elsewhere of type of dtype.
    """
    _x = as_tensor_variable(x)
    if dtype is None:
        dtype = _x.dtype
    return eye(_x.shape[0], _x.shape[1], k=0, dtype=dtype)


class CachedEquilibrimDB(EquilibriumDB):
    """A subclass of EquilibriumDB that allows caching of a default query for faster reuse."""

    def __init__(self, default_query):
        super().__init__()
        self._default_query = default_query
        self._cached_default_query = None

    def register(self, *args, **kwargs):
        # If new rewrites are registered, the default cached query is void
        self.cached_default_query = None
        super().register(*args, **kwargs)

    @property
    def default_query(self):
        if self._cached_default_query is None:
            self._cached_default_query = self.query(self._default_query)
        return self._cached_default_query


infer_shape_db = CachedEquilibrimDB(
    default_query=RewriteDatabaseQuery(include=("infer_shape",))
)


def register_infer_shape(rewrite, *tags, **kwargs):
    if isinstance(rewrite, str):

        def register(inner_lopt):
            return register_infer_shape(inner_lopt, rewrite, *tags, **kwargs)

        return register
    else:
        name = kwargs.pop("name", None) or rewrite.__name__

        infer_shape_db.register(name, rewrite, *tags, "infer_shape", **kwargs)
        return rewrite


def infer_static_shape(
    shape: Variable | Sequence[Variable | int],
) -> tuple[Sequence["TensorLike"], Sequence[int | None]]:
    """Infer the static shapes implied by the potentially symbolic elements in `shape`.

    `shape` will be validated and constant folded.  As a result, this function
    can be expensive and shouldn't be used unless absolutely necessary.

    It is often needed for `Op`s whose static shape and broadcastable flags
    depend on the values of their inputs, such as `Alloc` and `RandomVariable`.

    Returns
    -------
    A validated sequence of symbolic shape values, and a sequence of
    ``None``/``int`` values that can be used as `TensorType.shape` values.

    """
    from pytensor.tensor.rewriting.basic import topo_constant_folding
    from pytensor.tensor.rewriting.shape import ShapeFeature

    def check_type(s):
        if s.type.dtype in integer_dtypes:
            return s

        if config.exception_verbosity == "high":
            s_as_str = "\n" + min_informative_str(s)
        else:
            s_as_str = str(s)

        raise TypeError(f"Shapes must be scalar integers; got {s_as_str}")

    sh = folded_shape = [check_type(as_tensor_variable(s, ndim=0)) for s in shape]

    if not all(isinstance(s, Constant) for s in folded_shape):
        shape_fg = FunctionGraph(outputs=sh, features=[ShapeFeature()], clone=True)
        with config.change_flags(optdb__max_use_ratio=10, cxx=""):
            infer_shape_db.default_query.rewrite(shape_fg)
            if not all(isinstance(s, Constant) for s in shape_fg.outputs):
                topo_constant_folding.rewrite(shape_fg)
        folded_shape = shape_fg.outputs

    static_shape = tuple(
        s.data.item() if isinstance(s, Constant) else None for s in folded_shape
    )
    return sh, static_shape


class Alloc(COp):
    """Create a `TensorVariable` from an initial value and a desired shape.

    Usage:

        alloc(value, shape0, shape1, ..., shapeN)

    Returns an N-dimensional tensor initialized by a value, using something
    equivalent to

        z = numpy.zeros(shape, value.dtype)
        z += value

    The result has N dimensions, has the dtype of the given value, and is
    obtained by broadcasting value over the output array.

    This `Op` is used to replace ``fill`` during optimizations, because, after
    shapes are lifted, the first argument to ``fill`` can often be pruned from
    the graph.

    """

    _f16_ok = True
    _output_type_depends_on_input_value = True

    __props__ = ()

    _runtime_broadcast_error_msg = (
        "Runtime broadcasting not allowed. "
        "The output of Alloc requires broadcasting a dimension of the input value, which was not marked as broadcastable. "
        "If broadcasting was intended, use `specify_broadcastable` on the relevant input."
    )

    def make_node(self, value, *shape):
        value = as_tensor_variable(value)
        shape, static_shape = infer_static_shape(shape)
        if value.ndim > len(shape):
            raise TypeError(
                "The Alloc value to use has more dimensions"
                " than the specified dimensions",
                value.ndim,
                len(shape),
            )

        # Combine static shape information from value and shape
        combined_static_shape = list(static_shape).copy()
        new_dims = len(shape) - value.type.ndim
        extended_value_static_shape = (None,) * new_dims + value.type.shape
        extended_value_broadcastable = (False,) * new_dims + value.type.broadcastable
        for i, (v_bc, v_st, sh_st) in enumerate(
            zip(
                extended_value_broadcastable,
                extended_value_static_shape,
                static_shape,
            )
        ):
            # If value is not broadcastable and we don't know the target static shape: use value static shape
            if (not v_bc) and (sh_st is None):
                combined_static_shape[i] = v_st
            # Otherwise check if static shapes are compatible
            elif (v_st is not None) and (sh_st is not None):
                # They must match or if not, the value must be broadcastable
                if v_st != sh_st and not v_bc:
                    raise ValueError(
                        f"Alloc static input type and target shape are incompatible: {value.type} vs {static_shape}"
                    )

        otype = TensorType(dtype=value.dtype, shape=combined_static_shape)
        return Apply(self, [value, *shape], [otype()])

    @staticmethod
    def _check_runtime_broadcast(node, value, shape):
        value_static_shape = node.inputs[0].type.shape
        for v_static_dim, value_dim, out_dim in zip(
            value_static_shape[::-1], value.shape[::-1], shape[::-1]
        ):
            if v_static_dim is None and value_dim == 1 and out_dim != 1:
                raise ValueError(Alloc._runtime_broadcast_error_msg)

    def perform(self, node, inputs, out_):
        (out,) = out_
        v = inputs[0]
        sh = tuple([int(i) for i in inputs[1:]])
        self._check_runtime_broadcast(node, v, sh)

        if out[0] is None or out[0].shape != sh:
            if v.size == 1 and v.item() == 0:
                out[0] = np.zeros(sh, dtype=v.dtype)
            else:
                out[0] = np.empty(sh, dtype=v.dtype)
                out[0][...] = v  # broadcast v to fill us up
        else:
            # reuse the allocated memory.
            out[0][...] = v  # broadcast v to fill us up

    def c_code(self, node, name, inp, out, sub):
        vv = inp[0]
        (zz,) = out
        fail = sub["fail"]

        v_static_shape = node.inputs[0].type.shape
        o_static_shape = node.outputs[0].type.shape
        v_ndim = len(v_static_shape)
        o_ndim = len(o_static_shape)
        assert o_ndim == len(inp[1:])

        # Declare variables
        code = f"""
            npy_intp shape[{o_ndim}];
            int need_new_out;
            """

        # Initialize shape
        for i, shp_i in enumerate(inp[1:]):
            code += f"""
                shape[{i}] = ((dtype_{shp_i}*) PyArray_DATA({shp_i}))[0];
            """

        # Add checks for runtime broadcasting
        for i, v_static_dim in enumerate(v_static_shape[::-1]):
            if v_static_dim is None:
                code += f"""
                if (PyArray_DIMS({vv})[{v_ndim - i - 1}] == 1 && shape[{o_ndim - i - 1}] != 1)
                {{
                    PyErr_Format(PyExc_ValueError, "{self._runtime_broadcast_error_msg}");
                    {fail}
                }}
                """

        code += f"""
            need_new_out = (NULL == {zz});
            for (int i = 0; i < {o_ndim}; i++)
                need_new_out = (need_new_out || (PyArray_DIMS({zz})[i] != shape[i]));

            if (need_new_out)
            {{
                Py_XDECREF({zz});
                {zz} = (PyArrayObject*) PyArray_SimpleNew({o_ndim}, shape, PyArray_TYPE({vv}));
                if (!{zz})
                {{
                    PyErr_SetString(PyExc_MemoryError, "alloc failed");
                    {fail}
                }}
            }}

            // This function takes care of broadcasting
            if (PyArray_CopyInto({zz}, {vv}) == -1)
              {fail}
            """

        return code

    def c_code_cache_version(self):
        return (4,)

    def infer_shape(self, fgraph, node, input_shapes):
        return [node.inputs[1:]]

    def connection_pattern(self, node):
        rval = [[True]]

        for ipt in node.inputs[1:]:
            rval.append([False])

        return rval

    def grad(self, inputs, grads):
        x = inputs[0]
        gz = grads[0]
        n_axes_to_sum = gz.ndim - x.ndim
        # The number of dimensions added
        axis = list(range(n_axes_to_sum))
        # The broadcasted dimensions
        axis_broadcasted = []
        axis_kept = []
        for i, (ib, gb) in enumerate(
            zip(
                inputs[0].type.shape,
                # We need the dimensions corresponding to x
                grads[0].type.shape[-inputs[0].ndim :],
            )
        ):
            if ib == 1 and gb != 1:
                axis_broadcasted.append(i + n_axes_to_sum)
            else:
                axis_kept.append(i)

        gx = gz.sum(axis=axis + axis_broadcasted)
        if axis_broadcasted:
            new_order = ["x"] * x.ndim
            for idx, axis in enumerate(axis_kept):
                new_order[axis] = idx
            gx = gx.dimshuffle(new_order)
        # Dimshuffle to add back the broadcasted dims
        # The *elements* of the output are not connected to
        # the inputs that specify the shape. If you grow the
        # shape by epsilon, the existing elements do not
        # change.
        return [gx] + [DisconnectedType()() for i in inputs[1:]]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self(eval_points[0], *inputs[1:], return_list=True)

    def do_constant_folding(self, fgraph, node):
        clients = fgraph.clients[node.outputs[0]]

        if not clients:
            return False

        for client, idx in clients:
            if client == "output":
                # If the output is a constant, it will have to be deepcopied
                # each time the function is called.  So we do not fold.
                return False
            # Allow alloc to be lifted out of Elemwise before constant folding it
            elif isinstance(client.op, Elemwise):
                return None
            # Same for Blockwise, unless it has no batch_dims
            elif isinstance(client.op, Blockwise) and client.op.batch_ndim(client):
                return None
            elif (
                # The following ops work inplace of their input id 0.
                idx == 0
                and isinstance(
                    client.op,
                    pytensor.tensor.subtensor.IncSubtensor
                    | pytensor.tensor.subtensor.AdvancedIncSubtensor1
                    | pytensor.tensor.subtensor.AdvancedIncSubtensor
                    | pytensor.tensor.blas.Gemv
                    | pytensor.tensor.blas_c.CGemv
                    | pytensor.tensor.blas.Ger
                    | pytensor.tensor.blas_c.CGer
                    | pytensor.tensor.blas_scipy.ScipyGer,
                )
            ):
                # Ops that will work inplace on the Alloc. So if they
                # get constant_folded, they would copy the constant
                # and this is less efficient.
                # Not doing the constant folding could also lower the
                # peak memory use, as the "constant" won't always exist.
                return False
        return True


alloc = Alloc()
pprint.assign(alloc, printing.FunctionPrinter(["alloc"]))


@_get_vector_length.register(Alloc)
def _get_vector_length_Alloc(var_inst, var):
    try:
        return get_underlying_scalar_constant_value(var.owner.inputs[1])
    except NotScalarConstantError:
        raise ValueError(f"Length of {var} cannot be determined")


def full(shape, fill_value, dtype=None):
    """Return a new array of given shape and type, filled with `fill_value`.

    See ``numpy.full``.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar or array_like
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array  The default, None, means
        `np.array(fill_value).dtype`.

    """
    fill_value = as_tensor_variable(fill_value)
    if dtype:
        fill_value = fill_value.astype(dtype)

    if np.ndim(shape) == 0:
        shape = (shape,)
    return alloc(fill_value, *shape)


def full_like(
    a: TensorVariable,
    fill_value: TensorVariable | int | float,
    dtype: str | np.generic | np.dtype = None,
) -> TensorVariable:
    """Equivalent of `numpy.full_like`.

    Returns
    -------
    tensor
        tensor the shape of `a` containing `fill_value` of the type of dtype.
    """
    fill_value = as_tensor_variable(fill_value)
    if dtype is not None:
        fill_value = fill_value.astype(dtype)
    return fill(a, fill_value)


class MakeVector(COp):
    """Concatenate a number of scalars together into a vector.

    This is a simple version of stack() that introduces far less cruft
    into the graph. Should work with 0 inputs. The constant_folding
    optimization will remove it.

    """

    __props__ = ("dtype",)

    def __init__(self, dtype="int64"):
        self.dtype = np.dtype(dtype).name

    def make_node(self, *inputs):
        inputs = [as_tensor_variable(x) for x in inputs]

        if not all(a.ndim == 0 for a in inputs):
            raise ValueError("All inputs to MakeVector must be scalars")

        if not all(a.type.dtype == inputs[0].type.dtype for a in inputs) or (
            len(inputs) > 0 and inputs[0].dtype != self.dtype
        ):
            dtype = ps.upcast(self.dtype, *[i.dtype for i in inputs])
            inputs = [cast(i, dtype=dtype) for i in inputs]

            if not all(self.dtype == i.dtype for i in inputs):
                raise TypeError(
                    f"Expected inputs to be upcastable to {self.dtype}; "
                    f"got {[i.dtype for i in inputs]}"
                )

        if inputs:
            dtype = inputs[0].type.dtype
        else:
            dtype = self.dtype

        otype = TensorType(dtype, shape=(len(inputs),))
        return Apply(self, inputs, [otype()])

    def perform(self, node, inputs, out_):
        (out,) = out_
        # not calling pytensor._asarray as optimization
        if (out[0] is None) or (out[0].size != len(inputs)):
            out[0] = _asarray(inputs, dtype=node.outputs[0].dtype)
        else:
            # assume that out has correct dtype. there is no cheap way to check
            out[0][...] = inputs

    def c_code_cache_version(self):
        return (2,)

    def c_code(self, node, name, inp, out_, props):
        (out,) = out_
        # Shouldn't use PyArray_TYPE(inp[0]) for the dtype
        # when len(inp) == 0 (we need to support this case.
        # So there will be (1 * nb_dtype) + ((nb len(inp) - 1 ))
        # different c code with the following algo
        out_shape = len(inp)
        out_num = np.dtype(node.outputs[0].dtype).num
        # don't use dtype_%(out)s as when check_input=False, it isn't defined.
        out_dtype = node.outputs[0].type.dtype_specs()[1]
        if len(inp) > 0:
            assert self.dtype == node.inputs[0].dtype
            out_num = f"PyArray_TYPE({inp[0]})"

        ret = """
        npy_intp dims[1];
        dims[0] = {out_shape};
        if(!{out} || PyArray_DIMS({out})[0] != {out_shape}){{
            Py_XDECREF({out});
            {out} = (PyArrayObject*)PyArray_EMPTY(1, dims, {out_num}, 0);
        }}
        """.format(**locals())
        for idx, i in enumerate(inp):
            ret += """
            *(({out_dtype} *)PyArray_GETPTR1({out}, {idx})) = *(({out_dtype} *) PyArray_DATA({i}));
            """.format(**locals())
        return ret

    def infer_shape(self, fgraph, node, ishapes):
        return [(len(ishapes),)]

    def grad(self, inputs, output_gradients):
        # If the output is of an integer dtype, no gradient shall pass
        if self.dtype in discrete_dtypes:
            return [ipt.zeros_like().astype(config.floatX) for ipt in inputs]

        grads = []
        for i, inp in enumerate(inputs):
            grads.append(output_gradients[0][i])
        return grads

    def R_op(self, inputs, eval_points):
        if None in eval_points:
            return [None]
        return self.make_node(*eval_points).outputs


make_vector = MakeVector()


class MakeVectorPrinter(Printer):
    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print make_vector.")
        elif isinstance(r.owner.op, MakeVector):
            with set_precedence(pstate):
                s = [pstate.pprinter.process(inp) for inp in r.owner.inputs]
            return f"[{', '.join(s)}]"
        else:
            raise TypeError("Can only print make_vector.")


pprint.assign(MakeVector, MakeVectorPrinter())


@_get_vector_length.register(MakeVector)
def _get_vector_length_MakeVector(op, var):
    return len(var.owner.inputs)


def transfer(var, target):
    """
    Return a version of `var` transferred to `target`.

    `cpu` mean a TensorType (on the CPU).  Other types may define
    additional targets.

    Parameters
    ----------
    var : variable
        A pytensor variable
    target : str
        The target of the transfer
    """
    if target == "cpu":
        return as_tensor_variable(var)
    else:
        for trans in transfer._others:
            res = trans(var, target)
            if res is not None:
                return res
    raise ValueError(f"Can't transfer to target {target}")


transfer._others = []


def register_transfer(fn):
    """
    Register a transfer function for alternative targets.

    Parameters
    ----------
    fn : callable
    """
    transfer._others.append(fn)


"""Create a duplicate of `a` (with duplicated storage)"""
tensor_copy = Elemwise(ps.identity)
pprint.assign(tensor_copy, printing.IgnorePrinter())


class Default(Op):
    """
    Takes an input x and a default value.

    If the input is not None, a reference to it is returned.
    If the input is None, a copy of the default value is returned instead.
    The input and the default must have exactly the same type.

    """

    view_map = {0: [0]}
    __props__ = ()

    def make_node(self, x, default):
        x, default = as_tensor_variable(x), as_tensor_variable(default)
        if not x.type.in_same_class(default.type):
            raise TypeError("Both arguments must have compatible types")
        return Apply(self, [x, default], [default.type()])

    def perform(self, node, inp, out_):
        x, default = inp
        (out,) = out_
        if x is None:
            # why copy?  PyTensor can't yet understand out[0] being a view of
            # either x or y, so we can be a view of x, but only a copy of y.
            out[0] = default.copy()
        else:
            out[0] = x


default = Default()


def extract_constant(x, elemwise=True, only_process_constants=False):
    """
    This function is basically a call to tensor.get_underlying_scalar_constant_value.

    The main difference is the behaviour in case of failure. While
    get_underlying_scalar_constant_value raises an TypeError, this function returns x,
    as a tensor if possible. If x is a ScalarVariable from a
    scalar_from_tensor, we remove the conversion. If x is just a
    ScalarVariable, we convert it to a tensor with tensor_from_scalar.

    """
    try:
        x = get_underlying_scalar_constant_value(x, elemwise, only_process_constants)
    except NotScalarConstantError:
        pass
    if isinstance(x, ps.ScalarVariable | ps.sharedvar.ScalarSharedVariable):
        if x.owner and isinstance(x.owner.op, ScalarFromTensor):
            x = x.owner.inputs[0]
        else:
            x = tensor_from_scalar(x)
    return x


def transpose(x, axes=None):
    """
    Reorder the dimensions of x. (Default: reverse them)

    This is a macro around dimshuffle that matches the numpy.transpose function.

    """
    _x = as_tensor_variable(x)

    if axes is None:
        axes = list(range((_x.type.ndim - 1), -1, -1))
    ret = DimShuffle(tuple(s == 1 for s in _x.type.shape), axes)(_x)

    if _x.name and axes == list(range((_x.type.ndim - 1), -1, -1)):
        ret.name = _x.name + ".T"

    return ret


def matrix_transpose(x: "TensorLike") -> TensorVariable:
    """
    Transposes each 2-dimensional matrix tensor along the last two dimensions of a higher-dimensional tensor.

    Parameters
    ----------
    x : array_like
        Input tensor with shape (..., M, N), where `M` and `N` represent the dimensions
        of the matrices. Each matrix is of shape (M, N).

    Returns
    -------
    out : tensor
        Transposed tensor with the shape (..., N, M), where each 2-dimensional matrix
        in the input tensor has been transposed along the last two dimensions.

    Examples
    --------
    >>> import pytensor as pt
    >>> import numpy as np
    >>> x = np.arange(24).reshape((2, 3, 4))
    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]

     [[12 13 14 15]
      [16 17 18 19]
      [20 21 22 23]]]


    >>> pt.matrix_transpose(x).eval()
    [[[ 0  4  8]
      [ 1  5  9]
      [ 2  6 10]
      [ 3  7 11]]

     [[12 16 20]
      [13 17 21]
      [14 18 22]
      [15 19 23]]]


    Notes
    -----
    This function transposes each 2-dimensional matrix within the input tensor along
    the last two dimensions. If the input tensor has more than two dimensions, it
    transposes each 2-dimensional matrix independently while preserving other dimensions.
    """
    x = as_tensor_variable(x)
    if x.ndim < 2:
        raise ValueError(
            f"Input array must be at least 2-dimensional, but it is {x.ndim}"
        )
    return swapaxes(x, -1, -2)


def split(x, splits_size, n_splits, axis=0):
    the_split = Split(n_splits)
    return the_split(x, axis, splits_size)


class Split(COp):
    """Partition a `TensorVariable` along some axis.

    Examples
    --------
    >>> x = vector()
    >>> splits = lvector()
    You have to declare right away how many split_points there will be.
    >>> ra, rb, rc = split(x, splits, n_splits = 3, axis = 0)
    >>> f = function([x, splits], [ra, rb, rc])
    >>> a, b, c = f([0,1,2,3,4,5], [3, 2, 1])
    a == [0,1,2]
    b == [3, 4]
    c == [5]

    TODO: Don't make a copy in C impl
    """

    len_splits = None
    """A Split instance will have this many outputs, and require that
    the splits argument to `perform` have exactly this many elements.
    """
    __props__ = ("len_splits",)

    def __init__(self, len_splits):
        self.len_splits = int(len_splits)
        self.view_map = {i: [0] for i in range(self.len_splits)}

    def __str__(self):
        return f"{self.__class__.__name__ }{{{self.len_splits}}}"

    def make_node(self, x, axis, splits):
        """WRITEME"""
        x = as_tensor_variable(x)
        axis = as_tensor_variable(axis)
        splits = as_tensor_variable(splits)

        if splits.type.ndim == 1 and splits.type.dtype not in integer_dtypes:
            raise TypeError("`splits` parameter must be tensors of integer type")

        if axis.type.dtype not in integer_dtypes or axis.ndim != 0:
            raise TypeError("`axis` parameter must be an integer scalar")

        inputs = [x, axis, splits]
        out_type = TensorType(dtype=x.dtype, shape=(None,) * x.type.ndim)
        outputs = [out_type() for i in range(self.len_splits)]

        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        x, axis, splits = inputs

        if len(splits) != self.len_splits:
            raise ValueError("Length of splits is not equal to n_splits")
        if np.sum(splits) != x.shape[axis]:
            raise ValueError(
                f"Split sizes sum to {np.sum(splits)}; expected {x.shape[axis]}"
            )
        if np.any(splits < 0):
            raise ValueError("Split sizes cannot be negative")

        split_outs = np.split(x, np.cumsum(splits[:-1]), axis=axis)
        for i, out in enumerate(split_outs):
            outputs[i][0] = out

    def infer_shape(self, fgraph, node, in_shapes):
        axis = node.inputs[1]
        splits = node.inputs[2]
        shp_x, shp_axis, shp_splits = in_shapes
        out_shapes = []
        for i in range(self.len_splits):
            temp = as_tensor_variable(shp_x)
            temp = pytensor.tensor.subtensor.set_subtensor(temp[axis], splits[i])
            temp = [temp[i] for i in range(len(shp_x))]
            out_shapes.append(temp)
        return out_shapes

    def grad(self, inputs, g_outputs):
        """Join the gradients along the axis that was used to split x."""
        x, axis, n = inputs
        outputs = self(*inputs, return_list=True)
        # If all the output gradients are disconnected, then so are the inputs
        if builtins.all(isinstance(g.type, DisconnectedType) for g in g_outputs):
            return [
                DisconnectedType()(),
                grad_undefined(self, 1, axis),
                grad_undefined(self, 2, n),
            ]
        # Else, we have to make them zeros before joining them
        new_g_outputs = []
        for o, g in zip(outputs, g_outputs):
            if isinstance(g.type, DisconnectedType):
                new_g_outputs.append(o.zeros_like())
            else:
                new_g_outputs.append(g)

        return [
            join(axis, *new_g_outputs),
            grad_undefined(self, 1, axis),
            grad_undefined(self, 2, n),
        ]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None for i in self.len_splits]
        return self.make_node(eval_points[0], *inputs[1:]).outputs

    def c_code_cache_version(self):
        return (2,)

    def c_support_code(self, **kwargs):
        return """
        /* Return 1 if output has the correct shape. */
        int split_output_shape_is_correct (
            PyArrayObject* output, PyArrayObject* array_to_split, int axis_to_split, npy_intp split_size
        ) {
            return
                PyArray_NDIM(output) == PyArray_NDIM(array_to_split)
                && memcmp(
                    PyArray_DIMS(output),
                    PyArray_DIMS(array_to_split),
                    axis_to_split * sizeof(npy_intp)
                ) == 0
                && memcmp(
                    PyArray_DIMS(output) + axis_to_split + 1,
                    PyArray_DIMS(array_to_split) + axis_to_split + 1,
                    (PyArray_NDIM(array_to_split) - axis_to_split - 1) * sizeof(npy_intp)
                ) == 0
                && split_size == PyArray_DIM(output, axis_to_split);
        }
        """

    def c_code(self, node, name, inputs, outputs, sub):
        if self.len_splits == 0:
            # There are no outputs, then nothing to do.
            return ""

        # outputs_pointers lists the addresses of the pointers to the outputs.
        outputs_pointers = "&" + (", &".join(outputs))
        x, axis, splits = inputs
        fail = sub["fail"]
        x_typenum = np.dtype(node.inputs[0].dtype).num
        x_itemsize = np.dtype(node.inputs[0].dtype).itemsize
        axis_dtype = node.inputs[1].type.dtype_specs()[1]
        splits_dtype = node.inputs[2].type.dtype_specs()[1]
        expected_splits_count = self.len_splits

        return """
        int ndim = PyArray_NDIM({x});
        int axis = (int)(*({axis_dtype}*)PyArray_GETPTR1({axis}, 0));
        int splits_count = PyArray_DIM({splits}, 0);
        npy_intp len_along_axis, sum_of_splits = 0, current_split_length = 0, current_split_start = 0;
        npy_intp* split_dims = NULL;
        PyObject* split_view = NULL;
        npy_intp data_offset;
        int i;
        PyArrayObject** outputs[] = {{{outputs_pointers}}};

        /* Check inputs. */

        if (splits_count != {expected_splits_count}) {{
            PyErr_Format(PyExc_ValueError,
                "Split: splits count (%d) != expected count (%d).", splits_count, {expected_splits_count});
            {fail}
        }}

        if (axis < 0) {{
            axis += ndim;
        }}
        if (axis < 0 || axis >= ndim) {{
            PyErr_Format(PyExc_IndexError, "Split: invalid axis %d for a %d-D array.", axis, ndim);
            {fail}
        }}
        len_along_axis = PyArray_DIM({x}, axis);

        for (i = 0; i < splits_count; ++i) {{
            current_split_length = (npy_intp)(*({splits_dtype}*)PyArray_GETPTR1({splits}, i));
            if (current_split_length < 0) {{
                PyErr_Format(PyExc_ValueError,
                    "Split: you try to take a negative number (%ld) of elements.", current_split_length);
                {fail}
            }}
            sum_of_splits += current_split_length;
        }}
        if (sum_of_splits != len_along_axis) {{
            PyErr_Format(PyExc_ValueError, "Split: the splits sums to %ld, expected %ld.", sum_of_splits, len_along_axis);
            {fail}
        }}

        /* Check outputs. */

        split_dims = (npy_intp*) malloc(ndim * sizeof(npy_intp));
        if (split_dims == NULL) {{
            PyErr_NoMemory();
            {fail}
        }}

        memcpy(split_dims, PyArray_DIMS({x}), ndim * sizeof(npy_intp));

        for (i = 0; i < splits_count; ++i) {{
            PyArrayObject** output = outputs[i];
            current_split_length = (npy_intp) (* ({splits_dtype}*) PyArray_GETPTR1({splits}, i));
            if (*output == NULL || !split_output_shape_is_correct(*output, {x}, axis, current_split_length)) {{
                Py_XDECREF(*output);
                split_dims[axis] = current_split_length;
                *output = (PyArrayObject*)PyArray_EMPTY(ndim, split_dims, {x_typenum}, PyArray_IS_F_CONTIGUOUS({x}));
                if (outputs == NULL) {{
                    PyErr_SetString(PyExc_RuntimeError, "Split: unable to allocate an output.");
                    free(split_dims);
                    {fail}
                }}
            }}
        }}

        /* Compute split. */

        for (i = 0; i < splits_count; ++i) {{
            current_split_length = (npy_intp) (* ({splits_dtype}*) PyArray_GETPTR1({splits}, i));
            data_offset = PyArray_STRIDE({x}, axis) * current_split_start;
            split_dims[axis] = current_split_length;
            split_view = PyArray_New(&PyArray_Type,
                                    ndim, split_dims,
                                    {x_typenum},
                                    PyArray_STRIDES({x}),
                                    PyArray_BYTES({x}) + data_offset,
                                    {x_itemsize},
                                    PyArray_FLAGS({x}),
                                    NULL);
            if (split_view == NULL) {{
                PyErr_SetString(PyExc_RuntimeError, "Split: unable to create a view for a split.");
                free(split_dims);
                {fail}
            }}
            if (PyArray_CopyInto(*outputs[i], (PyArrayObject*)split_view) != 0) {{
                PyErr_SetString(PyExc_RuntimeError, "Split: unable to copy a split view into the output.");
                Py_XDECREF(split_view);
                free(split_dims);
                {fail}
            }}
            Py_XDECREF(split_view);
            current_split_start += current_split_length;
        }}

        free(split_dims);
        """.format(**locals())


class Join(COp):
    r"""
    Concatenate multiple `TensorVariable`\s along some axis.

    The axis must be given as first argument. All tensors must have the same
    shape along all dimensions other than this axis.
    Of course, TensorVariable instances do not have a shape, so this error
    cannot be caught until runtime.  See `perform()`.

    See Also
    --------
    stack : For joins involving scalar values

    Examples
    --------
    >>> x, y, z = tensor.matrix(), tensor.matrix(), tensor.matrix()
    >>> u = tensor.vector()

    >>> r = join(0, x, y, z)
    >>> c = join(1, x, y, z)
    >>> join(2, x, y, z)    # WRONG: the axis has to be an index into the shape
    >>> join(0, x, u)       # WRONG: joined tensors must have the same rank

    """

    check_input = False
    __props__ = ("view",)

    def __init__(self, view=-1):
        self.view = view
        if view != -1:
            # since the first input is always the axis, the tensors
            # start from index 1.
            self.view_map = {0: [1 + view]}

    def __str__(self):
        if self.view == -1:
            return self.__class__.__name__
        else:
            return "{}{{{}}}".format(
                self.__class__.__name__,
                ", ".join(f"{p}={getattr(self, p)!r}" for p in self.__props__),
            )

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "view"):
            self.view = -1

    def make_node(self, axis, *tensors):
        """
        Parameters
        ----------
        axis
            The axis upon which to join `tensors`.
        tensors
            A variable number of tensors to join along the specified axis.
            These tensors must have the same shape along all dimensions other
            than `axis`.

        """
        if not tensors:
            raise ValueError("Cannot join an empty list of tensors")

        tensors = [as_tensor_variable(x) for x in tensors]
        out_dtype = ps.upcast(*[x.type.dtype for x in tensors])

        if not builtins.all(targs.type.ndim for targs in tensors):
            raise TypeError(
                "Join cannot handle arguments of dimension 0."
                " Use `stack` to join scalar values."
            )

        if len(tensors) == 1:
            out_shape = tensors[0].type.shape
        else:
            # When the axis is fixed, a dimension should be
            # broadcastable if at least one of the inputs is
            # broadcastable on that dimension (see justification below),
            # except for the axis dimension.
            # Initialize bcastable all false, and then fill in some trues with
            # the loops.

            if not isinstance(axis, int):
                try:
                    axis = int(get_underlying_scalar_constant_value(axis))
                except NotScalarConstantError:
                    pass

            ndim = tensors[0].type.ndim
            if isinstance(axis, int):
                # Basically, broadcastable -> length 1, but the
                # converse does not hold. So we permit e.g. T/F/T
                # joins, and if they fail at runtime they fail, but if
                # they don't then it means that the argument where
                # that broadcastable flag was False had length 1 along
                # this dimension, and therefore this dimension should
                # be broadcastable for the output.

                if axis < -ndim:
                    raise IndexError(
                        f"Axis value {axis} is out of range for the given input dimensions"
                    )
                if axis < 0:
                    axis += ndim
                if axis > ndim - 1:
                    raise ValueError(
                        f"Axis value {axis} is out of range for the given input dimensions"
                    )
                # NOTE: Constant negative axis can no longer be negative at this point.

                in_shapes = [x.type.shape for x in tensors]
                in_ndims = [len(s) for s in in_shapes]
                if set(in_ndims) != {ndim}:
                    raise TypeError(
                        "Only tensors with the same number of dimensions can be joined."
                        f" Input ndims were: {in_ndims}."
                    )

                # Determine output shapes from a matrix of input shapes
                in_shapes = np.array(in_shapes)
                out_shape = [None] * ndim
                for d in range(ndim):
                    ins = in_shapes[:, d]
                    if d == axis:
                        # Any unknown size along the axis means we can't sum
                        if None in ins:
                            out_shape[d] = None
                        else:
                            out_shape[d] = sum(ins)
                    else:
                        inset = set(in_shapes[:, d])
                        # Other dims must match exactly,
                        # or if a mix of None and ? the output will be ?
                        # otherwise the input shapes are incompatible.
                        if len(inset) == 1:
                            (out_shape[d],) = inset
                        elif len(inset - {None}) == 1:
                            (out_shape[d],) = inset - {None}
                        else:
                            raise ValueError(
                                f"all input array dimensions other than the specified `axis` ({axis})"
                                " must match exactly, or be unknown (None),"
                                f" but along dimension {d}, the inputs shapes are incompatible: {ins}"
                            )
            else:
                # When the axis may vary, no dimension can be guaranteed to be
                # broadcastable.
                out_shape = [None] * tensors[0].type.ndim

            if not builtins.all(x.ndim == len(out_shape) for x in tensors):
                raise TypeError(
                    "Only tensors with the same number of dimensions can be joined"
                )

        inputs = [as_tensor_variable(axis), *tensors]

        if inputs[0].type.dtype not in int_dtypes:
            raise TypeError(f"Axis value {inputs[0]} must be an integer type")

        return Apply(self, inputs, [tensor(dtype=out_dtype, shape=out_shape)])

    def perform(self, node, axis_and_tensors, out_):
        (out,) = out_
        view = self.view
        axis, tens = axis_and_tensors[0], axis_and_tensors[1:]
        # we check these tensors for being empty.
        if (view != -1) and all(
            tensor.shape[axis] == 0 for tensor in tens[0:view] + tens[view + 1 :]
        ):
            out[0] = tens[view]

        else:
            ndim = tens[0].ndim
            if axis < -ndim:
                raise IndexError(
                    f"Join axis {int(axis)} out of bounds [0, {int(ndim)})"
                )

            out[0] = _asarray(
                np.concatenate(tens, axis=axis), dtype=node.outputs[0].type.dtype
            )

    def c_code_cache_version(self):
        return (5,)

    def c_code(self, node, name, inputs, outputs, sub):
        axis, tens = inputs[0], inputs[1:]
        view = self.view
        non_empty_tensor = tens[view]
        input_1 = tens[0]
        l = len(tens)
        (out,) = outputs
        fail = sub["fail"]
        adtype = node.inputs[0].type.dtype_specs()[1]
        copy_to_list = []

        for i, inp in enumerate(tens):
            copy_to_list.append(
                f"""Py_INCREF({inp});
                   PyList_SetItem(list, {i}, (PyObject*){inp});"""
            )

        copy_inputs_to_list = "\n".join(copy_to_list)
        n = len(tens)

        code = """
        int axis = (({adtype} *)PyArray_DATA({axis}))[0];
        PyObject* list = PyList_New({l});
        {copy_inputs_to_list}
        int tensors_lens_sum;
        if({view} != -1) {{
            tensors_lens_sum = 0;

            for(int i=0; i < {n}; i++){{
                tensors_lens_sum += PyArray_DIM((PyArrayObject *)(PyList_GetItem(list, i)), axis);
            }}
            tensors_lens_sum -= PyArray_DIM({non_empty_tensor}, axis);
        }}
        if({view} != -1 && tensors_lens_sum == 0) {{
            Py_XDECREF({out});
            Py_INCREF({non_empty_tensor});
            {out} = {non_empty_tensor};
        }}else{{
            //PyObject* PyArray_Concatenate(PyObject* obj, int axis)
            int ndim = PyArray_NDIM({input_1});
            if( axis < -ndim ){{
                PyErr_Format(PyExc_IndexError,
                             "Join axis %d out of bounds [0, %d)", axis, ndim);
                {fail}
            }}
            Py_XDECREF({out});
            {out} = (PyArrayObject *)PyArray_Concatenate(list, axis);
            Py_DECREF(list);
            if(!{out}){{
                {fail}
            }}
        }}
        """.format(**locals())
        return code

    def R_op(self, inputs, eval_points):
        if None in eval_points[1:]:
            return [None]
        return self.make_node(inputs[0], *eval_points[1:]).outputs

    def grad(self, axis_and_tensors, grads):
        """The gradient wrt a join op is a `Split`, used to partition
        the gradient along the `axis` which was used for joining.
        """
        (gz,) = grads
        axis, tens = axis_and_tensors[0], axis_and_tensors[1:]

        rval = [grad_undefined(self, 0, axis)]

        dtypes = [as_tensor_variable(x).type.dtype for x in tens]
        out_dtype = ps.upcast(*dtypes)

        if "float" in out_dtype or "complex" in out_dtype:
            # assume that this is differentiable
            split = Split(len(tens))
            split_gz = split(gz, axis, stack([shape(x)[axis] for x in tens]))
            # If there is only one split, it might not be in a list.
            if not isinstance(split_gz, list):
                split_gz = [split_gz]
            # Split.make_node isn't always able to infer the right
            # broadcast. As the grad need to keep the information,
            # read it if needed.
            split_gz = [
                g
                if g.type.shape == t.type.shape == 1
                else specify_broadcastable(
                    g, *(ax for (ax, s) in enumerate(t.type.shape) if s == 1)
                )
                for t, g in zip(tens, split_gz)
            ]
            rval = rval + split_gz
        else:
            # the output has integer type, so the gradient through it
            # is 0
            rval = rval + [t.zeros_like(dtype=config.floatX) for t in tens]

        return rval

    def infer_shape(self, fgraph, node, ishapes):
        from pytensor.tensor.math import eq, ge

        # ishapes[0] contains the size of the axis on which we join
        # Join op should get at least one input to join
        assert len(ishapes) > 1
        n_dim = len(ishapes[1])
        for shp in ishapes[1:]:
            assert shp is not None
            assert len(shp) == n_dim

        # The joining dimension could be negative, but we need it to be
        # in [0, n_dim) in the loop below.
        # An axis < -n_dim or >= ndim would be invalid, but this is
        # not checked here. A `CheckAndRaise` `Op` would be a way of
        # addressing that, but it may disrupt optimizations.
        join_dim = switch(ge(node.inputs[0], 0), node.inputs[0], node.inputs[0] + n_dim)
        out_shapes = []
        for dim in range(n_dim):
            # we have to deal with 2 possible cases in here :
            #   a) we are dealing with the dimension for which we join
            #     (called t_side from true side of the if, where the if
            #     compares current dimension with the joining dimension)
            #   b) a non joining dimension ( in which maybe a symbolic
            #      assertion can be used to make sure all tensors have
            #      the same number of elements on this non-joined dimension
            #      this is f_side
            # initialize
            t_side = ishapes[1][dim]
            f_side = ishapes[1][dim]
            # loop over tensors and sum for the joining dimension
            for shp in ishapes[2:]:
                t_side = t_side + shp[dim]
            # return the dimensions found
            out_shapes.append(switch(eq(dim, join_dim), t_side, f_side))

        return [tuple(out_shapes)]


join_ = Join()
pprint.assign(Join, printing.FunctionPrinter(["join"]))


@_get_vector_length.register(Join)
def _get_vector_length_Join(op, var):
    axis, *arrays = var.owner.inputs
    try:
        axis = get_underlying_scalar_constant_value(axis)
        assert axis == 0 and builtins.all(a.ndim == 1 for a in arrays)
        return builtins.sum(get_vector_length(a) for a in arrays)
    except NotScalarConstantError:
        raise ValueError(f"Length of {var} cannot be determined")


def join(axis, *tensors_list):
    r"""
    Convenience function to concatenate `TensorType`\s along the given axis.

    This function will not add the op in the graph when it is not useful.
    For example, in the case that the list of tensors to be concatenated
    is one, it will just return the tensor.

    Parameters
    ----------
    axis : int (symbolic or literal)
        On which dimension should the tensors be joined?  The `axis`
        must be a valid index into the shape of the tensors to be
        concatenated.
        The `axis` parameter may either be an integer or an object that
        can be converted to a scalar using `as_scalar`(`axis`). In the
        former case, the axis is fixed at construction, while in the
        latter it may vary over time depending on the value of the
        `axis` variable.
    tensors_list : list of TensorVariable (or list-like)
        A list of tensors to be concatenated along the given axis.
        The shapes of the tensors to be concatenated must be all
        identical, except in the dimension (`axis`) on which they are to
        be joined.
    """
    if len(tensors_list) == 1:
        return tensors_list[0]
    else:
        return join_(axis, *tensors_list)


@_vectorize_node.register(Join)
def vectorize_join(op: Join, node, batch_axis, *batch_inputs):
    original_axis, *old_inputs = node.inputs
    # We can vectorize join as a shifted axis on the batch inputs if:
    # 1. The batch axis is a constant and has not changed
    # 2. All inputs are batched with the same broadcastable pattern
    if (
        original_axis.type.ndim == 0
        and isinstance(original_axis, Constant)
        and equal_computations([original_axis], [batch_axis])
    ):
        batch_ndims = {
            batch_input.type.ndim - old_input.type.ndim
            for batch_input, old_input in zip(batch_inputs, old_inputs)
        }
        if len(batch_ndims) == 1:
            [batch_ndim] = batch_ndims
            batch_bcast = batch_inputs[0].type.broadcastable[:batch_ndim]
            if all(
                batch_input.type.broadcastable[:batch_ndim] == batch_bcast
                for batch_input in batch_inputs[1:]
            ):
                original_ndim = node.outputs[0].type.ndim
                original_axis = normalize_axis_index(original_axis.data, original_ndim)
                batch_axis = original_axis + batch_ndim
                return op.make_node(batch_axis, *batch_inputs)

    return vectorize_node_fallback(op, node, batch_axis, *batch_inputs)


def roll(x, shift, axis=None):
    """
    Convenience function to roll TensorTypes along the given axis.

    Syntax copies numpy.roll function.

    Parameters
    ----------
    x : tensor_like
        Input tensor.
    shift : int (symbolic or literal)
        The number of places by which elements are shifted.
    axis : int (symbolic or literal), optional
        The axis along which elements are shifted. By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    tensor
        Output tensor, with the same shape as ``x``.

    """
    _x = as_tensor_variable(x)
    if axis is None:
        if _x.ndim > 1:
            y = _x.flatten()
            return roll(y, shift, axis=0).reshape(_x.shape)
        else:
            axis = 0

    if axis < 0:
        axis += _x.ndim

    # Shift may be larger than the size of the axis. If so, since the
    # roll operation is cyclic, we can take the shift modulo the size
    # of the axis
    shift = shift % _x.shape[axis]

    # A slice of all elements in a dimension ':'
    allslice = slice(None)
    # List of slices describing the front half [:, :, shift:, :]
    front_slice = slice(-shift, None)
    front_list = [allslice] * axis + [front_slice] + [allslice] * (_x.ndim - axis - 1)
    # List of slices describing the back half [:, :, :shift, :]
    end_slice = slice(0, -shift)
    end_list = [allslice] * axis + [end_slice] + [allslice] * (_x.ndim - axis - 1)
    return join(
        axis, _x.__getitem__(tuple(front_list)), _x.__getitem__(tuple(end_list))
    )


def stack(tensors: Sequence["TensorLike"], axis: int = 0):
    """Stack tensors in sequence on given axis (default is 0).

    Take a sequence of tensors or tensor-like constant and stack them on
    given axis to make a single tensor. The size in dimension `axis` of the
    result will be equal to the number of tensors passed.

    Parameters
    ----------
    tensors : Sequence[TensorLike]
        A list of tensors or tensor-like constants to be stacked.
    axis : int
        The index of the new axis. Default value is 0.

    Examples
    --------
    >>> a = pytensor.tensor.type.scalar()
    >>> b = pytensor.tensor.type.scalar()
    >>> c = pytensor.tensor.type.scalar()
    >>> x = pytensor.tensor.stack([a, b, c])
    >>> x.ndim # x is a vector of length 3.
    1
    >>> a = pytensor.tensor.type.tensor4()
    >>> b = pytensor.tensor.type.tensor4()
    >>> c = pytensor.tensor.type.tensor4()
    >>> x = pytensor.tensor.stack([a, b, c])
    >>> x.ndim # x is a 5d tensor.
    5
    >>> rval = x.eval(dict((t, np.zeros((2, 2, 2, 2))) for t in [a, b, c]))
    >>> rval.shape # 3 tensors are stacked on axis 0
    (3, 2, 2, 2, 2)
    >>> x = pytensor.tensor.stack([a, b, c], axis=3)
    >>> x.ndim
    5
    >>> rval = x.eval(dict((t, np.zeros((2, 2, 2, 2))) for t in [a, b, c]))
    >>> rval.shape # 3 tensors are stacked on axis 3
    (2, 2, 2, 3, 2)
    >>> x = pytensor.tensor.stack([a, b, c], axis=-2)
    >>> x.ndim
    5
    >>> rval = x.eval(dict((t, np.zeros((2, 2, 2, 2))) for t in [a, b, c]))
    >>> rval.shape # 3 tensors are stacked on axis -2
    (2, 2, 2, 3, 2)
    """
    if not isinstance(tensors, Sequence):
        raise TypeError("First argument should be a Sequence.")
    elif len(tensors) == 0:
        raise ValueError("No tensor arguments provided.")

    # If all tensors are scalars, call make_vector.
    # It makes the graph simpler, by not adding DimShuffles and SpecifyShapes

    # This should be an optimization!
    # Doing it here make the graph less canonicalized
    # (more type need to be understood by all optimization)
    # And DebugMode can't detect error in this code as it is not in an
    # optimization.
    # See ticket #660
    if all(
        # In case there are explicit scalars in tensors
        isinstance(t, Number)
        or (isinstance(t, np.ndarray) and t.ndim == 0)
        or (isinstance(t, Variable) and isinstance(t.type, TensorType) and t.ndim == 0)
        for t in tensors
    ):
        # In case there is direct scalar
        tensors = list(map(as_tensor_variable, tensors))
        dtype = ps.upcast(*[i.dtype for i in tensors])
        return MakeVector(dtype)(*tensors)
    return join(axis, *[shape_padaxis(t, axis) for t in tensors])


def concatenate(tensor_list, axis=0):
    """Alias for `join`(axis, *tensor_list).

    This function is similar to `join`, but uses the signature of
    numpy's concatenate function.

    Raises
    ------
    TypeError
        The tensor_list must be a tuple or list.

    """
    # Check someone did not make the common mistake to do something like:
    #   c = concatenate(x, y)
    # instead of
    #   c = concatenate((x, y))
    if not isinstance(tensor_list, tuple | list):
        raise TypeError(
            "The 'tensors' argument must be either a tuple "
            "or a list, make sure you did not forget () or [] around "
            "arguments of concatenate.",
            tensor_list,
        )
    return join(axis, *tensor_list)


def horizontal_stack(*args):
    r"""Stack arrays in sequence horizontally (column wise)."""
    # Note: 'horizontal_stack' and 'vertical_stack' do not behave exactly like
    # Numpy's hstack and vstack functions. This is intended, because Numpy's
    # functions have potentially confusing/incoherent behavior (try them on 1D
    # arrays). If this is fixed in a future version of Numpy, it may be worth
    # trying to get closer to Numpy's way of doing things. In the meantime,
    # better keep different names to emphasize the implementation divergences.

    if len(args) < 2:
        raise ValueError("Too few arguments")

    _args = []
    for arg in args:
        _arg = as_tensor_variable(arg)
        if _arg.type.ndim != 2:
            raise ValueError("All arguments must have two dimensions")
        _args.append(_arg)

    return concatenate(_args, axis=1)


def vertical_stack(*args):
    r"""Stack arrays in sequence vertically (row wise)."""

    if len(args) < 2:
        raise ValueError("Too few arguments")

    _args = []
    for arg in args:
        _arg = as_tensor_variable(arg)
        if _arg.type.ndim != 2:
            raise ValueError("All arguments must have two dimensions")
        _args.append(_arg)

    return concatenate(_args, axis=0)


def is_flat(var, ndim=1):
    """
    Verifies the dimensionality of the var is equal to
    ndim. This method is usually called after flatten method on a
    variable, where the first ndim-1 dimension size(s) of the variable
    is kept intact, and the last dimension size of the variable is made
    equal to the multiplication of its remaining dimension size(s), such that
    the variable would end up with as many dimension as ndim.

    Parameters
    ----------
    var : pytensor.tensor.var.TensorVariable
        the pytensor var on which the dimensionality is checked.

    ndim : int
        the expected dimensionality of var.

    Returns
    -------
    bool
        the comparison result of var's dim
        and the expected outdim.
    """
    return var.ndim == ndim


def flatten(x, ndim=1):
    """Return a copy of the array collapsed into one dimension.

    Reshapes the variable `x` by keeping the first outdim-1 dimension size(s)
    of `x` the same, and making the last dimension size of `x` equal to the
    multiplication of its remaining dimension size(s).

    Parameters
    ----------
    x : pytensor.tensor.var.TensorVariable
        The variable to be reshaped.
    ndim : int
        The number of dimensions of the returned variable
        The default value is ``1``.

    Returns
    -------
    pytensor.tensor.var.TensorVariable
        the flattened variable with dimensionality of outdim
    """
    if ndim is None:
        ndim = 1

    _x = as_tensor_variable(x)

    # Any input variable can be flattened to have ndim of 1,
    # even if it's a scalar. Otherwise, ndim must be positive
    # and smaller than x.ndim.
    if ndim < 1 or (ndim > 1 and ndim > _x.ndim):
        raise ValueError(f"ndim {ndim} out of bound [1, {_x.ndim + 1})")

    if ndim > 1:
        dims = (*_x.shape[: ndim - 1], -1)
    else:
        dims = (-1,)

    x_reshaped = _x.reshape(dims)
    shape_kept_dims = _x.type.shape[: ndim - 1]
    bcast_new_dim = builtins.all(s == 1 for s in _x.type.shape[ndim - 1 :])
    out_shape = (*shape_kept_dims, 1 if bcast_new_dim else None)
    bcasted_indices = tuple(i for i in range(ndim) if out_shape[i] == 1)
    x_reshaped = specify_broadcastable(x_reshaped, *bcasted_indices)
    return x_reshaped


def tile(x, reps, ndim=None):
    """
    Tile input array `x` according to `reps`.

    See the docstring of `numpy.tile` for details.

    'reps' can be constant integer (e.g. 3), constant vector(e.g. [2 3]),
    symbolic scalar (e.g. tensor.iscalar()), symbolic vector (e.g. tensor.ivector())
    or a list of symbolic scalar (e.g. [tensor.iscalar(), tensor.iscalar()]).

    ndim is the number of the dimensions of the output, if it is provided, ndim
    should be equal or larger than x.ndim and len(reps), otherwise, we will use
    max(x.ndim, len(reps)) as ndim. If reps is symbolic vector, the ndim has to
    be provided.

    """
    from pytensor.tensor.math import ge

    _x = as_tensor_variable(x)
    if ndim is not None and ndim < _x.ndim:
        raise ValueError("ndim should be equal or larger than _x.ndim")

    # If reps is a scalar, integer or vector, we convert it to a list.
    if not isinstance(reps, list | tuple):
        reps_astensor = as_tensor_variable(reps)
        ndim_check = reps_astensor.ndim
        if reps_astensor.dtype not in discrete_dtypes:
            raise ValueError("elements of reps must be integer dtype")

        # The scalar/integer case
        if ndim_check == 0:
            reps = [reps]

        # The vector case
        elif ndim_check == 1:
            if ndim is None:
                raise ValueError(
                    "if reps is tensor.vector, you should specify the ndim"
                )
            else:
                offset = ndim - reps.shape[0]

                # assert that reps.shape[0] does not exceed ndim
                offset = assert_op(offset, ge(offset, 0))

                # if reps.ndim is less than _x.ndim, we pad the reps with
                # "1" so that reps will have the same ndim as _x.
                reps_ = [switch(i < offset, 1, reps[i - offset]) for i in range(ndim)]
                reps = reps_

        # For others, raise an error
        else:
            raise ValueError("the dimension of reps should not exceed 1")
    else:
        if ndim is not None and len(reps) > ndim:
            raise ValueError("len(reps) should be equal or less than ndim")
        if not all(
            isinstance(r, int)
            or (isinstance(r, TensorVariable) and r.dtype in discrete_dtypes)
            for r in reps
        ):
            raise ValueError("elements of reps must be scalars of integer dtype")

    # If reps.ndim is less than _x.ndim, we pad the reps with
    # "1" so that reps will have the same ndim as _x
    reps = list(reps)
    if ndim is None:
        ndim = builtins.max(len(reps), _x.ndim)
    if len(reps) < ndim:
        reps = [1] * (ndim - len(reps)) + reps

    _shape = [1] * (ndim - _x.ndim) + [_x.shape[i] for i in range(_x.ndim)]
    alloc_shape = reps + _shape
    y = alloc(_x, *alloc_shape)
    shuffle_ind = np.arange(ndim * 2).reshape(2, ndim)
    shuffle_ind = shuffle_ind.transpose().flatten()
    y = y.dimshuffle(*shuffle_ind)
    new_shapes = [sh * reps[i] for i, sh in enumerate(_shape)]
    y = y.reshape(new_shapes)

    return y


class ARange(Op):
    """Create an array containing evenly spaced values within a given interval.

    Parameters and behaviour are the same as numpy.arange().

    """

    __props__ = ("dtype",)

    def __init__(self, dtype):
        self.dtype = dtype

    def make_node(self, start, stop, step):
        start, stop, step = map(as_tensor_variable, (start, stop, step))
        assert start.ndim == 0
        assert stop.ndim == 0
        assert step.ndim == 0

        inputs = [start, stop, step]
        outputs = [tensor(dtype=self.dtype, shape=(None,))]

        return Apply(self, inputs, outputs)

    @config.change_flags(warn_float64="ignore")
    def infer_shape(self, fgraph, node, i_shapes):
        from pytensor.tensor.math import ceil, maximum

        # Note start, stop and step can be float numbers.
        start, stop, step = node.inputs

        def is_constant_value(var, value):
            try:
                v = get_underlying_scalar_constant_value(var)
                return np.all(v == value)
            except NotScalarConstantError:
                pass
            return False

        def upcast(var):
            if (
                var.dtype in integer_dtypes
                and
                # We do not want to cast uint64 to int64 as this can
                # loose information. If we upcast uint64 with int64,
                # this give float64. This is safer then checking for
                # uint64 in case we support [u]int128 or other in the
                # future.
                ps.upcast(var.dtype, "int64") == "int64"
            ):
                return cast(var, "int64")
            return var

        if is_constant_value(step, 1):
            if is_constant_value(start, 0):
                return [(cast(stop, "int64"),)]
            else:
                stop = upcast(stop)
                start = upcast(start)
                return [(maximum(cast(stop - start, "int64"), 0),)]
        else:
            stop = upcast(stop)
            start = upcast(start)
            return [
                (
                    maximum(
                        cast(ceil(cast((stop - start), "float64") / step), "int64"), 0
                    ),
                )
            ]

    def perform(self, node, inp, out_):
        start, stop, step = inp
        (out,) = out_
        start = start.item()
        stop = stop.item()
        step = step.item()
        out[0] = np.arange(start, stop, step, dtype=self.dtype)

    def connection_pattern(self, node):
        return [[True], [False], [True]]

    def L_op(self, inputs, outputs, grads):
        start, stop, step = inputs
        (gz,) = grads
        # `start` and `step` affect the output values
        # but the outputs are integers so there's
        # no gradient through them.
        # When they are not integers, the gradients are
        # as expressed below.
        # `stop` does not affect the output values,
        # just the output shape, so it is disconnected.

        if self.dtype in discrete_dtypes:
            return [
                start.zeros_like(dtype=config.floatX),
                DisconnectedType()(),
                step.zeros_like(dtype=config.floatX),
            ]
        else:
            num_steps_taken = outputs[0].shape[0]
            return [
                gz.sum(),
                DisconnectedType()(),
                (gz * arange(num_steps_taken, dtype=self.dtype)).sum(),
            ]

    def R_op(self, inputs, eval_points):
        return [None]


_arange = {}


def arange(start, stop=None, step=1, dtype=None):
    # If only one argument is provided, it is in fact the "stop" argument,
    # and start is 0.
    if stop is None:
        start, stop = 0, start

    start, stop, step = map(as_tensor_variable, (start, stop, step))
    # If dtype is not provided, infer it from the other arguments
    if dtype is None:
        dtype = ps.upcast(start.type.dtype, stop.type.dtype, step.type.dtype)
        # don't try to be stingy and byte-optimize, this leads to
        # overflow problems.
        if dtype in int_dtypes:
            dtype = "int64"
        if dtype in uint_dtypes:
            dtype = "uint64"
        if config.cast_policy in ("numpy", "numpy+floatX"):
            # We enforce numpy semantics, except in the special case where
            # `config.cast_policy` is 'numpy+floatX' and we want to use float32
            # rather than float64.
            # As an example, if `start`, `stop` and `step` are all int32,
            # `numpy.arange` returns an int64 array (on 64-bit platforms),
            # while the upcast above returns int32.
            numpy_dtype = np.arange(
                start=np.array(0, dtype=start.dtype),
                stop=np.array(1, dtype=stop.dtype),
                step=np.array(1, dtype=step.dtype),
            ).dtype
            if numpy_dtype != dtype:
                if (
                    config.cast_policy == "numpy+floatX"
                    and config.floatX == "float32"
                    and numpy_dtype == "float64"
                    and
                    # No explicit float64 in the three arguments?
                    builtins.all(
                        dt != "float64" for dt in [s.dtype for s in (start, stop, step)]
                    )
                ):
                    # We use float32 instead.
                    assert dtype != "float64"
                    dtype = "float32"
                else:
                    # We use the same dtype as numpy instead of the result of
                    # the upcast.
                    dtype = str(numpy_dtype)

    if dtype not in _arange:
        _arange[dtype] = ARange(dtype)
    return _arange[dtype](start, stop, step)


class _nd_grid:
    """Create a dense n-dimensional 'meshgrid' with equally spaced points.

    Used to create the instance ``mgrid`` and ``ogrid`` which act similarly
    to their numpy equivalents.

    Parameters
    ----------
    sparse : boolean, optional, default=True
        Specifying False leads to the equivalent of numpy's mgrid functionality.
        Specifying True leads to the equivalent of ogrid.

    Examples
    --------
    >>> a = at.mgrid[0:5, 0:3]
    >>> a[0].eval()
    array([[0, 0, 0],
           [1, 1, 1],
           [2, 2, 2],
           [3, 3, 3],
           [4, 4, 4]], dtype=int8)
    >>> a[1].eval()
    array([[0, 1, 2],
           [0, 1, 2],
           [0, 1, 2],
           [0, 1, 2],
           [0, 1, 2]], dtype=int8)
    >>> b = at.ogrid[0:5, 0:3]
    >>> b[0].eval()
    array([[0],
           [1],
           [2],
           [3],
           [4]], dtype=int8)
    >>> b[1].eval()
    array([[0, 1, 2, 3]], dtype=int8)

    """

    def __init__(self, sparse=False):
        self.sparse = sparse

    def __getitem__(self, *args):
        if isinstance(args[0], slice):
            sl = args[0]
            return arange(sl.start or 0, sl.stop, sl.step or 1)

        ndim = len(args[0])
        for sl in args[0]:
            if isinstance(sl.step, builtins.complex):
                raise NotImplementedError(
                    "Not implemented for slices whose step is complex"
                )
        ranges = [arange(sl.start or 0, sl.stop, sl.step or 1) for sl in args[0]]
        shapes = [
            tuple([1] * j + [r.shape[0]] + [1] * (ndim - 1 - j))
            for j, r in enumerate(ranges)
        ]
        ranges = [r.reshape(shape) for r, shape in zip(ranges, shapes)]
        if self.sparse:
            grids = ranges
        else:
            grids = []
            ones = [ones_like(r) for r in ranges]
            for i in range(ndim):
                grid = 1
                for j in range(ndim):
                    if j == i:
                        grid = grid * ranges[j]
                    else:
                        grid = grid * ones[j]
                grids.append(grid)
        return grids


mgrid = _nd_grid()
ogrid = _nd_grid(sparse=True)


class PermuteRowElements(Op):
    """Permute the elements of each row (inner-most dim) of a tensor.

    A permutation will be applied to every row (vector) of the input tensor x.
    Depending on the dimensionality of x and the permutation tensor y,
    different cases are possible.
    If y.ndim = 1, y is a single permutation, that will be applied to every
    vector of x. For instance, if x is a matrix, the same permutation will be
    applied to each row of x.
    If x.ndim = y.ndim, each row of x corresponds to a row of y, containing
    a permutation that will be applied to that row. For instance, if x and y
    are two matrices, a different permutation will be applied to each row of x.
    If x.ndim > y.ndim, y will be broadcasted to fit x, then each row (vector)
    of x will be reordered according to the corresponding row of y. (This is
    a generalization of the first case).
    If x.ndim = 1, every permutation in y will be applied to x, and the output
    will contain all the results.
    If x.ndim < y.ndim, x will be broadcasted to fit y, and different
    permutations contained in y will be applied to each vector in x. (This is
    a generalization of the previous case).

    If the "inverse" argument is True, the Op will perform the inverse
    permutation instead.
    """

    __props__ = ()

    def make_node(self, x, y, inverse):
        x = as_tensor_variable(x)
        y = as_tensor_variable(y)
        if inverse:  # as_tensor_variable does not accept booleans
            inverse = as_tensor_variable(1)
        else:
            inverse = as_tensor_variable(0)

        # y should contain integers
        assert y.type.dtype in integer_dtypes
        # Inverse should be an integer scalar
        assert inverse.type.ndim == 0 and inverse.type.dtype in integer_dtypes

        # Match shapes of x and y
        x_dim = x.type.ndim
        y_dim = y.type.ndim

        if x_dim > y_dim:
            y = shape_padleft(y, n_ones=(x_dim - y_dim))
        elif x_dim < y_dim:
            x = shape_padleft(x, n_ones=(y_dim - x_dim))

        out_shape = [
            1 if xb == 1 and yb == 1 else None
            for xb, yb in zip(x.type.shape, y.type.shape)
        ]
        out_type = tensor(dtype=x.type.dtype, shape=out_shape)

        inputlist = [x, y, inverse]
        outputlist = [out_type]
        return Apply(self, inputlist, outputlist)

    def _rec_perform(self, node, x, y, inverse, out, curdim):
        """Perform the permutation by doing a recursion over the input
        dimensions.

        For every dimension, starting with the leftmost, the right set of
        indices is determined (depending if broadcasting or not), then
        the function is recursively called on the appropriate subtensors.

        The terminal case is reached when the current tensors are vector,
        then the permutation contained in y is applied to x.

        Parameters
        ----------
        x: TensorVariable
            The input tensor, on which the permutation is applied.
        y: TensorVariable
            Tensor containing the permutations to apply.
        inverse: bool
            Whether to apply permutations or their inverse.
        out: TensorVariable
            Tensor storing the output result.
        curdim: int
            Counter of the current depth of recursion.

        """
        if len(x.shape) == 1:
            # Numpy advanced indexing works in this case
            if inverse:
                out[y] = x[:]
            else:
                out[:] = x[y]
        else:
            xs0 = x.shape[0]
            ys0 = y.shape[0]
            if xs0 == ys0:
                for i in range(xs0):
                    self._rec_perform(node, x[i], y[i], inverse, out[i], curdim + 1)
            elif ys0 == 1 and node.inputs[1].type.shape[curdim] == 1:
                # Broadcast y
                for i in range(xs0):
                    self._rec_perform(node, x[i], y[0], inverse, out[i], curdim + 1)
            elif xs0 == 1 and node.inputs[0].type.shape[curdim] == 1:
                # Broadcast x
                for i in range(ys0):
                    self._rec_perform(node, x[0], y[i], inverse, out[i], curdim + 1)
            else:
                raise ValueError(f"Dimension mismatch: {xs0}, {ys0}")

    def perform(self, node, inp, out):
        x, y, inverse = inp
        (outs,) = out
        x_s = x.shape
        y_s = y.shape
        assert len(x_s) == len(y_s)

        # Make sure the output is big enough
        out_s = []
        for xdim, ydim in zip(x_s, y_s):
            if xdim == ydim:
                outdim = xdim
            elif xdim == 1:
                outdim = ydim
            elif ydim == 1:
                outdim = xdim
            else:
                raise ValueError(f"Dimension mismatch: {xdim}, {ydim}")
            out_s.append(outdim)

        if outs[0] is None or outs[0].shape != out_s:
            outs[0] = np.empty(out_s, dtype=x.dtype)

        self._rec_perform(node, x, y, inverse, outs[0], curdim=0)

    def infer_shape(self, fgraph, node, in_shapes):
        from pytensor.tensor.math import maximum

        shp_x = in_shapes[0]
        shp_y = in_shapes[1]
        assert len(shp_x) == len(shp_y)
        out_shape = []
        for i in range(len(shp_x)):
            out_shape.append(maximum(shp_x[i], shp_y[i]))
        return [out_shape]

    def grad(self, inp, grads):
        from pytensor.tensor.math import Sum, eq

        x, y, inverse = inp
        (gz,) = grads
        # First, compute the gradient wrt the broadcasted x.
        # If 'inverse' is False (0), apply the inverse of y on gz.
        # Else, apply y on gz.
        gx = permute_row_elements(gz, y, eq(inverse, 0))

        # If x has been broadcasted along some axes, we need to sum
        # the gradient over these axes, but keep the dimension (as
        # broadcastable)
        broadcasted_dims = [
            dim
            for dim in range(gz.type.ndim)
            if x.type.shape[dim] == 1 and gz.type.shape[dim] != 1
        ]
        gx = Sum(axis=broadcasted_dims)(gx)

        # Sum(...) removed the dimensions in broadcasted_dims,
        # so we need to put them back.
        newdims = []
        i = 0
        for dim in range(gz.type.ndim):
            if dim in broadcasted_dims:
                newdims.append("x")
            else:
                newdims.append(i)
                i += 1

        gx = DimShuffle(tuple(s == 1 for s in gx.type.shape), newdims)(gx)
        assert gx.type.ndim == x.type.ndim
        assert all(
            s1 == s2
            for s1, s2 in zip(gx.type.shape, x.type.shape)
            if s1 == 1 or s2 == 1
        )

        # if x is an integer type, then so is the output.
        # this means f(x+eps) = f(x) so the gradient with respect
        # to x is zero
        if x.type.dtype in discrete_dtypes:
            gx = x.zeros_like()

        # The elements of y and of inverse both affect the output,
        # so they are connected to the output,
        # and the transformation isn't defined if their values
        # are non-integer, so the gradient with respect to them is
        # undefined

        return [gx, grad_undefined(self, 1, y), grad_undefined(self, 1, inverse)]


_permute_row_elements = PermuteRowElements()


def permute_row_elements(x, y, inverse=0):
    return _permute_row_elements(x, y, inverse)


def inverse_permutation(perm):
    """Computes the inverse of permutations.

    Each row of input should contain a permutation of the first integers.

    """
    _perm = as_tensor_variable(perm)
    return permute_row_elements(
        arange(_perm.shape[-1], dtype=_perm.dtype), _perm, inverse=True
    )


# TODO: optimization to insert ExtractDiag with view=True
class ExtractDiag(Op):
    """
    Return specified diagonals.

    If x is 2-D, returns the diagonal of x with the given offset,
    i.e., the collection of elements of the form x[i, i+offset].
    If x has more than two dimensions, then the axes specified by
    axis1 and axis2 are used to determine the 2-D sub-array whose
    diagonal is returned. The shape of the resulting array can be
    determined by removing axis1 and axis2 and appending an index
    to the right equal to the size of the resulting diagonals.

    Parameters
    ----------
    x: A tensor variable with x.ndim >= 2.

    offset: Offset of the diagonal from the main diagonal.
        Can be positive or negative.
        Defaults to main diagonal (0).

    axis1: Axis to be used as the first axis of the 2-D
        sub-arrays from which the diagonals should be taken.
        Defaults to first axis (0).

    axis2: Axis to be used as the second axis of the 2-D
        sub-arrays from which the diagonals should be taken.
        Defaults to second axis (1).



    Returns
    -------
    array_of_diagonals:
        If x is 2-D, a 1-D array of the same type as a
        containing the diagonal is returned.
        If the dimension of x is greater than two, then an
        array of diagonals is returned, "packed" from left-most
        dimension to right-most (e.g., if x is 3-D, then the
        diagonals are "packed" along rows).



    Raises
    ------
    ValueError
        If the dimension of x is less than 2.


    See Also
    --------
    numpy.diagonal:
        https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.diagonal.html
    """

    __props__ = ("offset", "axis1", "axis2", "view")

    def __init__(self, offset=0, axis1=0, axis2=1, view=False):
        self.view = view
        if self.view:
            self.view_map = {0: [0]}
        if axis1 < 0 or axis2 < 0:
            raise NotImplementedError(
                "ExtractDiag does not support negative axis. Use pytensor.tensor.diagonal instead."
            )
        if axis1 == axis2:
            raise ValueError("axis1 and axis2 cannot be the same")
        # Sort axis
        if axis1 > axis2:
            axis1, axis2, offset = axis2, axis1, -offset
        self.axis1 = axis1
        self.axis2 = axis2
        self.offset = offset

    def make_node(self, x):
        x = as_tensor_variable(x)

        if x.ndim < 2:
            raise ValueError("ExtractDiag needs an input with 2 or more dimensions", x)

        out_shape = [
            st_dim
            for i, st_dim in enumerate(x.type.shape)
            if i not in (self.axis1, self.axis2)
        ] + [None]

        return Apply(
            self,
            [x],
            [x.type.clone(dtype=x.dtype, shape=tuple(out_shape))()],
        )

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        z[0] = x.diagonal(self.offset, self.axis1, self.axis2)
        if not self.view:
            z[0] = z[0].copy()

    def grad(self, inputs, gout):
        # Avoid circular import
        from pytensor.tensor.subtensor import set_subtensor

        (x,) = inputs
        (gz,) = gout

        axis1, axis2, offset = self.axis1, self.axis2, self.offset

        # Start with zeros (and axes in the front)
        x_grad = zeros_like(moveaxis(x, (axis1, axis2), (0, 1)))

        # Fill zeros with output diagonal
        xdiag = alloc_diag(gz, offset=0, axis1=0, axis2=1)
        z_len = xdiag.shape[0]
        if offset >= 0:
            diag_slices = (slice(None, z_len), slice(offset, offset + z_len))
        else:
            diag_slices = (slice(abs(offset), abs(offset) + z_len), slice(None, z_len))
        x_grad = set_subtensor(x_grad[diag_slices], xdiag)

        # Put axes back in their original positions
        x_grad = moveaxis(x_grad, (0, 1), (axis1, axis2))
        return [x_grad]

    def infer_shape(self, fgraph, node, shapes):
        from pytensor.tensor.math import clip, minimum

        (in_shape,) = shapes
        dim1 = in_shape[self.axis1]
        dim2 = in_shape[self.axis2]
        out_shape = [
            d for i, d in enumerate(in_shape) if i not in (self.axis1, self.axis2)
        ]
        # The following logic is inspired by C code of PyArray_Diagonal().
        offset = self.offset
        if offset > 0:
            diag_size = clip(dim2 - offset, 0, dim1)
        elif offset < 0:
            diag_size = clip(dim1 + offset, 0, dim2)
        else:
            diag_size = minimum(dim1, dim2)
        out_shape.append(diag_size)
        return [tuple(out_shape)]

    def __setstate__(self, state):
        self.__dict__.update(state)

        if self.view:
            self.view_map = {0: [0]}

        if "offset" not in state:
            self.offset = 0
        if "axis1" not in state:
            self.axis1 = 0
        if "axis2" not in state:
            self.axis2 = 1


def extract_diag(x):
    warnings.warn(
        "pytensor.tensor.extract_diag is deprecated. Use pytensor.tensor.diagonal instead.",
        FutureWarning,
    )
    return diagonal(x)


def diagonal(a, offset=0, axis1=0, axis2=1):
    """
    A helper function for `ExtractDiag`. It accepts tensor with
    `ndim >= 2` as input. The name `diagonal` is just meant to keep it
    consistent with numpy.

    Parameters
    ----------
    a : symbolic tensor
    offset : int
        offset
    axis1 : int
    axis2 : int

    Returns
    -------
    tensor : symbolic tensor

    """
    a = as_tensor_variable(a)
    axis1, axis2 = normalize_axis_tuple((axis1, axis2), ndim=a.type.ndim)
    return ExtractDiag(offset, axis1, axis2)(a)


@_vectorize_node.register(ExtractDiag)
def vectorize_extract_diag(op: ExtractDiag, node, batch_x):
    core_ndim = node.inputs[0].type.ndim
    batch_ndim = batch_x.type.ndim - core_ndim
    batch_axis1, batch_axis2 = get_normalized_batch_axes(
        (op.axis1, op.axis2), core_ndim, batch_ndim
    )

    return diagonal(
        batch_x,
        offset=op.offset,
        axis1=batch_axis1,
        axis2=batch_axis2,
    ).owner


def trace(a, offset=0, axis1=0, axis2=1):
    """
    Returns the sum along diagonals of the array.

    Equivalent to `numpy.trace`
    """
    return diagonal(a, offset=offset, axis1=axis1, axis2=axis2).sum(-1)


class AllocDiag(Op):
    """An `Op` that copies a vector to the diagonal of a zero-ed matrix."""

    __props__ = ("offset", "axis1", "axis2")

    def __init__(self, offset=0, axis1=0, axis2=1):
        """
        Parameters
        ----------
        offset: int
            Offset of the diagonal from the main diagonal defined by `axis1`
            and `axis2`. Can be positive or negative.  Defaults to main
            diagonal (i.e. 0).
        axis1: int
            Axis to be used as the first axis of the 2-D sub-arrays to which
            the diagonals will be allocated.  Defaults to first axis (i.e. 0).
        axis2: int
            Axis to be used as the second axis of the 2-D sub-arrays to which
            the diagonals will be allocated.  Defaults to second axis (i.e. 1).
        """
        warnings.warn(
            "AllocDiag is deprecated. Use `alloc_diag` instead",
            FutureWarning,
        )
        self.offset = offset
        if axis1 < 0 or axis2 < 0:
            raise NotImplementedError("AllocDiag does not support negative axis")
        if axis1 == axis2:
            raise ValueError("axis1 and axis2 cannot be the same")
        self.axis1 = axis1
        self.axis2 = axis2

    def make_node(self, diag):
        diag = as_tensor_variable(diag)
        if diag.type.ndim < 1:
            raise ValueError(
                "AllocDiag needs an input with 1 or more dimensions", diag.type
            )
        return Apply(
            self,
            [diag],
            [diag.type.clone(shape=(None,) * (diag.ndim + 1))()],
        )

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs

        axis1 = np.minimum(self.axis1, self.axis2)
        axis2 = np.maximum(self.axis1, self.axis2)
        offset = self.offset

        # Create array with one extra dimension for resulting matrix
        result_shape = x.shape[:-1] + (x.shape[-1] + abs(offset),) * 2
        result = np.zeros(result_shape, dtype=x.dtype)

        # Create slice for diagonal in final 2 axes
        idxs = np.arange(x.shape[-1])
        diagonal_slice = (len(result_shape) - 2) * [slice(None)] + [
            idxs + np.maximum(0, -offset),
            idxs + np.maximum(0, offset),
        ]

        # Fill in final 2 axes with x
        result[tuple(diagonal_slice)] = x

        if len(x.shape) > 1:
            # Re-order axes so they correspond to diagonals at axis1, axis2
            axes = list(range(len(x.shape[:-1])))
            last_idx = axes[-1]
            axes = axes[:axis1] + [last_idx + 1] + axes[axis1:]
            axes = axes[:axis2] + [last_idx + 2] + axes[axis2:]
            result = result.transpose(axes)

        z[0] = result

    def grad(self, inputs, gout):
        (gz,) = gout
        return [diagonal(gz, offset=self.offset, axis1=self.axis1, axis2=self.axis2)]

    def infer_shape(self, fgraph, nodes, shapes):
        (x_shape,) = shapes
        axis1 = np.minimum(self.axis1, self.axis2)
        axis2 = np.maximum(self.axis1, self.axis2)

        result_shape = list(x_shape[:-1])
        diag_shape = x_shape[-1] + abs(self.offset)
        result_shape = result_shape[:axis1] + [diag_shape] + result_shape[axis1:]
        result_shape = result_shape[:axis2] + [diag_shape] + result_shape[axis2:]
        return [tuple(result_shape)]

    def __setstate__(self, state):
        if "view_map" in state:
            del state["view_map"]

        self.__dict__.update(state)

        if "offset" not in state:
            self.offset = 0
        if "axis1" not in state:
            self.axis1 = 0
        if "axis2" not in state:
            self.axis2 = 1


def alloc_diag(diag, offset=0, axis1=0, axis2=1):
    """Insert a vector on the diagonal of a zero-ed matrix.

    diagonal(alloc_diag(x)) == x
    """
    from pytensor.tensor import set_subtensor

    diag = as_tensor_variable(diag)
    axis1, axis2 = normalize_axis_tuple((axis1, axis2), ndim=diag.type.ndim + 1)
    if axis1 > axis2:
        axis1, axis2 = axis2, axis1

    # Create array with one extra dimension for resulting matrix
    result_shape = tuple(diag.shape)[:-1] + (diag.shape[-1] + abs(offset),) * 2
    result = zeros(result_shape, dtype=diag.dtype)

    # Create slice for diagonal in final 2 axes
    idxs = arange(diag.shape[-1])
    diagonal_slice = (slice(None),) * (len(result_shape) - 2) + (
        idxs + np.maximum(0, -offset),
        idxs + np.maximum(0, offset),
    )

    # Fill in final 2 axes with diag
    result = set_subtensor(result[diagonal_slice], diag)

    if diag.type.ndim > 1:
        # Re-order axes so they correspond to diagonals at axis1, axis2
        axes = list(range(diag.type.ndim - 1))
        last_idx = axes[-1]
        axes = axes[:axis1] + [last_idx + 1] + axes[axis1:]
        axes = axes[:axis2] + [last_idx + 2] + axes[axis2:]
        result = result.transpose(axes)

    return result


def diag(v, k=0):
    """
    A helper function for two ops: `ExtractDiag` and
    `AllocDiag`. The name `diag` is meant to keep it consistent
    with numpy. It both accepts tensor vector and tensor matrix.
    While the passed tensor variable `v` has `v.ndim==2`, it builds a
    `ExtractDiag` instance, and returns a vector with its entries equal to
    `v`'s main diagonal; otherwise if `v.ndim` is `1`, it builds an `AllocDiag`
    instance, and returns a matrix with `v` at its k-th diaogonal.

    Parameters
    ----------
    v : symbolic tensor
    k : int
        offset

    Returns
    -------
    tensor : symbolic tensor

    """

    _v = as_tensor_variable(v)

    if _v.ndim == 1:
        return alloc_diag(_v, offset=k)
    elif _v.ndim == 2:
        return diagonal(_v, offset=k)
    else:
        raise ValueError("Input must be 1- or 2-d.")


def stacklists(arg):
    """
    Recursively stack lists of tensors to maintain similar structure.

    This function can create a tensor from a shaped list of scalars:

    Examples
    --------
    >>> from pytensor.tensor import stacklists
    >>> from pytensor.tensor.type import scalars, matrices
    >>> from pytensor import function
    >>> a, b, c, d = scalars('abcd')
    >>> X = stacklists([[a, b], [c, d]])
    >>> f = function([a, b, c, d], X)
    >>> f(1, 2, 3, 4)
    array([[ 1.,  2.],
           [ 3.,  4.]], dtype=float32)

    We can also stack arbitrarily shaped tensors. Here we stack matrices into
    a 2 by 2 grid:

    >>> from numpy import ones
    >>> a, b, c, d = matrices('abcd')
    >>> X = stacklists([[a, b], [c, d]])
    >>> f = function([a, b, c, d], X)
    >>> x = ones((4, 4), 'float32')
    >>> f(x, x, x, x).shape
    (2, 2, 4, 4)

    """
    if isinstance(arg, tuple | list):
        return stack(list(map(stacklists, arg)))
    else:
        return arg


def swapaxes(y, axis1: int, axis2: int) -> TensorVariable:
    "Swap the axes of a tensor."
    y = as_tensor_variable(y)
    ndim = y.ndim
    li = list(range(0, ndim))
    li[axis1], li[axis2] = li[axis2], li[axis1]
    return y.dimshuffle(li)


def moveaxis(
    a: np.ndarray | TensorVariable,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> TensorVariable:
    """Move axes of a TensorVariable to new positions.

    Other axes remain in their original order.

    Parameters
    ----------
    a
        The TensorVariable whose axes should be reordered.
    source
        Original positions of the axes to move. These must be unique.
    destination
        Destination positions for each of the original axes. These must also be
        unique.

    Returns
    -------
    result
        TensorVariable with moved axes.

    """

    a = as_tensor_variable(a)

    source = normalize_axis_tuple(source, a.ndim, "source")
    destination = normalize_axis_tuple(destination, a.ndim, "destination")

    if len(source) != len(destination):
        raise ValueError(
            "`source` and `destination` arguments must have the same number of elements"
        )

    order = [n for n in range(a.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    result = a.dimshuffle(order)
    return result


def choose(a, choices, mode="raise"):
    """
    Construct an array from an index array and a set of arrays to choose from.

    First of all, if confused or uncertain, definitely look at the Examples -
    in its full generality, this function is less simple than it might seem
    from the following code description (below ndi = numpy.lib.index_tricks):

    np.choose(a,c) == np.array([c[a[I]][I] for I in ndi.ndindex(a.shape)]).

    But this omits some subtleties. Here is a fully general summary:

    Given an ``index`` array (a) of integers and a sequence of n arrays
    (choices), a and each choice array are first broadcast, as necessary,
    to arrays of a common shape; calling these Ba and
    Bchoices[i], i = 0,...,n-1 we have that, necessarily,
    Ba.shape == Bchoices[i].shape for each i.
    Then, a new array with shape Ba.shape is created as follows:

    - if mode=raise (the default), then, first of all, each element of a
      (and thus Ba) must be in the range [0, n-1]; now, suppose that
      i (in that range) is the value at the (j0, j1, ..., jm) position in Ba -
      then the value at the same position in the new array is the value in
      Bchoices[i] at that same position;

    - if mode=wrap, values in a (and thus Ba) may be any (signed) integer;
      modular arithmetic is used to map integers outside the range [0, n-1]
      back into that range; and then the new array is constructed as above;

    - if mode=clip, values in a (and thus Ba) may be any (signed) integer;
      negative integers are mapped to 0; values greater than n-1 are mapped
      to n-1; and then the new array is constructed as above.

    Parameters
    ----------
    a : int array
        This array must contain integers in [0, n-1], where n is the number of
        choices, unless mode=wrap or mode=clip, in which cases any integers
        are permissible.
    choices : sequence of arrays
        Choice arrays. a and all of the choices must be broadcastable to
        the same shape. If choices is itself an array (not recommended),
        then its outermost dimension (i.e., the one corresponding to
        choices.shape[0]) is taken as defining the ``sequence``.
    mode : {``raise`` (default), ``wrap``, ``clip``}, optional
        Specifies how indices outside [0, n-1] will be treated:
        ``raise`` : an exception is raised
        ``wrap`` : value becomes value mod n
        ``clip`` : values < 0 are mapped to 0, values > n-1 are mapped to n-1

    Returns
    -------
    merged_array - array
        The merged result.

    Raises
    ------
    ValueError - shape mismatch
        If a and each choice array are not all broadcastable to the same shape.

    """
    return Choose(mode)(a, choices)


class Choose(Op):
    __props__ = ("mode",)

    def __init__(self, mode):
        assert mode in ("raise", "wrap", "clip")
        self.mode = mode

    def infer_shape(self, fgraph, node, shapes):
        a_shape, choices_shape = shapes
        out_shape = pytensor.tensor.extra_ops.broadcast_shape(
            a_shape, choices_shape[1:], arrays_are_shapes=True
        )

        return [out_shape]

    def make_node(self, a, choices):
        # Import here as it isn't imported by default and we can't
        # import at the top as it would cause circular import.
        import pytensor.typed_list

        a = as_tensor_variable(a)
        if a.dtype not in discrete_dtypes:
            raise TypeError(
                f"choose first argument must have an [u]int* dtype. Got {a.dtype}."
            )

        # Only use make_list if choices have inconsistent shapes
        # otherwise use as_tensor_variable
        if isinstance(choices, tuple | list):
            choice = pytensor.typed_list.make_list(choices)
        else:
            choice = as_tensor_variable(choices)

        (out_shape,) = self.infer_shape(
            None, None, [shape_tuple(a), shape_tuple(choice)]
        )

        static_out_shape = ()
        for s in out_shape:
            try:
                s_val = pytensor.get_underlying_scalar_constant(s)
            except (NotScalarConstantError, AttributeError):
                s_val = None

            if s_val == 1:
                static_out_shape += (1,)
            else:
                static_out_shape += (None,)

        o = TensorType(choice.dtype, shape=static_out_shape)
        return Apply(self, [a, choice], [o()])

    def perform(self, node, inputs, outputs):
        (z,) = outputs
        a = inputs[0]
        choice = inputs[1]
        # TODO reuse out?
        z[0] = np.choose(a, choice, mode=self.mode)


class AllocEmpty(COp):
    """Implement Alloc on the cpu, but without initializing memory."""

    _output_type_depends_on_input_value = True

    __props__ = ("dtype",)
    params_type = ParamsType(typecode=int32)

    # specify the type of the data
    def __init__(self, dtype):
        assert isinstance(dtype, str), dtype
        self.dtype = dtype.lower()

    @property
    def typecode(self):
        return np.dtype(self.dtype).num

    def make_node(self, *_shape):
        _shape, static_shape = infer_static_shape(_shape)
        otype = TensorType(dtype=self.dtype, shape=static_shape)
        output = otype()

        output.tag.values_eq_approx = values_eq_approx_always_true
        # The output can contain nan/inf.  output.type is a new
        # instance, so we can do this only for that variable.
        output.type.filter_checks_isfinite = False

        # We can't reuse filter_checks_isfinite as by default it is
        # False and it is set to true only in DebugMode.
        # We can't set it in the type as other make_node can reuse the type.
        # We can't set it in the variable as it isn't copied when we copy
        # the variable. So we set it in the tag.
        output.tag.nan_guard_mode_check = False
        return Apply(self, _shape, [output])

    def debug_perform(self, node, inputs, out_):
        self.perform(node, inputs, out_)
        out_[0][0].fill(-123456789)

    def perform(self, node, inputs, out_):
        (out,) = out_
        sh = tuple([int(i) for i in inputs])
        if out[0] is None or out[0].shape != sh:
            out[0] = np.empty(sh, dtype=self.dtype)

    def c_code(self, node, name, inputs, out_, sub):
        (out,) = out_
        fail = sub["fail"]
        shps = inputs
        nd = len(shps)
        params = sub["params"]
        str = f"npy_intp dims[{nd}];\n"
        for idx, sh in enumerate(shps):
            str += (
                "dims[{idx}] ="
                "((npy_intp)((dtype_{sh}*)"
                " PyArray_DATA({sh}))[0]);\n".format(**locals())
            )

        # Validate that the output storage exists
        str += f"if({out}==NULL\n"
        for idx, sh in enumerate(shps):
            str += f"||PyArray_DIMS({out})[{idx}]!=dims[{idx}]"

        str += """){{
            /* Reference received to invalid output variable.
            Decrease received reference's ref count and allocate new
            output variable */
            Py_XDECREF({out});
            {out} = (PyArrayObject*)PyArray_EMPTY({nd},
                                                    dims,
                                                    {params}->typecode,
                                                    0);
            if (!{out})
            {{
                PyErr_SetString(PyExc_MemoryError, "alloc failed");
                {fail};
            }}
        }}
        """.format(**locals())
        return str

    def infer_shape(self, fgraph, node, input_shapes):
        return [node.inputs]

    def c_code_cache_version(self):
        return (4,)

    def do_constant_folding(self, fgraph, node):
        return False

    def connection_pattern(self, node):
        return [[False] for i in node.inputs]

    def grad(self, inputs, grads):
        return [DisconnectedType()() for i in inputs]

    def R_op(self, inputs, eval_points):
        return [zeros(inputs, self.dtype)]


def empty(shape, dtype=None):
    """Return a new array of given shape and type, without initializing entries.

    See ``numpy.empty``.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        Desired output data-type for the array, e.g, `numpy.int8`. Default is
        `numpy.float64`.
    """
    if not (
        isinstance(shape, np.ndarray | Sequence)
        or (isinstance(shape, TensorVariable) and shape.ndim > 0)
    ):
        shape = [shape]
    if dtype is None:
        dtype = config.floatX
    return AllocEmpty(dtype)(*shape)


def empty_like(
    prototype: np.ndarray | TensorVariable,
    dtype: str | np.generic | np.dtype | None = None,
) -> TensorVariable:
    """Return a new array with the same shape and type as a given array.

    See ``numpy.empty_like``.

    Parameters
    ----------
    prototype
        The shape and data-type of `prototype` define these same attributes
        of the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    """
    if dtype is None:
        dtype = prototype.dtype

    return empty(shape(prototype), dtype)


def atleast_Nd(
    *arys: np.ndarray | TensorVariable, n: int = 1, left: bool = True
) -> TensorVariable:
    """Convert inputs to arrays with at least `n` dimensions."""
    res = []
    for ary in arys:
        ary = as_tensor(ary)

        if ary.ndim >= n:
            result = ary
        else:
            result = (
                shape_padleft(ary, n - ary.ndim)
                if left
                else shape_padright(ary, n - ary.ndim)
            )

        res.append(result)

    if len(res) == 1:
        return res[0]
    else:
        return res


atleast_1d = partial(atleast_Nd, n=1)
atleast_2d = partial(atleast_Nd, n=2)
atleast_3d = partial(atleast_Nd, n=3)


def expand_dims(
    a: np.ndarray | TensorVariable, axis: tuple[int, ...]
) -> TensorVariable:
    """Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded
    array shape.

    Parameters
    ----------
    a :
        The input array.
    axis :
        Position in the expanded axes where the new axis is placed.
        If `axis` is empty, `a` will be returned immediately.
    Returns
    -------
    `a` with a new axis at the `axis` position.
    """
    a = as_tensor(a)

    if not isinstance(axis, tuple | list):
        axis = (axis,)

    out_ndim = len(axis) + a.ndim
    axis = np.core.numeric.normalize_axis_tuple(axis, out_ndim)

    if not axis:
        return a

    dim_it = iter(range(a.ndim))
    pattern = ["x" if ax in axis else next(dim_it) for ax in range(out_ndim)]

    return a.dimshuffle(pattern)


def _make_along_axis_idx(arr_shape, indices, axis):
    """Take from `numpy.lib.shape_base`."""
    if str(indices.dtype) not in int_dtypes:
        raise IndexError("`indices` must be an integer array")

    shape_ones = (1,) * indices.ndim
    dest_dims = [*range(axis), None, *range(axis + 1, indices.ndim)]

    # build a fancy index, consisting of orthogonal aranges, with the
    # requested index inserted at the right location
    fancy_index = []
    for dim, n in zip(dest_dims, arr_shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1 :]
            fancy_index.append(arange(n).reshape(ind_shape))

    return tuple(fancy_index)


def take_along_axis(arr, indices, axis=0):
    """Take values from the input array by matching 1d index and data slices.

    This iterates over matching 1d slices oriented along the specified axis in
    the index and data arrays, and uses the former to look up values in the
    latter. These slices can be different lengths.

    Functions returning an index along an axis, like `argsort` and
    `argpartition`, produce suitable indices for this function.
    """
    arr = as_tensor_variable(arr)
    indices = as_tensor_variable(indices)
    # normalize inputs
    if axis is None:
        arr = arr.flatten()
        axis = 0
    else:
        axis = normalize_axis_index(axis, arr.ndim)

    if arr.ndim != indices.ndim:
        raise ValueError("`indices` and `arr` must have the same number of dimensions")

    # use the fancy index
    return arr[_make_along_axis_idx(arr.shape, indices, axis)]


def ix_(*args):
    """
    PyTensor np.ix_ analog

    See numpy.lib.index_tricks.ix_ for reference
    """
    out = []
    nd = len(args)
    for k, new in enumerate(args):
        if new is None:
            out.append(slice(None))
        new = as_tensor(new)
        if new.ndim != 1:
            raise ValueError("Cross index must be 1 dimensional")
        new = new.reshape((1,) * k + (new.size,) + (1,) * (nd - k - 1))
        out.append(new)
    return tuple(out)


__all__ = [
    "take_along_axis",
    "expand_dims",
    "atleast_Nd",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "choose",
    "swapaxes",
    "moveaxis",
    "stacklists",
    "diag",
    "diagonal",
    "inverse_permutation",
    "permute_row_elements",
    "mgrid",
    "ogrid",
    "arange",
    "tile",
    "flatten",
    "is_flat",
    "vertical_stack",
    "horizontal_stack",
    "get_vector_length",
    "concatenate",
    "stack",
    "roll",
    "join",
    "split",
    "transpose",
    "matrix_transpose",
    "extract_constant",
    "default",
    "tensor_copy",
    "transfer",
    "alloc",
    "identity_like",
    "eye",
    "triu",
    "tril",
    "tri",
    "nonzero_values",
    "flatnonzero",
    "nonzero",
    "ones",
    "zeros",
    "zeros_like",
    "ones_like",
    "fill",
    "second",
    "where",
    "switch",
    "cast",
    "scalar_from_tensor",
    "tensor_from_scalar",
    "get_scalar_constant_value",
    "get_underlying_scalar_constant_value",
    "constant",
    "as_tensor_variable",
    "as_tensor",
    "extract_diag",
    "full",
    "full_like",
    "empty",
    "empty_like",
    "trace",
    "tril_indices",
    "tril_indices_from",
    "triu_indices",
    "triu_indices_from",
]
