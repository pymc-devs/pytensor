import operator
import sys
import warnings
from copy import copy
from functools import singledispatch
from textwrap import dedent

import numba
import numba.np.unsafe.ndarray as numba_ndarray
import numpy as np
import scipy
import scipy.special
from llvmlite import ir
from numba import types
from numba.core.errors import NumbaWarning, TypingError
from numba.cpython.unsafe.tuple import tuple_setitem  # noqa: F401
from numba.extending import box, overload

from pytensor import In, config
from pytensor.compile import NUMBA
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.function.types import add_supervisor_to_fgraph
from pytensor.compile.ops import DeepCopyOp
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.type import Type
from pytensor.ifelse import IfElse
from pytensor.link.numba.dispatch.sparse import CSCMatrixType, CSRMatrixType
from pytensor.link.utils import (
    compile_function_src,
    fgraph_to_python,
)
from pytensor.scalar.basic import ScalarType
from pytensor.scalar.math import Softplus
from pytensor.sparse import SparseTensorType
from pytensor.tensor.basic import Nonzero
from pytensor.tensor.blas import BatchedDot
from pytensor.tensor.math import Dot
from pytensor.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape
from pytensor.tensor.slinalg import Solve
from pytensor.tensor.type import TensorType
from pytensor.tensor.type_other import MakeSlice, NoneConst


def global_numba_func(func):
    """Use to return global numba functions in numba_funcify_*.

    This allows tests to remove the compilation using mock.
    """
    return func


def numba_njit(*args, fastmath=None, **kwargs):
    kwargs.setdefault("cache", config.numba__cache)
    kwargs.setdefault("no_cpython_wrapper", True)
    kwargs.setdefault("no_cfunc_wrapper", True)
    if fastmath is None:
        if config.numba__fastmath:
            # Opinionated default on fastmath flags
            # https://llvm.org/docs/LangRef.html#fast-math-flags
            fastmath = {
                "arcp",  # Allow Reciprocal
                "contract",  # Allow floating-point contraction
                "afn",  # Approximate functions
                "reassoc",
                "nsz",  # no-signed zeros
            }
        else:
            fastmath = False

    # Suppress cache warning for internal functions
    # We have to add an ansi escape code for optional bold text by numba
    warnings.filterwarnings(
        "ignore",
        message=(
            "(\x1b\\[1m)*"  # ansi escape code for bold text
            "Cannot cache compiled function "
            '"(numba_funcified_fgraph|store_core_outputs|cholesky|solve|solve_triangular|cho_solve)" '
            "as it uses dynamic globals"
        ),
        category=NumbaWarning,
    )

    if len(args) > 0 and callable(args[0]):
        return numba.njit(*args[1:], fastmath=fastmath, **kwargs)(args[0])

    return numba.njit(*args, fastmath=fastmath, **kwargs)


def numba_vectorize(*args, **kwargs):
    if len(args) > 0 and callable(args[0]):
        return numba.vectorize(*args[1:], cache=config.numba__cache, **kwargs)(args[0])

    return numba.vectorize(*args, cache=config.numba__cache, **kwargs)


def get_numba_type(
    pytensor_type: Type,
    layout: str = "A",
    force_scalar: bool = False,
    reduce_to_scalar: bool = False,
) -> numba.types.Type:
    r"""Create a Numba type object for a :class:`Type`.

    Parameters
    ----------
    pytensor_type
        The :class:`Type` to convert.
    layout
        The :class:`numpy.ndarray` layout to use.
    force_scalar
        Ignore dimension information and return the corresponding Numba scalar types.
    reduce_to_scalar
        Return Numba scalars for zero dimensional :class:`TensorType`\s.
    """

    if isinstance(pytensor_type, TensorType):
        dtype = pytensor_type.numpy_dtype
        numba_dtype = numba.from_dtype(dtype)
        if force_scalar or (
            reduce_to_scalar and getattr(pytensor_type, "ndim", None) == 0
        ):
            return numba_dtype
        return numba.types.Array(numba_dtype, pytensor_type.ndim, layout)
    elif isinstance(pytensor_type, ScalarType):
        dtype = np.dtype(pytensor_type.dtype)
        numba_dtype = numba.from_dtype(dtype)
        return numba_dtype
    elif isinstance(pytensor_type, SparseTensorType):
        dtype = pytensor_type.numpy_dtype
        numba_dtype = numba.from_dtype(dtype)
        if pytensor_type.format == "csr":
            return CSRMatrixType(numba_dtype)
        if pytensor_type.format == "csc":
            return CSCMatrixType(numba_dtype)

        raise NotImplementedError()
    else:
        raise NotImplementedError(f"Numba type not implemented for {pytensor_type}")


def create_numba_signature(
    node_or_fgraph: FunctionGraph | Apply,
    force_scalar: bool = False,
    reduce_to_scalar: bool = False,
) -> numba.types.Type:
    """Create a Numba type for the signature of an `Apply` node or `FunctionGraph`."""
    input_types = [
        get_numba_type(
            inp.type, force_scalar=force_scalar, reduce_to_scalar=reduce_to_scalar
        )
        for inp in node_or_fgraph.inputs
    ]

    output_types = [
        get_numba_type(
            out.type, force_scalar=force_scalar, reduce_to_scalar=reduce_to_scalar
        )
        for out in node_or_fgraph.outputs
    ]

    if len(output_types) > 1:
        return numba.types.Tuple(output_types)(*input_types)
    elif len(output_types) == 1:
        return output_types[0](*input_types)
    else:
        return numba.types.void(*input_types)


def slice_new(self, start, stop, step):
    fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.pyobj, self.pyobj])
    fn = self._get_function(fnty, name="PySlice_New")
    return self.builder.call(fn, [start, stop, step])


def enable_slice_boxing():
    """Enable boxing for Numba's native ``slice``s.

    TODO: this can be removed when https://github.com/numba/numba/pull/6939 is
    merged and a release is made.
    """

    @box(types.SliceType)
    def box_slice(typ, val, c):
        """Implement boxing for ``slice`` objects in Numba.

        This makes it possible to return an Numba's internal representation of a
        ``slice`` object as a proper ``slice`` to Python.
        """
        start = c.builder.extract_value(val, 0)
        stop = c.builder.extract_value(val, 1)

        none_val = ir.Constant(ir.IntType(64), sys.maxsize)

        start_is_none = c.builder.icmp_signed("==", start, none_val)
        start = c.builder.select(
            start_is_none,
            c.pyapi.get_null_object(),
            c.box(types.int64, start),
        )

        stop_is_none = c.builder.icmp_signed("==", stop, none_val)
        stop = c.builder.select(
            stop_is_none,
            c.pyapi.get_null_object(),
            c.box(types.int64, stop),
        )

        if typ.has_step:
            step = c.builder.extract_value(val, 2)
            step_is_none = c.builder.icmp_signed("==", step, none_val)
            step = c.builder.select(
                step_is_none,
                c.pyapi.get_null_object(),
                c.box(types.int64, step),
            )
        else:
            step = c.pyapi.get_null_object()

        slice_val = slice_new(c.pyapi, start, stop, step)

        return slice_val

    @numba.extending.overload(operator.contains)
    def in_seq_empty_tuple(x, y):
        if isinstance(x, types.Tuple) and not x.types:
            return lambda x, y: False


enable_slice_boxing()


def to_scalar(x):
    return np.asarray(x).item()


@numba.extending.overload(to_scalar)
def impl_to_scalar(x):
    if isinstance(x, numba.types.Number | numba.types.Boolean):
        return lambda x: x
    elif isinstance(x, numba.types.Array):
        return lambda x: x.item()
    else:
        raise TypingError(f"{x} must be a scalar compatible type.")


def create_tuple_creator(f, n):
    """Construct a compile-time ``tuple``-comprehension-like loop.

    See https://github.com/numba/numba/issues/2771#issuecomment-414358902
    """
    assert n > 0

    f = numba_njit(f)

    @numba_njit
    def creator(args):
        return (f(0, *args),)

    for i in range(1, n):

        @numba_njit
        def creator(args, creator=creator, i=i):
            return (*creator(args), f(i, *args))

    return numba_njit(lambda *args: creator(args))


def create_tuple_string(x):
    args = ", ".join(x + ([""] if len(x) == 1 else []))
    return f"({args})"


def create_arg_string(x):
    args = ", ".join(x)
    return args


@singledispatch
def numba_typify(data, dtype=None, **kwargs):
    return data


def generate_fallback_impl(op, node=None, storage_map=None, **kwargs):
    """Create a Numba compatible function from a Pytensor `Op`."""

    warnings.warn(
        f"Numba will use object mode to run {op}'s perform method",
        UserWarning,
    )

    n_outputs = len(node.outputs)

    if n_outputs > 1:
        ret_sig = numba.types.Tuple([get_numba_type(o.type) for o in node.outputs])
    else:
        ret_sig = get_numba_type(node.outputs[0].type)

    output_types = tuple(out.type for out in node.outputs)

    def py_perform(inputs):
        outputs = [[None] for i in range(n_outputs)]
        op.perform(node, inputs, outputs)
        return outputs

    if n_outputs == 1:

        def py_perform_return(inputs):
            return output_types[0].filter(py_perform(inputs)[0][0])

    else:

        def py_perform_return(inputs):
            # strict=False because we are in a hot loop
            return tuple(
                out_type.filter(out[0])
                for out_type, out in zip(output_types, py_perform(inputs), strict=False)
            )

    @numba_njit
    def perform(*inputs):
        with numba.objmode(ret=ret_sig):
            ret = py_perform_return(inputs)
        return ret

    return perform


@singledispatch
def numba_funcify(op, node=None, storage_map=None, **kwargs):
    """Generate a numba function for a given op and apply node.

    The resulting function will usually use the `no_cpython_wrapper`
    argument in numba, so it can not be called directly from python,
    but only from other jit functions.
    """
    return generate_fallback_impl(op, node, storage_map, **kwargs)


@numba_funcify.register(OpFromGraph)
def numba_funcify_OpFromGraph(op, node=None, **kwargs):
    _ = kwargs.pop("storage_map", None)

    # Apply inner rewrites
    # TODO: Not sure this is the right place to do this, should we have a rewrite that
    #  explicitly triggers the optimization of the inner graphs of OpFromGraph?
    #  The C-code defers it to the make_thunk phase
    fgraph = op.fgraph
    add_supervisor_to_fgraph(
        fgraph=fgraph,
        input_specs=[In(x, borrow=True, mutable=False) for x in fgraph.inputs],
        accept_inplace=True,
    )
    NUMBA.optimizer(fgraph)
    fgraph_fn = numba_njit(numba_funcify(op.fgraph, **kwargs))

    if len(op.fgraph.outputs) == 1:

        @numba_njit
        def opfromgraph(*inputs):
            return fgraph_fn(*inputs)[0]

    else:

        @numba_njit
        def opfromgraph(*inputs):
            return fgraph_fn(*inputs)

    return opfromgraph


@numba_funcify.register(FunctionGraph)
def numba_funcify_FunctionGraph(
    fgraph,
    node=None,
    fgraph_name="numba_funcified_fgraph",
    **kwargs,
):
    return fgraph_to_python(
        fgraph,
        numba_funcify,
        type_conversion_fn=numba_typify,
        fgraph_name=fgraph_name,
        **kwargs,
    )


def deepcopyop(x):
    return copy(x)


@overload(deepcopyop)
def dispatch_deepcopyop(x):
    if isinstance(x, types.Array):
        return lambda x: np.copy(x)

    return lambda x: x


@numba_funcify.register(DeepCopyOp)
def numba_funcify_DeepCopyOp(op, node, **kwargs):
    return deepcopyop


@numba_njit
def makeslice(*x):
    return slice(*x)


@numba_funcify.register(MakeSlice)
def numba_funcify_MakeSlice(op, **kwargs):
    return global_numba_func(makeslice)


@numba_njit
def shape(x):
    return np.asarray(np.shape(x))


@numba_funcify.register(Shape)
def numba_funcify_Shape(op, **kwargs):
    return global_numba_func(shape)


@numba_funcify.register(Shape_i)
def numba_funcify_Shape_i(op, **kwargs):
    i = op.i

    @numba_njit
    def shape_i(x):
        return np.asarray(np.shape(x)[i])

    return shape_i


@numba.extending.intrinsic
def direct_cast(typingctx, val, typ):
    if isinstance(typ, numba.types.TypeRef):
        casted = typ.instance_type
    elif isinstance(typ, numba.types.DTypeSpec):
        casted = typ.dtype
    else:
        casted = typ

    sig = casted(casted, typ)

    def codegen(context, builder, signature, args):
        val, _ = args
        context.nrt.incref(builder, signature.return_type, val)
        return val

    return sig, codegen


@numba_funcify.register(Reshape)
def numba_funcify_Reshape(op, **kwargs):
    ndim = op.ndim

    if ndim == 0:

        @numba_njit
        def reshape(x, shape):
            return np.asarray(x.item())

    else:

        @numba_njit
        def reshape(x, shape):
            # TODO: Use this until https://github.com/numba/numba/issues/7353 is closed.
            return np.reshape(
                np.ascontiguousarray(np.asarray(x)),
                numba_ndarray.to_fixed_tuple(shape, ndim),
            )

    return reshape


@numba_funcify.register(SpecifyShape)
def numba_funcify_SpecifyShape(op, node, **kwargs):
    shape_inputs = node.inputs[1:]
    shape_input_names = ["shape_" + str(i) for i in range(len(shape_inputs))]

    func_conditions = [
        f"assert x.shape[{i}] == {shape_input_names}"
        for i, (shape_input, shape_input_names) in enumerate(
            zip(shape_inputs, shape_input_names, strict=True)
        )
        if shape_input is not NoneConst
    ]

    func = dedent(
        f"""
        def specify_shape(x, {create_arg_string(shape_input_names)}):
            {"; ".join(func_conditions)}
            return x
        """
    )

    specify_shape = compile_function_src(func, "specify_shape", globals())
    return numba_njit(specify_shape)


def int_to_float_fn(inputs, out_dtype):
    """Create a Numba function that converts integer and boolean ``ndarray``s to floats."""

    if all(
        input.type.numpy_dtype == np.dtype(out_dtype) for input in inputs
    ) and isinstance(np.dtype(out_dtype), np.floating):

        @numba_njit
        def inputs_cast(x):
            return x

    elif any(i.type.numpy_dtype.kind in "ib" for i in inputs):
        args_dtype = np.dtype(f"f{out_dtype.itemsize}")

        @numba_njit
        def inputs_cast(x):
            return x.astype(args_dtype)

    else:
        args_dtype_sz = max(_arg.type.numpy_dtype.itemsize for _arg in inputs)
        args_dtype = np.dtype(f"f{args_dtype_sz}")

        @numba_njit
        def inputs_cast(x):
            return x.astype(args_dtype)

    return inputs_cast


@numba_funcify.register(Dot)
def numba_funcify_Dot(op, node, **kwargs):
    # Numba's `np.dot` does not support integer dtypes, so we need to cast to
    # float.

    out_dtype = node.outputs[0].type.numpy_dtype
    inputs_cast = int_to_float_fn(node.inputs, out_dtype)

    @numba_njit
    def dot(x, y):
        return np.asarray(np.dot(inputs_cast(x), inputs_cast(y))).astype(out_dtype)

    return dot


@numba_funcify.register(Softplus)
def numba_funcify_Softplus(op, node, **kwargs):
    x_dtype = np.dtype(node.inputs[0].dtype)

    @numba_njit
    def softplus(x):
        if x < -37.0:
            value = np.exp(x)
        elif x < 18.0:
            value = np.log1p(np.exp(x))
        elif x < 33.3:
            value = x + np.exp(-x)
        else:
            value = x
        return direct_cast(value, x_dtype)

    return softplus


@numba_funcify.register(Solve)
def numba_funcify_Solve(op, node, **kwargs):
    assume_a = op.assume_a
    # check_finite = op.check_finite

    if assume_a != "gen":
        lower = op.lower

        warnings.warn(
            (
                "Numba will use object mode to allow the "
                "`compute_uv` argument to `numpy.linalg.svd`."
            ),
            UserWarning,
        )

        ret_sig = get_numba_type(node.outputs[0].type)

        @numba_njit
        def solve(a, b):
            with numba.objmode(ret=ret_sig):
                ret = scipy.linalg.solve_triangular(
                    a,
                    b,
                    lower=lower,
                    # check_finite=check_finite
                )
            return ret

    else:
        out_dtype = node.outputs[0].type.numpy_dtype
        inputs_cast = int_to_float_fn(node.inputs, out_dtype)

        @numba_njit
        def solve(a, b):
            return np.linalg.solve(
                inputs_cast(a),
                inputs_cast(b),
                # assume_a=assume_a,
                # check_finite=check_finite,
            ).astype(out_dtype)

    return solve


@numba_funcify.register(BatchedDot)
def numba_funcify_BatchedDot(op, node, **kwargs):
    dtype = node.outputs[0].type.numpy_dtype

    @numba_njit
    def batched_dot(x, y):
        # Numba does not support 3D matmul
        # https://github.com/numba/numba/issues/3804
        shape = x.shape[:-1] + y.shape[2:]
        z0 = np.empty(shape, dtype=dtype)
        for i in range(z0.shape[0]):
            z0[i] = np.dot(x[i], y[i])

        return z0

    return batched_dot


# NOTE: The remaining `pytensor.tensor.blas` `Op`s appear unnecessary, because
# they're only used to optimize basic `Dot` nodes, and those GEMV and GEMM
# optimizations are apparently already performed by Numba


@numba_funcify.register(IfElse)
def numba_funcify_IfElse(op, **kwargs):
    n_outs = op.n_outs

    if n_outs > 1:

        @numba_njit
        def ifelse(cond, *args):
            if cond:
                res = args[:n_outs]
            else:
                res = args[n_outs:]

            return res

    else:

        @numba_njit
        def ifelse(cond, *args):
            if cond:
                res = args[:n_outs]
            else:
                res = args[n_outs:]

            return res[0]

    return ifelse


@numba_funcify.register(Nonzero)
def numba_funcify_Nonzero(op, node, **kwargs):
    @numba_njit
    def nonzero(a):
        result_tuple = np.nonzero(a)
        if a.ndim == 1:
            return result_tuple[0]
        return list(result_tuple)

    return nonzero
