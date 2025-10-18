import warnings
from functools import singledispatch

import numba
import numpy as np
from numba.core.errors import NumbaWarning
from numba.cpython.unsafe.tuple import tuple_setitem  # noqa: F401

from pytensor import In, config
from pytensor.compile import NUMBA
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.function.types import add_supervisor_to_fgraph
from pytensor.compile.ops import DeepCopyOp, TypeCastingOp
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.type import Type
from pytensor.ifelse import IfElse
from pytensor.link.numba.dispatch.sparse import CSCMatrixType, CSRMatrixType
from pytensor.link.utils import (
    fgraph_to_python,
)
from pytensor.scalar.basic import ScalarType
from pytensor.sparse import SparseTensorType
from pytensor.tensor.type import TensorType


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
            '"(numba_funcified_fgraph|store_core_outputs|cholesky|solve|solve_triangular|cho_solve|lu_factor)" '
            "as it uses dynamic globals"
        ),
        category=NumbaWarning,
    )

    if len(args) > 0 and callable(args[0]):
        return numba.njit(*args[1:], fastmath=fastmath, **kwargs)(args[0])

    return numba.njit(*args, fastmath=fastmath, **kwargs)


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


def int_to_float_fn(inputs, out_dtype):
    """Create a Numba function that converts integer and boolean ``ndarray``s to floats."""

    if (
        all(inp.type.dtype == out_dtype for inp in inputs)
        and np.dtype(out_dtype).kind == "f"
    ):

        @numba_njit(inline="always")
        def inputs_cast(x):
            return x

    elif any(i.type.numpy_dtype.kind in "uib" for i in inputs):
        args_dtype = np.dtype(f"f{out_dtype.itemsize}")

        @numba_njit(inline="always")
        def inputs_cast(x):
            return x.astype(args_dtype)

    else:
        args_dtype_sz = max(_arg.type.numpy_dtype.itemsize for _arg in inputs)
        args_dtype = np.dtype(f"f{args_dtype_sz}")

        @numba_njit(inline="always")
        def inputs_cast(x):
            return x.astype(args_dtype)

    return inputs_cast


@singledispatch
def numba_typify(data, dtype=None, **kwargs):
    return data


def generate_fallback_impl(op, node, storage_map=None, **kwargs):
    """Create a Numba compatible function from a Pytensor `Op`."""

    warnings.warn(
        f"Numba will use object mode to run {op}'s perform method. "
        f"Set `pytensor.config.compiler_verbose = True` to see more details.",
        UserWarning,
    )

    if config.compiler_verbose:
        node.dprint(depth=5, print_type=True)

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
            # zip strict not specified because we are in a hot loop
            return tuple(
                out_type.filter(out[0])
                for out_type, out in zip(output_types, py_perform(inputs))
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


@numba_funcify.register(TypeCastingOp)
def numba_funcify_type_casting(op, **kwargs):
    @numba_njit
    def identity(x):
        return x

    return identity


@numba_funcify.register(DeepCopyOp)
def numba_funcify_DeepCopyOp(op, node, **kwargs):
    if isinstance(node.inputs[0].type, TensorType):

        @numba_njit
        def deepcopy(x):
            return np.copy(x)

    else:

        @numba_njit
        def deepcopy(x):
            return x

    return deepcopy


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
