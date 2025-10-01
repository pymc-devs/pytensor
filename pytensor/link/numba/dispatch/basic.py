import operator
import sys
import warnings
from collections.abc import Callable
from functools import singledispatch

import numba
import numpy as np
from llvmlite import ir
from numba import types
from numba.core.errors import TypingError
from numba.cpython.unsafe.tuple import tuple_setitem  # noqa: F401
from numba.extending import box

from pytensor import In, config
from pytensor.compile import NUMBA
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.function.types import add_supervisor_to_fgraph
from pytensor.compile.ops import DeepCopyOp
from pytensor.graph.fg import FunctionGraph
from pytensor.ifelse import IfElse
from pytensor.link.numba.cache import (
    cache_node_key,
)
from pytensor.link.numba.compile import (
    compile_and_cache_numba_function_src,
    get_numba_type,
    numba_njit,
)
from pytensor.link.utils import fgraph_to_python
from pytensor.tensor import TensorType
from pytensor.tensor.basic import Nonzero
from pytensor.tensor.blas import BatchedDot
from pytensor.tensor.math import Dot
from pytensor.tensor.sort import ArgSortOp, SortOp
from pytensor.tensor.type_other import MakeSlice


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


def generate_fallback_impl(op, node=None, storage_map=None, **kwargs):
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


def numba_funcify_njit(op, node, **kwargs):
    jitable_func_and_key = numba_funcify(op, node=node, **kwargs)

    match jitable_func_and_key:
        case Callable():
            jitable_func = jitable_func_and_key
            key = cache_node_key(node)
        case (Callable(), str() | int()):
            jitable_func, funcify_key = jitable_func_and_key
            key = cache_node_key(node, funcify_key)
        case (Callable(), None):
            # We were explicitly told by the dispatch not to try and cache this function
            jitable_func, key = jitable_func_and_key
        case _:
            raise TypeError(
                f"numpy_funcify should return a callable or a callable, key pair, got {jitable_func_and_key}"
            )

    if key is not None:
        # To force numba to use our cache, we must compile the function so that any closure
        # becomes a global variable...
        op_name = op.__class__.__name__
        cached_func = compile_and_cache_numba_function_src(
            src=f"def {op_name}(*args): return jitable_func(*args)",
            function_name=op_name,
            global_env=globals() | dict(jitable_func=jitable_func),
            key=key,
        )
        return numba_njit(cached_func, final_function=True, cache=True)
    else:
        return numba_njit(
            lambda *args: jitable_func(*args), final_function=True, cache=False
        )


@numba_funcify.register(FunctionGraph)
def numba_funcify_FunctionGraph(
    fgraph,
    node=None,
    fgraph_name="numba_funcified_fgraph",
    jit_nodes: bool = False,
    **kwargs,
):
    return fgraph_to_python(
        fgraph,
        op_conversion_fn=numba_funcify_njit if jit_nodes else numba_funcify,
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

    # We can't cache this correctly until we can define a key for it
    return opfromgraph, None


@numba_funcify.register(DeepCopyOp)
def numba_funcify_DeepCopyOp(op, node, **kwargs):
    if isinstance(node.inputs[0].type, TensorType):

        @numba_njit
        def deepcopy_fn(x):
            return np.copy(x)

    else:

        @numba_njit
        def deepcopy_fn(x):
            return x

    return deepcopy_fn


@numba_funcify.register(MakeSlice)
def numba_funcify_MakeSlice(op, **kwargs):
    @numba_njit
    def makeslice(*x):
        return slice(*x)

    return makeslice


@numba_funcify.register(SortOp)
def numba_funcify_SortOp(op, node, **kwargs):
    @numba_njit
    def sort_f(a, axis):
        axis = axis.item()

        a_swapped = np.swapaxes(a, axis, -1)
        a_sorted = np.sort(a_swapped)
        a_sorted_swapped = np.swapaxes(a_sorted, -1, axis)

        return a_sorted_swapped

    if op.kind != "quicksort":
        warnings.warn(
            (
                f'Numba function sort doesn\'t support kind="{op.kind}"'
                " switching to `quicksort`."
            ),
            UserWarning,
        )

    return sort_f


@numba_funcify.register(ArgSortOp)
def numba_funcify_ArgSortOp(op, node, **kwargs):
    def argsort_f_kind(kind):
        @numba_njit
        def argort_vec(X, axis):
            axis = axis.item()

            Y = np.swapaxes(X, axis, 0)
            result = np.empty_like(Y, dtype="int64")

            indices = list(np.ndindex(Y.shape[1:]))

            for idx in indices:
                result[(slice(None), *idx)] = np.argsort(
                    Y[(slice(None), *idx)], kind=kind
                )

            result = np.swapaxes(result, 0, axis)

            return result

        return argort_vec

    kind = op.kind

    if kind not in ["quicksort", "mergesort"]:
        kind = "quicksort"
        warnings.warn(
            (
                f'Numba function argsort doesn\'t support kind="{op.kind}"'
                " switching to `quicksort`."
            ),
            UserWarning,
        )

    return argsort_f_kind(kind)


@numba_funcify.register(Dot)
def numba_funcify_Dot(op, node, **kwargs):
    # Numba's `np.dot` does not support integer dtypes, so we need to cast to float.
    x, y = node.inputs
    [out] = node.outputs

    x_dtype = x.type.dtype
    y_dtype = y.type.dtype
    dot_dtype = f"float{max((32, out.type.numpy_dtype.itemsize * 8))}"
    out_dtype = out.type.dtype

    if x_dtype == dot_dtype and y_dtype == dot_dtype:

        @numba_njit
        def dot(x, y):
            return np.asarray(np.dot(x, y))

    elif x_dtype == dot_dtype and y_dtype != dot_dtype:

        @numba_njit
        def dot(x, y):
            return np.asarray(np.dot(x, y.astype(dot_dtype)))

    elif x_dtype != dot_dtype and y_dtype == dot_dtype:

        @numba_njit
        def dot(x, y):
            return np.asarray(np.dot(x.astype(dot_dtype), y))

    else:

        @numba_njit()
        def dot(x, y):
            return np.asarray(np.dot(x.astype(dot_dtype), y.astype(dot_dtype)))

    if out_dtype == dot_dtype:
        return dot

    else:

        @numba_njit
        def dot_with_cast(x, y):
            return dot(x, y).astype(out_dtype)

    return dot_with_cast


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
