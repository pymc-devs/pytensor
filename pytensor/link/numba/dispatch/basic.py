import warnings
from collections.abc import Callable
from functools import singledispatch, wraps
from hashlib import sha256
from pickle import dumps

import numba
import numpy as np
from numba import njit as _njit
from numba.core.extending import register_jitable
from numba.cpython.unsafe.tuple import tuple_setitem  # noqa: F401

from pytensor import config
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.type import Type
from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch.sparse import CSCMatrixType, CSRMatrixType
from pytensor.link.utils import (
    fgraph_to_python,
)
from pytensor.scalar.basic import ScalarType
from pytensor.sparse import SparseTensorType
from pytensor.tensor.type import TensorType
from pytensor.tensor.utils import hash_from_ndarray


def numba_njit(
    *args, fastmath=None, final_function: bool = False, **kwargs
) -> Callable:
    """A thin wrapper around `numba.njit` or `numba.extending.register_jitable`.

    It dispatches to `numba.njit` if `final_function` is `True`, otherwise to `numba.extending.register_jitable`.
    This function also sets opinionated defaults for the `fastmath` argument based on the
    `pytensor.config.numba__fastmath` configuration variable.

    Only final_functions should be called directly from Python.
    """

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

    if not final_function:
        if kwargs.get("inline") == "always":
            raise ValueError(
                "Cannot inline non-final functions, as they are just `register_overload`ed not `jit`ed"
            )
        # These slow down compilation and are not necessary for functions not called directly from Python
        kwargs.setdefault("no_cpython_wrapper", True)
        kwargs.setdefault("no_cfunc_wrapper", True)

    # register_jitable promises numba a function can be jitted without actually jitting it.
    # This also seems necessary to avoid cache invalidation on numba's part.
    func = _njit if final_function else register_jitable
    if len(args) > 0 and callable(args[0]):
        return func(*args[1:], fastmath=fastmath, **kwargs)(args[0])  # type: ignore
    else:
        return func(*args, fastmath=fastmath, **kwargs)  # type: ignore


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
    warnings.warn(
        "create_numba_signature is deprecated and will be removed in a future release",
        FutureWarning,
    )
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
    warnings.warn(
        "create_tuple_creator is deprecated and will be removed in a future release",
        FutureWarning,
    )

    assert n > 0

    f = numba_njit(f)

    @numba_njit(final_function=True)
    def creator(args):
        return (f(0, *args),)

    for i in range(1, n):

        @numba_njit(final_function=True)
        def creator(args, creator=creator, i=i):
            return (*creator(args), f(i, *args))

    return numba_njit(lambda *args: creator(args), final_function=True)


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

        @numba_njit
        def inputs_cast(x):
            return x

    elif any(i.type.numpy_dtype.kind in "uib" for i in inputs):
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
    single_out = n_outputs == 1

    if single_out:
        ret_sig = get_numba_type(node.outputs[0].type)
    else:
        ret_sig = numba.types.Tuple([get_numba_type(o.type) for o in node.outputs])

    def py_perform(inputs):
        output_storage = [[None] for _i in range(n_outputs)]
        op.perform(node, inputs, output_storage)
        outputs = tuple(o[0] for o in output_storage)
        return outputs[0] if single_out else outputs

    @numba_njit
    def perform(*inputs):
        with numba.objmode(ret=ret_sig):
            ret = py_perform(inputs)
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


@singledispatch
def numba_funcify_default_op_cache_key(
    op, node=None, **kwargs
) -> Callable | tuple[Callable, int]:
    """Funcify an Op and allow a default cache key to be generated for it.

    Wrapped function can return an integer in addition to the generated numba function.

    See docstrings of `register_funcify_default_op_cache_key` for details.
    """
    raise NotImplementedError()


def register_funcify_default_op_cache_key(op_type):
    """Funcify an Op and allow a default cache key to be generated for it.

    This function is a helper that dispatches to both `numba_funcify_default_op_cache_key`
    and the legacy `numba_funcify`.

    The cache key will ultimately be generated by the base case of `numba_funcify_and_cache_key`
    when a more specialized dispatch for the Op is not registered. Function wrapped by this decorator
    can return an integer in addition to the numba function.
    This will be added to the default cache key, and can be used to signal changes over versions.

    The default cache key is based on the string representations of: `type(op)` and the
    bytes of the props serialized by pickle.
    It does not take into account the input types or any other graph context.
    Note that numba will use the input array dtypes, rank and layout as part of its own cache key,
    but not the static shape, broadcastable pattern or constant values.

    If the funcify implementation exploits information that is not unique to either the Op class
    or it's `_props` as described above, or the information numba uses, then this method should not be used.
    Instead, use
    which won't use any cache key.
    """

    def decorator(dispatch_func):
        numba_funcify_default_op_cache_key.register(op_type)(dispatch_func)

        # Create a wrapper that can be dispatched to the legacy `numba_funcify
        @wraps(dispatch_func)
        def dispatch_func_wrapper(*args, **kwargs):
            # Discard the potential key salt for the non-cache version
            func_and_int = dispatch_func(*args, **kwargs)
            if isinstance(func_and_int, tuple):
                func, _int = func_and_int
            else:
                func = func_and_int
            return func

        numba_funcify.register(op_type)(dispatch_func_wrapper)

        # Return the original function
        return dispatch_func

    return decorator


@singledispatch
def numba_funcify_and_cache_key(op, node=None, **kwargs) -> tuple[Callable, str | None]:
    """Funcify an Op and return a unique cache key that can be used by numba caching.

    A cache key of `None` can be returned to indicate that a function can't be cached.

    See docstrings of `register_funcify_default_op_cache_key` for details.
    """

    # The base case of this dispatch (if nothing specialized was registered), is to
    # 1. Attempt to use `numba_funcify_default_op_cache_key`,
    #   which indicates a simple cache key based on the Op and its _props can be
    #   safely used to uniquely identify the returned numba function
    # 2. If that fails, attempt to use the legacy `numba_funcify`.
    #   In this case a `None` is returned as the cache_key to indicate the function
    #   cannot be safely cached.

    try:
        func_and_int = numba_funcify_default_op_cache_key(op, node=node, **kwargs)
    except NotImplementedError:
        # Fallback
        return numba_funcify(op, node=node, **kwargs), None

    if isinstance(func_and_int, tuple):
        func, integer = func_and_int
        if isinstance(integer, int):
            integer_str = str(integer)
        else:
            # Input validation
            if integer is None:  # type: ignore[unreachable]
                raise TypeError(
                    "The function wrapped by `numba_funcify_default_op_cache_key` returned None as it's second output, "
                    "but only integers are allowed.\nIf the function cannot be cached, the wrapper shouldn't be used. "
                    "You can use `numba_funcify_and_cache_key` to optionally return None",
                )
            else:
                raise TypeError(
                    f"The function wrapped by numba_funcify_default_op_cache_key returned {integer} of type {type(integer)} "
                    "as it's second output, but only integers are allowed."
                )
    else:
        func, integer_str = func_and_int, "None"

    try:
        props_dict = op._props_dict()
    except AttributeError:
        raise ValueError(
            "The function wrapped by `numba_funcify_default_op_cache_key` can only be used with Ops with `_props`, "
            f"but {op} of type {type(op)} has no _props defined (not even empty)."
        )
    if not props_dict:
        # Simple op, just use the type string as key
        key_bytes = f"({type(op)}, {integer_str})".encode()
    else:
        # Simple props, can use string representation of props as key
        simple_types = (str, bool, int, type(None), float)
        container_types = (tuple, frozenset)
        if all(
            isinstance(v, simple_types)
            or (
                isinstance(v, container_types)
                and all(isinstance(i, simple_types) for i in v)
            )
            for v in props_dict.values()
        ):
            key_bytes = (
                f"({type(op)}, {tuple(props_dict.items())}, {integer_str})".encode()
            )
        else:
            # Complex props, use pickle to serialize them
            key_bytes = dumps(
                (str(type(op)), tuple(props_dict.items()), integer_str),
                protocol=-1,
            )
    return func, sha256(key_bytes).hexdigest()


def register_funcify_and_cache_key(op_type):
    """Funcify an Op and return a unique cache key that can be used by numba caching.

    This function is a helper that dispatches to both `numba_funcify_and_cache_key`
    and the legacy `numba_funcify`.

    Note that numba will use the input array dtypes, rank and layout as part of its own cache key,
    but not the static shape, broadcastable pattern or constant values.

    The cache_key should be unique to indentify the function that was generated by the dispatch
    function among all possible PyTensor Ops and graphs, modulo the information numba already uses.

    A cache key of `None` can be returned to indicate that a function can't be cached.

    For simple cases, it may be possible to use the helper `register_funcify_default_op_cache_key`.
    Be sure to read the limitations in the respective docstrings!
    """

    def decorator(dispatch_func):
        numba_funcify_and_cache_key.register(op_type)(dispatch_func)

        # Create a wrapper for the legacy dispatcher
        @wraps(dispatch_func)
        def dispatch_func_wrapper(*args, **kwargs):
            func, _key = dispatch_func(*args, **kwargs)
            # Discard the key for the non-cache version
            return func

        numba_funcify.register(op_type)(dispatch_func_wrapper)

        return dispatch_func

    return decorator


def numba_funcify_ensure_cache(
    op, *args, final_function: bool = True, **kwargs
) -> tuple[Callable, str | None]:
    """Obtain a numba function for an Op and ensure it can be cached by numba.

    If `config.numba__cache` is `True`, and `numba_funcify_and_cache_key` returns a non-None key,
    the returned function will be wrapped in a python-compiled function that hoists any closures
    to the global scope. This, together with the NumbaPyTensorCacheLocator ensures numba will use our cache.

    Without this strategy, numba would often consider caches to be invalid. This was always the case for:
    1. Ops using the custom vectorize intrinsic: Elemwise, Blockwise, RandomVariables
    2. String generated functions: Alloc, Scan, OpFromGraph, and FunctionGraph itself
    """
    if config.numba__cache:
        jitable_func, cache_key = numba_funcify_and_cache_key(op, *args, **kwargs)
    else:
        jitable_func, cache_key = numba_funcify(op, *args, **kwargs), None

    if cache_key is None:
        if config.numba__cache and config.compiler_verbose:
            print(f"{op} of type {type(op)} will not be cached by PyTensor.\n")  # noqa: T201
        if final_function:
            # Wrap the overloaded function and jit it
            return numba_njit(
                lambda *args: jitable_func(*args),
                final_function=True,
                cache=False,
            ), None
        else:
            # No need to do anything if it's not the final function
            return jitable_func, None
    else:
        op_name = op.__class__.__name__
        cached_func = compile_numba_function_src(
            src=f"def {op_name}(*args): return jitable_func(*args)",
            function_name=op_name,
            global_env=globals() | {"jitable_func": jitable_func},
            cache_key=cache_key,
        )
        return numba_njit(
            cached_func,
            final_function=final_function,
            cache=True,
        ), cache_key


def cache_key_for_constant(data):
    """Create a cache key for a constant value."""
    if isinstance(data, np.number):
        return sha256(data).hexdigest()
    elif isinstance(data, np.ndarray):
        return hash_from_ndarray(data)
    elif data is None:
        return "None"
    elif isinstance(data, int | float | bool):
        # These shouldn't all really be np.number, but we keep this branch just in case
        return str(data)
    else:
        # Fallback for arbitrary types
        return sha256(dumps(data)).hexdigest()


@register_funcify_and_cache_key(FunctionGraph)
def numba_funcify_FunctionGraph(
    fgraph: FunctionGraph,
    node=None,
    fgraph_name="numba_funcified_fgraph",
    **kwargs,
):
    # Collect cache keys of every Op/Constant in the FunctionGraph
    # so we can create a global cache key for the whole FunctionGraph
    cache_keys = []
    toposort = fgraph.toposort()
    clients = fgraph.clients
    toposort_indices = {node: i for i, node in enumerate(toposort)}
    # Add dummy output clients which are not included of the toposort
    toposort_indices |= {
        clients[out][0][0]: i
        for i, out in enumerate(fgraph.outputs, start=len(toposort))
    }

    def op_conversion_and_key_collection(*args, **kwargs):
        # Convert an Op to a funcified function and store the cache_key

        # We also Cache each Op so Numba can do less work next time it sees it
        func, key = numba_funcify_ensure_cache(*args, final_function=False, **kwargs)
        cache_keys.append(key)
        return func

    def type_conversion_and_key_collection(value, variable, **kwargs):
        # Convert a constant type to a numba compatible one and compute a cache key for it

        # We need to know where in the graph the constants are used
        # Otherwise we would hash stack(x, 5.0, 7.0), and stack(5.0, x, 7.0) the same
        client_indices = tuple(
            (toposort_indices[node], inp_idx) for node, inp_idx in clients[variable]
        )
        cache_keys.append((client_indices, cache_key_for_constant(value)))
        return numba_typify(value, variable=variable, **kwargs)

    py_func = fgraph_to_python(
        fgraph,
        op_conversion_fn=op_conversion_and_key_collection,
        type_conversion_fn=type_conversion_and_key_collection,
        fgraph_name=fgraph_name,
        **kwargs,
    )
    if any(key is None for key in cache_keys):
        # If a single element couldn't be cached, we can't cache the whole FunctionGraph either
        fgraph_key = None
    else:
        # Compose individual cache_keys into a global key for the FunctionGraph
        fgraph_key = sha256(
            f"({type(fgraph)}, {tuple(cache_keys)}, {len(fgraph.inputs)}, {len(fgraph.outputs)})".encode()
        ).hexdigest()
    return numba_njit(py_func), fgraph_key
