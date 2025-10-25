import math
from hashlib import sha256

import numpy as np

from pytensor.graph.basic import Variable
from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    numba_funcify_and_cache_key,
    register_funcify_and_cache_key,
)
from pytensor.link.numba.dispatch.cython_support import wrap_cython_function
from pytensor.link.utils import (
    get_name_for_object,
    unique_name_generator,
)
from pytensor.scalar.basic import (
    Add,
    Cast,
    Clip,
    Composite,
    Identity,
    Mul,
    Reciprocal,
    ScalarOp,
    Second,
    Switch,
)
from pytensor.scalar.math import Erf, Erfc, GammaLn, Log1mexp, Sigmoid, Softplus


def scalar_op_cache_key(op):
    # Scalar Ops don't have _props, because of their weird outputs_types_preference function
    # So we create hash differently
    return sha256(str(type(op)).encode()).hexdigest()


@register_funcify_and_cache_key(ScalarOp)
def numba_funcify_ScalarOp(op, node, **kwargs):
    if not hasattr(op, "nfunc_spec"):
        return generate_fallback_impl(op, node=node, **kwargs), None

    scalar_func_path = op.nfunc_spec[0]
    scalar_func_numba = None

    *module_path, scalar_func_name = scalar_func_path.split(".")
    if not module_path:
        # Assume it is numpy, and numba has an implementation
        scalar_func_numba = getattr(np, scalar_func_name)

    input_dtypes = [np.dtype(input.type.dtype) for input in node.inputs]
    output_dtypes = [np.dtype(output.type.dtype) for output in node.outputs]

    if len(output_dtypes) != 1:
        raise ValueError("ScalarOps with more than one output are not supported")

    output_dtype = output_dtypes[0]

    input_inner_dtypes = None
    output_inner_dtype = None

    # Cython functions might have an additional argument
    cython_func = None
    has_pyx_skip_dispatch = False

    if scalar_func_path.startswith("scipy.special"):
        import scipy.special.cython_special

        cython_func = getattr(scipy.special.cython_special, scalar_func_name, None)
        if cython_func is not None:
            scalar_func_numba = wrap_cython_function(
                cython_func, output_dtype, input_dtypes
            )
            has_pyx_skip_dispatch = scalar_func_numba.has_pyx_skip_dispatch
            input_inner_dtypes = scalar_func_numba.numpy_arg_dtypes()
            output_inner_dtype = scalar_func_numba.numpy_output_dtype()

    if scalar_func_numba is None:
        scalar_func_numba = generate_fallback_impl(op, node, **kwargs)

    scalar_op_fn_name = get_name_for_object(scalar_func_numba)

    global_env = {"scalar_func_numba": scalar_func_numba}

    if input_inner_dtypes is None and output_inner_dtype is None:
        unique_names = unique_name_generator(
            [scalar_op_fn_name, "scalar_func_numba"], suffix_sep="_"
        )
        input_names = ", ".join(unique_names(v, force_unique=True) for v in node.inputs)
        if not has_pyx_skip_dispatch:
            scalar_op_src = f"""
def {scalar_op_fn_name}({input_names}):
    return scalar_func_numba({input_names})
            """
        else:
            scalar_op_src = f"""
def {scalar_op_fn_name}({input_names}):
    return scalar_func_numba({input_names}, np.intc(1))
            """

    else:
        global_env["direct_cast"] = numba_basic.direct_cast
        global_env["output_dtype"] = np.dtype(output_inner_dtype)
        input_tmp_dtype_names = {
            f"inp_tmp_dtype_{i}": i_dtype
            for i, i_dtype in enumerate(input_inner_dtypes)
        }
        global_env.update(input_tmp_dtype_names)

        unique_names = unique_name_generator(
            [scalar_op_fn_name, "scalar_func_numba", *global_env.keys()],
            suffix_sep="_",
        )

        input_names = [unique_names(v, force_unique=True) for v in node.inputs]
        converted_call_args = ", ".join(
            f"direct_cast({i_name}, {i_tmp_dtype_name})"
            for i_name, i_tmp_dtype_name in zip(
                input_names, input_tmp_dtype_names, strict=False
            )
        )
        if not has_pyx_skip_dispatch:
            scalar_op_src = f"""
def {scalar_op_fn_name}({", ".join(input_names)}):
    return direct_cast(scalar_func_numba({converted_call_args}), output_dtype)
            """
        else:
            scalar_op_src = f"""
def {scalar_op_fn_name}({", ".join(input_names)}):
    return direct_cast(scalar_func_numba({converted_call_args}, np.intc(1)), output_dtype)
            """

    scalar_op_fn = compile_numba_function_src(
        scalar_op_src,
        scalar_op_fn_name,
        {**globals(), **global_env},
    )

    # Functions that call a function pointer can't be cached
    cache_key = None if cython_func else scalar_op_cache_key(op)
    return numba_basic.numba_njit(scalar_op_fn), cache_key


@register_funcify_and_cache_key(Switch)
def numba_funcify_Switch(op, node, **kwargs):
    @numba_basic.numba_njit
    def switch(condition, x, y):
        if condition:
            return x
        else:
            return y

    return switch, scalar_op_cache_key(op)


def binary_to_nary_func(inputs: list[Variable], binary_op_name: str, binary_op: str):
    """Create a Numba-compatible N-ary function from a binary function."""
    unique_names = unique_name_generator(["binary_op_name"], suffix_sep="_")
    input_names = [unique_names(v, force_unique=True) for v in inputs]
    input_signature = ", ".join(input_names)
    output_expr = binary_op.join(input_names)

    nary_src = f"""
def {binary_op_name}({input_signature}):
    return {output_expr}
    """
    nary_fn = compile_numba_function_src(nary_src, binary_op_name, globals())

    return nary_fn


@register_funcify_and_cache_key(Add)
def numba_funcify_Add(op, node, **kwargs):
    nary_add_fn = binary_to_nary_func(node.inputs, "add", "+")

    return numba_basic.numba_njit(nary_add_fn), scalar_op_cache_key(op)


@register_funcify_and_cache_key(Mul)
def numba_funcify_Mul(op, node, **kwargs):
    nary_mul_fn = binary_to_nary_func(node.inputs, "mul", "*")

    return numba_basic.numba_njit(nary_mul_fn), scalar_op_cache_key(op)


@register_funcify_and_cache_key(Cast)
def numba_funcify_Cast(op, node, **kwargs):
    dtype = np.dtype(op.o_type.dtype)

    @numba_basic.numba_njit
    def cast(x):
        return numba_basic.direct_cast(x, dtype)

    return cast, sha256(str((type(op), op.o_type.dtype)).encode()).hexdigest()


@register_funcify_and_cache_key(Identity)
def numba_funcify_type_casting(op, **kwargs):
    @numba_basic.numba_njit
    def identity(x):
        return x

    return identity, scalar_op_cache_key(op)


@register_funcify_and_cache_key(Clip)
def numba_funcify_Clip(op, **kwargs):
    @numba_basic.numba_njit
    def clip(x, min_val, max_val):
        if x < min_val:
            return min_val
        elif x > max_val:
            return max_val
        else:
            return x

    return clip, scalar_op_cache_key(op)


@register_funcify_and_cache_key(Composite)
def numba_funcify_Composite(op, node, **kwargs):
    _ = kwargs.pop("storage_map", None)

    composite_fn, fgraph_key = numba_funcify_and_cache_key(
        op.fgraph, squeeze_output=True, **kwargs
    )
    if fgraph_key is None:
        composite_key = None
    else:
        composite_key = sha256(str((type(op), fgraph_key)).encode()).hexdigest()
    return composite_fn, composite_key


@register_funcify_and_cache_key(Second)
def numba_funcify_Second(op, node, **kwargs):
    @numba_basic.numba_njit
    def second(x, y):
        return y

    return second, scalar_op_cache_key(op)


@register_funcify_and_cache_key(Reciprocal)
def numba_funcify_Reciprocal(op, node, **kwargs):
    @numba_basic.numba_njit
    def reciprocal(x):
        # TODO FIXME: This isn't really the behavior or `numpy.reciprocal` when
        # `x` is an `int`
        return 1 / x

    return reciprocal, scalar_op_cache_key(op)


@register_funcify_and_cache_key(Sigmoid)
def numba_funcify_Sigmoid(op, node, **kwargs):
    @numba_basic.numba_njit
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    return sigmoid, scalar_op_cache_key(op)


@register_funcify_and_cache_key(GammaLn)
def numba_funcify_GammaLn(op, node, **kwargs):
    @numba_basic.numba_njit
    def gammaln(x):
        return math.lgamma(x)

    return gammaln, scalar_op_cache_key(op)


@register_funcify_and_cache_key(Log1mexp)
def numba_funcify_Log1mexp(op, node, **kwargs):
    @numba_basic.numba_njit
    def logp1mexp(x):
        if x < np.log(0.5):
            return np.log1p(-np.exp(x))
        else:
            return np.log(-np.expm1(x))

    return logp1mexp, scalar_op_cache_key(op)


@register_funcify_and_cache_key(Erf)
def numba_funcify_Erf(op, **kwargs):
    @numba_basic.numba_njit
    def erf(x):
        return math.erf(x)

    return erf, scalar_op_cache_key(op)


@register_funcify_and_cache_key(Erfc)
def numba_funcify_Erfc(op, **kwargs):
    @numba_basic.numba_njit
    def erfc(x):
        return math.erfc(x)

    return erfc, scalar_op_cache_key(op)


@register_funcify_and_cache_key(Softplus)
def numba_funcify_Softplus(op, node, **kwargs):
    out_dtype = np.dtype(node.outputs[0].type.dtype)

    @numba_basic.numba_njit
    def softplus(x):
        if x < -37.0:
            value = np.exp(x)
        elif x < 18.0:
            value = np.log1p(np.exp(x))
        elif x < 33.3:
            value = x + np.exp(-x)
        else:
            value = x
        return numba_basic.direct_cast(value, out_dtype)

    return softplus, scalar_op_cache_key(op)
