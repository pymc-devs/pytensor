import warnings
from copy import deepcopy
from functools import singledispatch
from types import NoneType

import mlx.core as mx
import numpy as np

from pytensor.compile.builders import OpFromGraph
from pytensor.compile.mode import MLX
from pytensor.compile.ops import DeepCopyOp
from pytensor.graph import Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.ifelse import IfElse
from pytensor.link.utils import fgraph_to_python
from pytensor.raise_op import Assert, CheckAndRaise


def convert_dtype_to_mlx(dtype_str, auto_cast_unsupported=True):
    """Convert PyTensor dtype strings to MLX dtype objects.

    MLX expects dtype objects rather than string literals for type conversion.
    This function maps common dtype strings to their MLX equivalents.

    Parameters
    ----------
    dtype_str : str or MLX dtype
        The dtype to convert
    auto_cast_unsupported : bool
        If True, automatically cast unsupported dtypes to supported ones with warnings

    Returns
    -------
    MLX dtype object
    """
    import warnings

    if isinstance(dtype_str, str):
        if dtype_str == "bool":
            return mx.bool_
        elif dtype_str == "int8":
            return mx.int8
        elif dtype_str == "int16":
            return mx.int16
        elif dtype_str == "int32":
            return mx.int32
        elif dtype_str == "int64":
            return mx.int64
        elif dtype_str == "uint8":
            return mx.uint8
        elif dtype_str == "uint16":
            return mx.uint16
        elif dtype_str == "uint32":
            return mx.uint32
        elif dtype_str == "uint64":
            return mx.uint64
        elif dtype_str == "float16":
            return mx.float16
        elif dtype_str == "float32":
            return mx.float32
        elif dtype_str == "float64":
            if auto_cast_unsupported:
                warnings.warn(
                    "MLX does not support float64 on GPU. Automatically casting to float32. "
                    "This may result in reduced precision. To avoid this warning, "
                    "explicitly use float32 in your code or set floatX='float32' in PyTensor config.",
                    UserWarning,
                    stacklevel=3,
                )
                return mx.float32
            else:
                return mx.float64
        elif dtype_str == "bfloat16":
            return mx.bfloat16
        elif dtype_str == "complex64":
            return mx.complex64
        elif dtype_str == "complex128":
            if auto_cast_unsupported:
                warnings.warn(
                    "MLX does not support complex128. Automatically casting to complex64. "
                    "This may result in reduced precision. To avoid this warning, "
                    "explicitly use complex64 in your code.",
                    UserWarning,
                    stacklevel=3,
                )
                return mx.complex64
            else:
                # Return the original even though it might fail
                # This allows users to opt out of auto-casting if needed
                return mx.complex64  # MLX doesn't have complex128, so fallback
    # Return as is if it's already an MLX dtype or not a recognized string
    return dtype_str


@singledispatch
def mlx_typify(data, **kwargs):
    raise NotImplementedError(f"mlx_typify is not implemented for {type(data)}")


@mlx_typify.register(np.ndarray)
def mlx_typify_tensor(data, dtype=None, **kwargs):
    return mx.array(data, dtype=dtype)


@mlx_typify.register(slice)
@mlx_typify.register(NoneType)
@mlx_typify.register(mx.array)
def mlx_typify_no_conversion_needed(data, **kwargs):
    return data


@mlx_typify.register(int)
@mlx_typify.register(float)
def mlx_typify_python_scalar(data, **kwargs):
    return mx.array(data)


@mlx_typify.register(bool)
@mlx_typify.register(np.bool_)
def mlx_typify_bool(data, **kwargs):
    return bool(data)


@mlx_typify.register(np.integer)
@mlx_typify.register(np.floating)
@mlx_typify.register(np.complexfloating)
def mlx_typify_numpy_scalar(data, **kwargs):
    return mx.array(data)


@singledispatch
def mlx_funcify(op, node=None, storage_map=None, **kwargs):
    """Create a MLX compatible function from an PyTensor `Op`."""
    raise NotImplementedError(
        f"No MLX conversion for the given `Op`: {op}.\nCheck out `https://github.com/pymc-devs/pytensor/issues/1350` for progress or to request we prioritize this operation"
    )


@mlx_funcify.register(FunctionGraph)
def mlx_funcify_FunctionGraph(
    fgraph,
    node=None,
    fgraph_name="mlx_funcified_fgraph",
    conversion_func=mlx_funcify,
    **kwargs,
):
    built_kwargs = {"conversion_func": conversion_func, **kwargs}
    return fgraph_to_python(
        fgraph,
        conversion_func,
        type_conversion_fn=mlx_typify,
        fgraph_name=fgraph_name,
        **built_kwargs,
    )


@mlx_funcify.register(DeepCopyOp)
def mlx_funcify_DeepCopyOp(op, **kwargs):
    def deepcopyop(x):
        return deepcopy(x)

    return deepcopyop


@mlx_funcify.register(IfElse)
def mlx_funcify_IfElse(op, **kwargs):
    n_outs = op.n_outs

    def ifelse(cond, *args, n_outs=n_outs):
        # MLX has no conditional Op, the best we can do is mx.where to select between branches elementwise.
        res = tuple(mx.where(cond, args[i], args[n_outs + i]) for i in range(n_outs))
        return res if n_outs > 1 else res[0]

    return ifelse

@mlx_funcify.register(Assert)
@mlx_funcify.register(CheckAndRaise)
def mlx_funcify_CheckAndRaise(op, node, **kwargs):
    conds = node.inputs[1:]
    if any(isinstance(cond, Constant) and not bool(cond.data) for cond in conds):
        raise op.exc_type(op.msg)

    warnings.warn(
        f"""Skipping `{type(op).__name__}` Op (assertion: {op.msg}) as MLX tracing would remove it.""",
        stacklevel=2,
    )

    def assert_fn(x, *inputs):
        return x

    return assert_fn


@mlx_funcify.register(OpFromGraph)
def mlx_funcify_OpFromGraph(ofg: OpFromGraph, node=None, **kwargs):
    _ = kwargs.pop("storage_map", None)

    MLX.optimizer(ofg.fgraph)
    fgraph_fn = mlx_funcify(ofg.fgraph, squeeze_output=True, **kwargs)

    return fgraph_fn
