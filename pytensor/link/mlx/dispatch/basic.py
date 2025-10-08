import warnings
from copy import deepcopy
from functools import singledispatch
from types import NoneType

import mlx.core as mx
import numpy as np

from pytensor.compile.ops import DeepCopyOp
from pytensor.graph.fg import FunctionGraph
from pytensor.link.utils import fgraph_to_python
from pytensor.raise_op import Assert, CheckAndRaise


@singledispatch
def mlx_typify(data, **kwargs):
    raise NotImplementedError(f"mlx_typify is not implemented for {type(data)}")


@mlx_typify.register(np.ndarray)
@mlx_typify.register(mx.array)
def mlx_typify_tensor(data, dtype=None, **kwargs):
    return mx.array(data, dtype=dtype)


@mlx_typify.register(slice)
@mlx_typify.register(NoneType)
@mlx_typify.register(np.number)
def mlx_typify_no_conversion_needed(data, **kwargs):
    return data


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


@mlx_funcify.register(Assert)
@mlx_funcify.register(CheckAndRaise)
def mlx_funcify_CheckAndRaise(op, **kwargs):
    warnings.warn(
        f"""Skipping `CheckAndRaise` Op (assertion: {op.msg}) as MLX tracing would remove it.""",
        stacklevel=2,
    )

    def assert_fn(x, *inputs):
        return x

    return assert_fn
