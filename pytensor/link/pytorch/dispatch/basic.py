import warnings
from functools import singledispatch

import torch
import numpy as np

from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.ifelse import IfElse
from pytensor.link.utils import fgraph_to_python
from pytensor.raise_op import Assert, CheckAndRaise


@singledispatch
def torch_typify(data, dtype=None, **kwargs):
    r"""Convert instances of PyTensor `Type`\s to PyTorch types."""
    if dtype is None:
        return data
    else:
        return torch.tensor(data, dtype=dtype)


@torch_typify.register(np.ndarray)
def torch_typify_ndarray(data, dtype=None, **kwargs):
    if len(data.shape) == 0:
        return data.item()
    return torch.tensor(data, dtype=dtype)


@singledispatch
def torch_funcify(op, node=None, storage_map=None, **kwargs):
    """Create a PyTorch compatible function from an PyTensor `Op`."""
    raise NotImplementedError(f"No PyTorch conversion for the given `Op`: {op}")


@torch_funcify.register(FunctionGraph)
def torch_funcify_FunctionGraph(
    fgraph,
    node=None,
    fgraph_name="torch_funcified_fgraph",
    **kwargs,
):
    return fgraph_to_python(
        fgraph,
        torch_funcify,
        type_conversion_fn=torch_typify,
        fgraph_name=fgraph_name,
        **kwargs,
    )


@torch_funcify.register(IfElse)
def torch_funcify_IfElse(op, **kwargs):
    n_outs = op.n_outs

    def ifelse(cond, *args, n_outs=n_outs):
        res = torch.where(
            cond, lambda _: args[:n_outs], lambda _: args[n_outs:], operand=None
        )
        return res if n_outs > 1 else res[0]

    return ifelse


@torch_funcify.register(Assert)
@torch_funcify.register(CheckAndRaise)
def torch_funcify_CheckAndRaise(op, **kwargs):
    warnings.warn(
        f"""Skipping `CheckAndRaise` Op (assertion: {op.msg}) as PyTorch tracing would remove it.""",
        stacklevel=2,
    )

    def assert_fn(x, *inputs):
        return x

    return assert_fn


def torch_safe_copy(x):
    try:
        res = torch.clone(x)
    except NotImplementedError:
        warnings.warn(
            "`torch.clone` is not implemented yet. Using the object's `copy` method."
        )
        if hasattr(x, "copy"):
            res = torch.tensor(x.copy())
        else:
            warnings.warn(f"Object has no `copy` method: {x}")
            res = x

    return res


@torch_funcify.register(DeepCopyOp)
def torch_funcify_DeepCopyOp(op, **kwargs):
    def deepcopyop(x):
        return torch_safe_copy(x)

    return deepcopyop


@torch_funcify.register(ViewOp)
def torch_funcify_ViewOp(op, **kwargs):
    def viewop(x):
        return x

    return viewop