import warnings
from functools import singledispatch

import torch

from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.graph.fg import FunctionGraph
from pytensor.ifelse import IfElse
from pytensor.link.utils import fgraph_to_python
from pytensor.raise_op import CheckAndRaise


@singledispatch
def pytorch_typify(data, dtype=None, **kwargs):
    r"""Convert instances of PyTensor `Type`\s to PyTorch types."""
    if dtype is None:
        return data
    else:
        return torch.tensor(data, dtype=dtype)


@pytorch_typify.register(torch.Tensor)
def pytorch_typify_tensor(data, dtype=None, **kwargs):
    # if len(data.shape) == 0:
    #     return data.item()
    return torch.tensor(data, dtype=dtype)


@singledispatch
def pytorch_funcify(op, node=None, storage_map=None, **kwargs):
    """Create a PyTorch compatible function from an PyTensor `Op`."""
    raise NotImplementedError(f"No PyTorch conversion for the given `Op`: {op}")


@pytorch_funcify.register(FunctionGraph)
def pytorch_funcify_FunctionGraph(
    fgraph,
    node=None,
    fgraph_name="pytorch_funcified_fgraph",
    **kwargs,
):
    return fgraph_to_python(
        fgraph,
        pytorch_funcify,
        type_conversion_fn=pytorch_typify,
        fgraph_name=fgraph_name,
        **kwargs,
    )


@pytorch_funcify.register(IfElse)
def pytorch_funcify_IfElse(op, **kwargs):
    n_outs = op.n_outs

    def ifelse(cond, *args, n_outs=n_outs):
        res = torch.where(
            cond,
            args[:n_outs][0],
            args[n_outs:][0],
        )
        return res

    return ifelse


@pytorch_funcify.register(CheckAndRaise)
def pytorch_funcify_CheckAndRaise(op, **kwargs):
    def assert_fn(x, *conditions):
        for cond in conditions:
            assert cond.item()
        return x

    return assert_fn


def pytorch_safe_copy(x):
    try:
        res = x.clone()
    except NotImplementedError:
        # warnings.warn(
        #     "`jnp.copy` is not implemented yet. Using the object's `copy` method."
        # )
        if hasattr(x, "copy"):
            res = torch.tensor(x.copy())
        else:
            warnings.warn(f"Object has no `copy` method: {x}")
            res = x

    return res


@pytorch_funcify.register(DeepCopyOp)
def pytorch_funcify_DeepCopyOp(op, **kwargs):
    def deepcopyop(x):
        return pytorch_safe_copy(x)

    return deepcopyop


@pytorch_funcify.register(ViewOp)
def pytorch_funcify_ViewOp(op, **kwargs):
    def viewop(x):
        return x

    return viewop
