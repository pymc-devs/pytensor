import warnings
from functools import singledispatch

import torch

from pytensor.compile.ops import DeepCopyOp
from pytensor.graph.fg import FunctionGraph
from pytensor.ifelse import IfElse
from pytensor.link.utils import fgraph_to_python
from pytensor.raise_op import CheckAndRaise


@singledispatch
def pytorch_typify(data, dtype=None, **kwargs):
    r"""Convert instances of PyTensor `Type`\s to PyTorch types."""
    if dtype is None:
        return torch.tensor(data)
    else:
        return torch.as_tensor(data, dtype=dtype)


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
    # Cannot use try-except due to: https://github.com/pytorch/pytorch/issues/93720

    if hasattr(x, "clone"):
        res = torch.clone(x)
    else:
        warnings.warn(f"Object has no `clone` method: {x}")
        res = x

    return res


@pytorch_funcify.register(DeepCopyOp)
def pytorch_funcify_DeepCopyOp(op, **kwargs):
    def deepcopyop(x):
        return pytorch_safe_copy(x)

    return deepcopyop
