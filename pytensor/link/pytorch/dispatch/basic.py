from functools import singledispatch

import torch

from pytensor.compile.ops import DeepCopyOp
from pytensor.graph.fg import FunctionGraph
from pytensor.link.utils import fgraph_to_python
from pytensor.raise_op import CheckAndRaise
from pytensor.tensor.basic import Alloc, AllocEmpty, ARange


@singledispatch
def pytorch_typify(data, dtype=None, **kwargs):
    r"""Convert instances of PyTensor `Type`\s to PyTorch types."""
    return torch.as_tensor(data, dtype=dtype)


@singledispatch
def pytorch_funcify(op, node=None, storage_map=None, **kwargs):
    """Create a PyTorch compatible function from an PyTensor `Op`."""
    raise NotImplementedError(
        f"No PyTorch conversion for the given `Op`: {op}.\nCheck out `https://github.com/pymc-devs/pytensor/issues/821` for progress or to request we prioritize this operation"
    )


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


@pytorch_funcify.register(CheckAndRaise)
def pytorch_funcify_CheckAndRaise(op, **kwargs):
    error = op.exc_type
    msg = op.msg

    def assert_fn(x, *conditions):
        for cond in conditions:
            if not cond.item():
                raise error(msg)
        return x

    return assert_fn


@pytorch_funcify.register(DeepCopyOp)
def pytorch_funcify_DeepCopyOp(op, **kwargs):
    def deepcopyop(x):
        return x.clone()

    return deepcopyop


@pytorch_funcify.register(AllocEmpty)
def pytorch_funcify_AllocEmpty(op, **kwargs):
    dtype = getattr(torch, op.dtype)

    def alloc_empty(*shape):
        return torch.empty(shape, dtype=dtype)

    return alloc_empty


@pytorch_funcify.register(Alloc)
def pytorch_funcify_alloc(op, **kwargs):
    def alloc(value, *shape):
        out = torch.empty(shape, dtype=value.dtype)
        out[...] = value  # broadcast value to shape of out
        return out

    return alloc


@pytorch_funcify.register(ARange)
def pytorch_funcify_arange(op, **kwargs):
    dtype = getattr(torch, op.dtype)

    def arange(start, stop, step):
        return torch.arange(start, stop, step, dtype=dtype)

    return arange
