from functools import singledispatch
from types import NoneType

import torch

from pytensor.compile.ops import DeepCopyOp
from pytensor.graph.fg import FunctionGraph
from pytensor.link.utils import fgraph_to_python
from pytensor.raise_op import CheckAndRaise
from pytensor.tensor.basic import Alloc, AllocEmpty, ARange, Eye, Join, MakeVector


@singledispatch
def pytorch_typify(data, dtype=None, **kwargs):
    r"""Convert instances of PyTensor `Type`\s to PyTorch types."""
    return torch.as_tensor(data, dtype=dtype)


@pytorch_typify.register(NoneType)
def pytorch_typify_None(data, **kwargs):
    return None


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


@pytorch_funcify.register(Join)
def pytorch_funcify_Join(op, **kwargs):
    def join(axis, *tensors):
        # tensors could also be tuples, and in this case they don't have a ndim
        tensors = [torch.tensor(tensor) for tensor in tensors]

        return torch.cat(tensors, dim=axis)

    return join


@pytorch_funcify.register(Eye)
def pytorch_funcify_eye(op, **kwargs):
    torch_dtype = getattr(torch, op.dtype)

    def eye(N, M, k):
        major, minor = (M, N) if k > 0 else (N, M)
        k_abs = torch.abs(k)
        zeros = torch.zeros(N, M, dtype=torch_dtype)
        if k_abs < major:
            l_ones = torch.min(major - k_abs, minor)
            return zeros.diagonal_scatter(torch.ones(l_ones, dtype=torch_dtype), k)
        return zeros

    return eye


@pytorch_funcify.register(MakeVector)
def pytorch_funcify_MakeVector(op, **kwargs):
    torch_dtype = getattr(torch, op.dtype)

    def makevector(*x):
        return torch.tensor(x, dtype=torch_dtype)

    return makevector
