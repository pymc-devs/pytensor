from functools import singledispatch
from types import NoneType

import numpy as np
import torch
import torch.compiler

from pytensor.compile import PYTORCH
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.ops import DeepCopyOp
from pytensor.graph.fg import FunctionGraph
from pytensor.ifelse import IfElse
from pytensor.link.utils import fgraph_to_python
from pytensor.raise_op import CheckAndRaise
from pytensor.tensor.basic import (
    Alloc,
    AllocEmpty,
    ARange,
    Eye,
    Join,
    MakeVector,
    Split,
    TensorFromScalar,
)


@singledispatch
def pytorch_typify(data, **kwargs):
    raise NotImplementedError(f"pytorch_typify is not implemented for {type(data)}")


@pytorch_typify.register(np.ndarray)
@pytorch_typify.register(torch.Tensor)
def pytorch_typify_tensor(data, dtype=None, **kwargs):
    return torch.as_tensor(data, dtype=dtype)


@pytorch_typify.register(slice)
@pytorch_typify.register(NoneType)
@pytorch_typify.register(np.number)
def pytorch_typify_no_conversion_needed(data, **kwargs):
    return data


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
    conversion_func=pytorch_funcify,
    **kwargs,
):
    built_kwargs = {"conversion_func": conversion_func, **kwargs}
    return fgraph_to_python(
        fgraph,
        conversion_func,
        type_conversion_fn=pytorch_typify,
        fgraph_name=fgraph_name,
        **built_kwargs,
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


@pytorch_funcify.register(IfElse)
def pytorch_funcify_IfElse(op, **kwargs):
    n_outs = op.n_outs

    def ifelse(cond, *true_and_false, n_outs=n_outs):
        if cond:
            return true_and_false[:n_outs]
        else:
            return true_and_false[n_outs:]

    return ifelse


@pytorch_funcify.register(OpFromGraph)
def pytorch_funcify_OpFromGraph(op, node, **kwargs):
    kwargs.pop("storage_map", None)

    # Apply inner rewrites
    PYTORCH.optimizer(op.fgraph)
    fgraph_fn = pytorch_funcify(op.fgraph, **kwargs, squeeze_output=True)
    return fgraph_fn


@pytorch_funcify.register(TensorFromScalar)
def pytorch_funcify_TensorFromScalar(op, **kwargs):
    def tensorfromscalar(x):
        return torch.as_tensor(x)

    return tensorfromscalar


@pytorch_funcify.register(Split)
def pytorch_funcify_Split(op, node, **kwargs):
    def inner_fn(x, dim, split_amounts):
        return x.split(split_amounts.tolist(), dim=dim.item())

    return inner_fn
