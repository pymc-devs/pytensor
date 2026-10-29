import importlib

import torch

from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.scalar import ScalarLoop
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import All, Any, Max, Min, Prod, ProdWithoutZeros, Sum
from pytensor.tensor.special import LogSoftmax, Softmax


@pytorch_funcify.register(Elemwise)
def pytorch_funcify_Elemwise(op, node, **kwargs):
    scalar_op = op.scalar_op

    base_fn = pytorch_funcify(scalar_op, node=node, **kwargs)

    def check_special_scipy(func_name):
        if "scipy." not in func_name:
            return False
        loc = func_name.split(".")[1:]
        try:
            mod = importlib.import_module(".".join(loc[:-1]), "torch")
            return getattr(mod, loc[-1], False)
        except ImportError:
            return False

    if hasattr(scalar_op, "nfunc_spec") and (
        hasattr(torch, scalar_op.nfunc_spec[0])
        or check_special_scipy(scalar_op.nfunc_spec[0])
    ):
        # torch can handle this scalar
        # broadcast, we'll let it.
        def elemwise_fn(*inputs):
            Elemwise._check_runtime_broadcast(node, inputs)
            return base_fn(*inputs)

    elif isinstance(scalar_op, ScalarLoop):
        return elemwise_ravel_fn(base_fn, op, node, **kwargs)

    else:

        def elemwise_fn(*inputs):
            Elemwise._check_runtime_broadcast(node, inputs)
            broadcast_inputs = torch.broadcast_tensors(*inputs)
            ufunc = base_fn
            for _ in range(broadcast_inputs[0].dim()):
                ufunc = torch.vmap(ufunc)
            return ufunc(*broadcast_inputs)

    return elemwise_fn


@pytorch_funcify.register(DimShuffle)
def pytorch_funcify_DimShuffle(op, **kwargs):
    def dimshuffle(x):
        res = torch.permute(x, op._transposition)

        shape = list(res.shape[: len(op.shuffle)])

        for augm in op.augment:
            shape.insert(augm, 1)

        return torch.reshape(res, shape)

    return dimshuffle


@pytorch_funcify.register(Sum)
def pytorch_funcify_sum(op, **kwargs):
    def torch_sum(x):
        return torch.sum(x, dim=op.axis)

    return torch_sum


@pytorch_funcify.register(All)
def pytorch_funcify_all(op, **kwargs):
    def torch_all(x):
        return torch.all(x, dim=op.axis)

    return torch_all


@pytorch_funcify.register(Prod)
def pytorch_funcify_prod(op, node, **kwargs):
    acc_dtype = getattr(torch, op.acc_dtype)
    out_dtype = getattr(torch, node.outputs[0].dtype)
    # PyTorch has no uint16/32/64 prod kernels; int64 preserves their low bits.
    use_signed_accumulator = op.acc_dtype in {"uint16", "uint32", "uint64"}
    reduction_dtype = torch.int64 if use_signed_accumulator else acc_dtype

    def torch_prod(x):
        if op.axis == ():
            return x.to(dtype=out_dtype)

        if use_signed_accumulator and x.dtype == torch.uint64:
            # Preserve bits so signed multiplication has the same modulo-2**64 result.
            x = x.view(torch.int64)

        # Keep intermediate reductions in the accumulator dtype, then cast once.
        if isinstance(op.axis, tuple):
            for d in sorted(op.axis, reverse=True):
                x = torch.prod(x, dim=d, dtype=reduction_dtype)
        else:
            x = torch.prod(x.flatten(), dim=0, dtype=reduction_dtype)

        if use_signed_accumulator:
            # Restore unsigned values before converting to a floating output dtype.
            x = x.to(dtype=acc_dtype)
        return x.to(dtype=out_dtype)

    return torch_prod


@pytorch_funcify.register(ProdWithoutZeros)
def pytorch_funcify_ProdWithoutZeros(op, **kwargs):
    axis = op.axis

    def prod_without_zeros(x):
        nonzero = x != 0
        prod = torch.where(nonzero, x, 1)
        if axis is None:
            prod = torch.prod(prod.flatten(), dim=0)
        else:
            for d in sorted(axis, reverse=True):
                prod = torch.prod(prod, dim=d)
        # All-zero and empty slices reduce to 0, not 1
        return torch.where(torch.any(nonzero, dim=axis), prod, 0)

    return prod_without_zeros


@pytorch_funcify.register(Any)
def pytorch_funcify_any(op, **kwargs):
    def torch_any(x):
        return torch.any(x, dim=op.axis)

    return torch_any


@pytorch_funcify.register(Max)
def pytorch_funcify_max(op, **kwargs):
    def torch_max(x):
        if isinstance(op.axis, tuple):
            for d in sorted(op.axis, reverse=True):
                x = torch.max(x, dim=d).values
            return x
        else:
            return torch.max(x.flatten(), dim=0).values

    return torch_max


@pytorch_funcify.register(Min)
def pytorch_funcify_min(op, **kwargs):
    def torch_min(x):
        if isinstance(op.axis, tuple):
            for d in sorted(op.axis, reverse=True):
                x = torch.min(x, dim=d).values
            return x
        else:
            return torch.min(x.flatten(), dim=0).values

    return torch_min


def _pytorch_softmax_dispatch(torch_fn, axis):
    if axis is None:

        def fn(x):
            return torch_fn(x.ravel(), dim=0).reshape(x.shape)

    elif len(axis) == 1:

        def fn(x):
            return torch_fn(x, dim=axis[0])

    else:

        def fn(x):
            orig_shape = x.shape
            x = torch.movedim(x, axis, tuple(range(-len(axis), 0)))
            unflatten_shape = x.shape[: -len(axis)]
            x = x.reshape(*unflatten_shape, -1)
            x = torch_fn(x, dim=-1)
            x = x.reshape(*unflatten_shape, *[orig_shape[a] for a in axis])
            x = torch.movedim(x, tuple(range(-len(axis), 0)), axis)
            return x

    return fn


@pytorch_funcify.register(Softmax)
def pytorch_funcify_Softmax(op, **kwargs):
    dtype = kwargs["node"].inputs[0].dtype

    if not dtype.startswith("float"):
        raise NotImplementedError(
            "Pytorch Softmax is not currently implemented for non-float types."
        )

    return _pytorch_softmax_dispatch(torch.softmax, op.axis)


@pytorch_funcify.register(LogSoftmax)
def pytorch_funcify_LogSoftmax(op, **kwargs):
    dtype = kwargs["node"].inputs[0].dtype

    if not dtype.startswith("float"):
        raise NotImplementedError(
            "Pytorch LogSoftmax is not currently implemented for non-float types."
        )

    return _pytorch_softmax_dispatch(torch.log_softmax, op.axis)


def elemwise_ravel_fn(base_fn, op, node, **kwargs):
    """
    Dispatch methods using `.item()` (ScalarLoop + Elemwise) is common, but vmap
    in torch has a limitation: https://github.com/pymc-devs/pytensor/issues/1031,
    Instead, we can ravel all the inputs, broadcasted according to torch
    """

    n_outputs = len(node.outputs)

    def elemwise_fn(*inputs):
        bcasted_inputs = torch.broadcast_tensors(*inputs)
        raveled_inputs = [inp.ravel() for inp in bcasted_inputs]

        out_shape = bcasted_inputs[0].size()
        out_size = out_shape.numel()
        raveled_outputs = [torch.empty(out_size) for out in node.outputs]

        for i in range(out_size):
            core_outs = base_fn(*(inp[i] for inp in raveled_inputs))
            if n_outputs == 1:
                raveled_outputs[0][i] = core_outs
            else:
                for o in range(n_outputs):
                    raveled_outputs[o][i] = core_outs[o]

        outputs = tuple(out.view(out_shape) for out in raveled_outputs)
        if n_outputs == 1:
            return outputs[0]
        else:
            return outputs

    return elemwise_fn
