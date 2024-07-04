import torch

from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import All, Any, Max, Min, Prod, Sum
from pytensor.tensor.special import LogSoftmax, Softmax, SoftmaxGrad


@pytorch_funcify.register(Elemwise)
def pytorch_funcify_Elemwise(op, node, **kwargs):
    scalar_op = op.scalar_op
    base_fn = pytorch_funcify(scalar_op, node=node, **kwargs)

    def elemwise_fn(*inputs):
        Elemwise._check_runtime_broadcast(node, inputs)
        return base_fn(*inputs)

    return elemwise_fn


@pytorch_funcify.register(DimShuffle)
def pytorch_funcify_DimShuffle(op, **kwargs):
    def dimshuffle(x):
        res = torch.permute(x, op.transposition)

        shape = list(res.shape[: len(op.shuffle)])

        for augm in op.augment:
            shape.insert(augm, 1)

        res = torch.reshape(res, shape)

        if not op.inplace:
            res = res.clone()

        return res

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
def pytorch_funcify_prod(op, **kwargs):
    def torch_prod(x):
        if isinstance(op.axis, tuple):
            for d in sorted(op.axis, reverse=True):
                x = torch.prod(x, dim=d)
            return x
        else:
            return torch.prod(x.flatten(), dim=0)

    return torch_prod


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


@pytorch_funcify.register(Softmax)
def pytorch_funcify_Softmax(op, **kwargs):
    axis = op.axis
    dtype = kwargs["node"].inputs[0].dtype

    if not dtype.startswith("float"):
        raise NotImplementedError(
            "Pytorch Softmax is not currently implemented for non-float types."
        )

    def softmax(x):
        if axis is not None:
            return torch.softmax(x, dim=axis)
        else:
            return torch.softmax(x.ravel(), dim=0).reshape(x.shape)

    return softmax


@pytorch_funcify.register(LogSoftmax)
def pytorch_funcify_LogSoftmax(op, **kwargs):
    axis = op.axis
    dtype = kwargs["node"].inputs[0].dtype

    if not dtype.startswith("float"):
        raise NotImplementedError(
            "Pytorch LogSoftmax is not currently implemented for non-float types."
        )

    def log_softmax(x):
        if axis is not None:
            return torch.log_softmax(x, dim=axis)
        else:
            return torch.log_softmax(x.ravel(), dim=0).reshape(x.shape)

    return log_softmax


@pytorch_funcify.register(SoftmaxGrad)
def jax_funcify_SoftmaxGrad(op, **kwargs):
    axis = op.axis

    def softmax_grad(dy, sm):
        dy_times_sm = dy * sm
        return dy_times_sm - torch.sum(dy_times_sm, dim=axis, keepdim=True) * sm

    return softmax_grad
