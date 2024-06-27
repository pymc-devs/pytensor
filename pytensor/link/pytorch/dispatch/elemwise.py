import torch

from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.tensor.elemwise import DimShuffle, Elemwise
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
