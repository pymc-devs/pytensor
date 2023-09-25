import torch

from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.special import LogSoftmax, Softmax, SoftmaxGrad


@pytorch_funcify.register(Elemwise)
def pytorch_funcify_Elemwise(op, node, **kwargs):
    scalar_op = op.scalar_op
    base_fn = pytorch_funcify(scalar_op, node=node, **kwargs)

    def elemwise_fn(*inputs):
        # ScalarVariables in PyTorch are passed as int/float.
        # We wrap them in tensors just for the broadcast check
        Elemwise._check_runtime_broadcast(node, tuple(map(torch.tensor, inputs)))
        return base_fn(*inputs)

    return elemwise_fn


@pytorch_funcify.register(CAReduce)
def pytorch_funcify_CAReduce(op, **kwargs):
    axis = op.axis
    op_nfunc_spec = getattr(op, "nfunc_spec", None)
    scalar_nfunc_spec = getattr(op.scalar_op, "nfunc_spec", None)
    scalar_op_name = getattr(op.scalar_op, "name", None)
    scalar_op_identity = getattr(op.scalar_op, "identity", None)
    acc_dtype = getattr(op, "acc_dtype", None)

    def careduce(x):
        nonlocal axis, op_nfunc_spec, scalar_nfunc_spec, scalar_op_name, scalar_op_identity, acc_dtype

        if axis is None:
            axis = list(range(x.ndim))

        if acc_dtype is None:
            acc_dtype = x.dtype.type

        if op_nfunc_spec:
            torch_op = getattr(torch, op_nfunc_spec[0])
            return torch_op(x, axis=axis).type(acc_dtype)

        # The PyTensor `Op` didn't tell us which PyTorch equivalent to use (or
        # there isn't one), so we use this fallback approach
        if scalar_nfunc_spec:
            scalar_fn_name = scalar_nfunc_spec[0]
        elif scalar_op_name:
            scalar_fn_name = scalar_op_name

        to_reduce = sorted(axis, reverse=True)

        if to_reduce:
            # In this case, we need to use the `torch` function (if there
            # is one), and not the `torch` version.
            torch_op = getattr(torch, scalar_fn_name)
            init_value = torch.tensor(scalar_op_identity, dtype=acc_dtype)
            return torch.reduce(x, init_value, torch_op, to_reduce).type(acc_dtype)
        else:
            return x

    return careduce


@pytorch_funcify.register(DimShuffle)
def pytorch_funcify_DimShuffle(op, **kwargs):
    def dimshuffle(x):
        res = torch.transpose(x, op.transposition)

        shape = list(res.shape[: len(op.shuffle)])

        for augm in op.augment:
            shape.insert(augm, 1)

        res = torch.reshape(res, shape)

        if not op.inplace:
            res = torch.clone(res)

        return res

    return dimshuffle


@pytorch_funcify.register(Softmax)
def pytorch_funcify_Softmax(op, **kwargs):
    axis = op.axis

    def softmax(x):
        return torch.nn.functional.softmax(x, dim=axis)

    return softmax


@pytorch_funcify.register(SoftmaxGrad)
def pytorch_funcify_SoftmaxGrad(op, **kwargs):
    axis = op.axis

    def softmax_grad(dy, sm):
        dy_times_sm = dy * sm
        return dy_times_sm - torch.sum(dy_times_sm, dim=axis, keepdim=True) * sm

    return softmax_grad


@pytorch_funcify.register(LogSoftmax)
def pytorch_funcify_LogSoftmax(op, **kwargs):
    axis = op.axis

    def log_softmax(x):
        return torch.nn.functional.log_softmax(x, dim=axis)

    return log_softmax
