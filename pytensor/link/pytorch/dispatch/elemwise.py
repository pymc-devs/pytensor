import importlib
from itertools import chain

import torch

from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.scalar import ScalarLoop
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import All, Any, Max, Min, Prod, Sum
from pytensor.tensor.special import LogSoftmax, Softmax, SoftmaxGrad


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
        return elemwise_scalar_loop(base_fn, op, node, **kwargs)

    else:

        def elemwise_fn(*inputs):
            Elemwise._check_runtime_broadcast(node, inputs)
            broadcast_inputs = torch.broadcast_tensors(*inputs)
            ufunc = base_fn
            for _ in range(broadcast_inputs[0].dim()):
                ufunc = torch.vmap(ufunc)
            return ufunc(*broadcast_inputs)
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


def elemwise_scalar_loop(base_fn, op, node, **kwargs):
    """
    ScalarLoop + Elemwise is too common
    to not work, but @1031, vmap won't allow it.
    Instead, we can do the following strategy
    1. `.unbind(dim)` will return a list of tensors
       representing `dim` but "unwrapped". e.x.
       ```
       t = torch.ones(3, 4, 2)
       len(t.unbind(0)) == 3
       t[0].shape == torch.Size[4, 2]
    2. If we successfully apply, the length of the list will grow
       by the next dimension in the tensor if we flatten the previous
       dimension result
       ```
       inputs = [torch.ones(3, 4, 2)]
       level_1 = chain.from_iterable(t.unbind(0) for t in inputs)
       level_2 = chain.from_iterable(t.unbind(0) for t in level_1)
       len(level_2) == 3 * 4
       ```
    3. Eventually we'll reach single dimension tensors. At that point
       we can iterate over each input in an element by element manner
       and call some function

    For scalar loop, we need to broadcast the tensors so all
    the necessary values are repeated, and we "evenly" iterate through everything
    """

    def elemwise_fn(*inputs):
        Elemwise._check_runtime_broadcast(node, inputs)
        shaped_inputs = torch.broadcast_tensors(*inputs)
        expected_size = shaped_inputs[0].numel()
        final_inputs = [s.clone() for s in shaped_inputs]
        for _ in range(shaped_inputs[0].dim() - 1):
            for i, _ in enumerate(shaped_inputs):
                layer = chain.from_iterable([s.unbind(0) for s in final_inputs[i]])
                final_inputs[i] = list(layer)

        # make sure we still have the same number of things
        assert len(final_inputs) == len(shaped_inputs)

        # make sure each group of things are the expected size
        assert all(len(x) == expected_size for x in final_inputs)

        # make sure they are all single elements
        assert all(len(x.shape) == 0 for tensor in final_inputs for x in tensor)
        res = [base_fn(*args) for args in zip(*final_inputs)]

        return [torch.stack(tuple(out[i] for out in res)) for i in range(len(res[0]))]

    return elemwise_fn
