import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.scalar import Softplus
from pytensor.tensor.elemwise import CAReduce, DimShuffle
from pytensor.tensor.special import Softmax, SoftmaxGrad


@mlx_funcify.register(DimShuffle)
def mlx_funcify_DimShuffle(op, **kwargs):
    def dimshuffle(x):
        res = mx.transpose(x, op.transposition)

        shape = list(res.shape[: len(op.shuffle)])

        for augm in op.augment:
            shape.insert(augm, 1)

        return mx.reshape(res, shape)

    return dimshuffle


@mlx_funcify.register(CAReduce)
def mlx_funcify_CAReduce(op, **kwargs):
    axis = op.axis
    op_nfunc_spec = getattr(op, "nfunc_spec", None)
    scalar_nfunc_spec = getattr(op.scalar_op, "nfunc_spec", None)
    scalar_op_name = getattr(op.scalar_op, "name", None)
    scalar_op_identity = getattr(op.scalar_op, "identity", None)
    acc_dtype = getattr(op, "acc_dtype", None)

    def careduce(x):
        nonlocal \
            axis, \
            op_nfunc_spec, \
            scalar_nfunc_spec, \
            scalar_op_name, \
            scalar_op_identity, \
            acc_dtype

        if axis is None:
            axis = list(range(x.ndim))

        if acc_dtype is None:
            acc_dtype = x.dtype

        if op_nfunc_spec:
            mlx_op = getattr(mx, op_nfunc_spec[0])
            return mlx_op(x, axis=axis)
            # return mlx_op(x, axis=axis).astype(acc_dtype)

        # The PyTensor `Op` didn't tell us which NumPy equivalent to use (or
        # there isn't one), so we use this fallback approach
        if scalar_nfunc_spec:
            scalar_fn_name = scalar_nfunc_spec[0]
        elif scalar_op_name:
            scalar_fn_name = scalar_op_name

        to_reduce = sorted(axis, reverse=True)

        if to_reduce:
            raise NotImplementedError("Not implemented yet")
            # In this case, we need to use the `jax.lax` function (if there
            # is one), and not the `jnp` version.
            mlx_op = getattr(mx, scalar_fn_name)
            init_value = mx.array(scalar_op_identity, dtype=acc_dtype)
            return mx.reduce(x, init_value, mlx_op, to_reduce).astype(acc_dtype)
        else:
            return x

    return careduce


@mlx_funcify.register(Softmax)
def mlx_funcify_Softmax(op, **kwargs):
    axis = op.axis

    def softmax(x):
        return mx.softmax(x, axis=axis)

    return softmax


@mlx_funcify.register(SoftmaxGrad)
def mlx_funcify_SoftmaxGrad(op, **kwargs):
    axis = op.axis

    def softmax_grad(dy, sm):
        dy_times_sm = dy * sm
        return dy_times_sm - mx.sum(dy_times_sm, axis=axis, keepdims=True) * sm

    return softmax_grad


@mlx_funcify.register(Softplus)
def mlx_funcify_Softplus(op, **kwargs):
    def softplus(x):
        return mx.where(
            x < -37.0,
            mx.exp(x),
            mx.where(
                x < 18.0,
                mx.log1p(mx.exp(x)),
                mx.where(
                    x < 33.3,
                    x + mx.exp(-x),
                    x,
                ),
            ),
        )

    return softplus
