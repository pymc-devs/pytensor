import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.link.mlx.dispatch.core import convert_dtype_to_mlx
from pytensor.scalar.basic import Cast, Composite, Identity, Mod, ScalarOp, Second
from pytensor.scalar.math import Erfc, Erfcx, Sigmoid, Softplus


# MLX name overrides for nfunc_spec names that don't match mlx.core
MLX_NFUNC_OVERRIDES = {
    "true_divide": "divide",
    "invert": "bitwise_invert",
}


@mlx_funcify.register(ScalarOp)
def mlx_funcify_ScalarOp(op, node=None, **kwargs):
    """Generic handler for scalar ops, using nfunc_spec for auto-resolution.

    Most scalar ops have a ``nfunc_spec`` attribute like ``('add', 2, 1)`` that
    names the corresponding numpy function, along with the number of inputs and
    outputs. Since MLX mirrors numpy's API, we resolve most ops via
    ``getattr(mx, name)``.
    """
    nfunc_spec = getattr(op, "nfunc_spec", None)
    if nfunc_spec is None:
        raise NotImplementedError(
            f"No MLX conversion for scalar op {op}. "
            "It has no nfunc_spec and no specific dispatch."
        )

    # MLX doesn't have scipy submodules, everything is in the main mlx.core namespace
    func_name = nfunc_spec[0].removeprefix("scipy.special.")

    func_name = MLX_NFUNC_OVERRIDES.get(func_name, func_name)
    mlx_func = getattr(mx, func_name, None)
    if mlx_func is None:
        raise NotImplementedError(
            f"No MLX conversion for scalar op {op} (mx.{func_name} not found)"
        )

    # Handle variadic ops (e.g. Add with 3+ inputs)
    if node is not None and len(node.inputs) > nfunc_spec[1]:
        variadic_name = getattr(op, "nfunc_variadic", None)
        if variadic_name:
            mlx_variadic_func = getattr(mx, variadic_name, None)
            if mlx_variadic_func:

                def variadic_fn(*args):
                    return mlx_variadic_func(mx.stack(list(args), axis=0), axis=0)

                return variadic_fn

        # Fallback: fold binary op
        def fold_fn(*args):
            result = args[0]
            for arg in args[1:]:
                result = mlx_func(result, arg)
            return result

        return fold_fn

    return mlx_func


@mlx_funcify.register(Cast)
def mlx_funcify_Cast(op, **kwargs):
    # Cast can be called as a tensor-level op (op.scalar_op.o_type)
    # or as a scalar op directly (op.o_type). Handle both.
    scalar_op = getattr(op, "scalar_op", op)

    def cast(x):
        dtype = convert_dtype_to_mlx(scalar_op.o_type.dtype)
        try:
            return x.astype(dtype)
        except ValueError as e:
            if "is not supported on the GPU" in str(e):
                import warnings

                warnings.warn(
                    f"MLX GPU limitation: {e}. Attempting automatic fallback casting.",
                    UserWarning,
                    stacklevel=2,
                )
                fallback_dtype = convert_dtype_to_mlx(
                    scalar_op.o_type.dtype, auto_cast_unsupported=True
                )
                return x.astype(fallback_dtype)
            else:
                raise

    return cast


@mlx_funcify.register(Mod)
def mlx_funcify_Mod(op, **kwargs):
    def mlx_mod(x, y):
        _, res = mx.divmod(x, y)
        return res

    return mlx_mod


@mlx_funcify.register(Identity)
def mlx_funcify_Identity(op, **kwargs):
    def identity(x):
        return x

    return identity


@mlx_funcify.register(Second)
def mlx_funcify_Second(op, **kwargs):
    def second(x, y):
        x = mx.array(x)
        y = mx.array(y)
        _, out = mx.broadcast_arrays(x, y)
        return out

    return second


@mlx_funcify.register(Sigmoid)
def mlx_funcify_Sigmoid(op, **kwargs):
    return mx.sigmoid


@mlx_funcify.register(Erfc)
def mlx_funcify_Erfc(op, **kwargs):
    def erfc(x):
        return 1.0 - mx.erf(x)

    return erfc


@mlx_funcify.register(Erfcx)
def mlx_funcify_Erfcx(op, **kwargs):
    def erfcx(x):
        return mx.exp(x * x) * (1.0 - mx.erf(x))

    return erfcx


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


@mlx_funcify.register(Composite)
def mlx_funcify_Composite(op, node=None, **kwargs):
    mlx_impl = mlx_funcify(op.fgraph)

    if len(op.fgraph.outputs) == 1:

        def composite(*args):
            return mlx_impl(*args)[0]

    else:

        def composite(*args):
            return mlx_impl(*args)

    return composite
