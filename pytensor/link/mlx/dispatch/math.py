import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.scalar.basic import Add, Cos, Exp, Log, Mul, Sin, Sub
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.math import Dot


@mlx_funcify.register(Dot)
def mlx_funcify_Dot(op, **kwargs):
    def dot(x, y):
        return mx.matmul(x, y)

    return dot


@mlx_funcify.register(Elemwise)
def mlx_funcify_Elemwise(op, **kwargs):
    if isinstance(op.scalar_op, Add):

        def add(x, y):
            return mx.add(x, y)

        return add
    elif isinstance(op.scalar_op, Sub):

        def sub(x, y):
            return mx.subtract(x, y)

        return sub
    elif isinstance(op.scalar_op, Mul):

        def mul(x, y):
            return mx.multiply(x, y)

        return mul
    elif isinstance(op.scalar_op, Exp):

        def exp(x):
            return mx.exp(x)

        return exp
    elif isinstance(op.scalar_op, Log):

        def log(x):
            return mx.log(x)

        return log
    elif isinstance(op.scalar_op, Sin):

        def sin(x):
            return mx.sin(x)

        return sin
    elif isinstance(op.scalar_op, Cos):

        def cos(x):
            return mx.cos(x)

        return cos
    else:
        raise NotImplementedError(f"MLX does not support {op.scalar_op}")
