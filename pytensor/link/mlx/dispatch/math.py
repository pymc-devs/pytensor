import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.scalar.basic import (
    EQ,
    GE,
    GT,
    LE,
    LT,
    NEQ,
    Add,
    Cos,
    Exp,
    Log,
    Mul,
    Pow,
    Sin,
    Sub,
    Switch,
    TrueDiv,
)
from pytensor.scalar.math import Sigmoid
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

        def add(*args):
            result = args[0]
            for arg in args[1:]:
                result = mx.add(result, arg)
            return result

        return add
    elif isinstance(op.scalar_op, Sub):

        def sub(x, y):
            return mx.subtract(x, y)

        return sub
    elif isinstance(op.scalar_op, Mul):

        def mul(*args):
            result = args[0]
            for arg in args[1:]:
                result = mx.multiply(result, arg)
            return result

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
    elif isinstance(op.scalar_op, Sigmoid):

        def sigmoid(x):
            return mx.sigmoid(x)

        return sigmoid
    elif isinstance(op.scalar_op, LE):

        def le(x, y):
            return mx.less_equal(x, y)

        return le
    elif isinstance(op.scalar_op, LT):

        def lt(x, y):
            return mx.less(x, y)

        return lt
    elif isinstance(op.scalar_op, GE):

        def ge(x, y):
            return mx.greater_equal(x, y)

        return ge
    elif isinstance(op.scalar_op, GT):

        def gt(x, y):
            return mx.greater(x, y)

        return gt
    elif isinstance(op.scalar_op, EQ):

        def eq(x, y):
            return mx.equal(x, y)

        return eq
    elif isinstance(op.scalar_op, NEQ):

        def neq(x, y):
            return mx.not_equal(x, y)

        return neq
    elif isinstance(op.scalar_op, Switch):

        def switch(cond, x, y):
            return mx.where(cond, x, y)

        return switch
    elif isinstance(op.scalar_op, Pow):

        def pow(x, y):
            return mx.power(x, y)

        return pow
    elif isinstance(op.scalar_op, TrueDiv):

        def true_div(x, y):
            return mx.divide(x, y)

        return true_div
    else:
        raise NotImplementedError(f"MLX does not support {op.scalar_op}")
