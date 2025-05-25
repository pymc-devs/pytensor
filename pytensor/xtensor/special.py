from pytensor.xtensor.math import exp
from pytensor.xtensor.reduction import REDUCE_DIM


def softmax(x, dim: REDUCE_DIM = None):
    exp_x = exp(x)
    return exp_x / exp_x.sum(dim=dim)
