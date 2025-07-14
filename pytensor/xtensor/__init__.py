import warnings

import pytensor.xtensor.rewriting
from pytensor.xtensor import linalg, math, random
from pytensor.xtensor.basic import ones, xtensor_from_tensor, zeros
from pytensor.xtensor.math import dot
from pytensor.xtensor.shape import broadcast, concat, full_like, ones_like, zeros_like
from pytensor.xtensor.type import (
    as_xtensor,
    dim,
    xtensor,
    xtensor_constant,
)


warnings.warn("xtensor module is experimental and full of bugs")
