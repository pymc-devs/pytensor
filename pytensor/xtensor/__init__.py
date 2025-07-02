import warnings

import pytensor.xtensor.rewriting
from pytensor.xtensor import linalg, random
from pytensor.xtensor.math import dot
from pytensor.xtensor.shape import concat
from pytensor.xtensor.type import (
    as_xtensor,
    dim,
    xtensor,
    xtensor_constant,
)


warnings.warn("xtensor module is experimental and full of bugs")
