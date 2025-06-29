import warnings

import pytensor.xtensor.rewriting
from pytensor.xtensor import linalg, random
from pytensor.xtensor.math import dot
from pytensor.xtensor.shape import broadcast, concat
from pytensor.xtensor.type import (
    as_xtensor,
    xtensor,
    xtensor_constant,
)


warnings.warn("xtensor module is experimental and full of bugs")
