import warnings

import pytensor.xtensor.rewriting
from pytensor.xtensor.spaces import BaseSpace, Dim, DimLike, OrderedSpace
from pytensor.xtensor.type import XTensorType
from pytensor.xtensor.variable import (
    XTensorConstant,
    XTensorVariable,
    as_xtensor,
    as_xtensor_variable,
)


warnings.warn("xtensor module is experimental and full of bugs")
