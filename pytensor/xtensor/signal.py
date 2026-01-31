from typing import Literal

import numpy as np

from pytensor.scalar import as_scalar
from pytensor.tensor import zeros
from pytensor.tensor.signal.conv import _convolve_1d
from pytensor.xtensor import concat
from pytensor.xtensor.type import as_xtensor
from pytensor.xtensor.vectorization import XBlockwise


def convolve1d(
    in1,
    in2,
    mode: Literal["full", "valid", "same"] = "full",
    *,
    dim: str,
):
    """Convolve two arrays along a dimension.

    Convolve in1 and in2, with the output size determined by the mode argument.

    Parameters
    ----------
    in1 : XTensorVariable
        First input.
    in2 : XTensorVariable
        Second input.
    mode : {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        - 'full': The output is the full discrete linear convolution of the inputs, with shape (..., N+M-1,).
        - 'valid': The output consists only of elements that do not rely on zero-padding, with shape (..., max(N, M) - min(N, M) + 1,).
        - 'same': The output is the same size as in1, centered with respect to the 'full' output.
    dim: str
        The dimension along which to convolve. Must be present in both inputs, and will be present in the output

    Returns
    -------
    out: XTensorVariable
        The discrete linear convolution of in1 with in2.

    """

    if mode == "same":
        # We implement "same" as "valid" with padded `in1`.
        in2_core_size = in2.sizes[dim]
        zeros_left = as_xtensor(zeros(in2_core_size // 2, dtype=in1.dtype), dims=(dim,))
        zeros_right = as_xtensor(
            zeros((in2_core_size - 1) // 2, dtype=in1.dtype), dims=(dim,)
        )
        in1 = concat([zeros_left, in1, zeros_right], dim=dim)
        mode = "valid"

    full_mode = as_scalar(np.bool_(mode == "full"))

    xop = XBlockwise(
        _convolve_1d,
        core_dims=(((dim,), (dim,), ()), ((dim,),)),
        signature=_convolve_1d.gufunc_signature,
        unaligned_core_dims=(dim,),
    )
    return xop(in1, in2, full_mode)
