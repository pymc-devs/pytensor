import numpy as np
from numba.core.extending import overload

from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch.string_codegen import create_tuple_string


def _copyto(dst, src):
    """Copy equal-shaped arrays without broadcasting or overlap handling."""
    for idx in np.ndindex(dst.shape):
        dst[idx] = src[idx]


@overload(_copyto, inline="always")
def _copyto_impl(dst, src):
    axes = range(dst.ndim)
    if dst.layout == "F":
        axes = reversed(axes)
    lines = ["def copyto(dst, src):"]
    indent = "    "
    for axis in axes:
        lines.append(f"{indent}for i{axis} in range(dst.shape[{axis}]):")
        indent += "    "
    idx = create_tuple_string([f"i{axis}" for axis in range(dst.ndim)])
    lines.append(f"{indent}dst[{idx}] = src[{idx}]")
    return compile_numba_function_src("\n".join(lines), "copyto")
