import numpy as np

from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.link.numba.dispatch.string_codegen import (
    CODE_TOKEN,
    build_source_code,
)
from pytensor.tensor.linalg.constructors import BlockDiagonal


@register_funcify_default_op_cache_key(BlockDiagonal)
def numba_funcify_BlockDiagonal(op, node, **kwargs):
    """

    Because we have variadic arguments we need to use codegen.

    The generated code looks something like:

    def block_diagonal(arr0, arr1, arr2):
        out_r = arr0.shape[0] + arr1.shape[0] + arr2.shape[0]
        out_c = arr0.shape[1] + arr1.shape[1] + arr2.shape[1]
        out = np.zeros((out_r, out_c), dtype=np.float64)

        r, c = 0, 0
        rr, cc = arr0.shape
        out[r: r + rr, c: c + cc] = arr0
        r += rr
        c += cc

        rr, cc = arr1.shape
        out[r: r + rr, c: c + cc] = arr1
        r += rr
        c += cc

        rr, cc = arr2.shape
        out[r: r + rr, c: c + cc] = arr2
        r += rr
        c += cc

        return out
    """
    dtype = node.outputs[0].dtype
    n_inp = len(node.inputs)

    arg_names = [f"arr{i}" for i in range(n_inp)]
    code = [
        f"def block_diagonal({', '.join(arg_names)}):",
        CODE_TOKEN.INDENT,
        f"out_r = {' + '.join(f'{a}.shape[0]' for a in arg_names)}",
        f"out_c = {' + '.join(f'{a}.shape[1]' for a in arg_names)}",
        f"out = np.zeros((out_r, out_c), dtype=np.{dtype})",
        CODE_TOKEN.EMPTY_LINE,
        "r, c = 0, 0",
    ]
    for i, arg_name in enumerate(arg_names):
        code.extend(
            [
                f"rr, cc = {arg_name}.shape",
                f"out[r: r + rr, c: c + cc] = {arg_name}",
                "r += rr",
                "c += cc",
                CODE_TOKEN.EMPTY_LINE,
            ]
        )
    code.append("return out")

    code_txt = build_source_code(code)
    block_diag = compile_numba_function_src(
        code_txt,
        "block_diagonal",
        globals() | {"np": np},
    )

    cache_version = 1
    return numba_basic.numba_njit(block_diag), cache_version
