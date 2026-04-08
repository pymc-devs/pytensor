import numpy as np

from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.tensor.rewriting.join_inplace import WriteJoin, WriteSplit


@register_funcify_default_op_cache_key(WriteSplit)
def numba_funcify_WriteSplit(op, node, **kwargs):
    n_splits = op.n_splits
    axis = op.axis

    slice_lines = []
    offset_expr = "0"
    for i in range(n_splits):
        slice_lines.append(f"    sz_{i} = s{i}.item()")
        idx = ", ".join(
            f"{offset_expr}:{offset_expr} + sz_{i}" if d == axis else ":"
            for d in range(node.inputs[0].type.ndim)
        )
        slice_lines.append(f"    out_{i} = buffer[{idx}]")
        offset_expr = f"{offset_expr} + sz_{i}"

    return_vars = ", ".join(f"out_{i}" for i in range(n_splits))
    size_params = ", ".join(f"s{i}" for i in range(n_splits))

    func_src = f"""
def write_split(buffer, {size_params}):
{chr(10).join(slice_lines)}
    return ({return_vars},)
"""
    fn = compile_numba_function_src(func_src, "write_split", {"np": np})
    return numba_basic.numba_njit(fn)


@register_funcify_default_op_cache_key(WriteJoin)
def numba_funcify_WriteJoin(op, node, **kwargs):
    n_deps = len(node.inputs) - 1

    dep_params = ", ".join(f"dep{i}" for i in range(n_deps))
    sig = f"buffer, {dep_params}" if dep_params else "buffer"

    func_src = f"""
def write_join({sig}):
    return buffer
"""
    fn = compile_numba_function_src(func_src, "write_join")
    return numba_basic.numba_njit(fn)
