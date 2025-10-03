import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify, mlx_typify
from pytensor.tensor.basic import get_underlying_scalar_constant_value
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.signal.conv import Convolve1d


@mlx_funcify.register(Convolve1d)
def mlx_funcify_Convolve1d(op, node, **kwargs):
    _, _, full_mode_var = node.inputs

    try:
        full_mode = bool(get_underlying_scalar_constant_value(full_mode_var))
        runtime_mode_static = True
    except NotScalarConstantError:
        full_mode = True
        runtime_mode_static = False

    def conv1d(raw_data, raw_kernel, runtime_full_mode):
        data = mlx_typify(raw_data, dtype=None)
        kernel = mlx_typify(raw_kernel, dtype=None)

        if runtime_mode_static:
            runtime_mode = full_mode
        else:
            runtime_full_mode = mx.array(runtime_full_mode)
            runtime_mode = bool(runtime_full_mode.reshape(-1)[0])

        mode = "full" if runtime_mode else "valid"
        return mx.convolve(data, kernel, mode=mode)

    return conv1d
