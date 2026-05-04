from pytensor.compile.builders import SymbolicOp
from pytensor.tensor.basic import as_tensor_variable


class TensorSymbolicOp(SymbolicOp):
    """SymbolicOp that converts inputs via ``as_tensor_variable``."""

    @staticmethod
    def filter_inputs(*inputs):
        return tuple(as_tensor_variable(inp) for inp in inputs)
