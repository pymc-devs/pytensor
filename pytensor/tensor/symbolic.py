from pytensor.compile.builders import OpFromGraph, SymbolicOp
from pytensor.tensor.basic import as_tensor_variable


class TensorSymbolicOp(SymbolicOp):
    """SymbolicOp that converts inputs via ``as_tensor_variable``."""

    @staticmethod
    def filter_inputs(*inputs):
        return tuple(as_tensor_variable(inp) for inp in inputs)


class FusibleSymbolicOpDerivative(OpFromGraph):
    """A type-specialized derivative graph, exposed with symbolic forward graphs at fusion.

    Unlike `SymbolicOp`, this graph cannot rebuild itself for new input types.
    """

    @property
    def derivative_op_type(self):
        return type(self)


class FusibleSymbolicOp(TensorSymbolicOp):
    """Inline the forward graph and generated derivatives at tensor fusion.

    Use for graphs that can fuse without first running the full rewrite pipeline.
    """

    derivative_op_type = FusibleSymbolicOpDerivative
