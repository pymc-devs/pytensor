from pytensor.compile.builders import OpFromGraph, SymbolicOp
from pytensor.scalar.basic import (
    _resolve_float_dtypes,
    resolve_weak_dtypes,
)
from pytensor.tensor.utils import as_tensor_variables


class TensorSymbolicOp(SymbolicOp):
    """SymbolicOp that resolves input dtypes before converting tensor arguments."""

    def resolve_dtypes(self, input_dtypes):
        """Resolve input conversions; output dtypes are inferred by the inner graph."""
        return resolve_weak_dtypes(input_dtypes), None

    def filter_inputs(self, *inputs):
        return as_tensor_variables(*inputs, resolve_dtypes=self.resolve_dtypes)


def resolve_float_dtypes(self, input_dtypes):
    inputs = resolve_weak_dtypes(input_dtypes)
    if any(dtype.startswith(("float", "complex")) for dtype in inputs):
        return inputs, None
    inputs, _ = _resolve_float_dtypes(self, input_dtypes)
    return inputs, None


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
