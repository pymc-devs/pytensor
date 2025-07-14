from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

import numpy as np

from pytensor.graph.basic import Apply
from pytensor.graph.op import Op, Variable
from pytensor.xtensor.type import (
    DIM_LENGTH_TYPE,
    DIM_LENGTH_VARIABLE,
    BasicDim,
    CloneDim,
    DimType,
    DimVariable,
    XTensorVariable,
)


class DimOp(Op):
    def perform(self, node, inputs, outputs):
        raise NotImplementedError(
            f"xtensor operation {self} must be lowered to equivalent tensor operations"
        )


# Not a dim op, because it doesn't return a DimVariable
class Length(Op):
    __props__ = ()

    def make_node(self, *inputs: Variable) -> Apply:
        (x,) = inputs
        if not isinstance(x, DimVariable):
            raise TypeError(f"x must be a DimVariable, got {type(x.type)}")
        return Apply(self, [x], [DIM_LENGTH_TYPE()])

    def perform(self, node, inputs, outputs):
        # outputs[0][0] = np.int64(inputs[0])
        outputs[0][0] = np.array(inputs[0], dtype=DIM_LENGTH_TYPE.dtype)


def _dim_size(dim: DimVariable) -> DIM_LENGTH_VARIABLE:
    if dim.type.size is not None:
        return DIM_LENGTH_TYPE.filter_variable(dim.type.size)
    return Length()(dim)


class FromLength(DimOp):
    __props__ = ("dim_type",)

    def __init__(self, dim_type: DimType):
        super().__init__()
        self.dim_type = dim_type

    def make_node(self, *inputs: Variable) -> Apply:
        (length,) = inputs
        if not isinstance(length, DIM_LENGTH_VARIABLE):
            raise TypeError(
                f"length must be a DIM_LENGTH_VARIABLE, got {type(length.type)}"
            )
        if length.type != DIM_LENGTH_TYPE:
            raise TypeError(
                f"length must be of dtype 'DIM_LENGTH_SCALAR', got {length.type.dtype}"
            )
        return Apply(self, [length], [self.dim_type()])

    def perform(self, node, inputs, outputs):
        """Convert the length to a list of lengths."""
        outputs[0][0] = inputs[0]


def from_length(length: DIM_LENGTH_VARIABLE, name: str | None = None) -> DimVariable:
    # TODO add check for dtype
    if not isinstance(length, DIM_LENGTH_VARIABLE):
        raise TypeError(
            f"length must be a DIM_LENGTH_VARIABLE, got {type(length.type)}"
        )
    if length.type != DIM_LENGTH_TYPE:
        raise TypeError(
            f"length must be of dtype 'DIM_LENGTH_SCALAR', got {length.type.dtype}"
        )

    uuid = uuid4()
    dim_type = BasicDim(uuid=uuid, name=name)
    op = FromLength(dim_type)
    return op(length, name=name)


class DimFromTensor(Op):
    __props__ = ("dim_type",)

    def __init__(self, dim_type: DimType):
        super().__init__()
        self.dim_type = dim_type

    def make_node(self, *inputs: Variable) -> Apply:
        (x,) = inputs
        if not isinstance(x, XTensorVariable):
            raise TypeError(f"x must be an XTensorVariable, got {type(x.type)}")
        return Apply(self, [x], [self.dim_type()])

    def perform(self, node, inputs, outputs):
        """Convert the tensor to a dimension variable."""
        (x,) = inputs
        (x_var,) = node.inputs
        for i, dim in enumerate(x_var.type.dims):
            if dim == self.dim_type:
                # outputs[0][0] = np.int64(x.shape[i])
                outputs[0][0] = np.array(x.shape[i], dtype=DIM_LENGTH_TYPE.dtype)
                return
        raise ValueError(f"Dimension {self.dim_type} not found in tensor {x.type.dims}")


def _dim_from_tensor(x: XTensorVariable, idx: int) -> DimVariable:
    op = DimFromTensor(dim_type=x.type.dims[idx])
    return op(x, name=x.type.dims[idx].name)


class Clone(Op):
    __props__ = ("dim_type",)

    def __init__(self, dim_type):
        super().__init__()
        self.dim_type = dim_type

    def make_node(self, *inputs: Variable) -> Apply:
        (x,) = inputs
        if not isinstance(x, DimVariable):
            raise TypeError(f"x must be a DimVariable, got {type(x.type)}")
        return Apply(self, [x], [self.dim_type()])

    def perform(self, node, inputs, outputs):
        outputs[0][0] = inputs[0]


def _clone_dim(dim: DimVariable, *, name: str | None = None) -> DimVariable:
    """Rename a dimension variable.

    Args:
        name: The new name for the dimension.

    Returns:
        A new DimVariable with the updated name.
    """
    dim_type = CloneDim(uuid=uuid4(), base=dim.type, name=name)
    return Clone(dim_type)(dim, name=name)


class Product(Op):
    __props__ = ()

    def make_node(self, *dims: Variable) -> Apply:
        if not all(isinstance(dim, DimVariable) for dim in dims):
            raise TypeError("All inputs must be DimVariables.")
        out = dim_type()
        return Apply(self, list(dims), [out])

    def perform(self, node, inputs, outputs):
        outputs[0][0] = np.prod(inputs, dtype=DIM_LENGTH_TYPE.dtype).item()


def product_dim(*dims: DimVariable, name: str | None = None) -> DimVariable:
    return Product()(*dims, name=name)


def rebase_dim(dim: DimVariable | DimType, *tensors: XTensorVariable) -> DimVariable:
    if not isinstance(dim, DimVariable | DimType):
        raise TypeError(f"dim must be a DimVariable, got {type(dim)}")

    if not tensors:
        raise ValueError("At least one tensor must be provided for rebasing.")

    if isinstance(dim, DimVariable):
        dim_type = dim.type
    else:
        dim_type = dim

    for tensor in tensors:
        for i, tensor_dim in enumerate(tensor.type.dims):
            if dim_type == tensor_dim:
                return _dim_from_tensor(tensor, idx=i)
    raise ValueError(f"Dimension {dim} not found in any of the provided tensors.")


def rebase_dims(
    dims: Iterable[DimVariable | DimType], *tensors: XTensorVariable
) -> list[DimVariable]:
    return [rebase_dim(dim, *tensors) for dim in dims]
