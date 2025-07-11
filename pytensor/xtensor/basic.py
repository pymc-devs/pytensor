from collections.abc import Sequence

from pytensor.compile.ops import TypeCastingOp
from pytensor.graph import Apply, Op
from pytensor.tensor.type import TensorType
from pytensor.xtensor.type import XTensorType, as_xtensor, xtensor


class XOp(Op):
    """A base class for XOps that shouldn't be materialized"""

    def perform(self, node, inputs, outputs):
        raise NotImplementedError(
            f"xtensor operation {self} must be lowered to equivalent tensor operations"
        )


class XTypeCastOp(TypeCastingOp):
    """Base class for Ops that type cast between TensorType and XTensorType.

    This is like a `ViewOp` but without the expectation the input and output have identical types.
    """


class TensorFromXTensor(XTypeCastOp):
    __props__ = ()

    def make_node(self, x):
        if not isinstance(x.type, XTensorType):
            raise TypeError(f"x must be have an XTensorType, got {type(x.type)}")
        output = TensorType(x.type.dtype, shape=x.type.shape)()
        return Apply(self, [x], [output])

    def L_op(self, inputs, outs, g_outs):
        [x] = inputs
        [g_out] = g_outs
        return [xtensor_from_tensor(g_out, dims=x.type.dims)]


tensor_from_xtensor = TensorFromXTensor()


class XTensorFromTensor(XTypeCastOp):
    __props__ = ("dims",)

    def __init__(self, dims: Sequence[str]):
        super().__init__()
        self.dims = tuple(dims)

    def make_node(self, x):
        if not isinstance(x.type, TensorType):
            raise TypeError(f"x must be an TensorType type, got {type(x.type)}")
        output = xtensor(dtype=x.type.dtype, dims=self.dims, shape=x.type.shape)
        return Apply(self, [x], [output])

    def L_op(self, inputs, outs, g_outs):
        [g_out] = g_outs
        return [tensor_from_xtensor(g_out)]


def xtensor_from_tensor(x, dims, name=None):
    return XTensorFromTensor(dims=dims)(x, name=name)


class Rename(XTypeCastOp):
    __props__ = ("new_dims",)

    def __init__(self, new_dims: tuple[str, ...]):
        super().__init__()
        self.new_dims = new_dims

    def make_node(self, x):
        x = as_xtensor(x)
        output = x.type.clone(dims=self.new_dims)()
        return Apply(self, [x], [output])

    def L_op(self, inputs, outs, g_outs):
        [x] = inputs
        [g_out] = g_outs
        return [rename(g_out, dims=x.type.dims)]


def rename(x, name_dict: dict[str, str] | None = None, **names: str):
    if name_dict is not None:
        if names:
            raise ValueError("Cannot use both positional and keyword names in rename")
        names = name_dict

    x = as_xtensor(x)
    old_names = x.type.dims
    new_names = list(old_names)
    for old_name, new_name in names.items():
        try:
            new_names[old_names.index(old_name)] = new_name
        except ValueError:
            raise ValueError(
                f"Cannot rename {old_name} to {new_name}: {old_name} not in {old_names}"
            )

    return Rename(tuple(new_names))(x)
