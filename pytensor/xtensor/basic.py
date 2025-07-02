from pytensor.compile.ops import TypeCastingOp
from pytensor.graph import Apply, Op
from pytensor.scalar.basic import uint64
from pytensor.tensor.basic import ones as tensor_ones
from pytensor.tensor.basic import zeros as tensor_zeros
from pytensor.tensor.shape import specify_shape
from pytensor.tensor.type import TensorType
from pytensor.xtensor.type import DimVariable, XTensorType, as_xtensor, xtensor


DIM_LENGTH_SCALAR = uint64


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
        # TODO fix
        [x] = inputs
        [g_out] = g_outs
        return [xtensor_from_tensor(g_out, dims=x.type.dims)]


tensor_from_xtensor = TensorFromXTensor()


class XTensorFromTensor(XTypeCastOp):
    __props__ = ()

    def make_node(self, x, *dims):
        if not isinstance(x.type, TensorType):
            raise TypeError(f"x must be an TensorType type, got {type(x.type)}")
        output = xtensor(dtype=x.type.dtype, dims=dims)
        return Apply(self, [x, *dims], [output])

    def L_op(self, inputs, outs, g_outs):
        # TODO fix
        [g_out] = g_outs
        return [tensor_from_xtensor(g_out)]


def xtensor_from_tensor(x, dims, name=None, check: bool = True):
    if check:
        x = specify_shape(x, [dim.size for dim in dims])
    return XTensorFromTensor()(x, *dims, name=name)


class MapDims(XTypeCastOp):
    __props__ = ("new_dim_indices",)

    def __init__(self, new_dim_indices: tuple[int, ...]):
        self.new_dims_indices = new_dim_indices

    def make_node(self, x, *new_dims):
        x = as_xtensor(x)
        new_dims = list(x.dims)
        for i, idx in enumerate(self.new_dims_indices):
            new_dims[idx] = new_dims[i]

        output = x.type.clone(dims=new_dims)()
        return Apply(self, [x], [output])

    def L_op(self, inputs, outs, g_outs):
        # TODO fix
        [x] = inputs
        [g_out] = g_outs
        return [map_dims(g_out, dims=x.type.dims)]


def map_dims(x, name_dict: dict[DimVariable, DimVariable] | None = None, **names):
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

    return MapDims(tuple(new_names))(x)


def zeros(*dims, dtype=None, name=None):
    """Create a new XTensor filled with zeros."""
    if not dims:
        raise ValueError("At least one dimension must be specified")

    return xtensor_from_tensor(
        tensor_zeros(shape=[dim.size for dim in dims], dtype=dtype),
        dims=dims,
        name=name,
        check=False,
    )


def ones(*dims, dtype=None, name=None):
    """Create a new XTensor filled with zeros."""
    if not dims:
        raise ValueError("At least one dimension must be specified")

    return xtensor_from_tensor(
        tensor_ones(shape=[dim.size for dim in dims], dtype=dtype),
        dims=dims,
        name=name,
        check=False,
    )
