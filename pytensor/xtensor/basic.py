import pytensor.scalar as ps
import pytensor.xtensor as px
from pytensor.graph import Apply, Op
from pytensor.tensor import TensorType


class TensorFromXTensor(Op):
    # TODO: May need mapping of named dims to positional dims?

    def make_node(self, x) -> Apply:
        if not isinstance(x.type, px.XTensorType):
            raise TypeError(f"x must be have an XTensorType, got {type(x.type)}")
        output = TensorType(x.type.dtype, shape=x.type.shape)()
        return Apply(self, [x], [output])

    def perform(self, node, inputs, output_storage) -> None:
        [x] = inputs
        output_storage[0][0] = x.copy()


tensor_from_xtensor = TensorFromXTensor()


class XTensorFromTensor(Op):
    __props__ = ("dims",)

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def make_node(self, x) -> Apply:
        if not isinstance(x.type, TensorType):
            raise TypeError(f"x must be an TensorType type, got {type(x.type)}")
        output = px.XTensorType(x.type.dtype, dims=self.dims, shape=x.type.shape)()
        return Apply(self, [x], [output])

    def perform(self, node, inputs, output_storage) -> None:
        [x] = inputs
        output_storage[0][0] = x.copy()


def xtensor_from_tensor(x, dims):
    return XTensorFromTensor(dims=dims)(x)


class XElemwise(Op):
    __props__ = ("scalar_op",)

    def __init__(self, scalar_op):
        super().__init__()
        self.scalar_op = scalar_op

    def make_node(self, *inputs):
        inputs = [px.as_xtensor(inp) for inp in inputs]

        # TODO: This ordering is different than what xarray does
        unique_dims: dict[str, int | None] = {}
        for inp in inputs:
            for dim, dim_length in zip(inp.type.dims, inp.type.shape):
                if dim not in unique_dims:
                    unique_dims[dim] = dim_length
                elif dim_length is not None:
                    # Check for conflicting shapes
                    if (unique_dims[dim] is not None) and (
                        unique_dims[dim] != dim_length
                    ):
                        raise ValueError(f"Dimension {dim} has conflicting shapes")
                    # Keep the non-None shape
                    unique_dims[dim] = dim_length

        dims, shape = zip(*sorted(unique_dims.items()))

        # TODO: Fix dtype
        output_type = px.XTensorType("float64", dims=dims, shape=shape)
        outputs = [output_type() for _ in range(self.scalar_op.nout)]
        return Apply(self, inputs, outputs)

    def perform(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "xtensor operations must be rewritten as tensor operations"
        )


add = XElemwise(ps.add)
exp = XElemwise(ps.exp)
