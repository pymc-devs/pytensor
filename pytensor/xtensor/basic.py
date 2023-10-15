from itertools import chain
from typing import Any, Iterable, Sequence

import pytensor.scalar as ps
import pytensor.xtensor as px
from pytensor.graph import Apply, Op
from pytensor.graph.basic import Variable
from pytensor.graph.op import OutputStorageType, ParamsInputType
from pytensor.tensor import TensorType
from pytensor.xtensor.spaces import DimLike, Space


class TensorFromXTensor(Op):
    def make_node(self, *inputs: Variable) -> Apply:
        [x] = inputs
        if not isinstance(x.type, px.XTensorType):
            raise TypeError(f"x must be have an XTensorType, got {type(x.type)}")
        output = TensorType(x.type.dtype, shape=x.type.shape)()
        return Apply(self, inputs, [output])

    def perform(
        self,
        node: Apply,
        inputs: Sequence[Any],
        output_storage: OutputStorageType,
        params: ParamsInputType = None,
    ) -> None:
        [x] = inputs
        output_storage[0][0] = x.copy()


tensor_from_xtensor = TensorFromXTensor()


class XTensorFromTensor(Op):
    __props__ = ("dims",)

    def __init__(self, dims: Iterable[DimLike]):
        super().__init__()
        self.dims = Space(dims)

    def make_node(self, *inputs: Variable) -> Apply:
        [x] = inputs
        if not isinstance(x.type, TensorType):
            raise TypeError(f"x must be an TensorType type, got {type(x.type)}")
        output = px.XTensorType(x.type.dtype, dims=self.dims, shape=x.type.shape)()
        return Apply(self, inputs, [output])

    def perform(
        self,
        node: Apply,
        inputs: Sequence[Any],
        output_storage: OutputStorageType,
        params: ParamsInputType = None,
    ) -> None:
        [x] = inputs
        output_storage[0][0] = x.copy()


def xtensor_from_tensor(x, dims: Iterable[DimLike]):
    return XTensorFromTensor(dims=dims)(x)


class XElemwise(Op):
    __props__ = ("scalar_op",)

    def __init__(self, scalar_op):
        super().__init__()
        self.scalar_op = scalar_op

    def make_node(self, *inputs):
        # TODO: Check dim lengths match
        inputs = [px.as_xtensor_variable(inp) for inp in inputs]
        # NOTE: The output will have unordered dims
        output_dims = set(chain.from_iterable(inp.type.dims for inp in inputs))
        # TODO: Fix dtype
        output_type = px.XTensorType(
            "float64", dims=output_dims, shape=(None,) * len(output_dims)
        )
        outputs = [output_type() for _ in range(self.scalar_op.nout)]
        return Apply(self, inputs, outputs)

    def perform(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "xtensor operations must be rewritten as tensor operations"
        )


add = XElemwise(ps.add)
