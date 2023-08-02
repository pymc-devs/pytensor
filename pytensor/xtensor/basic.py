from itertools import chain

import pytensor.scalar as ps
from pytensor.graph import Apply, Op
from pytensor.tensor import TensorType, tensor
from pytensor.tensor.utils import _parse_gufunc_signature
from pytensor.xtensor.type import XTensorType, as_xtensor, xtensor


class XOp(Op):
    """A base class for XOps that shouldn't be materialized"""

    def perform(self, node, inputs, outputs):
        raise NotImplementedError(
            "xtensor operations must be rewritten as tensor operations"
        )


class XViewOp(Op):
    # Make this a View Op with C-implementation
    view_map = {0: [0]}

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = inputs[0]


class TensorFromXTensor(XViewOp):
    __props__ = ()

    def make_node(self, x) -> Apply:
        if not isinstance(x.type, XTensorType):
            raise TypeError(f"x must be have an XTensorType, got {type(x.type)}")
        output = TensorType(x.type.dtype, shape=x.type.shape)()
        return Apply(self, [x], [output])


tensor_from_xtensor = TensorFromXTensor()


class XTensorFromTensor(XViewOp):
    __props__ = ("dims",)

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def make_node(self, x) -> Apply:
        if not isinstance(x.type, TensorType):
            raise TypeError(f"x must be an TensorType type, got {type(x.type)}")
        output = xtensor(dtype=x.type.dtype, dims=self.dims, shape=x.type.shape)
        return Apply(self, [x], [output])


def xtensor_from_tensor(x, dims):
    return XTensorFromTensor(dims=dims)(x)


class Rename(XViewOp):
    __props__ = ("new_dims",)

    def __init__(self, new_dims: tuple[str, ...]):
        super().__init__()
        self.new_dims = new_dims

    def make_node(self, x):
        x = as_xtensor(x)
        output = x.type.clone(dims=self.new_dims)()
        return Apply(self, [x], [output])


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
        except IndexError:
            raise ValueError(
                f"Cannot rename {old_name} to {new_name}: {old_name} not in {old_names}"
            )

    return Rename(tuple(new_names))(x)


class XElemwise(XOp):
    __props__ = ("scalar_op",)

    def __init__(self, scalar_op):
        super().__init__()
        self.scalar_op = scalar_op

    def make_node(self, *inputs):
        inputs = [as_xtensor(inp) for inp in inputs]
        if (self.scalar_op.nin != -1) and (len(inputs) != self.scalar_op.nin):
            raise ValueError(
                f"Wrong number of inputs, expected {self.scalar_op.nin}, got {len(inputs)}"
            )

        dims_and_shape: dict[str, int | None] = {}
        for inp in inputs:
            for dim, dim_length in zip(inp.type.dims, inp.type.shape):
                if dim not in dims_and_shape:
                    dims_and_shape[dim] = dim_length
                elif dim_length is not None:
                    # Check for conflicting shapes
                    if (dims_and_shape[dim] is not None) and (
                        dims_and_shape[dim] != dim_length
                    ):
                        raise ValueError(f"Dimension {dim} has conflicting shapes")
                    # Keep the non-None shape
                    dims_and_shape[dim] = dim_length

        output_dims, output_shape = zip(*dims_and_shape.items())

        dummy_scalars = [ps.get_scalar_type(inp.type.dtype)() for inp in inputs]
        output_dtypes = [
            out.type.dtype for out in self.scalar_op.make_node(*dummy_scalars).outputs
        ]
        outputs = [
            xtensor(dtype=output_dtype, dims=output_dims, shape=output_shape)
            for output_dtype in output_dtypes
        ]
        return Apply(self, inputs, outputs)


class XBlockwise(XOp):
    __props__ = ("core_op", "signature", "core_dims")

    def __init__(
        self,
        core_op: Op,
        signature: str,
        core_dims: tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]],
    ):
        super().__init__()
        self.core_op = core_op
        self.signature = signature
        self.inputs_sig, self.outputs_sig = _parse_gufunc_signature(signature)
        self.core_dims = core_dims

    def make_node(self, *inputs):
        inputs = [as_xtensor(i) for i in inputs]
        if len(inputs) != len(self.inputs_sig):
            raise ValueError(
                f"Wrong number of inputs, expected {len(self.inputs_sig)}, got {len(inputs)}"
            )

        dims_and_shape: dict[str, int | None] = {}
        for inp in inputs:
            for dim, dim_length in zip(inp.type.dims, inp.type.shape):
                if dim not in dims_and_shape:
                    dims_and_shape[dim] = dim_length
                elif dim_length is not None:
                    # Check for conflicting shapes
                    if (dims_and_shape[dim] is not None) and (
                        dims_and_shape[dim] != dim_length
                    ):
                        raise ValueError(f"Dimension {dim} has conflicting shapes")
                    # Keep the non-None shape
                    dims_and_shape[dim] = dim_length

        core_inputs_dims, core_outputs_dims = self.core_dims
        # TODO: Avoid intermediate dict
        core_dims = set(chain.from_iterable(core_inputs_dims))
        batched_dims_and_shape = {
            k: v for k, v in dims_and_shape.items() if k not in core_dims
        }
        batch_dims, batch_shape = zip(*batched_dims_and_shape.items())

        dummy_core_inputs = []
        for inp, core_inp_dims in zip(inputs, core_inputs_dims):
            try:
                core_static_shape = [
                    inp.type.shape[inp.type.dims.index(d)] for d in core_inp_dims
                ]
            except IndexError:
                raise ValueError(
                    f"At least one core dim={core_inp_dims} missing from input {inp} with dims={inp.type.dims}"
                )
            dummy_core_inputs.append(
                tensor(dtype=inp.type.dtype, shape=core_static_shape)
            )
        core_node = self.core_op.make_node(*dummy_core_inputs)

        outputs = [
            xtensor(
                dtype=core_out.type.dtype,
                shape=batch_shape + core_out.type.shape,
                dims=batch_dims + core_out_dims,
            )
            for core_out, core_out_dims in zip(core_node.outputs, core_outputs_dims)
        ]
        return Apply(self, inputs, outputs)
