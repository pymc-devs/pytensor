from itertools import chain

from pytensor import scalar as ps
from pytensor.graph import Apply, Op
from pytensor.tensor import tensor
from pytensor.xtensor.basic import XOp
from pytensor.xtensor.type import as_xtensor, xtensor


def combine_dims_and_shape(inputs):
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
    return dims_and_shape


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

        dims_and_shape = combine_dims_and_shape(inputs)
        if dims_and_shape:
            output_dims, output_shape = zip(*dims_and_shape.items())
        else:
            output_dims, output_shape = (), ()

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
    __props__ = ("core_op", "core_dims")

    def __init__(
        self,
        core_op: Op,
        core_dims: tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]],
        signature: str | None = None,
    ):
        super().__init__()
        self.core_op = core_op
        self.core_dims = core_dims
        self.signature = signature  # Only used for lowering, not for validation

    def make_node(self, *inputs):
        inputs = [as_xtensor(i) for i in inputs]
        if len(inputs) != len(self.core_dims[0]):
            raise ValueError(
                f"Wrong number of inputs, expected {len(self.core_dims[0])}, got {len(inputs)}"
            )

        dims_and_shape = combine_dims_and_shape(inputs)

        core_inputs_dims, core_outputs_dims = self.core_dims
        core_input_dims_set = set(chain.from_iterable(core_inputs_dims))
        batch_dims, batch_shape = zip(
            *((k, v) for k, v in dims_and_shape.items() if k not in core_input_dims_set)
        )

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
