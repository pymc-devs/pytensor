from collections.abc import Sequence
from itertools import chain

import numpy as np

from pytensor import scalar as ps
from pytensor import shared
from pytensor.graph import Apply, Op
from pytensor.scalar import discrete_dtypes
from pytensor.tensor import tensor
from pytensor.tensor.random.op import RNGConsumerOp
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.utils import (
    get_static_shape_from_size_variables,
)
from pytensor.xtensor.basic import XOp
from pytensor.xtensor.type import XTensorVariable, as_xtensor, xtensor


def combine_dims_and_shape(
    inputs: Sequence[XTensorVariable], exclude: Sequence[str] | None = None
) -> dict[str, int | None]:
    """Combine information of static dimensions and shapes from multiple xtensor inputs.

    Exclude
    """
    exclude_set: set[str] = set() if exclude is None else set(exclude)
    dims_and_shape: dict[str, int | None] = {}
    for inp in inputs:
        for dim, dim_length in zip(inp.type.dims, inp.type.shape):
            if dim in exclude_set:
                continue
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


class XRV(XOp, RNGConsumerOp):
    """Wrapper for RandomVariable operations that follows xarray-like broadcasting semantics.

    Xarray does not offer random generators, so this class implements a new API.

    It mostly works like a gufunc (or XBlockwise), which specifies core dimensions for inputs and output, and
    enforces dim-based broadcasting between inputs and output.

    It differs from XBlockwise in a couple of ways:
    1. It is restricted to one sample output
    2. It takes a random generator as the first input and returns the consumed generator as the first output.
    3. It has the concept of extra dimensions, which determine extra batch dimensions of the output, that are not
    implied by batch dimensions of the parameters.
    """

    default_output = 1
    __props__ = ("core_op", "core_dims", "extra_dims")

    def __init__(
        self,
        core_op,
        core_dims: tuple[tuple[tuple[str, ...], ...], tuple[str, ...]],
        extra_dims: tuple[str, ...],
        name: str | None = None,
    ):
        super().__init__()
        if name is None:
            name = getattr(core_op, "name", None)
        self.name = name
        self.core_op = core_op
        inps_core_dims, out_core_dims = core_dims
        for operand_dims in (*inps_core_dims, out_core_dims):
            if len(set(operand_dims)) != len(operand_dims):
                raise ValueError(f"Operand has repeated dims {operand_dims}")
        self.core_dims = (tuple(i for i in inps_core_dims), tuple(out_core_dims))
        if len(set(extra_dims)) != len(extra_dims):
            raise ValueError("size_dims must be unique")
        self.extra_dims = tuple(extra_dims)

    def __str__(self):
        if self.name is not None:
            name = self.name
            attrs = f"(core_dims={self.core_dims}, extra_dims={self.extra_dims})"
        else:
            name = self.__class__.__name__
            attrs = f"(core_op={self.core_op}, core_dims={self.core_dims}, extra_dims={self.extra_dims})"
        return f"{name}({attrs})"

    def update(self, node):
        # RNG input and update are the first input and output respectively
        return {node.inputs[0]: node.outputs[0]}

    def make_node(self, rng, *extra_dim_lengths_and_params):
        if rng is None:
            rng = shared(np.random.default_rng())
        elif not isinstance(rng.type, RandomType):
            raise TypeError(
                "The type of rng should be an instance of RandomGeneratorType "
            )

        extra_dim_lengths = [
            as_xtensor(dim_length).values
            for dim_length in extra_dim_lengths_and_params[: len(self.extra_dims)]
        ]
        if not all(
            (dim_length.type.ndim == 0 and dim_length.type.dtype in discrete_dtypes)
            for dim_length in extra_dim_lengths
        ):
            raise TypeError("All dimension lengths should be scalar discrete dtype.")

        params = [
            as_xtensor(param)
            for param in extra_dim_lengths_and_params[len(self.extra_dims) :]
        ]
        if len(params) != len(self.core_op.ndims_params):
            raise ValueError(
                f"Expected {len(self.core_op.ndims_params)} parameters + {len(self.extra_dims)} dim_lengths, "
                f"got {len(extra_dim_lengths_and_params)}"
            )

        param_core_dims, output_core_dims = self.core_dims
        input_core_dims_set = set(chain.from_iterable(param_core_dims))

        # Check parameters don't have core dimensions they shouldn't have
        for param, core_param_dims in zip(params, param_core_dims):
            if invalid_core_dims := (
                set(param.type.dims) - set(core_param_dims)
            ).intersection(input_core_dims_set):
                raise ValueError(
                    f"Parameter {param} has invalid core dimensions {sorted(invalid_core_dims)}"
                )

        extra_dims_and_shape = dict(
            zip(
                self.extra_dims, get_static_shape_from_size_variables(extra_dim_lengths)
            )
        )
        params_dims_and_shape = combine_dims_and_shape(params)

        # Check that no parameter dims conflict with size dims
        if conflict_dims := set(extra_dims_and_shape).intersection(
            params_dims_and_shape
        ):
            raise ValueError(
                f"Size dimensions {sorted(conflict_dims)} conflict with parameter dimensions. They should be unique."
            )

        batch_dims_and_shape = [
            (dim, dim_length)
            for dim, dim_length in (
                extra_dims_and_shape | params_dims_and_shape
            ).items()
            if dim not in input_core_dims_set
        ]
        if batch_dims_and_shape:
            batch_output_dims, batch_output_shape = zip(*batch_dims_and_shape)
        else:
            batch_output_dims, batch_output_shape = (), ()

        dummy_core_inputs = []
        for param, core_param_dims in zip(params, param_core_dims):
            try:
                core_static_shape = [
                    param.type.shape[param.type.dims.index(d)] for d in core_param_dims
                ]
            except ValueError:
                raise ValueError(
                    f"At least one core dim={core_param_dims} missing from input {param} with dims={param.type.dims}"
                )
            dummy_core_inputs.append(
                tensor(dtype=param.type.dtype, shape=core_static_shape)
            )
        core_node = self.core_op.make_node(rng, None, *dummy_core_inputs)

        if not len(core_node.outputs) == 2:
            raise NotImplementedError(
                "XRandomVariable only supports core ops with two outputs (rng, out)"
            )

        _, core_out = core_node.outputs
        out = xtensor(
            dtype=core_out.type.dtype,
            shape=batch_output_shape + core_out.type.shape,
            dims=batch_output_dims + output_core_dims,
        )

        return Apply(self, [rng, *extra_dim_lengths, *params], [rng.type(), out])
