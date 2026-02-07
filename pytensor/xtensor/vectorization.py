from collections.abc import Mapping, Sequence
from functools import singledispatch
from itertools import chain

import numpy as np

from pytensor import Variable, shared
from pytensor import scalar as ps
from pytensor.graph import Apply, Op
from pytensor.graph.replace import _vectorize_node, graph_replace
from pytensor.graph.traversal import toposort, truncated_graph_inputs
from pytensor.graph.type import HasShape
from pytensor.scalar import discrete_dtypes
from pytensor.tensor import tensor
from pytensor.tensor.random.op import RNGConsumerOp
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.utils import (
    get_static_shape_from_size_variables,
)
from pytensor.utils import unzip
from pytensor.xtensor.basic import XOp, XTypeCastOp
from pytensor.xtensor.type import XTensorType, XTensorVariable, as_xtensor, xtensor


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

        output_dims, output_shape = unzip(combine_dims_and_shape(inputs).items(), n=2)
        dummy_scalars = [ps.get_scalar_type(inp.type.dtype)() for inp in inputs]
        output_dtypes = [
            out.type.dtype for out in self.scalar_op.make_node(*dummy_scalars).outputs
        ]
        outputs = [
            xtensor(dtype=output_dtype, dims=output_dims, shape=output_shape)
            for output_dtype in output_dtypes
        ]
        return Apply(self, inputs, outputs)

    def vectorize_node(self, node, *new_inputs, new_dim):
        return self(*new_inputs, return_list=True)


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

        core_inputs_dims, core_outputs_dims = self.core_dims
        core_input_dims_set = set(chain.from_iterable(core_inputs_dims))

        # Check no input has a core_dim it shouldn't have
        for i, (inp, core_inp_dims) in enumerate(
            zip(inputs, core_inputs_dims, strict=True)
        ):
            if invalid_dims := (
                set(inp.dims) & (core_input_dims_set - set(core_inp_dims))
            ):
                raise ValueError(
                    f"Input {i} has invalid core dims {sorted(invalid_dims)}. Allowed: {core_inp_dims}"
                )

        dims_and_shape = combine_dims_and_shape(inputs)
        batch_dims, batch_shape = unzip(
            ((k, v) for k, v in dims_and_shape.items() if k not in core_input_dims_set),
            n=2,
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

    def vectorize_node(self, node, *new_inputs, new_dim):
        return self(*new_inputs, return_list=True)


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

        batch_output_dims, batch_output_shape = unzip(
            (
                (dim, dim_length)
                for dim, dim_length in (
                    extra_dims_and_shape | params_dims_and_shape
                ).items()
                if dim not in input_core_dims_set
            ),
            n=2,
        )

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

    def vectorize_node(self, node, *new_inputs, new_dim):
        if len(new_inputs) != len(node.inputs):
            raise NotImplementedError(
                f"Vectorization of {self} with additional extra_dim_lengths not implemented, "
                "as it can't infer new dimension labels"
            )
        new_rng, *new_extra_dim_lengths_and_params = new_inputs
        new_extra_dim_lengths, new_params = (
            new_extra_dim_lengths_and_params[: len(self.extra_dims)],
            new_extra_dim_lengths_and_params[len(self.extra_dims) :],
        )

        new_extra_dim_lengths = [dl.squeeze() for dl in new_extra_dim_lengths]
        if not all(dl.type.ndim == 0 for dl in new_extra_dim_lengths):
            raise NotImplementedError(
                f"Vectorization of {self} with batched extra_dim_lengths not implemented, "
            )

        return self.make_node(new_rng, *new_extra_dim_lengths, *new_params).outputs


@_vectorize_node.register(XOp)
@_vectorize_node.register(XTypeCastOp)
def vectorize_xop(op, node, *new_inputs) -> Apply:
    # This gets called by regular graph_replace, which isn't aware of xtensor and doesn't have a concept of `new_dim`
    return vectorize_xnode(node.op, node, *new_inputs, new_dim=None)


@singledispatch
def vectorize_xnode(
    op: XOp | XTypeCastOp,
    node: Apply,
    *batched_inputs: Variable,
    new_dim: str | None = None,
) -> tuple[Variable]:
    """Returns vectorized version of node with new batched inputs."""

    all_old_dims_set = set(
        chain.from_iterable(
            x.dims
            for x in (*node.inputs, *node.outputs)
            if isinstance(x.type, XTensorType)
        )
    )
    for new_inp, old_inp in zip(batched_inputs, node.inputs, strict=True):
        if not (
            isinstance(new_inp.type, XTensorType)
            and isinstance(old_inp.type, XTensorType)
        ):
            continue

        old_dims_set = set(old_inp.dims)
        new_dims_set = set(new_inp.dims)

        # Validate that new inputs didn't drop pre-existing dims
        if missing_dims := old_dims_set - new_dims_set:
            raise ValueError(
                f"Vectorized input {new_inp} is missing pre-existing dims: {sorted(missing_dims)}"
            )
        # Or have new dimensions that were already in the graph
        if new_core_dims := ((new_dims_set - old_dims_set) & all_old_dims_set):
            raise ValueError(
                f"Vectorized input {new_inp} has new dimensions that were present in the original graph: {new_core_dims}"
            )

    def align_dims(new_x, old_x):
        if isinstance(new_x.type, XTensorType):
            if new_dim is not None and new_dim in new_x.dims:
                return new_x.transpose(new_dim, *old_x.dims)
            else:
                return new_x.transpose(..., *old_x.dims)
        else:
            return new_x

    vectorized_outs = op.vectorize_node(
        node,
        *(
            align_dims(new_x, old_x)
            for new_x, old_x in zip(batched_inputs, node.inputs)
        ),
        new_dim=new_dim,
    )

    return tuple(
        align_dims(new_out, old_out)
        for new_out, old_out in zip(vectorized_outs, node.outputs, strict=True)
    )


def _vectorize_single_dim(outputs, replace, new_dim: str):
    inputs = truncated_graph_inputs(outputs, ancestors_to_include=replace.keys())
    new_inputs = [replace.get(inp, inp) for inp in inputs]

    vect_vars = dict(zip(inputs, new_inputs, strict=True))
    for node in toposort(outputs, blockers=inputs):
        vect_inputs = [vect_vars.get(inp, inp) for inp in node.inputs]

        if isinstance(node.op, XOp | XTypeCastOp):
            node_vect_outs = vectorize_xnode(
                node.op, node, *vect_inputs, new_dim=new_dim
            )
        else:
            node_vect_outs = _vectorize_node(node.op, node, *vect_inputs)
            if isinstance(node_vect_outs, Apply):
                # Old API
                node_vect_outs = node_vect_outs.outputs

        for output, vect_output in zip(node.outputs, node_vect_outs, strict=True):
            if output in vect_vars:
                # This can happen when some outputs of a multi-output node are given a replacement,
                # while some of the remaining outputs are still needed in the graph.
                # We make sure we don't overwrite the provided replacement with the newly vectorized output
                continue
            vect_vars[output] = vect_output

    return [vect_vars[out] for out in outputs]


def vectorize_graph(
    outputs: Variable | Sequence[Variable],
    replace: Mapping[XTensorVariable, XTensorVariable],
):
    new_dims = []
    for old, new in replace.items():
        if not old.type.in_same_class(new.type):
            raise ValueError(
                f"Vectorized input {new} is not of the same type as the variable {old} "
                f"it is trying to replace {new.type} vs {old.type}"
            )
        if not isinstance(new.type, XTensorType):
            if isinstance(new.type, HasShape) and new.type.ndim != old.type.ndim:
                # We have no way of knowing what `new_dims` the new axis correspond to
                raise ValueError(
                    "A non-XTensorType input with batch dimensions was provided. "
                    "The semantics of xtensor.vectorize_graph are not well defined in this case."
                )
            continue

        old_dims_set = set(old.dims)
        new_dims_set = set(new.dims)
        if missing_dims := old_dims_set - new_dims_set:
            raise ValueError(
                f"Vectorized input {new} is missing pre-existing dims: {sorted(missing_dims)}"
            )
        new_dims.extend(dim for dim in new.dims if dim not in old_dims_set)

    if isinstance(outputs, Sequence):
        seq_outputs = outputs
    else:
        seq_outputs = [outputs]

    # Align batch dims on the left
    replace = {
        k: v.transpose(*new_dims, ..., missing_dims="ignore")
        for k, v in replace.items()
    }

    if not new_dims:
        return graph_replace(seq_outputs, replace, strict=False)

    seq_vect_outputs = seq_outputs
    remaining_new_dims = list(new_dims)
    while new_dims:
        new_dim = remaining_new_dims.pop()

        if remaining_new_dims:
            # We need to use a dummy inputs to batch graph once at a time
            # We drop all the dims that are still in `remaining_new_dims`
            single_dim_replace = {
                k: v.type.clone(
                    dims=tuple(dim for dim in v.dims if dim not in remaining_new_dims)
                )
                for k, v in replace.items()
            }
            replace = dict(zip(single_dim_replace.values(), replace.keys()))
        else:
            single_dim_replace = replace
        seq_vect_outputs = _vectorize_single_dim(
            seq_vect_outputs, single_dim_replace, new_dim
        )

    if isinstance(outputs, Sequence):
        return seq_vect_outputs
    else:
        [vect_output] = seq_vect_outputs
        return vect_output
