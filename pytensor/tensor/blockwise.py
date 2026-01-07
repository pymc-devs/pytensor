from collections.abc import Callable, Sequence
from typing import Any, Literal, cast, overload

import numpy as np
from numpy import broadcast_shapes, empty

from pytensor import config
from pytensor.compile.builders import OpFromGraph
from pytensor.gradient import DisconnectedType
from pytensor.graph import FunctionGraph
from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.null_type import NullType
from pytensor.graph.op import Op
from pytensor.graph.replace import (
    _vectorize_node,
    _vectorize_not_needed,
    vectorize_graph,
)
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.link.c.op import COp
from pytensor.scalar import ScalarType
from pytensor.tensor import as_tensor_variable
from pytensor.tensor.shape import shape_padleft
from pytensor.tensor.type import TensorType, tensor
from pytensor.tensor.utils import (
    _parse_gufunc_signature,
    broadcast_static_dim_lengths,
    faster_broadcast_to,
    faster_ndindex,
    import_func_from_string,
    safe_signature,
)
from pytensor.tensor.variable import TensorVariable


def _squeeze_left(x, stop_at_dim: int | None = None):
    """Squeeze any leading dims of `x` until a real dim or `stop_at_dim` (if not None) is reached."""
    x_dims = x.type.broadcastable
    squeeze_ndim = len(x_dims) if all(x_dims) else x_dims.index(False)
    if stop_at_dim is not None:
        squeeze_ndim = min(squeeze_ndim, stop_at_dim)
    if squeeze_ndim == 0:
        return x
    return x.squeeze(axis=tuple(range(squeeze_ndim)))


def _vectorize_node_perform(
    core_node: Apply,
    batch_bcast_patterns: Sequence[tuple[bool, ...]],
    batch_ndim: int,
    impl: str | None,
) -> Callable:
    """Creates a vectorized `perform` function for a given core node.

    Similar behavior of np.vectorize, but specialized for PyTensor Blockwise Op.
    """

    storage_map = {var: [None] for var in core_node.inputs + core_node.outputs}
    try:
        core_thunk = core_node.op.make_thunk(
            core_node, storage_map, None, [], impl=impl
        )
    except NotImplementedError:
        if impl == "c":
            # Try again with py impl
            core_thunk = core_node.op.make_thunk(
                core_node, storage_map, None, [], impl="py"
            )
        else:
            raise
    single_in = len(core_node.inputs) == 1
    core_input_storage = [storage_map[inp] for inp in core_node.inputs]
    core_output_storage = [storage_map[out] for out in core_node.outputs]
    core_storage = core_input_storage + core_output_storage

    def vectorized_perform(
        *args,
        batch_bcast_patterns=batch_bcast_patterns,
        batch_ndim=batch_ndim,
        single_in=single_in,
        core_thunk=core_thunk,
        core_input_storage=core_input_storage,
        core_output_storage=core_output_storage,
        core_storage=core_storage,
    ):
        if single_in:
            batch_shape = args[0].shape[:batch_ndim]
        else:
            _check_runtime_broadcast_core(args, batch_bcast_patterns, batch_ndim)
            batch_shape = broadcast_shapes(*(arg.shape[:batch_ndim] for arg in args))
            args = list(args)
            for i, arg in enumerate(args):
                if arg.shape[:batch_ndim] != batch_shape:
                    args[i] = faster_broadcast_to(
                        arg, batch_shape + arg.shape[batch_ndim:]
                    )

        ndindex_iterator = faster_ndindex(batch_shape)
        # Call once to get the output shapes
        try:
            # TODO: Pass core shape as input like BlockwiseWithCoreShape does?
            index0 = next(ndindex_iterator)
        except StopIteration:
            raise NotImplementedError("vectorize with zero size not implemented")
        else:
            for core_input, arg in zip(core_input_storage, args):
                core_input[0] = np.asarray(arg[index0])
            core_thunk()
            outputs = tuple(
                empty(batch_shape + core_output[0].shape, dtype=core_output[0].dtype)
                for core_output in core_output_storage
            )
            for output, core_output in zip(outputs, core_output_storage):
                output[index0] = core_output[0]

        for index in ndindex_iterator:
            for core_input, arg in zip(core_input_storage, args):
                core_input[0] = np.asarray(arg[index])
            core_thunk()
            for output, core_output in zip(outputs, core_output_storage):
                output[index] = core_output[0]

        # Clear storage
        for core_val in core_storage:
            core_val[0] = None
        return outputs

    return vectorized_perform


def _check_runtime_broadcast_core(numerical_inputs, batch_bcast_patterns, batch_ndim):
    # strict=None because we are in a hot loop
    # We zip together the dimension lengths of each input and their broadcast patterns
    for dim_lengths_and_bcast in zip(
        *[
            zip(input.shape[:batch_ndim], batch_bcast_pattern)
            for input, batch_bcast_pattern in zip(
                numerical_inputs, batch_bcast_patterns
            )
        ],
    ):
        # If for any dimension where an entry has dim_length != 1,
        # and another a dim_length of 1 and broadcastable=False, we have runtime broadcasting.
        if (
            any(d != 1 for d, _ in dim_lengths_and_bcast)
            and (1, False) in dim_lengths_and_bcast
        ):
            raise ValueError(
                "Runtime broadcasting not allowed. "
                "At least one input has a distinct batch dimension length of 1, but was not marked as broadcastable.\n"
                "If broadcasting was intended, use `specify_broadcastable` on the relevant input."
            )


class Blockwise(COp):
    """Generalizes a core `Op` to work with batched dimensions.

    TODO: C implementation?
    TODO: Fuse Blockwise?
    """

    __props__ = ("core_op", "signature")

    def __init__(
        self,
        core_op: Op,
        signature: str | None = None,
        name: str | None = None,
        gufunc_spec: tuple[str, int, int] | None = None,
        destroy_map=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        core_op
            An instance of a subclass of `Op` which works on the core case.
        signature
            Generalized universal function signature,
            e.g., (m,n),(n)->(m) for vectorized matrix-vector multiplication
        gufunc: tuple, Optional
            Tuple containing:
                1. String import path for a numpy/scipy function (e.g., "numpy.matmul", "scipy.special.softmax")
                that implements the blockwised operation of the scalar op.
                2 Number of inputs of the function
                3 Number of outputs of the function
        """
        if isinstance(core_op, Blockwise):
            raise TypeError("Core Op is already a Blockwise")

        if signature is None:
            signature = getattr(core_op, "gufunc_signature", None)
            if signature is None:
                raise ValueError(
                    f"Signature not provided nor found in core_op {core_op}"
                )

        self.core_op = core_op
        self.signature = signature
        self.name = name
        self.inputs_sig, self.outputs_sig = _parse_gufunc_signature(signature)
        self.gufunc_spec = gufunc_spec
        if destroy_map is not None:
            self.destroy_map = destroy_map
        if self.destroy_map != core_op.destroy_map:
            # Note: Should be fine for destroy_map of Blockwise to be more extensive than that of core_op
            # But we are not using that anywhere yet, so this check is fine for now
            raise ValueError(
                f"Blockwise destroy_map {self.destroy_map} must be the same as that of the core_op {core_op} {core_op.destroy_map}"
            )

        super().__init__(**kwargs)

    @overload
    def _create_dummy_core_node(
        self,
        inputs: Sequence[TensorVariable],
        *,
        propagate_unbatched_core_inputs: bool = False,
        return_dummy_inputs: Literal[False] = ...,
    ) -> Apply: ...

    @overload
    def _create_dummy_core_node(
        self,
        inputs: Sequence[TensorVariable],
        *,
        propagate_unbatched_core_inputs: bool = False,
        return_dummy_inputs: Literal[True] = ...,
    ) -> tuple[Apply, list[TensorVariable]]: ...

    def _create_dummy_core_node(
        self,
        inputs: Sequence[TensorVariable],
        *,
        propagate_unbatched_core_inputs: bool = False,
        return_dummy_inputs: bool = False,
    ) -> Apply | tuple[Apply, list[TensorVariable]]:
        core_inputs = []
        core_dummy_inputs = []
        for i, (inp, sig) in enumerate(zip(inputs, self.inputs_sig, strict=True)):
            if inp.type.ndim < len(sig):
                raise ValueError(
                    f"Input {i} {inp} has insufficient core dimensions for signature {self.signature}"
                )
            # ndim_supp = 0 case
            inp_ndim = inp.type.ndim
            batch_ndim = inp_ndim - len(sig)
            core_shape = inp.type.shape[batch_ndim:]
            if propagate_unbatched_core_inputs and all(
                inp.type.broadcastable[:batch_ndim]
            ):
                core_inputs.append(_squeeze_left(inp, batch_ndim))
            else:
                dummy_inp = tensor(dtype=inp.type.dtype, shape=core_shape)
                core_inputs.append(dummy_inp)
                core_dummy_inputs.append(dummy_inp)

        core_node = self.core_op.make_node(*core_inputs)

        if len(core_node.outputs) != len(self.outputs_sig):
            raise ValueError(
                f"Insufficient number of outputs for signature {self.signature}: {len(core_node.outputs)}"
            )
        for i, (core_out, sig) in enumerate(
            zip(core_node.outputs, self.outputs_sig, strict=True)
        ):
            if core_out.type.ndim != len(sig):
                raise ValueError(
                    f"Output {i} of {self.core_op} has wrong number of core dimensions for signature {self.signature}: {core_out.type.ndim}"
                )

        if return_dummy_inputs:
            return core_node, core_dummy_inputs

        return core_node

    def make_node(self, *inputs):
        inputs = [as_tensor_variable(i) for i in inputs]

        core_node = self._create_dummy_core_node(inputs)

        batch_ndims = max(
            inp.type.ndim - len(sig)
            for inp, sig in zip(inputs, self.inputs_sig, strict=True)
        )

        batched_inputs = []
        batch_shapes = []
        for i, (inp, sig) in enumerate(zip(inputs, self.inputs_sig, strict=True)):
            # Append missing dims to the left
            missing_batch_ndims = batch_ndims - (inp.type.ndim - len(sig))
            if missing_batch_ndims:
                inp = shape_padleft(inp, missing_batch_ndims)
            batched_inputs.append(inp)

            if not sig:
                batch_shapes.append(inp.type.shape)
            else:
                batch_shapes.append(inp.type.shape[: -len(sig)])

        try:
            batch_shape = tuple(
                broadcast_static_dim_lengths(batch_dims)
                for batch_dims in zip(*batch_shapes, strict=True)
            )
        except ValueError:
            raise ValueError(
                f"Incompatible Blockwise batch input shapes {[inp.type.shape for inp in inputs]}"
            )

        batched_outputs = [
            tensor(dtype=core_out.type.dtype, shape=batch_shape + core_out.type.shape)
            for core_out in core_node.outputs
        ]

        return Apply(self, batched_inputs, batched_outputs)

    def batch_ndim(self, node: Apply) -> int:
        return cast(int, node.outputs[0].type.ndim - len(self.outputs_sig[0]))

    def infer_shape(
        self, fgraph, node, input_shapes
    ) -> list[tuple[TensorVariable, ...]]:
        from pytensor.tensor import broadcast_shape
        from pytensor.tensor.shape import Shape_i

        batch_ndims = self.batch_ndim(node)
        core_dims: dict[str, Any] = {}
        batch_shapes = [input_shape[:batch_ndims] for input_shape in input_shapes]
        for input_shape, sig in zip(input_shapes, self.inputs_sig, strict=True):
            core_shape = input_shape[batch_ndims:]

            for core_dim, dim_name in zip(core_shape, sig, strict=True):
                prev_core_dim = core_dims.get(core_dim)
                if prev_core_dim is None:
                    core_dims[dim_name] = core_dim
                # Prefer constants
                elif not isinstance(prev_core_dim, Constant):
                    core_dims[dim_name] = core_dim

        batch_shape = broadcast_shape(*batch_shapes, arrays_are_shapes=True)

        def extract_core_shape_from_infer_shape():
            # Try to extract the core shapes from the core_op
            core_op_infer_shape = getattr(self.core_op, "infer_shape", None)
            if core_op_infer_shape is None:
                return [[None] * out.ndim for out in node.outputs]

            dummy_core_node, dummy_core_inputs = self._create_dummy_core_node(
                node.inputs,
                return_dummy_inputs=True,
                propagate_unbatched_core_inputs=True,
            )
            dummy_fgraph = FunctionGraph(outputs=dummy_core_node.outputs, clone=False)
            core_input_shapes = [
                input_shape[batch_ndims:] for input_shape in input_shapes
            ]
            core_output_shapes = core_op_infer_shape(
                dummy_fgraph, dummy_core_node, core_input_shapes
            )

            if not dummy_core_inputs:
                # All inputs are unbatched, so the core_shape can be used as is
                return core_output_shapes
            else:
                # Set to None those core_shapes that depend on dummy_core_inputs,
                # meaning their value may not be constant across batch dims of the Blockwise
                set_dummy_core_inputs = set(dummy_core_inputs)
                safe_core_output_shapes = [list(shape) for shape in core_output_shapes]
                for core_out_shape in safe_core_output_shapes:
                    for o, core_out_dim in enumerate(core_out_shape):
                        if (
                            # Some Ops return integers / literals from infer_shape...
                            # If it's not a Variable it can't depend on inputs
                            isinstance(core_out_dim, Variable)
                            and set_dummy_core_inputs
                            & set(explicit_graph_inputs([core_out_dim]))
                        ):
                            core_out_shape[o] = None

                return safe_core_output_shapes

        safe_core_out_shape = None

        out_shapes = []
        for o, (output, sig) in enumerate(
            zip(node.outputs, self.outputs_sig, strict=True)
        ):
            core_out_shape = []
            for i, dim_name in enumerate(sig):
                if dim_name in core_dims:
                    # The output dim is the same as another input dim
                    core_out_shape.append(core_dims[dim_name])
                elif str.isnumeric(dim_name):
                    # The core_dim has a constant size
                    from pytensor.tensor.basic import constant

                    core_out_shape.append(
                        constant(np.array(int(dim_name), dtype="int64"))
                    )
                else:
                    if safe_core_out_shape is None:
                        # Extract the core shape from the core_op infer_shape on demand
                        # For many Ops we never need to do this, because all info is in their signature
                        safe_core_out_shape = extract_core_shape_from_infer_shape()
                    if (core_out_dim := safe_core_out_shape[o][i]) is not None:
                        core_out_shape.append(core_out_dim)
                    else:
                        # Fallback shape requires evaluating the Blockwise Op
                        core_out_shape.append(Shape_i(batch_ndims + i)(output))
            out_shapes.append((*batch_shape, *core_out_shape))

        return out_shapes

    def connection_pattern(self, node):
        if hasattr(self.core_op, "connection_pattern"):
            return self.core_op.connection_pattern(node)

        return [[True for _ in node.outputs] for _ in node.inputs]

    def L_op(self, inputs, outputs, output_gradients):
        batch_ndim = self.batch_ndim(outputs[0].owner)

        # Obtain core_op gradients
        with config.change_flags(compute_test_value="off"):
            core_inputs = [
                tensor(
                    dtype=inp.type.dtype,
                    shape=inp.type.shape[batch_ndim:],
                )
                for inp in inputs
            ]
            core_outputs = self._create_dummy_core_node(core_inputs).outputs

            # Define core output_gradients, but keep original disconnected/null output_gradients (if any)
            core_output_gradients = [
                output_grad
                if isinstance(output_grad.type, NullType | DisconnectedType)
                else core_output.type()
                for output_grad, core_output in zip(
                    output_gradients, core_outputs, strict=True
                )
            ]

            core_input_gradients = self.core_op.L_op(
                core_inputs, core_outputs, core_output_gradients
            )

        # Vectorize core gradients to original inputs
        input_gradients = list(
            vectorize_graph(
                core_input_gradients,
                replace=dict(
                    zip(
                        core_inputs + core_outputs + core_output_gradients,
                        inputs + outputs + output_gradients,
                        strict=True,
                    )
                ),
            )
        )

        # Sum out the broadcasted batch dimensions
        batch_shape = outputs[0].type.shape[:batch_ndim]
        for i, (inp, sig) in enumerate(zip(inputs, self.inputs_sig, strict=True)):
            if isinstance(input_gradients[i].type, NullType | DisconnectedType):
                continue

            assert inp.type.ndim == batch_ndim + len(sig)

            if to_sum := [
                j
                for j, (inp_s, out_s) in enumerate(
                    zip(inp.type.shape, batch_shape, strict=False)
                )
                if inp_s == 1 and out_s != 1
            ]:
                input_gradients[i] = input_gradients[i].sum(axis=to_sum, keepdims=True)

        return input_gradients

    def _create_node_gufunc(self, node: Apply, impl) -> Callable:
        """Define (or retrieve) the node gufunc used in `perform`.

        If the Blockwise or core_op have a `gufunc_spec`, the relevant numpy or scipy gufunc is used directly.
        Otherwise, we default to `np.vectorize` of the core_op `perform` method for a dummy node.

        The gufunc is stored in the tag of the node.
        """
        batch_ndim = self.batch_ndim(node)
        batch_bcast_patterns = [
            inp.type.broadcastable[:batch_ndim] for inp in node.inputs
        ]
        if (
            gufunc_spec := self.gufunc_spec
            or getattr(self.core_op, "gufunc_spec", None)
        ) is not None:
            core_func = import_func_from_string(gufunc_spec[0])
            if core_func is None:
                raise ValueError(f"Could not import gufunc {gufunc_spec[0]} for {self}")

            if len(node.outputs) == 1:

                def gufunc(
                    *inputs,
                    batch_bcast_patterns=batch_bcast_patterns,
                    batch_ndim=batch_ndim,
                ):
                    _check_runtime_broadcast_core(
                        inputs, batch_bcast_patterns, batch_ndim
                    )
                    return (core_func(*inputs),)
            else:

                def gufunc(
                    *inputs,
                    batch_bcast_patterns=batch_bcast_patterns,
                    batch_ndim=batch_ndim,
                ):
                    _check_runtime_broadcast_core(
                        inputs, batch_bcast_patterns, batch_ndim
                    )
                    return core_func(*inputs)
        else:
            core_node = self._create_dummy_core_node(
                cast(list[TensorVariable], node.inputs),
                propagate_unbatched_core_inputs=True,
            )
            gufunc = _vectorize_node_perform(
                core_node,
                batch_bcast_patterns=batch_bcast_patterns,
                batch_ndim=self.batch_ndim(node),
                impl=impl,
            )

        return gufunc

    def _check_runtime_broadcast(self, node, inputs):
        batch_ndim = self.batch_ndim(node)
        batch_bcast = [pt_inp.type.broadcastable[:batch_ndim] for pt_inp in node.inputs]
        _check_runtime_broadcast_core(inputs, batch_bcast, batch_ndim)

    def prepare_node(self, node, storage_map, compute_map, impl=None):
        node.tag.gufunc = self._create_node_gufunc(node, impl=impl)

    def perform(self, node, inputs, output_storage):
        try:
            gufunc = node.tag.gufunc
        except AttributeError:
            gufunc = node.tag.gufunc = self._create_node_gufunc(node, impl=None)
        for out_storage, result in zip(output_storage, gufunc(*inputs)):
            out_storage[0] = result

    def __str__(self):
        if self.name is None:
            return f"{type(self).__name__}{{{self.core_op}, {self.signature}}}"
        else:
            return self.name

    def c_code(self, *args, **kwargs):
        # Blockwise is a C_Op just so we can propagate compilation mode to the inner Op.
        # It doesn't itself have a C implementation yet.
        raise NotImplementedError()

    def c_code_cache_version(self):
        return (-1,)


@_vectorize_node.register(Op)
def vectorize_node_fallback(op: Op, node: Apply, *bached_inputs) -> Apply:
    for inp in node.inputs:
        if not isinstance(inp.type, TensorType | ScalarType):
            raise NotImplementedError(
                f"Cannot vectorize node {node} with input {inp} of type {inp.type}"
            )

    if hasattr(op, "gufunc_signature"):
        signature = op.gufunc_signature
    else:
        # TODO: This is pretty bad for shape inference and merge optimization!
        #  Should get better as we add signatures to our Ops
        signature = safe_signature(
            [inp.type.ndim for inp in node.inputs],
            [out.type.ndim for out in node.outputs],
        )
    return cast(Apply, Blockwise(op, signature=signature).make_node(*bached_inputs))


_vectorize_node.register(Blockwise, _vectorize_not_needed)


class OpWithCoreShape(OpFromGraph):
    """Generalizes an `Op` to include core shape as an additional input."""

    def __init__(self, *args, on_unused_input="ignore", **kwargs):
        # We set on_unused_inputs="ignore" so that we can easily wrap nodes with repeated inputs
        # In this case the subsequent appearance of repeated inputs get disconnected in the inner graph
        # I can't think of a scenario where this will backfire, but if there's one
        # I bet on inplacing operations (time will tell)
        return super().__init__(*args, on_unused_input=on_unused_input, **kwargs)


class BlockwiseWithCoreShape(OpWithCoreShape):
    """Generalizes a Blockwise `Op` to include a core shape parameter."""

    @property
    def core_op(self):
        [blockwise_node] = self.fgraph.apply_nodes
        return cast(Blockwise, blockwise_node.op).core_op

    def __str__(self):
        [blockwise_node] = self.fgraph.apply_nodes
        return f"[{blockwise_node.op!s}]"
