"""Expose native scalar values for Numba graph compilation."""

from pytensor.compile.builders import FinalizedOpFromGraph
from pytensor.graph.rewriting.basic import GraphRewriter, copy_stack_trace, dfs_rewriter
from pytensor.graph.utils import InconsistencyError
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.basic import (
    Join,
    MakeVector,
    ScalarFromTensor,
    TensorFromScalar,
    atleast_Nd,
    join,
    scalar_from_tensor,
    tensor_from_scalar,
)
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.rewriting.basic import (
    local_scalar_tensor_scalar,
    local_tensor_scalar_tensor,
)
from pytensor.tensor.rewriting.fused_elemwise import (
    FusedElemwise,
    fused_elemwise_optdb,
)
from pytensor.tensor.rewriting.subtensor import local_subtensor_merge_integer
from pytensor.tensor.subtensor import IncSubtensor, Subtensor


class ScalarCAReduce(FinalizedOpFromGraph):
    """A full reduction with a native scalar output."""

    def __str__(self):
        [node] = [n for n in self.fgraph.apply_nodes if isinstance(n.op, CAReduce)]
        return f"Scalar{node.op}"


class ScalarSubtensor(FinalizedOpFromGraph):
    """A Subtensor read with a native scalar output."""

    def __str__(self):
        [node] = [n for n in self.fgraph.apply_nodes if isinstance(n.op, Subtensor)]
        return f"Scalar{node.op}"


class ScalarizedIncSubtensor(FinalizedOpFromGraph):
    """A basic indexed update with a native scalar update value."""

    def __str__(self):
        [node] = [n for n in self.fgraph.apply_nodes if isinstance(n.op, IncSubtensor)]
        return f"Scalarized{node.op}"


class ScalarizedJoin(FinalizedOpFromGraph):
    """A vector Join with some native ScalarType inputs."""

    def __str__(self):
        return "ScalarizedJoin"


class ScalarizedElemwise(FinalizedOpFromGraph):
    """An ``Elemwise`` with some native ``ScalarType`` inputs."""

    def __str__(self):
        [node] = [n for n in self.fgraph.apply_nodes if isinstance(n.op, Elemwise)]
        return f"Scalarized{node.op}"


class ScalarizeRewriter(GraphRewriter):
    """Pass scalar values between Ops without allocating intermediate size-1 arrays.

    Start scalar computations for clients that can accept scalars, or continue
    from existing scalar inputs. Leave isolated producers unchanged.
    Elemwise uses its ScalarOp for scalar computations. TensorFromScalar preserves
    array clients of the same value.

    Runs after FuseElemwise and updates FusedElemwise inputs and outputs.
    Preserves array inplace buffers.
    """

    scalar_consumers = (MakeVector, Elemwise, ScalarFromTensor)

    def apply(self, fgraph):
        for node in fgraph.toposort():
            if node not in fgraph.apply_nodes:
                continue
            op = node.op
            outputs = None
            if isinstance(op, Subtensor) and all(node.outputs[0].type.broadcastable):
                outputs = self.scalarize_subtensor(fgraph, node)
            elif isinstance(op, CAReduce) and all(node.outputs[0].type.broadcastable):
                outputs = self.scalarize_careduce(fgraph, node)
            elif isinstance(op, FusedElemwise):
                outputs = self.scalarize_fused(fgraph, node)
            elif isinstance(op, Elemwise):
                outputs = self.scalarize_elemwise(fgraph, node)
            elif isinstance(op, IncSubtensor) and all(
                node.inputs[1].type.broadcastable
            ):
                outputs = self.scalarize_incsubtensor(node)
            elif isinstance(op, MakeVector) or (
                isinstance(op, Join) and op.axis == 0 and node.outputs[0].ndim == 1
            ):
                outputs = self.scalarize_join(node)
            elif isinstance(op, ScalarFromTensor):
                scalar = self.scalar_behind(node.inputs[0])
                if scalar is not None:
                    outputs = [scalar]
            if outputs is None:
                continue
            replacements = list(zip(node.outputs, outputs, strict=True))
            for old, new in replacements:
                copy_stack_trace(old, new)
            try:
                fgraph.replace_all_validate(replacements, reason="scalarize")
            except InconsistencyError:
                continue

    @staticmethod
    def scalar_behind(var):
        """Find the scalar inside TensorFromScalar, through DimShuffle views, or None."""
        while isinstance(var.owner_op, DimShuffle):
            var = var.owner.inputs[0]
        if isinstance(var.owner_op, TensorFromScalar):
            return var.owner.inputs[0]
        return None

    @staticmethod
    def has_scalar_consumer(fgraph, var):
        for client, input_index in fgraph.clients[var]:
            if isinstance(client.op, DimShuffle):
                if ScalarizeRewriter.has_scalar_consumer(fgraph, client.outputs[0]):
                    return True
            elif isinstance(client.op, ScalarizeRewriter.scalar_consumers):
                return True
            elif isinstance(client.op, Join):
                if client.op.axis == 0 and client.outputs[0].ndim == 1:
                    return True
            elif isinstance(client.op, FusedElemwise):
                if input_index in client.op.scalarizable_inputs:
                    return True
            elif isinstance(client.op, IncSubtensor):
                if input_index == 1 and all(client.inputs[1].type.broadcastable):
                    return True
        return False

    @staticmethod
    def scalarize_subtensor(fgraph, node):
        """Return a single-element indexed read as a native scalar."""
        if not ScalarizeRewriter.has_scalar_consumer(fgraph, node.outputs[0]):
            return None
        original_outputs = node.outputs
        if node.outputs[0].ndim:
            # Merge away the singleton view so the wrapper keeps one scalar read.
            indexed = node.outputs[0][(0,) * node.outputs[0].ndim]
            merged = local_subtensor_merge_integer.transform(fgraph, indexed.owner)
            if not merged or merged[0].owner.inputs[0] is not node.inputs[0]:
                return None
            node = merged[0].owner
        inner_inputs = [inp.type() for inp in node.inputs]
        inner_output = scalar_from_tensor(node.op(*inner_inputs))
        scalar = ScalarSubtensor(inner_inputs, [inner_output])(*node.inputs)
        return ScalarizeRewriter.tensor_outputs(original_outputs, [scalar])

    @staticmethod
    def scalarize_careduce(fgraph, node):
        """Return a single-element reduction as a native scalar."""
        if not ScalarizeRewriter.has_scalar_consumer(fgraph, node.outputs[0]):
            return None
        [x] = node.inputs
        inner_input = x.type()
        inner_output = scalar_from_tensor(node.op.clone(axis=None)(inner_input))
        scalar = ScalarCAReduce([inner_input], [inner_output])(x)
        return ScalarizeRewriter.tensor_outputs(node.outputs, [scalar])

    @staticmethod
    def scalarize_elemwise(fgraph, node):
        """Use ScalarOps for scalar arithmetic and ScalarizedElemwise for array outputs."""
        scalars = [ScalarizeRewriter.scalar_behind(inp) for inp in node.inputs]
        has_scalar_input = any(scalar is not None for scalar in scalars)

        # Only real array destinations need to retain an inplace relationship.
        inplace_pattern = {
            out: inp
            for out, inp in node.op.inplace_pattern.items()
            if scalars[inp] is None
        }
        all_scalar = all(all(inp.type.broadcastable) for inp in node.inputs)
        if all_scalar and not inplace_pattern:
            if not has_scalar_input and not any(
                ScalarizeRewriter.has_scalar_consumer(fgraph, out)
                for out in node.outputs
            ):
                return None
            scalar_inputs = []
            for inp, scalar in zip(node.inputs, scalars, strict=True):
                if scalar is None:
                    squeezed = copy_stack_trace(inp, inp.squeeze())
                    scalar = copy_stack_trace(inp, scalar_from_tensor(squeezed))
                scalar_inputs.append(scalar)
            outputs = node.op.scalar_op(*scalar_inputs, return_list=True)
            return ScalarizeRewriter.tensor_outputs(node.outputs, outputs)

        if not has_scalar_input:
            return None
        outer_inputs = []
        inner_inputs = []
        tensor_inputs = []
        for inp, scalar in zip(node.inputs, scalars, strict=True):
            if scalar is None:
                inner = inp.type()
                outer_inputs.append(inp)
                tensor_inputs.append(inner)
            else:
                inner = scalar.type()
                outer_inputs.append(scalar)
                tensor_inputs.append(atleast_Nd(tensor_from_scalar(inner), n=inp.ndim))
            inner_inputs.append(inner)
        elemwise = Elemwise(node.op.scalar_op, inplace_pattern=inplace_pattern)
        inner_outputs = elemwise(*tensor_inputs, return_list=True)
        scalarized = ScalarizedElemwise(
            inner_inputs, inner_outputs, destroy_map=elemwise.destroy_map
        )
        return scalarized(*outer_inputs, return_list=True)

    @staticmethod
    def scalarize_fused(fgraph, node):
        scalars = {
            i: scalar
            for i in node.op.scalarizable_inputs
            if (scalar := ScalarizeRewriter.scalar_behind(node.inputs[i])) is not None
        }
        scalar_outputs = {
            i
            for i, spec in enumerate(node.op.reduced_outputs)
            if spec is not None
            and not isinstance(node.outputs[i].type, ScalarType)
            and all(node.outputs[i].type.broadcastable)
            and (
                scalars
                or ScalarizeRewriter.has_scalar_consumer(fgraph, node.outputs[i])
            )
        }
        if not scalars and not scalar_outputs:
            return None
        op = node.op.with_scalar_io(set(scalars), scalar_outputs)
        inputs = [scalars.get(i, inp) for i, inp in enumerate(node.inputs)]
        outputs = op(*inputs, return_list=True)
        return ScalarizeRewriter.tensor_outputs(node.outputs, outputs)

    @staticmethod
    def scalarize_incsubtensor(node):
        """Pass scalar updates directly while retaining the array destination."""
        x, value, *indices = node.inputs
        target_ndim = x.ndim - sum(isinstance(idx, int) for idx in node.op.idx_list)
        if value.ndim > target_ndim:
            return None
        scalar = ScalarizeRewriter.scalar_behind(value)
        if scalar is None:
            scalar = scalar_from_tensor(value.squeeze())
        outer_inputs = [x, scalar, *indices]
        inner_inputs = [inp.type() for inp in outer_inputs]
        inner_output = node.op(
            inner_inputs[0],
            atleast_Nd(tensor_from_scalar(inner_inputs[1]), n=value.ndim),
            *inner_inputs[2:],
        )
        scalarized = ScalarizedIncSubtensor(
            inner_inputs, [inner_output], destroy_map=node.op.destroy_map
        )
        return [scalarized(*outer_inputs)]

    @staticmethod
    def scalarize_join(node):
        """Let MakeVector and vector Join consume scalars directly."""
        scalars = [ScalarizeRewriter.scalar_behind(inp) for inp in node.inputs]
        if all(scalar is None for scalar in scalars):
            return None
        outer_inputs = []
        inner_inputs = []
        tensor_inputs = []
        for inp, scalar in zip(node.inputs, scalars, strict=True):
            if scalar is None and inp.ndim == 0:
                scalar = copy_stack_trace(inp, scalar_from_tensor(inp))
            if scalar is None:
                inner = inp.type()
                outer_inputs.append(inp)
                tensor_inputs.append(inner)
            else:
                inner = scalar.type()
                outer_inputs.append(scalar)
                tensor_inputs.append(tensor_from_scalar(inner)[None])
            inner_inputs.append(inner)
        scalarized = ScalarizedJoin(inner_inputs, [join(0, *tensor_inputs)])
        return [scalarized(*outer_inputs)]

    @staticmethod
    def tensor_outputs(original_outputs, scalars):
        """Wrap scalar outputs for clients that still need arrays."""
        outputs = []
        for original, scalar in zip(original_outputs, scalars, strict=True):
            copy_stack_trace(original, scalar)
            if isinstance(scalar.type, ScalarType) and not isinstance(
                original.type, ScalarType
            ):
                tensor = copy_stack_trace(original, tensor_from_scalar(scalar))
                outputs.append(atleast_Nd(tensor, n=original.ndim))
            else:
                outputs.append(scalar)
        return outputs


fused_elemwise_optdb.register("scalarize", ScalarizeRewriter(), "numba", position=1)
fused_elemwise_optdb.register(
    "remove_scalar_tensor_boundaries",
    dfs_rewriter(
        local_scalar_tensor_scalar, local_tensor_scalar_tensor, ignore_newtrees=True
    ),
    "numba",
    position=2,
)
