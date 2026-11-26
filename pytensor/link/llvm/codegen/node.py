"""The function of one node: unpacks its arguments into what its emitter gets
(interface.py), and calls the emitter, or the kernel that implements the node."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pytensor.graph.basic import Apply
from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.codegen.array import Array
from pytensor.link.llvm.codegen.context import Codegen
from pytensor.link.llvm.codegen.destination import (
    OutputDestination,
    ProvidedDestination,
    ScalarDestination,
    SlotDestination,
)
from pytensor.link.llvm.codegen.implementation import GENERATED, Implementation
from pytensor.link.llvm.codegen.interface import (
    EmitterInput,
)
from pytensor.link.llvm.codegen.num import constant, store
from pytensor.link.llvm.codegen.signature import (
    Passing,
    Signature,
    node_signature,
    skipped_inputs,
)
from pytensor.link.llvm.codegen.types import (
    ScalarRepr,
    TensorRepr,
    element_type,
    ir,
    is_absent,
    memory_type,
    repr_of,
)


if TYPE_CHECKING:
    from pytensor.link.llvm.codegen.call import LinkTable
    from pytensor.link.llvm.codegen.kernel_module import KernelModule
    from pytensor.link.llvm.codegen.plan import ValueId


@dataclass
class NodeFunction:
    function: ir.Function
    signature: Signature
    #: its link table (call.py): the config to call it with, or None
    links: LinkTable | None = None
    #: of a graph's function: where its array constants are in the link table
    #: ({value id: entry})
    constants: dict[ValueId, int] = field(default_factory=dict)
    #: of a node's function: the outputs it puts into scratch memory of the code around
    #: it (the rest of its new outputs are runtime buffers)
    in_scratch: frozenset[int] = frozenset()


def _debug_name(node: Apply) -> str:
    """The name of the function of `node` in debug information: the node with the names
    of its inputs (shortened). A compiled unit is named after the nodes inside it."""
    from pytensor.compile.builders import Execution

    if getattr(node.op, "execution", None) is Execution.COMPILED:
        text = "unit: " + ", ".join(str(n) for n in node.op.fgraph.toposort())
    else:
        text = str(node)
    return text if len(text) <= 200 else text[:197] + "..."


def node_function(
    unit: KernelModule,
    name: str,
    node: Apply,
    implementation: Implementation,
    signature: Signature,
    scratch_outputs: Collection[int] = frozenset(),
    parallel: bool | None = None,
) -> NodeFunction:
    """Build the LLVM function of `node` in `unit`, always inlined: its parameters from
    `signature` (`node_signature` for `implementation`), its body written by
    `implementation`.

    * `scratch_outputs`: positions of the outputs that are temporaries of the code
      around it: they go into its scratch memory instead of runtime buffers, where the
      implementation allocates through codegen (`NodeFunction.in_scratch`);
    * `parallel`: may its loops run in parallel? (None: as `unit`'s options say; the
      code around it passes False inside a parallel loop).
    """
    function = unit.module.function(
        name,
        signature.param_types(),
        ir.i32,
        inline=ir.Inline.Always,
        debug_name=_debug_name(node),
    )
    b = function.builder()
    params = iter(function.params)
    ctx, config = next(params), next(params)
    cg = Codegen(b, ctx, config, unit, name=name, parallel=parallel)
    folded, writable = signature.folded_values, signature.writable
    in_scratch = (
        frozenset(scratch_outputs)
        if implementation.allocates_through_codegen
        else frozenset()
    )
    scopes = _AliasScopes(b, name, node, signature, folded, writable)
    inputs = _inputs(cg, node, signature, folded, writable, params, scopes)
    destinations = _destinations(cg, signature, in_scratch, params, scopes)
    # (the scratch memory taken inside is freed before the function returns)
    with cg.scratch_scope(function=True):
        # copies of the inputs it overwrites that aren't writable
        for position in sorted(implementation.always_overwrites - writable):
            original = inputs[position]
            if isinstance(original, Array):
                mine = cg.temporary(original.scalar_type, original.shape)
                mine.copy_from(cg, original)
                inputs[position] = mine
        implementation.emit(cg, node, signature, inputs, destinations)
    if not b.terminated:
        b.ret(0)
    return NodeFunction(function, signature, unit.links, in_scratch=in_scratch)


def build_node_function(
    unit: KernelModule,
    node: Apply,
    name: str,
    *,
    input_layouts: Sequence[Layout | None] | None = None,
    folded: Mapping[int, float] | None = None,
    writable: Collection[int] = frozenset(),
    provided: Sequence[Layout | None] | None = None,
) -> NodeFunction:
    """Build the function of `node`'s generated code (`GENERATED`) in `unit`, with the
    signature `node_signature` gives for these arguments (see there)."""
    signature = node_signature(
        node, input_layouts, folded=folded, writable=writable, provided=provided
    )
    return node_function(unit, name, node, GENERATED, signature)


class _AliasScopes:
    """What LLVM is told about aliasing. Every array the function writes (an output's
    new memory, an input it may overwrite) overlaps no other array of the function;
    inputs that are only read may overlap each other."""

    def __init__(
        self,
        b: ir.Builder,
        name: str,
        node: Apply,
        signature: Signature,
        folded: Collection[int],
        writable: Collection[int],
    ):
        self.b = b
        b.alias_domain(name)
        arguments = _argument_positions(node, folded)
        tensors = [
            position
            for position, representation in zip(
                arguments, signature.inputs, strict=True
            )
            if isinstance(representation, TensorRepr)
        ]
        self.read_only: list[str] = [f"i{k}" for k in tensors if k not in writable]
        self.written: list[str] = [f"i{k}" for k in tensors if k in writable] + [
            f"o{j}" for j, output in enumerate(signature.outputs) if output.is_tensor
        ]

    def mark(self, data: ir.Value, scope: str) -> None:
        """Mark the memory at `data` as the array `scope`."""
        if scope in self.read_only:
            self.b.mark_array(data, scope, self.written)
        else:
            self.b.mark_array(data, scope, self.others(scope))

    def others(self, scope: str) -> list[str]:
        return self.read_only + [w for w in self.written if w != scope]


def _argument_positions(node: Apply, folded: Collection[int]) -> list[int]:
    """Return the node input position of each argument."""
    skipped = skipped_inputs(node, folded)
    return [k for k in range(len(node.inputs)) if k not in skipped]


def _inputs(
    cg: Codegen,
    node: Apply,
    signature: Signature,
    folded: Mapping[int, float],
    writable: Collection[int],
    params: Iterator[ir.Value],
    scopes: _AliasScopes,
) -> list[EmitterInput]:
    """Return what the emitter gets for each input (interface.py)."""
    b = cg.b
    representations = iter(signature.inputs)
    inputs: list[EmitterInput] = []
    for position, var in enumerate(node.inputs):
        if is_absent(var.type):
            inputs.append(None)
        elif position in folded:
            scalar_type = element_type(var.type)
            value = constant(b, scalar_type, folded[position])
            if isinstance(repr_of(var.type), ScalarRepr):
                inputs.append(value)
            else:
                memory = b.alloca(memory_type(scalar_type))
                store(b, scalar_type, value, memory)
                inputs.append(Array.c(scalar_type, memory, [], []))
        else:
            representation, param = next(representations), next(params)
            if not isinstance(representation, TensorRepr):  # a ScalarType value
                inputs.append(param)
                continue
            array = Array.unpack(b, representation, param)
            array.base, array.writable = position, position in writable
            scopes.mark(array.data, f"i{position}")
            inputs.append(array)
    return inputs


def _destinations(
    cg: Codegen,
    signature: Signature,
    scratch_outputs: Collection[int],
    params: Iterator[ir.Value],
    scopes: _AliasScopes,
) -> list[OutputDestination | None]:
    """Return where each output goes, from its parameter."""
    destinations: list[OutputDestination | None] = []
    for j, output in enumerate(signature.outputs):
        if output.is_absent:
            destinations.append(None)
            continue
        param = next(params)
        representation = output.representation
        if isinstance(representation, ScalarRepr):
            destinations.append(
                ScalarDestination(cg, param, representation.scalar_type)
            )
        elif output.passing is Passing.CALLER_BUFFER:
            array = Array.unpack(cg.b, output.tensor, param)
            scopes.mark(array.data, f"o{j}")
            destinations.append(ProvidedDestination(cg, array))
        else:
            destinations.append(
                SlotDestination.for_output(
                    cg,
                    param,
                    output,
                    aliased_inputs=output.aliased_inputs,
                    in_scratch_memory=j in scratch_outputs,
                    alias_scopes=(f"o{j}", scopes.others(f"o{j}")),
                )
            )
    return destinations


__all__ = [
    "NodeFunction",
    "build_node_function",
    "node_function",
]
