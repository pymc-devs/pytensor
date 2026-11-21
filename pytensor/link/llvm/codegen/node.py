"""The function of one node: unpacks its arguments into what its emitter gets
(interface.py), and calls the emitter, or the kernel that implements the node."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pytensor.link.llvm.codegen.destination import (
    OutputDestination,
    ProvidedDestination,
    ScalarDestination,
    SlotDestination,
)
from pytensor.link.llvm.codegen.emit import Codegen, c_array, constant, store, unpack
from pytensor.link.llvm.codegen.interface import (
    EmitterInput,
    EmitterOutput,
    InPlaceOutput,
    InPlaceUpdate,
    InputView,
    InputViewOutput,
    NewBufferOutput,
    ScalarOutput,
    ViewOrCopy,
    ViewOrCopyOutput,
    llvm_emit,
)
from pytensor.link.llvm.codegen.signature import (
    Output,
    Passing,
    Signature,
    node_signature,
    skipped_inputs,
)
from pytensor.link.llvm.codegen.types import (
    ScalarRepr,
    TensorRepr,
    ir,
    is_absent,
    memory_type,
    repr_of,
)


if TYPE_CHECKING:
    from pytensor.link.llvm.codegen.call import LinkTable


@dataclass
class NodeFunction:
    function: ir.Function
    signature: Signature
    #: its link table (call.py): the config to call it with, or None
    links: LinkTable | None = None
    #: of a graph's function: the arena ({vid: offset} of the temporary values in it,
    #: its size in bytes)
    arena: tuple = ({}, 0)
    #: of a graph's function: the vids of all its temporary values (in the arena or
    #: on the stack)
    temporaries: frozenset = frozenset()
    #: of a graph's function: where its array constants are in the link table
    #: ({vid: entry})
    constants: dict = field(default_factory=dict)


def debug_name(node) -> str:
    """The name of the function of `node` in debug information: the node with the names
    of its inputs (shortened). A compiled unit is named after the nodes inside it."""
    from pytensor.compile.builders import Execution

    if getattr(node.op, "execution", None) is Execution.COMPILED:
        text = "unit: " + ", ".join(str(n) for n in node.op.fgraph.toposort())
    else:
        text = str(node)
    return text if len(text) <= 200 else text[:197] + "..."


def build_node_function(
    module,
    node,
    name: str,
    scratch_outputs=frozenset(),
    folded=None,
    input_layouts=None,
    provided=None,
    parallel=True,
    kernel=None,
    links=None,
    planner=None,
    writable=frozenset(),
    check_layouts=False,
) -> NodeFunction:
    """Build the LLVM function of one node, always inlined.

    * `scratch_outputs`: positions of the outputs that are temporaries of a compiled
      unit: they go into the unit's scratch memory instead of runtime buffers;
    * `folded`: {input position: value} for 0-d constant inputs compiled into the code;
    * `input_layouts`, `provided`, `writable`: see `node_signature`;
    * `kernel`: a `Kernel` implementing the node (a Rust or numba kernel). The function
      calls it (call.py) through an entry of `links`, the link table that is the
      config of the kernel being generated. Code nested in the node (a Scan's loop
      body, a Blockwise's core) links its kernels there too;
    * `planner`: decides how the nodes of inner graphs run (a Scan's loop body, a
      Blockwise's core: `Codegen.graph_function`);
    * `check_layouts`: verify views' layouts at run time (`Codegen.check_layouts`).
    """
    folded = folded or {}
    signature = node_signature(
        node,
        frozenset(folded),
        input_layouts,
        provided,
        kernel.outputs if kernel is not None else None,
        writable,
    )
    function = module.function(
        name,
        signature.param_types(),
        ir.i32,
        inline=ir.Inline.Always,
        debug_name=debug_name(node),
    )
    builder = function.builder()
    params = iter(function.params)
    ctx, config = next(params), next(params)
    codegen = Codegen(
        builder,
        ctx,
        config,
        parallel=parallel,
        check_layouts=check_layouts,
        links=links,
        planner=planner,
        module=module,
        name=name,
    )
    scopes = _AliasScopes(builder, name, node, signature, folded, writable)
    inputs = _inputs(codegen, node, signature, folded, writable, params, scopes)
    destinations = _destinations(codegen, signature, scratch_outputs, params, scopes)
    # (the op's scratch memory is freed before the function returns)
    with codegen.scratch_scope(function=True):
        if kernel is not None:
            from pytensor.link.llvm.codegen.call import call_destroying

            if links is None:
                raise ValueError("a node function calling a kernel needs a links")
            kernel_inputs = [
                (array, getattr(var.type, "dtype", None))
                for array, var in zip(inputs, node.inputs, strict=True)
            ]
            call_destroying(
                codegen, kernel, links.link_kernel(kernel), kernel_inputs, destinations
            )
        else:
            handles = [
                None if destination is None else _handle(destination, output, inputs)
                for destination, output in zip(
                    destinations, signature.outputs, strict=True
                )
            ]
            llvm_emit(node.op, node, codegen, inputs, handles)
            if not builder.terminated:
                for k, destination in enumerate(destinations):
                    if destination is not None:
                        try:
                            destination.finish()
                        except ValueError as e:
                            raise ValueError(f"{node.op}: output {k}: {e}") from None
    if not builder.terminated:
        builder.ret(0)
    return NodeFunction(function, signature, links)


class _AliasScopes:
    """What LLVM is told about aliasing. Every array the function writes (an output's
    new memory, an input it may overwrite) overlaps no other array of the function;
    inputs that are only read may overlap each other."""

    def __init__(self, builder, name, node, signature, folded, writable):
        self.builder = builder
        builder.alias_domain(name)
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

    def mark(self, data, scope: str) -> None:
        """Mark the memory at `data` as the array `scope`."""
        if scope in self.read_only:
            self.builder.mark_array(data, scope, self.written)
        else:
            self.builder.mark_array(data, scope, self.others(scope))

    def others(self, scope: str) -> list[str]:
        return self.read_only + [w for w in self.written if w != scope]


def _argument_positions(node, folded) -> list[int]:
    """Return the node input position of each argument."""
    skipped = skipped_inputs(node, folded)
    return [k for k in range(len(node.inputs)) if k not in skipped]


def _inputs(codegen, node, signature, folded, writable, params, scopes) -> list:
    """Return what the emitter gets for each input (interface.py)."""
    builder = codegen.b
    representations = iter(signature.inputs)
    inputs: list[EmitterInput] = []
    for position, var in enumerate(node.inputs):
        dtype: str = getattr(var.type, "dtype", "")
        if is_absent(var.type):
            inputs.append(None)
        elif position in folded:
            value = constant(builder, dtype, folded[position])
            if isinstance(repr_of(var.type), ScalarRepr):
                inputs.append(value)
            else:
                memory = builder.alloca(memory_type(dtype))
                store(builder, dtype, value, memory)
                inputs.append(c_array(dtype, memory, [], []))
        else:
            representation, param = next(representations), next(params)
            if isinstance(representation, ScalarRepr):
                inputs.append(param)
                continue
            array = unpack(builder, representation, param)
            array.base, array.writable = position, position in writable
            scopes.mark(array.data, f"i{position}")
            inputs.append(array)
    return inputs


def _destinations(
    codegen, signature: Signature, scratch_outputs, params, scopes
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
            destinations.append(ScalarDestination(codegen, param, representation.dtype))
        elif output.passing is Passing.CALLER_BUFFER:
            array = unpack(codegen.b, output.tensor, param)
            scopes.mark(array.data, f"o{j}")
            destinations.append(ProvidedDestination(codegen, array))
        else:
            destinations.append(
                SlotDestination.for_output(
                    codegen,
                    param,
                    output,
                    aliased_inputs=output.aliased_inputs,
                    in_scratch_memory=j in scratch_outputs,
                    alias_scopes=(f"o{j}", scopes.others(f"o{j}")),
                )
            )
    return destinations


def _handle(
    destination: OutputDestination, output: Output, inputs: list
) -> EmitterOutput:
    """Return what the emitter gets for `output`: the handle matching its storage."""
    if isinstance(destination, ScalarDestination):
        return ScalarOutput(destination)
    storage = output.storage
    layout = output.layout
    assert layout is not None
    match storage:
        case InputView():
            return InputViewOutput(destination, storage, layout)
        case ViewOrCopy():
            return ViewOrCopyOutput(destination, layout)
        case InPlaceUpdate(input=input):
            return InPlaceOutput(destination, inputs[input])
    return NewBufferOutput(destination)


__all__ = ["NodeFunction", "build_node_function", "debug_name"]
