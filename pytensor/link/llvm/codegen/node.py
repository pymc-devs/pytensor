"""The function of one node: unpacks its arguments into what its emitter gets
(interface.py), and calls the emitter, or the kernel that implements the node."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pytensor.link.llvm.codegen.emit import Codegen, c_array, constant, store, unpack
from pytensor.link.llvm.codegen.interface import (
    ProvidedOutput,
    ScalarOutput,
    SlotOutput,
    llvm_emit,
)
from pytensor.link.llvm.codegen.signature import (
    OutputMode,
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
    temporary_outputs=frozenset(),
    folded=None,
    in_layouts=None,
    provided=None,
    parallel=True,
    kernel=None,
    links=None,
    planner=None,
    writable=frozenset(),
) -> NodeFunction:
    """The LLVM function of one node, always inlined.

    * `temporary_outputs`: allocated outputs that are temporaries of a compiled unit;
      they go into temporary memory instead of runtime buffers;
    * `folded`: {input position: value} for 0-d constant inputs compiled into the code;
    * `kernel`: a `Kernel` implementing the node (a Rust kernel, a numba kernel):
      the function calls it (call.py), through an entry of `links`, the
      link table that is the config of the kernel being generated. Code nested in the
      node (a Scan's loop body, a Blockwise's core) links its kernels there too;
    * `planner`: decides how the nodes of inner graphs run (a Scan's loop body, a
      Blockwise's core: `Codegen.graph_function`);
    * `writable`: positions of the inputs the node may overwrite (`node_outputs`).
    """
    folded = folded or {}
    sig = node_signature(
        node,
        frozenset(folded),
        in_layouts,
        provided,
        kernel.outputs if kernel is not None else None,
        writable,
    )
    fn = module.function(
        name,
        sig.param_types(),
        ir.i32,
        inline=ir.Inline.Always,
        debug_name=debug_name(node),
    )
    b = fn.builder()
    params = iter(fn.params)
    ctx, config = next(params), next(params)
    cg = Codegen(
        b,
        ctx,
        config,
        parallel=parallel,
        links=links,
        planner=planner,
        module=module,
        name=name,
    )
    scopes = _AliasScopes(b, name, node, sig, folded, writable)
    inputs = _inputs(cg, node, sig, folded, writable, params, scopes)
    outputs = _outputs(cg, node, sig, folded, temporary_outputs, params, scopes)
    # (the op's scratch memory is freed before the function returns)
    with cg.scratch_scope(function=True):
        if kernel is not None:
            from pytensor.link.llvm.codegen.call import call_destroying

            if links is None:
                raise ValueError("a node function calling a kernel needs a links")
            ins = [
                (a, getattr(var.type, "dtype", None))
                for a, var in zip(inputs, node.inputs, strict=True)
            ]
            call_destroying(cg, kernel, links.link_kernel(kernel), ins, outputs)
        else:
            llvm_emit(node.op, node, cg, inputs, outputs)
            if not b.terminated:
                for k, out in enumerate(outputs):
                    if out is not None:
                        try:
                            out.finish()
                        except ValueError as e:
                            raise ValueError(f"{node.op}: output {k}: {e}") from None
    if not b.terminated:
        b.ret(0)
    return NodeFunction(fn, sig, links)


class _AliasScopes:
    """What LLVM is told about aliasing. Every array the function writes (an output's
    new memory, an input it may overwrite) overlaps no other array of the function;
    inputs that are only read may overlap each other."""

    def __init__(self, b, name, node, sig, folded, writable):
        self.b = b
        b.alias_domain(name)
        args = _argument_positions(node, folded)
        tensors = [
            pos
            for pos, r in zip(args, sig.inputs, strict=True)
            if isinstance(r, TensorRepr)
        ]
        self.read_only = [f"i{pos}" for pos in tensors if pos not in writable]
        self.written = [f"i{pos}" for pos in tensors if pos in writable] + [
            f"o{j}"
            for j, o in enumerate(sig.outputs)
            if o.mode in (OutputMode.PROVIDED, OutputMode.RETURNED)
        ]

    def mark(self, data, scope):
        """Mark the memory at `data` as the array `scope`."""
        if scope in self.read_only:
            self.b.mark_array(data, scope, self.written)
        else:
            self.b.mark_array(data, scope, self.others(scope))

    def others(self, scope):
        return self.read_only + [w for w in self.written if w != scope]


def _argument_positions(node, folded) -> list[int]:
    """The node input each argument is."""
    skipped = skipped_inputs(node, folded)
    return [pos for pos in range(len(node.inputs)) if pos not in skipped]


def _inputs(cg, node, sig, folded, writable, params, scopes) -> list:
    """What the emitter gets for each input (interface.py)."""
    b = cg.b
    reprs = iter(sig.inputs)
    inputs = []
    for pos, var in enumerate(node.inputs):
        dtype = getattr(var.type, "dtype", None)
        if is_absent(var.type):
            inputs.append(None)
        elif pos in folded:
            value = constant(b, dtype, folded[pos])
            if isinstance(repr_of(var.type), ScalarRepr):
                inputs.append(value)
            else:
                slot = b.alloca(memory_type(dtype))
                store(b, dtype, value, slot)
                inputs.append(c_array(dtype, slot, [], []))
        else:
            r, v = next(reprs), next(params)
            if isinstance(r, ScalarRepr):
                inputs.append(v)
                continue
            a = unpack(b, r, v)
            a.base, a.writable = pos, pos in writable
            scopes.mark(a.data, f"i{pos}")
            inputs.append(a)
    return inputs


def _outputs(cg, node, sig, folded, temporary_outputs, params, scopes) -> list:
    """What the emitter gets for each output (interface.py)."""
    b = cg.b
    args = _argument_positions(node, folded)
    outputs = []
    for j, o in enumerate(sig.outputs):
        if o.mode is OutputMode.ABSENT:
            outputs.append(None)
            continue
        v = next(params)
        if o.mode is OutputMode.SCALAR:
            outputs.append(ScalarOutput(cg, v, o.repr.dtype))
        elif o.mode is OutputMode.PROVIDED:
            a = unpack(b, o.repr, v)
            scopes.mark(a.data, f"o{j}")
            outputs.append(ProvidedOutput(cg, a))
        else:
            outputs.append(
                SlotOutput.for_output(
                    cg,
                    v,
                    o,
                    aliases=tuple(args[i] for i in o.aliases),
                    temporary=j in temporary_outputs,
                    alias=(f"o{j}", scopes.others(f"o{j}")),
                )
            )
    return outputs


__all__ = ["NodeFunction", "build_node_function", "debug_name"]
