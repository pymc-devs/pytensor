"""OpFromGraph: a compiled unit (`execution=COMPILED`) runs its inner graph.

Other OpFromGraphs never get to code generation: the planner inlines them.
"""

from pytensor.compile.builders import Execution, OpFromGraph
from pytensor.graph.features import op_view_parents
from pytensor.link.llvm.analysis.layout import infer_layouts, new_memory_layout
from pytensor.link.llvm.codegen.emit import Array
from pytensor.link.llvm.codegen.interface import (
    NewBuffer,
    ViewOrCopy,
    destinations,
    llvm_emit,
    llvm_outputs,
)


@llvm_outputs.register(OpFromGraph)
def _(op, node, input_layouts):
    """An output the inner graph may pass an input through is a view or a copy;
    the others are new memory. Their layouts are what the inner graph gives them. A
    compiled unit has them where new memory can, since it copies an output that isn't
    in its inputs' memory (`new_memory_layout`)."""
    inner_layouts = infer_layouts(op.fgraph, input_layouts)
    parents = op_view_parents(op, node)
    storage = []
    for inner_output, output, output_parents in zip(
        op.fgraph.outputs, node.outputs, parents, strict=True
    ):
        layout = inner_layouts[inner_output]
        if layout is None:  # not a tensor
            storage.append(NewBuffer())
            continue
        if op.execution is Execution.COMPILED:
            layout = new_memory_layout(layout, output.type)
        viewed = tuple(i for i, var in enumerate(node.inputs) if var in output_parents)
        storage.append(ViewOrCopy(viewed, layout) if viewed else NewBuffer(layout))
    return storage


@llvm_emit.register(OpFromGraph)
def _(op, node, codegen, inputs, outputs):
    """A compiled unit (an OpFromGraph with `execution=COMPILED`): its code runs its
    inner graph. (Other OpFromGraphs never get here: they are inlined.)"""
    layouts = [a.layout if isinstance(a, Array) else None for a in inputs]
    graph = codegen.graph_function(op.fgraph.inputs, op.fgraph.outputs, layouts, "unit")
    graph.emit_call(codegen, inputs, destinations(outputs))
