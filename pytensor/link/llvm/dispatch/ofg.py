"""OpFromGraph: a compiled unit (`execution=COMPILED`) runs its inner graph.

Other OpFromGraphs never get to code generation: the planner inlines them.
"""

from pytensor.compile.builders import Execution, OpFromGraph
from pytensor.link.llvm.analysis.layout import infer_layouts, new_memory_layout
from pytensor.link.llvm.codegen.array import Array
from pytensor.link.llvm.codegen.interface import llvm_emit, llvm_layout


@llvm_layout.register(OpFromGraph)
def _(op, node, input_layouts):
    """The layouts of the inner outputs. Inlined, that is what they are; a compiled
    unit has them where new memory can, since it copies an output that isn't in its
    inputs' memory (`new_memory_layout`)."""
    inner_layouts = infer_layouts(op.fgraph, input_layouts)
    layouts = [inner_layouts[output] for output in op.fgraph.outputs]
    if op.execution is Execution.COMPILED:
        return [
            None if layout is None else new_memory_layout(layout, output.type)
            for layout, output in zip(layouts, node.outputs, strict=True)
        ]
    return layouts


@llvm_emit.register(OpFromGraph)
def _(op, node, cg, inputs, outputs):
    """A compiled unit (an OpFromGraph with `execution=COMPILED`): its code runs its
    inner graph. (Other OpFromGraphs never get here: they are inlined.)"""
    layouts = [a.layout if isinstance(a, Array) else None for a in inputs]
    graph = cg.graph_function(op.fgraph.inputs, op.fgraph.outputs, layouts, "unit")
    graph.emit_call(cg, inputs, outputs)
