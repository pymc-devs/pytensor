"""What everything generated into one LLVM module shares (`KernelModule`), and the
options of code generation (`CodegenOptions`).

Every generated module ends up as one kernel: the program, a step of it compiled on its
own, a node compiled for a test. The functions built in it share the module, the
options, the link table (call.py) through which they reach the kernels they call and
their array constants, and the way inner graphs are planned.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pytensor.graph.basic import Variable
from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.codegen.call import LinkTable
from pytensor.link.llvm.codegen.types import ir


if TYPE_CHECKING:
    from pytensor.link.llvm.codegen.plan import PlannedGraph


@dataclass(frozen=True)
class CodegenOptions:
    """The options of code generation (the code-generation part of the program's
    `Options`)."""

    #: may generated loops run in parallel (`Codegen.for_each`, `for_range`)?
    parallel: bool = True
    #: verify at run time that views and arguments have the layouts the planner relies
    #: on (`Codegen.check_layouts`, the kernel wrapper's argument check)
    check_layouts: bool = False
    #: LLVM's fast-math flags on float arithmetic
    fastmath: bool = True
    #: locate every instruction at the Python code that emitted it
    debug_info: bool = False


#: Plans an inner graph (a Scan's body, a Blockwise's core, a unit) for the layouts of
#: its inputs: `Planner.graph`.
type PlanGraph = Callable[
    [Sequence[Variable], Sequence[Variable], Sequence[Layout | None]], PlannedGraph
]


@dataclass(eq=False)
class KernelModule:
    """One LLVM module that becomes one kernel: what everything generated into it
    shares."""

    module: ir.Module
    options: CodegenOptions
    #: how inner graphs are planned; None: code generated here can't run inner graphs
    plan_graph: PlanGraph | None = None
    #: the module's link table: the kernels its code calls, its array constants
    links: LinkTable = field(default_factory=LinkTable)

    @classmethod
    def new(
        cls,
        name: str,
        options: CodegenOptions | None = None,
        plan_graph: PlanGraph | None = None,
    ) -> KernelModule:
        """A new, empty module named `name`."""
        options = CodegenOptions() if options is None else options
        module = ir.Module(
            name, fastmath=options.fastmath, debug_info=options.debug_info
        )
        return cls(module, options, plan_graph)


__all__ = ["CodegenOptions", "KernelModule", "PlanGraph"]
