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
from pytensor.link.llvm.codegen.call import LinkTable
from pytensor.link.llvm.codegen.errors import ExceptionTable
from pytensor.link.llvm.codegen.types import ir
from pytensor.link.llvm.layout import Layout


if TYPE_CHECKING:
    from pytensor.link.llvm.planning.implementation import OwnModule
    from pytensor.link.llvm.planning.plan import PlannedGraph


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
    #: what it runs, for stats and perf maps
    description: str = ""
    #: the compiled module, once linked (`link`)
    compiled: ir.CompiledModule | None = None
    #: may its code run work on the runtime's thread pool (a parallel loop, a kernel
    #: that does)?
    uses_pool: bool = False
    #: the exception classes of its error codes beyond the fixed kinds: one table for
    #: every module of a program (built in `program._build`)
    exceptions: ExceptionTable = field(default_factory=ExceptionTable)
    #: the modules of the steps compiled on their own that its code calls, built
    #: before it (`kernel.build_own_module`)
    own_modules: dict[OwnModule, KernelModule] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        name: str,
        options: CodegenOptions | None = None,
        plan_graph: PlanGraph | None = None,
        exceptions: ExceptionTable | None = None,
    ) -> KernelModule:
        """A new, empty module named `name`; its error codes in `exceptions` (default:
        a table of its own)."""
        options = CodegenOptions() if options is None else options
        module = ir.Module(
            name, fastmath=options.fastmath, debug_info=options.debug_info
        )
        return cls(
            module,
            options,
            plan_graph,
            description=name,
            exceptions=ExceptionTable() if exceptions is None else exceptions,
        )

    @property
    def address(self) -> int:
        """The address of its kernel (the function `kernel`), once linked."""
        if self.compiled is None:
            raise ValueError(f"{self.module.name} isn't linked yet")
        return self.compiled.address(KERNEL)

    @property
    def config(self) -> int:
        """The config to call its kernel with: its link table, once linked."""
        return self.links.address


#: The name of a module's kernel (`kernel.build_kernel_wrapper`).
KERNEL = "kernel"


def link(units: Sequence[KernelModule], opt_level: int = 3) -> None:
    """Compile the modules `units` together and link them: fix their link tables, and
    write into them the address and config of the modules they call (which must be
    among `units`). Their kernels can be called from then on (`address`, `config`)."""
    compiled = ir.compile_many([unit.module for unit in units], opt_level=opt_level)
    for unit, c in zip(units, compiled, strict=True):
        unit.compiled = c
        unit.links.fix()
    for unit in units:
        unit.links.write_module_entries()


__all__ = ["KERNEL", "CodegenOptions", "KernelModule", "PlanGraph", "link"]
