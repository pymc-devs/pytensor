"""What the planner decides about a graph and codegen runs: the steps of a graph, each
with how it runs (`Step`), and the graph with them (`PlannedGraph`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pytensor.graph.basic import Apply
from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.codegen.signature import Signature
from pytensor.link.llvm.codegen.types import is_absent
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.type import TensorType


if TYPE_CHECKING:
    from pytensor.link.llvm.codegen.call import Kernel


@dataclass
class Step:
    """One thing a planned graph runs, in order, with everything decided about how."""

    #: the node it runs; None for code compiled on its own, which it calls with
    #: `inputs` (a node or a unit of the program)
    node: Apply | None
    #: the values (vids) it reads: one per input of the node, or per argument of the
    #: code compiled on its own
    inputs: list[int]
    #: the values it writes, one per output
    outputs: list[int]
    #: the layouts of `inputs`
    in_layouts: list[Layout | None]
    #: the kernel it calls; None: its code is generated (and inlined)
    kernel: Kernel | None
    #: the signature of its function: representations of inputs and outputs
    signature: Signature
    #: what it runs, for dumps and profiles
    description: str

    @property
    def arguments(self) -> list[int]:
        """The values passed, one per input of the signature (inputs without data
        aren't passed)."""
        if self.node is None:
            return self.inputs
        return [
            vid
            for vid, var in zip(self.inputs, self.node.inputs, strict=True)
            if not is_absent(var.type)
        ]


@dataclass
class PlannedGraph:
    """A graph with every decision made about how its nodes run (`Planner.graph`,
    `Planner.program`): a Scan's loop body, a Blockwise's core, a unit's inner graph,
    or the program.

    Its steps are in the order they run, and refer to values by their vids."""

    #: the values (vids) of the graph's inputs and outputs
    in_ids: list[int]
    out_ids: list[int]
    steps: list[Step]
    #: {vid: (type, data)} of the constants the steps read
    constants: dict[int, tuple[TensorType | ScalarType, Any]]
    #: the layouts of its values (vid -> Layout; constants: `Planner.layout`)
    layouts: dict[int, Layout | None]


__all__ = ["PlannedGraph", "Step"]
