"""What the planner decides about a graph and codegen runs: the steps of a graph, each
with how it runs (`Step`), and the graph with them (`PlannedGraph`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NewType

from pytensor.graph.basic import Apply
from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.codegen.signature import Signature
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.type import TensorType


if TYPE_CHECKING:
    from pytensor.link.llvm.codegen.implementation import Implementation


#: A value of a planned graph (`planner.Values`): what a variable of the graph holds,
#: one per value even where a variable appears several times (in inlined OpFromGraphs).
ValueId = NewType("ValueId", int)


@dataclass
class Step:
    """One thing a planned graph runs, in order, with everything decided about how."""

    #: the node it runs
    node: Apply
    #: the values it reads, one per input of the node
    inputs: list[ValueId]
    #: the values it writes, one per output
    outputs: list[ValueId]
    #: the layouts of `inputs`
    in_layouts: list[Layout | None]
    #: how the node runs: generated code, a kernel, a module of its own
    implementation: Implementation
    #: the signature of its function (`node_signature` for the implementation)
    signature: Signature
    #: what it runs, for dumps and profiles
    description: str

    @property
    def arguments(self) -> list[ValueId]:
        """The values passed, one per input of the signature (folded constants and
        inputs without data aren't passed)."""
        positions = self.signature.argument_of_input
        if positions is None:
            return self.inputs
        return [
            value_id
            for value_id, position in zip(self.inputs, positions, strict=True)
            if position is not None
        ]


@dataclass
class PlannedGraph:
    """A graph with every decision made about how its nodes run (`Planner.graph`,
    `Planner.program`): a Scan's loop body, a Blockwise's core, a unit's inner graph,
    or the program.

    Its steps are in the order they run, and refer to values by their `ValueId`."""

    #: the values of the graph's inputs and outputs
    in_ids: list[ValueId]
    out_ids: list[ValueId]
    steps: list[Step]
    #: {value id: (type, data)} of the constants the steps read
    constants: dict[ValueId, tuple[TensorType | ScalarType, Any]]
    #: the layouts of its values (value id -> Layout; constants: `Planner.layout`)
    layouts: dict[ValueId, Layout | None]


__all__ = ["PlannedGraph", "Step", "ValueId"]
