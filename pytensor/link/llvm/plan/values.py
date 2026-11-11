"""The values of a program being planned, and the items that read and write them."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pytensor.graph.basic import Constant
from pytensor.link.llvm.codegen.types import is_absent


@dataclass
class NodeItem:
    """A node to run, with the value ids (vids: the planner's names for values) of what
    it reads and writes. The same PyTensor node can appear with different vids (inlined
    OpFromGraphs)."""

    node: object
    inputs: list
    outputs: list


@dataclass
class UnitItem:
    """Several items compiled as one function."""

    items: list
    inputs: list  # vids read from outside
    outputs: list  # (inner vid, outer vid)
    description: str


@dataclass
class SubprogramItem:
    """An interpreted OpFromGraph: a subprogram."""

    node: object
    inputs: list
    outputs: list


class Values:
    """The values of a program: a type per value id, and the data of constants."""

    def __init__(self):
        self.types = {}  # vid -> PyTensor type
        self.consts = {}  # vid -> numpy data
        self._next = 0

    def new(self, t) -> int:
        """A new value of type `t`."""
        k = self._next
        self._next += 1
        self.types[k] = t
        return k

    def of(self, var, scope) -> int:
        """The value of `var`: what `scope` says, or a new value for a constant."""
        if var in scope:
            return scope[var]
        if is_absent(var.type):
            # None inputs have no value; nothing ever reads this vid
            k = self.new(var.type)
            scope[var] = k
            return k
        if isinstance(var, Constant):
            k = self.new(var.type)
            self.consts[k] = np.asarray(var.data)
            scope[var] = k
            return k
        raise ValueError(f"variable {var} used before it is defined")

    def folded(self, k) -> bool:
        """0-d constants are folded into compiled units."""
        return k in self.consts and self.consts[k].ndim == 0

    def unit_inputs(self, items) -> list:
        """The values `items` read from outside (in order of first use)."""
        produced = {k for it in items for k in it.outputs}
        ins = []
        for it in items:
            for k in it.inputs:
                if (
                    k in produced
                    or k in ins
                    or self.folded(k)
                    or is_absent(self.types[k])
                ):
                    continue
                ins.append(k)
        return ins

    def constants_of(self, items, extra=()) -> dict:
        """{vid: (type, data)} of the constants `items` read (and of the vids `extra`)."""
        vids = [k for it in items for k in it.inputs] + list(extra)
        return {k: (self.types[k], self.consts[k]) for k in vids if k in self.consts}
