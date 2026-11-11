"""Stage 3: where each value of a program lives, and when it is released.

Plain data in, plain data out: nothing here touches LLVM or the interpreter.

Every value has a kind:

* `input`, `const`: memory the program doesn't own;
* `provided`: memory the program gives the task that writes it: a place in the arena,
  or a fresh buffer if the value is a graph output;
* `returned`: the task hands the array back: a buffer it allocated, or a view of one
  of the values it may alias (`MemoryPlan.aliases`). Which one is only known when the
  program runs, so every value a returned value may alias is kept alive as long as it.

A task may also work in place: an output is then not a value of its own but the input
it overwrites (`MemoryPlan.same`), whatever kind that is.

The arena is laid out by liveness: a value's place is free again after the last task
that needs it, or anything that may alias it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from pytensor.link.llvm.codegen.function import OutputMode
from pytensor.link.llvm.plan.values import Values


@dataclass
class MemoryPlan:
    #: vid -> "input", "const", "provided" or "returned"
    kind: dict
    #: vid -> the values a returned value may be a view of
    aliases: dict
    #: vid -> the value it is: outputs of tasks that work in place. The other
    #: dictionaries have no entries for these
    same: dict
    #: vid -> index of the task that produces it (-1: inputs and constants)
    defined: dict
    #: vid -> index of the last task that needs it alive (`forever` for graph outputs)
    end: dict
    #: vid -> arena offset, for provided values that aren't graph outputs
    offsets: dict
    arena_bytes: int
    #: vid -> position among the graph outputs (the first one, if several)
    first_out: dict
    forever: int

    def release_points(self, n_tasks) -> list:
        """Per task: the values to release after it."""
        release = [[] for _ in range(n_tasks)]
        for k, what in self.kind.items():
            if (
                what in ("returned", "provided")
                and self.end.get(k, self.forever) < self.forever
            ):
                release[max(self.end[k], self.defined[k])].append(k)
        return release

    def value(self, k) -> int:
        """The value that holds `k`'s memory: `k`, or the value it overwrote."""
        return self.same.get(k, k)

    def in_order(self) -> list:
        """The values, each after the values it may alias."""
        return sorted(self.kind, key=lambda k: self.defined[k])


def lifetimes(tasks, kind, aliases, out_ids, forever) -> tuple[dict, dict]:
    """(defined, end) per value: the task that produces it and the last task that
    needs it alive. A value is needed as long as anything that may alias it."""
    defined = dict.fromkeys(kind, -1)
    end = {}
    for i, task in enumerate(tasks):
        for k in task.inputs:
            end[k] = max(end.get(k, -1), i)
        for k in task.outputs:
            if k in task.inputs:
                continue  # overwritten in place: it was there before
            defined[k] = i
            end.setdefault(k, i)
    for k in out_ids:
        defined.setdefault(k, -1)
        end[k] = forever
    # later values first, so chains of views propagate
    for k in sorted(aliases, key=lambda k: defined[k], reverse=True):
        for source in aliases[k]:
            end[source] = max(end.get(source, -1), end[k])
    return defined, end


def first_fit(intervals) -> tuple[dict, int]:
    """Place `intervals` ((start, stop, nbytes, key), live from start to stop
    inclusive) at 64-byte aligned offsets, reusing space that is free again. Returns
    ({key: offset}, total bytes)."""
    offsets, total = {}, 0
    active = []  # (offset, size, stop)
    for start, stop, nbytes, key in sorted(intervals, key=lambda x: (x[0], x[1])):
        size = max(64, -(-nbytes // 64) * 64)
        active = [a for a in active if a[2] >= start]
        off = 0
        for a_off, a_size, _ in sorted(active):
            if off + size <= a_off:
                break
            off = max(off, a_off + a_size)
        active.append((off, size, stop))
        offsets[key] = off
        total = max(total, off + size)
    return offsets, total


def plan_memory(values: Values, tasks, in_ids, out_ids) -> MemoryPlan:
    """`tasks` have `inputs`, `outputs` (vids), and per output a mode (`modes`) and the
    positions of the task inputs it may alias or, in place, is (`aliases`)."""
    forever = len(tasks) + 1
    # outputs of tasks working in place are the value they overwrite, from here on
    same = {}
    for task in tasks:
        for k, mode, source in zip(task.outputs, task.modes, task.aliases, strict=True):
            if mode is OutputMode.INPLACE:
                overwritten = task.inputs[source[0]]
                same[k] = same.get(overwritten, overwritten)
    if same:
        tasks = [
            SimpleNamespace(
                inputs=[same.get(k, k) for k in task.inputs],
                outputs=[same.get(k, k) for k in task.outputs],
                modes=task.modes,
                aliases=task.aliases,
            )
            for task in tasks
        ]
        out_ids = [same.get(k, k) for k in out_ids]

    kind, aliases = {}, {}
    for k in in_ids:
        kind[k] = "input"
    for task in tasks:
        for k in task.inputs:
            if k in values.consts and k not in kind:
                kind[k] = "const"
    for task in tasks:
        for k, mode, may_alias in zip(
            task.outputs, task.modes, task.aliases, strict=True
        ):
            if mode is OutputMode.INPLACE:
                continue
            if mode is OutputMode.PROVIDED:
                kind[k] = "provided"
            else:
                kind[k] = "returned"
                aliases[k] = [task.inputs[pos] for pos in may_alias]
    for k in out_ids:
        if k in values.consts and k not in kind:
            kind[k] = "const"
    defined, end = lifetimes(tasks, kind, aliases, out_ids, forever)

    first_out = {}
    for position, k in enumerate(out_ids):
        first_out.setdefault(k, position)
    intervals = []
    for k, what in kind.items():
        if what == "provided" and k not in first_out:
            t = values.types[k]
            nbytes = int(np.prod(t.shape, dtype=np.int64)) * np.dtype(t.dtype).itemsize
            intervals.append((defined[k], end[k], nbytes, k))
    offsets, arena = first_fit(intervals)
    return MemoryPlan(
        kind, aliases, same, defined, end, offsets, arena, first_out, forever
    )
