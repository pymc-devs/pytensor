"""Stage 3: where each value of a program lives, and when it is released.

Plain data in, plain data out: nothing here touches LLVM or the interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pytensor.link.llvm.codegen.function import OutputMode
from pytensor.link.llvm.plan.values import Values


@dataclass
class MemoryPlan:
    #: vid -> (kind, source): "input", "const", "provided", "allocated", or "view" of
    #: the value `source`
    kind: dict
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
        for k, (what, _) in self.kind.items():
            if (
                what in ("allocated", "view", "provided")
                and self.end.get(k, self.forever) < self.forever
            ):
                release[max(self.end[k], self.defined[k])].append(k)
        return release


def plan_memory(values: Values, tasks, in_ids, out_ids) -> MemoryPlan:
    """`tasks` have `inputs`, `outputs` (vids), `modes` and `view_of` per output."""
    n_tasks = len(tasks)
    INF = n_tasks + 1
    kind = {}  # vid -> ("input"|"const"|"provided"|"allocated"|"view", source)
    for k in in_ids:
        kind[k] = ("input", None)
    for s in tasks:
        for k in s.inputs:
            if k in values.consts and k not in kind:
                kind[k] = ("const", None)
    defined, end = {}, {}
    for k in kind:
        defined[k] = -1
    for i, s in enumerate(tasks):
        for k in s.inputs:
            end[k] = max(end.get(k, -1), i)
        for j, (k, mode) in enumerate(zip(s.outputs, s.modes, strict=True)):
            defined[k] = i
            end.setdefault(k, i)
            if mode is OutputMode.VIEW:
                kind[k] = ("view", s.inputs[s.view_of[j]])
            elif mode is OutputMode.PROVIDED:
                kind[k] = ("provided", None)
            else:
                kind[k] = ("allocated", None)
    for k in out_ids:
        if k in values.consts and k not in kind:
            kind[k] = ("const", None)
            defined[k] = -1
        end[k] = INF
    # views keep their source alive (later values first, so chains propagate)
    for k in sorted(kind, key=lambda k: defined[k], reverse=True):
        what, src = kind[k]
        if what == "view":
            end[src] = max(end.get(src, -1), end[k])

    # arena: static provided temporaries, first fit by liveness
    intervals = []
    first_out = {}
    for o, k in enumerate(out_ids):
        first_out.setdefault(k, o)
    for k, (what, _) in kind.items():
        if what == "provided" and k not in first_out:
            t = values.types[k]
            nbytes = int(np.prod(t.shape, dtype=np.int64)) * np.dtype(t.dtype).itemsize
            intervals.append((defined[k], end[k], nbytes, ("value", k)))
    offsets, arena = {}, 0
    active = []
    for start, stop, nbytes, what in sorted(intervals, key=lambda x: (x[0], x[1])):
        size = max(64, -(-nbytes // 64) * 64)
        active = [a for a in active if a[2] >= start]
        off = 0
        for a_off, a_size, _ in sorted(active):
            if off + size <= a_off:
                break
            off = max(off, a_off + a_size)
        active.append((off, size, stop))
        offsets[what[1]] = off
        arena = max(arena, off + size)
    return MemoryPlan(kind, defined, end, offsets, arena, first_out, INF)
