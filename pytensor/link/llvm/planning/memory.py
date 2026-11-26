"""Where the values of a planned graph live: their buffers, and the frames that hold the
buffers of a fixed size.

A buffer is one piece of memory with one lifetime. A step creates it, or it is the
memory of a graph input or constant. It may hold several values:

* the value it is created for,
* values that may be views of it, or may have overwritten it in place.

Whether a value really is a view, or really overwrote its input, is often only known
at run time. So a value may be in several buffers: its own, and those of the values it
may be a view of. A buffer is free after the last step that reads any value that may
be in it.

For example, if step 0 writes `y = exp(x)` into memory it is given, step 1 returns
`z = y.T`, which may be a view of `y`, and step 2 reads `z`:

    buffer 0: given to step 0, may hold y and z, free after step 2
    buffer 1: allocated by step 1, may hold z, free after step 2

Buffers of a fixed size that a step is given are placed in one of two frames: blocks
of memory the graph gets once per call. Buffers that are never needed at the same time
share space in a frame (`first_fit`). That is all there is to reuse: two buffers share
memory exactly where their places in a frame overlap. The stack frame holds the
buffers of up to `STACK_TEMPORARY_BYTES`. Code generation puts it on the stack, where
LLVM can keep its values in registers, unless the function's stack budget is used up;
then it goes into scratch memory too. The scratch frame holds the others.

A buffer created for a graph output isn't in a frame. A call usually provides memory
for the output, and the step writes into that. If it doesn't, code generation gives
the value a temporary of its own for that call.

Plain data in, plain data out: nothing here generates code.
"""

from __future__ import annotations

import enum
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NewType

import numpy as np

from pytensor.link.llvm.codegen.context import STACK_TEMPORARY_BYTES
from pytensor.link.llvm.planning.signature import Output, Passing


if TYPE_CHECKING:
    from pytensor.link.llvm.planning.plan import Step, ValueId


#: A buffer of a planned graph: its position in `Memory.buffers`.
BufferId = NewType("BufferId", int)


class BufferKind(enum.Enum):
    """Where the memory of a buffer comes from."""

    #: The memory of a graph input.
    INPUT = "input"
    #: An array constant of the graph.
    CONSTANT = "constant"
    #: Memory the step that creates the buffer is given to write its result into.
    GIVEN = "given"
    #: Memory the step that creates the buffer allocates itself.
    ALLOCATED = "allocated"


class Frame(enum.Enum):
    """The frames that hold the buffers of a fixed size, see the module docstring."""

    STACK = "stack"
    SCRATCH = "scratch"


@dataclass(frozen=True)
class Buffer:
    """One piece of memory of a planned graph, with one lifetime."""

    kind: BufferKind
    #: The step that creates it. None for graph inputs and constants.
    created_by: int | None
    #: The values that may be in it: the one it is created for first, then the values
    #: that may be views of it or may have overwritten it.
    values: tuple[ValueId, ...]
    #: The last step that may read it. None if it is still needed after the graph ran,
    #: because a graph output may be in it, and for the memory of graph inputs and
    #: constants, which isn't the graph's to free.
    free_after: int | None
    #: Its size in bytes, for memory of a fixed size that a step is given.
    nbytes: int | None = None
    #: The frame it is placed in, and its offset in bytes there. None if it isn't in a
    #: frame.
    frame: Frame | None = None
    offset: int = 0


@dataclass(frozen=True)
class Memory:
    """The buffers of a planned graph, made by `plan_memory`."""

    #: All buffers. A `BufferId` is a position in this tuple.
    buffers: tuple[Buffer, ...]
    #: For each value that is in memory: the buffers it may be in.
    buffer_of: dict[ValueId, tuple[BufferId, ...]]
    #: For each value a step produces into memory: the buffer created for it.
    created_for: dict[ValueId, BufferId]
    #: The size in bytes of each frame.
    frame_bytes: dict[Frame, int]

    def place_in_frame(self, value_id: ValueId) -> tuple[Frame, int] | None:
        """The frame and offset of the buffer created for `value_id`, or None if it
        isn't in a frame."""
        buffer_id = self.created_for.get(value_id)
        if buffer_id is None:
            return None
        buffer = self.buffers[buffer_id]
        return None if buffer.frame is None else (buffer.frame, buffer.offset)


def plan_memory(
    steps: Sequence[Step],
    in_ids: Collection[ValueId],
    out_ids: Collection[ValueId],
    constants: Collection[ValueId],
) -> Memory:
    """The buffers of the graph with the steps `steps`, inputs `in_ids`, outputs
    `out_ids` and constants `constants`, and their place in the frames.

    Graph inputs and constants only get a buffer if a step's result may be a view of
    them."""
    outputs = set(out_ids)
    inputs = set(in_ids)
    last_read: dict[ValueId, int] = {}
    for i, step in enumerate(steps):
        for value_id in step.arguments:
            last_read[value_id] = i

    kinds: list[BufferKind] = []
    created_by: list[int | None] = []
    sizes: list[int | None] = []
    buffer_of: dict[ValueId, tuple[BufferId, ...]] = {}
    created_for: dict[ValueId, BufferId] = {}

    def new(kind: BufferKind, step: int | None, nbytes: int | None) -> BufferId:
        kinds.append(kind)
        created_by.append(step)
        sizes.append(nbytes)
        return BufferId(len(kinds) - 1)

    def buffers_of_source(value_id: ValueId) -> tuple[BufferId, ...]:
        if value_id not in buffer_of:
            kind = BufferKind.INPUT if value_id in inputs else BufferKind.CONSTANT
            if value_id not in inputs and value_id not in constants:
                raise ValueError(f"a view of value {value_id}, which no step produces")
            buffer_of[value_id] = (new(kind, None, None),)
        return buffer_of[value_id]

    for i, step in enumerate(steps):
        for value_id, output in zip(step.outputs, step.signature.outputs, strict=True):
            if not output.is_tensor:
                continue
            kind = (
                BufferKind.GIVEN
                if output.passing is Passing.CALLER_BUFFER
                else BufferKind.ALLOCATED
            )
            own = new(kind, i, _fixed_bytes(output))
            created_for[value_id] = own
            sources = [
                buffer_id
                for a in step.signature.alias_arguments(output)
                for buffer_id in buffers_of_source(step.arguments[a])
            ]
            buffer_of[value_id] = tuple(dict.fromkeys([own, *sources]))

    values: list[list[ValueId]] = [[] for _ in kinds]
    for value_id, buffer_ids in buffer_of.items():
        for buffer_id in buffer_ids:
            values[buffer_id].append(value_id)

    free_after: list[int | None] = []
    for creator, held in zip(created_by, values, strict=True):
        if creator is None or any(v in outputs for v in held):
            free_after.append(None)
        else:
            reads = [last_read.get(v, creator) for v in held]
            free_after.append(max([creator, *reads]))

    # A buffer in a frame is needed from the step that creates it until it is free, or
    # until the end of the graph.
    places: dict[int, tuple[Frame, int]] = {}
    frame_bytes: dict[Frame, int] = {}
    for frame in Frame:
        intervals: list[tuple[int, int, int, int]] = []
        for index, (creator, nbytes, end) in enumerate(
            zip(created_by, sizes, free_after, strict=True)
        ):
            if creator is None or nbytes is None or values[index][0] in outputs:
                continue
            if (nbytes <= STACK_TEMPORARY_BYTES) == (frame is Frame.STACK):
                stop = len(steps) if end is None else end
                intervals.append((creator, stop, nbytes, index))
        offsets, frame_bytes[frame] = first_fit(intervals)
        places.update((index, (frame, offset)) for index, offset in offsets.items())

    buffers = []
    for index, kind in enumerate(kinds):
        in_frame, offset = places.get(index, (None, 0))
        buffers.append(
            Buffer(
                kind,
                created_by[index],
                tuple(values[index]),
                free_after[index],
                sizes[index],
                in_frame,
                offset,
            )
        )
    return Memory(tuple(buffers), buffer_of, created_for, frame_bytes)


def _fixed_bytes(output: Output) -> int | None:
    """The size in bytes of the memory a step is given for `output`, if it is fixed."""
    if output.passing is not Passing.CALLER_BUFFER or not output.tensor.static:
        return None
    tensor = output.tensor
    elements = int(np.prod(tensor.fixed_shape, dtype=np.int64))
    return elements * np.dtype(tensor.scalar_type.dtype).itemsize


def first_fit[K](
    intervals: Sequence[tuple[int, int, int, K]],
) -> tuple[dict[K, int], int]:
    """Place `intervals` at 64-byte aligned offsets in one block of memory, reusing
    space that is free again. Each interval is (start, stop, nbytes, key), and is live
    from step `start` to step `stop`, both included. Returns ({key: offset}, the size
    of the block in bytes)."""
    offsets: dict[K, int] = {}
    total = 0
    active: list[tuple[int, int, int]] = []  # (offset, size, stop)
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


__all__ = [
    "Buffer",
    "BufferId",
    "BufferKind",
    "Frame",
    "Memory",
    "first_fit",
    "plan_memory",
]
