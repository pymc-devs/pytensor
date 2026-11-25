"""When the static temporaries of generated code are live; their place in an arena.

A provided output of a step that isn't produced in the function's own output memory is
a temporary of static size. Those that don't go on the stack share one arena
(`Codegen.temporaries`): a temporary's place is free again after the last step that
reads it, or anything that may be a view of it.

Plain data in, plain data out: nothing here touches LLVM.
"""

from __future__ import annotations


def lifetimes(steps, kept) -> tuple[dict, dict]:
    """(defined, end) per value of the steps `steps` (`planner.Step`): the
    step that produces it and the last step that needs it alive. A value is needed as
    long as anything that may be a view of it (or overwrote it in place); `kept` are
    needed after the last step (`len(steps)`)."""
    defined, end, aliases = {}, {}, {}
    for i, step in enumerate(steps):
        args = step.arguments
        for k in args:
            end[k] = i
        for k, output in zip(step.outputs, step.signature.outputs, strict=True):
            if output.is_absent:
                continue
            defined[k] = i
            end.setdefault(k, i)
            if output.aliased_inputs:
                aliases[k] = [args[a] for a in step.signature.alias_arguments(output)]
    for k in kept:
        end[k] = len(steps)
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


def temporary_lives(steps, value_ids, kept) -> dict:
    """When each value in `value_ids` is needed, as {value id: (first, last)}: from the
    step that produces it to the last step that needs it (`lifetimes`), positions in
    `steps`.

    `kept`: values still needed after the last step (the function's outputs)."""
    defined, end = lifetimes(steps, kept)
    return {k: (defined[k], end[k]) for k in value_ids}


def place_temporaries(steps, sizes: dict, kept) -> tuple[dict, int]:
    """Arena offsets for the temporaries `sizes` ({value id: nbytes}) of the steps
    `steps`; `kept`: values needed after the last step (the function's outputs).
    Returns ({value id: offset}, arena bytes)."""
    lives = temporary_lives(steps, sizes, kept)
    return first_fit([(*lives[k], n, k) for k, n in sizes.items()])
