"""When the static temporaries of generated code are live; their place in an arena.

A provided output of an item that isn't produced in the function's own output memory is
a temporary of static size. Those that don't go on the stack share one arena
(`Codegen.temporaries`): a temporary's place is free again after the last item that
reads it, or anything that may be a view of it.

Plain data in, plain data out: nothing here touches LLVM.
"""

from __future__ import annotations

from pytensor.link.llvm.codegen.signature import OutputMode


def lifetimes(resolved, kept) -> tuple[dict, dict]:
    """(defined, end) per value of the items `resolved` (`plan.resolve.Resolved`): the
    item that produces it and the last item that needs it alive. A value is needed as
    long as anything that may be a view of it (or overwrote it in place); `kept` are
    needed after the last item (`len(resolved)`)."""
    defined, end, aliases = {}, {}, {}
    for i, r in enumerate(resolved):
        args = r.arguments
        for k in args:
            end[k] = i
        for k, o in zip(r.item.outputs, r.signature.outputs, strict=True):
            if o.mode is OutputMode.ABSENT:
                continue
            defined[k] = i
            end.setdefault(k, i)
            if o.aliases:
                aliases[k] = [args[a] for a in o.aliases]
    for k in kept:
        end[k] = len(resolved)
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


def temporary_lives(resolved, vids, kept) -> dict:
    """When each value in `vids` is needed, as {vid: (first, last)}: from the item that
    produces it to the last item that needs it (`lifetimes`), positions in `resolved`.

    `kept`: values still needed after the last item (the function's outputs)."""
    defined, end = lifetimes(resolved, kept)
    return {k: (defined[k], end[k]) for k in vids}


def place_temporaries(resolved, sizes: dict, kept) -> tuple[dict, int]:
    """Arena offsets for the temporaries `sizes` ({vid: nbytes}) of the items
    `resolved`; `kept`: values needed after the last item (the function's outputs).
    Returns ({vid: offset}, arena bytes)."""
    lives = temporary_lives(resolved, sizes, kept)
    return first_fit([(*lives[k], n, k) for k, n in sizes.items()])
