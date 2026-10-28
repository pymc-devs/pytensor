"""Capture and propagate compact source provenance for graph variables."""

import linecache
import sys
import traceback
from collections.abc import Iterable, Sequence
from io import StringIO


Frame = tuple[str | None, int, str, str | None]
StackTrace = tuple[Frame, ...]


class TraceSet(tuple):
    """Immutable, ordered, deduplicated source provenance for a variable."""

    def __new__(cls, traces: Iterable[StackTrace] = ()):
        return super().__new__(cls, tuple(tuple(stack) for stack in traces))

    def merge(self, other: "TraceSet") -> "TraceSet":
        return TraceSet(dict.fromkeys((*self, *other)))


def as_trace_set(trace) -> TraceSet:
    """Normalize current and historical ``tag.trace`` representations."""
    if isinstance(trace, TraceSet):
        return trace
    if not trace:
        return TraceSet()
    if isinstance(trace[0], tuple) and len(trace[0]) == 4:
        trace = [trace]
    return TraceSet(
        dict.fromkeys(tuple(tuple(frame) for frame in stack) for stack in trace)
    )


def resolve_frame(frame: Frame) -> Frame:
    """Load a source line when a diagnostic actually needs to display it."""
    filename, lineno, name, source_line = frame
    if source_line is None and filename:
        source_line = linecache.getline(filename, lineno).strip() or None
    return (filename, lineno, name, source_line)


def capture_stack(
    f=None, limit: int | None = None, skips: Sequence[str] = ()
) -> StackTrace:
    """Capture frame coordinates without source lookup or retaining frame objects."""
    if f is None:
        f = sys._getframe(1)

    frames = []
    skipping_internal_frames = True
    while f is not None and (limit is None or len(frames) < limit):
        code = f.f_code
        filename = code.co_filename
        if skipping_internal_frames:
            is_internal = any(path in filename for path in skips)
            # Keep the construction stack in test files, as PyTensor has always done.
            if is_internal and "tests" not in filename:
                f = f.f_back
                continue
            skipping_internal_frames = False
        frames.append((filename, f.f_lineno, code.co_name, None))
        f = f.f_back

    frames.reverse()
    return tuple(frames)


def format_trace_set(trace, header=True) -> str:
    """Format provenance using the legacy PyTensor diagnostic wording."""
    traces = as_trace_set(trace)
    if not traces:
        return ""
    stream = StringIO()
    if header:
        print(" \nBacktrace when that variable is created:\n", file=stream)
    for idx, stack in enumerate(traces):
        if len(traces) > 1:
            print(f"trace {idx}", file=stream)
        traceback.print_list([resolve_frame(frame) for frame in stack], stream)
    return stream.getvalue()


def get_trace_frame(trace, trace_index=0, frame_index=-1):
    """Return one lazily formatted source frame, supporting historical traces."""
    traces = as_trace_set(trace)
    if not traces:
        return None
    return resolve_frame(traces[trace_index][frame_index])


TRACEBACK_SKIP_PATHS = (
    "pytensor/tensor/",
    "pytensor\\tensor\\",
    "pytensor/compile/",
    "pytensor\\compile\\",
    "pytensor/graph/",
    "pytensor\\graph\\",
    "pytensor/scalar/basic.py",
    "pytensor\\scalar\\basic.py",
    "pytensor/scan/",
    "pytensor\\scan\\",
    "pytensor/sparse/",
    "pytensor\\sparse\\",
    "pytensor/typed_list/",
    "pytensor\\typed_list\\",
)
