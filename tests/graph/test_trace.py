import inspect
import pickle

from pytensor.graph.rewriting.basic import copy_stack_trace
from pytensor.graph.trace import TraceSet
from pytensor.graph.utils import get_variable_trace_string, simple_extract_stack
from pytensor.tensor.type import vector


def test_capture_is_lazy_and_tag_trace_is_immutable():
    expected_line = inspect.currentframe().f_lineno + 1
    value = vector("value")

    assert isinstance(value.tag.trace, TraceSet)
    stack = value.tag.trace[0]
    assert stack[-1][0] == __file__
    assert stack[-1][1] == expected_line
    assert stack[-1][3] is None
    assert 'value = vector("value")' in get_variable_trace_string(value)


def test_copy_stack_trace_merges_origins_without_duplicates():
    first = vector("first")
    second = vector("second")

    copy_stack_trace(first, second)
    assert len(second.tag.trace) == 2
    copy_stack_trace(first, second)
    assert len(second.tag.trace) == 2


def test_legacy_trace_format_is_readable_and_trace_set_is_picklable():
    value = vector("value")
    value.tag.trace = [[(__file__, 1, "legacy", "legacy source line")]]

    assert "legacy source line" in get_variable_trace_string(value)

    traces = TraceSet((((__file__, 1, "legacy", None),),))
    assert pickle.loads(pickle.dumps(traces)) == traces


def test_simple_extract_stack_keeps_legacy_source_filled_format():
    trace = simple_extract_stack(limit=1, skips=())

    assert len(trace) == 1
    assert len(trace[0]) == 4
    assert trace[0][3]
