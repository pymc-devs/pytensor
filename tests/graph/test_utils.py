from types import SimpleNamespace
from unittest.mock import patch

import pytest

import pytensor
from pytensor.graph.utils import simple_extract_stack
from pytensor.tensor.type import vector


def test_stack_trace():
    with pytensor.config.change_flags(traceback__limit=1):
        v = vector()
        assert len(v.tag.trace) == 1
        assert len(v.tag.trace[0]) == 1

    with pytensor.config.change_flags(traceback__limit=2):
        v = vector()
        assert len(v.tag.trace) == 1
        assert len(v.tag.trace[0]) == 2


@pytest.mark.parametrize("limit", [0, 1, 2, None])
@pytest.mark.parametrize("skip_internal", [True, False])
def test_extract_stack_skips(limit, skip_internal):
    frames = [
        ("pytensor/tensor/type.py", 1, "inner", "inner source"),
        ("tests/tensor/test_type.py", 2, "user", "user source"),
        ("pytensor/compile/builders.py", 3, "outer", "outer source"),
    ]
    frame = None
    for filename, lineno, name, _ in reversed(frames):
        frame = SimpleNamespace(
            f_code=SimpleNamespace(co_filename=filename, co_name=name),
            f_lineno=lineno,
            f_globals={},
            f_back=frame,
        )
    expected = frames[1:] if skip_internal else frames
    expected = expected[:limit]
    sources = {filename: f"  {source}  " for filename, _, _, source in frames}
    with patch(
        "pytensor.graph.utils.linecache.getline",
        side_effect=lambda filename, *_: sources[filename],
    ) as getline:
        trace = simple_extract_stack(frame, limit=limit, skip_internal=skip_internal)

    assert trace == expected[::-1]
    assert [call.args[0] for call in getline.call_args_list] == [
        filename for filename, *_ in expected
    ]
