import sys

import pytensor.tensor as pt
from pytensor.compile.rebuild import rebuild_collect_shared


def test_rebuild_collect_shared_deep_graph():
    # Cloning must not recurse, or graphs deeper than the interpreter stack fail
    x = pt.dscalar("x")
    out = x
    for i in range(sys.getrecursionlimit() + 500):
        out = out + i

    input_variables, cloned_outputs, (clone_d, *_) = rebuild_collect_shared([out], [x])
    assert input_variables == [x]
    assert cloned_outputs == [clone_d[out]]
