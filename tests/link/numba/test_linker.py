from textwrap import dedent

import pytest

from pytensor import function
from pytensor.compile.mode import Mode
from pytensor.link.numba import NumbaLinker
from pytensor.tensor import vector


pytest.importorskip("numba")


def test_debug_mode(capsys):
    x = vector("x")
    y = (x + 1).sum()

    debug_mode = Mode(linker=NumbaLinker(debug=True))
    fn = function([x], y, mode=debug_mode)

    assert fn([0, 1]) == 3.0
    captured = capsys.readouterr()
    assert captured.out == dedent(
        """
        Op:  Add
            inputs:  [1.] [0. 1.]
            outputs:  [1. 2.]

        Op:  Sum{axes=None}
            inputs:  [1. 2.]
            outputs:  3.0
        """
    )
