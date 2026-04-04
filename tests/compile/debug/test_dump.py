import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np

from pytensor.compile.debug.dump import function_dump
from pytensor.compile.maker import function
from pytensor.tensor.type import vector


def test_function_dump():
    v = vector()
    fct1 = function([v], v + 1)

    try:
        tmpdir = Path(tempfile.mkdtemp())
        fname = tmpdir / "test_function_dump.pkl"
        function_dump(fname, [v], v + 1)
        with fname.open("rb") as f:
            l = pickle.load(f)
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir)

    fct2 = function(**l)
    x = [1, 2, 3]
    assert np.allclose(fct1(x), fct2(x))
