import inspect
import os
import tempfile
import time
from functools import singledispatch
from pathlib import Path

import numpy as np
import pytest

from pytensor import config
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.link.utils import (
    clear_old_generated_src,
    compile_function_src,
    fgraph_to_python,
    get_name_for_object,
    unique_name_generator,
    write_generated_src,
)
from pytensor.scalar.basic import Add, float64
from pytensor.tensor import constant
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.type import scalar, vector
from pytensor.tensor.type_other import NoneConst


@singledispatch
def to_python(op, **kwargs):
    raise NotImplementedError()


@to_python.register(Elemwise)
def to_python_Elemwise(op, **kwargs):
    scalar_op = op.scalar_op
    return to_python(scalar_op, **kwargs)


@to_python.register(Add)
def to_python_Add(op, **kwargs):
    def add(*args):
        return np.add(*args)

    return add


def test_fgraph_to_python_names():
    import inspect

    x = scalar("1x")
    y = scalar("_")
    z = float64()
    q = scalar("def")
    r = NoneConst

    out_fg = FunctionGraph([x, y, z, q, r], [x, y, z, q, r], clone=False)
    out_jx = fgraph_to_python(out_fg, to_python)

    sig = inspect.signature(out_jx)
    assert (
        "tensor_variable",
        "_",
        "scalar_variable",
        "tensor_variable_1",
        r.name,
    ) == tuple(sig.parameters)
    assert (1, 2, 3, 4, 5) == out_jx(1, 2, 3, 4, 5)

    obj = object()
    assert get_name_for_object(obj) == type(obj).__name__


def test_fgraph_to_python_once():
    """Make sure that an output is only computed once when it's referenced multiple times."""

    x = vector("x")
    y = vector("y")

    class TestOp(Op):
        def __init__(self):
            self.called = 0

        def make_node(self, *args):
            return Apply(self, list(args), [x.type() for x in args])

        def perform(self, inputs, outputs):
            for i, inp in enumerate(inputs):
                outputs[i][0] = inp[0]

    @to_python.register(TestOp)
    def to_python_TestOp(op, **kwargs):
        def func(*args, op=op):
            op.called += 1
            return list(args)

        return func

    op1 = TestOp()
    op2 = TestOp()

    q, r = op1(x, y)
    outs = op2(q + r, q + r)

    out_fg = FunctionGraph([x, y], outs, clone=False)
    assert len(out_fg.outputs) == 2

    out_py = fgraph_to_python(out_fg, to_python)

    x_val = np.r_[1, 2].astype(config.floatX)
    y_val = np.r_[2, 3].astype(config.floatX)

    res = out_py(x_val, y_val)
    assert len(res) == 2
    assert op1.called == 1
    assert op2.called == 1

    res = out_py(x_val, y_val)
    assert len(res) == 2
    assert op1.called == 2
    assert op2.called == 2


def test_fgraph_to_python_multiline_str():
    """Make sure that multiline `__str__` values are supported by `fgraph_to_python`."""

    x = vector("x")
    y = vector("y")

    class TestOp(Op):
        def __init__(self):
            super().__init__()

        def make_node(self, *args):
            return Apply(self, list(args), [x.type() for x in args])

        def perform(self, inputs, outputs):
            for i, inp in enumerate(inputs):
                outputs[i][0] = inp[0]

        def __str__(self):
            return "Test\nOp()"

    @to_python.register(TestOp)
    def to_python_TestOp(op, **kwargs):
        def func(*args, op=op):
            return list(args)

        return func

    op1 = TestOp()
    op2 = TestOp()

    q, r = op1(x, y)
    outs = op2(q + r, q + r)

    out_fg = FunctionGraph([x, y], outs, clone=False)
    assert len(out_fg.outputs) == 2

    out_py = fgraph_to_python(out_fg, to_python)

    out_py_src = inspect.getsource(out_py)

    assert (
        """
    # Add(Test
    # Op().0, Test
    # Op().1)
    """
        in out_py_src
    )


def test_fgraph_to_python_constant_outputs():
    """Make sure that constant outputs are handled properly."""

    y = constant(1)

    out_fg = FunctionGraph([], [y], clone=False)

    out_py = fgraph_to_python(out_fg, to_python)

    assert out_py()[0] is y.data


def test_fgraph_to_python_constant_inputs():
    x = constant([1.0])
    y = vector("y")

    out = x + y
    out_fg = FunctionGraph(outputs=[out], clone=False)

    out_py = fgraph_to_python(out_fg, to_python, storage_map=None)

    res = out_py(2.0)
    assert res == (3.0,)

    storage_map = {out: [None], x: [np.r_[2.0]], y: [None]}
    out_py = fgraph_to_python(out_fg, to_python, storage_map=storage_map)

    res = out_py(2.0)
    assert res == (4.0,)


def test_unique_name_generator():
    unique_names = unique_name_generator(["blah"], suffix_sep="_")

    x = vector("blah")
    x_name = unique_names(x)
    assert x_name == "blah_1"

    y = vector("blah_1")
    y_name = unique_names(y)
    assert y_name == "blah_1_1"

    # Make sure that the old name associations are still good
    x_name = unique_names(x)
    assert x_name == "blah_1"
    y_name = unique_names(y)
    assert y_name == "blah_1_1"

    # Try a name that overlaps with the original name
    z = vector("blah")
    z_name = unique_names(z)
    assert z_name == "blah_2"

    # Try a name that overlaps with an extended name
    w = vector("blah_1")
    w_name = unique_names(w)
    assert w_name == "blah_1_2"

    q = vector()
    q_name_1 = unique_names(q)
    q_name_2 = unique_names(q)

    assert q_name_1 == q_name_2 == "tensor_variable"

    unique_names = unique_name_generator()

    r = vector()
    r_name_1 = unique_names(r)
    r_name_2 = unique_names(r, force_unique=True)

    assert r_name_1 != r_name_2

    r_name_3 = unique_names(r)
    assert r_name_2 == r_name_3


SRC_ONE = "def one():\n    return 1\n"


@pytest.fixture
def generated_src_dir(tmp_path):
    dirname = tmp_path / "generated_src"
    with config.change_flags(generated_src_dir=str(dirname)):
        yield dirname


def _make_old(path):
    long_ago = time.time() - 60 * 60 * 24 * 365
    os.utime(path, (long_ago, long_ago))


def test_write_generated_src_is_content_addressed(generated_src_dir):
    first = write_generated_src(SRC_ONE)
    second = write_generated_src(SRC_ONE)

    assert first == second
    # One file and no leftover .tmp from the atomic write.
    assert list(generated_src_dir.iterdir()) == [Path(first)]
    assert Path(first).read_text() == SRC_ONE
    assert write_generated_src("def one():\n    return 2\n") != first


def test_compile_function_src_leaves_nothing_in_tempdir(
    generated_src_dir, tmp_path, monkeypatch
):
    """Generated source used to pile up in TMPDIR, one file per compiled function."""
    tempdir = tmp_path / "tmp"
    tempdir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tempdir))
    monkeypatch.setenv("TMPDIR", str(tempdir))

    fn = compile_function_src(SRC_ONE, "one")

    assert fn() == 1
    assert list(tempdir.iterdir()) == []


@pytest.mark.parametrize("src_dir", ["", "generated_src"])
def test_compile_function_src_source_stays_readable(src_dir, tmp_path):
    """Tracebacks and Numba's type inference read the source back from disk."""
    with config.change_flags(generated_src_dir=src_dir and str(tmp_path / src_dir)):
        fn = compile_function_src(SRC_ONE, "one")

    assert inspect.getsource(fn) == SRC_ONE


def test_write_generated_src_rewrites_truncated_file(generated_src_dir):
    """A file truncated by an earlier crash must not be reused for good."""
    path = Path(write_generated_src(SRC_ONE))
    path.write_text("")

    assert Path(write_generated_src(SRC_ONE)) == path
    assert path.read_text() == SRC_ONE


def test_clear_old_generated_src(generated_src_dir):
    fresh = Path(write_generated_src(SRC_ONE))
    stale = Path(write_generated_src("def two():\n    return 2\n"))
    orphan = generated_src_dir / f"{'0' * 32}.py.partial.tmp"
    orphan.write_text("partially written")
    # The directory can be shared, so files PyTensor did not write are kept.
    unrelated = generated_src_dir / "unrelated.py"
    unrelated.write_text("x")
    for path in (stale, orphan, unrelated):
        _make_old(path)

    clear_old_generated_src()

    assert fresh.exists()
    assert not stale.exists()
    assert not orphan.exists()
    assert unrelated.exists()


def test_write_generated_src_keeps_reused_file_young(generated_src_dir):
    """Reuse refreshes the mtime, so the pruner spares source still in use."""
    path = Path(write_generated_src(SRC_ONE))
    _make_old(path)

    write_generated_src(SRC_ONE)
    clear_old_generated_src()

    assert path.exists()
