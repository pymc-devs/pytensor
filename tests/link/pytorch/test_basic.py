from collections.abc import Callable, Iterable
from functools import partial

import numpy as np
import pytest

import pytensor.tensor.basic as ptb
from pytensor.compile.function import function
from pytensor.compile.mode import get_mode
from pytensor.compile.sharedvalue import SharedVariable, shared
from pytensor.configdefaults import config
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.raise_op import CheckAndRaise
from pytensor.tensor import alloc, arange, as_tensor, empty, eye
from pytensor.tensor.type import matrix, scalar, vector


torch = pytest.importorskip("torch")


pytorch_mode = get_mode("PYTORCH")
py_mode = get_mode("FAST_COMPILE")


def compare_pytorch_and_py(
    fgraph: FunctionGraph,
    test_inputs: Iterable,
    assert_fn: Callable | None = None,
    must_be_device_array: bool = True,
    pytorch_mode=pytorch_mode,
    py_mode=py_mode,
):
    """Function to compare python graph output and pytorch compiled output for testing equality

    Parameters
    ----------
    fgraph: FunctionGraph
        PyTensor function Graph object
    test_inputs: iter
        Numerical inputs for testing the function graph
    assert_fn: func, opt
        Assert function used to check for equality between python and pytorch. If not
        provided uses np.testing.assert_allclose
    must_be_device_array: Bool
        Checks if torch.device.type is cuda


    """
    if assert_fn is None:
        assert_fn = partial(np.testing.assert_allclose)

    fn_inputs = [i for i in fgraph.inputs if not isinstance(i, SharedVariable)]

    pytensor_torch_fn = function(fn_inputs, fgraph.outputs, mode=pytorch_mode)
    pytorch_res = pytensor_torch_fn(*test_inputs)

    if must_be_device_array:
        if isinstance(pytorch_res, list):
            assert all(isinstance(res, torch.Tensor) for res in pytorch_res)
        else:
            assert pytorch_res.device.type == "cuda"

    pytensor_py_fn = function(fn_inputs, fgraph.outputs, mode=py_mode)
    py_res = pytensor_py_fn(*test_inputs)

    if len(fgraph.outputs) > 1:
        for j, p in zip(pytorch_res, py_res, strict=True):
            assert_fn(j.cpu(), p)
    else:
        assert_fn([pytorch_res[0].cpu()], py_res)

    return pytensor_torch_fn, pytorch_res


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pytorch_FunctionGraph_once(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    """Make sure that an output is only computed once when it's referenced multiple times."""
    from pytensor.link.pytorch.dispatch import pytorch_funcify

    with torch.device(device):
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

        @pytorch_funcify.register(TestOp)
        def pytorch_funcify_TestOp(op, **kwargs):
            def func(*args, op=op):
                op.called += 1
                for arg in args:
                    assert arg.device.type == device
                return list(args)

            return func

        op1 = TestOp()
        op2 = TestOp()

        q, r = op1(x, y)
        outs = op2(q + r, q + r)

        out_fg = FunctionGraph([x, y], outs, clone=False)
        assert len(out_fg.outputs) == 2

        out_torch = pytorch_funcify(out_fg)

        x_val = torch.tensor([1, 2]).to(getattr(torch, config.floatX))
        y_val = torch.tensor([2, 3]).to(getattr(torch, config.floatX))

        res = out_torch(x_val, y_val)

        for output in res:
            assert torch.equal(
                output, torch.tensor([3, 5]).to(getattr(torch, config.floatX))
            )

        assert len(res) == 2
        assert op1.called == 1
        assert op2.called == 1

        res = out_torch(x_val, y_val)

        for output in res:
            assert torch.equal(
                output, torch.tensor([3, 5]).to(getattr(torch, config.floatX))
            )

        assert len(res) == 2
        assert op1.called == 2
        assert op2.called == 2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    with torch.device(device):
        a = shared(np.array([1, 2, 3], dtype=config.floatX))
        pytensor_torch_fn = function([], a, mode="PYTORCH")
        pytorch_res = pytensor_torch_fn()

        assert isinstance(pytorch_res, torch.Tensor)
        assert isinstance(a.get_value(), np.ndarray)
        np.testing.assert_allclose(pytorch_res.cpu(), a.get_value())

        pytensor_torch_fn = function([], a * 2, mode="PYTORCH")
        pytorch_res = pytensor_torch_fn()

        assert isinstance(pytorch_res, torch.Tensor)
        assert isinstance(a.get_value(), np.ndarray)
        np.testing.assert_allclose(pytorch_res.cpu(), a.get_value() * 2)

        new_a_value = np.array([3, 4, 5], dtype=config.floatX)
        a.set_value(new_a_value)

        pytorch_res = pytensor_torch_fn()
        assert isinstance(pytorch_res, torch.Tensor)
        np.testing.assert_allclose(pytorch_res.cpu(), new_a_value * 2)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_updates(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    with torch.device(device):
        a = shared(0)

        pytensor_torch_fn = function([], a, updates={a: a + 1}, mode="PYTORCH")
        res1, res2 = pytensor_torch_fn(), pytensor_torch_fn()
        assert res1 == 0
        assert res2 == 1
        assert a.get_value() == 2
        assert isinstance(a.get_value(), np.ndarray)

        a.set_value(5)
        res1, res2 = pytensor_torch_fn(), pytensor_torch_fn()
        assert res1 == 5
        assert res2 == 6
        assert a.get_value() == 7
        assert isinstance(a.get_value(), np.ndarray)


def test_checkandraise():
    check_and_raise = CheckAndRaise(AssertionError, "testing")

    x = scalar("x")
    conds = (x > 0, x > 3)
    y = check_and_raise(x, *conds)

    y_fn = function([x], y, mode="PYTORCH")

    with pytest.raises(AssertionError, match="testing"):
        y_fn(0.0)
    assert y_fn(4).item() == 4


def test_alloc_and_empty():
    dim0 = as_tensor(5, dtype="int64")
    dim1 = scalar("dim1", dtype="int64")

    out = empty((dim0, dim1, 3), dtype="float32")
    fn = function([dim1], out, mode=pytorch_mode)
    res = fn(7)
    assert res.shape == (5, 7, 3)
    assert res.dtype == torch.float32

    v = vector("v", shape=(3,), dtype="float64")
    out = alloc(v, (dim0, dim1, 3))
    compare_pytorch_and_py(
        FunctionGraph([v, dim1], [out]),
        [np.array([1, 2, 3]), np.array(7)],
    )


def test_arange():
    start = scalar("start", dtype="int64")
    stop = scalar("stop", dtype="int64")
    step = scalar("step", dtype="int64")

    out = arange(start, stop, step, dtype="int16")

    compare_pytorch_and_py(
        FunctionGraph([start, stop, step], [out]),
        [np.array(1), np.array(10), np.array(2)],
    )


def test_pytorch_Join():
    a = matrix("a")
    b = matrix("b")

    x = ptb.join(0, a, b)
    x_fg = FunctionGraph([a, b], [x])
    compare_pytorch_and_py(
        x_fg,
        [
            np.c_[[1.0, 2.0, 3.0]].astype(config.floatX),
            np.c_[[4.0, 5.0, 6.0]].astype(config.floatX),
        ],
    )
    compare_pytorch_and_py(
        x_fg,
        [
            np.c_[[1.0, 2.0, 3.0]].astype(config.floatX),
            np.c_[[4.0, 5.0]].astype(config.floatX),
        ],
    )

    x = ptb.join(1, a, b)
    x_fg = FunctionGraph([a, b], [x])
    compare_pytorch_and_py(
        x_fg,
        [
            np.c_[[1.0, 2.0, 3.0]].astype(config.floatX),
            np.c_[[4.0, 5.0, 6.0]].astype(config.floatX),
        ],
    )
    compare_pytorch_and_py(
        x_fg,
        [
            np.c_[[1.0, 2.0], [3.0, 4.0]].astype(config.floatX),
            np.c_[[5.0, 6.0]].astype(config.floatX),
        ],
    )


@pytest.mark.parametrize(
    "dtype",
    ["int64", config.floatX],
)
def test_eye(dtype):
    N = scalar("N", dtype="int64")
    M = scalar("M", dtype="int64")
    k = scalar("k", dtype="int64")

    out = eye(N, M, k, dtype=dtype)

    fn = function([N, M, k], out, mode=pytorch_mode)

    for _N in range(1, 6):
        for _M in range(1, 6):
            for _k in list(range(_M + 2)) + [-x for x in range(1, _N + 2)]:
                np.testing.assert_array_equal(fn(_N, _M, _k), np.eye(_N, _M, _k))


def test_pytorch_MakeVector():
    x = ptb.make_vector(1, 2, 3)
    x_fg = FunctionGraph([], [x])

    compare_pytorch_and_py(x_fg, [])
