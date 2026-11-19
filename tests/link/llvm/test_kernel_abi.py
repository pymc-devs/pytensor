"""What kernels check about the arguments they are called with:
every `PtArray` carries the number of axes and the dtype of its array, and a call the
number of its inputs and outputs."""

import numpy as np
import pytest

import pytensor.tensor as pt


ir = pytest.importorskip("pytensor_llvm.ir")
interp = pytest.importorskip("pytensor_llvm.interp")

from pytensor.link.llvm.codegen.function import (  # noqa: E402
    build_kernel_wrapper,
    build_node_function,
)


f64 = np.dtype("float64")


def call(kernel, inputs, outputs):
    return ir.call_kernel(kernel.address, inputs, outputs, kernel.config)


def test_rust_kernels_run_as_they_were_made():
    values = np.array([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]])
    sort = interp.Sort(interp.OutputMode.Allocated, f64, 1, True)
    (result,) = call(sort, [values], [(f64, 2)])
    np.testing.assert_array_equal(result, np.sort(values, axis=1))
    # into memory the caller provides, with strides of its own
    sort = interp.Sort(interp.OutputMode.Provided, f64, 1, True)
    memory = np.empty((3, 2)).T
    call(sort, [values], [memory])
    np.testing.assert_array_equal(memory, np.sort(values, axis=1))
    # a view of the input
    transpose = interp.DimShuffle([1, 0])
    (result,) = call(transpose, [values], [(f64, 2)])
    np.testing.assert_array_equal(result, values.T)


@pytest.mark.parametrize(
    "inputs, outputs, message",
    [
        # arrays of another dtype than the kernel's
        ([np.zeros((2, 3), "float32")], [(f64, 2)], "input 0: an array of float64"),
        ([np.zeros((2, 3), "int64")], [(f64, 2)], "input 0: an array of float64"),
        ([np.zeros((2, 3))], [(np.dtype("float32"), 2)], "output 0: it is an array"),
        # an output with another number of axes
        ([np.zeros((2, 3))], [(f64, 1)], "output 0: the output has 1 axes"),
        # other numbers of arguments
        ([], [(f64, 2)], "takes 1 inputs and 1 outputs"),
        ([np.zeros((2, 3))] * 2, [(f64, 2)], "takes 1 inputs and 1 outputs"),
        ([np.zeros((2, 3))], [], "takes 1 inputs and 1 outputs"),
        # memory for an output the kernel returns
        ([np.zeros((2, 3))], [np.zeros((2, 3))], "provides memory"),
    ],
)
def test_rust_kernels_reject_other_arguments(inputs, outputs, message):
    sort = interp.Sort(interp.OutputMode.Allocated, f64, 1, True)
    with pytest.raises(RuntimeError, match=message):
        call(sort, inputs, outputs)


def test_rust_kernels_reject_missing_or_other_memory():
    values = np.zeros((2, 3))
    sort = interp.Sort(interp.OutputMode.Provided, f64, 1, True)
    # (no memory, and a shape that is still all zeros: not the array it writes)
    with pytest.raises(RuntimeError, match=r"expected \[2, 3\]"):
        call(sort, [values], [(f64, 2)])
    with pytest.raises(RuntimeError, match=r"it has shape \[3, 2\]"):
        call(sort, [values], [np.zeros((3, 2))])
    transpose = interp.DimShuffle([1, 0])
    with pytest.raises(RuntimeError, match="provides memory"):
        call(transpose, [values], [np.zeros((3, 2))])
    with pytest.raises(RuntimeError, match="axis 1 of an array with 1 axes"):
        call(transpose, [np.zeros(3)], [(f64, 2)])


def test_generated_kernels_check_their_arguments_when_asked():
    x = pt.dmatrix("x")
    module = ir.Module("argument_check")
    nf = build_node_function(module, pt.exp(x).owner, "node")
    build_kernel_wrapper(module, nf, "kernel", check_layouts=True)
    kernel = module.compile(cache=False).address("kernel")
    values = np.ones((2, 3))
    (result,) = ir.call_kernel(kernel, [values], [(f64, 2)])
    np.testing.assert_allclose(result, np.exp(values))
    for inputs, outputs, message in [
        (
            [values.astype("float32")],
            [(f64, 2)],
            "argument 0 isn't a 2-d array of float64",
        ),
        ([values[0]], [(f64, 2)], "argument 0 isn't a 2-d array of float64"),
        ([values], [(f64, 1)], "argument 1 isn't a 2-d array of float64"),
        ([values, values], [(f64, 2)], "the kernel takes 1 inputs and 1 outputs"),
    ]:
        with pytest.raises(RuntimeError, match=message):
            ir.call_kernel(kernel, inputs, outputs)


def test_calls_between_kernels_pass_the_checks():
    """Generated code that calls a Rust kernel (a dot inside a loop body) fills in what
    the kernel checks."""
    import pytensor
    from pytensor.compile.mode import get_mode

    x = pt.tensor("x", shape=(4, 3))
    w = pt.dvector("w")
    out = pytensor.scan(
        lambda row, acc: acc + pt.sort(row * w).sum() + pt.dot(row, w),
        sequences=[x],
        outputs_info=[pt.zeros(())],
        return_updates=False,
    )
    f = pytensor.function([x, w], out, mode=get_mode("LLVM"))
    xv, wv = np.arange(12.0).reshape(4, 3), np.array([1.0, -2.0, 0.5])
    expected = np.cumsum([np.sort(r * wv).sum() + r @ wv for r in xv])
    np.testing.assert_allclose(f(xv, wv), expected)
