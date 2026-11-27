"""What kernels check about the arguments they are called with:
every `PtArray` carries the number of axes and the dtype of its array, and a call the
number of its inputs and outputs."""

import numpy as np
import pytest

import pytensor.tensor as pt


ir = pytest.importorskip("pytensor_llvm.ir")
runtime = pytest.importorskip("pytensor_llvm.runtime")

from pytensor.link.llvm.codegen.kernel import build_kernel_wrapper  # noqa: E402
from pytensor.link.llvm.codegen.kernel_module import (  # noqa: E402
    CodegenOptions,
    KernelModule,
)
from pytensor.link.llvm.codegen.node import build_node_function  # noqa: E402


f64 = np.dtype("float64")


def call(kernel, inputs, outputs):
    return ir.call_kernel(kernel.address, inputs, outputs, kernel.config)


def sort(mode, input_layout=None):
    """A kernel sorting the rows of a matrix, made for an input with `input_layout`."""
    return runtime.Sort(mode, f64, 1, True, [input_layout], [2])


def test_rust_kernels_run_as_they_were_made():
    values = np.array([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]])
    (result,) = call(sort(runtime.OutputMode.Allocated), [values], [(f64, 2)])
    np.testing.assert_array_equal(result, np.sort(values, axis=1))
    assert result.flags.c_contiguous
    # made for an input in column-major order: so is what it allocates
    column_major = ([1, 0], ["P", "P"])
    (result,) = call(
        sort(runtime.OutputMode.Allocated, column_major), [values], [(f64, 2)]
    )
    np.testing.assert_array_equal(result, np.sort(values, axis=1))
    assert result.flags.f_contiguous
    # into memory the caller provides, with strides of its own
    memory = np.empty((3, 2)).T
    call(sort(runtime.OutputMode.Provided), [values], [memory])
    np.testing.assert_array_equal(memory, np.sort(values, axis=1))


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
    with pytest.raises(RuntimeError, match=message):
        call(sort(runtime.OutputMode.Allocated), inputs, outputs)


def test_rust_kernels_reject_missing_or_other_memory():
    values = np.zeros((2, 3))
    provided = sort(runtime.OutputMode.Provided)
    # (no memory, and a shape that is still all zeros: not the array it writes)
    with pytest.raises(RuntimeError, match=r"expected \[2, 3\]"):
        call(provided, [values], [(f64, 2)])
    with pytest.raises(RuntimeError, match=r"it has shape \[3, 2\]"):
        call(provided, [values], [np.zeros((3, 2))])


def test_generated_kernels_check_their_arguments_when_asked():
    x = pt.dmatrix("x")
    unit = KernelModule.new("argument_check", CodegenOptions(check_layouts=True))
    nf = build_node_function(unit, pt.exp(x).owner, "node")
    build_kernel_wrapper(unit, nf)
    kernel = unit.module.compile().address("kernel")
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
