import numpy as np
import pytest

from pytensor import function, shared


def careduce_benchmark_tester(axis, layout, N, mode, benchmark):
    if layout == "c_contiguous":
        x_test = np.random.uniform(size=(N, N, N))
        transpose_axis = (0, 1, 2)
    elif layout == "transposed":
        x_test = np.random.uniform(size=(N, N, N))
        transpose_axis = (2, 0, 1)
    elif layout == "strided":
        # Non-dense: strided on first axis (forces transposed loop fallback)
        x_test = np.random.uniform(size=(N * 2, N, N))
        transpose_axis = (2, 0, 1)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    x = shared(x_test, name="x", shape=x_test.shape)
    out = (
        (x[::2] if layout == "strided" else x).transpose(transpose_axis).sum(axis=axis)
    )
    fn = function([], out, mode=mode, trust_input=True)

    expected = (
        (x_test[::2] if layout == "strided" else x_test)
        .transpose(transpose_axis)
        .sum(axis=axis)
    )
    np.testing.assert_allclose(fn(), expected)
    benchmark(fn)


@pytest.mark.parametrize(
    "axis",
    (0, 1, 2, (0, 1), (0, 2), (1, 2), None),
    ids=lambda x: f"axis={x}",
)
@pytest.mark.parametrize(
    "layout",
    ("c_contiguous", "transposed", "strided"),
    ids=lambda x: f"layout={x}",
)
def test_careduce_benchmark_c_large(axis, layout, benchmark):
    careduce_benchmark_tester(axis, layout, N=256, mode="CVM", benchmark=benchmark)


@pytest.mark.parametrize(
    "axis",
    (0, 1, 2, (0, 1), (0, 2), (1, 2), None),
    ids=lambda x: f"axis={x}",
)
@pytest.mark.parametrize(
    "layout",
    ("c_contiguous", "transposed", "strided"),
    ids=lambda x: f"layout={x}",
)
def test_careduce_benchmark_c_small(axis, layout, benchmark):
    careduce_benchmark_tester(axis, layout, N=3, mode="CVM", benchmark=benchmark)


@pytest.mark.parametrize(
    "axis",
    (0, 1, 2, (0, 1), (0, 2), (1, 2), None),
    ids=lambda x: f"axis={x}",
)
@pytest.mark.parametrize(
    "layout",
    ("c_contiguous", "transposed", "strided"),
    ids=lambda x: f"layout={x}",
)
def test_careduce_benchmark_numba_large(axis, layout, benchmark):
    careduce_benchmark_tester(axis, layout, N=256, mode="NUMBA", benchmark=benchmark)


@pytest.mark.parametrize(
    "axis",
    (0, 1, 2, (0, 1), (0, 2), (1, 2), None),
    ids=lambda x: f"axis={x}",
)
@pytest.mark.parametrize(
    "layout",
    ("c_contiguous", "transposed", "strided"),
    ids=lambda x: f"layout={x}",
)
def test_careduce_benchmark_numba_small(axis, layout, benchmark):
    careduce_benchmark_tester(axis, layout, N=3, mode="NUMBA", benchmark=benchmark)
