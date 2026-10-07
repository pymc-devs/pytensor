"""The PyTensor rewrite graph lowered to V8; shapes can change between calls."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode, get_mode
from pytensor.link.js.dispatch.basic import js_typify
from pytensor.link.js.linker import JSLinker, lower
from pytensor.raise_op import CheckAndRaise
from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise


def _evaluate_in_node(tmp_path, fgraph, calls):
    if shutil.which("node") is None:
        pytest.skip("Node.js is required to test the emitted JavaScript")
    source = tmp_path / "function.js"
    values = tmp_path / "inputs.json"
    program = lower(fgraph)
    source.write_text(program.source)

    def serialize(array):
        return [
            str(value) if not np.isfinite(value) else float(value)
            for value in array.ravel()
        ]

    values.write_text(
        json.dumps(
            {
                "calls": [
                    [{"shape": list(x.shape), "data": serialize(x)} for x in inputs]
                    for inputs in calls
                ],
                "constants": [
                    {
                        "shape": list(x.shape),
                        "dtype": str(x.dtype),
                        "data": serialize(x),
                    }
                    for x in program.constants
                ],
            }
        )
    )
    result = subprocess.run(
        [
            "node",
            str(Path(__file__).with_name("run_generated.mjs")),
            str(source),
            str(values),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        [np.asarray(out["data"]).reshape(out["shape"]) for out in results]
        for results in json.loads(result.stdout)
    ]


def test_dynamic_shapes_and_gradient(tmp_path):
    x, y, beta = pt.vector("x"), pt.vector("y"), pt.scalar("beta")
    logp = pt.sum(-0.5 * (y - x * beta) ** 2)
    grad = pt.grad(logp, beta)
    mode = Mode(linker="py", optimizer=get_mode("JS").optimizer)
    compiled = pytensor.function([x, y, beta], [logp, grad], mode=mode)
    graph = compiled.maker.fgraph
    assert any("FusedElemwise" in type(n.op).__name__ for n in graph.toposort())
    calls = [
        [np.arange(n, dtype="float64") / 3, np.linspace(-1, 2, n), np.array(0.37)]
        for n in (3, 11)
    ]
    for arrays, actual in zip(
        calls, _evaluate_in_node(tmp_path, graph, calls), strict=True
    ):
        expected = compiled(*arrays)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-12)


def test_node_fused_indexed_reduction(tmp_path):
    x, idx = pt.vector("x"), pt.ivector("idx")
    out = pt.sum(pt.exp(x[idx]))
    mode = Mode(linker="py", optimizer=get_mode("JS").optimizer)
    compiled = pytensor.function([x, idx], out, mode=mode)
    graph = compiled.maker.fgraph
    assert any("FusedElemwise" in type(n.op).__name__ for n in graph.toposort())
    calls = [
        [np.arange(size, dtype="float64") / 10, np.asarray(take, dtype="int32")]
        for size, take in [(6, [1, 4, -1]), (15, [4, 3, 2, 1, 0])]
    ]
    for arrays, [actual] in zip(
        calls, _evaluate_in_node(tmp_path, graph, calls), strict=True
    ):
        np.testing.assert_allclose(actual, compiled(*arrays), atol=1e-12)


def test_uniform_switch_preserves_branches_and_dynamic_shapes():
    condition = pt.scalar("condition", dtype="bool")
    x = pt.matrix("x")
    value = pt.switch(condition, pt.log(x), pt.exp(x))
    fn = pytensor.function(
        [condition, x], [value.sum(), pt.grad(value.sum(), x)], mode="JS"
    )
    try:
        assert "if (a" in fn.vm.jit_fn.program.source
        for flag, array in [
            (True, np.array([[1.0, 1000.0], [0.5, 4.0]])[:, ::-1]),
            (False, np.array([[-1000.0, -40.0, 0.0]])),
            (True, np.empty((0, 2))),
            (False, np.array([[-2.0], [1.0]])),
        ]:
            actual, gradient = fn(flag, array)
            expected = np.log(array) if flag else np.exp(array)
            expected_gradient = 1 / array if flag else np.exp(array)
            np.testing.assert_allclose(actual, expected.sum(), atol=1e-12)
            np.testing.assert_allclose(gradient, expected_gradient, atol=1e-12)
    finally:
        fn.vm.jit_fn.close()


def test_switch_numeric_nan_condition():
    condition = pt.scalar("condition")
    x = pt.vector("x")
    fn = pytensor.function(
        [condition, x], pt.switch(condition, x + 1, x - 1).sum(), mode="JS"
    )
    try:
        for flag in [0.0, -0.0, 1.0, -1.0, np.nan]:
            expected = np.arange(3.0) + (1 if bool(flag) else -1)
            np.testing.assert_allclose(fn(flag, np.arange(3.0)), expected.sum())
    finally:
        fn.vm.jit_fn.close()


def test_shared_softplus_preserves_small_tail_values():
    x = pt.vector("x")
    positive, negative = pt.softplus(x), pt.softplus(-x)
    fn = pytensor.function([x], [positive, negative], mode="JS")
    try:
        values = np.array([-1000.0, -100.0, -40.0, 0.0, 40.0, 100.0, 1000.0])
        for actual, expected in zip(
            fn(values), [np.logaddexp(0, values), np.logaddexp(0, -values)], strict=True
        ):
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0)
    finally:
        fn.vm.jit_fn.close()


def test_reduction_fuses_through_parameter_check():
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for the JS linker")
    values = pt.vector("values")
    valid = pt.scalar("valid", dtype="bool")
    checked = CheckAndRaise(ValueError, "valid")(pt.exp(values), valid)
    fn = pytensor.function([values, valid], pt.sum(checked), mode="JS")
    assert any(
        isinstance(node.op, FusedElemwise)
        and any(spec is not None for spec in node.op.reduced_outputs)
        for node in fn.maker.fgraph.toposort()
    )
    np.testing.assert_allclose(
        fn(np.arange(7, dtype="float64"), True), np.exp(np.arange(7)).sum()
    )
    with pytest.raises(RuntimeError, match="parameter check failed"):
        fn(np.arange(7, dtype="float64"), False)
    fn.vm.jit_fn.close()


def test_js_linker_dynamic_shapes():
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for the JS linker")
    x, y = pt.vector("x"), pt.vector("y")
    f = pytensor.function([x, y], pt.sum((x - y) ** 2), mode="JS")
    for n in (2, 7, 3):
        a, b = np.arange(n, dtype="float64"), np.arange(n, dtype="float64") / 4
        np.testing.assert_allclose(f(a, b), np.sum((a - b) ** 2))
    assert isinstance(f.maker.linker, JSLinker)
    assert f.vm.jit_fn.repeat(100) > 0
    f.vm.jit_fn.close()


def test_js_linker_strided_output():
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for the JS linker")
    x = pt.matrix("x")
    f = pytensor.function([x], x.T, mode="JS")
    for shape in ((2, 3), (4, 2)):
        value = np.arange(np.prod(shape), dtype="float64").reshape(shape)
        np.testing.assert_array_equal(f(value), value.T)
    f.vm.jit_fn.close()


def test_js_linker_recovers_from_evaluation_error():
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for the JS linker")
    x, idx = pt.vector("x"), pt.ivector("idx")
    f = pytensor.function([x, idx], pt.sum(x[idx]), mode="JS")
    values = np.arange(5, dtype="float64")
    with pytest.raises(RuntimeError, match="index out of bounds"):
        f(values, np.array([10], dtype="int32"))
    np.testing.assert_allclose(f(values, np.array([1, 3], dtype="int32")), 4)
    f.vm.jit_fn.close()


def test_js_linker_rejects_unsupported_op():
    x = pt.vector("x")
    with pytest.raises(NotImplementedError, match=r"JS (Op|scalar Op|FusedElemwise)"):
        pytensor.function([x], pt.erf(x), mode="JS")


def test_js_mode_uses_its_own_rewrites():
    mode = get_mode("JS")
    assert "js" in mode.provided_optimizer.include
    assert "numba" not in mode.provided_optimizer.include
    assert "js" in mode.linker.required_rewrites


def test_js_typify_refuses_inexact_int64():
    value = np.array([2**53], dtype="int64")
    with pytest.raises(ValueError, match="exact Number range"):
        js_typify(value, "int64")


def test_special_functions():
    scipy = pytest.importorskip("scipy.special")
    x = pt.vector("x")
    fn = pytensor.function([x], [pt.gammaln(x), pt.psi(x)], mode="JS")
    try:
        values = np.array(
            [
                -40.5,
                -5.25,
                -1,
                -0.5,
                -0.0,
                0.0,
                1e-10,
                0.1,
                0.99,
                1,
                2,
                3.7,
                4,
                10,
                100,
                1e4,
                1e50,
                np.inf,
                -np.inf,
                np.nan,
            ]
        )
        for array in (values, values[:6], np.empty(0)):
            for actual, expected in zip(
                fn(array), (scipy.gammaln(array), scipy.digamma(array)), strict=True
            ):
                np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    finally:
        fn.vm.jit_fn.close()


def test_tensor_shape_and_cumulative_ops():
    x = pt.matrix("x")
    view = x[:, ::-1]
    outputs = [
        pt.concatenate([view, view], axis=1),
        pt.cumsum(view, axis=0),
        pt.cumprod(view, axis=1),
        pt.arange(x.shape[0], dtype="int64"),
    ]
    fn = pytensor.function([x], outputs, mode="JS")
    try:
        for shape in ((3, 4), (5, 2), (0, 3)):
            values = np.arange(np.prod(shape), dtype="float64").reshape(shape) / 20
            sliced = values[:, ::-1]
            expected = [
                np.concatenate([sliced, sliced], axis=1),
                sliced.cumsum(axis=0),
                sliced.cumprod(axis=1),
                np.arange(shape[0]),
            ]
            for actual, wanted in zip(fn(values), expected, strict=True):
                np.testing.assert_allclose(actual, wanted, atol=1e-12)
    finally:
        fn.vm.jit_fn.close()


@pytest.mark.parametrize("axis", [0, 1])
def test_join_distinct_inputs(axis):
    from pytensor.tensor.basic import Join

    if shutil.which("node") is None:
        pytest.skip("Node.js is required for the JS linker")
    x, y = pt.matrix("x"), pt.matrix("y")
    fn = pytensor.function([x, y], pt.concatenate([x, y], axis=axis), mode="JS")
    assert any(isinstance(node.op, Join) for node in fn.maker.fgraph.toposort())
    try:
        for rows, cols in [(3, 2), (5, 4), (0, 2)]:
            shape = (rows, cols) if axis == 0 else (cols, rows)
            other = (2, shape[1]) if axis == 0 else (shape[0], 2)
            left = np.arange(np.prod(shape), dtype="float64").reshape(shape)
            right = -np.arange(np.prod(other), dtype="float64").reshape(other) - 1
            np.testing.assert_array_equal(
                fn(left, right), np.concatenate([left, right], axis=axis)
            )
    finally:
        fn.vm.jit_fn.close()


def test_scalar_casts():
    x = pt.vector("x")
    dtypes = ["bool", "int8", "int16", "int32", "uint8", "uint16", "uint32", "float32"]
    fn = pytensor.function([x], [x.astype(dtype) for dtype in dtypes], mode="JS")
    try:
        values = np.array([-65537.25, -257.5, -0.5, 0, 0.5, 257.5, 65537.25])
        for actual, dtype in zip(fn(values), dtypes, strict=True):
            np.testing.assert_array_equal(actual, values.astype(dtype))
    finally:
        fn.vm.jit_fn.close()


def test_basic_indexing_views_and_gradient():
    x = pt.matrix("x")
    view = x[1::2, ::-1]
    out = pt.exp(view)
    outputs = [view, view.reshape((-1,)), out.sum(axis=0), pt.grad(out.sum(), x)]
    fn = pytensor.function([x], outputs, mode="JS")
    try:
        for shape in ((4, 3), (7, 5), (0, 3)):
            value = np.arange(np.prod(shape), dtype="float64").reshape(shape) / 20
            sliced = value[1::2, ::-1]
            gradient = np.zeros_like(value)
            gradient[1::2, ::-1] = np.exp(sliced)
            expected = [sliced, sliced.ravel(), np.exp(sliced).sum(axis=0), gradient]
            for actual, wanted in zip(fn(value), expected, strict=True):
                np.testing.assert_allclose(actual, wanted, atol=1e-12)
    finally:
        fn.vm.jit_fn.close()


def test_fused_matrix_gather_partial_reduction_and_scatter():
    x, indices = pt.matrix("x"), pt.ivector("indices")
    out = pt.exp(x[indices]).sum(axis=1)
    fn = pytensor.function([x, indices], [out, pt.grad(out.sum(), x)], mode="JS")
    assert any(
        isinstance(node.op, FusedElemwise) for node in fn.maker.fgraph.toposort()
    )
    try:
        for shape, take in [((4, 3), [1, 1, -1]), ((7, 5), [4, 0]), ((4, 3), [])]:
            value = np.arange(np.prod(shape), dtype="float64").reshape(shape) / 20
            take = np.asarray(take, dtype="int32")
            gradient = np.zeros_like(value)
            np.add.at(gradient, take, np.exp(value[take]))
            expected = [np.exp(value[take]).sum(axis=1), gradient]
            for actual, wanted in zip(fn(value, take), expected, strict=True):
                np.testing.assert_allclose(actual, wanted, atol=1e-12)
    finally:
        fn.vm.jit_fn.close()


def test_multiple_index_axes_and_duplicates():
    x, rows, cols = pt.matrix("x"), pt.ivector("rows"), pt.ivector("cols")
    selected = pt.exp(x[rows, cols])
    fn = pytensor.function(
        [x, rows, cols], [selected, pt.grad(selected.sum(), x)], mode="JS"
    )
    try:
        values = np.arange(12, dtype="float64").reshape(3, 4) / 20
        row = np.array([1, 1, -1], dtype="int32")
        col = np.array([2, 2, 0], dtype="int32")
        gradient = np.zeros_like(values)
        np.add.at(gradient, (row, col), np.exp(values[row, col]))
        actual, actual_gradient = fn(values, row, col)
        np.testing.assert_allclose(actual, np.exp(values[row, col]), atol=1e-12)
        np.testing.assert_allclose(actual_gradient, gradient, atol=1e-12)
    finally:
        fn.vm.jit_fn.close()


@pytest.mark.parametrize("lower", [True, False])
def test_dense_linalg_and_gradient(lower):
    from pytensor.tensor.linalg import cholesky, solve_triangular

    x, rhs = pt.matrix("x"), pt.matrix("rhs")
    factor = cholesky(x @ x.T, lower=lower)
    solution = solve_triangular(factor, rhs, lower=lower)
    outputs = [factor, solution, pt.grad(pt.log(pt.diagonal(factor)).sum(), x)]
    reference = pytensor.function([x, rhs], outputs, mode="FAST_RUN")
    fn = pytensor.function([x, rhs], outputs, mode="JS")
    try:
        rng = np.random.default_rng(458)
        for n in (2, 3):
            values, b = rng.normal(size=(n, n + 2)), rng.normal(size=(n, 4))
            for actual, expected in zip(
                fn(values, b), reference(values, b), strict=True
            ):
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
        for actual, expected in zip(
            fn(np.zeros((2, 4)), np.zeros((2, 4))),
            reference(np.zeros((2, 4)), np.zeros((2, 4))),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, equal_nan=True)
    finally:
        fn.vm.jit_fn.close()


def test_pymc_logp_gradient_with_checks_and_multiple_outputs(tmp_path):
    if shutil.which("node") is None:
        pytest.skip("Node.js is required to test the emitted JavaScript")
    pm = pytest.importorskip("pymc")
    with pm.Model() as model:
        beta0 = pm.Normal("beta0", 0, 2)
        beta1 = pm.Normal("beta1", 0, 2)
        probability = pm.math.sigmoid(
            beta0 + beta1 * np.arange(12, dtype="float64") / 12
        )
        pm.Binomial("y", n=5, p=probability, observed=np.arange(12) % 6)
    logp = model.logp()
    variables = model.value_vars
    compiled = pytensor.function(
        variables,
        [logp, *pt.grad(logp, variables)],
        mode=Mode(linker="py", optimizer=get_mode("JS").optimizer),
    )
    assert any(
        isinstance(node.op, FusedElemwise)
        and sum(spec is not None for spec in node.op.reduced_outputs) >= 3
        for node in compiled.maker.fgraph.toposort()
    )
    points = [
        [np.array(0.2), np.array(-0.1)],
        [np.array(-0.8), np.array(0.7)],
    ]
    for point, actual in zip(
        points, _evaluate_in_node(tmp_path, compiled.maker.fgraph, points), strict=True
    ):
        for value, expected in zip(actual, compiled(*point), strict=True):
            np.testing.assert_allclose(value, expected, atol=1e-11)


def test_pymc_slice_sampler_in_node():
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for the JS sampler")
    pytest.importorskip("pymc")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("pymc_slice_demo.py"))],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    assert abs(summary["mean"] - summary["expected_mean"]) < 0.03
    assert abs(summary["sd"] - summary["expected_sd"]) < 0.03
    assert summary["evaluations"] > 11_000


@pytest.mark.parametrize("axis", [0, 1])
def test_fused_partial_reduction_locals(axis):
    x = pt.matrix("x")
    view = x[:, ::-1]
    fn = pytensor.function(
        [x],
        [pt.exp(view).sum(axis=axis), (pt.exp(view) ** 2).sum(axis=axis)],
        mode="JS",
    )
    assert any(
        isinstance(node.op, FusedElemwise)
        and any(spec and len(spec[1]) == 1 for spec in node.op.reduced_outputs)
        for node in fn.maker.fgraph.toposort()
    )
    try:
        for shape in [(3, 4), (0, 4), (4, 0), (5, 2)]:
            values = np.arange(np.prod(shape), dtype="float64").reshape(shape) / 20
            exp = np.exp(values[:, ::-1])
            for actual, expected in zip(
                fn(values), [exp.sum(axis), (exp**2).sum(axis)], strict=True
            ):
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    finally:
        fn.vm.jit_fn.close()


def test_compact_integer_constants_and_output_transport():
    x = pt.vector("x")
    indices = np.array([2, 0, -1], dtype="int64")
    limits = np.array([-(2**31), 2**31 - 1], dtype="int64")
    wide = np.array([2**31, 2**40], dtype="int64")
    fn = pytensor.function(
        [x],
        [pt.exp(x[indices]).sum(), pt.constant(limits), pt.constant(wide)],
        mode="JS",
    )
    constants = fn.vm.jit_fn.program.constants
    assert any(
        value.dtype == "int32" and np.array_equal(value, indices) for value in constants
    )
    assert any(
        value.dtype == "float64" and np.array_equal(value, wide) for value in constants
    )
    try:
        for n in [3, 7]:
            values = np.arange(n, dtype="float64") / 10
            actual = fn(values)
            np.testing.assert_allclose(actual[0], np.exp(values[indices]).sum())
            np.testing.assert_array_equal(actual[1], limits)
            np.testing.assert_array_equal(actual[2], wide)
            assert actual[1].dtype == actual[2].dtype == np.dtype("int64")
    finally:
        fn.vm.jit_fn.close()


def test_fused_index_bounds_for_single_element_target():
    x, indices = pt.matrix("x"), pt.ivector("indices")
    fn = pytensor.function(
        [x, indices], pt.grad(pt.exp(x[indices]).sum(), x), mode="JS"
    )
    try:
        value = np.ones((1, 3))
        np.testing.assert_allclose(
            fn(value, np.array([0, -1], dtype="int32")), 2 * np.exp(value)
        )
        with pytest.raises(RuntimeError, match="index out of bounds"):
            fn(value, np.array([1], dtype="int32"))
        np.testing.assert_allclose(
            fn(value, np.array([0], dtype="int32")), np.exp(value)
        )
    finally:
        fn.vm.jit_fn.close()


def test_inplace_elemwise_on_strided_copy():
    from pytensor.compile.ops import deep_copy_op
    from pytensor.scalar.basic import add
    from pytensor.tensor.elemwise import Elemwise

    x, y = pt.matrix("x"), pt.matrix("y")
    view = deep_copy_op(x)[:, ::-1]
    result = Elemwise(add, {0: 0})(view, y)
    fn = pytensor.function(
        [x, y],
        [result, x],
        mode=Mode(linker=JSLinker(), optimizer=None),
        accept_inplace=True,
    )
    assert any(
        isinstance(node.op, Elemwise) and node.op.inplace_pattern
        for node in fn.maker.fgraph.toposort()
    )
    try:
        for shape in [(3, 4), (5, 2), (0, 3)]:
            values = np.arange(np.prod(shape), dtype="float64").reshape(shape)
            addend = values / 10
            out, original = fn(values, addend)
            np.testing.assert_allclose(out, values[:, ::-1] + addend)
            np.testing.assert_array_equal(original, values)
    finally:
        fn.vm.jit_fn.close()


@pytest.mark.parametrize("ignore_duplicates", [False, True])
@pytest.mark.parametrize("set_instead", [False, True])
def test_inplace_advanced_update_strided_target(ignore_duplicates, set_instead):
    from pytensor.tensor.subtensor import AdvancedIncSubtensor

    x, y, indices = pt.vector("x"), pt.vector("y"), pt.ivector("indices")
    base = pt.exp(x)[::-1]
    result = AdvancedIncSubtensor(
        (0,),
        inplace=True,
        set_instead_of_inc=set_instead,
        ignore_duplicates=ignore_duplicates,
    )(base, y, indices)
    fn = pytensor.function(
        [x, y, indices],
        result,
        mode=Mode(linker=JSLinker(), optimizer=None),
        accept_inplace=True,
    )
    assert any(
        isinstance(node.op, AdvancedIncSubtensor) and node.op.inplace
        for node in fn.maker.fgraph.toposort()
    )
    try:
        for n, take in [(5, [1, 1, -1]), (3, [0, -1]), (0, [])]:
            values = np.arange(n, dtype="float64") / 10
            takes = np.array(take, dtype="int32")
            addend = np.arange(len(take), dtype="float64") + 1
            expected = np.exp(values)[::-1].copy()
            if set_instead:
                expected[takes] = addend
            elif ignore_duplicates:
                expected[takes] += addend
            else:
                np.add.at(expected, takes, addend)
            np.testing.assert_allclose(fn(values, addend, takes), expected)
    finally:
        fn.vm.jit_fn.close()


def test_inplace_basic_update_strided_value():
    from pytensor.tensor.subtensor import IncSubtensor

    x = pt.vector("x")
    base = pt.exp(x)[::-1]
    view = base[::2]
    result = IncSubtensor(
        view.owner.op.idx_list, inplace=True, destroyhandler_tolerate_aliased=((0, 1),)
    )(base, base[1::2], *view.owner.inputs[1:])
    fn = pytensor.function(
        [x], result, mode=Mode(linker=JSLinker(), optimizer=None), accept_inplace=True
    )
    assert any(
        isinstance(node.op, IncSubtensor) and node.op.inplace
        for node in fn.maker.fgraph.toposort()
    )
    try:
        for n in [6, 2, 0]:
            values = np.arange(n, dtype="float64") / 10
            expected = np.exp(values)[::-1].copy()
            expected[::2] += expected[1::2].copy()
            np.testing.assert_allclose(fn(values), expected)
    finally:
        fn.vm.jit_fn.close()
