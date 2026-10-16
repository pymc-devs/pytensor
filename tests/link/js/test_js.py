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


def _evaluate_in_node(tmp_path, fgraph, calls, *, disable_wasm=False, check_wasm=False):
    if shutil.which("node") is None:
        pytest.skip("Node.js is required to test the emitted JavaScript")
    source = tmp_path / "function.js"
    values = tmp_path / "inputs.json"
    program = lower(fgraph)
    source.write_text(
        ("globalThis.WebAssembly = undefined;\n" if disable_wasm else "")
        + program.source
    )

    def serialize(array):
        return [
            "NaN"
            if np.isnan(value)
            else "Infinity"
            if value == np.inf
            else "-Infinity"
            if value == -np.inf
            else float(value)
            for value in array.ravel()
        ]

    values.write_text(
        json.dumps(
            {
                "check_wasm": check_wasm,
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
        [
            np.asarray(out["data"], dtype="float64").reshape(out["shape"])
            for out in results
        ]
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


def test_scalar_log_checks_nonfinite_and_signed_zero():
    x = pt.vector("x", shape=(12,))
    factors = np.array([0, -0.0] * 5 + [1, -2], dtype="float64")
    logarithm = pt.log(x)
    outputs = [factors * logarithm, pt.eq(logarithm, -np.inf)]
    fn = pytensor.function([x], outputs, mode="JS")
    try:
        values = np.array([0.5, 0.5, 1, 1, 2, 2, 0, np.inf, -1, np.nan, 2, 3])
        with np.errstate(divide="ignore", invalid="ignore"):
            actual, check = fn(values)
            expected = factors * np.log(values)
            np.testing.assert_allclose(actual, expected, rtol=1e-14, equal_nan=True)
            np.testing.assert_equal(
                np.signbit(actual[actual == 0]), np.signbit(expected[expected == 0])
            )
            np.testing.assert_equal(check, np.log(values) == -np.inf)
    finally:
        fn.vm.jit_fn.close()


def test_static_fused_layout_and_constant_indices():
    x = pt.matrix("x", shape=(4, 6))
    step = pt.iscalar("step")
    source = pt.specify_shape(x[:, ::step], (4, 6))
    indices = np.array([1, 1, 3], dtype="int32")
    cost = pt.square(source[indices]).sum()
    fn = pytensor.function([x, step], [cost, pt.grad(cost, x)], mode="JS")
    try:
        values = np.arange(24, dtype="float64").reshape(4, 6) / 10
        gradient = np.zeros_like(values)
        np.add.at(gradient, indices, 2 * values[indices])
        for stride in (1, -1, 1):
            actual_cost, actual_grad = fn(values, stride)
            np.testing.assert_allclose(actual_cost, np.square(values[indices]).sum())
            np.testing.assert_allclose(actual_grad, gradient)
    finally:
        fn.vm.jit_fn.close()


def test_short_fused_axes_preserve_reduction_order():
    x = pt.matrix("x", shape=(2, 12))
    row = pt.vector("row", shape=(2,))
    step = pt.iscalar("step")
    source = pt.specify_shape(x[:, ::step], (2, 12))
    terms = source + row[:, None]
    fn = pytensor.function([x, row, step], [terms.sum(), terms], mode="JS")
    try:
        values = np.tile([1e16, 1, -1e16, 1], (2, 3))
        offset = np.array([0, 0.25])
        for stride in (1, -1, 1):
            expected = values[:, ::stride] + offset[:, None]
            total, actual = fn(values, offset, stride)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_equal(total, np.cumsum(expected.ravel())[-1])
    finally:
        fn.vm.jit_fn.close()


def test_dynamic_matvec_reduction_order_and_strided_inputs():
    matrix, vector = pt.matrix("matrix"), pt.vector("vector")
    fn = pytensor.function(
        [matrix, vector],
        [matrix @ vector, matrix[:, ::-1] @ vector[::-1]],
        mode="JS",
    )
    try:
        for rows, columns in [(5, 0), (3, 3), (205, 25), (7, 13), (2, 17), (5, 12)]:
            values = np.resize([1e16, 1, -1e16, 1], (rows, columns)).astype("float64")
            weights = np.ones(columns)
            expected = [
                np.cumsum(source, axis=1)[:, -1] if columns else np.zeros(rows)
                for source in (values, values[:, ::-1])
            ]
            for actual, wanted in zip(fn(values, weights), expected, strict=True):
                np.testing.assert_array_equal(actual, wanted)
    finally:
        fn.vm.jit_fn.close()


def test_dynamic_outer_products_and_signed_zero():
    a, b = pt.matrix("a"), pt.matrix("b")
    fn = pytensor.function([a, b], [a @ b, a[::-1, ::-1] @ b[::-1, ::-1]], mode="JS")
    try:
        for rows, extent, columns in [
            (5, 1, 3),
            (257, 1, 17),
            (4, 3, 2),
            (0, 1, 7),
            (3, 0, 5),
        ]:
            left = np.resize([0.0, -0.0, 2, -2, np.inf], (rows, extent))
            right = np.resize([0.0, -0.0, 1, -1], (extent, columns))
            with np.errstate(invalid="ignore"):
                actual = fn(left, right)
                for result, x, y in zip(
                    actual,
                    (left, left[::-1, ::-1]),
                    (right, right[::-1, ::-1]),
                    strict=True,
                ):
                    expected = np.zeros((rows, columns))
                    for index in range(extent):
                        expected += x[:, index, None] * y[index, None, :]
                    np.testing.assert_array_equal(result, expected)
                    np.testing.assert_array_equal(
                        np.signbit(result[result == 0]),
                        np.signbit(expected[expected == 0]),
                    )
    finally:
        fn.vm.jit_fn.close()


def test_plain_sum_preserves_sequential_order():
    x = pt.matrix("x")
    fn = pytensor.function(
        [x],
        [x.sum(), x.sum(0), x.sum(1), x[::-1, ::-1].sum()],
        mode=Mode(linker=JSLinker(), optimizer=None),
    )
    try:
        for rows, columns in [(0, 3), (5, 0), (5, 13), (205, 25)]:
            values = np.resize([1e16, 1, -1e16, 1], (rows, columns))
            expected = [
                np.cumsum(values.ravel())[-1] if values.size else 0,
                np.cumsum(values, axis=0)[-1] if rows else np.zeros(columns),
                np.cumsum(values, axis=1)[:, -1] if columns else np.zeros(rows),
                np.cumsum(values[::-1, ::-1].ravel())[-1] if values.size else 0,
            ]
            for actual, wanted in zip(fn(values), expected, strict=True):
                np.testing.assert_array_equal(actual, wanted)
    finally:
        fn.vm.jit_fn.close()


def test_fused_sum_products_with_dynamic_axes_and_strides(tmp_path):
    x, y = pt.matrix("x"), pt.matrix("y")
    products = [
        x.dimshuffle(0, 1, "x") * y.dimshuffle(0, "x", 1),
        y.dimshuffle("x", 0, 1) * x.dimshuffle(1, 0, "x"),
        x.dimshuffle(1, "x", 0) * y.dimshuffle("x", 1, 0),
    ]
    fn = pytensor.function(
        [x, y],
        [product.sum(axis) for axis, product in enumerate(products)],
        mode=Mode(linker="py", optimizer=get_mode("JS").optimizer),
    )
    calls, expected = [], []
    for rows, left, right in [(0, 3, 7), (205, 3, 7), (1, 2, 3), (257, 5, 9)]:
        a = np.resize([1e16, 1, -1e16, 1], (rows, left))
        b = np.resize([1.0, -1.0, 0.0, -0.0], (rows, right))
        for u, v in [(a, b), (a[::-1, ::-1], b[::-1, ::-1])]:
            calls.append((u, v))
            terms = u[:, :, None] * v[:, None, :]
            expected.append(
                np.cumsum(terms, axis=0)[-1] if rows else np.zeros((left, right))
            )
    results = _evaluate_in_node(tmp_path, fn.maker.fgraph, calls)
    for outputs, wanted in zip(results, expected, strict=True):
        for actual in outputs:
            np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize("disable_wasm", [False, True])
def test_squared_norms_preserve_order_and_wasm_fallback(tmp_path, disable_wasm):
    x = pt.matrix("x")
    fn = pytensor.function(
        [x],
        [pt.sqr(x).sum(0), pt.sqr(x.T).sum(1), pt.sqr(x[:, ::-1]).sum(0)],
        mode=Mode(linker="py", optimizer=get_mode("JS").optimizer),
    )
    calls = []
    for rows, columns in [(205, 25), (257, 17), (0, 5), (5, 0), (1, 5001)]:
        values = np.resize([1e8, 1, -1e8, -0.0, 0.0], (rows, columns))
        calls.append((values,))
    values = calls[0][0].copy()
    values[0, :3] = [np.inf, -np.inf, np.nan]
    calls.append((values,))
    results = _evaluate_in_node(
        tmp_path,
        fn.maker.fgraph,
        calls,
        disable_wasm=disable_wasm,
        check_wasm=not disable_wasm,
    )
    for outputs, (values,) in zip(results, calls, strict=True):
        with np.errstate(invalid="ignore"):
            expected = (
                np.cumsum(values * values, axis=0)[-1]
                if len(values)
                else np.zeros(values.shape[1])
            )
        for actual, wanted in zip(
            outputs, [expected, expected, expected[::-1]], strict=True
        ):
            np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize("disable_wasm", [False, True])
def test_basic_elemwise_simd_and_scalar_broadcasts(tmp_path, disable_wasm):
    x, y, weight = pt.vector("x"), pt.vector("y"), pt.scalar("weight")
    fn = pytensor.function(
        [x, y, weight],
        [
            -x,
            pt.sqr(x),
            x + y,
            x - y,
            x * y,
            x / y,
            x + weight,
            weight - x,
            x * weight,
            weight / x,
        ],
        mode=Mode(linker="py", optimizer=None),
    )
    calls = [
        (
            np.resize([-0.0, 0.0, 2, -2, np.inf, -np.inf, np.nan, 1e308], count),
            np.resize([1.0, -1.0, -0.0, 0.0, 1e-308, np.inf, np.nan], count),
            np.asarray(value),
        )
        for count, value in [
            (5001, -0.0),
            (0, 2.0),
            (1, -1.0),
            (8192, 2.0),
            (5001, 2.0),
        ]
    ]
    results = _evaluate_in_node(
        tmp_path,
        fn.maker.fgraph,
        calls,
        disable_wasm=disable_wasm,
        check_wasm=not disable_wasm,
    )
    for actuals, (a, b, w) in zip(results, calls, strict=True):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            expected = [
                -a,
                a * a,
                a + b,
                a - b,
                a * b,
                a / b,
                a + w,
                w - a,
                a * w,
                w / a,
            ]
        for actual, wanted in zip(actuals, expected, strict=True):
            np.testing.assert_array_equal(actual, wanted)
            np.testing.assert_array_equal(
                np.signbit(actual[actual == 0]), np.signbit(wanted[wanted == 0])
            )


@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize("disable_wasm", [False, True])
def test_negated_solve_graph_fusion(tmp_path, lower, disable_wasm):
    from pytensor.tensor.rewriting.js import NegatedSolveTriangular

    a, b = pt.matrix("a"), pt.matrix("b")
    expression = pt.linalg.solve_triangular(
        a.T, -b[:, ::-1], lower=lower, unit_diagonal=lower
    )
    optimizer = get_mode("JS").optimizer
    fn = pytensor.function(
        [a, b], expression, mode=Mode(linker="py", optimizer=optimizer)
    )
    baseline = pytensor.function(
        [a, b],
        expression,
        mode=Mode(
            linker="py",
            optimizer=get_mode("JS").excluding("js_negated_solve").optimizer,
        ),
    )
    assert any(
        isinstance(node.op, NegatedSolveTriangular)
        for node in fn.maker.fgraph.toposort()
    )
    calls = []
    for n, columns in [(0, 5), (3, 7), (2, 5001), (5, 513), (7, 0)]:
        triangle = np.eye(n) * 2 + np.arange(n * n).reshape(n, n) / max(1, n * n)
        triangle = np.tril(triangle) if lower else np.triu(triangle)
        values = np.resize([0.0, -0.0, 1, -1, 2], (n, columns))
        calls.append((triangle.T.copy(), values))
    wanted = _evaluate_in_node(
        tmp_path, baseline.maker.fgraph, calls, disable_wasm=disable_wasm
    )
    actual = _evaluate_in_node(
        tmp_path,
        fn.maker.fgraph,
        calls,
        disable_wasm=disable_wasm,
        check_wasm=not disable_wasm,
    )
    for [result], [reference] in zip(actual, wanted, strict=True):
        np.testing.assert_allclose(result, reference, atol=1e-14, rtol=1e-14)
        np.testing.assert_array_equal(
            np.signbit(result[result == 0]), np.signbit(reference[reference == 0])
        )


def test_arithmetic_fused_nonfinite_and_dynamic_shapes(tmp_path):
    a, b, row = pt.matrix("a"), pt.matrix("b"), pt.vector("row")
    value = a * row[:, None] * b[:, ::-1]
    valid = value >= 0
    optimizer = get_mode("JS").excluding("inplace").optimizer
    reference = pytensor.function(
        [a, b, row],
        [value, valid, valid.all()],
        mode=Mode(linker="py", optimizer=optimizer),
    )
    calls = []
    for m, n in ((205, 25), (257, 17), (3, 1), (0, 25), (205, 25)):
        x = np.resize(np.array([0.0, -0.0, 1.0, -2.0, np.inf, np.nan]), (m, n))
        y = np.resize(np.array([1.0, -1.0, 0.0]), (m, n))
        rows = np.resize(np.array([1.0, -1.0, 0.0]), m)
        calls.append([x, y, rows])
    actual = _evaluate_in_node(
        tmp_path,
        reference.maker.fgraph,
        calls,
    )
    with np.errstate(invalid="ignore"):
        for inputs, outputs in zip(calls, actual, strict=True):
            for result, expected in zip(outputs, reference(*inputs), strict=True):
                np.testing.assert_equal(result, expected)
                if expected.dtype.kind == "f":
                    np.testing.assert_equal(
                        np.signbit(result[expected == 0]),
                        np.signbit(expected[expected == 0]),
                    )


def test_wasm_constant_operands_and_shape_changes():
    x = pt.matrix("x")
    constant = np.arange(97 * 97, dtype="float64").reshape(97, 97) / 1000
    fn = pytensor.function([x], [constant @ x, x.T @ constant], mode="JS")
    try:
        for columns in (9, 1, 11, 0, 9):
            value = np.linspace(-1, 1, 97 * columns).reshape(97, columns)
            for actual, expected in zip(
                fn(value), [constant @ value, value.T @ constant], strict=True
            ):
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    finally:
        fn.vm.jit_fn.close()


def test_uniform_switch_preserves_branches_and_dynamic_shapes():
    condition = pt.scalar("condition", dtype="bool")
    x = pt.matrix("x")
    value = pt.switch(condition, pt.log(x), pt.exp(x))
    fn = pytensor.function(
        [condition, x], [value.sum(), value.sum(1), pt.grad(value.sum(), x)], mode="JS"
    )
    try:
        assert "if (a" in fn.vm.jit_fn.program.source
        for flag, array in [
            (True, np.array([[1.0, 1000.0], [0.5, 4.0]])[:, ::-1]),
            (False, np.array([[-1000.0, -40.0, 0.0]])),
            (True, np.empty((0, 2))),
            (False, np.array([[-2.0], [1.0]])),
        ]:
            actual, rows, gradient = fn(flag, array)
            expected = np.log(array) if flag else np.exp(array)
            expected_gradient = 1 / array if flag else np.exp(array)
            np.testing.assert_allclose(actual, expected.sum(), atol=1e-12)
            np.testing.assert_allclose(rows, expected.sum(1), atol=1e-12)
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


def test_shared_branches_with_partial_reductions():
    x, y = pt.matrix("x"), pt.matrix("y")
    mask = pt.matrix("mask", dtype="bool")
    selected = (x > 0) & mask
    value = pt.switch(selected, pt.log(x) + y, 0)
    gradient = pt.grad(value.sum(), x)
    product = pt.switch(selected, x * y, 0)
    fn = pytensor.function(
        [x, y, mask], [value.sum(), gradient.sum(1), gradient, product], mode="JS"
    )
    try:
        assert "let branch" in fn.vm.jit_fn.program.source
        for rows, columns in ((5, 7), (0, 3), (3, 0), (2, 11)):
            a = np.resize([0.0, -0.0, 1.0, -1.0, np.inf, np.nan, 2.0], (rows, columns))
            b = np.arange(rows * columns, dtype="float64").reshape(rows, columns)[
                :, ::-1
            ]
            flags = np.resize([True, False, True], (rows, columns))
            with np.errstate(divide="ignore", invalid="ignore"):
                expected = np.where((a > 0) & flags, np.log(a) + b, 0)
                expected_gradient = np.where((a > 0) & flags, 1 / a, 0)
                expected_product = np.where((a > 0) & flags, a * b, 0)
            total, row_gradient, actual_gradient, actual_product = fn(a, b, flags)
            np.testing.assert_equal(total, expected.sum())
            np.testing.assert_equal(row_gradient, expected_gradient.sum(1))
            np.testing.assert_equal(actual_gradient, expected_gradient)
            np.testing.assert_equal(actual_product, expected_product)
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
    for shape in ((2, 3), (4, 2), (2, 4), (2, 3)):
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


def test_op_from_graph_outputs_and_gradient():
    from pytensor.compile.builders import OpFromGraph

    x, y = pt.vectors("x", "y")
    inner = OpFromGraph([x, y], [pt.exp(x + y) + 2, pt.square(x - y).sum()])
    vector, scalar = inner(x, y)
    gradient = pt.grad(vector.sum() + scalar, x)
    fn = pytensor.function([x, y], [vector, scalar, gradient], mode="JS")
    assert any(isinstance(node.op, OpFromGraph) for node in fn.maker.fgraph.toposort())
    try:
        for size in (3, 7):
            a, b = np.arange(size, dtype="float64") / 10, np.linspace(-1, 1, size)
            expected = [
                np.exp(a + b) + 2,
                np.square(a - b).sum(),
                np.exp(a + b) + 2 * (a - b),
            ]
            for actual, wanted in zip(fn(a, b), expected, strict=True):
                np.testing.assert_allclose(actual, wanted)
    finally:
        fn.vm.jit_fn.close()


def test_op_from_graph_keeps_view_outputs_independent():
    from pytensor.compile.builders import OpFromGraph

    x = pt.vector("x")
    inner = OpFromGraph([x], [x, x[::-1]])
    direct, reverse = inner(x)
    fn = pytensor.function([x], [x, direct + 1, reverse + 2], mode="JS")
    try:
        values = np.arange(5, dtype="float64")
        for actual, expected in zip(
            fn(values), [values, values + 1, values[::-1] + 2], strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
    finally:
        fn.vm.jit_fn.close()


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


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_join_distinct_inputs(axis):
    from pytensor.tensor.basic import Join

    if shutil.which("node") is None:
        pytest.skip("Node.js is required for the JS linker")
    x, y = pt.tensor3("x"), pt.tensor3("y")
    fn = pytensor.function(
        [x, y],
        [
            pt.concatenate([x, y], axis=axis),
            pt.concatenate([x[1:], y[1:]], axis=axis),
            pt.concatenate([x[:, :, ::-1], y[:, :, ::-1]], axis=axis),
        ],
        mode="JS",
    )
    assert any(isinstance(node.op, Join) for node in fn.maker.fgraph.toposort())
    try:
        for extent, outer in [(3, 2), (5, 4), (0, 2), (3, 0)]:
            shape = [outer] * 3
            shape[axis] = extent
            other = shape.copy()
            other[axis] = 2
            left = np.arange(np.prod(shape), dtype="float64").reshape(shape)
            right = -np.arange(np.prod(other), dtype="float64").reshape(other) - 1
            expected = [
                np.concatenate([left, right], axis=axis),
                np.concatenate([left[1:], right[1:]], axis=axis),
                np.concatenate([left[:, :, ::-1], right[:, :, ::-1]], axis=axis),
            ]
            for actual, wanted in zip(fn(left, right), expected, strict=True):
                np.testing.assert_array_equal(actual, wanted)
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


def test_broadcast_matrix_indices_and_gradient():
    x = pt.tensor3("x")
    indices = pt.imatrix("indices")
    rows, columns = pt.arange(x.shape[0])[:, None], pt.arange(x.shape[1])[None, :]
    selected = x[rows, columns, indices[:, ::-1]]
    cost = pt.square(selected).sum()
    fn = pytensor.function([x, indices], [selected, pt.grad(cost, x)], mode="JS")
    try:
        for shape in ((3, 2, 4), (2, 5, 3), (0, 2, 4)):
            values = np.arange(np.prod(shape), dtype="float64").reshape(shape) / 10
            take = (
                np.arange(shape[0] * shape[1], dtype="int32").reshape(shape[:2])
                % shape[2]
            )
            take -= 1
            r, c = np.arange(shape[0])[:, None], np.arange(shape[1])[None, :]
            expected = values[r, c, take[:, ::-1]]
            gradient = np.zeros_like(values)
            np.add.at(gradient, (r, c, take[:, ::-1]), 2 * expected)
            actual, grad = fn(values, take)
            np.testing.assert_allclose(actual, expected)
            np.testing.assert_allclose(grad, gradient)
    finally:
        fn.vm.jit_fn.close()


def test_split_strided_views_and_gradient():
    x, lengths = pt.matrix("x"), pt.ivector("lengths")
    selected = x[::2, ::-1]
    parts = pt.split(selected, lengths, n_splits=3, axis=0)
    cost = pt.square(parts[0]).sum() + 3 * pt.square(parts[2]).sum()
    fn = pytensor.function([x, lengths], [*parts, pt.grad(cost, x)], mode="JS")
    try:
        for rows, splits in [(5, [1, 0, 2]), (7, [2, 0, 2])]:
            values = np.arange(rows * 4, dtype="float64").reshape(rows, 4) / 10
            expected_parts = np.split(values[::2, ::-1], np.cumsum(splits[:-1]))
            gradient = np.zeros_like(values)
            gradient[::2] = 2 * values[::2]
            gradient[2 * splits[0] :: 2] *= 3
            for actual, expected in zip(
                fn(values, np.asarray(splits, dtype="int32")),
                [*expected_parts, gradient],
                strict=True,
            ):
                np.testing.assert_allclose(actual, expected)
        with pytest.raises(RuntimeError, match="split sizes"):
            fn(values, np.array([1, 0, 1], dtype="int32"))
    finally:
        fn.vm.jit_fn.close()


def test_fused_scatter_targets_sharing_indices():
    x, y, weights = pt.vector("x"), pt.vector("y"), pt.vector("weights")
    indices = pt.ivector("indices")
    selected = x[indices] + y[indices] * weights
    cost = pt.square(selected).sum()
    fn = pytensor.function(
        [x, y, weights, indices], [cost, *pt.grad(cost, [x, y])], mode="JS"
    )
    assert any(
        isinstance(node.op, FusedElemwise)
        and any(
            spec is not None and len(spec[0]) > 1 for spec in node.op.indexed_outputs
        )
        for node in fn.maker.fgraph.toposort()
    )
    try:
        for take in ([1, 1, -1], [0, 2], []):
            values_x = np.arange(4, dtype="float64") / 10
            values_y = np.arange(4, dtype="float64") + 1
            weights = np.arange(len(take), dtype="float64") + 2
            take = np.asarray(take, dtype="int32")
            selected = values_x[take] + values_y[take] * weights
            grad_x, grad_y = np.zeros_like(values_x), np.zeros_like(values_y)
            np.add.at(grad_x, take, 2 * selected)
            np.add.at(grad_y, take, 2 * selected * weights)
            for actual, expected in zip(
                fn(values_x, values_y, weights, take),
                [np.square(selected).sum(), grad_x, grad_y],
                strict=True,
            ):
                np.testing.assert_allclose(actual, expected, atol=1e-12)
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
        for n in (2, 3, 33):
            values, b = rng.normal(size=(n, n + 2)), rng.normal(size=(n, 9))
            for actual, expected in zip(
                fn(values, b), reference(values, b), strict=True
            ):
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
        for n in (2, 33):
            values, b = np.zeros((n, n + 2)), np.zeros((n, 9))
            for actual, expected in zip(
                fn(values, b), reference(values, b), strict=True
            ):
                np.testing.assert_allclose(actual, expected, equal_nan=True)
    finally:
        fn.vm.jit_fn.close()


@pytest.mark.parametrize("disable_wasm", [False, True])
def test_dense_nodes_in_subgraph_and_wasm_fallback(tmp_path, disable_wasm):
    from pytensor.compile.builders import OpFromGraph
    from pytensor.tensor.linalg import cholesky, solve_triangular

    x, rhs = pt.matrix("x"), pt.matrix("rhs")
    product = x @ x.T
    block = OpFromGraph([x], [product + 1, product.sum()], inline=False)
    factor = cholesky(product)
    outputs = [*block(x), factor, solve_triangular(factor.T, rhs, lower=False)]
    reference = pytensor.function(
        [x, rhs], outputs, mode=Mode(linker="py", optimizer=get_mode("JS").optimizer)
    )
    rng = np.random.default_rng(183)
    calls = [
        [rng.normal(size=(n, n + 2)), rng.normal(size=(n, 11))] for n in (33, 2, 34)
    ]
    for inputs, actual in zip(
        calls,
        _evaluate_in_node(
            tmp_path,
            reference.maker.fgraph,
            calls,
            disable_wasm=disable_wasm,
            check_wasm=True,
        ),
        strict=True,
    ):
        for value, expected in zip(actual, reference(*inputs), strict=True):
            np.testing.assert_allclose(value, expected, rtol=1e-11, atol=1e-12)


def test_clip_bounds_and_gradient():
    x = pt.vector("x")
    lower, upper = pt.scalars("lower", "upper")
    clipped = pt.clip(x, lower, upper)
    fn = pytensor.function(
        [x, lower, upper], [clipped, pt.grad(clipped.sum(), x)], mode="JS"
    )
    try:
        values = np.array([-np.inf, -1, 0, 1, 2, np.inf, np.nan])
        for bounds, result, gradient in [
            ((0, 1), [0, 0, 0, 1, 1, 1, np.nan], [0, 0, 1, 1, 0, 0, 0]),
            ((2, 1), [2, 2, 2, 2, 1, 1, np.nan], [0, 0, 0, 0, 0, 0, 0]),
        ]:
            actual, grad = fn(values, *bounds)
            np.testing.assert_array_equal(actual, result)
            np.testing.assert_array_equal(grad, gradient)
    finally:
        fn.vm.jit_fn.close()


def test_sign_nan_and_zero():
    x = pt.vector("x")
    fn = pytensor.function(
        [x],
        [pt.sign(x), pt.isnan(x), pt.isinf(x), pt.isnan(x) | pt.isinf(x)],
        mode="JS",
    )
    try:
        values = np.array([-np.inf, -1, -0.0, 0.0, 1, np.inf, np.nan])
        result, isnan, isinf, nonfinite = fn(values)
        np.testing.assert_array_equal(result, np.sign(values))
        assert not np.any(np.signbit(result[2:4]))
        np.testing.assert_array_equal(isnan, np.isnan(values))
        np.testing.assert_array_equal(isinf, np.isinf(values))
        np.testing.assert_array_equal(nonfinite, ~np.isfinite(values))
    finally:
        fn.vm.jit_fn.close()


def test_extrema_reductions():
    x = pt.matrix("x")
    fn = pytensor.function([x], [x.max(1), x.min(0), pt.isnan(x).any(1)], mode="JS")
    try:
        values = np.array([[1, -3, np.nan], [2, np.inf, -np.inf]])
        for actual, expected in zip(
            fn(values[:, ::-1]),
            [values[:, ::-1].max(1), values[:, ::-1].min(0), np.isnan(values).any(1)],
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        with pytest.raises(RuntimeError, match="empty extrema"):
            fn(np.empty((2, 0)))
    finally:
        fn.vm.jit_fn.close()


def test_bitwise_or_and_invert():
    x, y = pt.lvectors("x", "y")
    fn = pytensor.function([x, y], [x | y, ~x], mode="JS")
    try:
        a = np.array([0, 3, 2**40, -(2**40)], dtype="int64")
        b = np.array([7, 2**41, -1, 1], dtype="int64")
        for actual, expected in zip(fn(a, b), [a | b, ~a], strict=True):
            np.testing.assert_array_equal(actual, expected)
    finally:
        fn.vm.jit_fn.close()


@pytest.mark.parametrize("n", [1, 2, 3, 4, 33])
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("unit_diagonal", [True, False])
def test_static_triangular_solve_strides_and_empty_rhs(n, lower, unit_diagonal):
    from scipy.linalg import solve_triangular as scipy_solve

    from pytensor.tensor.linalg import solve_triangular

    a = pt.matrix("a", shape=(n, n))
    b = pt.matrix("b", shape=(n, None))
    solution = solve_triangular(
        a.T, b[:, ::2], lower=lower, unit_diagonal=unit_diagonal
    )
    fn = pytensor.function([a, b], solution, mode="JS")
    rng = np.random.default_rng(173)
    matrix = rng.normal(size=(n, n)) + np.eye(n) * (n + 1)
    if unit_diagonal:
        np.fill_diagonal(matrix, 0)
    try:
        for cols in (0, 2, 18):
            rhs = rng.normal(size=(n, cols))
            expected = scipy_solve(
                matrix.T, rhs[:, ::2], lower=lower, unit_diagonal=unit_diagonal
            )
            np.testing.assert_allclose(
                fn(matrix, rhs), expected, rtol=1e-13, atol=1e-14
            )
        if not unit_diagonal:
            matrix[-1, -1] = 0
            # scalar_solve_to_division replaces the 1x1 solve before lowering.
            expected = np.inf if n == 1 else np.nan
            for cols in (2, 18):
                np.testing.assert_array_equal(
                    fn(matrix, np.ones((n, cols))), np.full((n, cols // 2), expected)
                )
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
        [
            pt.exp(view).sum(axis=axis),
            (pt.exp(view) ** 2).sum(axis=axis),
            pt.exp(view).sum(),
            pt.exp(view),
        ],
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
                fn(values),
                [exp.sum(axis), (exp**2).sum(axis), exp.sum(), exp],
                strict=True,
            ):
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    finally:
        fn.vm.jit_fn.close()


def test_static_small_axes_with_strides_and_empty_reductions():
    x = pt.tensor3("x", shape=(None, 2, 3))
    view = x[:, :, ::-1]
    squared = view * (view + 1)
    fn = pytensor.function(
        [x], [view.sum(0), squared.sum(0), squared.sum((1, 2))], mode="JS"
    )
    try:
        for n in [0, 1, 7]:
            values = np.arange(n * 6, dtype="float64").reshape(3, 2, n).T / 10
            expected_view = values[:, :, ::-1]
            expected_squared = expected_view * (expected_view + 1)
            expected = [
                expected_view.sum(0),
                expected_squared.sum(0),
                expected_squared.sum((1, 2)),
            ]
            for actual, reference in zip(fn(values), expected, strict=True):
                np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
    finally:
        fn.vm.jit_fn.close()


def test_small_output_reduction_with_strided_operands():
    x = pt.matrix("x", shape=(None, 2))
    y = pt.matrix("y", shape=(None, 3))
    result = (x[::-1, :, None] * y[::-1, None, ::-1]).sum(0)
    fn = pytensor.function([x, y], result, mode="JS")
    try:
        for n in [0, 1, 7]:
            left = np.arange(n * 2, dtype="float64").reshape(2, n).T / 10
            right = np.arange(n * 3, dtype="float64").reshape(3, n).T / 10
            expected = (left[::-1, :, None] * right[::-1, None, ::-1]).sum(0)
            np.testing.assert_allclose(
                fn(left, right), expected, rtol=1e-12, atol=1e-12
            )
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


def test_vector_scatter_aliased_strided_slices():
    from pytensor.tensor.subtensor import AdvancedIncSubtensor

    x, indices = pt.tensor3("x"), pt.ivector("indices")
    base = pt.exp(x)[:, ::-1, ::2]
    update = AdvancedIncSubtensor((0,), inplace=True)
    update.destroyhandler_tolerate_aliased = ((0, 1),)
    result = update(base, base[1:], indices)
    fn = pytensor.function(
        [x, indices],
        result,
        mode=Mode(linker=JSLinker(), optimizer=None),
        accept_inplace=True,
    )
    try:
        for n, columns, take in [(4, 4, [1, 1, -1]), (1, 4, []), (4, 0, [10, 10, 10])]:
            values = (
                np.arange(n * 2 * columns, dtype="float64").reshape(n, 2, columns) / 100
            )
            expected = np.exp(values)[:, ::-1, ::2].copy()
            if columns:
                np.add.at(expected, take, expected[1:].copy())
            np.testing.assert_allclose(
                fn(values, np.array(take, dtype="int32")), expected
            )
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
