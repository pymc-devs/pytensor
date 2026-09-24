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


def _evaluate_in_node(tmp_path, fgraph, calls):
    if shutil.which("node") is None:
        pytest.skip("Node.js is required to test the emitted JavaScript")
    source = tmp_path / "function.js"
    values = tmp_path / "inputs.json"
    program = lower(fgraph)
    source.write_text(program.source)
    values.write_text(
        json.dumps(
            {
                "calls": [
                    [
                        {"shape": list(x.shape), "data": x.ravel().tolist()}
                        for x in inputs
                    ]
                    for inputs in calls
                ],
                "constants": [
                    {"shape": list(x.shape), "data": x.ravel().tolist()}
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
        pytensor.function([x], pt.gammaln(x), mode="JS")


def test_js_mode_uses_its_own_rewrites():
    mode = get_mode("JS")
    assert "js" in mode.provided_optimizer.include
    assert "numba" not in mode.provided_optimizer.include
    assert "js" in mode.linker.required_rewrites


def test_js_typify_refuses_inexact_int64():
    value = np.array([2**53], dtype="int64")
    with pytest.raises(ValueError, match="exact Number range"):
        js_typify(value, "int64")


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
