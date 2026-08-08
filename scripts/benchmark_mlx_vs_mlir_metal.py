# ruff: noqa: T201

"""Benchmark equivalent PyTensor MLX and MLIR Metal float32 graphs on Apple Silicon.

The timings intentionally include the host NumPy input to device and device to host
output behavior visible to a PyTensor caller.  See the methodology printed by the
script for what each timing column includes.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from statistics import median
from time import perf_counter

import iree.runtime
import mlx.core as mx
import numpy as np

import pytensor
import pytensor.tensor as pt


ATOL = 1e-5
BATCH = 512
COLD_SAMPLES = 5
FEATURES = 256
OUTPUTS = 64
RTOL = 1e-5
SEED = 20260729
SMALL_COLUMNS = 192
SMALL_ROWS = 128
STEADY_CALLS_PER_REPEAT = 20
STEADY_REPEATS = 9
STEADY_WARMUP_CALLS = 3
BACKENDS = ("MLX", "MLIR_METAL")


@dataclass(frozen=True)
class SymbolicGraph:
    inputs: tuple
    outputs: tuple


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    description: str
    shapes: str
    values: tuple[np.ndarray, ...]
    build_graph: Callable[[], SymbolicGraph]


@dataclass(frozen=True)
class TimingResult:
    case: str
    backend: str
    build_samples: tuple[float, ...]
    first_call_samples: tuple[float, ...]
    cold_total_samples: tuple[float, ...]
    steady_call_samples: tuple[float, ...]


def float32_values(
    rng: np.random.Generator, *shapes: tuple[int, ...]
) -> tuple[np.ndarray, ...]:
    return tuple(rng.standard_normal(shape, dtype=np.float32) for shape in shapes)


def make_elementwise_case(rng: np.random.Generator) -> BenchmarkCase:
    shape = (SMALL_ROWS, SMALL_COLUMNS)
    values = float32_values(rng, shape, shape, shape)

    def build_graph() -> SymbolicGraph:
        x = pt.matrix("elementwise_x", dtype="float32", shape=shape)
        y = pt.matrix("elementwise_y", dtype="float32", shape=shape)
        z = pt.matrix("elementwise_z", dtype="float32", shape=shape)
        output = (x + y) * (z + x) + y * z
        return SymbolicGraph((x, y, z), (output,))

    return BenchmarkCase(
        name="elementwise_chain",
        description="(X + Y) * (Z + X) + Y * Z",
        shapes=f"X, Y, Z: {shape}; output: {shape}",
        values=values,
        build_graph=build_graph,
    )


def make_matrix_case(rng: np.random.Generator) -> BenchmarkCase:
    x_shape = (BATCH, FEATURES)
    w_shape = (FEATURES, OUTPUTS)
    b_shape = (OUTPUTS,)
    output_shape = (BATCH, OUTPUTS)
    values = float32_values(rng, x_shape, w_shape, b_shape, output_shape)

    def build_graph() -> SymbolicGraph:
        x = pt.matrix("matrix_x", dtype="float32", shape=x_shape)
        w = pt.matrix("matrix_w", dtype="float32", shape=w_shape)
        b = pt.vector("matrix_b", dtype="float32", shape=b_shape)
        residual = pt.matrix("matrix_residual", dtype="float32", shape=output_shape)
        output = (pt.dot(x, w) + b) * residual + residual
        return SymbolicGraph((x, w, b, residual), (output,))

    return BenchmarkCase(
        name="matrix_bias_chain",
        description="(X @ W + b) * residual + residual",
        shapes=(
            f"X: {x_shape}; W: {w_shape}; b: {b_shape}; residual/output: {output_shape}"
        ),
        values=values,
        build_graph=build_graph,
    )


def make_logistic_vjp_case(rng: np.random.Generator) -> BenchmarkCase:
    x_shape = (BATCH, FEATURES)
    w_shape = (FEATURES, OUTPUTS)
    b_shape = (OUTPUTS,)
    output_shape = (BATCH, OUTPUTS)
    values = float32_values(rng, x_shape, w_shape, b_shape, output_shape)

    def build_graph() -> SymbolicGraph:
        x = pt.matrix("logistic_x", dtype="float32", shape=x_shape)
        w = pt.matrix("logistic_w", dtype="float32", shape=w_shape)
        b = pt.vector("logistic_b", dtype="float32", shape=b_shape)
        cotangent = pt.matrix("logistic_cotangent", dtype="float32", shape=output_shape)
        activations = pt.sigmoid(pt.dot(x, w) + b)
        cost = (activations * cotangent).sum()
        grad_w = pt.grad(cost, w)
        return SymbolicGraph((x, w, b, cotangent), (activations, grad_w))

    return BenchmarkCase(
        name="logistic_forward_vjp",
        description=(
            "outputs sigmoid(X @ W + b) and grad_W of "
            "sum(sigmoid(X @ W + b) * cotangent)"
        ),
        shapes=(
            f"X: {x_shape}; W: {w_shape}; b: {b_shape}; "
            f"cotangent/activation: {output_shape}; grad_W: {w_shape}"
        ),
        values=values,
        build_graph=build_graph,
    )


def materialize_outputs(result, backend: str) -> tuple[np.ndarray, ...]:
    outputs = result if isinstance(result, (list, tuple)) else (result,)
    if backend == "MLX":
        mx.eval(*outputs)
    return tuple(np.asarray(output) for output in outputs)


def compile_function(graph: SymbolicGraph, backend: str):
    return pytensor.function(list(graph.inputs), list(graph.outputs), mode=backend)


def invoke(function, case: BenchmarkCase, backend: str) -> tuple[np.ndarray, ...]:
    return materialize_outputs(function(*case.values), backend)


def assert_correctness(case: BenchmarkCase, backend: str) -> None:
    reference = compile_function(case.build_graph(), "FAST_COMPILE")
    expected = invoke(reference, case, "FAST_COMPILE")
    actual = invoke(compile_function(case.build_graph(), backend), case, backend)

    if len(actual) != len(expected):
        raise AssertionError(
            f"{case.name} {backend} returned {len(actual)} outputs; expected {len(expected)}"
        )
    for output_index, (observed, wanted) in enumerate(
        zip(actual, expected, strict=True)
    ):
        np.testing.assert_allclose(
            observed,
            wanted,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{case.name} {backend} output {output_index}",
        )


def time_cold_sample(case: BenchmarkCase, backend: str) -> tuple[float, float]:
    graph = case.build_graph()
    build_start = perf_counter()
    function = compile_function(graph, backend)
    build_seconds = perf_counter() - build_start

    first_call_start = perf_counter()
    invoke(function, case, backend)
    first_call_seconds = perf_counter() - first_call_start
    return build_seconds, first_call_seconds


def time_steady_calls(case: BenchmarkCase, backend: str) -> tuple[float, ...]:
    function = compile_function(case.build_graph(), backend)
    invoke(function, case, backend)
    for _ in range(STEADY_WARMUP_CALLS):
        invoke(function, case, backend)

    samples = []
    for _ in range(STEADY_REPEATS):
        start = perf_counter()
        for _ in range(STEADY_CALLS_PER_REPEAT):
            invoke(function, case, backend)
        samples.append((perf_counter() - start) / STEADY_CALLS_PER_REPEAT)
    return tuple(samples)


def benchmark_case(case: BenchmarkCase, backend: str) -> TimingResult:
    build_samples = []
    first_call_samples = []
    for _ in range(COLD_SAMPLES):
        build_seconds, first_call_seconds = time_cold_sample(case, backend)
        build_samples.append(build_seconds)
        first_call_samples.append(first_call_seconds)

    cold_totals = tuple(
        build_seconds + first_call_seconds
        for build_seconds, first_call_seconds in zip(
            build_samples, first_call_samples, strict=True
        )
    )
    return TimingResult(
        case=case.name,
        backend=backend,
        build_samples=tuple(build_samples),
        first_call_samples=tuple(first_call_samples),
        cold_total_samples=cold_totals,
        steady_call_samples=time_steady_calls(case, backend),
    )


def format_ms(samples: tuple[float, ...]) -> str:
    values = tuple(sample * 1_000 for sample in samples)
    return f"{median(values):.3f} [{min(values):.3f}, {max(values):.3f}]"


def format_raw_ms(samples: tuple[float, ...]) -> str:
    return ", ".join(f"{sample * 1_000:.3f}" for sample in samples)


def package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not installed"


def print_environment(cases: tuple[BenchmarkCase, ...]) -> None:
    driver = iree.runtime.get_driver("metal")
    devices = driver.query_available_devices()
    if not devices:
        raise RuntimeError("IREE Metal driver reported no available devices")

    print("Environment")
    print(f"  Python: {sys.version.split()[0]}; interpreter: {sys.executable}")
    print(
        f"  NumPy: {np.__version__}; PYTENSOR_FLAGS: {os.environ.get('PYTENSOR_FLAGS', '<unset>')}"
    )
    print(f"  PyTensor: {pytensor.__version__}")
    print(f"  MLX: {mx.__version__}; default device: {mx.default_device()}")
    print(
        "  IREE: "
        f"compiler={package_version('iree-base-compiler')}; "
        f"runtime={package_version('iree-base-runtime')}; Metal driver=metal"
    )
    print(f"  IREE Metal devices: {devices}")
    print(f"  NumPy seed: {SEED}; dtype: float32")
    print("  Cases:")
    for case in cases:
        print(f"    {case.name}: {case.description}")
        print(f"      {case.shapes}")
    print()


def print_methodology() -> None:
    print("Methodology")
    print(
        "  Correctness gate: every MLX and MLIR_METAL output is compared with a "
        f"FAST_COMPILE NumPy result using np.testing.assert_allclose(rtol={RTOL}, atol={ATOL}) "
        "before timing."
    )
    print(
        "  Each backend is compiled from the same static-shape expression for a case and "
        "receives the same seeded NumPy float32 arrays."
    )
    print(
        "  Timing scope: each call starts with the same host NumPy float32 inputs and ends "
        "with host NumPy outputs. MLX outputs are synchronized with mx.eval and then "
        "materialized with np.asarray; MLIR_METAL already returns owned NumPy outputs."
    )
    print(
        f"  Cold samples: {COLD_SAMPLES} fresh symbolic graphs and pytensor.function calls "
        "per backend and case; build and first fully synchronized call are timed separately."
    )
    print(
        "  Cold samples do not clear process-level Python, MLX, IREE, or Metal driver/compiler "
        "caches; imports and the correctness gate occur before timing."
    )
    print(
        f"  Steady state: one first call plus {STEADY_WARMUP_CALLS} warm-up calls, then "
        f"{STEADY_REPEATS} perf_counter timeit-style repeated batches of "
        f"{STEADY_CALLS_PER_REPEAT} fully synchronized calls."
    )
    print("  Values are milliseconds; summaries are median [minimum, maximum].")
    print()


def print_results(results: tuple[TimingResult, ...]) -> None:
    headers = (
        "case",
        "backend",
        "correctness",
        "build",
        "first call",
        "cold total",
        "steady call",
    )
    rows = [
        (
            result.case,
            result.backend,
            "PASS",
            format_ms(result.build_samples),
            format_ms(result.first_call_samples),
            format_ms(result.cold_total_samples),
            format_ms(result.steady_call_samples),
        )
        for result in results
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    print("Results (milliseconds; median [minimum, maximum])")
    print(
        "  "
        + "  ".join(
            header.ljust(width) for header, width in zip(headers, widths, strict=True)
        )
    )
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  "
            + "  ".join(
                value.ljust(width) for value, width in zip(row, widths, strict=True)
            )
        )

    print("\nRaw samples (milliseconds)")
    for result in results:
        print(
            f"  {result.case} / {result.backend}: "
            f"build=[{format_raw_ms(result.build_samples)}]; "
            f"first=[{format_raw_ms(result.first_call_samples)}]; "
            f"cold_total=[{format_raw_ms(result.cold_total_samples)}]; "
            f"steady_per_call=[{format_raw_ms(result.steady_call_samples)}]"
        )
    print("\nNo statistical-significance claims are made from these samples.")


def main() -> None:
    rng = np.random.default_rng(SEED)
    cases = (
        make_elementwise_case(rng),
        make_matrix_case(rng),
        make_logistic_vjp_case(rng),
    )
    print_environment(cases)
    print_methodology()

    for case in cases:
        for backend in BACKENDS:
            assert_correctness(case, backend)
    print("Correctness gate: PASS for every backend and case.\n")

    results = tuple(
        benchmark_case(case, backend) for case in cases for backend in BACKENDS
    )
    print_results(results)


if __name__ == "__main__":
    main()
