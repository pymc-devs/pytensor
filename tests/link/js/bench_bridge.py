"""Manual timing of the experimental JS linker's V8 and Python-call paths.

Run from the PyTensor checkout after setting PYTENSOR_FLAGS to a writable
compiledir. Compilation and first-call times are reported separately.
"""

import json
import statistics
import sys
import time

import numpy as np

import pytensor
import pytensor.tensor as pt


def measure_public(call, count):
    samples = []
    for _ in range(5):
        begin = time.perf_counter_ns()
        for _ in range(count):
            call()
        samples.append((time.perf_counter_ns() - begin) / count / 1000)
    return statistics.median(samples)


def bench(n, fixed_data):
    x_data = np.arange(n, dtype="float64") / 3
    y_data = np.linspace(-1, 2, n)
    x = pt.constant(x_data) if fixed_data else pt.vector("x")
    y = pt.constant(y_data) if fixed_data else pt.vector("y")
    beta = pt.scalar("beta")
    logp = pt.sum(-0.5 * (y - x * beta) ** 2)
    inputs = [beta] if fixed_data else [x, y, beta]
    values = [0.37] if fixed_data else [x_data, y_data, 0.37]
    begin = time.perf_counter_ns()
    fn = pytensor.function(inputs, [logp, pt.grad(logp, beta)], mode="JS")
    compile_ms = (time.perf_counter_ns() - begin) / 1e6
    begin = time.perf_counter_ns()
    actual = fn(*values)
    first_call_us = (time.perf_counter_ns() - begin) / 1000
    expected = [
        -0.5 * np.sum((y_data - x_data * 0.37) ** 2),
        np.sum((y_data - x_data * 0.37) * x_data),
    ]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    runtime = fn.vm.jit_fn
    curve = {str(k): runtime.repeat(k) for k in (1, 10, 100, 1000, 10000)}
    engine_us = statistics.median(runtime.repeat(1000) for _ in range(5))
    count = max(100, min(2000, 200000 // n))
    python_call_us = measure_public(lambda: fn(*values), count)
    runtime.close()
    return {
        "elements": n,
        "data_constant": fixed_data,
        "compile_ms": compile_ms,
        "first_python_call_us": first_call_us,
        "v8_warmup_curve_us_per_eval": curve,
        "warm_v8_us_per_eval": engine_us,
        "warm_python_call_us_per_eval": python_call_us,
    }


if __name__ == "__main__":
    rows = [bench(n, fixed) for n in (4, 1000, 10000) for fixed in (False, True)]
    sys.stdout.write(json.dumps(rows, indent=2) + "\n")
