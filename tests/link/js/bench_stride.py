"""Interleaved V8 A/B: dynamic-stride reads inside versus outside the loop."""

import json
import statistics
import sys

import numpy as np

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode, get_mode
from pytensor.link.js.dispatch import elemwise
from pytensor.link.js.dispatch.basic import lower
from pytensor.link.js.node_runtime import NodeJSFunction


results = []
for n in (4, 1000, 10000):
    x, y, beta = pt.vector("x"), pt.vector("y"), pt.scalar("beta")
    logp = pt.sum(-0.5 * (y - x * beta) ** 2)
    mode = Mode(linker="py", optimizer=get_mode("JS").optimizer)
    graph = pytensor.function(
        [x, y, beta], [logp, pt.grad(logp, beta)], mode=mode
    ).maker.fgraph

    newer = lower(graph)
    original_setup = elemwise.stride_setup
    original_address = elemwise.address
    try:
        elemwise.stride_setup = lambda var, record, prefix: []
        elemwise.address = (
            lambda record, rank, own_rank=None, indexed=None, strides=None: (
                original_address(record, rank, own_rank, indexed, None)
            )
        )
        older = lower(graph)
    finally:
        elemwise.stride_setup = original_setup
        elemwise.address = original_address

    old = NodeJSFunction(older)
    new = NodeJSFunction(newer)
    inputs = (np.arange(n, dtype="float64") / 3, np.linspace(-1, 2, n), np.array(0.37))
    for a, b in zip(old(*inputs), new(*inputs), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    count = {4: 100000, 1000: 10000, 10000: 1000}[n]
    old.repeat(count)
    new.repeat(count)
    samples = {"old": [], "new": []}
    for round_index in range(9):
        order = (("old", old), ("new", new))
        if round_index % 2:
            order = order[::-1]
        for name, function in order:
            samples[name].append(function.repeat(count))
    old.close()
    new.close()
    median_old = statistics.median(samples["old"])
    median_new = statistics.median(samples["new"])
    results.append(
        {
            "n": n,
            "old_us": median_old,
            "new_us": median_new,
            "old_over_new": median_old / median_new,
            "samples": samples,
        }
    )
sys.stdout.write(json.dumps(results, indent=2) + "\n")
