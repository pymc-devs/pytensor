# Experimental JS backend

This is a proof of concept for lowering an ordinary PyTensor graph to JavaScript
and running it in V8. PyTensor still builds and rewrites the graph and computes
gradients. The linker consumes the existing scalar `Composite` and
`FusedElemwise` representation, including vector indexed reads and sum
reductions. It does not use TyMC.

`JS` mode selects `fast_run` and the `js` rewrite tag. The indexed/reduction
fusion rewrites are shared with Numba but registered under both tags; JS does
not select Numba-only rewrites. The linker delegates Op and scalar expression
generation to `dispatch/`, using type-based dispatch and a shared indentation
helper (`pytensor/link/string_codegen.py`). `js_typify` checks supported dtypes
and converts values for the binary transport.

Install Node.js so `node` is on `PATH`, then:

```python
import numpy as np
import pytensor
import pytensor.tensor as pt

x, y, beta = pt.vector("x"), pt.vector("y"), pt.scalar("beta")
logp = pt.sum(-0.5 * (y - beta * x) ** 2)
fn = pytensor.function([x, y, beta], [logp, pt.grad(logp, beta)], mode="JS")

print(fn(np.arange(3.), np.ones(3), 0.2))
print(fn(np.arange(11.), np.ones(11), 0.2))  # same graph, different shape
```

Input rank and dtype are set by PyTensor; axis lengths are read at runtime.
Generated loops and buffer sizes adjust when they change. The default
Python/V8 bridge keeps one Node process alive per compiled function and sends
typed arrays through binary pipes. This avoids per-call Python↔V8 object
dispatch but still copies inputs and outputs across the process boundary.
`fn.vm.jit_fn.source` holds the generated JavaScript, and
`fn.vm.jit_fn.repeat(n)` returns microseconds per evaluation when the already
loaded inputs run repeatedly inside V8, excluding Python and transport.
Call `fn.vm.jit_fn.close()` when finished to release the process promptly.

The supported set is intentionally small: float64 elementary arithmetic and
several transcendental scalar Ops, `Composite`, `DimShuffle`, sum reductions,
and rank-one fused elementwise graphs with optional integer-vector gather and
one or more sum outputs. Int32/int64/bool inputs are used for indexing and
conditions; int64 values outside JavaScript's exact integer range are refused.
Unsupported Ops and scalar casts raise `NotImplementedError` during lowering.
This backend is not yet a replacement for the C or Numba linker: it lacks
general linear algebra, arbitrary indexed writes, `Scan`, many special
functions, and complete dtype semantics. It deliberately reports such gaps
rather than silently falling back to Python.

The included tests cover a single compiled graph called with changing vector
lengths, matching log density and gradient to PyTensor, and a fused indexed
sum with changing index lengths. They assert that PyTensor's fusion pass fired.
They do not claim speed parity with Numba; bridge cost and warm-up need to be
measured separately.

## Node-side slice sampler PoC with a PyMC model

`tests/link/js/pymc_slice_demo.py` builds a one-parameter PyMC Normal model,
compiles its scalar logp with `mode="JS"`, and sends the generated program,
constants, and initial point to `tests/link/js/slice_sampler.mjs` once. Node
loads the program into V8 and runs a stepping-out, shrinkage slice sampler.
All logp evaluations occur in that process; Python receives only the final
draws and statistics. Run with a Python environment containing PyMC:

```bash
python tests/link/js/pymc_slice_demo.py
```

The sampler deliberately supports just one scalar continuous parameter. It is
an example of the backend handoff, not a PyMC sampling method or general NUTS
implementation. Its `sample_seconds` measures the Node sampling loop only;
PyMC model construction, PyTensor compilation, Node startup, program loading,
and returning the draws are outside that timer. On the Ryzen 5 2400G, seed
12345, 1,000 tuning and 10,000 retained iterations produced 52,828 logp
evaluations in 0.062 seconds; sample mean 0.791 and standard deviation 0.446
versus the analytic posterior mean 0.800 and standard deviation 0.447.

## A measured Gaussian example

The two manual scripts in `tests/link/js/` isolate the loop from the bridge:

```bash
python tests/link/js/bench_stride.py
python tests/link/js/bench_bridge.py
```

On an AMD Ryzen 5 2400G with Node 22.21.1 (2026-09-24), nine interleaved
old/new V8 runs of the same optimized Gaussian logp+gradient graph gave these
medians. "Old" rereads each input's shape and stride inside the element loop;
"current" reads them once per call, outside it. Both versions accept new
dimension lengths on later calls.

| Elements | Old V8 µs/eval | Current V8 µs/eval | Old / current |
| ---: | ---: | ---: | ---: |
| 4 | 0.49 | 0.45 | 1.09 |
| 1,000 | 10.81 | 4.48 | 2.42 |
| 10,000 | 102.85 | 38.39 | 2.68 |

An independent TyMC run of the equivalent Gaussian graph, on the same Node
runtime, had medians of roughly 0.17, 4.38, and 41.54 µs respectively. Those
are **in-engine** comparisons, not Python-call comparisons, and the two
projects still generate different loop bodies. The stride change brought the
larger examples close to TyMC; it did not erase the small-model difference.

`bench_bridge.py` times the public `pytensor.function` call separately. In one
run with observed data embedded as constants, warm calls took about 76, 106,
and 189 µs at those sizes; the first calls were about 2.6, 3.2, and 7.1 ms.
The V8 loop warmed over hundreds to thousands of evaluations (up to 10,000
for one case in this run). Public-call
times vary with process scheduling and include PyTensor input filtering,
serialization, pipe transport, and output reconstruction. The binary bridge
is useful for larger kernels but remains material below about 50 µs of
in-engine work; `mode="JS"` is not a claim that Python-to-JS calls are free.
