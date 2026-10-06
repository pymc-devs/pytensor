# Experimental JS backend

This is a proof of concept for lowering an ordinary PyTensor graph to JavaScript
and running it in V8. PyTensor still builds and rewrites the graph and computes
gradients. The linker consumes the existing scalar `Composite` and
`FusedElemwise` representation, including indexed reads/writes and partial sum
reductions. It uses PyTensor's graph and rewrites; the scalar gamma/digamma
kernels are ported from TyMC (see `THIRD_PARTY_NOTICES`). TyMC is not a runtime
dependency.

`JS` mode selects `fast_run` and the `js` rewrite tag. The indexed/reduction
fusion rewrites are shared with Numba but registered under both tags; JS does
not select Numba-only rewrites. The linker delegates Op and scalar expression
generation to `dispatch/`, using type-based dispatch and a shared indentation
helper (`pytensor/link/string_codegen.py`). `js_typify` checks supported dtypes
and converts values for the binary transport. Each Op or fused cluster gets
its own generated function, called by a topological driver. Tensor records
carry data, shape, strides and an offset; basic slices and transposes are
views. Complex indexed fusion patterns without a fused emitter lower their
inner graph through ordinary Op dispatch.

Install Node.js so `node` is on `PATH`, then:

```python
import numpy as np
import pytensor
import pytensor.tensor as pt

x, y, beta = pt.vector("x"), pt.vector("y"), pt.scalar("beta")
logp = pt.sum(-0.5 * (y - beta * x) ** 2)
fn = pytensor.function([x, y, beta], [logp, pt.grad(logp, beta)], mode="JS")

print(fn(np.arange(3.0), np.ones(3), 0.2))
print(fn(np.arange(11.0), np.ones(11), 0.2))  # same graph, different shape
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

The supported set includes elementary arithmetic, `gammaln` and `psi`,
`Composite`, `DimShuffle`, sum and boolean all reductions, multidimensional
fused loops, integer scalar/vector indexing and indexed updates. Allocation,
shape operations, reshape, concatenate, cumulative sum/product, vector/matrix
dot, Cholesky and triangular solves also lower. Simple PyMC parameter checks,
boolean arrays, and float32 constants/intermediates are supported. Int64
inputs outside JavaScript's exact integer range are refused.
Unsupported Ops and casts raise `NotImplementedError` during lowering. This
backend is not yet a replacement for C or Numba: it lacks BLAS, `Scan`, many
special functions, general advanced indexing and complete dtype semantics.
The dense linear algebra kernels are straightforward compatibility loops.

The included tests cover changing shapes, strided views, indexed gradients
with duplicate indices, partial reductions, special functions and small dense
linear algebra, as well as PyMC log density and gradient. Fusion tests assert
that PyTensor's fusion pass fired.
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

Three PyMC benchmark-suite twins were also checked at their stored reference
points (2026-09-24, Ryzen 5 2400G, Node 22.21.1). Both paths were compiled and
warmed before timing. The PyTensor JS timer calls the generated logp+gradient
function directly inside Node; the TyMC timer calls `compiled.logp_and_grad`
inside the same Node runtime. Neither timer includes model building,
compilation, Python transport, or diagnostics. Log density and every gradient
component matched the PyMC references to 1e-9 relative tolerance. Medians of
nine interleaved runs were:

| Model | Data rows | PyTensor JS µs/eval | TyMC µs/eval |
| --- | ---: | ---: | ---: |
| Eight schools | 8 | 1.02 | 0.50 |
| Binomial GLM | 30 | 1.98 | 0.82 |
| Poisson GLM | 4,000 | 64.84 | 40.68 |

The JS-only rewrite now moves a scalar parameter check after the likelihood
sum, so fusion can accumulate the log density without a row-sized temporary.
The generated loop also holds reduction accumulators in JS locals and hoists
singleton inputs. In an interleaved A/B run on the Poisson graph, moving the
check reduced 81.78 to 72.45 µs/eval; accumulator and input changes reduced
71.58 to 58.56 µs/eval. TyMC remains faster: its Poisson loop has fewer
per-row checks and branch cases. A scratch experiment removing those branches
at the reference point helped, but is not a valid general rewrite at boundary
values and is not part of the backend.

These historical measurements are single points, not sampler throughput.
Negative-binomial and hierarchical models were unsupported in that run;
the current prototype lowers their required Ops.

`bench_bridge.py` times the public `pytensor.function` call separately. In one
run with observed data embedded as constants, warm calls took about 76, 106,
and 189 µs at those sizes; the first calls were about 2.6, 3.2, and 7.1 ms.
The V8 loop warmed over hundreds to thousands of evaluations (up to 10,000
for one case in this run). Public-call
times vary with process scheduling and include PyTensor input filtering,
serialization, pipe transport, and output reconstruction. The binary bridge
is useful for larger kernels but remains material below about 50 µs of
in-engine work; `mode="JS"` is not a claim that Python-to-JS calls are free.


## Expanded coverage, interleaved rerun (2026-10-06)

The expanded prototype runs all 12 variants in the recovered benchmark grid,
including both negative-binomial datasets, hierarchical beta-binomial,
fitted covariance and matrix factorization. Representative results on the
same Ryzen 5 2400G / Node 22.21.1 are below. These supersede the earlier
single-point comparisons as a broader audit; they are not a controlled
regression comparison against those historical runs.

Each runtime cell is the median of 12 interleaved round means, averaging
the same five saved unconstrained parameter points. Numba uses the committed
`beat_js` tree (`50ad847`) and alchemize's native Rust `cfunc` timer. Both JS
engines evaluate directly inside Node and copy the full gradient to the
caller buffer; Python transport is excluded. PyMC's normal compile path
converts parameter checks to switches for both PyTensor backends. All variants
matched the saved C reference; maximum component-scaled error was 1.62e-9.

Current TyMC is remote main `b1c0389`, with its default WASM/SIMD linear
algebra enabled. The earlier local `38d0a20` snapshot is not used for these
TyMC timings.

| Model | Numba µs/eval | PyTensor JS µs/eval | Current TyMC µs/eval |
| --- | ---: | ---: | ---: |
| Eight schools | 1.20 | 1.12 | 0.38 |
| Poisson GLM | 37.43 | 136.49 | 60.96 |
| Catalogue negative binomial | 541.14 | 1012.94 | 465.86 |
| Hierarchical beta-binomial | 8.35 | 12.65 | 4.86 |
| LKJ multivariate Normal | 292.12 | 1350.86 | 251.22 |
| Matrix factorization | 1322.22 | 18631.21 | 4723.79 |

Compile starts from a built/frozen model and includes logp/gradient building,
rewrites and lowering. Numba has one fresh native callback compile per model,
with caches disabled. JS/TyMC include binding and elapsed evaluation warmup
through the observed performance plateau; their cells are medians of two
cold instances. Process startup and imports are excluded. The plateau uses
a 200 ms observation window within 10% of the best later window, not the
last JIT event. Runtime follows at least four seconds of evaluations.

| Model | Numba cache off, s | PyTensor JS through plateau, s | Current TyMC through plateau, s |
| --- | ---: | ---: | ---: |
| Eight schools | 5.04 | 0.74 | 0.30 |
| Poisson GLM | 6.71 | 0.62 | 0.68 |
| Catalogue negative binomial | 9.28 | 0.73 | 0.52 |
| Hierarchical beta-binomial | 9.03 | 0.82 | 0.89 |
| LKJ multivariate Normal | 34.29 | 1.56 | 0.89 |
| Matrix factorization | 6.69 | 0.86 | 0.76 |

The machine was not otherwise isolated; the full report retains paired ratios,
round ranges, CPU times, per-point timings and both cold warmup traces.
These figures show the cold-compile advantage and remaining runtime gaps of
this compatibility PoC. They do not establish general V8 versus LLVM speed,
or sampling throughput. The focused JS suite passed 19 tests.
