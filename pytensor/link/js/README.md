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


## Expanded coverage, interleaved rerun (2026-10-08)

AMD Ryzen 5 2400G, x86 Linux, CPU3 pinned, one numerical thread. Node 22.21.1 /
V8 12.4. TyMC base b1c038979eb006b2d9ad1842a5b9ec2640688944 with default
WASM/SIMD; stable adds NB/softplus precision fixes, parameter guards and zero-
mask/broadcast corrections. PyTensor js_backend commit 8c5317186. The grid keeps
models whose slowest backend takes at least 50 microseconds per evaluation.

### Runtime: microseconds per full logp + gradient

Python crossing excluded. Median of three independent V8 replicas, each with
four seconds of all-point warmup and twelve rotating interleaved rounds over the
same five C reference points.

| Model | CVM | Numba | PyTensor JS | TyMC | TyMC-Stable |
|---|---:|---:|---:|---:|---:|
| Poisson GLM | 106.30 | 42.36 | 122.52 | 66.56 | 120.12 |
| NB catalogue | 628.45 | 602.04 | 838.90 | 486.82 | 748.33 |
| NB corpus | 660.25 | 598.20 | 893.35 | 515.00 | 772.47 |
| LKJ covariance | 574.67 | 198.78 | 644.97 | 246.65 | 293.52 |
| Matrix factorization | 19,073.19 | 1,390.83 | 7,253.71 | 5,211.04 | 5,216.61 |

JS means PyTensor JS. TyMC-Stable is the precision-corrected TyMC branch. Numba
uses the alchemize native cfunc harness; C uses CVM n_calls. JS/TyMC and Numba
include the full caller-gradient copy; C writes its output buffer without an
extra external copy.

TyMC includes its default WASM/SIMD kernels. The LKJ profile confirms WASM
triangular solves; this is not a pure-JS TyMC comparison. Gaussian 10,000 is
omitted because the pinned current TyMC snapshot fails to compile its broadcast-
lift graph.

### Compilation: seconds from a built model

JS and TyMC include observed evaluation warmup through the plateau. Cold LLVM/C
and reused-cache observations are shown separately.

| Model | CVM cold | CVM cached | Numba cold | Numba cached | PyTensor JS | TyMC | TyMC-Stable |
|---|---:|---:|---:|---:|---:|---:|---:|
| Poisson GLM | 10.518 | 1.026 | 6.125 | 0.926 | ~2.562 | ~1.750 | ~1.161 |
| NB catalogue | 16.204 | 1.359 | 9.318 | 1.296 | ~1.302 | ~0.341 | ~0.414 |
| NB corpus | 18.373 | 1.289 | 6.971 | 1.343 | ~1.152 | ~0.351 | ~0.428 |
| LKJ covariance | 44.924 | 1.654 | 32.862 | 2.185 | ~3.184 | ~1.101 | ~0.529 |
| Matrix factorization | 13.072 | 0.859 | 5.335 | 0.617 | ~0.618 | ~0.691 | ~1.280 |

Numba cold: caches disabled. Numba cached: graph and native wrapper caches
reused (prototype, not stock nutpie). CVM cold/cached: fresh/reused C module
cache.

The JS estimate is graph-to-worker-ready plus the median observed plateau
interval. TyMC/stable estimates are the median of each paired core-compile +
plateau interval. Plateau means the first 200 ms window within 10% of the best
later window of a four-second curve; first-call and parity preparation precede
it. These are approximate active-work figures, not exact total-first-use
measurements or proof JIT compilation has stopped. JS graph-to-ready includes
Node startup; TyMC core compile excludes Node startup and module import. Model
construction is outside every compile timer. The JS/TyMC startup timing scopes
therefore differ.

Cold Numba/CVM are saved 2026-10-07 observations: one uncached Numba compile and
the mean of two C subprocesses with fresh cache directories. Model
initialization can populate C helpers before the timed compile; module counts
and cache paths are recorded. No fresh LLVM/C compile was repeated for this
report update.

Numba cached and CVM cached use fresh-process reloads measured 2026-10-08 in
compile_cache_refresh_20261008. Every fully cached Numba prototype record has
one observed adapter cache hit. This prototype is not stock nutpie; the C module
counts do not grow on reload.

### Evaluations where JS wins in projected wall time

Each column compares JS with the named backend and cache state. A number is the
estimated maximum evaluation count where JS wins; “any number” means JS wins at
every count.

Use T = compile_seconds + evaluations × runtime_us / 1e6. The count is
logp/gradient evaluations, not posterior draws. “never” means JS does not win
for any nonnegative count using these measured cells. All numbers are estimates;
runtimes vary with parameter points and compile/warmup timings vary between
runs.

| Model | CVM cold | CVM cached | Numba cold | Numba cached |
|---|---:|---:|---:|---:|
| Poisson GLM | ~490,600 | never | ~44,400 | never |
| NB catalogue | ~70,800 | ~269 | ~33,800 | never |
| NB corpus | ~73,900 | ~589 | ~19,700 | ~648 |
| LKJ covariance | ~593,800 | never | ~66,500 | never |
| Matrix factorization | any number | any number | ~804 | never |

These are arithmetic projections, not observed sampling-wall-time comparisons,
holding the evaluation count equal across backends. They use the raw evaluator
runtimes above; Python/Node crossing would lower the JS thresholds. The cache
scenarios match the compilation table, and the cold/reused comparison uses
explicitly dated observations. Very small compile-time savings may be
measurement variation.

Runtime and JS/TyMC compile row provenance: Poisson GLM: 20261008T140040; NB
catalogue: 20261008T125014; NB corpus: 20261008T125014; LKJ covariance:
20261008T130534; Matrix factorization: 20261008T125014.

Only Poisson was remeasured after the final rate-check patch; the other four
rows are the preceding observations on the earlier stable commit. Raw paths and
commits remain in row_origins.json, commits.json and provenance/.


## Python crossing and a five-second sampling budget (2026-10-07)

The existing persistent-Node bridge adds roughly 75–95 µs per call for small
parameter/gradient buffers on this machine, measured with the same evaluator
inside Node and through Python. `trust_input=True` and direct VM access only
remove a few µs. A 12,000-entry parameter/gradient payload adds roughly
220–240 µs in a matching cheap-graph control. Twelve interleaved rounds
follow at least four seconds of evaluation warmup; model compilation and
startup are outside the timing window. The full gradient is returned or
copied to a caller buffer in both paths.

For a tolerance of five seconds of added sampling time, the useful criterion
is `calls_on_critical_path * extra_us / 1e6 < 5`: about 60,000–65,000 calls
for a small payload, or 21,000–23,000 for the large control. These are
transport-based ballparks; graph-specific cost may lower the limits.
Parallel-chain wall time and GIL contention were not measured.

Nutpie's existing Python-function adapter can call the JS evaluator while
nuts-rs runs NUTS. Its Rust/Python adapter copies the parameter and gradient;
the binary Node pipe copies them again. A float64 gradient view of the packed
reply can be unaligned, so the adapter uses
`np.require(gradient, requirements=["C", "A"])` before returning it to Rust.
This can introduce another copy. An array protocol does not make the current
subprocess transport zero-copy.

`catalogue_negative_binomial`: 4 serial chains, 1,000 tuning + 1,000 retained draws per chain,
low-rank adaptation, seed 20134234. The actual Rust sampler made **30,010**
logp/gradient calls ([0, 7216, 7739, 7683, 7372] by evaluator), and the measured
sampling window was **40.20 s**. That window includes sampler
launch, initialization, tuning, draws and Arrow storage; compilation, JS warmup, Arrow
extraction and diagnostics are outside it.

Holding that observed call count fixed, the matching payload control projects
**2.25 s** for the Node crossing alone.
The five-second limit is **166.6 µs/call** for this run.
This is a projection, not an observed native-vs-JS sampling delta. It excludes
Rust-to-Python adapter overhead and the aligned-gradient copy.
The fixed-point model comparison shows additional per-call differences beyond
the payload control; do not assume the control explains the entire gap.
Its paired direct-wrapper excess at the five benchmark points would instead
project 4.45 s,
also excluding the Rust/Python adapter. Neither projection is an observed
sampling delta against an in-process JS evaluator.

The measured crossing alone need not justify a native integration for every
model. The existing JS evaluator's Numba gaps on covariance and matrix
factorization also matter, independently of Python. In-process Python V8,
wasmtime-py and Rust-native V8/WASM routes were not benchmarked here.
The focused suite passed 21 JS tests after fixing Join's lowering to the
current `op.axis` API.
