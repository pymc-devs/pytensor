# Scalar graph audit for PR #2310 — 2026-10-13

The scalar-carrying changes still help after rebasing. The largest remaining
miss in the three PyMC models was a statically size-one tensor input keeping a
multi-output Composite array-valued. The working branch now reads that input as
a scalar and cancels scalar/tensor boundaries through DimShuffle chains.

The original branch is preserved at
`backup/rust_ai_catchup-pre-rebase-20261013` (`cb8e0e0c3`). The first rebase used
upstream `a5aa8667e`; the final rebase also includes `3270561e1`, `7d10a2b1d`,
and `dde153ddb`. Those three later commits change linalg rewrites, tridiagonal
assumptions, and ProdWithoutZeros in other backends; the measured Numba dispatch
and fusion code is unchanged.

## Measurements

Native calls, nanoseconds per evaluation. “Rebased” is the PR before the new
singleton-input change; “final” includes it. Ratios are medians of paired round
ratios, so they need not equal the ratio of the two reported medians.

| Workload | Upstream ns | Rebased ns | Final ns | Upstream/final | Rebased/final | Native allocations: upstream → rebased → final |
|---|---:|---:|---:|---:|---:|---|
| PyMC normal, 100 observations | 1,072 | 406 | 296 | 3.54× | 1.37× | 13 → 4 → **2** |
| PyMC linear regression | 2,186 | 901 | 504 | 4.11× | 1.63× | 16 → 5 → **2** |
| PyMC hierarchical, 170 observations, 8 groups | 2,477 | 1,159 | 1,051 | 2.35× | 1.10× | 24 → 7 → **5** |
| Synthetic logp/gradient, 100 observations, 5 groups | 1,536 | 607 | 614 | 2.48× | 0.98× | 17 → 5 → **5** |
| Synthetic logp/gradient, 10,000 observations, 100 groups | 17,578 | 16,733 | 16,965 | 1.04× | 1.00× | 17 → 5 → **5** |

The singleton change improves the real scalar models and is neutral on these
synthetic graphs. On the large graph, allocation overhead is a small fraction
of runtime. Its roughly 4% aggregate improvement is close to the observed
timing noise; the allocation reduction is exact.

Absolute timings vary substantially on this workstation. A second environment
(Numba 0.65.1 / llvmlite 0.47.0) also confirmed the singleton improvement and the
same LLVM allocation counts. Use the paired comparisons and mechanism evidence,
not these absolute values as universal performance claims.

## Which optimizations remain necessary?

These are separate ablations of the **final** implementation. Each number is
the exact native allocation count with that feature disabled. Features interact;
their effects cannot be added.

| Disabled feature | Normal | Linear regression | Hierarchical | Synthetic small/large |
|---|---:|---:|---:|---:|
| None | **2** | **2** | **5** | **5** |
| Scalar Subtensor producer | 10 | 12 | 19 | 13 |
| ScalarJoin consumer | 4 | 5 | 8 | 7 |
| Mixed Elemwise scalar inputs / rescalarization | 8 | 10 | 17 | 11 |
| Second scalarization pass after fusion | 4 | 5 | 8 | 7 |
| Stack-backed buffers for scalar reduction handout | 5 | 6 | 10 | 9 |
| Scalar-valued fused reduction outputs | 5 | 6 | 10 | 9 |
| Standalone ScalarCAReduce wrapper | 2 | 2 | 5 | 5 |

- **Keep scalar indexing, mixed inputs, ScalarJoin, post-fusion scalarization,
  scalar reduction handout and stack-backed temporary result buffers.** Each removes
  allocations; the rebased runtime ablations independently confirm benefits on
  the small workloads. For example, disabling scalar indexing costs 1.68–1.83×
  in the three real models; disabling mixed inputs costs 1.51–1.58×; returning
  reductions as arrays costs 1.47–1.71×. Those runtime ablations were collected
  before the singleton change, rather than combining several removals.
- **Reuse upstream's register reduction accumulator.** The PR no longer needs
  the old store-to-output core wrapper. `reorder_scalar_inputs` now only reorders
  constant arguments and returns scalar values to upstream's output handlers.
  Keeping the reduction's eventual handout buffer off the heap is still useful:
  upstream's accumulator alone does not eliminate that allocation.
- **Keep standalone ScalarCAReduce, but do not attribute a gain to these five
  models.** It does not fire in them. A focused bare reduction with a direct
  scalar consumer falls from 4 to 3 LLVM allocation sites and improves about
  14–19% across two timing runs. A reduction also returned as a tensor retains
  one shared reduction in both graphs; no duplicate computation was observed.
- **The as_strided epilogue is a codegen/contiguity improvement, not an established
  timing win on these scalar graphs.** They have no partial fused reductions;
  the reshape ablation is effectively a control. Its correctness is covered by
  the existing partial-reduction tests. Full scalar handout uses `.item()`.
- **The existing upstream rank-zero Elemwise lift is redundant for these tested
  graphs once this PR is enabled.** Excluding it leaves node and allocation counts
  unchanged; runtime stays at the noise level. This is not evidence to remove it
  globally: this audit does not cover every Numba graph.

## Comparison with PR #1967

[PR #1967](https://github.com/pymc-devs/pytensor/pull/1967), tested at
`9fc4ce685`, keeps Elemwise tensor inputs at their native ranks and uses implicit
right-aligned broadcasting. This overlaps with removing dummy dimensions, but
does not carry these values as native `ScalarType` or eliminate their tensor
boxes. Its advertised follow-up of squeezing genuinely singleton inputs is not
part of that commit.

The comparison was made in an isolated checkout: this scalar branch plus
`9fc4ce685`, with only cache-key/version conflicts resolved by retaining the
scalar branch's fields. The main working branch remains based on upstream main.
The baseline arm loads the actual #1967 fusion pass and dispatcher from an
archive and excludes this PR's scalarization passes. Numerical checks, timings,
and allocator instrumentation use the same method as the main comparison.

| Workload | Main allocations | #1967 alone | #1967 + scalar PR | Paired speedup over #1967 alone |
|---|---:|---:|---:|---:|
| Normal | 13 | 12 | **2** | 3.10× |
| Linear regression | 16 | 15 | **2** | 3.43× |
| Hierarchical | 24 | 22 | **4** | 2.20× |
| Synthetic small | 17 | 17 | **5** | 2.48× |
| Synthetic large | 17 | 17 | **5** | 1.05×, close to timing noise |

With #1967, the rebased scalar PR already reaches two allocations in normal and
linear regression; the singleton-input addition has no further effect there.
It still matters in hierarchical: disabling just singleton-input scalarization
raises allocations from **4 to 9** and costs **1.30×** runtime in a focused run.
An explicit size-one slice control similarly rises **2 to 4** and costs **1.69×**.
Disabling the longer DimShuffle-chain traversal has no effect in these cases.
Thus #1967 supersedes the measured need for the singleton addition in two
models, not the optimization generally.

The following independent ablations confirm the remaining consumers and
temporary-buffer change after #1967. Runtime allocation counts, frees, and LLVM
allocation sites agree in every row.

| Disabled feature, on top of #1967 | Normal | Linear regression | Hierarchical | Explicit unit slice |
|---|---:|---:|---:|---:|
| None | **2** | **2** | **4** | **2** |
| ScalarJoin | 4 | 5 | 8 | 4 |
| Stack-backed temporary reduction buffer | 5 | 6 | 9 | 3 |
| Singleton-input scalarization | 2 | 2 | 9 | 4 |
| Longer DimShuffle-chain traversal | 2 | 2 | 4 | 2 |

For hierarchical, disabling the stack buffer costs **1.24×** runtime and
disabling ScalarJoin costs **1.20×**. In the explicit unit-slice control, those
cost **1.22×** and **1.61×** respectively. These are separate ablations and cannot
be added together. Eight focused scalarization/implicit-rank tests pass in the
combined checkout.

### What the new Op and stack temporary actually do

`ScalarJoin` is an internal `OpFromGraph` subclass introduced in the original
PR, not during this audit. Its faithful inner graph is an ordinary one-dimensional
Join with scalars expanded to length-one tensors. The Numba dispatcher instead
accepts scalar values and vectors together, allocates only the joined output,
and writes each scalar directly into it. Other backends can evaluate the inner
tensor graph. #1967 changes Elemwise broadcasting, not this Join boundary.

The stack buffer is a one-element **temporary result buffer**, not a second
reduction accumulator and not an array returned to the user. Upstream's register
accumulator writes its result through the vectorizer's existing array-output
interface. For a scalar-returning full reduction, this PR supplies that interface
with a stack slot and reads it in the epilogue before returning a native scalar.
The temporary has no NRT ownership and never escapes. Extending `_vectorized`
to return native scalar outputs directly would remove this compatibility bridge;
the present bridge already avoids the measured heap allocations.

## LLVM confirmation

The script follows defined callees reachable from the native graph function.
It counts actual `call` instructions, excluding declarations, Python wrappers,
and unrelated functions in the module. LLVM site counts and separately measured
native allocator counts agree for every workload/ablation in the count run.

For normal, the final reachable LLVM contains only:

```llvm
... = tail call ptr @NRT_MemInfo_alloc_aligned(i64 16, i32 32) ; gradient
... = tail call ptr @NRT_MemInfo_alloc_aligned(i64 8, i32 32)  ; logp
```

Linear regression similarly has only the 24-byte gradient and 8-byte logp
allocations. The discarded scalar boxes were replaced by scalar
loads and arithmetic, not hidden in another dispatch function. Normal's static
`NRT_incref`/`NRT_decref` call sites fall from 26/37 upstream to 2/2; linear
regression falls from 35/49 to 3/4. These are static sites, not dynamic execution
counts.

The reachable `numba_attempt_nocopy_reshape` call sites fall from 3/4/5 upstream
to zero for normal/linear regression/hierarchical. Normal and linear regression
each retain one scalar `llvm.exp.f64` site; hierarchical retains two. The
synthetic graphs go from four scalar exp sites to two. This audit does not claim
new SIMD math or a per-element speedup from scalarization.

Hierarchical still has five allocations: logp, joined gradient, two group-sized
vector buffers, and an 8-byte standalone reduction feeding an inplace unit
Elemwise. The latter remains boxed because the post-fusion pass conservatively
preserves destroy maps. It is a next candidate; the two vector buffers need an
ownership/buffer-sharing optimization, rather than scalarization.
In the combined #1967 checkout the standalone reduction is scalarized, leaving
four allocations; this particular remaining box is already eliminated there.

## Other measured gaps

These remain **prototypes**, not changes enabled on the branch:

| Candidate | Focused evidence | Decision |
|---|---|---|
| Allow mixed, multi-output, non-unit Elemwise/Composite to carry scalar inputs | Wide two-output Composite: 4 → 3 LLVM allocation sites; 371 → 310 ns; paired runtime ratio 0.873, with unchanged outputs | Worth a follow-up. Dispatch already supports the case; the rewriter's single-output/unit-output gate skips it. |
| Treat ScalarFromTensor as a scalar consumer for Subtensor producers | Bare reduction plus explicitly scalar-indexed parameter: 3 → 2 LLVM sites; 319 → 274 ns; paired ratio 0.855 | Worth a follow-up. The current consumer tuple omits this direct scalar boundary. |
| Extend scalar handout to partial reductions with only singleton surviving dimensions | Same graphs and LLVM counts in all three real models | No demonstrated gain here; earlier handoff's partial-reduction shape no longer appears. |

The direct-scalar-consumer prototype also removes one allocation from the
10,000-element shared-reduction control, but its timing is neutral (ratio
1.006). Removing an allocation is not sufficient evidence of a meaningful
end-to-end win when the array work dominates.

## Method and reproduction

- AMD Ryzen 5 2400G, CPU 2 pinned, single-threaded BLAS and Numba, fastmath enabled.
- Main audit: Python 3.13.9, NumPy 2.4.6, Numba 0.67.0, llvmlite 0.49.0,
  LLVM 22.1.0. The local PyMC checkout is 6.1.0+25.gf7a5d40b4.dirty; its existing
  changes were preserved and both arms use the same checkout/data.
- Each kernel is called through a Numba C function pointer from an opaque C loop.
  LLVM cannot hoist the graph evaluation out of this loop. Python call and
  ndarray-adaptation costs are outside the measurement.
- All variants compile in one process before timing. Two warmup sweeps, 31
  interleaved rounds, alternating forward/reverse order, median paired ratios.
- Allocations are measured separately with `NUMBA_NRT_STATS=1`, through the same
  native calling convention. Each allocation is freed in the native count runs.
  Instrumented timings are never reported.
- Every arm is compared numerically at four fresh parameter points. Outputs are
  copied before another call can reuse storage.
- The upstream arm executes the actual upstream fusion pass and FusedElemwise
  dispatch loaded from an archive. The rest of the tensor dispatch is preserved
  from main. The rebased arm restores the scalar rewriter from `436776b51`
  (identical to the `ce923bd50` source used before the final rebase).

```bash
git archive a5aa8667e -o /tmp/scalar-upstream-main.tar
mkdir -p /tmp/scalar-upstream-main
tar -xf /tmp/scalar-upstream-main.tar -C /tmp/scalar-upstream-main

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONPATH="$PWD:/home/ricardo/Documents/.pymc-worktree6"
export PYTENSOR_FLAGS='base_compiledir=/tmp/scalar-audit,numba__cache=False,on_opt_error=raise'

taskset -c 2 /home/ricardo/Documents/pytensor/.venv/bin/python -c '
import sys, runpy
sys.path.append("/home/ricardo/Documents/.pymc-worktree6/.venv/lib/python3.13/site-packages")
runpy.run_path("scalar_graph_audit/run.py", run_name="__main__")
' --arms upstream,rebased,full --rounds 31 --output /tmp/scalar-audit-runtime

# Repeat separately with NUMBA_NRT_STATS=1, --count, and another --output.
# Extra cases: --models bare_reduce,wide_multi,shared_reduce
# Extra prototype arms: wide_multi,scalar_clients,singleton
# Singleton/chain ablations: no_unit_inputs,single_view_boundary
# Explicit size-one slice control: --models unit_tensor
```

For the #1967 comparison, archive `9fc4ce685` to `/tmp/scalar1967-baseline`,
cherry-pick it onto an isolated checkout of this scalar branch, resolve the
cache-key/version conflicts as described above, and run the same script there:
`--upstream /tmp/scalar1967-baseline --arms upstream,rebased,full --rounds 31`.
Run counts separately with `--count` and `NUMBA_NRT_STATS=1`; the extra ablation
arms are `heap,no_join,no_unit_inputs,single_view_boundary`.

`results.json` preserves samples, environment metadata, native allocation counts,
and LLVM call/allocation-site evidence. Each run also writes the full `.ll`,
caller LLVM, and optimized graph to its output directory. Original outputs for
this audit remain under `/tmp/scalar-audit-final-runtime`,
`/tmp/scalar-audit-counts`, `/tmp/scalar-audit-numba067`,
`/tmp/scalar-audit-remaining`, and `/tmp/scalar-audit-scalar-clients`.
The #1967 comparison outputs are in `/tmp/scalar1967-runtime`,
`/tmp/scalar1967-counts`, `/tmp/scalar1967-ablation`, and
`/tmp/scalar1967-focused`.

Validation before the three additional upstream commits:

- Fused Elemwise, Elemwise, and new scalarization tests: **190 passed, 2 xfailed**.
- Cache-enabled scalarization and OpFromGraph tests: **54 passed, 4 xfailed**.
- New tests cover float32/float64, reversed input views, empty reductions,
  widened accumulation, retained output ownership, Python fallback, unit tensor
  inputs, and boundaries introduced after fusion.

After rebasing onto `dde153ddb`, all five new scalarization tests pass again.
The five-model native/LLVM allocation check is unchanged at 2/2/5/5/5, with
matching frees. The isolated #1967 combination additionally passes eight focused
tests, including implicit leading axes and a shared index at different loop axes.
