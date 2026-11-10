Compare two revisions across platforms
=====================================

The manual ``Benchmark revision comparison`` workflow accepts a repository,
base/head refs, and a case file. It resolves the refs once, then compares those
exact commits on Linux x86-64, Linux ARM64, Windows x86-64, and macOS ARM64.
For example, once the workflow is on the default branch::

    gh workflow run benchmark-compare.yml --repo ricardoV94/pytensor \
      --ref main -f base_ref=6cc22d53b303ecdddc2d39927b7ce33361c094d0 \
      -f head_ref=refs/pull/2647/head

To reuse it, add a case file exposing ``cases(size)``. Each yielded item is
``(unique_name, build, points, check)``: ``build()`` constructs an uncalled
PyTensor function, ``points`` maps input-range names to positional input
tuples, and ``check(result, inputs)`` validates a returned result. Prepare
symbolic expressions and data before yielding, but never compile or call the
function there: doing so would silently consume the measured first call.
The same case file from the workflow checkout runs against both source trees.
Compilation is measured once per function/dtype; execution is measured for
each input range. ``special.py`` has six function/dtype groups and 18 ranges.
The included ``special.py`` exercises the positive gammaln/psi kernels from
PR #2647 in both dtypes and includes fused value-plus-gradient evaluation.
Its numerical checks establish agreement with SciPy, rather than exhaustive
ULP accuracy coverage.

To run locally in an environment with the pinned requirements::

    python scripts/benchmarks/compare.py --base main \
      --head refs/pull/2647/head --case-file scripts/benchmarks/special.py

Dependencies are pinned in ``requirements.txt``; update them together when
needed. The runner imports archived source directly, so it needs no editable
install and no C compiler. This template targets Numba/Python workloads; C
extension benchmarks need an explicit build step.

Each function/dtype and round gets a fresh subprocess and empty compiler-cache
directories. Its reload uses another fresh process and exactly those same
directories. Base/head never share caches. Both ``PYTENSOR_FLAGS`` and
``NUMBA_CACHE_DIR`` are set before import; actual paths are checked and saved.
Cold means an empty compiler cache, not flushed OS filesystem/CPU caches.
Two rounds use base/head then head/base order.

Measurements separate ``build`` (``pytensor.function`` rewrites/lowering),
``first_call`` (lazy JIT or disk load plus one evaluation), extra ``warmup``
calls, and steady execution. ``ready`` is build plus first call. Imports and
symbolic graph/data preparation are outside these intervals. Validation
follows the measured first call and is outside execution timing. Numba has
no JS-style tiering/plateau detector here: warmup is an explicit number of
extra calls to every input range (default three). Runtime uses the reload
process, with nine batches of three calls per range. All raw samples, phase
wall/process-CPU times, metadata, per-round ratios, CSV and Markdown summaries
are uploaded. Process CPU time excludes child processes. Native pools use
one thread; Linux/Windows workers pin to one allowed CPU, while macOS lacks
process CPU affinity through psutil.

A reused directory does not prove a full cache hit. Artifacts save top-level
dispatcher hit/miss counters and fresh ``numba:compile`` events, including
function names and phases. ``full_hit`` requires a top-level hit, no misses
and zero fresh native compilations. A hit with fresh helper compilation is
``partial_hit``; missing/uncacheable cases retain that label. In the #2647
comparison, the baseline SciPy pointer resolver can compile on reload even
when the graph cache hits. A partially cached reload may therefore still
retain some compiler memory; the report does not call it a full cache hit.

RAM uses portable process RSS in bytes (reported as MiB), sampled by the
parent every 10 ms plus worker phase endpoints. There is no sampling thread
in the timed worker. Each phase records baseline, after, sampled peak and
peak increase. Compile peak covers build/first call in the cold process;
reload-ready RSS is recorded immediately after the first reload call;
evaluation peaks/increases come from the reload process after validation
and warmup. Input data, libraries and allocator-retained memory are included.
These are process measurements, not function-object sizes. RSS includes
shared pages; short allocations between samples may be missed, and a zero
increase does not mean no allocation. A lifetime high-water mark is not
reused as an evaluation peak. ``setup-summary.*`` contains phase/RSS summaries;
per-worker ``*.json``, ``*.memory.json`` and logs retain the evidence.

Ratios compare revisions within one machine: above one means the head is
faster. Hosted runner noise and different CPUs mean absolute timings across
operating systems are not directly comparable. Review per-round ratios and
repeat noisy or close results; the workflow deliberately has no timing pass
threshold. Numerical checks do fail the job. Adjust ``--size``, ``--rounds``,
``--batches``, ``--calls``, ``--warmup-calls`` and ``--rss-interval`` locally
for a quick smoke or a longer run.

Using the unmerged template branch
----------------------------------

GitHub normally requires a manual workflow to exist on the default branch
before ``workflow_dispatch`` works. Until this template is merged, rerun its
existing four-platform comparison without changing the default branch::

    gh run rerun 42798595487 --repo ricardoV94/pytensor

That reruns the exact workflow commit and its original revisions. To compare
different refs immediately, check out ``ci/benchmark-template-2647`` and use
the local command above. For a reproducible comparison with the tested PR,
use ``--base 6cc22d53b303ecdddc2d39927b7ce33361c094d0`` and
``--head 18b1d004bbe32c924756dbf4ec9a614e14fe540f``. Artifacts record the
resolved commits in ``revisions.json`` and arguments plus the runner/case
file hashes in ``configuration.json``. Branch names such as ``main`` and
PR head refs may change; use the recorded commits when repeating a result.
