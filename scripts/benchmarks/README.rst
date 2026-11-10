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
``(unique_name, callable, positional_inputs)``. Construct functions and check
numerical results before yielding; the runner warms up and times only calls.
The same case file from the workflow checkout runs against both source trees.
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
extension benchmarks need an explicit build step. Compilation is excluded.

Both revisions run in separate processes with separate compilation caches.
Two rounds use base/head then head/base order. Each result reports the median
of nine batches of three calls; raw samples, package/CPU metadata, per-round
ratios, CSV, and a Markdown summary are uploaded as artifacts. Native thread
pools use one thread, and Linux/Windows processes pin to one allowed CPU.
macOS does not expose process CPU affinity through psutil.

Ratios compare revisions within one machine: above one means the head is
faster. Hosted runner noise and different CPUs mean absolute timings across
operating systems are not directly comparable. Review per-round ratios and
repeat noisy or close results; the workflow deliberately has no timing pass
threshold. Numerical checks do fail the job. Adjust ``--size``, ``--rounds``,
``--batches``, and ``--calls`` locally for a quick smoke or a longer run.

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
