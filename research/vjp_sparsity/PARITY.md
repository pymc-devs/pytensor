# Advanced gist parity

Source: [the gist](https://gist.github.com/jessegrabowski/1cd7ce787755c514c239be29c9e583d9),
revision `524b2af7118d48c9ade25babb3e44f763fbe4baa`.
The downloaded notebook was left untouched. This audit covers Jacobian/Hessian
patterns, including the derivative-semantics examples and inputs to coloring.
It excludes the marginalization/dependence API and axis-label inference.

## Result

The VJP → nonzero rewrites → boolean basis batching pipeline covers most of
the advanced examples.
**Full precision parity has not been reached: Cholesky exposes cancellation
inside the numerical VJP that local boolean lifting cannot retain.**

Direction inputs are classified at the end of lifting, using the explicit
`cotangents` argument. Root queries remain unresolved until then.

The full audit and its per-case results are in [gist_results.json](gist_results.json).
Of 84 cases, 80 completed across 88 evaluations, with no missed numerical
nonzeros. In 68 cases the masks exactly matched the numerical reference at
every evaluated point. Four dynamic-QR cases errored.
Tests against numerical Jacobians are evidence at the supplied points, not a
proof that every structural nonzero has been found for every input.

| Representative case | Prototype | Gist | Numerical |
| --- | ---: | ---: | ---: |
| Five-equation nonlinear system | 9 | 9 | 9 |
| Rosenbrock Hessian | 22 | 22 | 22 |
| Five-sector CGE | 168 | 168 | 168 |
| CAR Hessian, either formulation | 73 | 73 | 73 |
| `kron(small, other_small)` | 48 | 48 | 48 |
| Batched dot, sparse batched constant | 64 | 120 | 64 |
| Dense Cholesky, 4×4 | 50 | 45 | 45 |
| Tridiagonal Cholesky, 4×4 | 36 | 28 | 28 |
| Arrow-first Cholesky, 4×4 | 40 | 36 | 36 |
| Arrow-last Cholesky, 4×4 | 16 | 16 | 16 |
| QR with a block-diagonal mask | 44 | 44 | 36 |

Entries are inferred nonzero counts, not timings. The README's graph metrics
are 443 versus 487 nodes over the 17 PR elementwise cases. These counts do not
establish a performance advantage.

## Supported operations

The rewrite database contains producer rules for:

- Shape-derived conditions and triangular masks preserve their actual values.
  Conditions depending on unknown floating-point data still union both branches.
- Gather/scatter applies the existing indexing Ops to boolean masks when the
  indices are fixed by constants, shapes, or held-fixed integer inputs.
- Join/Split supports the checkout's axis-on-the-Op API and multiple outputs.
- Dense dot uses boolean matrix multiplication, including batching.
- Eye, diagonal extraction/allocation, and block-diagonal construction retain
  their structural zeros.
- `OpFromGraph` exposes a cloned inner expression to the same inference rules.
- Solve values use boolean inverse reachability, respecting triangular,
  symmetric and tridiagonal interpretations. Cholesky solves compose the
  factor's inverse masks.
- Cholesky **value** masks include elimination fill-in. Economic QR **value**
  masks use boolean Gram–Schmidt. These describe coefficient values used by
  the ordinary VJP; they do not supply bespoke derivative-connectivity rules.

Reuse of existing indexing and lifting rewrites has not been audited.

## Remaining precision gap: Cholesky

For a lower Cholesky factor,

```text
L[0, 0] = sqrt(A[0, 0])
L[2, 0] = A[2, 0] / L[0, 0]
```

Therefore `d L[2, 0] / d A[1, 0]` is identically zero. The prototype reports
that entry as potentially nonzero, even with exact triangular factor masks.
The 3×3 dense case has 20 inferred entries versus 19 numerical entries;
this is its only extra connection.

The native pullback uses triangular solves around
`Phi(L.T @ cotangent)`, followed by symmetrization. Dependencies cancel between
the terms of those solves. AND/OR lifting retains the paths but discards their
signs and algebraic relationships, so later boolean cleanup cannot recover the
zero. This is different from having an unimplemented lift.

The gist instead propagates dependencies through the entry-by-entry Cholesky
recurrence, which exposes this independence directly. To close this gap while
keeping a VJP-based design, we need a numerical/algebraic rewrite before
boolean abstraction, or a different symbolic pullback representation. Adding
more factor-value masks alone does not fix it. A bespoke Cholesky derivative
pattern would also work, but would reintroduce the specialization this
experiment is testing whether we can avoid.

The gap is recorded as a strict expected failure in
[test_advanced.py](test_advanced.py); it is not counted as precision parity.
Five of the six dedicated Cholesky cases remain looser than the gist. The
dynamic-size Cholesky example also remains loose at both evaluated sizes.

## Other conservative results and execution issue

`max`, data-dependent `switch`, `y - y.T`, and masked QR have the same masks
as the gist in a separate direct comparison. They can exceed a numerical
Jacobian's nonzeros at one point. The sparse batched-constant dot mask is a
strict subset of the gist's mask: 64 versus 120 entries, with no numerical
nonzeros missing at the audit point.

Four unknown-shape QR cases hit `Blockwise{IfElse}` evaluation with
`compute_map=None`. They are recorded as errors.

Boolean matmul keeps the generic core Dot path,
with an AND/OR kernel for Numba, including preallocated batch outputs. The
regression tests exercise empty contraction dimensions as well as ordinary
batched products. No broadcasted three-dimensional AND tensor is introduced.

## Reproduction

Download the notebook at the pinned gist revision, then run:

```bash
PYTENSOR_FLAGS=mode=FAST_COMPILE,on_opt_error=raise python -m research.vjp_sparsity.gist_audit \
  /path/to/sparsity.ipynb --output /tmp/vjp-gist-results.json
```

The runner loads only graph-construction prefixes and the gist's numerical
oracle. It does not run the notebook's sparsity detector, global registrations,
plots, or marginalization examples. The numerical oracle uses static-shaped
input copies and a relative `1e-12` threshold to suppress pullback roundoff;
the inferred pattern itself uses boolean operations, not numerical thresholds.

Repository regression checks:

```bash
PYTENSOR_FLAGS=mode=FAST_COMPILE,on_opt_error=raise python -m pytest \
  research/vjp_sparsity/test_prototype.py research/vjp_sparsity/test_advanced.py -q
```

The focused inference suite passes **51 tests in each of FAST_COMPILE, CVM,
and NUMBA**, with the Cholesky precision counterexample explicitly expected to
fail. The regressions cover root resolution and structured directions.
