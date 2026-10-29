# Advanced gist parity

Source: [the gist](https://gist.github.com/jessegrabowski/1cd7ce787755c514c239be29c9e583d9),
revision `524b2af7118d48c9ade25babb3e44f763fbe4baa`.
The downloaded notebook was left untouched. This audit covers Jacobian/Hessian
patterns, including the derivative-semantics examples and inputs to coloring.
It excludes the marginalization/dependence API and axis-label inference.

## Result

The VJP → numerical algebra → nonzero rewrites → boolean basis batching
pipeline matches or tightens the saved gist counts in every completed case.
All six dedicated Cholesky cases and both dynamic-size Cholesky evaluations
match the numerical reference exactly.

Direction inputs are classified at the end of lifting, using the explicit
`cotangents` argument. Root queries remain unresolved until then.

The full audit and its per-case results are in [gist_results.json](gist_results.json).
Of 84 cases, 80 completed across 88 evaluations, with no missed numerical
nonzeros. In 74 cases the masks exactly matched the numerical reference at
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
| Dense Cholesky, 4×4 | 45 | 45 | 45 |
| Tridiagonal Cholesky, 4×4 | 28 | 28 | 28 |
| Arrow-first Cholesky, 4×4 | 36 | 36 | 36 |
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

## Projected triangular solves

The numerical equilibrium database in [algebra.py](algebra.py) normalizes
unbatched Blockwise wrappers and simplifies a projected two-sided solve
before boolean lifting. The rule checks the existing upper-triangular
assumption; it applies to the expression independently of its source Op.

For a square matrix `M`, define:

```text
Phi(M) = tril(M)    - diag(diag(M)) / 2
Psi(M) = triu(M, 1) + diag(diag(M)) / 2
Phi(M) + Psi(M) = M
```

For invertible upper-triangular `T`, let `U = tril(V)`. The product of `T`
and a strictly upper-triangular matrix is strictly upper-triangular, so
`Phi(T @ V) = Phi(T @ U)`. Consequently:

```text
T^-1 Phi(T @ V) T^-T
    = (U - solve(T, Psi(T @ U))) T^-T
```

The rewrite cancels `solve(T, T @ U)` in real-valued algebra. Boolean
inference then acts on the simplified expression. Its solve rule continues
to use conservative inverse reachability; it does not assume an inverse in
the boolean semiring.

This identity exposes the first-column independence of a lower Cholesky
factor: `L[2, 0] = A[2, 0] / sqrt(A[0, 0])` has no derivative connection to
`A[1, 0]`. Dense 3×3 and 6×6 patterns contain exactly 19 and 161 connections.

[test_algebra.py](test_algebra.py) checks numerical equivalence with arbitrary
direction matrices, verifies the rewritten graph, and checks that unknown
triangularity prevents the transformation. A precision regression compares
the original and rewritten solve masks over all 64 upper-triangular 4×4
sparsity structures and all 16 basis directions; none gains a connection.
Numerical equivalence and mask precision are checked independently, since
equivalent numerical forms need not give equally precise boolean bounds.

The dedicated gist recurrence combines dependency propagation and factor
nonzeros into one Scan. VJP inference composes factor-fill and inverse-
reachability analyses. For `cholesky(x + 10*eye(n))`, after identical
ShapeOpt/canonicalization cleanup and before fusion:

| Shape | VJP outer + Scan-body nodes | Gist outer + Scan-body nodes |
| --- | ---: | ---: |
| Dynamic | 193 + 58 | 36 + 77 |
| Static 4×4 | 54 + 52 | 21 + 77 |

VJP inference has four Scans; the gist has one. Both produce 45 connections
for 4×4 inputs. These counts include Scan bodies separately and do not
estimate tensor sizes, runtime, or memory.

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
  research/vjp_sparsity/test_prototype.py research/vjp_sparsity/test_advanced.py \
  research/vjp_sparsity/test_algebra.py -q
```

The focused inference suite passes **55 tests in each of FAST_COMPILE, CVM,
and NUMBA**. The regressions cover root resolution, structured directions,
numerical algebra, and precision of the lifted masks.
