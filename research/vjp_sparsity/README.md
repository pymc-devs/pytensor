# Nonzero inference from an ordinary VJP

This prototype starts with the existing symbolic VJP and rewrites structural
nonzero queries over it. Its organization follows PyMC's inference rewriting:
an explicit database, separately registered local rules, and a late
pessimistic pass.

## Start directly from a regular VJP

```python
import pytensor.tensor as pt
from pytensor.gradient import pullback

from research.vjp_sparsity.rewriting import infer_nonzero

x = pt.vector("x", shape=(3,))
y = pt.sin(x * [0., 2., 0.])
cotangent = y.type("cotangent")

vjp = pullback(y, x, cotangent)
mask = infer_nonzero(vjp, cotangent)
```

The input to `infer_nonzero` is the ordinary VJP. The result is a boolean
expression depending on the cotangent's nonzeros and input shapes, with
unknown primal values treated conservatively. No second derivative engine or
separate `zeros` dictionary is involved.

For this example, the VJP contains:

```text
cotangent * cos(x * [0, 2, 0]) * [0, 2, 0]
```

Its inferred connectivity is equivalent to:

```text
(cotangent != 0) & [False, True, False]
```

The cosine coefficient becomes potentially nonzero everywhere. It is not
evaluated at the current input values to decide structural sparsity. Redundant
all-true broadcast masks can remain in intermediate graphs.

## Explicit databases and registered rewrites

[rewriting.py](rewriting.py) defines:

- `nonzero_rewrites_db`: a `SequenceDB` with unbatched-wrapper normalization,
  shape inference, pre-canonicalization, equilibrium lifting, final query
  resolution, and post-canonicalization. The final resolver is added to each
  call's query.
- `nonzero_lift_db`: an `EquilibriumDB` of independently registered rules.
- `register_nonzero(tracks)`: register a rule against the actual producing Ops.

For example, the multiplication rule is an ordinary registered node rewrite:

```python
@register_nonzero([elemwise_of(ps.Mul)])
def local_nonzero_mul(fgraph, node):
    return reduce(pt.and_, map(may_be_nonzero, node.inputs))
```

A multiplication rule therefore dispatches on multiplication, rather than
sharing the same query-Op tracker with every other rule. The registration
adapter replaces the producer's nonzero-query clients with the returned mask;
it does not replace the numerical producer or its other clients.

A package can register another rule without editing the inference driver.
The test suite demonstrates an external rule refining a pessimistic result.
Standard database queries can select or exclude rewrites.

`MayBeNonzero(value)` is temporary inference IR, not a replacement for the VJP.
It distinguishes structural nonzero queries from ordinary runtime
`value != 0` predicates already inside the graph. Rewriting those original
predicates as if they were analysis queries could change which branches are
possible. All query markers are removed before returning the result.

Root queries stay unresolved throughout lifting. At the end, `infer_nonzero`
uses its explicit `cotangents` argument to resolve direction roots to ordinary
`value != 0` predicates. Unknown primal roots and unsupported expressions
become all-true masks in that same final pass. Only the constant rule and this
final resolver track `MayBeNonzero` itself.

For example, the VJP of `y = x**2` is `2*x*v`. Lifting it produces queries on
the primal `x` and the direction `v`:

```text
may_be_nonzero(x) & may_be_nonzero(v)
```

With `cotangents=[v]`, final resolution gives `True & (v != 0)`.

A structured direction such as `zeros(n)[idx].set(amplitude)` already carries
its selection in the zero/scatter graph. Ordinary lifting preserves that
structure even without declaring a cotangent root. Declaring `amplitude` a
cotangent additionally preserves its actual zero/nonzero choices.

Lifting and final resolution are ordered stages of the same database.
Known rules first reach equilibrium, including processing queries they create
on upstream producers. Only then are roots classified and unsupported queries
made dense; applying that default earlier could prevent producer rules from
running.

The registered rules cover:

| VJP expression | Inferred nonzeros |
| --- | --- |
| Constant | Stored values compared with zero |
| Unknown primal input | All elements potentially nonzero |
| Cotangent | Ordinary `cotangent != 0` |
| Product | AND of masks |
| Sum/difference | OR of masks, conservatively |
| Division | Numerator mask, on valid differentiation domains |
| Zero-preserving unary operation | Input mask |
| Sum reduction | Boolean `any` |
| Product reduction | Boolean `all` |
| Shapes/broadcasting | Corresponding shape operation on the mask |
| Constant/shape-condition switch | Select branch masks |
| Unknown-condition switch | Union branch masks |
| Fixed-index gather/scatter | Apply the indexing Op to boolean masks |
| Dense dot | Boolean matrix multiplication |
| Join/Split, diagonal/block-diagonal | Corresponding boolean construction |
| Solve | Boolean inverse reachability times RHS mask |
| Cholesky/QR coefficients | Factor-value structure, including fill-in |
| CholeskyGrad | Reverse column recurrence on factor/cotangent masks |
| OpFromGraph | Expose its cloned inner expression to the same rules |

Cholesky's pullback emits `CholeskyGrad`, a `SymbolicOp`/`OpFromGraph` subclass.
Ordinary compilation inlines its numerical body, including through Blockwise.
This database keeps the operation intact and dispatches on it directly. With
known factor triangularity, its nonzero rule reverses the column recurrence
using boolean factor and cotangent masks.

Data-dependent indexing and sparse-dot rules remain conservative. Non-direction
queries left unresolved after lifting become all-true masks in the final pass.

Derivative behavior for `floor`, comparisons, and other operations comes
from the existing VJP rules. The prototype does not use the PR's new
`piecewise_constant` flags or its sparsity dispatchers.

## Full Jacobian pattern

[prototype.py](prototype.py) is a small convenience builder:

```text
regular pullback -> infer_nonzero -> batch boolean cotangent directions
```

It handles the PR's multiple outputs/inputs, repeats, intermediate variables,
and dynamic dimensions. The builder calls `infer_nonzero` once after AD.
Inside that call, the VJP is cloned and requested primal values are made opaque
before simplification, preserving direct perturbations of requested
intermediates even when their producers can be constant-folded.

The full rewrite pipeline runs on the private clone. Original primal bindings
are restored only after it finishes, with no further rewrites. This follows
PyMC's `construct_ir_fgraph` isolation pattern. A regression test snapshots the
user's original node inputs and validates their attached `FunctionGraph`.

There is no additional boolean seed-variable mapping: the existing cotangent
inputs are directly replaced by boolean basis directions during vectorization.

[demo.py](demo.py) displays the ordinary and rewritten VJP and compares the
resulting graphs with the PR under identical shape/canonicalization cleanup:

```bash
python -m research.vjp_sparsity.demo
```

These graph counts are structural observations, not performance benchmarks.
Both approaches currently build dense patterns and basis matrices; neither
has an established memory or runtime advantage.

Implementation size includes imports, docstrings, and graph-isolation handling.
The comparison excludes tests and demo code; the original PR's core and
elementwise modules total 471 lines.

Across all 17 elementwise cases, after the same shape/canonicalization cleanup:

| Metric | VJP inference | Original PR |
| --- | ---: | ---: |
| Nodes, summed over separate case graphs | 443 | 487 |
| Floating-point node outputs | 0 | 192 |

The VJP graphs are smaller in 10 cases and larger in seven. Connectivity
operations are boolean; shape and index operations are integer. These counts
do not measure the sizes of the tensors carried by the nodes.

For M output elements and N input elements, both return an O(MN) dense matrix.
The prototype seeds an M-by-M boolean output basis; the PR seeds an N-by-N
floating-point input basis. This direction choice can matter more than the
number of graph nodes.

## Validation

[test_prototype.py](test_prototype.py) reuses all 21 PR cases with the same
expectations: 17 elementwise cases and four core cases. The three explicitly
conservative nonzero counts also match.

Thirteen additional checks cover cancellation, the folded-intermediate regression
in three modes, repeated outputs, an existing zero-gradient rule, value
independence, ordinary VJP input, external rewrite registration, and keeping
runtime nonzero predicates separate from inference queries, and preserving the
original user graph, final classification of roots, and structured directions.

The PR cases and root-resolution regressions comprise **34
tests**, checked in each of FAST_COMPILE, CVM, and NUMBA.
The advanced regression cases are in [test_advanced.py](test_advanced.py).
The combined inference suite contains **55 tests**. The precision regression
compares the semantic gradient with its expanded body over all 64 lower- and
upper-triangular 4×4 sparsity structures and every individual direction.
Native tests also check normal inlining and second-order derivatives.

See [PARITY.md](PARITY.md) for the advanced gist audit: 84 cases, 80 completed,
no missed numerical nonzeros over 88 evaluations. All six dedicated Cholesky
examples and both dynamic-size evaluations match the numerical reference
exactly. Four dynamic-QR cases hit an IfElse batching error.

The sparse batched-constant dot case improves on the gist: **64 inferred
connections versus 120**, matching the numerical pattern at the audit point.
A direct mask comparison confirms that ours removes 56 connections and adds
none. The masks for `max`, data-dependent `switch`, `y - y.T`, and masked QR
match the gist exactly.

After shape/canonicalization cleanup, representative full pattern graphs have
14 nodes for the five-equation system, 37 for the Rosenbrock Hessian, 80 for
the CGE system, and 51 for the CAR Hessian. Their operations are boolean and
integer, with no floating-point node outputs. Factorization graphs are less
compact: dense Cholesky has 106 outer nodes and two boolean scans. These
are graph-size observations, not runtime or memory benchmarks.

For `cholesky(x + 10*eye(n))`, the gist's dedicated rule has a smaller graph
under that same cleanup for dynamic dimensions: VJP inference has 106 outer
nodes plus 75 nodes in Scan bodies; the gist has 36 plus 77. With static 4×4
dimensions, the counts are 23 plus 72 versus 21 plus 77. The respective Scan
counts are two and one. Both infer the same 45 connections at size four.

```bash
PYTENSOR_FLAGS=mode=FAST_COMPILE python -m pytest research/vjp_sparsity/test_prototype.py -q
PYTENSOR_FLAGS=mode=CVM python -m pytest research/vjp_sparsity/test_prototype.py -q
PYTENSOR_FLAGS=mode=NUMBA python -m pytest research/vjp_sparsity/test_prototype.py -q
```

The prototype supports real, dense endpoints and the tested sparse-constant
case with dense endpoints. It returns a dense result and currently batches a
full boolean output basis; streaming or chunking remains future work.
Pessimistic rules can overestimate nonzeros. Valid differentiation domains
are assumed.
