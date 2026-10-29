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

- `nonzero_rewrites_db`: a `SequenceDB` with shape inference,
  pre-canonicalization, equilibrium lifting, pessimistic completion, marker
  removal, and post-canonicalization.
- `nonzero_lift_db`: an `EquilibriumDB` of independently registered rules.
- `register_nonzero(tracks)`: register a rule against the actual producing Ops.
- `Cotangent(value)`: an IR demarcator identifying cotangent inputs without a
  graph feature or an external variable-to-mask mapping.

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
The test suite demonstrates an external rule improving a previously
pessimistic result. Standard database queries can select or exclude rewrites.

`MayBeNonzero(value)` is temporary inference IR, not a replacement for the VJP.
It distinguishes structural nonzero queries from ordinary runtime
`value != 0` predicates already inside the graph. Rewriting those original
predicates as if they were analysis queries could change which branches are
possible. All query markers are removed before returning the result.

`Cotangent(value)` is also temporary. Its nonzero query becomes the ordinary
`value != 0`; other unknown inputs become potentially nonzero everywhere.
The final cleanup removes the cotangent demarcators. Only the generic input
and pessimistic-completion rules track `MayBeNonzero` itself.

Lifting and pessimistic completion are ordered stages of the same database.
Known rules first reach equilibrium, including processing queries they create
on upstream producers. Only then can remaining queries become dense; applying
that default earlier could prevent those producer rules from running.

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
| Constant-condition switch | Select branch masks |
| Unknown-condition switch | Union branch masks |

Conservative indexing/scatter and dot rules preserve the fact that an
all-zero incoming mask contributes nothing. Any query left unresolved after
the lifting database reaches equilibrium becomes an all-true mask in the
final pessimistic pass.

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
intermediates. This avoids the constant-folding false negative identified in
the review.

The full rewrite pipeline runs on the private clone. Original primal bindings
are restored only after it finishes, with no further rewrites. This follows
PyMC's `construct_ir_fgraph` isolation pattern. A regression test snapshots the
user's original node inputs and validates their attached `FunctionGraph`; it
reproduced the earlier mutation caused by canonicalizing after restoration.

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

Eleven additional checks cover cancellation, the folded-intermediate regression
in three modes, repeated outputs, an existing zero-gradient rule, value
independence, ordinary VJP input, external rewrite registration, and keeping
runtime nonzero predicates separate from inference queries, and preserving the
original user graph.

All **32 tests pass in each of FAST_COMPILE, CVM, and NUMBA**.

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
