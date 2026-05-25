import pytensor.tensor as pt
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.rewriting.elemwise import FusionOptimizer


def test_fusion_no_cycle_after_multi_output():
    """Verify that multi-output fusions do not cause cycles in later discoveries.

    A bug in the ancestors_bitsets update logic for multi-output subgraphs
    incorrectly marked ancestors within the subgraph as depending on their
    own descendants.
    """
    in_a = pt.vector("a")
    a = pt.exp(in_a)
    b = pt.exp(a)
    c = pt.log(a)
    d = pt.exp(b)
    e = pt.exp(c)

    fgraph = FunctionGraph([in_a], [d, e])
    optimizer = FusionOptimizer()
    # Should not raise ValueError: graph contains cycles
    optimizer.apply(fgraph)


def test_fusion_cycle_diamond():
    """Test fusion in a diamond-like structure with mixed fuseability.

    This structure can trigger cycles if subgraphs overlap or are non-convex.
    """
    x = pt.matrix("x")
    a = pt.exp(x)
    # Path 1: fuseable
    b = pt.exp(a)
    # Path 2: unfuseable in the middle
    c = a[0]
    d = pt.exp(c)
    # Sink
    out = b + d

    fgraph = FunctionGraph([x], [out])
    # Should not raise ValueError: graph contains cycles
    FusionOptimizer().apply(fgraph)
