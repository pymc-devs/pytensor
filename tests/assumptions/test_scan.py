import pytensor.tensor as pt
from pytensor import function
from pytensor.assumptions import DIAGONAL, FactState
from pytensor.assumptions.specify import assume
from pytensor.scan.basic import scan
from pytensor.scan.op import Scan
from tests.assumptions.conftest import make_fgraph


def test_map_preserving_body_forwards_property():
    """map (nit-sot): ``s @ s`` of a diagonal ``s`` is diagonal, so the stack is."""
    x = pt.tensor3("seq")
    seq = assume(x, diagonal=True)
    out = scan(lambda s: s @ s, sequences=[seq], return_updates=False)
    _, af = make_fgraph(out, inputs=[x])
    assert af.check(out, DIAGONAL)


def test_map_breaking_body_is_unknown():
    """map: ``exp`` does not preserve the zero pattern, so the property is lost."""
    x = pt.tensor3("seq")
    seq = assume(x, diagonal=True)
    out = scan(lambda s: pt.exp(s), sequences=[seq], return_updates=False)
    _, af = make_fgraph(out, inputs=[x])
    assert af.get(out, DIAGONAL) == FactState.UNKNOWN


def test_non_sequence_tagged_input_forwards_property():
    """The delegate seeds non-sequence inner inputs; transposing a diagonal
    non-sequence keeps it diagonal."""
    x = pt.matrix("m", shape=(4, 4))
    m = assume(x, diagonal=True)
    out = scan(lambda mat: mat.T, non_sequences=[m], n_steps=5, return_updates=False)
    _, af = make_fgraph(out, inputs=[x])
    assert af.check(out, DIAGONAL)


def test_recurrence_preserving_body_forwards_property():
    """recurrence (sit-sot): ``2 * prev`` keeps a diagonal carried state diagonal
    at every step."""
    x = pt.matrix("init", shape=(3, 3))
    init = assume(x, diagonal=True)
    out = scan(
        lambda prev: 2.0 * prev,
        outputs_info=[init],
        n_steps=5,
        return_updates=False,
    )
    _, af = make_fgraph(out, inputs=[x])
    assert af.check(out, DIAGONAL)


def test_recurrence_breaking_body_is_unknown():
    """recurrence: adding a non-diagonal term each step breaks the property."""
    x = pt.matrix("init", shape=(3, 3))
    init = assume(x, diagonal=True)
    out = scan(
        lambda prev: prev + pt.ones_like(prev),
        outputs_info=[init],
        n_steps=5,
        return_updates=False,
    )
    _, af = make_fgraph(out, inputs=[x])
    assert af.get(out, DIAGONAL) == FactState.UNKNOWN


def test_mit_sot_multi_tap_recurrence_forwards_property():
    """2-tap recurrence: ``p1 + p2`` needs *both* taps diagonal, pinning that the
    delegate seeds both from the multi-tap init buffer."""
    x = pt.tensor("init", shape=(2, 3, 3))
    init = assume(x, diagonal=True)
    out = scan(
        lambda p2, p1: p1 + p2,
        outputs_info=[dict(initial=init, taps=[-2, -1])],
        n_steps=5,
        return_updates=False,
    )
    _, af = make_fgraph(out, inputs=[x])
    assert af.check(out, DIAGONAL)


def test_map_and_recurrence_combined_forwards_property():
    """A scan with both a sequence and a recurrence -- ``prev @ s`` needs both the
    carried state and the sequence element diagonal, pinning that the delegate
    seeds the recurrent and sequence inner inputs from the right outer inputs."""
    sx = pt.tensor("seq", shape=(5, 3, 3))
    seq = assume(sx, diagonal=True)
    ix = pt.matrix("init", shape=(3, 3))
    init = assume(ix, diagonal=True)
    out = scan(
        lambda s, prev: prev @ s,
        sequences=[seq],
        outputs_info=[init],
        return_updates=False,
    )
    _, af = make_fgraph(out, inputs=[sx, ix])
    assert af.check(out, DIAGONAL)


def test_multi_output_scan_maps_outputs_independently():
    """A scan with two outputs -- one diagonal-preserving, one breaking -- the
    delegate maps each inner output to its own outer output independently."""
    x = pt.tensor3("seq")
    seq = assume(x, diagonal=True)
    doubled, exponential = scan(
        lambda s: [s + s, pt.exp(s)], sequences=[seq], return_updates=False
    )
    _, af = make_fgraph(doubled, exponential, inputs=[x])
    assert af.check(doubled, DIAGONAL)
    assert af.get(exponential, DIAGONAL) == FactState.UNKNOWN


def test_nested_scan_forwards_property():
    """A scan whose body runs an inner scan -- the delegate recurses into the
    inner Scan while still inferring the outer one."""
    x = pt.tensor3("seq")
    seq = assume(x, diagonal=True)

    def body(s):
        inner = scan(
            lambda m: 2.0 * m, non_sequences=[s], n_steps=1, return_updates=False
        )
        return inner[-1]

    out = scan(body, sequences=[seq], return_updates=False)
    _, af = make_fgraph(out, inputs=[x])
    assert af.check(out, DIAGONAL)


def test_outer_assumption_lifts_into_scan_inner_graph():
    """`lift_assumptions_into_scan` re-asserts a sequence's assumption on the
    inner input, so a rewrite of the inner graph sees it: ``inv(X) @ y`` of a
    positive-definite ``X`` specializes to a Cholesky solve in the loop body."""
    Xs = pt.tensor("Xs", shape=(4, 3, 3))
    ys = pt.tensor("ys", shape=(4, 3, 3))
    out = scan(
        lambda Xt, yt: pt.linalg.inv(Xt) @ yt,
        sequences=[assume(Xs, positive_definite=True), ys],
        return_updates=False,
    )
    fn = function([Xs, ys], out)
    [scan_node] = [n for n in fn.maker.fgraph.toposort() if isinstance(n.op, Scan)]
    inner_ops = {type(n.op).__name__ for n in scan_node.op.fgraph.toposort()}
    assert "MatrixInverse" not in inner_ops
    assert "CholeskySolve" in inner_ops


def test_non_sequence_assumption_lifts_into_scan_inner_graph():
    """Mirror of the sequence test but for a non-sequence: a positive-definite
    ``X`` passed via ``non_sequences`` lifts into the inner graph too, so the
    per-step ``inv(X) @ y_t`` still specializes to a Cholesky solve."""
    X = pt.matrix("X", shape=(3, 3))
    ys = pt.tensor("ys", shape=(4, 3, 3))
    out = scan(
        lambda yt, X: pt.linalg.inv(X) @ yt,
        sequences=[ys],
        non_sequences=[assume(X, positive_definite=True)],
        return_updates=False,
    )
    fn = function([X, ys], out)
    [scan_node] = [n for n in fn.maker.fgraph.toposort() if isinstance(n.op, Scan)]
    inner_ops = {type(n.op).__name__ for n in scan_node.op.fgraph.toposort()}
    assert "MatrixInverse" not in inner_ops
    assert "CholeskySolve" in inner_ops


def test_scan_grad_compiles_with_recurrence_assumption():
    """Backward AD flips a sit-sot forward output into a sequence input of the
    backward scan. With ``diagonal=True`` on the recurrence's init the
    gradient must still compile and stay numerically correct."""
    import numpy as np

    init_raw = pt.matrix("init", shape=(3, 3))
    init = assume(init_raw, diagonal=True)
    out = scan(
        lambda prev: 2.0 * prev,
        outputs_info=[init],
        n_steps=4,
        return_updates=False,
    )
    g = pt.grad(out[-1].sum(), init_raw)
    fn = function([init_raw], g)
    np.testing.assert_allclose(fn(np.eye(3)), 16.0 * np.ones((3, 3)))
