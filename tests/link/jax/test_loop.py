import numpy as np
import pytest
from jax.tree_util import tree_leaves

from pytensor import config, function, shared
from pytensor.graph import FunctionGraph
from pytensor.loop.basic import scan
from pytensor.scan import until
from pytensor.tensor import scalar, vector, zeros
from pytensor.tensor.random import normal
from tests.link.jax.test_basic import compare_jax_and_py


def test_scan_with_single_sequence():
    xs = vector("xs")
    _, [ys] = scan(lambda x: x * 100, sequences=[xs])

    out_fg = FunctionGraph([xs], [ys])
    compare_jax_and_py(out_fg, [np.arange(10, dtype=config.floatX)])


def test_scan_with_single_sequence_shortened_by_nsteps():
    xs = vector("xs", shape=(10,))  # JAX needs the length to be constant
    _, [ys] = scan(
        lambda x: x * 100,
        sequences=[xs],
        n_steps=9,
    )

    out_fg = FunctionGraph([xs], [ys])
    compare_jax_and_py(out_fg, [np.arange(10, dtype=config.floatX)])


def test_scan_with_multiple_sequences():
    # JAX can only handle constant n_steps
    xs = vector("xs", shape=(10,))
    ys = vector("ys", shape=(10,))
    _, [zs] = scan(
        fn=lambda x, y: x * y,
        sequences=[xs, ys],
    )

    out_fg = FunctionGraph([xs, ys], [zs])
    compare_jax_and_py(
        out_fg, [np.arange(10, dtype=xs.dtype), np.arange(10, dtype=ys.dtype)]
    )


def test_scan_with_carried_and_non_carried_states():
    x = scalar("x")
    _, [ys1, ys2] = scan(
        fn=lambda xtm1: (xtm1 + 1, (xtm1 + 1) * 2),
        init_states=[x, None],
        n_steps=10,
    )
    out_fg = FunctionGraph([x], [ys1, ys2])
    compare_jax_and_py(out_fg, [-1])


def test_scan_with_sequence_and_carried_state():
    xs = vector("xs")
    _, [ys] = scan(
        fn=lambda x, ytm1: (ytm1 + 1) * x,
        init_states=[zeros(())],
        sequences=[xs],
    )
    out_fg = FunctionGraph([xs], [ys])
    compare_jax_and_py(out_fg, [[1, 2, 3]])


def test_scan_with_rvs():
    rng = shared(np.random.default_rng(123))

    [final_rng, _], [rngs, xs] = scan(
        fn=lambda prev_rng: normal(rng=prev_rng).owner.outputs,
        init_states=[rng, None],
        n_steps=10,
    )

    # First without updates
    fn = function([], xs, mode="JAX", updates=None)
    res1 = fn()
    res2 = fn()
    assert not set(tuple(np.array(res1))) ^ set(tuple(np.array(res2)))

    # Now with updates
    fn = function([], xs, mode="JAX", updates={rng: final_rng})
    res1 = fn()
    res2 = fn()
    assert not set(tuple(np.array(res1))) & set(tuple(np.array(res2)))

    # Test traced rngs
    fn = function([], [rngs, final_rng], mode="JAX")
    rngs_res, final_rng_res = fn()
    assert isinstance(rngs_res, list) and len(rngs_res) == 10
    assert [np.array(v).tolist() for v in tree_leaves(rngs_res[-1])] == [
        np.array(v).tolist() for v in tree_leaves(final_rng_res)
    ]


def test_while_scan_fails():
    _, [xs] = scan(
        fn=lambda x: (x + 1, until((x + 1) >= 9)),
        init_states=[-1],
        n_steps=20,
    )

    out_fg = FunctionGraph([], [xs])
    with pytest.raises(
        NotImplementedError,
        match="Scan ops with while condition cannot be transpiled JAX",
    ):
        compare_jax_and_py(out_fg, [])
