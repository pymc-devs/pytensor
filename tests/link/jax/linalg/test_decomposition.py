import numpy as np
import pytest

import pytensor.tensor as pt
import tests.unittest_tools as utt
from pytensor.configdefaults import config
from pytensor.tensor import subtensor as pt_subtensor
from pytensor.tensor.linalg.decomposition import lu, qr, schur, svd
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky, cholesky
from pytensor.tensor.linalg.decomposition.eigen import eig, eigh, eigvalsh
from pytensor.tensor.linalg.inverse import matrix_inverse
from pytensor.tensor.linalg.summary import det, slogdet
from pytensor.tensor.math import clip, cosh
from pytensor.tensor.type import matrix, vector
from pytensor.tensor.type_other import NoneConst
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def test_jax_basic():
    rng = np.random.default_rng(28494)

    x = matrix("x")
    y = matrix("y")
    b = vector("b")

    # `ScalarOp`
    z = cosh(x**2 + y / 3.0)

    # `[Inc]Subtensor`
    out = pt_subtensor.set_subtensor(z[0], -10.0)
    out = pt_subtensor.inc_subtensor(out[0, 1], 2.0)
    out = out[:5, :3]

    test_input_vals = [
        np.tile(np.arange(10), (10, 1)).astype(config.floatX),
        np.tile(np.arange(10, 20), (10, 1)).astype(config.floatX),
    ]
    _, [jax_res] = compare_jax_and_py([x, y], [out], test_input_vals)

    # Confirm that the `Subtensor` slice operations are correct
    assert jax_res.shape == (5, 3)

    # Confirm that the `IncSubtensor` operations are correct
    assert jax_res[0, 0] == -10.0
    assert jax_res[0, 1] == -8.0

    out = clip(x, y, 5)
    compare_jax_and_py([x, y], [out], test_input_vals)

    out = pt.diagonal(x, 0)
    compare_jax_and_py(
        [x], [out], [np.arange(10 * 10).reshape((10, 10)).astype(config.floatX)]
    )

    out = cholesky(x)
    compare_jax_and_py(
        [x],
        [out],
        [
            (np.eye(10) + rng.standard_normal(size=(10, 10)) * 0.01).astype(
                config.floatX
            )
        ],
    )

    # not sure why this isn't working yet with lower=False
    out = Cholesky(lower=False)(x)
    compare_jax_and_py(
        [x],
        [out],
        [
            (np.eye(10) + rng.standard_normal(size=(10, 10)) * 0.01).astype(
                config.floatX
            )
        ],
    )

    out = pt.linalg.solve(x, b)
    compare_jax_and_py(
        [x, b],
        [out],
        [
            np.eye(10).astype(config.floatX),
            np.arange(10).astype(config.floatX),
        ],
    )

    out = pt.diag(b)
    compare_jax_and_py([b], [out], [np.arange(10).astype(config.floatX)])

    out = det(x)
    compare_jax_and_py(
        [x], [out], [np.arange(10 * 10).reshape((10, 10)).astype(config.floatX)]
    )

    out = matrix_inverse(x)
    compare_jax_and_py(
        [x],
        [out],
        [
            (np.eye(10) + rng.standard_normal(size=(10, 10)) * 0.01).astype(
                config.floatX
            )
        ],
    )

    def assert_fn(x, y):
        np.testing.assert_allclose(x.astype(config.floatX), y, rtol=1e-3)

    M = rng.normal(size=(3, 3))
    X = M.dot(M.T)

    outs = qr.qr(x, mode="full")
    compare_jax_and_py([x], outs, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = qr.qr(x, mode="economic")
    compare_jax_and_py([x], outs, [X.astype(config.floatX)], assert_fn=assert_fn)


def test_jax_basic_multiout():
    rng = np.random.default_rng(213234)

    M = rng.normal(size=(3, 3))
    X = M.dot(M.T)

    x = matrix("x")

    outs = eig(x)

    def assert_fn(x, y):
        np.testing.assert_allclose(x.astype(config.floatX), y, rtol=1e-3)

    compare_jax_and_py([x], outs, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = eigh(x)
    compare_jax_and_py([x], outs, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = svd.svd(x)
    compare_jax_and_py([x], outs, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = slogdet(x)
    compare_jax_and_py([x], outs, [X.astype(config.floatX)], assert_fn=assert_fn)


@pytest.mark.parametrize("lower", [False, True])
def test_jax_eigvalsh(lower):
    A = matrix("A")
    B = matrix("B")

    out = eigvalsh(A, B, lower=lower)

    with pytest.raises(NotImplementedError):
        compare_jax_and_py(
            [A, B],
            [out],
            [
                np.array(
                    [[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]]
                ).astype(config.floatX),
                np.array(
                    [[10, 0, 1, 3], [0, 12, 7, 8], [1, 7, 14, 2], [3, 8, 2, 16]]
                ).astype(config.floatX),
            ],
        )
    out_no_b = eigvalsh(A, NoneConst, lower=lower)
    compare_jax_and_py(
        [A],
        [out_no_b],
        [
            np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]]).astype(
                config.floatX
            ),
        ],
    )


@pytest.mark.parametrize(
    "permute_l, p_indices",
    [(True, False), (False, True), (False, False)],
    ids=["PL", "p_indices", "P"],
)
@pytest.mark.parametrize("complex", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("shape", [(3, 5, 5), (5, 5)], ids=["batched", "not_batched"])
def test_jax_lu(permute_l, p_indices, complex, shape: tuple[int]):
    rng = np.random.default_rng()
    A = pt.tensor(
        "A",
        shape=shape,
        dtype=f"complex{int(config.floatX[-2:]) * 2}" if complex else config.floatX,
    )
    out = lu.lu(A, permute_l=permute_l, p_indices=p_indices)

    x = rng.normal(size=shape).astype(config.floatX)
    if complex:
        x = x + 1j * rng.normal(size=shape).astype(config.floatX)

    if p_indices:
        with pytest.raises(
            ValueError, match="JAX does not support the p_indices argument"
        ):
            compare_jax_and_py(graph_inputs=[A], graph_outputs=out, test_inputs=[x])
    else:
        compare_jax_and_py(graph_inputs=[A], graph_outputs=out, test_inputs=[x])


@pytest.mark.parametrize("shape", [(5, 5), (5, 5, 5)], ids=["matrix", "batch"])
def test_jax_lu_factor(shape):
    rng = np.random.default_rng(utt.fetch_seed())
    A = pt.tensor(name="A", shape=shape)
    A_value = rng.normal(size=shape).astype(config.floatX)
    out = lu.lu_factor(A)

    compare_jax_and_py(
        [A],
        out,
        [A_value],
    )


@pytest.mark.parametrize("mode", ["full", "r"])
def test_jax_qr(mode):
    rng = np.random.default_rng(utt.fetch_seed())
    A = pt.tensor(name="A", shape=(5, 5))
    A_val = rng.normal(size=(5, 5)).astype(config.floatX)
    out = qr.qr(A, mode=mode)

    compare_jax_and_py([A], out, [A_val])


@pytest.mark.parametrize("output", ["real", "complex"])
def test_jax_schur(output):
    rng = np.random.default_rng(utt.fetch_seed())
    A = pt.tensor(name="A", shape=(5, 5))
    A_val = rng.normal(size=(5, 5)).astype(config.floatX)
    T, Z = schur.schur(A, output=output)

    compare_jax_and_py([A], [T, Z], [A_val])
