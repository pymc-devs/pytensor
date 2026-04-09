from functools import partial

import numpy as np
import pytest
from scipy import linalg as scipy_linalg

from pytensor import function
from pytensor.configdefaults import config
from pytensor.tensor.linalg import qr
from pytensor.tensor.type import tensor
from tests import unittest_tools as utt


@pytest.mark.parametrize(
    "mode, names",
    [
        ("economic", ["Q", "R"]),
        ("full", ["Q", "R"]),
        ("r", ["R"]),
        ("raw", ["H", "tau", "R"]),
    ],
)
@pytest.mark.parametrize("pivoting", [True, False])
def test_qr_modes(mode, names, pivoting):
    rng = np.random.default_rng(utt.fetch_seed())
    A_val = rng.random((4, 4)).astype(config.floatX)

    if pivoting:
        names = [*names, "pivots"]

    A = tensor("A", dtype=config.floatX, shape=(None, None))

    f = function([A], qr(A, mode=mode, pivoting=pivoting))

    outputs_pt = f(A_val)
    outputs_sp = scipy_linalg.qr(A_val, mode=mode, pivoting=pivoting)

    if mode == "raw":
        # The first output of scipy's qr is a tuple when mode is raw; flatten it for easier iteration
        outputs_sp = (*outputs_sp[0], *outputs_sp[1:])
    elif mode == "r" and not pivoting:
        # Here there's only one output from the pytensor function; wrap it in a list for iteration
        outputs_pt = [outputs_pt]

    for out_pt, out_sp, name in zip(outputs_pt, outputs_sp, names):
        np.testing.assert_allclose(out_pt, out_sp, err_msg=f"{name} disagrees")


@pytest.mark.parametrize(
    "shape, gradient_test_case, mode",
    (
        [(s, c, "economic") for s in [(3, 3), (6, 3), (3, 6)] for c in [0, 1, 2]]
        + [(s, c, "full") for s in [(3, 3), (6, 3), (3, 6)] for c in [0, 1, 2]]
        + [(s, 0, "r") for s in [(3, 3), (6, 3), (3, 6)]]
        + [((3, 3), 0, "raw")]
    ),
    ids=(
        [
            f"shape={s}, gradient_test_case={c}, mode=economic"
            for s in [(3, 3), (6, 3), (3, 6)]
            for c in ["Q", "R", "both"]
        ]
        + [
            f"shape={s}, gradient_test_case={c}, mode=full"
            for s in [(3, 3), (6, 3), (3, 6)]
            for c in ["Q", "R", "both"]
        ]
        + [f"shape={s}, gradient_test_case=R, mode=r" for s in [(3, 3), (6, 3), (3, 6)]]
        + ["shape=(3, 3), gradient_test_case=Q, mode=raw"]
    ),
)
@pytest.mark.parametrize("is_complex", [True, False], ids=["complex", "real"])
def test_qr_grad(shape, gradient_test_case, mode, is_complex):
    rng = np.random.default_rng(utt.fetch_seed())

    def _test_fn(x, case=2, mode="reduced"):
        if case == 0:
            return qr(x, mode=mode)[0].sum()
        elif case == 1:
            return qr(x, mode=mode)[1].sum()
        elif case == 2:
            Q, R = qr(x, mode=mode)
            return Q.sum() + R.sum()

    if is_complex:
        pytest.xfail("Complex inputs currently not supported by verify_grad")

    m, n = shape
    a = rng.standard_normal(shape).astype(config.floatX)
    if is_complex:
        a += 1j * rng.standard_normal(shape).astype(config.floatX)

    if mode == "raw":
        with pytest.raises(NotImplementedError):
            utt.verify_grad(
                partial(_test_fn, case=gradient_test_case, mode=mode),
                [a],
                rng=np.random,
            )

    elif mode == "full" and m > n:
        with pytest.raises(AssertionError):
            utt.verify_grad(
                partial(_test_fn, case=gradient_test_case, mode=mode),
                [a],
                rng=np.random,
            )

    else:
        utt.verify_grad(
            partial(_test_fn, case=gradient_test_case, mode=mode), [a], rng=np.random
        )
