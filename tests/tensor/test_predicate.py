import numpy as np
import pytest

import pytensor.tensor.basic as ptb
from pytensor.tensor import constant, matrix, tensor3, vector
from pytensor.tensor.linalg.decomposition.cholesky import cholesky
from pytensor.tensor.math import exp, maximum, minimum, prod, sqr
from pytensor.tensor.predicate import (
    is_provably_non_negative,
    is_provably_positive,
)


class TestProvablyPositive:
    @pytest.mark.parametrize(
        "data, strict, expected",
        [
            ([1.0, 2.0], True, True),
            ([1.0, 2.0], False, True),
            ([0.0, 2.0], True, False),
            ([0.0, 2.0], False, True),
            ([-1.0, 2.0], True, False),
            ([-1.0, 2.0], False, False),
        ],
        ids=[
            "all-positive-strict",
            "all-positive-loose",
            "zero-fails-strict",
            "zero-passes-loose",
            "negative-fails-strict",
            "negative-fails-loose",
        ],
    )
    def test_constant_data_respects_strictness(self, data, strict, expected):
        assert is_provably_positive(constant(np.array(data)), strict=strict) is expected

    @pytest.mark.parametrize(
        "make_var",
        [
            pytest.param(lambda: vector("v", dtype="uint8"), id="uint-dtype"),
            pytest.param(lambda: matrix("m").shape[0], id="shape-dim"),
            pytest.param(lambda: constant(np.array([0.5])).astype("int64"), id="cast"),
            pytest.param(lambda: exp(vector("v")), id="non-negative-op"),
        ],
    )
    def test_proves_non_negative_but_not_strict_positive(self, make_var):
        """A uint value, a shape dimension, a cast, and the output of an op flagged
        ``non_negative`` can each equal zero, so they establish ``>= 0`` but never
        strict ``> 0``."""
        var = make_var()
        assert is_provably_non_negative(var) is True
        assert is_provably_positive(var, strict=True) is False

    @pytest.mark.parametrize(
        "expr, strict, expected",
        [
            (minimum(2, 3), True, True),
            (minimum(2, 0), True, False),
            (minimum(2, 0), False, True),
            (maximum(5, -10), True, True),
            (maximum(-1, -10), True, False),
        ],
        ids=[
            "min-needs-all-positive",
            "min-zero-fails-strict",
            "min-zero-passes-loose",
            "max-needs-any-positive",
            "max-none-positive",
        ],
    )
    def test_recurses_through_min_and_max(self, expr, strict, expected):
        assert is_provably_positive(expr, strict=strict) is expected

    @pytest.mark.parametrize(
        "make_var, strict, expected",
        [
            pytest.param(
                lambda: ptb.diagonal(cholesky(matrix("X"))),
                True,
                True,
                id="diag-of-cholesky",
            ),
            pytest.param(
                lambda: ptb.diagonal(constant(np.full((2, 2), 0.5))),
                True,
                True,
                id="diag-recurses-to-matrix",
            ),
            pytest.param(
                lambda: prod(constant(np.array([1.0, 2.0]))),
                True,
                True,
                id="prod-positive",
            ),
            pytest.param(
                lambda: prod(constant(np.array([0.0, 2.0]))),
                True,
                False,
                id="prod-zero-fails-strict",
            ),
            pytest.param(
                lambda: prod(constant(np.array([0.0, 2.0]))),
                False,
                True,
                id="prod-zero-passes-loose",
            ),
            pytest.param(
                lambda: sqr(constant(np.array([1.0, 2.0]))),
                True,
                True,
                id="sqr-positive-strict",
            ),
            pytest.param(
                lambda: sqr(vector("v")),
                True,
                False,
                id="sqr-indefinite-fails-strict",
            ),
            pytest.param(
                lambda: ptb.diagonal(cholesky(matrix("X")), offset=-1),
                True,
                False,
                id="off-diagonal-of-cholesky",
            ),
            pytest.param(
                lambda: ptb.diagonal(cholesky(tensor3("X")), axis1=0, axis2=1),
                True,
                False,
                id="batch-axis-diag-of-cholesky",
            ),
            pytest.param(
                lambda: prod(vector("v", dtype="uint8")),
                True,
                False,
                id="uint-prod-can-wrap-to-zero",
            ),
            pytest.param(
                lambda: prod(
                    constant(np.array([2**32, 2**32], dtype="uint64")),
                    dtype="float64",
                ),
                True,
                False,
                id="uint-accumulator-with-float-output-can-wrap-to-zero",
            ),
        ],
    )
    def test_recurses_through_diag_prod_and_sqr(self, make_var, strict, expected):
        assert is_provably_positive(make_var(), strict=strict) is expected
