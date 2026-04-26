import pytest

import pytensor.tensor as pt
from pytensor.graph import rewrite_graph
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.assumptions import (
    ALL_KEYS,
    DIAGONAL,
    LOWER_TRIANGULAR,
    POSITIVE_DEFINITE,
    SYMMETRIC,
    UPPER_TRIANGULAR,
    AssumptionFeature,
    AssumptionKey,
    FactState,
    register_assumption,
)
from pytensor.tensor.basic import alloc_diag
from pytensor.tensor.blockwise import Blockwise


def make_fgraph(*outputs, **kwargs):
    fg = FunctionGraph(outputs=outputs, clone=False, **kwargs)
    af = AssumptionFeature()
    fg.attach_feature(af)
    return fg, af


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (FactState.UNKNOWN, FactState.UNKNOWN, FactState.UNKNOWN),
        (FactState.TRUE, FactState.UNKNOWN, FactState.TRUE),
        (FactState.FALSE, FactState.UNKNOWN, FactState.FALSE),
        (FactState.TRUE, FactState.FALSE, FactState.CONFLICT),
    ],
    ids=["unknown+unknown", "true+unknown", "false+unknown", "true+false"],
)
def test_fact_state_join(left, right, expected):
    assert FactState.join(left, right) == expected


@pytest.mark.parametrize(
    "state, expected",
    [
        (FactState.TRUE, True),
        (FactState.UNKNOWN, False),
        (FactState.FALSE, False),
        (FactState.CONFLICT, False),
    ],
)
def test_fact_state_bool(state, expected):
    assert bool(state) is expected


def test_double_attach_keeps_single_feature():
    x = pt.matrix("x")
    fg = FunctionGraph([x], [x], clone=False)
    fg.attach_feature(AssumptionFeature())
    fg.attach_feature(AssumptionFeature())
    assert sum(isinstance(f, AssumptionFeature) for f in fg._features) == 1


def test_bare_input_is_unknown():
    x = pt.matrix("x")
    _, af = make_fgraph(x)
    assert af.get(x, DIAGONAL) == FactState.UNKNOWN


def test_user_fact_lifecycle():
    x = pt.matrix("x")
    _, af = make_fgraph(x)

    af.set_user_fact(x, DIAGONAL, FactState.TRUE)
    assert af.check(x, DIAGONAL)

    af.replace_user_fact(x, DIAGONAL, FactState.FALSE)
    assert af.get(x, DIAGONAL) == FactState.FALSE

    af.clear_user_fact(x, DIAGONAL)
    assert af.get(x, DIAGONAL) == FactState.UNKNOWN


def test_user_fact_invalidates_downstream_cache():
    x = pt.matrix("x", shape=(3, 3))
    inv_x = pt.linalg.inv(x)
    _, af = make_fgraph(inv_x, inputs=[x])

    assert af.get(inv_x, DIAGONAL) == FactState.UNKNOWN
    af.set_user_fact(x, DIAGONAL, FactState.TRUE)
    assert af.check(inv_x, DIAGONAL)
    af.clear_user_fact(x, DIAGONAL)
    assert af.get(inv_x, DIAGONAL) == FactState.UNKNOWN


def test_op_local_infer_assumption():
    from pytensor.graph.basic import Apply
    from pytensor.graph.op import Op
    from pytensor.tensor.type import TensorType

    class AlwaysSymmetricOp(Op):
        __props__ = ()

        def make_node(self, x):
            return Apply(self, [x], [TensorType(dtype=x.dtype, shape=(None, None))()])

        def perform(self, node, inputs, outputs):
            outputs[0][0] = inputs[0]

        def infer_assumption(self, key, feature, fgraph, node, input_states):
            if key == SYMMETRIC:
                return [FactState.TRUE]
            return NotImplemented

    x = pt.matrix("x")
    y = AlwaysSymmetricOp()(x)
    _, af = make_fgraph(y, inputs=[x])
    assert af.check(y, SYMMETRIC)
    assert af.get(y, DIAGONAL) == FactState.UNKNOWN


def test_register_custom_assumption_key():
    INVERTIBLE = AssumptionKey("invertible")
    from pytensor.tensor.basic import Eye

    @register_assumption(INVERTIBLE, Eye)
    def _eye_invertible(op, feature, fgraph, node, input_states):
        return [FactState.TRUE]

    e = pt.eye(5)
    _, af = make_fgraph(e)
    assert af.check(e, INVERTIBLE)


@pytest.mark.parametrize(
    "stronger, weaker",
    [
        (DIAGONAL, SYMMETRIC),
        (DIAGONAL, LOWER_TRIANGULAR),
        (DIAGONAL, UPPER_TRIANGULAR),
        (POSITIVE_DEFINITE, SYMMETRIC),
    ],
)
def test_implication(stronger, weaker):
    x = pt.matrix("x", shape=(3, 3))
    _, af = make_fgraph(x)
    af.set_user_fact(x, stronger, FactState.TRUE)
    assert af.check(x, weaker)


def test_symmetric_does_not_imply_diagonal():
    x = pt.matrix("x", shape=(3, 3))
    _, af = make_fgraph(x)
    af.set_user_fact(x, SYMMETRIC, FactState.TRUE)
    assert not af.check(x, DIAGONAL)


def test_eye_identity_has_all_properties():
    e = pt.eye(5)
    _, af = make_fgraph(e)
    for key in ALL_KEYS:
        assert af.check(e, key), f"Eye should be {key}"


@pytest.mark.parametrize(
    "eye_args",
    [
        pytest.param(dict(n=5, k=1), id="offset"),
        pytest.param(dict(n=5, m=3), id="rectangular"),
    ],
)
def test_eye_non_identity_is_unknown(eye_args):
    e = pt.eye(**eye_args)
    _, af = make_fgraph(e)
    assert af.get(e, DIAGONAL) == FactState.UNKNOWN


def test_eye_symbolic_same_shape_is_identity():
    n = pt.iscalar("n")
    e = pt.eye(n, n, 0)
    _, af = make_fgraph(e, inputs=[n])
    assert af.check(e, DIAGONAL)


@pytest.mark.parametrize(
    "key", [DIAGONAL, SYMMETRIC, LOWER_TRIANGULAR, UPPER_TRIANGULAR]
)
def test_alloc_diag_properties(key):
    v = pt.vector("v", shape=(5,))
    d = pt.diag(v)
    _, af = make_fgraph(d, inputs=[v])
    assert af.check(d, key)


def test_zeros_matrix_is_diagonal():
    z = pt.zeros((5, 5))
    _, af = make_fgraph(z)
    assert af.check(z, DIAGONAL)


def test_ones_matrix_is_not_diagonal():
    o = pt.ones((5, 5))
    _, af = make_fgraph(o)
    assert af.get(o, DIAGONAL) == FactState.UNKNOWN


@pytest.mark.parametrize(
    "lower, expected_true, expected_false",
    [
        (True, LOWER_TRIANGULAR, UPPER_TRIANGULAR),
        (False, UPPER_TRIANGULAR, LOWER_TRIANGULAR),
    ],
)
def test_cholesky_triangularity(lower, expected_true, expected_false):
    x = pt.matrix("x", shape=(3, 3))
    L = pt.linalg.cholesky(x, lower=lower)
    _, af = make_fgraph(L, inputs=[x])
    assert af.check(L, expected_true)
    assert not af.check(L, expected_false)


def test_cholesky_of_diagonal_is_diagonal():
    v = pt.vector("v", shape=(3,))
    L = pt.linalg.cholesky(pt.diag(v), lower=True)
    _, af = make_fgraph(L, inputs=[v])
    assert af.check(L, DIAGONAL)


@pytest.mark.parametrize(
    "key", [DIAGONAL, LOWER_TRIANGULAR, SYMMETRIC, POSITIVE_DEFINITE]
)
def test_inv_propagates_property(key):
    x = pt.matrix("x", shape=(3, 3))
    inv_x = pt.linalg.inv(x)
    _, af = make_fgraph(inv_x, inputs=[x])
    af.set_user_fact(x, key, FactState.TRUE)
    assert af.check(inv_x, key)


@pytest.mark.parametrize("key", [DIAGONAL, SYMMETRIC])
def test_pinv_propagates_property(key):
    x = pt.matrix("x", shape=(3, 3))
    px = pt.linalg.pinv(x)
    _, af = make_fgraph(px, inputs=[x])
    af.set_user_fact(x, key, FactState.TRUE)
    assert af.check(px, key)


def test_block_diag_of_diagonal_blocks_is_diagonal():
    bd = pt.linalg.block_diag(pt.eye(3), pt.eye(4))
    _, af = make_fgraph(bd)
    assert af.check(bd, DIAGONAL)


def test_block_diag_of_generic_blocks_is_unknown():
    x = pt.matrix("x", shape=(3, 3))
    y = pt.matrix("y", shape=(4, 4))
    bd = pt.linalg.block_diag(x, y)
    _, af = make_fgraph(bd, inputs=[x, y])
    assert af.get(bd, DIAGONAL) == FactState.UNKNOWN


@pytest.mark.parametrize("key", [DIAGONAL, SYMMETRIC, POSITIVE_DEFINITE])
def test_kron_of_eyes_propagates_property(key):
    k = pt.linalg.kron(pt.eye(3), pt.eye(4))
    _, af = make_fgraph(k)
    assert af.check(k, key)


def test_kron_with_generic_input_not_diagonal():
    x = pt.matrix("x", shape=(3, 3))
    k = pt.linalg.kron(x, pt.eye(4))
    _, af = make_fgraph(k, inputs=[x])
    assert af.get(k, DIAGONAL) == FactState.UNKNOWN


def test_matmul_diagonal_diagonal_is_diagonal():
    y = pt.eye(5) @ pt.diag(pt.ones(5))
    _, af = make_fgraph(y)
    assert af.check(y, DIAGONAL)


def test_matmul_diagonal_generic_is_unknown():
    x = pt.matrix("x", shape=(5, 5))
    y = pt.eye(5) @ x
    _, af = make_fgraph(y, inputs=[x])
    assert af.get(y, DIAGONAL) == FactState.UNKNOWN


def test_dot_xxt_is_psd_and_symmetric():
    """dot(x, x.T) is positive semi-definite and symmetric."""
    x = pt.matrix("x")
    gram = pt.dot(x, x.T)
    _, af = make_fgraph(gram, inputs=[x])
    assert af.check(gram, POSITIVE_DEFINITE)
    assert af.check(gram, SYMMETRIC)


def test_dot_xtx_is_psd_and_symmetric():
    """dot(x.T, x) is positive semi-definite and symmetric."""
    x = pt.matrix("x")
    gram = pt.dot(x.T, x)
    _, af = make_fgraph(gram, inputs=[x])
    assert af.check(gram, POSITIVE_DEFINITE)
    assert af.check(gram, SYMMETRIC)


def test_dot_xxH_complex_is_psd():
    """dot(x, x.conj().T) with complex dtype IS positive semi-definite."""
    x = pt.cmatrix("x")  # complex64

    # Test x @ x.conj().T
    gram1 = pt.dot(x, x.conj().T)
    _, af = make_fgraph(gram1, inputs=[x])
    assert af.check(gram1, POSITIVE_DEFINITE)

    # Test x @ x.T.conj() (equivalent)
    gram2 = pt.dot(x, x.T.conj())
    _, af2 = make_fgraph(gram2, inputs=[x])
    assert af2.check(gram2, POSITIVE_DEFINITE)


def test_dot_orthogonal_xxt_is_diagonal():
    """dot(x, x.T) is diagonal when x is orthogonal."""
    x = pt.dmatrix("x")
    x_orth = pt.specify_assumptions(x, orthogonal=True)
    y = pt.dot(x_orth, x_orth.T)
    _, af = make_fgraph(y, inputs=[x])
    assert af.check(y, DIAGONAL)


class TestSubtensorDiagonalPreservation:
    def test_set_diagonal_entries(self):
        d = pt.eye(5)
        idx = pt.arange(5)
        v = pt.vector("v", shape=(5,))
        y = pt.set_subtensor(d[idx, idx], v)
        _, af = make_fgraph(y, inputs=[v])
        assert af.check(y, DIAGONAL)

    def test_inc_diagonal_entries(self):
        d = pt.eye(5)
        idx = pt.arange(5)
        v = pt.vector("v", shape=(5,))
        y = pt.inc_subtensor(d[idx, idx], v)
        _, af = make_fgraph(y, inputs=[v])
        assert af.check(y, DIAGONAL)

    def test_set_scalar_diagonal_entry(self):
        d = pt.eye(5)
        i = pt.iscalar("i")
        y = rewrite_graph(pt.set_subtensor(d[i, i], 1.0), include=["merge"])
        _, af = make_fgraph(y, inputs=[i])
        assert af.check(y, DIAGONAL)

    def test_set_off_diagonal_entries_is_unknown(self):
        d = pt.eye(5)
        v = pt.vector("v", shape=(5,))
        y = pt.set_subtensor(d[pt.arange(5), pt.arange(1, 6)], v)
        _, af = make_fgraph(y, inputs=[v])
        assert af.get(y, DIAGONAL) == FactState.UNKNOWN

    def test_non_diagonal_base_is_unknown(self):
        x = pt.matrix("x", shape=(5, 5))
        v = pt.vector("v", shape=(5,))
        y = pt.set_subtensor(x[pt.arange(5), pt.arange(5)], v)
        _, af = make_fgraph(y, inputs=[x, v])
        assert af.get(y, DIAGONAL) == FactState.UNKNOWN


def test_transpose_preserves_diagonal():
    e = pt.eye(5)
    _, af = make_fgraph(e.T)
    assert af.check(e.T, DIAGONAL)


def test_expand_dims_preserves_diagonal():
    e = pt.eye(5)
    e3d = e.dimshuffle("x", 0, 1)
    _, af = make_fgraph(e3d)
    assert af.check(e3d, DIAGONAL)


def test_transpose_of_generic_matrix_is_unknown():
    x = pt.matrix("x", shape=(3, 3))
    xT = x.T
    _, af = make_fgraph(xT, inputs=[x])
    assert af.get(xT, DIAGONAL) == FactState.UNKNOWN


class TestElemwiseAssumptions:
    def test_mul_diagonal_by_anything_is_diagonal(self):
        x = pt.matrix("x", shape=(5, 5))
        y = pt.eye(5) * x
        _, af = make_fgraph(y, inputs=[x])
        assert af.check(y, DIAGONAL)

    def test_add_diagonal_plus_diagonal(self):
        y = pt.eye(5) + pt.diag(pt.ones(5))
        _, af = make_fgraph(y)
        assert af.check(y, DIAGONAL)

    def test_add_diagonal_plus_generic_is_unknown(self):
        x = pt.matrix("x", shape=(5, 5))
        y = pt.eye(5) + x
        _, af = make_fgraph(y, inputs=[x])
        assert not af.check(y, DIAGONAL)

    def test_sub_diagonal_minus_diagonal(self):
        y = pt.eye(5) - pt.diag(pt.ones(5))
        _, af = make_fgraph(y)
        assert af.check(y, DIAGONAL)

    def test_truediv_diagonal_over_anything_is_diagonal(self):
        x = pt.matrix("x", shape=(5, 5))
        y = pt.eye(5) / x
        _, af = make_fgraph(y, inputs=[x])
        assert af.check(y, DIAGONAL)

    def test_truediv_anything_over_diagonal_is_unknown(self):
        x = pt.matrix("x", shape=(5, 5))
        y = x / pt.eye(5)
        _, af = make_fgraph(y, inputs=[x])
        assert not af.check(y, DIAGONAL)

    @pytest.mark.parametrize(
        "transform",
        [
            pytest.param(lambda d: -d, id="neg"),
            pytest.param(lambda d: abs(d), id="abs"),
            pytest.param(lambda d: d**2, id="pow2"),
        ],
    )
    def test_zero_preserving_unary_preserves_diagonal(self, transform):
        y = transform(pt.eye(5))
        _, af = make_fgraph(y)
        assert af.check(y, DIAGONAL)


class TestTriangularElemwiseAssumptions:
    @pytest.mark.parametrize("key", [LOWER_TRIANGULAR, UPPER_TRIANGULAR])
    @pytest.mark.parametrize(
        "build, expected",
        [
            pytest.param(lambda a, b: a + b, True, id="add"),
            pytest.param(lambda a, b: a - b, True, id="sub"),
            pytest.param(lambda a, b: a * b, True, id="mul"),
            pytest.param(lambda a, b: 2.0 * a, True, id="scalar_mul"),
            pytest.param(lambda a, b: -a, True, id="neg"),
            pytest.param(lambda a, b: a**3, True, id="pow_const"),
            pytest.param(lambda a, b: pt.exp(a), False, id="non_zero_preserving_unary"),
        ],
    )
    def test_elemwise_propagation(self, key, build, expected):
        kwarg = {key.name: True}
        x = pt.matrix("x", shape=(5, 5))
        y = pt.matrix("y", shape=(5, 5))
        a = pt.specify_assumptions(x, **kwarg)
        b = pt.specify_assumptions(y, **kwarg)
        z = build(a, b)
        _, af = make_fgraph(z, inputs=[x, y])
        assert af.check(z, key) is expected


class TestSymmetricElemwiseAssumptions:
    @staticmethod
    def _sym(name, shape=(5, 5)):
        x = pt.tensor(name, shape=shape)
        return pt.specify_assumptions(x, symmetric=True), x

    @pytest.mark.parametrize(
        "binop",
        [
            pytest.param(lambda a, b: a + b, id="add"),
            pytest.param(lambda a, b: a - b, id="sub"),
            pytest.param(lambda a, b: a * b, id="mul"),
            pytest.param(lambda a, b: a / b, id="truediv"),
            pytest.param(lambda a, b: a**b, id="pow"),
        ],
    )
    def test_binop_of_symmetric_is_symmetric(self, binop):
        a, x = self._sym("a")
        b, y = self._sym("b")
        z = binop(a, b)
        _, af = make_fgraph(z, inputs=[x, y])
        assert af.check(z, SYMMETRIC)

    @pytest.mark.parametrize(
        "unop",
        [
            pytest.param(lambda a: -a, id="neg"),
            pytest.param(lambda a: pt.exp(a), id="exp"),
            pytest.param(abs, id="abs"),
        ],
    )
    def test_unop_of_symmetric_is_symmetric(self, unop):
        a, x = self._sym("a")
        z = unop(a)
        _, af = make_fgraph(z, inputs=[x])
        assert af.check(z, SYMMETRIC)

    @pytest.mark.parametrize(
        "shape, expected",
        [
            pytest.param((), True, id="0d_scalar"),
            pytest.param((1, 1), True, id="1x1_matrix"),
            pytest.param((5,), False, id="1d_vector"),
            pytest.param((5, 1), False, id="col_vector"),
            pytest.param((1, 5), False, id="row_vector"),
            pytest.param((5, 5), False, id="generic_matrix"),
        ],
    )
    def test_add_symmetric_plus_other(self, shape, expected):
        a, x = self._sym("a")
        y = pt.tensor("y", shape=shape)
        z = a + y
        _, af = make_fgraph(z, inputs=[x, y])
        assert af.check(z, SYMMETRIC) is expected

    def test_batched_symmetric_add(self):
        a, x = self._sym("a", shape=(4, 5, 5))
        b, y = self._sym("b", shape=(4, 5, 5))
        z = a + b
        _, af = make_fgraph(z, inputs=[x, y])
        assert af.check(z, SYMMETRIC)

    def test_scalar_only_elemwise_is_unknown(self):
        s = pt.scalar("s")
        z = s + s
        _, af = make_fgraph(z, inputs=[s])
        assert not af.check(z, SYMMETRIC)


class TestPSDElemwiseAssumptions:
    @staticmethod
    def _psd(name, shape=(5, 5)):
        x = pt.tensor(name, shape=shape)
        return pt.specify_assumptions(x, positive_definite=True), x

    def test_add_pd_plus_pd(self):
        a, x = self._psd("a")
        b, y = self._psd("b")
        z = a + b
        _, af = make_fgraph(z, inputs=[x, y])
        assert af.check(z, POSITIVE_DEFINITE)

    @pytest.mark.parametrize(
        "scale, expected",
        [
            pytest.param(2.5, True, id="positive"),
            pytest.param(-1.0, False, id="negative"),
            pytest.param(0.0, False, id="zero"),
        ],
    )
    def test_scalar_times_pd(self, scale, expected):
        a, x = self._psd("a")
        z = scale * a
        _, af = make_fgraph(z, inputs=[x])
        assert af.check(z, POSITIVE_DEFINITE) is expected

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((), id="0d_scalar"),
            pytest.param((1, 1), id="1x1_matrix"),
            pytest.param((5, 1), id="col_vector"),
            pytest.param((5, 5), id="generic_matrix"),
        ],
    )
    def test_add_pd_plus_other_is_unknown(self, shape):
        a, x = self._psd("a")
        y = pt.tensor("y", shape=shape)
        z = a + y
        _, af = make_fgraph(z, inputs=[x, y])
        assert not af.check(z, POSITIVE_DEFINITE)

    def test_batched_pd_plus_pd(self):
        a, x = self._psd("a", shape=(4, 5, 5))
        b, y = self._psd("b", shape=(4, 5, 5))
        z = a + b
        _, af = make_fgraph(z, inputs=[x, y])
        assert af.check(z, POSITIVE_DEFINITE)

    def test_unrelated_elemwise_is_unknown(self):
        a, x = self._psd("a")
        z = -a
        _, af = make_fgraph(z, inputs=[x])
        assert not af.check(z, POSITIVE_DEFINITE)


def test_blockwise_cholesky_is_lower_triangular():
    x = pt.tensor("x", shape=(5, 3, 3))
    L = pt.linalg.cholesky(x, lower=True)
    _, af = make_fgraph(L, inputs=[x])
    assert af.check(L, LOWER_TRIANGULAR)


def test_blockwise_diagonal_propagation():
    x = pt.tensor("x", shape=(5, 3, 3))
    L = pt.linalg.cholesky(x, lower=True)
    _, af = make_fgraph(L, inputs=[x])
    af.set_user_fact(x, DIAGONAL, FactState.TRUE)
    assert af.check(L, DIAGONAL)


@pytest.mark.parametrize(
    "key", [DIAGONAL, SYMMETRIC, LOWER_TRIANGULAR, UPPER_TRIANGULAR]
)
def test_blockwise_alloc_diag_properties(key):
    v_core = pt.vector("v", shape=(3,))
    d_core = alloc_diag(v_core, offset=0, axis1=0, axis2=1)

    bw = Blockwise(d_core.owner.op, signature="(n)->(n,n)")
    v_batch = pt.matrix("v_batch", shape=(5, 3))
    result = bw(v_batch)

    _, af = make_fgraph(result, inputs=[v_batch])

    assert af.check(result, key)


def test_deep_graph_no_recursion_error():
    x = pt.eye(5)
    for _ in range(2000):
        x = x * 1.0
    _, af = make_fgraph(x)
    assert af.check(x, DIAGONAL)
