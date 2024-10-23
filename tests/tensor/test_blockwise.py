import re
from itertools import product

import numpy as np
import pytest
import scipy.linalg

import pytensor
from pytensor import In, config, function
from pytensor.compile import get_default_mode, get_mode
from pytensor.gradient import grad
from pytensor.graph import Apply, Op
from pytensor.graph.replace import vectorize_node
from pytensor.raise_op import assert_op
from pytensor.tensor import diagonal, log, tensor
from pytensor.tensor.blockwise import Blockwise, vectorize_node_fallback
from pytensor.tensor.nlinalg import MatrixInverse
from pytensor.tensor.rewriting.blas import specialize_matmul_to_batched_dot
from pytensor.tensor.slinalg import (
    Cholesky,
    Solve,
    SolveBase,
    cho_solve,
    cholesky,
    solve,
    solve_triangular,
)
from pytensor.tensor.utils import _parse_gufunc_signature


def test_perform_method_per_node():
    """Confirm that Blockwise uses one perform method per node.

    This is important if the perform method requires node information (such as dtypes)
    """

    class NodeDependentPerformOp(Op):
        def make_node(self, x):
            return Apply(self, [x], [x.type()])

        def perform(self, node, inputs, outputs):
            [x] = inputs
            if node.inputs[0].type.dtype.startswith("float"):
                y = x + 1
            else:
                y = x - 1
            outputs[0][0] = y

    blockwise_op = Blockwise(core_op=NodeDependentPerformOp(), signature="()->()")
    x = tensor("x", shape=(3,), dtype="float32")
    y = tensor("y", shape=(3,), dtype="int32")

    out_x = blockwise_op(x)
    out_y = blockwise_op(y)
    fn = pytensor.function([x, y], [out_x, out_y])
    [op1, op2] = [node.op for node in fn.maker.fgraph.apply_nodes]
    # Confirm both nodes have the same Op
    assert op1 is blockwise_op
    assert op1 is op2

    res_out_x, res_out_y = fn(np.zeros(3, dtype="float32"), np.zeros(3, dtype="int32"))
    np.testing.assert_array_equal(res_out_x, np.ones(3, dtype="float32"))
    np.testing.assert_array_equal(res_out_y, -np.ones(3, dtype="int32"))


def test_vectorize_blockwise():
    mat = tensor(shape=(None, None))
    tns = tensor(shape=(None, None, None))

    # Something that falls back to Blockwise
    node = MatrixInverse()(mat).owner
    vect_node = vectorize_node(node, tns)
    assert isinstance(vect_node.op, Blockwise) and isinstance(
        vect_node.op.core_op, MatrixInverse
    )
    assert vect_node.op.signature == ("(m,m)->(m,m)")
    assert vect_node.inputs[0] is tns

    # Useless blockwise
    tns4 = tensor(shape=(5, None, None, None))
    new_vect_node = vectorize_node(vect_node, tns4)
    assert new_vect_node.op is vect_node.op
    assert isinstance(new_vect_node.op, Blockwise) and isinstance(
        new_vect_node.op.core_op, MatrixInverse
    )
    assert new_vect_node.inputs[0] is tns4


def test_vectorize_node_fallback_unsupported_type():
    x = tensor("x", shape=(2, 6))
    node = x[:, [0, 2, 4]].owner

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "Cannot vectorize node AdvancedSubtensor(x, MakeSlice.0, [0 2 4]) with input MakeSlice.0 of type slice"
        ),
    ):
        vectorize_node_fallback(node.op, node, node.inputs)


def check_blockwise_runtime_broadcasting(mode):
    a = tensor("a", shape=(None, 3, 5))
    b = tensor("b", shape=(None, 5, 3))

    out = a @ b
    fn = function(
        [a, b],
        out,
        mode=get_mode(mode).excluding(specialize_matmul_to_batched_dot.__name__),
    )
    assert isinstance(fn.maker.fgraph.outputs[0].owner.op, Blockwise)

    for valid_test_values in [
        (
            np.ones((2, 3, 5)).astype(config.floatX),
            np.ones((2, 5, 3)).astype(config.floatX),
        ),
        (
            np.ones((1, 3, 5)).astype(config.floatX),
            np.ones((1, 5, 3)).astype(config.floatX),
        ),
    ]:
        batch_dim = valid_test_values[0].shape[0]
        np.testing.assert_allclose(
            fn(*valid_test_values), np.full((batch_dim, 3, 3), 5.0)
        )

    for invalid_test_values in [
        (
            np.ones((1, 3, 5)).astype(config.floatX),
            np.ones((2, 5, 3)).astype(config.floatX),
        ),
        (
            np.ones((2, 3, 5)).astype(config.floatX),
            np.ones((1, 5, 3)).astype(config.floatX),
        ),
    ]:
        with pytest.raises(ValueError, match="Runtime broadcasting not allowed"):
            fn(*invalid_test_values)

    invalid_test_values = (
        np.ones((2, 3, 5)).astype(config.floatX),
        np.ones((3, 5, 3)).astype(config.floatX),
    )
    # Error message is backend specific
    with pytest.raises(ValueError):
        fn(*invalid_test_values)


@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_runtime_broadcast(mode):
    check_blockwise_runtime_broadcasting(mode)


class MyTestOp(Op):
    def make_node(self, *inputs):
        return Apply(self, inputs, [i.type() for i in inputs])

    def perform(self, *args, **kwargs):
        raise NotImplementedError("Test Op should not be present in final graph")


test_op = MyTestOp()


def test_vectorize_node_default_signature():
    vec = tensor(shape=(None,))
    mat = tensor(shape=(5, None))
    node = test_op.make_node(vec, mat)

    vect_node = vectorize_node(node, mat, mat)
    assert isinstance(vect_node.op, Blockwise) and isinstance(
        vect_node.op.core_op, MyTestOp
    )
    assert vect_node.op.signature == ("(i00),(i10,i11)->(o00),(o10,o11)")

    with pytest.raises(
        ValueError, match="Signature not provided nor found in core_op MyTestOp"
    ):
        Blockwise(test_op)

    vect_node = Blockwise(test_op, signature="(m),(n)->(m),(n)").make_node(vec, mat)
    assert vect_node.outputs[0].type.shape == (
        5,
        None,
    )
    assert vect_node.outputs[0].type.shape == (
        5,
        None,
    )


def test_blockwise_shape():
    # Single output
    inp = tensor(shape=(5, None, None))
    inp_test = np.zeros((5, 4, 3), dtype=config.floatX)

    # Shape can be inferred from inputs
    op = Blockwise(test_op, signature="(m, n) -> (n, m)")
    out = op(inp)
    assert out.type.shape == (5, None, None)

    shape_fn = pytensor.function([inp], out.shape)
    assert not any(
        isinstance(getattr(n.op, "core_op", n.op), MyTestOp)
        for n in shape_fn.maker.fgraph.apply_nodes
    )
    assert tuple(shape_fn(inp_test)) == (5, 3, 4)

    # Shape can only be partially inferred from inputs
    op = Blockwise(test_op, signature="(m, n) -> (m, k)")
    out = op(inp)
    assert out.type.shape == (5, None, None)

    shape_fn = pytensor.function([inp], out.shape)
    assert any(
        isinstance(getattr(n.op, "core_op", n.op), MyTestOp)
        for n in shape_fn.maker.fgraph.apply_nodes
    )

    shape_fn = pytensor.function([inp], out.shape[:-1])
    assert not any(
        isinstance(getattr(n.op, "core_op", n.op), MyTestOp)
        for n in shape_fn.maker.fgraph.apply_nodes
    )
    assert tuple(shape_fn(inp_test)) == (5, 4)

    # Mutiple outputs
    inp1 = tensor(shape=(7, 1, None, None))
    inp2 = tensor(shape=(1, 5, None, None))
    inp1_test = np.zeros((7, 1, 4, 3), dtype=config.floatX)
    inp2_test = np.zeros((1, 5, 4, 3), dtype=config.floatX)

    op = Blockwise(test_op, signature="(m, n), (m, n) -> (n, m), (m, k)")
    outs = op(inp1, inp2)
    assert outs[0].type.shape == (7, 5, None, None)
    assert outs[1].type.shape == (7, 5, None, None)

    shape_fn = pytensor.function([inp1, inp2], [out.shape for out in outs])
    assert any(
        isinstance(getattr(n.op, "core_op", n.op), MyTestOp)
        for n in shape_fn.maker.fgraph.apply_nodes
    )

    shape_fn = pytensor.function([inp1, inp2], outs[0].shape)
    assert not any(
        isinstance(getattr(n.op, "core_op", n.op), MyTestOp)
        for n in shape_fn.maker.fgraph.apply_nodes
    )
    assert tuple(shape_fn(inp1_test, inp2_test)) == (7, 5, 3, 4)

    shape_fn = pytensor.function([inp1, inp2], [outs[0].shape, outs[1].shape[:-1]])
    assert not any(
        isinstance(getattr(n.op, "core_op", n.op), MyTestOp)
        for n in shape_fn.maker.fgraph.apply_nodes
    )
    assert tuple(shape_fn(inp1_test, inp2_test)[0]) == (7, 5, 3, 4)
    assert tuple(shape_fn(inp1_test, inp2_test)[1]) == (7, 5, 4)


class BlockwiseOpTester:
    """Base class to test Blockwise works for specific Ops"""

    core_op = None
    signature = None
    batcheable_axes = None

    @classmethod
    def setup_class(cls):
        seed = sum(map(ord, str(cls.core_op)))
        cls.rng = np.random.default_rng(seed)
        cls.params_sig, cls.outputs_sig = _parse_gufunc_signature(cls.signature)
        if cls.batcheable_axes is None:
            cls.batcheable_axes = list(range(len(cls.params_sig)))
        batch_shapes = [(), (1,), (5,), (1, 1), (1, 5), (3, 1), (3, 5)]
        cls.test_batch_shapes = list(
            product(batch_shapes, repeat=len(cls.batcheable_axes))
        )
        cls.block_op = Blockwise(core_op=cls.core_op, signature=cls.signature)

    @staticmethod
    def parse_shape(shape: tuple[str | int, ...]) -> tuple[int, ...]:
        """
        Convert (5, "m", "n") -> (5, 7, 11)
        """
        mapping = {"m": 7, "n": 11, "k": 19}
        return tuple(mapping.get(p, p) for p in shape)

    def create_testvals(self, shape):
        return self.rng.normal(size=self.parse_shape(shape)).astype(config.floatX)

    def create_batched_inputs(self, batch_idx: int | None = None):
        for batch_shapes in self.test_batch_shapes:
            vec_inputs = []
            vec_inputs_testvals = []
            for idx, (batch_shape, param_sig) in enumerate(
                zip(batch_shapes, self.params_sig)
            ):
                if batch_idx is not None and idx != batch_idx:
                    # Skip out combinations in which other inputs are batched
                    if batch_shape != ():
                        break
                vec_inputs.append(tensor(shape=batch_shape + (None,) * len(param_sig)))
                vec_inputs_testvals.append(
                    self.create_testvals(shape=batch_shape + param_sig)
                )
            else:  # no-break
                yield vec_inputs, vec_inputs_testvals

    def test_perform(self):
        base_inputs = [
            tensor(shape=(None,) * len(param_sig)) for param_sig in self.params_sig
        ]
        core_func = pytensor.function(base_inputs, self.core_op(*base_inputs))
        np_func = np.vectorize(core_func, signature=self.signature)

        for vec_inputs, vec_inputs_testvals in self.create_batched_inputs():
            pt_func = pytensor.function(vec_inputs, self.block_op(*vec_inputs))
            if len(self.outputs_sig) != 1:
                raise NotImplementedError("Did not implement test for multi-output Ops")
            np.testing.assert_allclose(
                pt_func(*vec_inputs_testvals),
                np_func(*vec_inputs_testvals),
                rtol=1e-7 if config.floatX == "float64" else 1e-5,
                atol=1e-7 if config.floatX == "float64" else 1e-5,
            )

    def test_grad(self):
        base_inputs = [
            tensor(shape=(None,) * len(param_sig)) for param_sig in self.params_sig
        ]
        out = self.core_op(*base_inputs).sum()
        # Create separate numpy vectorized functions for each input
        np_funcs = []
        for i, inp in enumerate(base_inputs):
            core_grad_func = pytensor.function(base_inputs, grad(out, wrt=inp))
            params_sig = self.signature.split("->")[0]
            param_sig = f"({','.join(self.params_sig[i])})"
            grad_sig = f"{params_sig}->{param_sig}"
            np_func = np.vectorize(core_grad_func, signature=grad_sig)
            np_funcs.append(np_func)

        # We test gradient wrt to one batched input at a time
        for test_input_idx in range(len(base_inputs)):
            for vec_inputs, vec_inputs_testvals in self.create_batched_inputs(
                batch_idx=test_input_idx
            ):
                out = self.block_op(*vec_inputs).sum()
                pt_func = pytensor.function(
                    vec_inputs, grad(out, wrt=vec_inputs[test_input_idx])
                )
                pt_out = pt_func(*vec_inputs_testvals)
                np_out = np_funcs[test_input_idx](*vec_inputs_testvals)
                np.testing.assert_allclose(
                    pt_out,
                    np_out,
                    rtol=1e-7 if config.floatX == "float64" else 1e-5,
                    atol=1e-6 if config.floatX == "float64" else 1e-4,
                )


class MatrixOpBlockwiseTester(BlockwiseOpTester):
    def create_testvals(self, shape):
        # Return a posdef matrix
        X = super().create_testvals(shape)
        return np.einsum("...ij,...kj->...ik", X, X)


class TestCholesky(MatrixOpBlockwiseTester):
    core_op = Cholesky(lower=True)
    signature = "(m, m) -> (m, m)"


class TestMatrixInverse(MatrixOpBlockwiseTester):
    core_op = MatrixInverse()
    signature = "(m, m) -> (m, m)"


class TestSolveVector(BlockwiseOpTester):
    core_op = Solve(lower=True, b_ndim=1)
    signature = "(m, m),(m) -> (m)"


class TestSolveMatrix(BlockwiseOpTester):
    core_op = Solve(lower=True, b_ndim=2)
    signature = "(m, m),(m, n) -> (m, n)"


@pytest.mark.parametrize(
    "mu_batch_shape", [(), (1000,), (4, 1000)], ids=lambda arg: f"mu:{arg}"
)
@pytest.mark.parametrize(
    "cov_batch_shape", [(), (1000,), (4, 1000)], ids=lambda arg: f"cov:{arg}"
)
def test_batched_mvnormal_logp_and_dlogp(mu_batch_shape, cov_batch_shape, benchmark):
    rng = np.random.default_rng(sum(map(ord, "batched_mvnormal")))

    value_batch_shape = mu_batch_shape
    if len(cov_batch_shape) > len(mu_batch_shape):
        value_batch_shape = cov_batch_shape

    value = tensor("value", shape=(*value_batch_shape, 10))
    mu = tensor("mu", shape=(*mu_batch_shape, 10))
    cov = tensor("cov", shape=(*cov_batch_shape, 10, 10))

    test_values = [
        rng.normal(size=value.type.shape),
        rng.normal(size=mu.type.shape),
        np.eye(cov.type.shape[-1]) * np.abs(rng.normal(size=cov.type.shape)),
    ]

    chol_cov = cholesky(cov, lower=True, on_error="raise")
    delta_trans = solve_triangular(chol_cov, value - mu, b_ndim=1)
    quaddist = (delta_trans**2).sum(axis=-1)
    diag = diagonal(chol_cov, axis1=-2, axis2=-1)
    logdet = log(diag).sum(axis=-1)
    k = value.shape[-1]
    norm = -0.5 * k * (np.log(2 * np.pi))

    logp = norm - 0.5 * quaddist - logdet
    dlogp = grad(logp.sum(), wrt=[value, mu, cov])

    fn = pytensor.function([value, mu, cov], [logp, *dlogp])
    benchmark(fn, *test_values)


def test_cop_with_params():
    matrix_assert = Blockwise(core_op=assert_op, signature="(x1,x2),()->(x1,x2)")

    x = tensor("x", shape=(5, None, None), dtype="float64")
    x_shape = matrix_assert(x, (x >= 0).all())

    fn = pytensor.function([x], x_shape)
    [fn_out] = fn.maker.fgraph.outputs
    assert fn_out.owner.op == matrix_assert, "Blockwise should be in final graph"

    np.testing.assert_allclose(
        fn(np.zeros((5, 3, 2))),
        np.zeros((5, 3, 2)),
    )

    with pytest.raises(AssertionError):
        fn(np.zeros((5, 3, 2)) - 1)


@pytest.mark.skipif(
    config.mode == "FAST_COMPILE",
    reason="inplace rewrites disabled when mode is FAST_COMPILE",
)
class TestInplace:
    @pytest.mark.parametrize("is_batched", (False, True))
    def test_cholesky(self, is_batched):
        X = tensor("X", shape=(5, None, None) if is_batched else (None, None))
        L = cholesky(X, lower=True)
        f = function([In(X, mutable=True)], L)

        assert not L.owner.op.core_op.destroy_map

        if is_batched:
            [cholesky_op] = [
                node.op.core_op
                for node in f.maker.fgraph.apply_nodes
                if isinstance(node.op, Blockwise)
                and isinstance(node.op.core_op, Cholesky)
            ]
        else:
            [cholesky_op] = [
                node.op
                for node in f.maker.fgraph.apply_nodes
                if isinstance(node.op, Cholesky)
            ]
        assert cholesky_op.destroy_map == {0: [0]}

        rng = np.random.default_rng(441 + is_batched)
        X_val = rng.normal(size=(10, 10)).astype(config.floatX)
        X_val_in = X_val @ X_val.T
        if is_batched:
            X_val_in = np.broadcast_to(X_val_in, (5, *X_val_in.shape)).copy()
        X_val_in_copy = X_val_in.copy()

        f(X_val_in)

        np.testing.assert_allclose(
            X_val_in,
            np.linalg.cholesky(X_val_in_copy),
            atol=1e-5 if config.floatX == "float32" else 0,
        )

    @pytest.mark.parametrize("batched_A", (False, True))
    @pytest.mark.parametrize("batched_b", (False, True))
    @pytest.mark.parametrize("solve_fn", (solve, solve_triangular, cho_solve))
    def test_solve(self, solve_fn, batched_A, batched_b):
        A = tensor("A", shape=(5, 3, 3) if batched_A else (3, 3))
        b = tensor("b", shape=(5, 3) if batched_b else (3,))
        if solve_fn == cho_solve:
            # Special signature for cho_solve
            x = solve_fn((A, True), b, b_ndim=1)
        else:
            x = solve_fn(A, b, b_ndim=1)

        mode = get_default_mode().excluding("batched_vector_b_solve_to_matrix_b_solve")
        fn = function([In(A, mutable=True), In(b, mutable=True)], x, mode=mode)

        op = fn.maker.fgraph.outputs[0].owner.op
        if batched_A or batched_b:
            assert isinstance(op, Blockwise) and isinstance(op.core_op, SolveBase)
            if batched_A and not batched_b:
                if solve_fn == solve:
                    assert op.destroy_map == {0: [0]}
                else:
                    # SolveTriangular does not destroy A
                    assert op.destroy_map == {}
            else:
                assert op.destroy_map == {0: [1]}
        else:
            assert isinstance(op, SolveBase)
            assert op.destroy_map == {0: [1]}

        # We test with an F_CONTIGUOUS (core) A as only that will be destroyed by scipy
        rng = np.random.default_rng(
            487 + batched_A + 2 * batched_b + sum(map(ord, solve_fn.__name__))
        )
        A_val = np.swapaxes(rng.normal(size=A.type.shape).astype(A.type.dtype), -1, -2)
        b_val = np.random.normal(size=b.type.shape).astype(b.type.dtype)
        A_val_copy = A_val.copy()
        b_val_copy = b_val.copy()
        out = fn(A_val, b_val)

        if solve_fn == cho_solve:

            def core_scipy_fn(A, b):
                return scipy.linalg.cho_solve((A, True), b)

        else:
            core_scipy_fn = getattr(scipy.linalg, solve_fn.__name__)
        expected_out = np.vectorize(core_scipy_fn, signature="(m,m),(m)->(m)")(
            A_val_copy, b_val_copy
        )
        np.testing.assert_allclose(
            out, expected_out, atol=1e-6 if config.floatX == "float32" else 0
        )

        # Confirm input was destroyed
        assert (A_val == A_val_copy).all() == (op.destroy_map.get(0, None) != [0])
        assert (b_val == b_val_copy).all() == (op.destroy_map.get(0, None) != [1])
