from itertools import product
from typing import Optional, Tuple, Union

import numpy as np
import pytest

import pytensor
from pytensor import config
from pytensor.gradient import grad
from pytensor.graph import Apply, Op
from pytensor.graph.replace import vectorize_node
from pytensor.tensor import tensor
from pytensor.tensor.blockwise import Blockwise, _parse_gufunc_signature
from pytensor.tensor.nlinalg import MatrixInverse
from pytensor.tensor.slinalg import Cholesky, Solve


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


class TestOp(Op):
    def make_node(self, *inputs):
        return Apply(self, inputs, [i.type() for i in inputs])

    def perform(self, *args, **kwargs):
        raise NotImplementedError("Test Op should not be present in final graph")


test_op = TestOp()


def test_vectorize_node_default_signature():
    vec = tensor(shape=(None,))
    mat = tensor(shape=(5, None))
    node = test_op.make_node(vec, mat)

    vect_node = vectorize_node(node, mat, mat)
    assert isinstance(vect_node.op, Blockwise) and isinstance(
        vect_node.op.core_op, TestOp
    )
    assert vect_node.op.signature == ("(i00),(i10,i11)->(o00),(o10,o11)")

    with pytest.raises(
        ValueError, match="Signature not provided nor found in core_op TestOp"
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
        isinstance(getattr(n.op, "core_op", n.op), TestOp)
        for n in shape_fn.maker.fgraph.apply_nodes
    )
    assert tuple(shape_fn(inp_test)) == (5, 3, 4)

    # Shape can only be partially inferred from inputs
    op = Blockwise(test_op, signature="(m, n) -> (m, k)")
    out = op(inp)
    assert out.type.shape == (5, None, None)

    shape_fn = pytensor.function([inp], out.shape)
    assert any(
        isinstance(getattr(n.op, "core_op", n.op), TestOp)
        for n in shape_fn.maker.fgraph.apply_nodes
    )

    shape_fn = pytensor.function([inp], out.shape[:-1])
    assert not any(
        isinstance(getattr(n.op, "core_op", n.op), TestOp)
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
        isinstance(getattr(n.op, "core_op", n.op), TestOp)
        for n in shape_fn.maker.fgraph.apply_nodes
    )

    shape_fn = pytensor.function([inp1, inp2], outs[0].shape)
    assert not any(
        isinstance(getattr(n.op, "core_op", n.op), TestOp)
        for n in shape_fn.maker.fgraph.apply_nodes
    )
    assert tuple(shape_fn(inp1_test, inp2_test)) == (7, 5, 3, 4)

    shape_fn = pytensor.function([inp1, inp2], [outs[0].shape, outs[1].shape[:-1]])
    assert not any(
        isinstance(getattr(n.op, "core_op", n.op), TestOp)
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
    def parse_shape(shape: Tuple[Union[str, int], ...]) -> Tuple[int, ...]:
        """
        Convert (5, "m", "n") -> (5, 7, 11)
        """
        mapping = {"m": 7, "n": 11, "k": 19}
        return tuple(mapping.get(p, p) for p in shape)

    def create_testvals(self, shape):
        return self.rng.normal(size=self.parse_shape(shape)).astype(config.floatX)

    def create_batched_inputs(self, batch_idx: Optional[int] = None):
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
                    atol=1e-6 if config.floatX == "float64" else 1e-5,
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
