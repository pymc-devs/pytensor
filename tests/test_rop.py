"""
WRITE ME

Tests for the R operator / L operator

For the list of op with r op defined, with or without missing test
see this file: doc/library/tensor/basic.txt

For function to automatically test your Rop implementation, look at
the docstring of the functions: check_mat_rop_lop, check_rop_lop,
check_nondiff_rop,
"""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import config, function
from pytensor.gradient import (
    Lop,
    NullTypeGradError,
    Rop,
    grad,
    grad_undefined,
)
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor.math import argmax, dot
from pytensor.tensor.math import max as pt_max
from pytensor.tensor.type import matrix, vector
from tests import unittest_tools as utt


class BreakRop(Op):
    """
    Special Op created to test what happens when you have one op that is not
    differentiable in the computational graph

    @note: Non-differentiable.
    """

    __props__ = ()

    def make_node(self, x):
        return Apply(self, [x], [x.type()])

    def perform(self, node, inp, out_):
        (x,) = inp
        (out,) = out_
        out[0] = x

    def grad(self, inp, grads):
        return [grad_undefined(self, 0, inp[0])]

    def R_op(self, inputs, eval_points):
        return [None]


break_op = BreakRop()


class RopLopChecker:
    """
    Don't perform any test, but provide the function to test the
    Rop to class that inherit from it.
    """

    @staticmethod
    def rtol():
        return 1e-7 if config.floatX == "float64" else 1e-5

    def setup_method(self):
        # Using vectors make things a lot simpler for generating the same
        # computations using scan
        self.x = vector("x")
        self.v = vector("v")
        self.rng = np.random.default_rng(utt.fetch_seed())
        self.in_shape = (5 + self.rng.integers(3),)
        self.mx = matrix("mx")
        self.mv = matrix("mv")
        self.mat_in_shape = (5 + self.rng.integers(3), 5 + self.rng.integers(3))

    def check_nondiff_rop(self, y, x, v):
        """
        If your op is not differentiable(so you can't define Rop)
        test that an error is raised.
        """
        with pytest.raises(ValueError):
            Rop(y, x, v, use_op_rop_implementation=True)

    def check_mat_rop_lop(self, y, out_shape):
        """
        Test the Rop/Lop when input is a matrix and the output is a vector

        :param y: the output variable of the op applied to self.mx
        :param out_shape: Used to generate a random tensor
                          corresponding to the evaluation point of the Rop
                          (i.e. the tensor with which you multiply the
                          Jacobian). It should be a tuple of ints.

        If the Op has more than 1 input, one of them must be mx, while
        others must be shared variables / constants. We will test only
        against the input self.mx, so you must call
        check_mat_rop_lop/check_rop_lop for the other inputs.

        We expect all inputs/outputs have dtype floatX.

        If you want to test an Op with an output matrix, add a sum
        after the Op you want to test.
        """
        vx = np.asarray(
            self.rng.uniform(size=self.mat_in_shape), pytensor.config.floatX
        )
        vv = np.asarray(
            self.rng.uniform(size=self.mat_in_shape), pytensor.config.floatX
        )
        yv = Rop(y, self.mx, self.mv, use_op_rop_implementation=True)
        rop_f = function([self.mx, self.mv], yv, on_unused_input="ignore")

        yv_through_lop = Rop(y, self.mx, self.mv, use_op_rop_implementation=False)
        rop_through_lop_f = function(
            [self.mx, self.mv], yv_through_lop, on_unused_input="ignore"
        )

        sy, _ = pytensor.scan(
            lambda i, y, x, v: (grad(y[i], x) * v).sum(),
            sequences=pt.arange(y.shape[0]),
            non_sequences=[y, self.mx, self.mv],
        )
        scan_f = function([self.mx, self.mv], sy, on_unused_input="ignore")

        v_ref = scan_f(vx, vv)
        np.testing.assert_allclose(rop_f(vx, vv), v_ref)
        np.testing.assert_allclose(rop_through_lop_f(vx, vv), v_ref)

        self.check_nondiff_rop(
            pytensor.clone_replace(y, replace={self.mx: break_op(self.mx)}),
            self.mx,
            self.mv,
        )

        vv = np.asarray(self.rng.uniform(size=out_shape), pytensor.config.floatX)
        yv = Lop(y, self.mx, self.v)
        lop_f = function([self.mx, self.v], yv)

        sy = grad((self.v * y).sum(), self.mx)
        scan_f = function([self.mx, self.v], sy)

        v = lop_f(vx, vv)
        v_ref = scan_f(vx, vv)
        np.testing.assert_allclose(v, v_ref)

    def check_rop_lop(self, y, out_shape, check_nondiff_rop: bool = True):
        """
        As check_mat_rop_lop, except the input is self.x which is a
        vector. The output is still a vector.
        """
        rtol = self.rtol()

        # TEST ROP
        vx = np.asarray(self.rng.uniform(size=self.in_shape), pytensor.config.floatX)
        vv = np.asarray(self.rng.uniform(size=self.in_shape), pytensor.config.floatX)

        yv = Rop(y, self.x, self.v, use_op_rop_implementation=True)
        rop_f = function([self.x, self.v], yv, on_unused_input="ignore")

        yv_through_lop = Rop(y, self.x, self.v, use_op_rop_implementation=False)
        rop_through_lop_f = function(
            [self.x, self.v], yv_through_lop, on_unused_input="ignore"
        )

        J, _ = pytensor.scan(
            lambda i, y, x: grad(y[i], x),
            sequences=pt.arange(y.shape[0]),
            non_sequences=[y, self.x],
        )
        sy = dot(J, self.v)
        scan_f = function([self.x, self.v], sy, on_unused_input="ignore")

        v_ref = scan_f(vx, vv)
        np.testing.assert_allclose(rop_f(vx, vv), v_ref, rtol=rtol)
        np.testing.assert_allclose(rop_through_lop_f(vx, vv), v_ref, rtol=rtol)

        if check_nondiff_rop:
            self.check_nondiff_rop(
                pytensor.clone_replace(y, replace={self.x: break_op(self.x)}),
                self.x,
                self.v,
            )

        vx = np.asarray(self.rng.uniform(size=self.in_shape), pytensor.config.floatX)
        vv = np.asarray(self.rng.uniform(size=out_shape), pytensor.config.floatX)

        yv = Lop(y, self.x, self.v)
        lop_f = function([self.x, self.v], yv, on_unused_input="ignore")
        J, _ = pytensor.scan(
            lambda i, y, x: grad(y[i], x),
            sequences=pt.arange(y.shape[0]),
            non_sequences=[y, self.x],
        )
        sy = dot(self.v, J)
        scan_f = function([self.x, self.v], sy)

        v = lop_f(vx, vv)
        v_ref = scan_f(vx, vv)
        np.testing.assert_allclose(v, v_ref, rtol=rtol)


class TestRopLop(RopLopChecker):
    def test_max(self):
        self.check_mat_rop_lop(pt_max(self.mx, axis=0), (self.mat_in_shape[1],))
        self.check_mat_rop_lop(pt_max(self.mx, axis=1), (self.mat_in_shape[0],))

    def test_argmax(self):
        self.check_nondiff_rop(argmax(self.mx, axis=1), self.mx, self.mv)

    def test_subtensor(self):
        self.check_rop_lop(self.x[:4], (4,))

    def test_incsubtensor1(self):
        tv = np.asarray(self.rng.uniform(size=(3,)), pytensor.config.floatX)
        t = pytensor.shared(tv)
        out = pytensor.tensor.subtensor.inc_subtensor(self.x[:3], t)
        self.check_rop_lop(out, self.in_shape)

    def test_incsubtensor2(self):
        tv = np.asarray(self.rng.uniform(size=(10,)), pytensor.config.floatX)
        t = pytensor.shared(tv)
        out = pytensor.tensor.subtensor.inc_subtensor(t[:4], self.x[:4])
        self.check_rop_lop(out, (10,))

    def test_setsubtensor1(self):
        tv = np.asarray(self.rng.uniform(size=(3,)), pytensor.config.floatX)
        t = pytensor.shared(tv)
        out = pytensor.tensor.subtensor.set_subtensor(self.x[:3], t)
        self.check_rop_lop(out, self.in_shape)

    def test_print(self):
        out = pytensor.printing.Print("x", attrs=("shape",))(self.x)
        self.check_rop_lop(out, self.in_shape)

    def test_setsubtensor2(self):
        tv = np.asarray(self.rng.uniform(size=(10,)), pytensor.config.floatX)
        t = pytensor.shared(tv)
        out = pytensor.tensor.subtensor.set_subtensor(t[:4], self.x[:4])
        self.check_rop_lop(out, (10,))

    def test_dimshuffle(self):
        # I need the sum, because the setup expects the output to be a
        # vector
        self.check_rop_lop(self.x[:4].dimshuffle("x", 0).sum(axis=0), (4,))

    def test_join(self):
        tv = np.asarray(self.rng.uniform(size=(10,)), pytensor.config.floatX)
        t = pytensor.shared(tv)
        out = pt.join(0, self.x, t)
        self.check_rop_lop(out, (self.in_shape[0] + 10,))

    def test_dot(self):
        insh = self.in_shape[0]
        vW = np.asarray(self.rng.uniform(size=(insh, insh)), pytensor.config.floatX)
        W = pytensor.shared(vW)
        # check_nondiff_rop reveals an error in how legacy Rop handles non-differentiable paths
        # See: test_Rop_partially_differentiable_paths
        self.check_rop_lop(dot(self.x, W), self.in_shape, check_nondiff_rop=False)

    def test_elemwise0(self):
        # check_nondiff_rop reveals an error in how legacy Rop handles non-differentiable paths
        # See: test_Rop_partially_differentiable_paths
        self.check_rop_lop((self.x + 1) ** 2, self.in_shape, check_nondiff_rop=False)

    def test_elemwise1(self):
        self.check_rop_lop(self.x + pt.cast(self.x, "int32"), self.in_shape)

    def test_flatten(self):
        self.check_mat_rop_lop(
            self.mx.flatten(), (self.mat_in_shape[0] * self.mat_in_shape[1],)
        )

    def test_sum(self):
        self.check_mat_rop_lop(self.mx.sum(axis=1), (self.mat_in_shape[0],))

    def test_softmax(self):
        self.check_rop_lop(
            pytensor.tensor.special.softmax(self.x, axis=-1), self.in_shape
        )

    def test_alloc(self):
        # Alloc of the sum of x into a vector
        out1d = pt.alloc(self.x.sum(), self.in_shape[0])
        self.check_rop_lop(out1d, self.in_shape[0])

        # Alloc of x into a 3-D tensor, flattened
        out3d = pt.alloc(
            self.x, self.mat_in_shape[0], self.mat_in_shape[1], self.in_shape[0]
        )
        self.check_rop_lop(
            out3d.flatten(),
            self.mat_in_shape[0] * self.mat_in_shape[1] * self.in_shape[0],
        )

    @pytest.mark.parametrize("use_op_rop_implementation", [True, False])
    def test_invalid_input(self, use_op_rop_implementation):
        with pytest.raises(ValueError):
            Rop(
                0.0,
                [matrix()],
                [vector()],
                use_op_rop_implementation=use_op_rop_implementation,
            )

    @pytest.mark.parametrize("use_op_rop_implementation", [True, False])
    def test_multiple_outputs(self, use_op_rop_implementation):
        m = matrix("m")
        v = vector("v")
        m_ = matrix("m_")
        v_ = vector("v_")

        mval = self.rng.uniform(size=(3, 7)).astype(pytensor.config.floatX)
        vval = self.rng.uniform(size=(7,)).astype(pytensor.config.floatX)
        m_val = self.rng.uniform(size=(3, 7)).astype(pytensor.config.floatX)
        v_val = self.rng.uniform(size=(7,)).astype(pytensor.config.floatX)

        rop_out1 = Rop(
            [m, v, m + v],
            [m, v],
            [m_, v_],
            use_op_rop_implementation=use_op_rop_implementation,
        )
        assert isinstance(rop_out1, list)
        assert len(rop_out1) == 3
        rop_out2 = Rop(
            (m, v, m + v),
            [m, v],
            [m_, v_],
            use_op_rop_implementation=use_op_rop_implementation,
        )
        assert isinstance(rop_out2, tuple)
        assert len(rop_out2) == 3

        all_outs = []
        for o in rop_out1, rop_out2:
            all_outs.extend(o)
        f = pytensor.function([m, v, m_, v_], all_outs)
        f(mval, vval, m_val, v_val)

    @pytest.mark.parametrize(
        "use_op_rop_implementation",
        [pytest.param(True, marks=pytest.mark.xfail()), False],
    )
    def test_Rop_partially_differentiable_paths(self, use_op_rop_implementation):
        # This test refers to a bug reported by Jeremiah Lowin on 18th Oct
        # 2013. The bug consists when through a dot operation there is only
        # one differentiable path (i.e. there is no gradient wrt to one of
        # the inputs).
        x = pt.arange(20.0).reshape([1, 20])
        v = pytensor.shared(np.ones([20]), name="v")
        d = dot(x, v).sum()

        Rop(
            grad(d, v),
            v,
            v,
            use_op_rop_implementation=use_op_rop_implementation,
            # 2025: This is a tricky case, the gradient of the gradient does not depend on v
            # although v still exists in the graph inside a `Second` operator.
            # The original test was checking that Rop wouldn't raise an error, but Lop does.
            # Since the correct behavior is ambiguous, I let both implementations off the hook.
            disconnected_outputs="raise" if use_op_rop_implementation else "ignore",
        )

        # 2025: Here is an unambiguous test for the original commented issue:
        x = pt.matrix("x")
        y = pt.matrix("y")
        out = dot(x, break_op(y)).sum()
        # Should not raise an error
        Rop(
            out,
            [x],
            [x.type()],
            use_op_rop_implementation=use_op_rop_implementation,
            disconnected_outputs="raise",
        )

        # More extensive testing shows that the legacy Rop implementation FAILS to raise when
        # the cost is linked through strictly non-differentiable paths.
        # This is not Dot specific, we would observe the same with any operation where the gradient
        # with respect to one of the inputs does not depend on the original input (such as `mul`, `add`, ...)
        out = dot(break_op(x), y).sum()
        with pytest.raises((ValueError, NullTypeGradError)):
            Rop(
                out,
                [x],
                [x.type()],
                use_op_rop_implementation=use_op_rop_implementation,
                disconnected_outputs="raise",
            )

        # Only when both paths are non-differentiable is an error correctly raised again.
        out = dot(break_op(x), break_op(y)).sum()
        with pytest.raises((ValueError, NullTypeGradError)):
            Rop(
                out,
                [x],
                [x.type()],
                use_op_rop_implementation=use_op_rop_implementation,
                disconnected_outputs="raise",
            )
