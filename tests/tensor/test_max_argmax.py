import builtins

import numpy as np
import pytest

from pytensor.compile.function import function
from pytensor.compile.mode import get_default_mode
from pytensor.compile.sharedvalue import shared
from pytensor.configdefaults import config
from pytensor.gradient import grad, numeric_grad
from pytensor.graph.replace import vectorize_node
from pytensor.tensor.basic import (
    as_tensor_variable,
    constant,
    get_underlying_scalar_constant_value,
)
from pytensor.tensor.math import (
    Argmax,
    TensorMax,
    argmax,
    argmin,
    max,
    max_and_argmax,
    min,
)
from pytensor.tensor.type import (
    matrix,
    tensor,
)
from pytensor.tensor.type_other import NoneConst
from tests import unittest_tools as utt
from tests.tensor.utils import (
    eval_outputs,
    random,
)


if config.mode == "FAST_COMPILE":
    mode_opt = "FAST_RUN"
else:
    mode_opt = get_default_mode()


class TestMaxAndArgmax:
    def setup_method(self):
        TensorMax.debug = 0

    def test_basic(self):
        # dbt: for some reason, Argmax does not work when I pass: n = as_tensor_variable(5.0)
        n = as_tensor_variable([5.0])
        v, i = eval_outputs(max_and_argmax(n))
        assert v == 5.0
        assert i == 0
        assert i.dtype == "int64"
        v = eval_outputs(max_and_argmax(n)[0].shape)
        assert len(v) == 0
        v = eval_outputs(max_and_argmax(n)[1].shape)
        assert len(v) == 0

    def test_basic_1(self):
        n = as_tensor_variable([1, 2, 3, 2, -6])
        v, i = eval_outputs(max_and_argmax(n))
        assert v == 3
        assert i == 2
        assert i.dtype == "int64"
        v = eval_outputs(max_and_argmax(n)[0].shape)
        assert len(v) == 0

    @pytest.mark.parametrize(
        "axis,np_axis",
        [
            (-1, -1),
            (0, 0),
            (1, 1),
            (None, None),
            ([0, 1], None),
            ([1, 0], None),
            (NoneConst.clone(), None),
            (constant(0), 0),
        ],
    )
    def test_basic_2(self, axis, np_axis):
        data = random(2, 3)
        n = as_tensor_variable(data)
        # Test shape propagates (static & eval)
        vt, it = max_and_argmax(n, axis)
        np_max, np_argm = np.max(data, np_axis), np.argmax(data, np_axis)
        assert vt.type.shape == np_max.shape
        assert it.type.shape == np_argm.shape
        v_shape, i_shape = eval_outputs([vt.shape, it.shape])
        assert tuple(v_shape) == vt.type.shape
        assert tuple(i_shape) == it.type.shape
        # Test values
        v, i = eval_outputs([vt, it])
        assert i.dtype == "int64"
        assert np.all(v == np_max)
        assert np.all(i == np_argm)

    @pytest.mark.parametrize(
        "axis,np_axis",
        [
            (-1, -1),
            (0, 0),
            (1, 1),
            (None, None),
            ([0, 1], None),
            ([1, 0], None),
            (NoneConst.clone(), None),
            (constant(0), 0),
        ],
    )
    def test_basic_2_float16(self, axis, np_axis):
        # Test negative values and bigger range to make sure numpy don't do the argmax as on uint16
        data = (random(20, 30).astype("float16") - 0.5) * 20
        n = as_tensor_variable(data)
        # Test shape propagates (static & eval)
        vt, it = max_and_argmax(n, axis)
        np_max, np_argm = np.max(data, np_axis), np.argmax(data, np_axis)
        assert vt.type.shape == np_max.shape
        assert it.type.shape == np_argm.shape
        v_shape, i_shape = eval_outputs([vt.shape, it.shape])
        assert tuple(v_shape) == vt.type.shape
        assert tuple(i_shape) == it.type.shape
        # Test values
        v, i = eval_outputs([vt, it])
        assert i.dtype == "int64"
        assert np.all(v == np_max)
        assert np.all(i == np_argm)

    def test_basic_2_invalid(self):
        n = as_tensor_variable(random(2, 3))
        with pytest.raises(ValueError):
            eval_outputs(max_and_argmax(n, 3))

        n = as_tensor_variable(random(2, 3))
        with pytest.raises(ValueError):
            eval_outputs(max_and_argmax(n, -3))

    def test_basic_2_valid_neg(self):
        n = as_tensor_variable(random(2, 3))
        v, i = eval_outputs(max_and_argmax(n, -1))
        assert i.dtype == "int64"
        assert v.shape == (2,)
        assert i.shape == (2,)
        assert np.all(v == np.max(n.value, -1))
        assert np.all(i == np.argmax(n.value, -1))
        v, i = eval_outputs(max_and_argmax(n, -2))
        assert i.dtype == "int64"
        assert v.shape == (3,)
        assert i.shape == (3,)
        assert np.all(v == np.max(n.value, -2))
        assert np.all(i == np.argmax(n.value, -2))
        v = eval_outputs(max_and_argmax(n, -1)[0].shape)
        assert v == (2)
        v = eval_outputs(max_and_argmax(n, -2)[0].shape)
        assert v == (3)

    @pytest.mark.parametrize(
        "axis,np_axis",
        [
            (-1, -1),
            (0, 0),
            (1, 1),
            (None, None),
            ([0, 1, 2], None),
            ([1, 2, 0], None),
        ],
    )
    def test_basic_3(self, axis, np_axis):
        data = random(2, 3, 4)
        n = as_tensor_variable(data)
        # Test shape propagates (static & eval)
        vt, it = max_and_argmax(n, axis)
        np_max, np_argm = np.max(data, np_axis), np.argmax(data, np_axis)
        assert vt.type.shape == np_max.shape
        assert it.type.shape == np_argm.shape
        v_shape, i_shape = eval_outputs([vt.shape, it.shape])
        assert tuple(v_shape) == vt.type.shape
        assert tuple(i_shape) == it.type.shape
        # Test values
        v, i = eval_outputs([vt, it])
        assert i.dtype == "int64"
        assert np.all(v == np_max)
        assert np.all(i == np_argm)

    def test_arg_grad(self):
        # The test checks that the gradient of argmax(x).sum() is 0

        x = matrix()
        cost = argmax(x, axis=0).sum()
        gx = grad(cost, x)
        val = get_underlying_scalar_constant_value(gx)
        assert val == 0.0

    def test_grad(self):
        data = random(2, 3)
        n = as_tensor_variable(data)

        def safe_verify_grad(func, data):
            # Wrapper around 'verify_grad' that picks a proper value for epsilon.
            #
            # This is needed because 'verify_grad' may fail when its epsilon is
            # too large, due to the fact the argmax is not continuous.
            # We make sure epsilon is less than the minimum absolute value found
            # in the matrix of pairwise differences between all elements in the
            # data. This way, the argmax will not change when adding epsilon.

            # 'data' is a one-element list.
            (data_tensor,) = data
            # Flatten it into a 1D vector.
            data_vector = data_tensor.flatten()
            # Compute pairwise absolute differences.
            diff = np.abs(data_vector.reshape((-1, 1)) - data_vector)
            # Alter the diagonal to avoid a zero minimum.
            for i in range(len(diff)):
                diff[i, i] = 1
            # Find an appropriate epsilon.
            eps = builtins.min(numeric_grad.type_eps[config.floatX], diff.min() / 2)
            # Run gradient verification.
            utt.verify_grad(func, data, eps=eps)

        def check_grad_max(data, max_grad_data, axis=None):
            # Why this is needed? verify_grad is not enough?
            # This works only for axis in [0, None].
            assert axis in [0, None]
            z = np.zeros_like(data)
            z = z.flatten()
            argmax = np.argmax(data, axis=axis)
            if argmax.ndim == 0:
                z[argmax] += 1
            else:
                for id, v in enumerate(argmax):
                    z[v * np.prod(data.shape[data.ndim - 1 : axis : -1]) + id] += 1

            z = z.reshape(data.shape)
            assert np.all(max_grad_data == z)

        for axis in (-1, 0, 1, None):
            for j in range(2):
                safe_verify_grad(lambda v: max_and_argmax(v, axis=axis)[j], [data])
                if axis != 1:
                    safe_verify_grad(
                        lambda v: max_and_argmax(v.flatten(), axis=axis)[j], [data]
                    )
            if axis in (0, None):
                check_grad_max(
                    data,
                    eval_outputs(grad(max_and_argmax(n, axis=axis)[0].sum(), n)),
                    axis=axis,
                )
            check_grad_max(data, eval_outputs(grad(max_and_argmax(n.flatten())[0], n)))

        # Test 3d inner dimensions
        data = random(3, 4, 5)

        for i in [0, 1, 2]:
            safe_verify_grad(lambda v: max_and_argmax(v, axis=[i])[0], [data])
            safe_verify_grad(lambda v: max_and_argmax(v, axis=[i])[1], [data])

        # Test 4d inner dimensions
        data = random(2, 3, 4, 5)

        for i in [0, 1, 2, 3]:
            safe_verify_grad(lambda v: max_and_argmax(v, axis=[i])[0], [data])
            safe_verify_grad(lambda v: max_and_argmax(v, axis=[i])[1], [data])

        # Test grad with multiple axes
        for i in [[0, 1], [0, 0]]:
            safe_verify_grad(lambda v: max_and_argmax(v, axis=i)[0], [data])
            safe_verify_grad(lambda v: max_and_argmax(v, axis=i)[1], [data])

    def test_preserve_broadcastable(self):
        # Ensure the original broadcastable flags are preserved by Max/Argmax.
        x = matrix().dimshuffle("x", 0, "x", 1, "x")
        y = x.max(axis=1)
        assert y.type.shape == (1, 1, None, 1)
        assert y.type.broadcastable == (True, True, False, True)

    def test_multiple_axes(self):
        data = np.arange(24).reshape(3, 2, 4)
        x = as_tensor_variable(data)
        vt, it = max_and_argmax(x, [1, -1])
        assert vt.type.shape == it.type.shape == (3,)
        v, i = eval_outputs([vt, it])
        assert np.all(v == np.array([7, 15, 23]))
        assert np.all(i == np.array([7, 7, 7]))
        v = eval_outputs(vt.shape)
        assert tuple(v) == vt.type.shape

    def test_zero_shape(self):
        x = matrix()
        m, i = max_and_argmax(x, axis=1)
        f = function([x], [m, i])
        xv = np.zeros((0, 4), dtype=config.floatX)
        mv, iv = f(xv)
        assert mv.shape == (0,)
        assert iv.shape == (0,)

    def test_numpy_input(self):
        ar = np.array([1, 2, 3])
        max_pt, argmax_pt = max_and_argmax(ar, axis=None)
        assert max_pt.eval() == 3
        assert argmax_pt.eval() == 2

    @pytest.mark.parametrize(
        "core_axis, batch_axis",
        [
            (None, (1, 2, 3, 4)),
            (0, (1,)),
            ((1, -1), (2, 4)),
        ],
    )
    def test_vectorize(self, core_axis, batch_axis):
        x = tensor(shape=(5, 5, 5, 5))
        batch_x = tensor(shape=(3, 5, 5, 5, 5))

        # Test MaxAndArgmax
        max_x, argmax_x = max_and_argmax(x, axis=core_axis)
        node = max_x.owner
        assert isinstance(node.op, TensorMax)

        # dbt: how to make Argmax user facing?
        # new_node = vectorize_node(node, batch_x)
        # pytensor.dprint(new_node)
        # assert isinstance(new_node.op, Argmax)
        # assert new_node.op.axis == batch_axis

        # Test Argmax
        # Argmax is not user-facing, so we have to create it manually
        node = Argmax(axis=node.op.axis).make_node(x)

        new_node = vectorize_node(node, batch_x)
        # print()
        # pytensor.dprint(new_node)
        # print()
        assert isinstance(new_node.op, Argmax)
        assert new_node.op.axis == batch_axis


class TestArgminArgmax:
    def setup_method(self):
        TensorMax.debug = 0

    def test_scalar(self):
        for fct in [argmin, argmax]:
            n = as_tensor_variable([5.0])
            i = eval_outputs(fct(n))
            assert i == 0
            v = eval_outputs(fct(n).shape)
            assert len(v) == 0

    def test_list(self):
        n = as_tensor_variable([1, 2, 3, 2, -6])
        i = eval_outputs(argmin(n))
        assert i == 4
        v = eval_outputs(argmin(n).shape)
        assert len(v) == 0

        n = as_tensor_variable([1, 2, 3, 2, -6])
        i = eval_outputs(argmax(n))
        assert i == 2
        v = eval_outputs(argmax(n).shape)
        assert len(v) == 0

    def test2(self):
        data = random(2, 3)
        n = as_tensor_variable(data)
        for fct, nfct in [(argmax, np.argmax), (argmin, np.argmin)]:
            for axis, np_axis in [
                (-1, -1),
                (0, 0),
                (1, 1),
                (None, None),
                ([0, 1], None),
                ([1, 0], None),
            ]:
                v = eval_outputs(fct(n, axis))
                assert np.all(v == nfct(data, np_axis))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test2_float16(self):
        # Test negative values and bigger range to make sure numpy don't do the argmax as on uint16
        data = (random(20, 30).astype("float16") - 0.5) * 20
        n = shared(data)
        mode = get_default_mode().including("local_max_and_argmax", "uncanonicalize")
        for fct, nfct in [(argmax, np.argmax), (argmin, np.argmin)]:
            for axis, np_axis in [
                (-1, -1),
                (0, 0),
                (1, 1),
                (None, None),
                ([0, 1], None),
                ([1, 0], None),
            ]:
                v = eval_outputs(fct(n, axis), (Argmax,), mode=mode)
                assert np.all(v == nfct(data, np_axis))
                v_shape = eval_outputs(fct(n, axis).shape, mode=mode)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test2_invalid(self):
        for fct, nfct in [(argmax, np.argmax), (argmin, np.argmin)]:
            n = as_tensor_variable(random(2, 3))
            with pytest.raises(ValueError):
                eval_outputs(fct(n, 3))
            with pytest.raises(ValueError):
                eval_outputs(fct(n, -3))

    def test2_valid_neg(self):
        for fct, nfct in [(argmax, np.argmax), (argmin, np.argmin)]:
            n = as_tensor_variable(random(2, 3))
            i = eval_outputs(fct(n, -1))
            assert i.shape == (2,)
            assert np.all(i == nfct(n.value, -1))
            i = eval_outputs(fct(n, -2))
            assert i.shape == (3,)
            assert np.all(i == nfct(n.value, -2))

            v = eval_outputs(fct(n, -1).shape)
            assert v == (2)
            v = eval_outputs(fct(n, -2).shape)
            assert v == (3)

    def test3(self):
        data = random(2, 3, 4)
        n = as_tensor_variable(data)
        for fct, nfct in [(argmax, np.argmax), (argmin, np.argmin)]:
            for axis, np_axis in [
                (-1, -1),
                (0, 0),
                (1, 1),
                (2, 2),
                (None, None),
                ([0, 1, 2], None),
                ([1, 0, 2], None),
            ]:
                v = eval_outputs(fct(n, axis))
                assert np.all(v == nfct(data, np_axis))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test_grad_argmin(self):
        data = random(2, 3)
        n = as_tensor_variable(data)
        n.name = "n"

        # test grad of argmin
        utt.verify_grad(lambda v: argmin(v, axis=-1), [data])

        utt.verify_grad(lambda v: argmin(v, axis=[0]), [data])

        utt.verify_grad(lambda v: argmin(v, axis=[1]), [data])

        utt.verify_grad(lambda v: argmin(v.flatten()), [data])

        try:
            cost = argmin(n, axis=-1)
            cost.name = None
            grad(cost, n)
            raise Exception("Expected an error")
        except TypeError:
            pass

    def test_grad_argmax(self):
        data = random(2, 3)
        n = as_tensor_variable(data)

        # test grad of argmax
        utt.verify_grad(lambda v: argmax(v, axis=-1), [data])

        utt.verify_grad(lambda v: argmax(v, axis=[0]), [data])

        utt.verify_grad(lambda v: argmax(v, axis=[1]), [data])

        utt.verify_grad(lambda v: argmax(v.flatten()), [data])

        try:
            grad(argmax(n, axis=-1), n)
            raise Exception("Expected an error")
        except TypeError:
            pass

    def test_uint(self):
        for dtype in ("uint8", "uint16", "uint32", "uint64"):
            itype = np.iinfo(dtype)
            data = np.array([itype.min + 3, itype.min, itype.max - 5, itype.max], dtype)
            n = as_tensor_variable(data)
            i = eval_outputs(argmin(n))
            assert i == 1
            i = eval_outputs(argmax(n))
            assert i == 3

    def test_bool(self):
        data = np.array([True, False], "bool")
        n = as_tensor_variable(data)
        i = eval_outputs(argmin(n))
        assert i == 1
        i = eval_outputs(argmax(n))
        assert i == 0


class TestMinMax:
    def setup_method(self):
        TensorMax.debug = 0

    def test_scalar(self):
        for fct in [max, min]:
            n = as_tensor_variable(5.0)
            v = eval_outputs(fct(n))
            assert v == 5.0

            v = eval_outputs(fct(n).shape)
            assert len(v) == 0

    def test_list(self):
        for fct, nfct in [(max, np.max), (min, np.min)]:
            n = as_tensor_variable([1, 2, 3, 2, -6])
            v = eval_outputs([fct(n)])
            assert v == nfct(n.value)

            v = eval_outputs(fct(n).shape)
            assert len(v) == 0

    def test2(self):
        data = random(2, 3)
        n = as_tensor_variable(data)
        for fct, nfct in [(max, np.max), (min, np.min)]:
            for axis, np_axis in [
                (-1, -1),
                (0, 0),
                (1, 1),
                (None, None),
                ([0, 1], None),
                ([1, 0], None),
            ]:
                v = eval_outputs(fct(n, axis))
                assert np.all(v == nfct(data, np_axis))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test2_invalid(self):
        for fct in [max, min]:
            n = as_tensor_variable(random(2, 3))
            with pytest.raises(ValueError):
                eval_outputs(fct(n, 3))
            with pytest.raises(ValueError):
                eval_outputs(fct(n, -3))

    def test2_valid_neg(self):
        for fct, nfct in [(max, np.max), (min, np.min)]:
            n = as_tensor_variable(random(2, 3))
            v = eval_outputs(fct(n, -1))
            assert v.shape == (2,)
            assert np.all(v == nfct(n.value, -1))
            v = eval_outputs(fct(n, -2))
            assert v.shape == (3,)
            assert np.all(v == nfct(n.value, -2))

            v = eval_outputs(fct(n, -1).shape)
            assert v == (2)
            v = eval_outputs(fct(n, -2).shape)
            assert v == (3)

    def test3(self):
        # Test with 1 axis or all axis out of 3 dims
        data = random(2, 3, 4)
        n = as_tensor_variable(data)
        for fct, nfct in [(max, np.max), (min, np.min)]:
            for axis, np_axis in [
                (-1, -1),
                (0, 0),
                (1, 1),
                (2, 2),
                (None, None),
                ([0, 1, 2], None),
                ([1, 0, 2], None),
            ]:
                v = eval_outputs(fct(n, axis))
                assert np.all(v == nfct(data, np_axis))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test3b(self):
        # Test with 2 axis out of 3 dims
        data = random(2, 3, 4)
        n = as_tensor_variable(data)
        for fct, nfct in [(max, np.max), (min, np.min)]:
            for axis in [[0, 1], [1, 2], [0, 2]]:
                v = eval_outputs(fct(n, axis))
                np_v = nfct(nfct(data, axis[1]), axis[0])
                assert np.all(v == np_v)
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == np_v.shape

    def test_grad_max(self):
        data = random(2, 3)
        n = as_tensor_variable(data)

        def check_grad_max(data, max_grad_data, axis=None):
            # This work only for axis in [0,None]
            assert axis in [0, None]
            z = np.zeros_like(data)
            z = z.flatten()
            argmax = np.argmax(data, axis=axis)
            if argmax.ndim == 0:
                z[np.argmax(data, axis=axis)] += 1
            else:
                for id, v in enumerate(argmax):
                    z[v * np.prod(data.shape[data.ndim - 1 : axis : -1]) + id] += 1

            z = z.reshape(data.shape)
            assert np.all(max_grad_data == z)

        # test grad of max
        # axis is the last one
        utt.verify_grad(lambda v: max(v, axis=-1), [data])

        utt.verify_grad(lambda v: max(v, axis=[0]), [data])
        check_grad_max(data, eval_outputs(grad(max(n, axis=0).sum(), n)), axis=0)

        utt.verify_grad(lambda v: max(v, axis=[1]), [data])
        # check_grad_max(data,eval_outputs(grad(max(n,axis=1),n)),axis=1)

        utt.verify_grad(lambda v: max(v.flatten()), [data])
        check_grad_max(data, eval_outputs(grad(max(n.flatten()), n)))

    def test_grad_min(self):
        data = random(2, 3)
        n = as_tensor_variable(data)

        def check_grad_min(data, min_grad_data, axis=None):
            # This work only for axis in [0, None]
            assert axis in [0, None]
            z = np.zeros_like(data)
            z = z.flatten()
            argmin = np.argmin(data, axis=axis)
            if argmin.ndim == 0:
                z[np.argmin(data, axis=axis)] += 1
            else:
                for id, v in enumerate(argmin):
                    z[v * np.prod(data.shape[data.ndim - 1 : axis : -1]) + id] += 1

            z = z.reshape(data.shape)
            assert np.all(min_grad_data == z)

        # test grad of min
        # axis is the last one
        utt.verify_grad(lambda v: min(v, axis=-1), [data])

        utt.verify_grad(lambda v: min(v, axis=[0]), [data])
        check_grad_min(data, eval_outputs(grad(min(n, axis=0).sum(), n)), axis=0)

        utt.verify_grad(lambda v: min(v, axis=[1]), [data])
        # check_grad_min(data,eval_outputs(grad(min(n,axis=1),n)),axis=1)

        utt.verify_grad(lambda v: min(v.flatten()), [data])
        check_grad_min(data, eval_outputs(grad(min(n.flatten()), n)))

    def _grad_list(self):
        # Test the gradient when we have multiple axis at the same time.
        #
        # This not implemented, so we disable the test. See ticket:
        # http://www.assembla.com/spaces/pytensor/tickets/511
        data = random(2, 3)
        for fct in [max_and_argmax, max, min]:
            utt.verify_grad(lambda v: fct(v, axis=[0, 1]), [data])
        # n = as_tensor_variable(data)
        # check_grad_max(data, eval_outputs(grad(max_and_argmax(n,
        # axis=1)[0], n)),axis=1)

    def test_uint(self):
        for dtype in ("uint8", "uint16", "uint32", "uint64"):
            itype = np.iinfo(dtype)
            data = np.array([itype.min + 3, itype.min, itype.max - 5, itype.max], dtype)
            n = as_tensor_variable(data)
            assert min(n).dtype == dtype
            i = eval_outputs(min(n))
            assert i == itype.min
            assert max(n).dtype == dtype
            i = eval_outputs(max(n))
            assert i == itype.max

    def test_bool(self):
        data = np.array([True, False], "bool")
        n = as_tensor_variable(data)
        assert min(n).dtype == "bool"
        i = eval_outputs(min(n))
        assert i.ndim == 0
        assert not np.any(i)
        assert max(n).dtype == "bool"
        i = eval_outputs(max(n))
        assert i.ndim == 0
        assert np.all(i)
