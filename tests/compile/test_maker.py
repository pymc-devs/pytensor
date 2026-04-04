import copy
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy as sp
from numba.core.errors import TypingError as NumbaTypingError

import pytensor.tensor as pt
from pytensor.compile import UnusedInputError, get_mode
from pytensor.compile.debugmode import DebugMode, InvalidValueError
from pytensor.compile.function_maker import function, function_dump
from pytensor.compile.io import In, Out
from pytensor.compile.mode import Mode, get_default_mode
from pytensor.compile.sharedvalue import shared
from pytensor.configdefaults import config
from pytensor.graph.basic import Constant
from pytensor.graph.utils import MissingInputError
from pytensor.link.basic import Container
from pytensor.printing import debugprint
from pytensor.sparse import SparseTensorType
from pytensor.tensor.math import dot, tanh
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.type import (
    bscalar,
    bvector,
    dmatrices,
    dmatrix,
    dscalar,
    dscalars,
    dvector,
    fscalar,
    fvector,
    imatrix,
    iscalar,
    ivector,
    ivectors,
    lscalar,
    matrix,
    scalar,
    scalars,
    vector,
    wvector,
)
from pytensor.utils import PYTHON_INT_BITWIDTH


floatX = "float32"


def test_function_dump():
    v = vector()
    fct1 = function([v], v + 1)

    try:
        tmpdir = Path(tempfile.mkdtemp())
        fname = tmpdir / "test_function_dump.pkl"
        function_dump(fname, [v], v + 1)
        with fname.open("rb") as f:
            l = pickle.load(f)
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir)

    fct2 = function(**l)
    x = [1, 2, 3]
    assert np.allclose(fct1(x), fct2(x))


def test_function_name():
    x = vector("x")
    func = function([x], x + 1.0)

    assert __file__ in func.name


class TestFunctionIn:
    def test_in_strict(self):
        a = dvector()
        b = shared(7)
        out = a + b

        f = function([In(a, strict=False)], out)

        # works, rand generates float64 by default
        assert f(np.random.random((8,)).astype(np.float64)).dtype == np.float64

        # works, casting is allowed
        assert f(np.array([1, 2, 3, 4], dtype="int32")).dtype == np.float64

        f = function([In(a, strict=True)], out)

        with pytest.raises(TypeError):
            # fails, f expects float64
            f(np.array([1, 2, 3, 4], dtype="int32"))

    def test_explicit_shared_input(self):
        # This is not a test of the In class per se, but the In class relies
        # on the fact that shared variables cannot be explicit inputs
        a = shared(1.0)
        with pytest.raises(TypeError):
            function([a], a + 1)

    def test_in_shared_variable(self):
        # Ensure that an error is raised if the In wrapped is used to wrap
        # a shared variable
        a = shared(1.0)
        a_wrapped = In(a, update=a + 1)
        with pytest.raises(TypeError):
            function([a_wrapped])

    def test_in_mutable(self):
        a = dvector()
        a_out = a * 2  # assuming the op which makes this "in place" triggers

        # using mutable=True will let f change the value in aval
        f = function([In(a, mutable=True)], a_out, mode="FAST_RUN")
        aval = np.random.random((10,))
        aval2 = aval.copy()
        assert np.array_equal(f(aval), (aval2 * 2))
        assert not np.array_equal(aval, aval2)

        # using mutable=False should leave the input untouched
        f = function([In(a, mutable=False)], a_out, mode="FAST_RUN")
        aval = np.random.random((10,))
        aval2 = aval.copy()
        assert np.array_equal(f(aval), (aval2 * 2))
        assert np.array_equal(aval, aval2)

    def test_in_update(self):
        a = dscalar("a")
        # A shared variable by any other name
        c = Container(a, storage=[np.array(0.0)])
        f = function([In(a, value=c, implicit=True, update=a + 1)], a, mode="FAST_RUN")

        # Ensure that, through the executions of the function, the state of the
        # input is persistent and is updated as it should
        assert f() == 0.0
        assert f() == 1.0
        assert f() == 2.0

    def test_in_update_wrong_dtype(self):
        # Ensure that an error is raised if an In-wrapped variables has
        # an update of a different type
        a = dscalar("a")
        b = dvector("b")
        with pytest.raises(TypeError):
            In(a, update=b)

    def test_in_update_shared(self):
        # Test that using both In() with updates and shared variables with
        # updates in the same function behaves as expected
        shared_var = shared(1.0)
        a = dscalar("a")
        container = Container(a, storage=[np.array(0.0)])
        a_wrapped = In(a, value=container, update=shared_var)
        f = function([a_wrapped], [], updates={shared_var: a}, mode="FAST_RUN")

        # Ensure that, through the executions of the function, the state of
        # the input and the shared variable are appropriate (after N execution,
        # the values have swapped N times). This allows testing that the
        # changes occur at the same time and one doesn't overwrite the other.
        for i in range(5):
            f()
            assert np.allclose(shared_var.get_value(), i % 2)

    def test_in_allow_downcast_int(self):
        a = wvector("a")  # int16
        b = bvector("b")  # int8
        c = bscalar("c")  # int8
        f = function(
            [
                In(a, allow_downcast=True),
                In(b, allow_downcast=False),
                In(c, allow_downcast=None),
            ],
            (a + b + c),
        )

        # Both values are in range. Since they're not ndarrays (but lists),
        # they will be converted, and their value checked.
        assert np.array_equal(f([3], [6], 1), [10])

        # Values are in range, but a dtype too large has explicitly been given
        # For performance reasons, no check of the data is explicitly performed
        # (It might be OK to change this in the future.)
        with pytest.raises(TypeError):
            f([3], np.array([6], dtype="int16"), 1)

        # Value too big for a, silently ignored
        assert np.array_equal(f([2**20], np.ones(1, dtype="int8"), 1), [2])

        # Value too big for b, raises OverflowError (in numpy >= 2.0... TypeError in numpy < 2.0)
        with pytest.raises(OverflowError):
            f([3], [312], 1)

        # Value too big for c, raises OverflowError
        with pytest.raises(OverflowError):
            f([3], [6], 806)

    def test_in_allow_downcast_floatX(self):
        a = fscalar("a")
        b = fscalar("b")
        c = fscalar("c")

        f = function(
            [
                In(a, allow_downcast=True),
                In(b, allow_downcast=False),
                In(c, allow_downcast=None),
            ],
            (a + b + c),
        )

        # If the values can be accurately represented, everything is OK
        assert np.array_equal(f(0, 0, 0), 0)

        # If allow_downcast is True, idem
        assert np.allclose(f(0.1, 0, 0), 0.1)

        # If allow_downcast is False, nope
        with pytest.raises(TypeError):
            f(0, 0.1, 0)

        # If allow_downcast is None, it should work iff floatX=float32
        if config.floatX == "float32":
            assert np.allclose(f(0, 0, 0.1), 0.1)
        else:
            with pytest.raises(TypeError):
                f(0, 0, 0.1)

    def test_in_allow_downcast_vector_floatX(self):
        a = fvector("a")
        b = fvector("b")
        c = fvector("c")

        f = function(
            [
                In(a, allow_downcast=True),
                In(b, allow_downcast=False),
                In(c, allow_downcast=None),
            ],
            (a + b + c),
        )

        # If the values can be accurately represented, everything is OK
        z = [0]
        assert np.array_equal(f(z, z, z), [0])

        # If allow_downcast is True, idem
        assert np.allclose(f([0.1], z, z), 0.1)

        # If allow_downcast is False, nope
        with pytest.raises(TypeError):
            f(z, [0.1], z)

        # If allow_downcast is None, like False
        with pytest.raises(TypeError):
            f(z, z, [0.1])


def data_of(s):
    # Return the raw value of a shared variable
    return s.container.storage[0]


# Test used to be a distinict test_function and test_pfunc that were merged
# We need to check TestFunction and TestFunction2 and merge / remove redundancy


class TestFunction:
    def test_errors(self):
        a = lscalar()
        b = shared(1)

        with pytest.raises(TypeError):
            function({a}, a + b)

        with pytest.raises(TypeError):
            function([a], a + b, no_default_updates=1)

        with pytest.raises(TypeError):
            function([a], a + b, updates=[{b, a}])

        with pytest.raises(TypeError):
            function([a], a + b, updates=[(1, b)])

    def test_doc(self):
        # Ensure the code given in the docs works as expected

        # Example #1.
        a = lscalar()
        b = shared(1)
        f1 = function([a], (a + b))
        f2 = function([In(a)], a + b, updates={b: b + 1})
        assert b.get_value() == 1
        assert f1(3) == 4
        assert f2(3) == 4
        assert b.get_value() == 2
        assert f1(3) == 5
        b.set_value(0)
        assert f1(3) == 3

        # Example #2.
        a = lscalar()
        b = shared(7)
        f1 = function([a], a + b)
        f2 = function([a], a * b)
        assert f1(5) == 12
        b.set_value(8)
        assert f1(5) == 13
        assert f2(4) == 32

    def test_shared(self):
        # CHECK: two functions (f1 and f2) can share w
        w = shared(np.random.random((2, 2)), "w")
        wval = w.get_value(borrow=False)

        x = dmatrix()
        out1 = w + x
        out2 = w * x
        f1 = function([x], [out1])
        f2 = function([x], [out2])
        xval = np.random.random((2, 2))
        assert np.all(f1(xval) == xval + wval)
        assert np.all(f2(xval) == xval * wval)

        # CHECK: updating a shared value
        f3 = function([x], out1, updates=[(w, (w - 1))])
        # f3 changes the value of w
        assert np.all(f3(xval) == xval + wval)
        # this same value is read by f1
        assert np.all(f1(xval) == xval + (wval - 1))

        w.set_value(w.get_value(borrow=True) * 10, borrow=True)
        # this same value is read by f1
        assert np.all(f1(xval) == xval + w.get_value(borrow=True))

    def test_no_shared_as_input(self):
        # Test that shared variables cannot be used as function inputs.
        w_init = np.random.random((2, 2))
        w = shared(w_init.copy(), "w")
        with pytest.raises(
            TypeError, match=r"^Cannot use a shared variable \(w\) as explicit input"
        ):
            function([w], pt_sum(w * w))

    def test_default_container(self):
        # Ensure it is possible to (implicitly) use a shared variable in a
        # function, as a 'state' that can be updated at will.

        rng = np.random.default_rng(1827)
        w_init = rng.random(5)
        w = shared(w_init.copy(), "w")
        reg = pt_sum(w * w)
        f = function([], reg)

        assert f() == np.sum(w_init * w_init)
        # Change the value of w and ensure the output changes accordingly.
        w.set_value(w.get_value(borrow=True) + 1.0, borrow=True)
        assert f() == np.sum((w_init + 1) ** 2)

    def test_default_scalar_container(self):
        # Similar in spirit to test_default_container, but updating a scalar
        # variable. This is a sanity check for non mutable types.
        x = shared(0.0, "x")
        f = function([], x)
        assert f() == 0
        x.set_value(x.get_value(borrow=True) + 1, borrow=True)
        assert f() == 1

    def test_param_strict(self):
        a = dvector()
        b = shared(7)
        out = a + b

        f = function([In(a, strict=False)], [out])
        # works, random( generates float64 by default
        f(np.random.random(8))
        # works, casting is allowed
        f(np.array([1, 2, 3, 4], dtype="int32"))

        f = function([In(a, strict=True)], [out])
        try:
            # fails, f expects float64
            f(np.array([1, 2, 3, 4], dtype="int32"))
        except TypeError:
            pass

    def test_param_mutable(self):
        a = dvector()
        a_out = a * 2  # assuming the op which makes this "in place" triggers

        # using mutable=True will let fip change the value in aval
        fip = function([In(a, mutable=True)], [a_out], mode="FAST_RUN")
        aval = np.random.random(10)
        aval2 = aval.copy()
        assert np.all(fip(aval) == (aval2 * 2))
        assert not np.all(aval == aval2)

        # using mutable=False should leave the input untouched
        f = function([In(a, mutable=False)], [a_out], mode="FAST_RUN")
        aval = np.random.random(10)
        aval2 = aval.copy()
        assert np.all(f(aval) == (aval2 * 2))
        assert np.all(aval == aval2)

    def test_shared_mutable(self):
        bval = np.arange(5)
        b = shared(bval)
        b_out = b * 2

        # shared vars copy args.
        assert b.get_value(borrow=True) is not bval
        # so we do this to get at the underlying data
        bval = data_of(b)

        # by default, shared are not mutable unless doing an explicit update
        f = function([], [b_out], mode="FAST_RUN")
        assert (f() == np.arange(5) * 2).all()
        assert np.all(b.get_value(borrow=True) == np.arange(5))

        # using updates, b is now a mutable parameter
        f = function([], [b_out], updates=[(b, b_out)], mode="FAST_RUN")
        assert (f() == (np.arange(5) * 2)).all()
        # because of the update
        assert (b.get_value(borrow=True) == (np.arange(5) * 2)).all()
        assert (bval == (np.arange(5) * 2)).all()  # because of mutable=True

        # do not depend on updates being in-place though!
        bval = np.arange(5)
        b.set_value(bval, borrow=True)
        bval = data_of(b)
        f = function(
            [],
            [b_out],
            updates=[(b, (b_out + 3))],
            mode=get_mode("FAST_RUN").excluding("fusion"),
        )
        assert (f() == (np.arange(5) * 2)).all()
        # because of the update
        assert (b.get_value(borrow=True) == ((np.arange(5) * 2) + 3)).all()
        # bval got modified to something...
        assert not (bval == np.arange(5)).all()
        # ... but not to b.value !
        assert not (bval == b.get_value(borrow=True)).all()

    def test_param_allow_downcast_int(self):
        a = wvector("a")  # int16
        b = bvector("b")  # int8
        c = bscalar("c")  # int8
        f = function(
            [
                In(a, allow_downcast=True),
                In(b, allow_downcast=False),
                In(c, allow_downcast=None),
            ],
            (a + b + c),
        )

        # Both values are in range. Since they're not ndarrays (but lists),
        # they will be converted, and their value checked.
        assert np.all(f([3], [6], 1) == 10)

        # Values are in range, but a dtype too large has explicitly been given
        # For performance reasons, no check of the data is explicitly performed
        # (It might be OK to change this in the future.)
        with pytest.raises(TypeError):
            f([3], np.array([6], dtype="int16"), 1)

        # Value too big for a, silently ignored
        assert np.all(f([2**20], np.ones(1, dtype="int8"), 1) == 2)

        # Value too big for b, raises OverflowError in numpy >= 2.0, TypeError in numpy <2.0
        with pytest.raises(OverflowError):
            f([3], [312], 1)

        # Value too big for c, raises OverflowError in numpy >= 2.0, TypeError in numpy <2.0
        with pytest.raises(OverflowError):
            f([3], [6], 806)

    def test_param_allow_downcast_floatX(self):
        a = fscalar("a")
        b = fscalar("b")
        c = fscalar("c")

        f = function(
            [
                In(a, allow_downcast=True),
                In(b, allow_downcast=False),
                In(c, allow_downcast=None),
            ],
            (a + b + c),
        )

        # If the values can be accurately represented, everything is OK
        assert np.all(f(0, 0, 0) == 0)

        # If allow_downcast is True, idem
        assert np.allclose(f(0.1, 0, 0), 0.1)

        # If allow_downcast is False, nope
        with pytest.raises(TypeError):
            f(0, 0.1, 0)

        # If allow_downcast is None, it should work iff floatX=float32
        if config.floatX == "float32":
            assert np.allclose(f(0, 0, 0.1), 0.1)
        else:
            with pytest.raises(TypeError):
                f(0, 0, 0.1)

    def test_param_allow_downcast_vector_floatX(self):
        a = fvector("a")
        b = fvector("b")
        c = fvector("c")

        f = function(
            [
                In(a, allow_downcast=True),
                In(b, allow_downcast=False),
                In(c, allow_downcast=None),
            ],
            (a + b + c),
        )

        # If the values can be accurately represented, everything is OK
        z = [0]
        assert np.all(f(z, z, z) == 0)

        # If allow_downcast is True, idem
        assert np.allclose(f([0.1], z, z), 0.1)

        # If allow_downcast is False, nope
        with pytest.raises(TypeError):
            f(z, [0.1], z)

        # If allow_downcast is None, like False
        with pytest.raises(TypeError):
            f(z, z, [0.1])

    def test_allow_input_downcast_int(self):
        a = wvector("a")  # int16
        b = bvector("b")  # int8
        c = bscalar("c")  # int8

        f = function([a, b, c], (a + b + c), allow_input_downcast=True)
        # Value too big for a, b, or c, silently ignored
        assert f([2**20], [1], 0) == 1
        assert f([3], [312], 0) == 59
        assert f([3], [1], 806) == 42

        g = function([a, b, c], (a + b + c), allow_input_downcast=False)
        # All values are in range. Since they're not ndarrays (but lists
        # or scalars), they will be converted, and their value checked.
        assert np.all(g([3], [6], 0) == 9)

        # Values are in range, but a dtype too large has explicitly been given
        # For performance reasons, no check of the data is explicitly performed
        # (It might be OK to change this in the future.)
        with pytest.raises(TypeError):
            g([3], np.array([6], dtype="int16"), 0)

        # Value too big for b, raises OverflowError in numpy >= 2.0, TypeError in numpy <2.0
        with pytest.raises(OverflowError):
            g([3], [312], 0)

        h = function([a, b, c], (a + b + c))  # Default: allow_input_downcast=None
        # Everything here should behave like with False
        assert np.all(h([3], [6], 0) == 9)

        with pytest.raises(TypeError):
            h([3], np.array([6], dtype="int16"), 0)

        # Value too big for b, raises OverflowError in numpy >= 2.0, TypeError in numpy <2.0
        with pytest.raises(OverflowError):
            h([3], [312], 0)

    def test_allow_downcast_floatX(self):
        a = fscalar("a")
        b = fvector("b")

        f = function([a, b], (a + b), allow_input_downcast=True)
        g = function([a, b], (a + b), allow_input_downcast=False)
        h = function([a, b], (a + b), allow_input_downcast=None)

        # If the values can be accurately represented, OK
        assert np.all(f(0, [0]) == 0)
        assert np.all(g(0, [0]) == 0)
        assert np.all(h(0, [0]) == 0)

        # For the vector: OK iff allow_input_downcast is True
        assert np.allclose(f(0, [0.1]), 0.1)
        with pytest.raises(TypeError):
            g(0, [0.1])
        with pytest.raises(TypeError):
            h(0, [0.1])

        # For the scalar: OK if allow_input_downcast is True,
        # or None and floatX==float32
        assert np.allclose(f(0.1, [0]), 0.1)
        with pytest.raises(TypeError):
            g(0.1, [0])
        if config.floatX == "float32":
            assert np.allclose(h(0.1, [0]), 0.1)
        else:
            with pytest.raises(TypeError):
                h(0.1, [0])

    def test_update(self):
        # Test update mechanism in different settings.

        # Simple value assignment.
        x = shared(0)
        assign = function([], [], updates={x: 3})
        assign()
        assert x.get_value() == 3

        # Basic increment function.
        x.set_value(0)
        inc = function([], [], updates={x: x + 1})
        inc()
        assert x.get_value() == 1

        # Increment by a constant value.
        x.set_value(-1)
        y = shared(2)
        inc_by_y = function([], [], updates={x: x + y})
        inc_by_y()
        assert x.get_value() == 1

    def test_update_err_broadcast(self):
        # Test that broadcastable dimensions raise error
        data = np.random.random((10, 10)).astype("float32")
        output_var = shared(name="output", value=data)

        # the update_var has type matrix, and the update expression
        # is a broadcasted scalar, and that should be allowed.
        with pytest.raises(TypeError):
            function(
                inputs=[],
                outputs=[],
                updates={output_var: output_var.sum().dimshuffle("x", "x")},
            )

    def test_duplicate_updates(self):
        x, y = dmatrices("x", "y")
        z = shared(np.ones((2, 3)))
        with pytest.raises(ValueError):
            function([x, y], [z], updates=[(z, (z + x + y)), (z, (z - x))])

    def test_givens(self):
        x = shared(0)
        assign = function([], x, givens={x: 3})
        assert assign() == 3
        assert x.get_value(borrow=True) == 0

        y = ivector()
        f = function([y], (y * x), givens={x: 6})
        assert np.all(f([1, 1, 1]) == [6, 6, 6])
        assert x.get_value() == 0

        z = ivector()
        c = z * y
        f = function([y], (c + 7), givens={z: np.asarray([4, 4, 4], dtype="int32")})
        assert np.all(f([1, 1, 1]) == [11, 11, 11])
        assert x.get_value() == 0

    def test_clone0(self):
        x = shared(np.asarray([4, 4, 4]))
        y = shared(np.asarray([4, 4, 4]))
        z = shared(np.asarray([2, 2, 2]))
        up = function(
            [], [], updates={x: (x * 5), y: ((x * 5) + y), z: (((x * 5) + y) ** z)}
        )

        up()
        assert np.all(x.get_value() == 20)
        assert np.all(y.get_value() == 24)
        assert np.all(z.get_value() == (24**2))

    def test_default_updates(self):
        x = shared(0)
        x.default_update = x + 1

        f = function([], [x])
        f()
        assert x.get_value() == 1

        x.default_update = None

        f()
        assert x.get_value() == 2

        g = function([], [x])
        g()
        assert x.get_value() == 2

    def test_no_default_updates(self):
        x = shared(0)
        y = shared(1)
        x.default_update = x + 2

        # Test that the default update is taken into account in the right cases
        f1 = function([], [x], no_default_updates=True)
        f1()
        assert x.get_value() == 0

        f2 = function([], [x], no_default_updates=[x])
        f2()
        assert x.get_value() == 0

        f3 = function([], [x], no_default_updates=[x, y])
        f3()
        assert x.get_value() == 0

        f4 = function([], [x], no_default_updates=[y])
        f4()
        assert x.get_value() == 2

        f5 = function([], [x], no_default_updates=[])
        f5()
        assert x.get_value() == 4

        f5 = function([], [x], no_default_updates=False)
        f5()
        assert x.get_value() == 6

        with pytest.raises(TypeError):
            function([], [x], no_default_updates=(x))
        with pytest.raises(TypeError):
            function([], [x], no_default_updates=x)
        with pytest.raises(TypeError):
            function([], [x], no_default_updates="canard")

        # Mix explicit updates and no_default_updates
        g1 = function([], [x], updates=[(x, (x - 1))], no_default_updates=True)
        g1()
        assert x.get_value() == 5

        g2 = function([], [x], updates=[(x, (x - 1))], no_default_updates=[x])
        g2()
        assert x.get_value() == 4

        g3 = function([], [x], updates=[(x, (x - 1))], no_default_updates=[x, y])
        g3()
        assert x.get_value() == 3

        g4 = function([], [x], updates=[(x, (x - 1))], no_default_updates=[y])
        g4()
        assert x.get_value() == 2

        g5 = function([], [x], updates=[(x, (x - 1))], no_default_updates=[])
        g5()
        assert x.get_value() == 1

        g5 = function([], [x], updates=[(x, (x - 1))], no_default_updates=False)
        g5()
        assert x.get_value() == 0

    def test_default_updates_expressions(self):
        x = shared(0)
        y = shared(1)
        a = lscalar("a")

        z = a * x
        x.default_update = x + y

        f1 = function([a], z)
        f1(12)
        assert x.get_value() == 1

        f2 = function([a], z, no_default_updates=True)
        assert f2(7) == 7
        assert x.get_value() == 1

        f3 = function([a], z, no_default_updates=[x])
        assert f3(9) == 9
        assert x.get_value() == 1

    def test_default_updates_multiple(self):
        x = shared(0)
        y = shared(1)

        x.default_update = x - 1
        y.default_update = y + 1

        f1 = function([], [x, y])
        f1()
        assert x.get_value() == -1
        assert y.get_value() == 2

        f2 = function([], [x, y], updates=[(x, (x - 2))], no_default_updates=[y])
        f2()
        assert x.get_value() == -3
        assert y.get_value() == 2

        f3 = function([], [x, y], updates=[(x, (x - 2))], no_default_updates=True)
        f3()
        assert x.get_value() == -5
        assert y.get_value() == 2

        f4 = function([], [x, y], updates=[(y, (y - 2))])
        f4()
        assert x.get_value() == -6
        assert y.get_value() == 0

    def test_default_updates_chained(self):
        x = shared(2)
        y = shared(1)
        z = shared(-1)

        x.default_update = x - y
        y.default_update = z
        z.default_update = z - 1

        f1 = function([], [x])
        f1()
        assert x.get_value() == 1
        assert y.get_value() == -1
        assert z.get_value() == -2

        f2 = function([], [x, y])
        f2()
        assert x.get_value() == 2
        assert y.get_value() == -2
        assert z.get_value() == -3

        f3 = function([], [y])
        f3()
        assert x.get_value() == 2
        assert y.get_value() == -3
        assert z.get_value() == -4

        f4 = function([], [x, y], no_default_updates=[x])
        f4()
        assert x.get_value() == 2
        assert y.get_value() == -4
        assert z.get_value() == -5

        f5 = function([], [x, y, z], no_default_updates=[z])
        f5()
        assert x.get_value() == 6
        assert y.get_value() == -5
        assert z.get_value() == -5

    def test_default_updates_input(self):
        x = shared(0)
        y = shared(1)
        if PYTHON_INT_BITWIDTH == 32:
            a = iscalar("a")
        else:
            a = lscalar("a")

        x.default_update = y
        y.default_update = y + a

        f1 = function([], x, no_default_updates=True)
        f1()
        assert x.get_value() == 0
        assert y.get_value() == 1

        f2 = function([], x, no_default_updates=[x])
        f2()
        assert x.get_value() == 0
        assert y.get_value() == 1

        f3 = function([], x, no_default_updates=[y])
        f3()
        assert x.get_value() == 1
        assert y.get_value() == 1

        f4 = function([a], x)
        f4(2)
        assert x.get_value() == 1
        assert y.get_value() == 3

        f5 = function([], x, updates={y: (y - 1)})
        f5()
        assert x.get_value() == 3
        assert y.get_value() == 2

        # a is needed as input if y.default_update is used
        with pytest.raises(MissingInputError):
            function([], x)

    def test_default_updates_partial_graph(self):
        a = shared(0)
        a.default_update = a + 1  # Increment a each time it is used
        b = 2 * a
        # Use only the tip of the graph, a is not used
        f = function([b], b)
        assert a.get_value() == 0
        f(21)
        assert a.get_value() == 0

    def test_givens_replaces_shared_variable(self):
        a = shared(1.0, "a")
        a.default_update = a + 3.0
        b = dscalar("b")
        c = a + 10
        f = function([b], c, givens={a: b})

        assert len(f.maker.fgraph.inputs) == 1
        assert len(f.maker.fgraph.outputs) == 1

    def test_givens_replaces_shared_variable2(self):
        a = shared(1.0, "a")
        a.default_update = a + 3
        c = a + 10
        f = function([], c, givens={a: (a + 10)})

        assert f() == 21
        assert f() == 34

    def test_duplicate_inputs(self):
        x = lscalar("x")
        with pytest.raises(ValueError, match="is used twice in inputs"):
            function([x, x, x], x)

    def test_update_same(self):
        # There was a bug in CVM, triggered when a shared variable
        # was its own update expression.
        a = shared(1.0, "a")
        b = shared(np.ones((2, 3)), "b")

        # The order of the variables is not determined, so we try
        # both shared variables.
        # TODO: explain the above comment. By "not determined" does
        # this mean "not deterministic"?
        # This test originally wrote the updates using dictionaries,
        # and iterating over the dictionary was not deterministic.
        # Is that all the comment above meant, or is the CVM intended
        # to add extra non-determinism? Or is the CVM meant to
        # deterministically but arbitrarily pick an order for the updates?
        f = function([], [], updates=[(a, a), (b, (2 * b))])
        g = function([], [], updates=[(a, (a * 2)), (b, b)])

        f()
        assert a.get_value(borrow=True).shape == (), a.get_value()
        assert b.get_value(borrow=True).shape == (2, 3), b.get_value()
        g()
        assert a.get_value(borrow=True).shape == (), a.get_value()
        assert b.get_value(borrow=True).shape == (2, 3), b.get_value()

    def test_update_equiv(self):
        # Like test_update_same, but the update expression is simplified until
        # it is found to be equal to the original variable
        a = shared(1.0, "a")
        b = shared(np.ones((2, 3)), "b")

        # See comment in test_update_same about why we try both
        # shared variables.
        f = function([], [], updates=[(a, a), (b, (2 * b - b))])
        g = function([], [], updates=[(a, (a * 2 - a)), (b, b)])

        f()
        assert a.get_value(borrow=True).shape == (), a.get_value()
        assert b.get_value(borrow=True).shape == (2, 3), b.get_value()
        g()
        assert a.get_value(borrow=True).shape == (), a.get_value()
        assert b.get_value(borrow=True).shape == (2, 3), b.get_value()


@pytest.mark.filterwarnings(
    "error",
    r"ignore:^Numba will use object mode to run.*perform method\.:UserWarning",
    r"ignore:Cannot cache compiled function \"numba_funcified_fgraph\".*:numba.NumbaWarning",
)
class TestFunction2:
    @pytest.mark.xfail()
    def test_none(self):
        fn = function([], None)  # ok
        rval = fn()
        assert rval != [], (
            "See #254: Using None as function output leads to [] return value"
        )
        assert rval is None

    def test_empty(self):
        fn = function([], [])  # ok
        assert fn() == []

    def test_extra_inputs(self):
        x, _s = scalars("xs")
        fn = function([x], [x])
        with pytest.raises(TypeError):
            fn(1, 2)

    def test_missing_inputs(self):
        def fn():
            x, _s = scalars("xs")
            function([], [x])

        with pytest.raises(MissingInputError):
            fn()

        def fn():
            x, s = scalars("xs")
            # Ignore unused input s, as it hides the other error
            function([s], [x], on_unused_input="ignore")

        with pytest.raises(MissingInputError):
            fn()

        def fn():
            x, s = scalars("xs")
            function([s], [x])

        with pytest.raises(UnusedInputError):
            fn()

        def fn():
            x, s = scalars("xs")
            # Ignore unused input s, as it hides the other error
            function([s], x, on_unused_input="ignore")

        with pytest.raises(MissingInputError):
            fn()

        def fn():
            x, s = scalars("xs")
            function([s], x)

        with pytest.raises(UnusedInputError):
            fn()

        def fn():
            x, s = scalars("xs")
            # Ignore unused input s, as it hides the other error
            function([s], Out(x), on_unused_input="ignore")

        with pytest.raises(MissingInputError):
            fn()

        def fn():
            x, s = scalars("xs")
            function([s], Out(x))

        with pytest.raises(UnusedInputError):
            fn()

        def fn():
            x, s = scalars("xs")
            function([In(x, update=s + x)], x)

        with pytest.raises(MissingInputError):
            fn()

        def fn():
            x, s = scalars("xs")
            function([In(x, update=((s * s) + x))], x)

        with pytest.raises(MissingInputError):
            fn()

    def test_input_anon_singleton(self):
        x, s = scalars("xs")
        fn = function([s, x], [x + s])
        assert fn(2, 3) == [5]
        # no state
        assert fn(2, 3) == [5]

    def test_input_anon_unpack(self):
        x, s = scalars("xs")
        fn = function([s, x], x + s)
        assert fn(2, 3) == 5

    def test_naming_rule0(self):
        x, s = scalars("xs")
        f = function([x, s], x / s)
        assert f(1, 2) == 0.5
        assert f(2, 1) == 2.0
        assert f(s=2, x=1) == 0.5
        assert f(x=2, s=1) == 2.0
        assert f(2, s=1) == 2.0

        with pytest.raises(TypeError):
            # got multiple values for keyword argument 'x'
            f(2, x=2.0)
        with pytest.raises(TypeError):
            # takes exactly 2 non-keyword arguments (1 given)
            f(x=1)
        with pytest.raises(TypeError):
            # takes exactly 2 non-keyword arguments (0 given)
            f(s=1)

    def test_naming_rule1(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        _x, s = scalars("xs")
        f = function([a, s], a / s)
        assert f(1, 2) == 0.5
        assert f(2, 1) == 2.0
        assert f(2, s=1) == 2.0

        with pytest.raises(TypeError):
            # got unexpected keyword argument 'q'
            f(q=2, s=1)
        with pytest.raises(TypeError):
            # got unexpected keyword argument 'a'
            f(a=2, s=1)

    def test_naming_rule2(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")

        # x's name is ignored because it is followed by anonymous parameter a.
        # Ignore unused input x, as it hides the other error
        f = function([x, a, s], a / s, on_unused_input="ignore")
        assert f(9, 1, 2) == 0.5
        assert f(9, 2, 1) == 2.0
        assert f(9, 2, s=1) == 2.0

        with pytest.raises(TypeError):
            # got unexpected keyword argument 'x'
            f(x=9, a=2, s=1)
        with pytest.raises(TypeError):
            # got unexpected keyword argument 'x'
            f(5.0, x=9)

    def test_same_names(self):
        a, x, s = scalars("xxx")
        # implicit names would cause error.  What do we do?
        f = function([a, x, s], a + x + s)
        assert f(1, 2, 3) == 6
        with pytest.raises(TypeError):
            f(1, 2, x=3)

    def test_weird_names(self):
        a, x, s = scalars("xxx")

        with pytest.raises(TypeError):
            function([In(a, name=[])], [])

        def t():
            f = function(
                [
                    In(a, name={"adsf", ()}),
                    In(x, name=()),
                    In(s, name=scalar()),
                ],
                a + x + s,
            )
            return f

        with pytest.raises(TypeError):
            t()

    def test_trust_input(self):
        x = dvector()
        y = shared(1)
        z = x + y
        f = function([x], z)
        assert f.trust_input is False
        f = function([x], z, trust_input=True)
        assert f.trust_input is True

    def test_copy(self):
        a = scalar()
        x, s = scalars("xs")

        f = function(
            [
                x,
                In(a, name="a"),
                In(s, name="s"),
            ],
            s + a * x,
        )

        g = f.copy()

        assert f.unpack_single == g.unpack_single
        assert f.trust_input == g.trust_input

        assert g._finder[x].storage is not f._finder[x].storage
        assert g._finder[a].storage is not f._finder[a].storage
        assert g._finder[s].storage is not f._finder[s].storage

        assert g._finder[a].value is None and f._finder[a].value is None
        assert g._finder[s].value is None and f._finder[s].value is None

        assert np.array_equal(f(2, 1, 0), g(2, 1, 0))
        assert np.array_equal(f(2, 1, 0), g(2, 1, 0))

    def test_copy_share_memory(self):
        x = fscalar("x")
        # SharedVariable for tests, one of them has update
        y = shared(value=1)
        z = shared(value=2)
        out = tanh((x + y + 2) / (x + z - 0.2) ** 2)

        # Test for different linkers
        for mode in ("FAST_RUN", "FAST_COMPILE"):
            ori = function([x], [out], mode=mode, updates={z: z + 1})
            cpy = ori.copy(share_memory=True)

            # Test if memories shared
            storage_map_ori = ori.vm.storage_map
            storage_map_cpy = cpy.vm.storage_map
            fgraph_cpy = cpy.maker.fgraph

            # Assert intermediate and Constants storages are shared.
            # and output stoarges are not shared
            i_o_variables = fgraph_cpy.inputs + fgraph_cpy.outputs
            ori_storages = storage_map_ori.values()
            l = [
                val
                for key, val in storage_map_cpy.items()
                if key not in i_o_variables or isinstance(key, Constant)
            ]
            for storage in l:
                assert any(storage is s for s in ori_storages)

            # Assert storages of SharedVariable without updates are shared
            for input, here, there in zip(
                ori.maker.expanded_inputs,
                ori.input_storage,
                cpy.input_storage,
                strict=True,
            ):
                assert here.data is there.data

    def test_swap_SharedVariable(self):
        i = iscalar()
        x_list = shared(value=np.random.random((10,)).astype(config.floatX))

        x = scalar("x")
        # SharedVariable for tests, one of them has update
        y = shared(value=1, name="y")
        z = shared(value=2, name="z")
        m = shared(value=0, name="m")

        # SharedVariable to replace
        y_rpl = shared(value=3, name="y_rpl")
        z_rpl = shared(value=4, name="z_rpl")
        swap = {y: y_rpl, z: z_rpl}
        map_SV = {"y_rpl": y_rpl, "z_rpl": z_rpl}

        out = x + y + z + m

        # Test for different linkers
        # for mode in ["FAST_RUN","FAST_COMPILE"]:
        second_time = False
        for mode in ("FAST_RUN", "FAST_COMPILE"):
            ori = function(
                [i],
                [out],
                mode=mode,
                updates=[(z, z + 1), (m, m + 2)],
                givens={x: x_list[i]},
            )
            cpy = ori.copy(swap=swap)

            # run function several times
            ori(1), cpy(1), cpy(2)

            # assert same SharedVariable are update in different function
            if not second_time:
                # m should be updated 3 times
                assert m.get_value() == 6
                # z should be updated once
                assert z.get_value() == 3
                # z_rpl should be updated twice
                assert z_rpl.get_value() == 6
                # y and y_rpl should not be updated
                assert y_rpl.get_value() == 3
                assert y.get_value() == 1
            elif second_time:
                # doule update for sharedvariable
                assert m.get_value() == 12
                assert z.get_value() == 4
                assert z_rpl.get_value() == 8
                assert y_rpl.get_value() == 3

            # test cpy function:
            # 2. SharedVariable is updatable -> values did update(z == 5)
            # 1. sharedvariable is swap ->  Rpl sharedvariables share storage
            names = map_SV.keys()
            for key in cpy.vm.storage_map:
                if key.name in names:
                    assert (
                        map_SV[key.name].container.storage[0]
                        == cpy.vm.storage_map[key][0]
                    )

            second_time = True

    def test_swap_SharedVariable_with_given(self):
        # A special testcase for logistic_sgd.py in Deep Learning Tutorial
        # This test assert that SharedVariable in different function have same storage

        train_x = shared(value=np.random.random((10, 10)).astype(config.floatX))
        test_x = shared(value=np.random.random((10, 10)).astype(config.floatX))

        train_y = shared(value=np.random.random((10, 1)).astype(config.floatX))
        test_y = shared(value=np.random.random((10, 1)).astype(config.floatX))

        i = iscalar("index")
        x = vector("x")
        y = vector("y")
        # this formular has no sense but for a test
        out = (pt_sum(x) - y) ** 2
        train = function(
            [i],
            out,
            givens={x: train_x[i], y: train_y[i]},
            updates={train_x: train_x + 0.1},
        )

        test_def = function([i], out, givens={x: test_x[i], y: test_y[i]})
        test_cpy = train.copy(
            swap={train_x: test_x, train_y: test_y}, delete_updates=True
        )

        for in1, in2 in zip(test_def.maker.inputs, test_cpy.maker.inputs, strict=True):
            assert in1.value is in2.value

    def test_copy_delete_updates(self):
        w = iscalar("w")
        x = fscalar("x")
        # SharedVariable for tests, one of them has update
        y = shared(value=1, name="y")
        z = shared(value=2, name="z")
        out = x + y + z

        # Test for different linkers
        # for mode in ["FAST_RUN","FAST_COMPILE"]:
        # second_time = False
        for mode in ("FAST_RUN", "FAST_COMPILE"):
            ori = function([x], out, mode=mode, updates={z: z * 2})
            cpy = ori.copy(delete_updates=True)

            assert cpy(1) == 4
            assert cpy(1) == 4
            assert cpy(1) == 4

        # Test if unused implicit and explicit inputs from delete_updates
        # are ignored as intended.
        for mode in ("FAST_RUN", "FAST_COMPILE"):
            ori = function([x], x, mode=mode, updates={z: z * 2})
            cpy = ori.copy(delete_updates=True)

            ori = function([x, w], x, mode=mode, updates={z: z + w})
            cpy = ori.copy(delete_updates=True)

    def test_shared_state0(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")

        f = function(
            [
                x,
                In(a, name="a"),
                In(
                    s,
                    value=Container(s, storage=[np.array(0.0)]),
                    update=s + a * x,
                    mutable=True,
                ),
            ],
            s + a * x,
        )
        g = function(
            [
                x,
                In(a, name="a"),
                In(s, value=f._finder[s], update=s - a * x, mutable=True),
            ],
            s + a * x,
        )

        f(1, 2)
        assert f._finder[s].value == 2
        assert g._finder[s].value == 2
        g(1, 2)
        assert f._finder[s].value == 0
        assert g._finder[s].value == 0

    def test_shared_state1(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")

        f = function(
            [
                x,
                In(a, name="a"),
                In(
                    s,
                    value=Container(s, storage=[np.array(0.0)]),
                    update=s + a * x,
                    mutable=True,
                ),
            ],
            s + a * x,
        )
        g = function([x, In(a, name="a"), In(s, value=f._finder[s])], s + a * x)

        f(1, 2)
        assert f._finder[s].value == 2
        assert g._finder[s].value == 2
        f(1, 2)
        g(1, 2)
        assert f._finder[s].value == 4
        assert g._finder[s].value == 4

    def test_shared_state2(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")

        f = function(
            [
                x,
                In(a, name="a"),
                In(
                    s,
                    value=Container(s, storage=[np.array(0.0)]),
                    update=s + a * x,
                    mutable=False,
                ),
            ],
            s + a * x,
        )
        g = function([x, In(a, name="a"), In(s, value=f._finder[s])], s + a * x)

        f(1, 2)
        assert f._finder[s].value == 2
        assert g._finder[s].value == 2
        f(1, 2)
        assert f._finder[s].value == 4
        assert g._finder[s].value == 4
        g(1, 2)  # has no effect on state
        assert f._finder[s].value == 4
        assert g._finder[s].value == 4

    def test_shared_state_not_implicit(self):
        # This test is taken from the documentation in
        # doc/topics/function.txt. If it does not pass anymore and yet the
        # behavior is still intended the doc and the test should both be
        # updated accordingly.
        x, s = scalars("xs")
        inc = function(
            [x, In(s, update=(s + x), value=Container(s, storage=[np.array(10.0)]))], []
        )
        dec = function(
            [x, In(s, update=(s - x), value=inc._finder[s], implicit=False)], []
        )
        assert dec._finder[s].value is inc._finder[s].value
        inc._finder[s].value = 2
        assert dec._finder[s].value == 2
        dec(1)
        assert inc._finder[s].value == 1
        dec(1, 0)
        assert inc._finder[s].value == -1
        assert dec._finder[s].value == -1

    def test_constant_output(self):
        # Test that if the output is a constant, we respect the pytensor memory interface
        f = function([], pt.constant([4]), mode="CVM")
        # print f.maker.fgraph.toposort()
        out = f()
        assert (out == 4).all()
        out[0] = 3
        out2 = f()
        # If the following 2 asserts fail it mean PyTensor broke it's memory contract.
        assert out2 is not out
        assert (out2 == 4).all()

        # Test that if the output is a constant and borrow, we respect the pytensor memory interface
        f = function([], Out(pt.constant([4]), borrow=True), mode="CVM")
        # print f.maker.fgraph.toposort()
        out = f()
        assert (out == 4).all()
        out[0] = 3
        out2 = f()

        if isinstance(get_default_mode(), DebugMode):
            # In DebugMode, we don't implement optimization based on borrow on the output.
            assert (out2 == 4).all()
        else:
            assert out2 is out
            assert (out2 == 3).all()

    def test_borrow_input(self):
        # Tests that the contract for io.In is respected. When borrow=False, it should be
        # impossible for outputs to be aliased to the input variables provided by the user,
        # either through a view-map or a destroy map. New tests should be added in the future
        # when borrow=True is implemented.

        a = dmatrix()
        aval = np.random.random((3, 3))

        # when borrow=False, test that a destroy map cannot alias output to input
        f = function([In(a, borrow=False)], Out(a + 1, borrow=True))
        assert np.all(f(aval) == aval + 1)
        assert not np.may_share_memory(aval, f(aval))

        # when borrow=False, test that a viewmap cannot alias output to input
        f = function([In(a, borrow=False)], Out(a[0, :], borrow=True))
        assert np.all(f(aval) == aval[0, :])
        assert not np.may_share_memory(aval, f(aval))

    def test_borrow_output(self):
        a = dmatrix()
        f = function([a], Out(a, borrow=False))
        o = np.ones((3, 3))
        assert o is not f(o)  # function no longer permits aliasing outputs to inputs

        f = function([a], Out(a * 4, borrow=False))
        o = np.ones((3, 3))
        four = f(o)
        assert np.all(four == 4)
        f(o + 0.1)  # should not clobber the memory used to store four
        assert np.all(four == 4)

        f = function([a], Out(a * 4, borrow=True), mode=Mode("c|py_nogc", "fast_run"))
        o = np.ones((3, 3))
        four = f(o)
        assert np.all(four == 4)
        f(o + 0.1)  # should clobber the memory used to store four
        if config.cxx:
            assert not np.all(four == 4)
        else:
            # The Elemwise.perform method don't reuse memory
            # as some numpy version don't support that correctly.
            assert np.all(four == 4)

    def test_disconnected_input(self):
        a = scalar("a")
        v = vector("v")
        with pytest.raises(UnusedInputError):
            function([a, v], v * 2)

        function([a, v], v * 2, on_unused_input="ignore")

    def test_masked_input(self):
        m = matrix("m")
        mt = m.T
        mt.name = "m.T"
        with pytest.raises(UnusedInputError):
            function([m, mt], mt * 2)
        function([m, mt], mt * 2, on_unused_input="ignore")

    def test_givens_input_var(self):
        # Ensure error is raised when trying to replace an input variable.

        x = scalar("x")
        y = x * 2
        with pytest.raises(RuntimeError):
            function([x], y, givens={x: x + 1})

    def test_free(self):
        # Make test on free() function

        x = vector("x")
        func = function([x], x + 1)
        func.vm.allow_gc = False
        func([1])

        check_list = []
        for key, val in func.vm.storage_map.items():
            if not isinstance(key, Constant):
                check_list.append(val)
        assert any(val[0] for val in check_list)

        func.free()

        for key, val in func.vm.storage_map.items():
            if not isinstance(key, Constant):
                assert val[0] is None

    def test_check_for_aliased_inputs(self):
        b = np.random.random((5, 4))
        s1 = shared(b)
        s2 = shared(b)
        x1 = vector()
        x2 = vector(shape=(3,))
        x3 = vector(shape=(1,))

        # Assert cases we should not check for aliased inputs
        for d in [
            dict(outputs=[s1 + 1]),
            dict(outputs=[s1 + 1, s2 + 3]),
            dict(outputs=[s1 + 1], updates=[(s2, s2 + 3)]),
            dict(inputs=[x1], outputs=[x1 + 1], updates=[(s2, s2 + 3)]),
            dict(
                inputs=[In(x1, mutable=True)], outputs=[x1 + 1], updates=[(s2, s2 + 3)]
            ),
            dict(
                inputs=[In(x2, mutable=True), In(x3, mutable=True)],
                outputs=[x2 + 2, x3 + 3],
            ),
        ]:
            if "inputs" not in d:
                d["inputs"] = []
            f = function(**d)
            assert not f._potential_aliased_input_groups, d

        # Assert cases we should check for aliased inputs
        for d in [
            dict(
                inputs=[In(x1, mutable=True), In(x2, mutable=True)],
                outputs=[x1 + 1, x2 + 2],
                updates=[(s2, s2 + 3)],
            ),
            dict(
                inputs=[In(x1, mutable=True), In(x3, mutable=True)],
                outputs=[x1 + 1, x3 + 3],
                updates=[(s2, s2 + 3)],
            ),
        ]:
            if "inputs" not in d:
                d["inputs"] = []
            f = function(**d)

            assert f._potential_aliased_input_groups, d

    def test_output_list_still_works(self):
        # Test that function works if outputs is a list.
        x = scalar("x")

        f = function([x], outputs=[x * 3, x * 2, x * 4, x])

        result = f(5.0)

        assert result[0] == 15.0
        assert result[1] == 10.0
        assert result[2] == 20.0
        assert result[3] == 5.0

    def test_dprint(self):
        x = pt.scalar("x")
        out = x + 1
        f = function([x], out)
        assert f.dprint(file="str") == debugprint(f, file="str")

    def test_empty_givens_updates(self):
        # Regression test for bug fixed in 8625e03.

        # Empty givens / updates dictionaries were not properly detected before,
        # triggering useless crashes at compile time.
        x = scalar()
        y = x * 2
        function([In(x)], y, givens={})
        function([In(x)], y, updates={})

    def test_rebuild_strict(self):
        # Test fix for error reported at
        # https://groups.google.com/d/topic/theano-users/BRK0UEB72XA/discussion
        w = imatrix()
        x, y = ivectors("x", "y")
        z = x * y
        f = function([w, y], z, givens=[(x, w)], rebuild_strict=False)
        z_val = f(np.ones((3, 5), dtype="int32"), np.arange(5, dtype="int32"))
        assert z_val.ndim == 2
        assert np.all(z_val == np.ones((3, 5)) * np.arange(5))


class TestAliasingRules:
    # 1. PyTensor manages its own memory space, which typically does not overlap
    # with the memory of normal python variables that the user uses.
    #
    # 2. shared variables are allocated in this memory space, as are the
    # temporaries used for Function evaluation.
    #
    # 3. Physically, this managed memory space may be spread across the host,
    # on a GPU device(s), or even on a remote machine.
    #
    # 4. PyTensor assumes that shared variables are never aliased to one another,
    # and tries to make it impossible to accidentally alias them.
    #
    # 5. PyTensor's managed data is constant while PyTensor Functions are not running
    # and pytensor library code is not running.
    #
    # 6. The default behaviour of Function is to return user-space values for
    # outputs, but this can be overridden (borrow=True) for better performance,
    # in which case the returned value may be aliased to managed memory, and
    # potentially invalidated by the next PyTensor Function call or call to pytensor
    # library code.

    def shared(self, x):
        return shared(x)

    def test_shared_constructor_copies(self):
        # shared constructor makes copy
        # (rule #2)
        orig_a = np.zeros((2, 2))
        A = self.shared(orig_a)
        assert not np.may_share_memory(orig_a, data_of(A))

        # rule #2 reading back from pytensor-managed memory
        assert not np.may_share_memory(A.get_value(borrow=False), data_of(A))

    def test_sparse_input_aliasing_affecting_inplace_operations(self):
        # Note: to trigger this bug with pytensor rev 4586:2bc6fc7f218b,
        #        you need to make in inputs mutable (so that inplace
        #        operations are used) and to break the elemwise composition
        #        with some non-elemwise op (here dot)

        x = SparseTensorType("csc", dtype="float64")()
        y = SparseTensorType("csc", dtype="float64")()
        f = function([In(x, mutable=True), In(y, mutable=True)], (x + y) + (x + y))
        # Test 1. If the same variable is given twice

        # Compute bogus values
        m = sp.sparse.csc_matrix(
            np.asarray(
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ],
                dtype="float64",
            )
        )
        bogus_vals = f(m, m)
        # Since we used inplace operation v and m may be corrupted
        # so we need to recreate them

        m = sp.sparse.csc_matrix(
            np.asarray(
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ],
                dtype="float64",
            )
        )
        m_copy = m.copy()
        vals = f(m, m_copy)

        assert np.allclose(vals.todense(), bogus_vals.todense())

    def test_input_aliasing_affecting_inplace_operations(self):
        # Note: to trigger this bug with pytensor rev 4586:2bc6fc7f218b,
        #        you need to make in inputs mutable (so that inplace
        #        operations are used) and to break the elemwise composition
        #        with some non-elemwise op (here dot)
        x = dvector()
        y = dvector()
        m1 = dmatrix()
        m2 = dmatrix()
        f = function(
            [
                In(x, mutable=True),
                In(y, mutable=True),
                In(m1, mutable=True),
                In(m2, mutable=True),
            ],
            pt.dot((x * 2), m1) + pt.dot((y * 3), m2),
        )
        # Test 1. If the same variable is given twice

        # Compute bogus values
        v = np.asarray([1, 2, 3, 4, 5], dtype="float64")
        m = np.asarray(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype="float64",
        )
        bogus_vals = f(v, v, m, m)
        # Since we used inplace operation v and m may be corrupted
        # so we need to recreate them

        v = np.asarray([1, 2, 3, 4, 5], dtype="float64")
        m = np.asarray(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype="float64",
        )
        m_copy = m.copy()
        v_copy = v.copy()
        vals = f(v, v_copy, m, m_copy)

        assert np.allclose(vals, bogus_vals)

    def test_partial_input_aliasing_affecting_inplace_operations(self):
        # Note: to trigger this bug with pytensor rev 4586:2bc6fc7f218b,
        #        you need to make in inputs mutable ( so that inplace
        #        operations are used) and to break the elemwise composition
        #        with some non-elemwise op ( here dot )
        x = dvector()
        y = dvector()
        z = dvector()
        m1 = dmatrix()
        m2 = dmatrix()
        m3 = dmatrix()

        # Test 2. If variables only partial overlap
        #   more exactly we care about the case when we have a,b,c
        #   and a shares memory with b, b shares memory with c, but
        #   c does not share memory with a

        f = function(
            [
                In(x, mutable=True),
                In(y, mutable=True),
                In(z, mutable=True),
                In(m1, mutable=True),
                In(m2, mutable=True),
                In(m3, mutable=True),
            ],
            (pt.dot((x * 2), m1) + pt.dot((y * 3), m2) + pt.dot((z * 4), m3)),
        )

        # Compute bogus values
        v = np.asarray([1, 2, 3, 4, 5], dtype="float64")
        m = np.asarray([[1, 0], [0, 1]], dtype="float64")
        bogus_vals = f(v[:2], v[1:3], v[2:4], m, m, m)
        # Since we used inplace operation v and m may be corrupted
        # so we need to recreate them

        v = np.asarray([1, 2, 3, 4, 5], dtype="float64")
        m = np.asarray([[1, 0], [0, 1]], dtype="float64")
        m_copy1 = m.copy()
        v_copy1 = v.copy()
        m_copy2 = m.copy()
        v_copy2 = v.copy()
        vals = f(v[:2], v_copy1[1:3], v_copy2[2:4], m, m_copy1, m_copy2)

        assert np.allclose(vals, bogus_vals)

    def test_potential_output_aliasing_induced_by_updates(self):
        A = self.shared(np.zeros((2, 2)))
        B = self.shared(np.zeros((2, 2)))
        C = np.zeros((2, 2))
        D = dmatrix()
        DD = D + 5

        f = function([D], [], updates=[(A, D), (B, D)])
        f(C)

        assert not np.may_share_memory(data_of(A), data_of(B))
        f = function([D], [], updates=[(A, D[:]), (B, D)])
        f(C)
        assert not np.may_share_memory(data_of(A), data_of(B))
        f = function([D], [], updates=[(A, (D + 5)), (B, D[:])])
        f(C)
        assert not np.may_share_memory(data_of(A), data_of(B))

        f = function([D], [], updates=[(A, (D + 5)), (B, D)])
        f(C)
        assert not np.may_share_memory(data_of(A), data_of(B))

        f = function([D], DD, updates=[(A, DD[:1]), (B, DD)])
        R = f(C)
        assert not np.may_share_memory(data_of(A), data_of(B))
        assert not np.may_share_memory(R, data_of(B))
        assert not np.may_share_memory(R, data_of(A))

        f = function([D], DD, updates=[(A, DD[:1]), (B, (DD[:1] * 2))])
        R = f(C)
        assert not np.may_share_memory(data_of(A), data_of(B))
        assert not np.may_share_memory(R, data_of(B))
        assert not np.may_share_memory(R, data_of(A))

        f = function([D], (DD * 4), updates=[(A, (DD[:1] * 3)), (B, (DD[:1] * 2))])
        R = f(C)
        assert not np.may_share_memory(data_of(A), data_of(B))
        assert not np.may_share_memory(R, data_of(B))
        assert not np.may_share_memory(R, data_of(A))

        f = function([D], (DD * 4), updates=[(A, (DD[:1] * 3)), (B, (DD[:1] * 3))])
        R = f(C)
        assert not np.may_share_memory(data_of(A), data_of(B))
        assert not np.may_share_memory(R, data_of(B))
        assert not np.may_share_memory(R, data_of(A))

    def test_no_aliasing_0(self):
        # B is a shared variable, A is updated with B's contents
        # we need A to be copied to avoid aliasing
        A = self.shared(np.zeros((2, 2)) + 0.5)
        B = self.shared(np.zeros((2, 2)) - 0.5)
        f = function([], [], updates=[(A, B)])
        f()
        assert not np.may_share_memory(data_of(A), data_of(B))

    def test_no_aliasing_1(self):
        # B is a shared variable, A is updated with B's contents
        # since B is being updated as well, we don't need to copy anything
        # to avoid aliasing shared variables.
        A = self.shared(np.zeros((2, 2)) + 0.5)
        B = self.shared(np.zeros((2, 2)) - 0.5)
        C = dmatrix()
        f = function([C], [], updates=[(A, B), (B, C)])
        z = np.zeros((2, 2))
        f(z)
        assert not np.may_share_memory(data_of(A), data_of(B))
        # PyTensor tries to maintain its own memory space.
        assert not np.may_share_memory(z, data_of(B))
        assert np.all(data_of(B) == z)

    def test_no_aliasing_2(self):
        # B and A take one another's values
        # no copying is necessary since each one is updated.
        orig_a = np.zeros((2, 2)) + 0.5
        orig_b = np.zeros((2, 2)) - 0.5
        A = self.shared(orig_a)
        B = self.shared(orig_b)

        data_of_a = data_of(A)
        data_of_b = data_of(B)

        f = function([], [], updates=[(A, B), (B, A)])
        f()
        # correctness
        assert np.all(data_of(A) == -0.5)
        assert np.all(data_of(B) == +0.5)

        # shared vars may not be aliased
        assert not np.may_share_memory(data_of(A), data_of(B))

        # pytensor should have been smart enough to not make copies
        assert np.may_share_memory(data_of(A), data_of_b)
        assert np.may_share_memory(data_of(B), data_of_a)

    def test_no_aliasing_2b(self):
        # B and A take one another's values
        # no copying is necessary since each one is updated.
        # The twist one `test_no_aliasing_2` is that each shared var is updated
        # with a view of the other one.

        orig_a = np.zeros((2, 2)) + 0.5
        orig_b = np.zeros((2, 2)) - 0.5
        A = self.shared(orig_a)
        B = self.shared(orig_b)

        data_of_a = data_of(A)
        data_of_b = data_of(B)

        f = function([], [], updates=[(A, B[:, ::-1]), (B, A.T)])
        # pytensor.printing.debugprint(f)
        f()
        # correctness (doesn't actually test the view...)
        assert np.all(data_of(A) == -0.5)
        assert np.all(data_of(B) == +0.5)

        # shared vars may not be aliased
        assert not np.may_share_memory(data_of(A), data_of(B))

        # pytensor should have been smart enough to not make copies
        if config.mode not in ["DebugMode", "DEBUG_MODE", "FAST_COMPILE"]:
            # We don't ask DebugMode and FAST_COMPILE not to make copy.
            # We have the right to do so.
            assert np.all(data_of(A) < 5)
            data_of_b += 10
            assert np.all(data_of(A) > 5)
            data_of_b -= 10

            assert np.all(data_of(B) < 5)
            data_of_a += 10
            assert np.all(data_of(B) > 5)
            data_of_a -= 10

            # N.B. may_share_memory is what we mean, but does it work?
            assert np.may_share_memory(data_of(A), data_of_b)
            assert np.may_share_memory(data_of(B), data_of_a)

            # N.B. This pattern could form a memory leak - each shared
            # variable always points to a view, and that view gets
            # further and further from the (e.g. data_of_a) with each
            # call.  The memory leak is in the increasing number of view
            # objects forming a chain to the underlying data.


class SomethingToPickle:
    def __init__(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")
        v = vector("v")

        self.s = s
        self.x = x
        self.v = v

        self.e = a * x + s

        self.f1 = function(
            [
                x,
                In(a, name="a"),
                In(
                    s,
                    value=Container(s, storage=[np.array(0.0)]),
                    update=s + a * x,
                    mutable=True,
                ),
            ],
            s + a * x,
        )

        self.f2 = function(
            [
                x,
                In(a, name="a"),
                In(s, value=self.f1._finder[s], update=s + a * x, mutable=True),
            ],
            s + a * x,
        )


@pytest.mark.filterwarnings(
    "error",
    r"ignore:^Numba will use object mode to run.*perform method\.:UserWarning",
    r"ignore:Cannot cache compiled function \"numba_funcified_fgraph\".*:numba.NumbaWarning",
)
class TestPicklefunction:
    def test_deepcopy(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")

        f = function(
            [
                x,
                In(a, name="a", mutable=True),
                In(s, update=s + a * x, mutable=True),
            ],
            s + a * x,
        )
        try:
            g = copy.deepcopy(f)
        except NotImplementedError as e:
            if e[0].startswith("DebugMode is not picklable"):
                return
            else:
                raise
        # if they both return, assume that they return equivalent things.
        # print [(k, id(k)) for k in f.finder]
        # print [(k, id(k)) for k in g.finder]

        assert g._finder[0].storage is not f._finder[0].storage
        assert g._finder[1].storage is not f._finder[1].storage
        assert g._finder[2].storage is not f._finder[2].storage
        assert x not in g._finder
        # Shared variable is the first input
        assert (
            f._potential_aliased_input_groups
            == g._potential_aliased_input_groups
            == ((1, 2),)
        )
        assert f.name == g.name
        assert f.maker.fgraph.name == g.maker.fgraph.name

        assert g._finder[1].value is None and f._finder[1].value is None
        assert g._finder[2].value is None and f._finder[2].value is None

        assert f(2, 1, 0) == g(2, 1, 0)

    def test_deepcopy_trust_input(self):
        a = dscalar()  # the a is for 'anonymous' (un-named).
        x, s = dscalars("xs")

        f = function(
            [
                x,
                In(a, name="a"),
                In(s, update=s + a * x, mutable=True),
            ],
            s + a * x,
        )
        f.trust_input = True
        try:
            g = copy.deepcopy(f)
        except NotImplementedError as e:
            if e[0].startswith("DebugMode is not picklable"):
                return
            else:
                raise
        assert f.trust_input is g.trust_input
        f(np.array(2.0), np.array(1.0), np.array(0.0))
        with pytest.raises(
            (ValueError, AttributeError, InvalidValueError, NumbaTypingError)
        ):
            f(2.0, np.array(1.0), np.array(0.0))
        g(np.array(2.0), np.array(1.0), np.array(0.0))
        with pytest.raises(
            (ValueError, AttributeError, InvalidValueError, NumbaTypingError)
        ):
            g(2.0, np.array(1.0), np.array(0.0))

    def test_deepcopy_shared_container(self):
        # Ensure that shared containers remain shared after a deep copy.
        a, x = scalars("ax")

        h = function([In(a, value=Container(a, storage=[np.array(0.0)]))], a)
        f = function([x, In(a, value=h._finder[a], implicit=True)], x + a)

        try:
            memo = {}
            ac = copy.deepcopy(a)
            memo.update({id(a): ac})
            hc = copy.deepcopy(h, memo=memo)
            memo.update({id(h): hc})
            fc = copy.deepcopy(f, memo=memo)
        except NotImplementedError as e:
            if e[0].startswith("DebugMode is not picklable"):
                return
            else:
                raise
        h._finder[a].value = 1
        hc._finder[ac].value = 2
        assert f._finder[a].value == 1
        assert fc._finder[ac].value == 2

    def test_pickle(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")

        f = function(
            [
                x,
                In(a, name="a"),
                In(s, update=s + a * x, mutable=True),
            ],
            s + a * x,
        )

        try:
            # Note that here we also test protocol 0 on purpose, since it
            # should work (even though one should not use it).
            g = pickle.loads(pickle.dumps(f, protocol=0))
            g = pickle.loads(pickle.dumps(f, protocol=-1))
        except NotImplementedError as e:
            if e[0].startswith("DebugMode is not picklable"):
                return
            else:
                raise
        # if they both return, assume that they return equivalent things.
        # print [(k, id(k)) for k in f.finder]
        # print [(k, id(k)) for k in g.finder]

        assert g._finder[0].storage is not f._finder[0].storage
        assert g._finder[1].storage is not f._finder[1].storage
        assert g._finder[2].storage is not f._finder[2].storage
        assert x not in g._finder

        assert g._finder[1].value is None and f._finder[1].value is None
        assert g._finder[2].value is None and f._finder[2].value is None
        assert f(2, 1, 0) == g(2, 1, 0)

    def test_optimizations_preserved(self):
        a = dvector()  # the a is for 'anonymous' (un-named).
        x = dvector("x")
        s = dvector("s")
        xm = dmatrix("x")
        sm = dmatrix("s")

        f = function(
            [a, x, s, xm, sm],
            ((a.T.T) * (dot(xm, (sm.T.T.T)) + x).T * (x / x) + s),
        )
        old_default_mode = config.mode
        try:
            try:
                str_f = pickle.dumps(f, protocol=-1)
                config.mode = "NUMBA"
                g = pickle.loads(str_f)
                # print g.maker.mode
                # print compile.mode.default_mode
            except NotImplementedError as e:
                if e[0].startswith("DebugMode is not pickl"):
                    g = "ok"
        finally:
            config.mode = old_default_mode

        if g == "ok":
            return

        assert f.maker is not g.maker
        assert f.maker.fgraph is not g.maker.fgraph
        tf = f.maker.fgraph.toposort()
        tg = f.maker.fgraph.toposort()
        assert len(tf) == len(tg)
        for nf, ng in zip(tf, tg, strict=True):
            assert nf.op == ng.op
            assert len(nf.inputs) == len(ng.inputs)
            assert len(nf.outputs) == len(ng.outputs)
            assert [i.type for i in nf.inputs] == [i.type for i in ng.inputs]
            assert [i.type for i in nf.outputs] == [i.type for i in ng.outputs]

    def test_multiple_functions(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")
        v = vector("v")

        # put in some inputs
        list_of_things = [s, x, v]

        # some derived thing, whose inputs aren't all in the list
        list_of_things.append(a * x + s)

        f1 = function(
            [
                x,
                In(a, name="a"),
                In(
                    s,
                    value=Container(s, storage=[np.array(0.0)]),
                    update=s + a * x,
                    mutable=True,
                ),
            ],
            s + a * x,
        )
        list_of_things.append(f1)

        # now put in a function sharing container with the previous one
        f2 = function(
            [
                x,
                In(a, name="a"),
                In(s, value=f1._finder[s], update=s + a * x, mutable=True),
            ],
            s + a * x,
        )
        list_of_things.append(f2)

        assert isinstance(f2._finder[s].storage, list)
        assert f2._finder[s].storage is f1._finder[s].storage

        # now put in a function with non-scalar
        value = Container(v, storage=[np.asarray([2, 3, 4.0], dtype=config.floatX)])
        f3 = function([x, In(v, value=value)], x + v)
        list_of_things.append(f3)

        # try to pickle the entire things
        try:
            saved_format = pickle.dumps(list_of_things, protocol=-1)
            new_list_of_things = pickle.loads(saved_format)
        except NotImplementedError as e:
            if e[0].startswith("DebugMode is not picklable"):
                return
            else:
                raise

        # now test our recovered new_list_of_things
        # it should be totally unrelated to the original
        # it should be interdependent in the same way as the original

        ol = list_of_things
        nl = new_list_of_things

        for i in range(4):
            assert nl[i] != ol[i]
            assert nl[i].type == ol[i].type
            assert nl[i].type is not ol[i].type

        # see if the implicit input got stored
        assert ol[3].owner.inputs[1] is s
        assert nl[3].owner.inputs[1] is not s
        assert nl[3].owner.inputs[1].type == s.type

        # moving on to the functions...
        for i in range(4, 7):
            assert nl[i] != ol[i]

        # looking at function number 1, input 's'
        assert nl[4]._finder[nl[0]].value is not ol[4]._finder[ol[0]].value
        assert nl[4]._finder[nl[0]].value == ol[4]._finder[ol[0]].value
        assert nl[4](3, 1) == ol[4](3, 1)

        # looking at function number 2, input 's'
        # make sure it's shared with the first function
        assert ol[4]._finder[ol[0]].storage is ol[5]._finder[ol[0]].storage
        assert nl[4]._finder[nl[0]].storage is nl[5]._finder[nl[0]].storage
        assert nl[5](3, 1) == ol[5](3, 1)
        assert nl[4]._finder[nl[0]].value == 6

        assert np.all(nl[6]._finder[nl[2]].value == np.array([2, 3.0, 4]))

    def test_broken_pickle_with_shared(self):
        saves = []

        def pers_save(obj):
            if isinstance(obj, np.ndarray):
                saves.append(obj)
                return len(saves) - 1
            else:
                return None

        def pers_load(id):
            return saves[id]

        def exc_message(e):
            """
            In Python 3, when an exception is reraised it saves the original
            exception in its args, therefore in order to find the actual
            message, we need to unpack arguments recursively.
            """
            msg = e.args[0]
            if isinstance(msg, Exception):
                return exc_message(msg)
            return msg

        b = np.random.random((5, 4))

        x = matrix()
        y = shared(b)

        f = function([x], dot(x, y))

        from io import BytesIO

        fp = BytesIO()
        p = pickle.Pickler(fp, 2)
        p.persistent_id = pers_save
        try:
            p.dump(f)
        except NotImplementedError as e:
            if exc_message(e).startswith("DebugMode is not picklable"):
                return
            else:
                raise
        fp2 = BytesIO(fp.getvalue())
        fp.close()
        p = pickle.Unpickler(fp2)
        p.persistent_load = pers_load
        p.load()
        fp2.close()

    def test_pickle_class_with_functions(self):
        blah = SomethingToPickle()
        assert blah.f2._finder[blah.s].storage is blah.f1._finder[blah.s].storage

        try:
            blah2 = copy.deepcopy(blah)
        except NotImplementedError as e:
            if e[0].startswith("DebugMode is not picklable"):
                return
            else:
                raise

        assert blah2.f2._finder[blah2.s].storage is blah2.f1._finder[blah2.s].storage

        assert blah.f1._finder[blah.s].value == blah2.f1._finder[blah2.s].value

        blah.f2(5, 1)
        assert blah.f1._finder[blah.s].value != blah2.f1._finder[blah2.s].value
