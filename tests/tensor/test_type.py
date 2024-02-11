import os.path as path
from tempfile import mkdtemp

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.tensor.shape import SpecifyShape
from pytensor.tensor.type import (
    TensorType,
    col,
    matrix,
    row,
    scalar,
    tensor,
    tensor3,
    tensor4,
    tensor5,
    tensor6,
    tensor7,
    vector,
)


@pytest.mark.parametrize(
    "dtype, exp_dtype",
    [
        (np.int32, "int32"),
        (np.dtype(np.int32), "int32"),
        ("int32", "int32"),
        ("floatX", config.floatX),
    ],
)
def test_numpy_dtype(dtype, exp_dtype):
    test_type = TensorType(dtype, [])
    assert test_type.dtype == exp_dtype


def test_in_same_class():
    test_type = TensorType(config.floatX, shape=(None, None))
    test_type2 = TensorType(config.floatX, shape=(None, 1))

    assert test_type.in_same_class(test_type)
    assert not test_type.in_same_class(test_type2)

    test_type = TensorType(config.floatX, shape=())
    test_type2 = TensorType(config.floatX, shape=(None,))
    assert not test_type.in_same_class(test_type2)


def test_is_super():
    test_type = TensorType(config.floatX, shape=(None, None))
    test_type2 = TensorType(config.floatX, shape=(None, 1))

    assert test_type.is_super(test_type)
    assert test_type.is_super(test_type2)
    assert not test_type2.is_super(test_type)

    test_type3 = TensorType(config.floatX, shape=(None, None, None))
    assert not test_type3.is_super(test_type)


def test_convert_variable():
    test_type = TensorType(config.floatX, shape=(None, None))
    test_var = test_type()

    test_type2 = TensorType(config.floatX, shape=(1, None))
    test_var2 = test_type2()

    res = test_type.convert_variable(test_var)
    assert res is test_var

    res = test_type.convert_variable(test_var2)
    assert res is test_var2

    res = test_type2.convert_variable(test_var)
    assert res.type == test_type2

    test_type3 = TensorType(config.floatX, shape=(1, None, 1))
    test_var3 = test_type3()

    res = test_type2.convert_variable(test_var3)
    assert res is None

    const_var = pt.as_tensor([[1, 2], [3, 4]], dtype=config.floatX)
    res = test_type.convert_variable(const_var)
    assert res is const_var


def test_convert_variable_mixed_specificity():
    type1 = TensorType(config.floatX, shape=(1, None, 3))
    type2 = TensorType(config.floatX, shape=(None, 5, 3))
    type3 = TensorType(config.floatX, shape=(1, 5, 3))

    test_var1 = type1()
    test_var2 = type2()

    assert type1.convert_variable(test_var2).type == type3
    assert type2.convert_variable(test_var1).type == type3


def test_filter_variable():
    test_type = TensorType(config.floatX, shape=())

    with pytest.raises(TypeError):
        test_type.filter(test_type())

    test_type = TensorType(config.floatX, shape=(1, None))

    with pytest.raises(TypeError):
        test_type.filter(np.empty((0, 1), dtype=config.floatX))

    with pytest.raises(TypeError, match=".*not aligned.*"):
        test_val = np.empty((1, 2), dtype=config.floatX)
        test_val.flags.aligned = False
        test_type.filter(test_val)

    with pytest.raises(ValueError, match="Non-finite"):
        test_type.filter_checks_isfinite = True
        test_type.filter(np.full((1, 2), np.inf, dtype=config.floatX))

    test_type2 = TensorType(config.floatX, shape=(None, None))
    test_var = test_type()
    test_var2 = test_type2()

    res = test_type.filter_variable(test_var, allow_convert=True)
    assert res is test_var

    # Make sure it returns the more specific type
    res = test_type.filter_variable(test_var2, allow_convert=True)
    assert res.type == test_type

    test_type3 = TensorType(config.floatX, shape=(1, 20))
    res = test_type3.filter_variable(test_var, allow_convert=True)
    assert res.type == test_type3


def test_filter_strict():
    test_type = TensorType(config.floatX, shape=())

    with pytest.raises(TypeError):
        test_type.filter(1, strict=True)

    with pytest.raises(TypeError):
        test_type.filter(np.array(1, dtype=int), strict=True)


def test_filter_ndarray_subclass():
    """Make sure `TensorType.filter` can handle NumPy `ndarray` subclasses."""
    test_type = TensorType(config.floatX, shape=(None,))

    class MyNdarray(np.ndarray):
        pass

    test_val = np.array([1.0], dtype=config.floatX).view(MyNdarray)
    assert isinstance(test_val, MyNdarray)

    res = test_type.filter(test_val)
    assert isinstance(res, MyNdarray)
    assert res is test_val


def test_filter_float_subclass():
    """Make sure `TensorType.filter` can handle `float` subclasses."""
    with config.change_flags(floatX="float64"):
        test_type = TensorType("float64", shape=())

        nan = np.array([np.nan], dtype="float64")[0]
        assert isinstance(nan, float) and not isinstance(nan, np.ndarray)

        filtered_nan = test_type.filter(nan)
        assert isinstance(filtered_nan, np.ndarray)

    with config.change_flags(floatX="float32"):
        # Try again, except this time `nan` isn't a `float`
        test_type = TensorType("float32", shape=())

        nan = np.array([np.nan], dtype="float32")[0]
        assert isinstance(nan, np.floating) and not isinstance(nan, np.ndarray)

        filtered_nan = test_type.filter(nan)
        assert isinstance(filtered_nan, np.ndarray)


def test_filter_memmap():
    r"""Make sure `TensorType.filter` can handle NumPy `memmap`\s subclasses."""
    data = np.arange(12, dtype=config.floatX)
    data.resize((3, 4))
    filename = path.join(mkdtemp(), "newfile.dat")
    fp = np.memmap(filename, dtype=config.floatX, mode="w+", shape=(3, 4))

    test_type = TensorType(config.floatX, shape=(None, None))

    res = test_type.filter(fp)
    assert res is fp


def test_may_share_memory():
    a = np.array(2)
    b = np.broadcast_to(a, (2, 3))

    res = TensorType.may_share_memory(a, b)
    assert res

    res = TensorType.may_share_memory(a, None)
    assert res is False


def test_tensor_values_eq_approx():
    # test, inf, -inf and nan equal themselves
    a = np.asarray([-np.inf, -1, 0, 1, np.inf, np.nan])
    with pytest.warns(RuntimeWarning):
        assert TensorType.values_eq_approx(a, a)

    # test inf, -inf don't equal themselves
    b = np.asarray([np.inf, -1, 0, 1, np.inf, np.nan])
    with pytest.warns(RuntimeWarning):
        assert not TensorType.values_eq_approx(a, b)
    b = np.asarray([-np.inf, -1, 0, 1, -np.inf, np.nan])
    with pytest.warns(RuntimeWarning):
        assert not TensorType.values_eq_approx(a, b)

    # test allow_remove_inf
    b = np.asarray([np.inf, -1, 0, 1, 5, np.nan])
    assert TensorType.values_eq_approx(a, b, allow_remove_inf=True)
    b = np.asarray([np.inf, -1, 0, 1, 5, 6])
    assert not TensorType.values_eq_approx(a, b, allow_remove_inf=True)

    # test allow_remove_nan
    b = np.asarray([np.inf, -1, 0, 1, 5, np.nan])
    assert not TensorType.values_eq_approx(a, b, allow_remove_nan=False)
    b = np.asarray([-np.inf, -1, 0, 1, np.inf, 6])
    with pytest.warns(RuntimeWarning):
        assert not TensorType.values_eq_approx(a, b, allow_remove_nan=False)


def test_fixed_shape_basic():
    t1 = TensorType("float64", shape=(1, 1))
    assert t1.shape == (1, 1)
    assert t1.broadcastable == (True, True)

    t1 = TensorType("float64", shape=(0,))
    assert t1.shape == (0,)
    assert t1.broadcastable == (False,)

    t1 = TensorType("float64", shape=(None, None))
    assert t1.shape == (None, None)
    assert t1.broadcastable == (False, False)

    t1 = TensorType("float64", shape=(2, 3))
    assert t1.shape == (2, 3)
    assert t1.broadcastable == (False, False)

    assert str(t1) == "Matrix(float64, shape=(2, 3))"

    t1 = TensorType("float64", shape=(1,))
    assert t1.shape == (1,)
    assert t1.broadcastable == (True,)

    t2 = t1.clone()
    assert t1 is not t2
    assert t1 == t2

    t2 = t1.clone(dtype="float32", shape=(2, 4))
    assert t2.dtype == "float32"
    assert t2.shape == (2, 4)


def test_shape_type_conversion():
    t1 = TensorType("float64", shape=np.array([3], dtype=int))
    assert t1.shape == (3,)
    assert isinstance(t1.shape[0], int)
    assert t1.broadcastable == (False,)
    assert isinstance(t1.broadcastable[0], bool)

    t1 = TensorType("float64", shape=np.array([3], dtype=np.int32))
    assert t1.shape == (3,)
    assert isinstance(t1.shape[0], int)
    assert t1.broadcastable == (False,)
    assert isinstance(t1.broadcastable[0], bool)

    t2 = TensorType("float64", broadcastable=np.array([True, False], dtype="bool"))
    assert t2.shape == (1, None)
    assert isinstance(t2.shape[0], int)
    assert t2.broadcastable == (True, False)
    assert isinstance(t2.broadcastable[0], bool)
    assert isinstance(t2.broadcastable[1], bool)

    with pytest.raises(
        ValueError,
        match="TensorType broadcastable/shape must be a boolean, integer or None",
    ):
        TensorType("float64", shape=("1", "2"))


def test_fixed_shape_clone():
    t1 = TensorType("float64", (1,))

    t2 = t1.clone(dtype="float32", shape=(2, 4))
    assert t2.shape == (2, 4)

    t2 = t1.clone(dtype="float32", shape=(None, None))
    assert t2.shape == (None, None)


def test_fixed_shape_comparisons():
    t1 = TensorType("float64", shape=(1, 1))
    t2 = TensorType("float64", shape=(1, 1))
    assert t1 == t2

    assert t1.is_super(t2)
    assert t2.is_super(t1)

    assert hash(t1) == hash(t2)

    t3 = TensorType("float64", shape=(1, None))
    t4 = TensorType("float64", shape=(1, 2))
    assert t3 != t4

    t1 = TensorType("float64", shape=(1, 1))
    t2 = TensorType("float64", shape=())
    assert t1 != t2


def test_fixed_shape_convert_variable():
    # These are equivalent types
    t1 = TensorType("float64", shape=(1, 1))
    t2 = TensorType("float64", shape=(1, 1))

    assert t1 == t2
    assert t1.shape == t2.shape

    t2_var = t2()
    res = t2.convert_variable(t2_var)
    assert res is t2_var

    res = t1.convert_variable(t2_var)
    assert res is t2_var

    t1_var = t1()
    res = t2.convert_variable(t1_var)
    assert res is t1_var

    t3 = TensorType("float64", shape=(None, 1))
    t3_var = t3()
    res = t2.convert_variable(t3_var)
    assert isinstance(res.owner.op, SpecifyShape)

    t3 = TensorType("float64", shape=(None, None))
    t4 = TensorType("float64", shape=(3, 2))
    t4_var = t4()
    assert t3.shape == (None, None)
    res = t3.convert_variable(t4_var)
    assert res.type == t4
    assert res.type.shape == (3, 2)


def test_deprecated_kwargs():
    with pytest.warns(DeprecationWarning, match=".*broadcastable.*"):
        res = TensorType("float64", broadcastable=(True, False))

    assert res.shape == (1, None)

    with pytest.warns(DeprecationWarning, match=".*broadcastable.*"):
        new_res = res.clone(broadcastable=(False, True))

    assert new_res.shape == (None, 1)


def test_tensor_creator_helper():
    res = tensor(shape=(5, None))
    assert res.type == TensorType(config.floatX, shape=(5, None))
    assert res.name is None

    res = tensor(dtype="int64", shape=(5, None), name="custom")
    assert res.type == TensorType(dtype="int64", shape=(5, None))
    assert res.name == "custom"

    # Test with positional name argument
    res = tensor("custom", dtype="int64", shape=(5, None))
    assert res.type == TensorType(dtype="int64", shape=(5, None))
    assert res.name == "custom"

    with pytest.warns(
        DeprecationWarning, match="The `broadcastable` keyword is deprecated"
    ):
        res = tensor(dtype="int64", broadcastable=(True, False), name="custom")
        assert res.type == TensorType("int64", shape=(1, None))
        assert res.name == "custom"


@pytest.mark.parametrize("dtype", ("floatX", "float64", bool, np.int64))
def test_tensor_creator_dtype_catch(dtype):
    with pytest.raises(
        ValueError,
        match="This name looks like a dtype, which you should pass as a keyword argument only",
    ):
        tensor(dtype, shape=(None,))

    # This should work
    assert tensor(dtype=dtype, shape=(None,))


def test_tensor_creator_ignores_rare_dtype_name():
    # This could be a dtype, but we assume it's a name
    assert tensor("a", shape=(None,)).type.dtype == config.floatX


def test_scalar_creator_helper():
    default = scalar()
    assert default.type.dtype == config.floatX
    assert default.type.ndim == 0
    assert default.type.shape == ()
    assert default.name is None

    custom = scalar(name="custom", dtype="int64")
    assert custom.dtype == "int64"
    assert custom.type.ndim == 0
    assert custom.type.shape == ()


@pytest.mark.parametrize(
    "helper, ndims",
    [
        (vector, 1),
        (matrix, 2),
        (row, 2),
        (col, 2),
        (tensor3, 3),
        (tensor4, 4),
        (tensor5, 5),
        (tensor6, 6),
        (tensor7, 7),
    ],
)
def test_tensor_creator_helpers(helper, ndims):
    if helper is row:
        default_shape = (1, None)
        custom_shape = (1, 5)
    elif helper is col:
        default_shape = (None, 1)
        custom_shape = (5, 1)
    else:
        default_shape = (None,) * ndims
        custom_shape = tuple(range(ndims))

    default = helper()
    assert default.type.dtype == config.floatX
    assert default.type.ndim == ndims
    assert default.type.shape == default_shape
    assert default.name is None

    assert helper(shape=default_shape).type == default.type

    custom = helper(name="custom", dtype="int64", shape=custom_shape)
    assert custom.type.dtype == "int64"
    assert custom.type.ndim == ndims
    assert custom.type.shape == custom_shape
    assert custom.name == "custom"

    with pytest.raises(TypeError, match="Shape must be a tuple"):
        helper(shape=list(default_shape))

    with pytest.raises(ValueError, match=f"Shape must be a tuple of length {ndims}"):
        helper(shape=(None, *default_shape))

    with pytest.raises(TypeError, match="Shape entries must be None or integer"):
        helper(shape=(1.0,) * ndims)


@pytest.mark.parametrize("helper", (row, col))
def test_row_matrix_creator_helpers(helper):
    if helper is row:
        match = "The first dimension of a `row` must have shape 1, got 2"
    else:
        match = "The second dimension of a `col` must have shape 1, got 5"
    with pytest.raises(ValueError, match=match):
        helper(shape=(2, 5))
