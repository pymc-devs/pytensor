import numpy as np

from pytensor.configdefaults import config
from pytensor.scalar.basic import (
    IntDiv,
    ScalarType,
    TrueDiv,
    complex64,
    float32,
    float64,
    int8,
    int32,
)


def test_numpy_dtype():
    test_type = ScalarType(np.int32)
    assert test_type.dtype == "int32"


def test_div_types():
    a = int8()
    b = int32()
    c = complex64()
    d = float64()
    f = float32()

    assert isinstance((a // b).owner.op, IntDiv)
    assert isinstance((b // a).owner.op, IntDiv)
    assert isinstance((b / d).owner.op, TrueDiv)
    assert isinstance((b / f).owner.op, TrueDiv)
    assert isinstance((f / a).owner.op, TrueDiv)
    assert isinstance((d / b).owner.op, TrueDiv)
    assert isinstance((d / f).owner.op, TrueDiv)
    assert isinstance((f / c).owner.op, TrueDiv)
    assert isinstance((a / c).owner.op, TrueDiv)


def test_filter_float_subclass():
    """Make sure `ScalarType.filter` can handle `float` subclasses."""
    with config.change_flags(floatX="float64"):
        test_type = ScalarType("float64")

        nan = np.array([np.nan], dtype="float64")[0]
        assert isinstance(nan, float)

        filtered_nan = test_type.filter(nan)
        assert isinstance(filtered_nan, float)

    with config.change_flags(floatX="float32"):
        # Try again, except this time `nan` isn't a `float`
        test_type = ScalarType("float32")

        nan = np.array([np.nan], dtype="float32")[0]
        assert isinstance(nan, np.floating)

        filtered_nan = test_type.filter(nan)
        assert isinstance(filtered_nan, np.floating)

    with config.change_flags(floatX="float64"):
        filtered_nan = test_type.filter(nan)
        assert isinstance(filtered_nan, np.floating)


def test_clone():
    st = ScalarType("int64")
    assert st == st.clone()
    assert st.clone("float64").dtype == "float64"
