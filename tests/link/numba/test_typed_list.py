import numpy as np

from pytensor.tensor import matrix
from pytensor.typed_list import make_list
from tests.link.numba.test_basic import compare_numba_and_py


def test_list_basic_ops():
    x = matrix("x", shape=(3, None), dtype="int64")
    l = make_list([x[0], x[2]])

    x_test = np.arange(12).reshape(3, 4)
    compare_numba_and_py([x], [l, l.length()], [x_test])

    # Test nested list
    ll = make_list([l, l, l])
    compare_numba_and_py([x], [ll, ll.length()], [x_test])


def test_make_list_index_ops():
    x = matrix("x", shape=(3, None), dtype="int64")
    l = make_list([x[0], x[2]])

    x_test = np.arange(12).reshape(3, 4)
    compare_numba_and_py([x], [l[-1], l[:-1], l.reverse()], [x_test])


def test_make_list_extend_ops():
    x = matrix("x", shape=(3, None), dtype="int64")
    l = make_list([x[0], x[2]])

    x_test = np.arange(12).reshape(3, 4)
    compare_numba_and_py(
        [x], [l.append(x[1]), l.extend(l), l.insert(0, x[1])], [x_test]
    )


def test_make_list_find_ops():
    # Remove requires to first find it
    x = matrix("x", shape=(3, None), dtype="int64")
    y = x[0].type("y")
    l = make_list([x[0], x[2], x[0], x[2]])

    x_test = np.arange(12).reshape(3, 4)
    test_y = x_test[2]
    compare_numba_and_py([x, y], [l.ind(y), l.count(y), l.remove(y)], [x_test, test_y])
