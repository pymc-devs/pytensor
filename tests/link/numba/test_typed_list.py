import numpy as np

from pytensor import In
from pytensor.tensor import as_tensor, lscalar, matrix
from pytensor.typed_list import TypedListType, make_list
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


def test_inplace_ops():
    int64_list = TypedListType(lscalar)
    ls = [int64_list(f"list[{i}]") for i in range(5)]
    to_extend = lscalar("to_extend")

    ls_test = [np.arange(3, dtype="int64").tolist() for _ in range(5)]
    to_extend_test = np.array(99, dtype="int64")

    def as_lscalar(x):
        return as_tensor(x, ndim=0, dtype="int64")

    fn, _ = compare_numba_and_py(
        [*(In(l, mutable=True) for l in ls), to_extend],
        [
            ls[0].reverse(),
            ls[1].append(as_lscalar(99)),
            # This fails because it gets constant folded
            # ls_to_extend = make_list([as_lscalar(99), as_lscalar(100)])
            ls[2].extend(make_list([to_extend, to_extend + 1])),
            ls[3].insert(as_lscalar(1), as_lscalar(99)),
            ls[4].remove(as_lscalar(2)),
        ],
        [*ls_test, to_extend_test],
        numba_mode="NUMBA",  # So it triggers inplace
    )
    for out in fn.maker.fgraph.outputs:
        assert out.owner.op.destroy_map
