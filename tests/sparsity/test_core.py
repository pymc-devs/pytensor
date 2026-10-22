import numpy as np
import scipy.sparse

import pytensor.sparse as ps
import pytensor.tensor as pt
from pytensor import function, shared


def test_fallback_is_input_level_reachability():
    x = pt.vector("x")
    y = pt.matrix("y")
    held_fixed = pt.vector("held_fixed")
    held_fixed_shared = shared(np.ones(4), name="held_fixed_shared")

    # No rule in the plan covers sort, so its block stays dense as rules are added.
    outputs = [
        pt.sort(x) * held_fixed.sum() * held_fixed_shared.sum(),
        (y * 2).sum() + pt.ones(x.shape[0]).sum(),
        pt.exp(pt.constant(np.array([1.0, 2.0]))),
    ]
    pattern_function = function(
        [x, y, held_fixed], pt.jacobian_sparsity(outputs, [x, y])
    )

    for x_size, y_shape in ((3, (2, 2)), (2, (1, 3))):
        y_size = int(np.prod(y_shape))
        expected = np.block(
            [
                [np.ones((x_size, x_size)), np.zeros((x_size, y_size))],
                [np.zeros((1, x_size)), np.ones((1, y_size))],
                [np.zeros((2, x_size)), np.zeros((2, y_size))],
            ]
        )
        pattern = pattern_function(np.ones(x_size), np.ones(y_shape), np.ones(5))
        np.testing.assert_array_equal(pattern, expected.astype(bool))


def test_intermediate_and_empty_wrt():
    x = pt.vector("x")
    intermediate = pt.exp(x)

    pattern_function = function(
        [x],
        [
            pt.jacobian_sparsity(pt.sort(intermediate) + x, intermediate),
            pt.jacobian_sparsity(intermediate, []),
        ],
    )
    through_intermediate, empty = pattern_function(np.ones(3))

    np.testing.assert_array_equal(through_intermediate, np.ones((3, 3), dtype=bool))
    assert empty.shape == (3, 0)


def test_sparse_constant_in_graph():
    x = pt.vector("x")
    sparse_identity = ps.as_sparse_variable(scipy.sparse.csr_matrix(np.eye(3)))
    out = ps.structured_dot(sparse_identity, x[:, None])

    pattern = function([x], pt.jacobian_sparsity(out, x))(np.ones(3))

    np.testing.assert_array_equal(pattern, np.ones((3, 3), dtype=bool))
