import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config, shared
from pytensor.compile.function import function
from pytensor.compile.mode import Mode
from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import EquilibriumGraphRewriter
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.tensor import constant
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.random.basic import (
    categorical,
    dirichlet,
    multinomial,
    multivariate_normal,
    normal,
    poisson,
    uniform,
)
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.rewriting import (
    local_dimshuffle_rv_lift,
    local_rv_size_lift,
    local_subtensor_rv_lift,
)
from pytensor.tensor.rewriting.shape import ShapeFeature, ShapeOptimizer
from pytensor.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1, Subtensor
from pytensor.tensor.type import iscalar, vector


no_mode = Mode("py", RewriteDatabaseQuery(include=[], exclude=[]))


def apply_local_rewrite_to_rv(
    rewrite, op_fn, dist_op, dist_params, size, rng, name=None
):
    dist_params_pt = []
    for i, p in enumerate(dist_params):
        p_pt = pt.as_tensor(p).type(f"p_{i}")
        p_pt.tag.test_value = p
        dist_params_pt.append(p_pt)

    size_pt = []
    for s in size:
        # To test DimShuffle with dropping dims we need that size dimension to be constant
        if s == 1:
            s_pt = constant(np.array(1, dtype="int32"))
        else:
            s_pt = iscalar()
        s_pt.tag.test_value = s
        size_pt.append(s_pt)

    dist_st = op_fn(dist_op(*dist_params_pt, size=size_pt, rng=rng, name=name))

    f_inputs = [
        p for p in dist_params_pt + size_pt if not isinstance(p, (slice, Constant))
    ]

    mode = Mode(
        "py", EquilibriumGraphRewriter([ShapeOptimizer(), rewrite], max_use_ratio=100)
    )

    f_rewritten = function(
        f_inputs,
        dist_st,
        mode=mode,
    )

    (new_out,) = f_rewritten.maker.fgraph.outputs

    return new_out, f_inputs, dist_st, f_rewritten


def test_inplace_rewrites():
    out = normal(0, 1)
    out.owner.inputs[0].default_update = out.owner.outputs[0]

    assert out.owner.op.inplace is False

    f = function(
        [],
        out,
        mode="FAST_RUN",
    )

    (new_out, new_rng) = f.maker.fgraph.outputs
    assert new_out.type == out.type
    assert isinstance(new_out.owner.op, type(out.owner.op))
    assert new_out.owner.op.inplace is True
    assert all(
        np.array_equal(a.data, b.data)
        for a, b in zip(new_out.owner.inputs[2:], out.owner.inputs[2:])
    )
    assert np.array_equal(new_out.owner.inputs[1].data, [])


def test_inplace_rewrites_extra_props():
    class Test(RandomVariable):
        name = "test"
        ndim_supp = 0
        ndims_params = [0]
        __props__ = ("name", "ndim_supp", "ndims_params", "dtype", "inplace", "extra")
        dtype = "floatX"
        _print_name = ("Test", "\\operatorname{Test}")

        def __init__(self, extra, *args, **kwargs):
            self.extra = extra
            super().__init__(*args, **kwargs)

        def make_node(self, rng, size, dtype, sigma):
            return super().make_node(rng, size, dtype, sigma)

        def rng_fn(self, rng, sigma, size):
            return rng.normal(scale=sigma, size=size)

    out = Test(extra="some value")(1)
    out.owner.inputs[0].default_update = out.owner.outputs[0]

    assert out.owner.op.inplace is False

    f = function(
        [],
        out,
        mode="FAST_RUN",
    )

    (new_out, new_rng) = f.maker.fgraph.outputs
    assert new_out.type == out.type
    assert isinstance(new_out.owner.op, type(out.owner.op))
    assert new_out.owner.op.inplace is True
    assert new_out.owner.op.extra == out.owner.op.extra
    assert all(
        np.array_equal(a.data, b.data)
        for a, b in zip(new_out.owner.inputs[2:], out.owner.inputs[2:])
    )
    assert np.array_equal(new_out.owner.inputs[1].data, [])


@config.change_flags(compute_test_value="raise")
@pytest.mark.parametrize(
    "dist_op, dist_params, size",
    [
        (
            normal,
            [
                np.array(1.0, dtype=config.floatX),
                np.array(5.0, dtype=config.floatX),
            ],
            [],
        ),
        (
            normal,
            [
                np.array([0.0, 1.0], dtype=config.floatX),
                np.array(5.0, dtype=config.floatX),
            ],
            [],
        ),
        (
            normal,
            [
                np.array([0.0, 1.0], dtype=config.floatX),
                np.array(5.0, dtype=config.floatX),
            ],
            [3, 2],
        ),
        (
            multivariate_normal,
            [
                np.array([[0], [10], [100]], dtype=config.floatX),
                np.diag(np.array([1e-6], dtype=config.floatX)),
            ],
            [2, 3, 3],
        ),
        (
            dirichlet,
            [np.array([[100, 1, 1], [1, 100, 1], [1, 1, 100]], dtype=config.floatX)],
            [2, 3, 3],
        ),
        (
            multinomial,
            [
                np.array([10, 20], dtype="int64"),
                np.array([[0.999, 0.001], [0.001, 0.999]], dtype=config.floatX),
            ],
            [3, 2],
        ),
    ],
)
def test_local_rv_size_lift(dist_op, dist_params, size):
    rng = shared(np.random.default_rng(1233532), borrow=False)

    new_out, f_inputs, dist_st, f_rewritten = apply_local_rewrite_to_rv(
        local_rv_size_lift,
        lambda rv: rv,
        dist_op,
        dist_params,
        size,
        rng,
    )

    assert pt.get_vector_length(new_out.owner.inputs[1]) == 0


@pytest.mark.parametrize(
    "ds_order, lifted, dist_op, dist_params, size, rtol",
    [
        (
            ("x", 0),
            True,
            normal,
            (
                np.array([0.0, -100.0], dtype=np.float64),
                np.array(1e-6, dtype=np.float64),
            ),
            (),
            1e-7,
        ),
        (
            ("x",),
            True,
            normal,
            (
                np.array(-10.0, dtype=np.float64),
                np.array(1e-6, dtype=np.float64),
            ),
            (),
            1e-7,
        ),
        (
            ("x", "x", "x"),
            True,
            normal,
            (
                np.array(-10.0, dtype=np.float64),
                np.array(1e-6, dtype=np.float64),
            ),
            (),
            1e-7,
        ),
        (
            (1, 0, 2),
            True,
            normal,
            (
                np.arange(2 * 2 * 2).reshape((2, 2, 2)).astype(config.floatX),
                np.array(1e-6).astype(config.floatX),
            ),
            (),
            1e-3,
        ),
        (
            (0, 1, 2),
            True,
            normal,
            (np.array(0).astype(config.floatX), np.array(1e-6).astype(config.floatX)),
            (2, 1, 2),
            1e-3,
        ),
        (
            (0, 2, 1),
            True,
            normal,
            (np.array(0).astype(config.floatX), np.array(1e-6).astype(config.floatX)),
            (2, 1, 2),
            1e-3,
        ),
        (
            (1, 0, 2),
            True,
            normal,
            (np.array(0).astype(config.floatX), np.array(1e-6).astype(config.floatX)),
            (2, 1, 2),
            1e-3,
        ),
        (
            (0, 2, 1),
            True,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
            1e-3,
        ),
        (
            ("x", 0, 2, 1, "x"),
            True,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
            1e-3,
        ),
        (
            ("x", 0, "x", 2, "x", 1, "x"),
            True,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
            1e-3,
        ),
        (
            ("x", 0, 2, 1, "x"),
            True,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
            1e-3,
        ),
        (
            ("x", 1, 0, 2, "x"),
            True,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
            1e-3,
        ),
        # Only one distribution parameter
        (
            (0, 2, 1),
            True,
            poisson,
            (np.array([[10, 50], [100, 150]], dtype=config.floatX),),
            (3, 2, 2),
            1,
        ),
        # Supported multi-dimensional cases
        (
            (1, 0, 2),
            True,
            multivariate_normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.eye(2).astype(config.floatX) * 1e-6,
            ),
            (3, 2),
            1e-3,
        ),
        (
            (1, 0, "x", 2),
            True,
            multivariate_normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.eye(2).astype(config.floatX) * 1e-6,
            ),
            (3, 2),
            1e-3,
        ),
        # Not supported multi-dimensional cases where dimshuffle affects the support dimensionality
        (
            (0, 2, 1),
            False,
            multivariate_normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.eye(2).astype(config.floatX) * 1e-6,
            ),
            (3, 2),
            1e-3,
        ),
        (
            (0, 1, 2, "x"),
            False,
            multivariate_normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.eye(2).astype(config.floatX) * 1e-6,
            ),
            (3, 2),
            1e-3,
        ),
        pytest.param(
            (1,),
            True,
            normal,
            (0, 1),
            (1, 2),
            1e-3,
            marks=pytest.mark.xfail(reason="Dropping dimensions not supported yet"),
        ),
        pytest.param(
            (1,),
            True,
            normal,
            ([[0, 0]], 1),
            (1, 2),
            1e-3,
            marks=pytest.mark.xfail(reason="Dropping dimensions not supported yet"),
        ),
    ],
)
@config.change_flags(compute_test_value_opt="raise", compute_test_value="raise")
def test_DimShuffle_lift(ds_order, lifted, dist_op, dist_params, size, rtol):
    rng = shared(np.random.default_rng(1233532), borrow=False)

    new_out, f_inputs, dist_st, f_rewritten = apply_local_rewrite_to_rv(
        local_dimshuffle_rv_lift,
        lambda rv: rv.dimshuffle(ds_order),
        dist_op,
        dist_params,
        size,
        rng,
    )

    if lifted:
        assert new_out.owner.op == dist_op
        assert all(
            isinstance(i.owner.op, DimShuffle)
            for i in new_out.owner.inputs[3:]
            if i.owner
        )
    else:
        assert isinstance(new_out.owner.op, DimShuffle)
        return

    f_base = function(
        f_inputs,
        dist_st,
        mode=no_mode,
    )

    arg_values = [p.get_test_value() for p in f_inputs]
    res_base = f_base(*arg_values)
    res_rewritten = f_rewritten(*arg_values)

    np.testing.assert_allclose(res_base, res_rewritten, rtol=rtol)


def rand_bool_mask(shape, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.binomial(n=1, p=0.5, size=shape).astype(bool)


@pytest.mark.parametrize(
    "indices, lifted, dist_op, dist_params, size",
    [
        # 0
        (
            # `size`-less simple integer indexing
            (slice(None), 2),
            True,
            normal,
            (
                np.arange(30, dtype=config.floatX).reshape(3, 5, 2),
                np.full((1, 5, 1), 1e-6),
            ),
            (),
        ),
        (
            # `size`-only slice
            (2, -1),
            True,
            uniform,
            (
                np.array(0.9 - 1e-5, dtype=config.floatX),
                np.array(0.9, dtype=config.floatX),
            ),
            (5, 2),
        ),
        (
            # `size`-less slice
            (slice(None), slice(4, -6, -1), slice(1, None)),
            True,
            normal,
            (
                np.arange(30, dtype=config.floatX).reshape(3, 5, 2),
                np.full((1, 5, 1), 1e-6),
            ),
            (),
        ),
        (
            # `size`-only slice
            (slice(4, -6, -1),),
            True,
            uniform,
            (
                np.array(0.9 - 1e-5, dtype=config.floatX),
                np.array(0.9, dtype=config.floatX),
            ),
            (5, 2),
        ),
        (
            # `size`-less advanced boolean indexing
            (np.r_[True, False, False, True],),
            True,
            uniform,
            (
                (0.1 - 1e-5) * np.arange(4).astype(dtype=config.floatX),
                0.1 * np.arange(4).astype(dtype=config.floatX),
            ),
            (),
        ),
        # 5
        (
            # `size`-only advanced boolean indexing
            (np.r_[True, False, False, True],),
            True,
            uniform,
            (
                np.array(0.9 - 1e-5, dtype=config.floatX),
                np.array(0.9, dtype=config.floatX),
            ),
            (4,),
        ),
        (
            # Advanced integer indexing
            (slice(1, None), [0, 2]),
            False,  # Could have duplicates
            normal,
            (
                np.array([1, 10, 100], dtype=config.floatX),
                np.array([1e-5, 2e-5, 3e-5], dtype=config.floatX),
            ),
            (4, 3),
        ),
        (
            # Advanced integer indexing
            (np.array([1]), 0),
            False,  # We don't support expand_dims
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
        ),
        (
            # Advanced integer-boolean indexing
            (0, np.r_[True, False]),
            True,
            normal,
            (
                np.array([[1, 2], [3, 4]], dtype=config.floatX),
                np.array([1e-6], dtype=config.floatX),
            ),
            (3, 2, 2),
        ),
        (
            # Advanced non-consecutive integer-boolean indexing
            (slice(None), 0, slice(None), np.r_[True, False]),
            True,
            normal,
            (
                np.array([[1, 2], [3, 4]], dtype=config.floatX),
                np.array([[1e-6]], dtype=config.floatX),
            ),
            (7, 3, 2, 2),
        ),
        # 10
        (
            # Univariate distribution with core-vector parameters
            (1,),
            True,
            categorical,
            (np.array([0.0, 0.0, 1.0], dtype=config.floatX),),
            (4,),
        ),
        (
            # Univariate distribution with core-vector parameters
            (np.array([True, False, True, True]),),
            True,
            categorical,
            (np.array([0.0, 0.0, 1.0], dtype=config.floatX),),
            (4,),
        ),
        (
            # Univariate distribution with core-vector parameters
            (np.array([True, False, True]),),
            True,
            categorical,
            (
                np.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=config.floatX,
                ),
            ),
            (),
        ),
        (
            # Univariate distribution with core-vector parameters
            (slice(None), np.array([True, False, True])),
            True,
            categorical,
            (
                np.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=config.floatX,
                ),
            ),
            (4, 3),
        ),
        (
            # Boolean indexing where output is empty
            (np.array([False, False]),),
            True,
            normal,
            (np.array([[1.0, 0.0, 0.0]], dtype=config.floatX),),
            (2, 3),
        ),
        # 15
        (
            # Boolean indexing where output is empty
            (np.array([False, False]), slice(1, None)),
            True,
            categorical,
            (
                np.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=config.floatX,
                ),
            ),
            (2, 3),
        ),
        (
            # Empty-slice
            (slice(None), slice(10, None), slice(1, None)),
            True,
            normal,
            (
                np.arange(30).reshape(2, 3, 5),
                np.full((1, 5), 1e-6),
            ),
            (2, 3, 5),
        ),
        (
            # Multidimensional boolean indexing
            (rand_bool_mask((5, 3, 2)),),
            True,
            normal,
            (
                np.arange(30).reshape(5, 3, 2),
                1e-6,
            ),
            (),
        ),
        (
            # Multidimensional boolean indexing
            (rand_bool_mask((5, 3)),),
            True,
            normal,
            (
                np.arange(30).reshape(5, 3, 2),
                1e-6,
            ),
            (),
        ),
        (
            # Multidimensional boolean indexing
            (rand_bool_mask((5, 3)), slice(None)),
            True,
            normal,
            (
                np.arange(30).reshape(5, 3, 2),
                1e-6,
            ),
            (),
        ),
        # 20
        (
            # Multidimensional boolean indexing
            (slice(None), rand_bool_mask((3, 2))),
            True,
            normal,
            (
                np.arange(30).reshape(5, 3, 2),
                1e-6,
            ),
            (),
        ),
        (
            # Multidimensional boolean indexing
            (rand_bool_mask((5, 3)),),
            True,
            normal,
            (
                np.arange(3).reshape(1, 3, 1),
                np.full((5, 1, 2), 1e-6),
            ),
            (5, 3, 2),
        ),
        (
            # Multidimensional boolean indexing
            (
                np.array([True, False, True, False, False]),
                slice(None),
                (np.array([True, True])),
            ),
            True,
            normal,
            (
                np.arange(30).reshape(5, 3, 2),
                1e-6,
            ),
            (),
        ),
        (
            # Multidimensional boolean indexing,
            # requires runtime broadcasting of the zeros arrays
            (
                np.array([True, False, True, False, False]),  # nonzero().shape == (2,)
                slice(None),
                (np.array([True, False])),  # nonzero().shape == (1,)
            ),
            True,
            normal,
            (
                np.arange(30).reshape(5, 3, 2),
                1e-6,
            ),
            (),
        ),
        (
            # Multivariate distribution: indexing dips into core dimension
            (1, 0),
            False,
            multivariate_normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.eye(2).astype(config.floatX) * 1e-6,
            ),
            (),
        ),
        # 25
        (
            # Multivariate distribution: indexing dips into core dimension
            (rand_bool_mask((2, 2)),),
            False,
            multivariate_normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.eye(2).astype(config.floatX) * 1e-6,
            ),
            (),
        ),
        (
            # Multivariate distribution: advanced integer indexing
            (np.array([0, 0]),),
            False,  # Could have duplicates (it has in this case)!
            multivariate_normal,
            (
                np.array(
                    [[-100, -125, -150], [0, 0, 0], [200, 225, 250]],
                    dtype=config.floatX,
                ),
                np.eye(3, dtype=config.floatX) * 1e-6,
            ),
            (),
        ),
        (
            # Multivariate distribution: dummy slice "dips" into core dimension
            (np.array([True, False, True]), slice(None)),
            True,
            multivariate_normal,
            (
                np.array([200, 250], dtype=config.floatX),
                # Second covariance is invalid, to test it is not chosen
                np.dstack([np.eye(2), np.eye(2) * 0, np.eye(2)]).T.astype(config.floatX)
                * 1e-6,
            ),
            (3,),
        ),
        (
            # Multivariate distribution
            (0, slice(1, None), rand_bool_mask((4, 3))),
            True,
            multivariate_normal,
            (
                np.arange(4 * 3 * 2).reshape(4, 3, 2).astype(dtype=config.floatX),
                np.eye(2) * 1e-6,
            ),
            (5, 3, 4, 3),
        ),
    ],
)
@config.change_flags(compute_test_value_opt="raise", compute_test_value="raise")
def test_Subtensor_lift(indices, lifted, dist_op, dist_params, size):
    from pytensor.tensor.subtensor import as_index_constant

    rng = shared(np.random.default_rng(1233532), borrow=False)

    indices_pt = ()
    for i in indices:
        i_pt = as_index_constant(i)
        if not isinstance(i_pt, slice):
            i_pt.tag.test_value = i
        indices_pt += (i_pt,)

    new_out, f_inputs, dist_st, f_rewritten = apply_local_rewrite_to_rv(
        local_subtensor_rv_lift,
        lambda rv: rv[indices_pt],
        dist_op,
        dist_params,
        size,
        rng,
    )

    if lifted:
        assert isinstance(new_out.owner.op, RandomVariable)
        assert all(
            isinstance(i.owner.op, (AdvancedSubtensor, AdvancedSubtensor1, Subtensor))
            for i in new_out.owner.inputs[3:]
            if i.owner
        )
    else:
        assert isinstance(
            new_out.owner.op, (AdvancedSubtensor, AdvancedSubtensor1, Subtensor)
        )
        return

    f_base = function(
        f_inputs,
        dist_st,
        mode=no_mode,
    )

    arg_values = [p.get_test_value() for p in f_inputs]
    res_base = f_base(*arg_values)
    res_rewritten = f_rewritten(*arg_values)

    np.testing.assert_allclose(res_base, res_rewritten, rtol=1e-3, atol=1e-2)


def test_Subtensor_lift_restrictions():
    rng = shared(np.random.default_rng(1233532), borrow=False)

    std = vector("std")
    std.tag.test_value = np.array([1e-5, 2e-5, 3e-5], dtype=config.floatX)
    x = normal(pt.arange(2), pt.ones(2), rng=rng)
    y = x[1]
    # The non-`Subtensor` client depends on the RNG state, so we can't perform
    # the lift
    z = x - y

    fg = FunctionGraph([rng], [z], clone=False, features=[ShapeFeature()])
    _ = EquilibriumGraphRewriter([local_subtensor_rv_lift], max_use_ratio=100).apply(fg)

    subtensor_node = fg.outputs[0].owner.inputs[1].owner.inputs[0].owner
    assert subtensor_node == y.owner
    assert isinstance(subtensor_node.op, Subtensor)
    assert subtensor_node.inputs[0].owner.op == normal

    z = pt.ones(x.shape) - x[1]

    # We add `x` as an output to make sure that `is_rv_used_in_graph` handles
    # `"output"` "nodes" correctly.
    fg = FunctionGraph([rng], [z, x], clone=False, features=[ShapeFeature()])
    EquilibriumGraphRewriter([local_subtensor_rv_lift], max_use_ratio=100).apply(fg)

    assert fg.outputs[0] == z
    assert fg.outputs[1] == x

    # The non-`Subtensor` client doesn't depend on the RNG state, so we can
    # perform the lift
    fg = FunctionGraph([rng], [z], clone=False, features=[ShapeFeature()])
    EquilibriumGraphRewriter([local_subtensor_rv_lift], max_use_ratio=100).apply(fg)

    rv_node = fg.outputs[0].owner.inputs[1].owner.inputs[0].owner
    assert rv_node.op == normal
    assert isinstance(rv_node.inputs[-1].owner.op, Subtensor)
    assert isinstance(rv_node.inputs[-2].owner.op, Subtensor)


def test_Dimshuffle_lift_restrictions():
    rng = shared(np.random.default_rng(1233532), borrow=False)

    x = normal(pt.arange(2).reshape((2,)), 100, size=(2, 2, 2), rng=rng)
    y = x.dimshuffle(1, 0, 2)
    # The non-`Dimshuffle` client depends on the RNG state, so we can't
    # perform the lift
    z = x - y

    fg = FunctionGraph([rng], [z, y], clone=False)
    _ = EquilibriumGraphRewriter([local_dimshuffle_rv_lift], max_use_ratio=100).apply(
        fg
    )

    dimshuffle_node = fg.outputs[0].owner.inputs[1].owner
    assert dimshuffle_node == y.owner
    assert isinstance(dimshuffle_node.op, DimShuffle)
    assert dimshuffle_node.inputs[0].owner.op == normal

    z = pt.ones(x.shape) - y

    # We add `x` as an output to make sure that `is_rv_used_in_graph` handles
    # `"output"` "nodes" correctly.
    fg = FunctionGraph([rng], [z, x], clone=False)
    EquilibriumGraphRewriter([local_dimshuffle_rv_lift], max_use_ratio=100).apply(fg)

    assert fg.outputs[0] == z
    assert fg.outputs[1] == x

    # The non-`Dimshuffle` client doesn't depend on the RNG state, so we can
    # perform the lift
    fg = FunctionGraph([rng], [z], clone=False)
    EquilibriumGraphRewriter([local_dimshuffle_rv_lift], max_use_ratio=100).apply(fg)

    rv_node = fg.outputs[0].owner.inputs[1].owner
    assert rv_node.op == normal
    assert isinstance(rv_node.inputs[-1].owner.op, DimShuffle)
    assert isinstance(rv_node.inputs[-2].owner.op, DimShuffle)


@pytest.mark.parametrize(
    "ds_order, lifted, dist_op, dist_params, size, rtol",
    [
        (
            ("x",),
            True,
            normal,
            (
                np.array(-10.0, dtype=np.float64),
                np.array(1e-6, dtype=np.float64),
            ),
            (),
            1e-7,
        ),
        (
            (0, 1, 2),
            True,
            normal,
            (np.array(0).astype(config.floatX), np.array(1e-6).astype(config.floatX)),
            (2, 1, 2),
            1e-3,
        ),
    ],
)
def test_Dimshuffle_lift_rename(ds_order, lifted, dist_op, dist_params, size, rtol):
    rng = shared(np.random.default_rng(1233532), borrow=False)

    new_out, *_ = apply_local_rewrite_to_rv(
        local_dimshuffle_rv_lift,
        lambda rv: rv.dimshuffle(ds_order),
        dist_op,
        dist_params,
        size,
        rng,
        name="test_name",
    )

    assert new_out.name == "test_name_lifted"
