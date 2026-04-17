import pytest

from pytensor.compile import optdb
from pytensor.graph import FunctionGraph
from pytensor.tensor import tensor
from pytensor.tensor.basic import infer_shape_db
from pytensor.tensor.random.type import random_generator_type
from pytensor.tensor.rewriting.shape import ShapeFeature
from pytensor.tensor.shape import Shape_i
from pytensor.xtensor import as_xtensor, xtensor
from pytensor.xtensor.random import normal
from pytensor.xtensor.vectorization import XRV
from tests.unittest_tools import assert_equal_computations


def test_infer_shape_db_handles_xtensor_lowering():
    x = xtensor("x", dims=("a", "b"))
    y = x.sum(dim="a")
    shape_y = y.shape[0]

    # Without ShapeFeature
    fgraph = FunctionGraph([x], [shape_y], features=[], copy_inputs=False)
    infer_shape_db.default_query.rewrite(fgraph)
    [rewritten_shape_y] = fgraph.outputs
    assert_equal_computations([rewritten_shape_y], [(x.values.sum(0)).shape[0]])

    # With ShapeFeature
    fgraph = FunctionGraph([x], [shape_y], features=[ShapeFeature()], copy_inputs=False)
    infer_shape_db.default_query.rewrite(fgraph)
    [rewritten_shape_y] = fgraph.outputs
    assert_equal_computations([rewritten_shape_y], [Shape_i(1)(x)])


@pytest.mark.parametrize("with_shape_feature", [False, True])
def test_nested_xrv_lowering_does_not_leak_stale_xrv(with_shape_feature):
    # Nested XRV where the outer's extra_dims aren't in the inner's dims.
    # Lowering needs the inner's shape for the outer's size, which drags the
    # pre-lowering XRV back into the graph via a dormant Shape_i cached in
    # ShapeFeature.shape_of.
    a_size = tensor("a_size", shape=(), dtype="int64")
    b_size = tensor("b_size", shape=(), dtype="int64")
    rng1 = random_generator_type("rng1")
    rng2 = random_generator_type("rng2")
    mu = normal(0.0, 0.1, extra_dims={"a": as_xtensor(a_size)}, rng=rng1)
    out = normal(mu, 1.0, extra_dims={"b": as_xtensor(b_size)}, rng=rng2)

    features = [ShapeFeature()] if with_shape_feature else []
    fgraph = FunctionGraph(
        [a_size, b_size, rng1, rng2],
        [out.values],
        features=features,
        copy_inputs=False,
    )
    optdb.query(
        "+lower_xtensor",
        "+canonicalize",
        "-local_eager_useless_unbatched_blockwise",
    ).rewrite(fgraph)

    stale = [n for n in fgraph.apply_nodes if isinstance(n.op, XRV)]
    assert not stale, f"XRV remained after lowering: {stale}"
