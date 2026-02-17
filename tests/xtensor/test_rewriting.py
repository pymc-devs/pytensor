from pytensor.graph import FunctionGraph
from pytensor.tensor.basic import infer_shape_db
from pytensor.tensor.rewriting.shape import ShapeFeature
from pytensor.tensor.shape import Shape_i
from pytensor.xtensor import xtensor
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
