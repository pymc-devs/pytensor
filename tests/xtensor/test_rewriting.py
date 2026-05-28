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

    # With ShapeFeature — force caching shape of XRV output before lowering
    sf = ShapeFeature()
    fgraph = FunctionGraph([x], [shape_y], features=[sf], copy_inputs=False)
    # Force get_shape on the XRV sum output (y) before any rewriting lowers it.
    # This caches a shape expression referencing the XRV variable.
    y_in_graph = [
        v
        for v in fgraph.variables
        if hasattr(v.type, "ndim") and v.type.ndim == 1 and v is not x
    ]
    for v in y_in_graph:
        try:
            sf.get_shape(v, 0)
        except Exception:
            pass
    infer_shape_db.default_query.rewrite(fgraph)
    [rewritten_shape_y] = fgraph.outputs
    assert_equal_computations([rewritten_shape_y], [Shape_i(1)(x)])
