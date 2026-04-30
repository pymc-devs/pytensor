import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.graph.basic import Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.type import lscalar, matrix
from pytensor.tensor.utils import hash_from_ndarray


def shape_of_variables(
    fgraph: FunctionGraph, input_shapes
) -> dict[Variable, tuple[int, ...]]:
    """Compute the numeric shape of every variable in ``fgraph`` given
    the numeric shapes of its inputs (test helper).

    Builds scalar placeholders for each input dim, walks
    ``builders.infer_shape`` over the fgraph variables, then compiles a
    scalar-in / scalar-out function. Used only by the tests in this
    module.
    """
    if any(i not in fgraph.inputs for i in input_shapes):
        raise ValueError(
            "input_shapes keys aren't in the fgraph.inputs. FunctionGraph()"
            " interface changed. Now by default, it clones the graph it receives."
            " To have the old behavior, give it this new parameter `clone=False`."
        )
    from pytensor.compile.builders import infer_shape

    input_shape_scalars: dict[Variable, tuple[Variable, ...]] = {
        inp: tuple(lscalar() for _ in range(inp.type.ndim)) for inp in fgraph.inputs
    }
    input_dims = [s for inp in fgraph.inputs for s in input_shape_scalars[inp]]

    all_vars: list[Variable] = [v for v in fgraph.variables if hasattr(v.type, "ndim")]
    inferred_shapes = infer_shape(
        outs=all_vars,
        inputs=list(fgraph.inputs),
        input_shapes=[input_shape_scalars[inp] for inp in fgraph.inputs],
    )
    per_var_shape: dict = dict(zip(all_vars, inferred_shapes, strict=True))
    output_dims = [dim for shape in per_var_shape.values() for dim in shape]

    compute_shapes = pytensor.function(
        input_dims, output_dims, on_unused_input="ignore"
    )

    numeric_input_dims = [dim for inp in fgraph.inputs for dim in input_shapes[inp]]
    numeric_output_dims = compute_shapes(*numeric_input_dims)

    sym_to_num_dict = dict(zip(output_dims, numeric_output_dims, strict=True))

    return {
        var: tuple(sym_to_num_dict[sym] for sym in shape)
        for var, shape in per_var_shape.items()
    }


def test_hash_from_ndarray():
    hashes = []
    x = np.random.random((5, 5))

    for data in [
        -2,
        -1,
        0,
        1,
        2,
        np.zeros((1, 5)),
        np.zeros((1, 6)),
        # Data buffer empty but different shapes
        np.zeros((1, 0)),
        np.zeros((2, 0)),
        # Same data buffer and shapes but different strides
        np.arange(25).reshape(5, 5),
        np.arange(25).reshape(5, 5).T,
        # Same data buffer, shapes and strides but different dtypes
        np.zeros((5, 5), dtype="uint32"),
        np.zeros((5, 5), dtype="int32"),
        # Test slice
        x,
        x[1:],
        x[:4],
        x[1:3],
        x[::2],
        x[::-1],
    ]:
        data = np.asarray(data)
        hashes.append(hash_from_ndarray(data))

    assert len(set(hashes)) == len(hashes)

    # test that different type of views and their copy give the same hash
    assert hash_from_ndarray(x[1:]) == hash_from_ndarray(x[1:].copy())
    assert hash_from_ndarray(x[1:3]) == hash_from_ndarray(x[1:3].copy())
    assert hash_from_ndarray(x[:4]) == hash_from_ndarray(x[:4].copy())
    assert hash_from_ndarray(x[::2]) == hash_from_ndarray(x[::2].copy())
    assert hash_from_ndarray(x[::-1]) == hash_from_ndarray(x[::-1].copy())


class TestShapeOfVariables:
    def test_simple(self):
        x = matrix("x")
        y = x + x
        fgraph = FunctionGraph([x], [y], clone=False)
        shapes = shape_of_variables(fgraph, {x: (5, 5)})
        assert shapes == {x: (5, 5), y: (5, 5)}

        x = matrix("x")
        y = pt.dot(x, x.T)
        fgraph = FunctionGraph([x], [y], clone=False)
        shapes = shape_of_variables(fgraph, {x: (5, 1)})
        assert shapes[x] == (5, 1)
        assert shapes[y] == (5, 5)

    def test_subtensor(self):
        x = matrix("x")
        subx = x[1:]
        fgraph = FunctionGraph([x], [subx], clone=False)
        shapes = shape_of_variables(fgraph, {x: (10, 10)})
        assert shapes[subx] == (9, 10)

    def test_err(self):
        x = matrix("x")
        subx = x[1:]
        fgraph = FunctionGraph([x], [subx])
        with pytest.raises(ValueError):
            shape_of_variables(fgraph, {x: (10, 10)})
