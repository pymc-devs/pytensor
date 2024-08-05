from functools import partial

from pytensor.graph import ancestors, rewrite_graph
from pytensor.tensor import einsum, specify_shape, tensor
from pytensor.tensor.einsum import Einsum


specialize_rewrite = partial(rewrite_graph, include=("specialize",), clone=True)


def test_einsum_optimization():
    a = tensor("a", shape=(None, None))
    b = tensor("b", shape=(None, None))
    c = tensor("c", shape=(None, None))

    dynamic_shape_einsum = einsum("ij,ij,jk->ik", a, b, c)
    assert not dynamic_shape_einsum.owner.op.optimized

    rewritten_out = specialize_rewrite(dynamic_shape_einsum)
    assert isinstance(rewritten_out.owner.op, Einsum)

    a = specify_shape(a, (2, 3))
    b = specify_shape(b, (2, 3))
    c = specify_shape(c, (3, 5))

    static_shape_einsum = dynamic_shape_einsum.owner.clone_with_new_inputs(
        [a, b, c]
    ).default_output()
    assert not static_shape_einsum.owner.op.optimized

    rewritten_out = specialize_rewrite(static_shape_einsum)
    # Einsum was inlined because it was optimized
    assert not isinstance(rewritten_out.owner.op, Einsum)
    # Sanity check that it's not buried in the graph
    assert not any(
        isinstance(var.owner.op, Einsum)
        for var in ancestors([rewritten_out])
        if var.owner
    )
