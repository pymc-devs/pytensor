from functools import partial

from pytensor import function
from pytensor.graph import FunctionGraph, rewrite_graph
from pytensor.graph.basic import equal_computations
from pytensor.scalar import log as scalar_log
from pytensor.tensor import add, alloc, matrix, tensor, tensor3
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.nlinalg import MatrixPinv
from pytensor.tensor.rewriting.blockwise import local_useless_blockwise


def test_useless_blockwise_of_elemwise():
    x = matrix("x")
    out = Blockwise(Elemwise(scalar_log), signature="()->()")(x)
    assert isinstance(out.owner.op, Blockwise)
    assert isinstance(out.owner.op.core_op, Elemwise)

    fg = FunctionGraph([x], [out], clone=False)
    [new_out] = local_useless_blockwise.transform(fg, out.owner)
    assert isinstance(new_out.owner.op, Elemwise)


def test_useless_unbatched_blockwise():
    x = matrix("x")
    blockwise_op = Blockwise(MatrixPinv(hermitian=False), signature="(m,n)->(n,m)")
    out = blockwise_op(x)

    assert isinstance(out.owner.op, Blockwise)
    assert isinstance(out.owner.op.core_op, MatrixPinv)

    fn = function([x], out, mode="FAST_COMPILE")
    assert isinstance(fn.maker.fgraph.outputs[0].owner.op, MatrixPinv)

    # Test that it's not removed when there are batched dims
    x = tensor3("x")
    out = blockwise_op(x)
    fn = function([x], out, mode="FAST_COMPILE")
    assert isinstance(fn.maker.fgraph.outputs[0].owner.op, Blockwise)
    assert isinstance(fn.maker.fgraph.outputs[0].owner.op.core_op, MatrixPinv)


def test_blockwise_alloc():
    rewrite = partial(
        rewrite_graph,
        include=("ShapeOpt", "specialize"),
        exclude=("local_useless_unbatched_blockwise",),
    )

    vector_add = Blockwise(core_op=add, signature="(x),(x)->(x)")

    # Depending on the rewrites the Alloc shape may be upcast to int64 or not
    # We do not care about that for the purposes of this test
    equal = partial(equal_computations, strict_dtype=False)

    # Case where Alloc is not necessary
    x = tensor("x", shape=(7, 5))
    y = tensor("y", shape=(5,))
    out = vector_add(x, alloc(y, 7, 5))
    expected_out = vector_add(x, y)
    assert equal([rewrite(out)], [expected_out])

    # Cases where Alloc can be fully pushed
    x = tensor("x", shape=(5,))
    y = tensor("y", shape=(5,))
    out = vector_add(x, alloc(y, 7, 5))
    expected_out = alloc(vector_add(x, y), 7, 5)
    assert equal([rewrite(out)], [expected_out])

    x = tensor("x", shape=(1, 5))
    y = tensor("y", shape=(5,))
    out = vector_add(x, alloc(y, 7, 5))
    expected_out = alloc(vector_add(x.squeeze(0), y), 7, 5)
    assert equal([rewrite(out)], [expected_out])

    x = tensor("x", shape=(7, 5))
    y = tensor("y", shape=(7, 5))
    out = vector_add(x, alloc(y, 3, 7, 5))
    expected_out = alloc(vector_add(x, y), 3, 7, 5)
    assert equal([rewrite(out)], [expected_out])

    x = tensor("x", shape=(5,))
    y = tensor("y", shape=(7, 1, 5))
    out = vector_add(x, alloc(y, 7, 2, 5))
    expected_out = alloc(vector_add(x, y), 7, 2, 5)
    assert equal([rewrite(out)], [expected_out])

    # Case where Alloc can be partially pushed
    x = tensor("x", shape=(5,))
    y = tensor("y", shape=())
    out = vector_add(x, alloc(y, 7, 5))
    expected_out = alloc(vector_add(x, alloc(y, 5)), 7, 5)
    assert equal([rewrite(out)], [expected_out])

    x = tensor("x", shape=(5,))
    y = tensor("y", shape=(7, 1, 1))
    out = vector_add(x, alloc(y, 7, 2, 5))
    expected_out = alloc(vector_add(x, alloc(y, 7, 1, 5)), 7, 2, 5)
    assert equal([rewrite(out)], [expected_out], strict_dtype=False)

    # Cases involving multiple Allocs being pushed
    x = tensor("x", shape=())
    y = tensor("y", shape=())
    out = vector_add(alloc(x, 3, 1, 5), alloc(y, 7, 5))
    expected_out = alloc(vector_add(alloc(x, 5), alloc(y, 5)), 3, 7, 5)
    assert equal([rewrite(out)], [expected_out])

    x = tensor("x", shape=(5,))
    y = tensor("y", shape=())
    out = vector_add(alloc(x, 3, 1, 5), alloc(y, 7, 5))
    expected_out = alloc(vector_add(x, alloc(y, 5)), 3, 7, 5)
    assert equal([rewrite(out)], [expected_out])

    # Case where Alloc cannot be pushed
    x = tensor("x", shape=(5,))
    y = tensor("y", shape=(1,))
    out = vector_add(x, alloc(y, 5))
    expected_out = out
    assert equal([rewrite(out)], [expected_out])
