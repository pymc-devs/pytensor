from functools import partial

import numpy as np
import pytest

from pytensor import Mode, config, function
from pytensor.graph import FunctionGraph, rewrite_graph, vectorize_graph
from pytensor.graph.basic import equal_computations
from pytensor.graph.traversal import apply_ancestors
from pytensor.scalar import log as scalar_log
from pytensor.tensor import add, alloc, iscalar, matrix, scalar, tensor, tensor3
from pytensor.tensor.basic import AllocEmpty
from pytensor.tensor.blas import Gemv
from pytensor.tensor.blockwise import Blockwise, BlockwiseWithCoreShape
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.linalg.inverse import MatrixPinv
from pytensor.tensor.rewriting.blockwise import local_useless_blockwise
from pytensor.tensor.shape import Reshape


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
    assert isinstance(
        fn.maker.fgraph.outputs[0].owner.op, Blockwise | BlockwiseWithCoreShape
    )
    assert isinstance(fn.maker.fgraph.outputs[0].owner.op.core_op, MatrixPinv)


def test_local_blockwise_alloc_inputs():
    rewrite = partial(
        rewrite_graph,
        include=("ShapeOpt", "specialize"),
        exclude=("local_useless_unbatched_blockwise", "local_dimshuffle_alloc"),
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
    # pytensor.dprint([expected_out, rewrite(out)], print_type=True)
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


@pytest.mark.parametrize("implicit_dims", [True, False])
def test_local_blockwise_alloc(implicit_dims):
    """Test that Blockwise(Alloc) is rewritten to a plain Alloc."""
    x = scalar("x")
    n = iscalar("n")
    if implicit_dims:
        out = alloc(x, n)
    else:
        out = alloc(x[None], n)

    # Vectorize with a batch shape that is itself an Alloc.
    # This creates Blockwise(Alloc) because the shape is non-broadcastable.
    # Other rewrites lift the Alloc above the Blockwise, then
    # local_blockwise_alloc simplifies the remaining Blockwise(Alloc).
    vect_x = tensor("vect_x", shape=(5,))
    vect_out = vectorize_graph(out, {x: vect_x, n: alloc(n, 5)})
    assert isinstance(vect_out.owner.op, Blockwise)

    rewritten_vect_out = rewrite_graph(
        vect_out, include=("canonicalize", "specialize"), clone=True
    )
    assert not any(
        isinstance(node.op, Blockwise) for node in apply_ancestors([rewritten_vect_out])
    )

    n_val = np.int64(3)
    vect_x_test = np.random.normal(size=(5,)).astype(config.floatX)
    no_rewrites = Mode(linker="py", optimizer=None)
    np.testing.assert_allclose(
        vect_out.eval({"vect_x": vect_x_test, "n": n_val}, mode=no_rewrites),
        rewritten_vect_out.eval(
            {"vect_x": vect_x_test, "n": n_val}, on_unused_input="ignore"
        ),
    )


def test_blockwise_reshape():
    x = tensor("x", shape=(None, None, None))
    y = x.reshape([x.shape[0] * x.shape[1], -1])

    new_x = tensor("x", shape=(None, None, None, None))
    new_y = vectorize_graph(y, {x: new_x})
    assert not isinstance(new_y.owner.op, Reshape)
    assert isinstance(new_y.owner.op, Blockwise) and isinstance(
        new_y.owner.op.core_op, Reshape
    )

    rewritten_y = rewrite_graph(
        new_y, include=("canonicalize", "specialize"), clone=True
    )
    assert isinstance(rewritten_y.owner.op, Reshape)

    no_rewrites = Mode(linker="py", optimizer=None)
    test_x = np.arange(5 * 4 * 3 * 2).reshape(5, 4, 3, 2).astype(config.floatX)
    np.testing.assert_allclose(
        new_y.eval({"x": test_x}, mode=no_rewrites),
        rewritten_y.eval({"x": test_x}, mode=no_rewrites),
    )


def test_split_alloc_empty_clients_enables_inplace():
    """Two destructive clients of one `AllocEmpty` each get a buffer they can destroy."""
    x = matrix("x")
    y = tensor("y", shape=(None,))
    z = tensor("z", shape=(None,))

    f = function([x, y, z], [y @ x, z @ x], mode="cvm")
    nodes = f.maker.fgraph.apply_nodes
    gemvs = [n.op for n in nodes if isinstance(n.op, Gemv)]
    assert len(gemvs) == 2
    assert all(op.inplace for op in gemvs), gemvs
    # One buffer per destroyer, rather than one shared between them.
    assert len([n for n in nodes if isinstance(n.op, AllocEmpty)]) == 2

    rng = np.random.default_rng(sum(map(ord, "split_alloc_empty")))
    x_val = rng.normal(size=(3, 3))
    y_val = rng.normal(size=(3,))
    z_val = rng.normal(size=(3,))
    out_y, out_z = f(x_val, y_val, z_val)
    np.testing.assert_allclose(out_y, y_val @ x_val)
    np.testing.assert_allclose(out_z, z_val @ x_val)


def test_split_alloc_empty_clients_leaves_readers_alone():
    """An `AllocEmpty` shared by clients that cannot destroy it stays a single buffer."""
    shape = iscalar("shape")
    buffer = AllocEmpty(config.floatX)(shape)
    out = add(buffer * 2, buffer * 3)

    fg = FunctionGraph([shape], [out])
    rewrite_graph(fg, include=("fast_run", "inplace"))
    allocs = [n for n in fg.apply_nodes if isinstance(n.op, AllocEmpty)]
    assert len(allocs) == 1, allocs
