from pytensor import function
from pytensor.graph import FunctionGraph
from pytensor.scalar import log as scalar_log
from pytensor.tensor import matrix, tensor3
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
