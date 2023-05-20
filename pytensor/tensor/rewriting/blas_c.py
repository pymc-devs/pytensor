from pytensor.configdefaults import config
from pytensor.graph.rewriting.basic import in2out
from pytensor.tensor import basic as at
from pytensor.tensor.blas import gemv_inplace, gemv_no_inplace, ger, ger_destructive
from pytensor.tensor.blas_c import (
    CGemv,
    CGer,
    cgemv_inplace,
    cgemv_no_inplace,
    cger_inplace,
)
from pytensor.tensor.rewriting.blas import blas_optdb, node_rewriter, optdb


@node_rewriter([ger, ger_destructive])
def use_c_ger(fgraph, node):
    if not config.blas__ldflags:
        return
    # Only float32 and float64 are supported for now.
    if node.op == ger and node.outputs[0].dtype in ("float32", "float64"):
        return [CGer(False)(*node.inputs)]
    if node.op == ger_destructive and node.outputs[0].dtype in ("float32", "float64"):
        return [CGer(True)(*node.inputs)]


@node_rewriter([CGer(False)])
def make_c_ger_destructive(fgraph, node):
    if isinstance(node.op, CGer) and not node.op.destructive:
        return [cger_inplace(*node.inputs)]


@node_rewriter([gemv_inplace, gemv_no_inplace])
def use_c_gemv(fgraph, node):
    if not config.blas__ldflags:
        return
    # Only float32 and float64 are supported for now.
    if node.op == gemv_no_inplace and node.outputs[0].dtype in ("float32", "float64"):
        return [cgemv_no_inplace(*node.inputs)]
    if node.op == gemv_inplace and node.outputs[0].dtype in ("float32", "float64"):
        return [cgemv_inplace(*node.inputs)]


@node_rewriter([CGemv(inplace=False)])
def make_c_gemv_destructive(fgraph, node):
    if isinstance(node.op, CGemv) and not node.op.inplace:
        inputs = list(node.inputs)
        dest = inputs[0]
        if (
            dest.owner
            and isinstance(dest.owner.op, at.AllocEmpty)
            and len(fgraph.clients[dest]) > 1
        ):
            inputs[0] = at.AllocEmpty(dest.dtype)(*dest.owner.inputs)

        return [cgemv_inplace(*inputs)]


blas_optdb.register(
    "use_c_blas", in2out(use_c_ger, use_c_gemv), "fast_run", "c_blas", position=20
)

# this matches the InplaceBlasOpt defined in blas.py
optdb.register(
    "c_blas_destructive",
    in2out(make_c_ger_destructive, make_c_gemv_destructive, name="c_blas_destructive"),
    "fast_run",
    "inplace",
    "c_blas",
    position=70.0,
)
