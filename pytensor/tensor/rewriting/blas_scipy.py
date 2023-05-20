from pytensor.graph.rewriting.basic import in2out
from pytensor.tensor.blas import ger, ger_destructive, have_fblas
from pytensor.tensor.blas_scipy import scipy_ger_inplace, scipy_ger_no_inplace
from pytensor.tensor.rewriting.blas import blas_optdb, node_rewriter, optdb


@node_rewriter([ger, ger_destructive])
def use_scipy_ger(fgraph, node):
    if node.op == ger:
        return [scipy_ger_no_inplace(*node.inputs)]


@node_rewriter([scipy_ger_no_inplace])
def make_ger_destructive(fgraph, node):
    if node.op == scipy_ger_no_inplace:
        return [scipy_ger_inplace(*node.inputs)]


use_scipy_blas = in2out(use_scipy_ger)
make_scipy_blas_destructive = in2out(make_ger_destructive)

if have_fblas:
    # scipy_blas is scheduled in the blas_optdb very late, because scipy sortof
    # sucks, but it is almost always present.
    # C implementations should be scheduled earlier than this, so that they take
    # precedence. Once the original Ger is replaced, then these optimizations
    # have no effect.
    blas_optdb.register("scipy_blas", use_scipy_blas, "fast_run", position=100)

    # this matches the InplaceBlasOpt defined in blas.py
    optdb.register(
        "make_scipy_blas_destructive",
        make_scipy_blas_destructive,
        "fast_run",
        "inplace",
        position=70.0,
    )
