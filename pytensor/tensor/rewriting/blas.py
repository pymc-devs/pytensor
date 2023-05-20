"""optimizations for using BLAS calls

Optimizations
=============

The optimization pipeline works something like this:

    1. identify dot22 from dot
    2. identify gemm from dot22
    3. identify dot22scalar from dot22 that are not gemm
    4. specialize gemm to gemv where applicable
    5. specialize gemm to ger where applicable
    6. specialize dot22 -> gemv or ger where applicable

:note: GEMM is the most canonical BLAS signature that we deal with so far, it
    would be good to turn most things into GEMM (dot, inner, outer, dot22,
    dot22scalar), and then to specialize from gemm to the various other L2 and
    L3 operations.

Identify Dot22
--------------

Numpy's dot supports arguments that are of any rank, and we should support that
too (just for compatibility).  The BLAS optimizations work with Dot Ops whose
inputs are each either vector or matrix.  So the first part of the optimization
pipeline is to transform qualifying Dot Ops to Dot22 Ops. Dot22 Ops may be
transformed further, but they will get implemented by a BLAS call.

More precisely, Dot nodes whose inputs are all vectors or matrices and whose
inputs both have the same dtype, and whose dtype is float or complex, become
Dot22.  This is implemented in `local_dot_to_dot22`.


Identify Gemm from Dot22
------------------------

This is complicated, done in GemmOptimizer.

Identify Dot22Scalar from Dot22
-------------------------------

Dot22 Ops that remain after the GemmOptimizer is done have not
qualified as GEMM Ops. Still they might be scaled by a factor, in
which case we use Dot22Scalar which is like Gemm, but without the b
and the Z.  In the future it would be good to merge this into the
GemmOptimizer.

Specialize Gemm to Gemv
-----------------------

If arguments to GEMM are dimshuffled vectors, then we can use GEMV
instead. This optimization is `local_gemm_to_gemv`.

"""

import copy
import logging
import time

import numpy as np


try:
    import numpy.__config__  # noqa
except ImportError:
    pass


import pytensor.scalar
from pytensor.compile.mode import optdb
from pytensor.configdefaults import config
from pytensor.graph.features import ReplacementDidNotRemoveError, ReplaceValidate
from pytensor.graph.rewriting.basic import (
    EquilibriumGraphRewriter,
    GraphRewriter,
    copy_stack_trace,
    in2out,
    node_rewriter,
)
from pytensor.graph.rewriting.db import SequenceDB
from pytensor.graph.utils import InconsistencyError
from pytensor.printing import debugprint
from pytensor.tensor import basic as at
from pytensor.tensor.blas import (
    Dot22,
    _dot22,
    _dot22scalar,
    gemm_inplace,
    gemm_no_inplace,
    gemv_inplace,
    gemv_no_inplace,
    ger,
    ger_destructive,
)
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.math import Dot, add, mul, neg, sub
from pytensor.tensor.rewriting.elemwise import local_dimshuffle_lift
from pytensor.tensor.type import (
    DenseTensorType,
    TensorType,
    integer_dtypes,
    values_eq_approx_remove_inf_nan,
)


_logger = logging.getLogger("pytensor.tensor.rewriting.blas")


def res_is_a(fgraph, var, op, maxclients=None):
    if maxclients is not None and var in fgraph.clients:
        retval = len(fgraph.get_clients(var)) <= maxclients
    else:
        retval = True

    return var.owner and var.owner.op == op and retval


def _as_scalar(res, dtype=None):
    """Return ``None`` or a `TensorVariable` of float type"""
    if dtype is None:
        dtype = config.floatX
    if all(s == 1 for s in res.type.shape):
        while res.owner and isinstance(res.owner.op, DimShuffle):
            res = res.owner.inputs[0]
        # may still have some number of True's
        if res.type.ndim > 0:
            rval = res.dimshuffle()
        else:
            rval = res
        if rval.type.dtype in integer_dtypes:
            # We check that the upcast of res and dtype won't change dtype.
            # If dtype is float64, we will cast int64 to float64.
            # This is valid when res is a scalar used as input to a dot22
            # as the cast of the scalar can be done before or after the dot22
            # and this will give the same result.
            if pytensor.scalar.upcast(res.dtype, dtype) == dtype:
                return at.cast(rval, dtype)
            else:
                return None

        return rval


def _is_real_matrix(res):
    return (
        res.type.dtype in ("float16", "float32", "float64")
        and res.type.ndim == 2
        and res.type.shape[0] != 1
        and res.type.shape[1] != 1
    )  # cope with tuple vs. list


def _is_real_vector(res):
    return (
        res.type.dtype in ("float16", "float32", "float64")
        and res.type.ndim == 1
        and res.type.shape[0] != 1
    )


def _beta_L_plus_alpha_M(fgraph, beta, L, alpha, M, recurse_flip=True):
    # print 'BETA L + ALPHA M', beta, L, alpha, M, recurse_flip
    # EXPRESSION: (beta * L) + (alpha * M)

    # we've already checked the client counts, now just make the type check.
    # if res_is_a(M, _dot22, 1):
    if M.owner and M.owner.op == _dot22:
        Ml, Mr = M.owner.inputs
        rval = [gemm_no_inplace(L, alpha, Ml, Mr, beta)]
        return rval, M

    # it also might be the case that there is a dimshuffle between the +
    # and the dot22. local_dot_to_dot22 in particular will put in such things.
    if (
        M.owner
        and isinstance(M.owner.op, DimShuffle)
        and M.owner.inputs[0].owner
        and isinstance(M.owner.inputs[0].owner.op, Dot22)
    ):
        MM = M.owner.inputs[0]
        if M.owner.op.new_order == (0,):
            # it is making a column MM into a vector
            MMl, MMr = MM.owner.inputs
            g = gemm_no_inplace(L.dimshuffle(0, "x"), alpha, MMl, MMr, beta)
            rval = [g.dimshuffle(0)]
            return rval, MM
        if M.owner.op.new_order == (1,):
            # it is making a row MM into a vector
            MMl, MMr = MM.owner.inputs
            g = gemm_no_inplace(L.dimshuffle("x", 0), alpha, MMl, MMr, beta)
            rval = [g.dimshuffle(1)]
            return rval, MM
        if len(M.owner.op.new_order) == 0:
            # it is making a row MM into a vector
            MMl, MMr = MM.owner.inputs
            g = gemm_no_inplace(L.dimshuffle("x", "x"), alpha, MMl, MMr, beta)
            rval = [g.dimshuffle()]
            return rval, MM

    if recurse_flip:
        return _beta_L_plus_alpha_M(fgraph, alpha, M, beta, L, recurse_flip=False)
    else:
        return False, False


def _gemm_canonicalize(fgraph, r, scale, rval, maxclients):
    # Tries to interpret node as a sum of scalars * (vectors or matrices)
    def scaled(thing):
        if scale == 1:
            return thing
        if scale == -1 and thing.type.dtype != "bool":
            return -thing
        else:
            return scale * thing

    if not isinstance(r.type, TensorType):
        return None

    if (r.type.ndim not in (1, 2)) or r.type.dtype not in (
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ):
        rval.append(scaled(r))
        return rval

    if maxclients and len(fgraph.clients[r]) > maxclients:
        rval.append((scale, r))
        return rval

    if r.owner and r.owner.op == sub:
        _gemm_canonicalize(fgraph, r.owner.inputs[0], scale, rval, 1)
        _gemm_canonicalize(fgraph, r.owner.inputs[1], -scale, rval, 1)

    elif r.owner and r.owner.op == add:
        for i in r.owner.inputs:
            _gemm_canonicalize(fgraph, i, scale, rval, 1)

    elif r.owner and r.owner.op == neg:
        _gemm_canonicalize(fgraph, r.owner.inputs[0], -scale, rval, 1)

    elif r.owner and r.owner.op == mul:
        scalars = []
        vectors = []
        matrices = []
        for i in r.owner.inputs:
            if all(s == 1 for s in i.type.shape):
                while i.owner and isinstance(i.owner.op, DimShuffle):
                    i = i.owner.inputs[0]
                if i.type.ndim > 0:
                    scalars.append(i.dimshuffle())
                else:
                    scalars.append(i)
            elif _is_real_vector(i):
                vectors.append(i)
            elif _is_real_matrix(i):
                matrices.append(i)
            else:
                # just put the original arguments as in the base case
                rval.append((scale, r))
                return rval
        if len(matrices) == 1:
            assert len(vectors) == 0
            m = matrices[0]
            if len(scalars) == 0:
                _gemm_canonicalize(fgraph, m, scale, rval, 1)
            elif len(scalars) == 1:
                _gemm_canonicalize(fgraph, m, scaled(scalars[0]), rval, 1)
            else:
                _gemm_canonicalize(
                    fgraph, m, mul(scaled(scalars[0]), *scalars[1:]), rval, 1
                )
        elif len(vectors) == 1:
            assert len(matrices) == 0
            v = vectors[0]
            if len(scalars) == 0:
                _gemm_canonicalize(fgraph, v, scale, rval, 1)
            elif len(scalars) == 1:
                _gemm_canonicalize(fgraph, v, scaled(scalars[0]), rval, 1)
            else:
                _gemm_canonicalize(
                    fgraph, v, mul(scaled(scalars[0]), *scalars[1:]), rval, 1
                )
        else:  # lets not open this up
            rval.append((scale, r))
    else:
        rval.append((scale, r))
    return rval


def _factor_canonicalized(lst):
    # remove duplicates from canonicalized list

    # we only delete out of the right end of the list,
    # once i has touched a list element, it is permantent
    lst = list(lst)
    # print 'FACTOR', lst
    # for t in lst:
    #    if not isinstance(t, (list, tuple)):
    #        t = (t,)
    #    for e in t:
    #        try:
    #            pytensor.printing.debugprint(e)
    #        except TypeError:
    #            print e, type(e)
    i = 0
    while i < len(lst) - 1:
        try:
            s_i, M_i = lst[i]
        except Exception:
            i += 1
            continue

        j = i + 1
        while j < len(lst):
            try:
                s_j, M_j = lst[j]
            except Exception:
                j += 1
                continue

            if M_i is M_j:
                s_i = s_i + s_j
                lst[i] = (s_i, M_i)
                del lst[j]
            else:
                j += 1
        i += 1
    return lst


def _gemm_from_factored_list(fgraph, lst):
    """
    Returns None, or a list to replace node.outputs.

    """
    lst2 = []
    # Remove the tuple that can't be cast correctly.
    # This can happen when we try to cast a complex to a real
    for sM in lst:
        # Make every pair in list have matching dtypes
        # sM can be a tuple of 2 elements or an PyTensor variable.
        if isinstance(sM, tuple):
            sm0, sm1 = sM
            sm0 = at.as_tensor_variable(sm0)
            if pytensor.scalar.upcast(sm0.dtype, sm1.dtype) == sm1.dtype:
                lst2.append((at.cast(sm0, sm1.dtype), sM[1]))

    lst = lst2

    def item_to_var(t):
        try:
            s, M = t
        except Exception:
            return t
        if s == 1:
            return M
        if s == -1:
            return -M
        return s * M

    # Try every pair in the sM_list, trying to turn it into a gemm operation
    for i in range(len(lst) - 1):
        s_i, M_i = lst[i]

        for j in range(i + 1, len(lst)):
            s_j, M_j = lst[j]

            if not M_j.type.in_same_class(M_i.type):
                continue

            # print 'TRYING', (s_i, M_i, s_j, M_j)

            gemm_of_sM_list, old_dot22 = _beta_L_plus_alpha_M(
                fgraph, s_i, M_i, s_j, M_j
            )
            # print 'GOT IT', gemm_of_sM_list
            if gemm_of_sM_list:
                assert len(gemm_of_sM_list) == 1
                add_inputs = [
                    item_to_var(input) for k, input in enumerate(lst) if k not in (i, j)
                ]
                add_inputs.extend(gemm_of_sM_list)
                if len(add_inputs) > 1:
                    rval = [add(*add_inputs)]
                else:
                    rval = add_inputs
                # print "RETURNING GEMM THING", rval
                return rval, old_dot22


def _gemm_from_node2(fgraph, node):
    """

    TODO: In many expressions, there are many ways to turn it into a
    gemm.  For example dot(a,b) + c + d.  This function should return all
    of them, so that if one version of gemm causes a cycle in the graph, then
    another application of gemm can be tried.

    """
    lst = []
    t0 = time.perf_counter()
    _gemm_canonicalize(fgraph, node.outputs[0], 1.0, lst, 0)
    t1 = time.perf_counter()

    if len(lst) > 1:
        lst = _factor_canonicalized(lst)
        t2 = time.perf_counter()
        rval = _gemm_from_factored_list(fgraph, lst)
        t3 = time.perf_counter()

        # It can happen that _factor_canonicalized and
        # _gemm_from_factored_list return a node with an incorrect
        # type.  This happens in particular when one of the scalar
        # factors forces the upcast of the whole expression.  In that
        # case, we simply skip that candidate for Gemm.  This was
        # discussed in
        # http://groups.google.com/group/theano-dev/browse_thread/thread/a3096c82856e3ad5,
        # but never made it into a trac ticket.

        if rval and rval[0][0].type.in_same_class(node.outputs[0].type):
            return rval, t1 - t0, t2 - t1, t3 - t2

    return None, t1 - t0, 0, 0


class GemmOptimizer(GraphRewriter):
    """Graph optimizer for inserting Gemm operations."""

    def __init__(self):
        super().__init__()
        self.warned = False

    def add_requirements(self, fgraph):
        fgraph.attach_feature(ReplaceValidate())

    def apply(self, fgraph):
        did_something = True
        nb_iter = 0
        nb_replacement = 0
        nb_replacement_didn_t_remove = 0
        nb_inconsistency_make = 0
        nb_inconsistency_replace = 0
        time_canonicalize = 0
        time_factor_can = 0
        time_factor_list = 0
        time_toposort = 0
        if fgraph.profile:
            validate_before = fgraph.profile.validate_time
            callbacks_before = fgraph.execute_callbacks_times.copy()
            callback_before = fgraph.execute_callbacks_time

        def on_import(new_node):
            if new_node is not node:
                nodelist.append(new_node)

        u = pytensor.graph.rewriting.basic.DispatchingFeature(
            on_import, None, None, name="GemmOptimizer"
        )
        fgraph.attach_feature(u)
        while did_something:
            nb_iter += 1
            t0 = time.perf_counter()
            nodelist = pytensor.graph.basic.io_toposort(fgraph.inputs, fgraph.outputs)
            time_toposort += time.perf_counter() - t0
            did_something = False
            nodelist.reverse()
            for node in nodelist:
                if not (
                    isinstance(node.op, Elemwise)
                    and isinstance(
                        node.op.scalar_op,
                        (
                            pytensor.scalar.Add,
                            pytensor.scalar.Sub,
                            pytensor.scalar.Neg,
                            pytensor.scalar.Mul,
                        ),
                    )
                ):
                    continue
                if node not in fgraph.apply_nodes:
                    # This mean that we already removed this node from
                    # the graph
                    continue
                try:
                    new_outputs, time1, time2, time3 = _gemm_from_node2(fgraph, node)
                    time_canonicalize += time1
                    time_factor_can += time2
                    time_factor_list += time3
                except InconsistencyError:
                    nb_inconsistency_make += 1
                    continue
                if new_outputs:
                    new_outputs, old_dot22 = new_outputs
                    assert len(new_outputs) == len(node.outputs)
                    new_outputs[
                        0
                    ].tag.values_eq_approx = values_eq_approx_remove_inf_nan
                    try:
                        fgraph.replace_all_validate_remove(
                            list(zip(node.outputs, new_outputs)),
                            [old_dot22],
                            reason="GemmOptimizer",
                            # For now we disable the warning as we know case
                            # that we need to fix.
                            warn=False,  # warn=not self.warned
                        )
                        did_something = True
                        nb_replacement += 1
                    except InconsistencyError:
                        # TODO: retry other applications of gemm (see comment
                        # in _gemm_from_node)
                        nb_inconsistency_replace += 1
                    except ReplacementDidNotRemoveError:
                        nb_replacement_didn_t_remove += 1
                        self.warned = True
        fgraph.remove_feature(u)
        if fgraph.profile:
            validate_time = fgraph.profile.validate_time - validate_before
            callback_time = fgraph.execute_callbacks_time - callback_before
            callbacks_time = {}
            for k, v in fgraph.execute_callbacks_times.items():
                if k in callbacks_before:
                    callbacks_time[k] = v - callbacks_before[k]
                else:
                    callbacks_time[k] = v
        else:
            validate_time = None
            callback_time = None
            callbacks_time = {}

        return (
            self,
            nb_iter,
            nb_replacement,
            nb_replacement_didn_t_remove,
            nb_inconsistency_make,
            nb_inconsistency_replace,
            time_canonicalize,
            time_factor_can,
            time_factor_list,
            time_toposort,
            validate_time,
            callback_time,
            callbacks_time,
        )

    @classmethod
    def print_profile(cls, stream, prof, level=0):
        blanc = "    " * level
        print(blanc, cls.__name__, file=stream)
        print(blanc, " nb_iter", prof[1], file=stream)
        print(blanc, " nb_replacement", prof[2], file=stream)
        print(blanc, " nb_replacement_didn_t_remove", prof[3], file=stream)
        print(blanc, " nb_inconsistency_make", prof[4], file=stream)
        print(blanc, " nb_inconsistency_replace", prof[5], file=stream)
        print(blanc, " time_canonicalize", prof[6], file=stream)
        print(blanc, " time_factor_can", prof[7], file=stream)
        print(blanc, " time_factor_list", prof[8], file=stream)
        print(blanc, " time_toposort", prof[9], file=stream)
        print(blanc, " validate_time", prof[10], file=stream)
        print(blanc, " callback_time", prof[11], file=stream)
        if prof[11] > 1:
            print(blanc, " callbacks_time", file=stream)
            for i in sorted(prof[12].items(), key=lambda a: a[1]):
                if i[1] > 0:
                    print(i)


@node_rewriter([Dot])
def local_dot_to_dot22(fgraph, node):
    # This works for tensor.outer too because basic.outer is a macro that
    # produces a dot(dimshuffle,dimshuffle) of form 4 below
    if not isinstance(node.op, Dot):
        return

    if any(not isinstance(i.type, DenseTensorType) for i in node.inputs):
        return False

    x, y = node.inputs
    if y.type.dtype != x.type.dtype:
        # TODO: upcast one so the types match
        _logger.info(f"Not optimizing dot with inputs {x} {y} {x.type} {y.type}")
        return

    if y.type.dtype in ("float16", "float32", "float64", "complex64", "complex128"):
        if x.ndim == 2 and y.ndim == 2:
            new_out = [_dot22(*node.inputs)]
        elif x.ndim == 2 and y.ndim == 1:
            new_out = [_dot22(x, y.dimshuffle(0, "x")).dimshuffle(0)]
        elif x.ndim == 1 and y.ndim == 2:
            new_out = [_dot22(x.dimshuffle("x", 0), y).dimshuffle(1)]
        elif x.ndim == 1 and y.ndim == 1:
            new_out = [_dot22(x.dimshuffle("x", 0), y.dimshuffle(0, "x")).dimshuffle()]
        else:
            return
        copy_stack_trace(node.outputs, new_out)
        return new_out

    _logger.info(f"Not optimizing dot with inputs {x} {y} {x.type} {y.type}")


@node_rewriter([gemm_no_inplace], inplace=True)
def local_inplace_gemm(fgraph, node):
    if node.op == gemm_no_inplace:
        new_out = [gemm_inplace(*node.inputs)]
        copy_stack_trace(node.outputs, new_out)
        return new_out


@node_rewriter([gemv_no_inplace], inplace=True)
def local_inplace_gemv(fgraph, node):
    if node.op == gemv_no_inplace:
        new_out = [gemv_inplace(*node.inputs)]
        copy_stack_trace(node.outputs, new_out)
        return new_out


@node_rewriter([ger], inplace=True)
def local_inplace_ger(fgraph, node):
    if node.op == ger:
        new_out = [ger_destructive(*node.inputs)]
        copy_stack_trace(node.outputs, new_out)
        return new_out


@node_rewriter([gemm_no_inplace])
def local_gemm_to_gemv(fgraph, node):
    """GEMM acting on row or column matrices -> GEMV."""
    if node.op == gemm_no_inplace:
        z, a, x, y, b = node.inputs
        if z.broadcastable == x.broadcastable == (True, False):
            r = gemv_no_inplace(z.dimshuffle(1), a, y.T, x.dimshuffle(1), b)
            new_out = [r.dimshuffle("x", 0)]
        elif z.broadcastable == y.broadcastable == (False, True):
            r = gemv_no_inplace(z.dimshuffle(0), a, x, y.dimshuffle(0), b)
            new_out = [r.dimshuffle(0, "x")]
        else:
            return
        copy_stack_trace(node.outputs, new_out)
        return new_out


@node_rewriter([gemm_no_inplace])
def local_gemm_to_ger(fgraph, node):
    """GEMM computing an outer-product -> GER."""
    if node.op == gemm_no_inplace:
        z, a, x, y, b = node.inputs
        if x.broadcastable[1] and y.broadcastable[0]:
            # x and y are both vectors so this might qualifies for a GER
            xv = x.dimshuffle(0)
            yv = y.dimshuffle(1)
            try:
                bval = at.get_underlying_scalar_constant_value(b)
            except NotScalarConstantError:
                # b isn't a constant, GEMM is doing useful pre-scaling
                return

            if bval == 1:  # best case a natural GER
                rval = ger(z, a, xv, yv)
                new_out = [rval]
            elif bval == 0:  # GER on zeros_like should be faster than GEMM
                zeros = at.zeros([x.shape[0], y.shape[1]], x.dtype)
                rval = ger(zeros, a, xv, yv)
                new_out = [rval]
            else:
                # if bval is another constant, then z is being usefully
                # pre-scaled and GER isn't really the right tool for the job.
                return
            copy_stack_trace(node.outputs, new_out)
            return new_out


# TODO: delete this optimization when we have the proper dot->gemm->ger pipeline
#      working
@node_rewriter([_dot22])
def local_dot22_to_ger_or_gemv(fgraph, node):
    """dot22 computing an outer-product -> GER."""
    if node.op == _dot22:
        x, y = node.inputs
        xb = x.broadcastable
        yb = y.broadcastable
        one = at.as_tensor_variable(np.asarray(1, dtype=x.dtype))
        zero = at.as_tensor_variable(np.asarray(0, dtype=x.dtype))
        if xb[1] and yb[0]:
            # x and y are both vectors so this might qualifies for a GER
            xv = x.dimshuffle(0)
            yv = y.dimshuffle(1)
            zeros = at.zeros([x.shape[0], y.shape[1]], dtype=x.dtype)
            rval = ger(zeros, one, xv, yv)
            new_out = [rval]
        elif xb[0] and yb[1]:
            # x and y are both vectors so this qualifies for a sdot / ddot
            # TODO: PyTensor doesn't have a sdot, but gemv is better than _dot22
            xv = x.dimshuffle(1)
            zeros = at.AllocEmpty(x.dtype)(1)
            rval = gemv_no_inplace(zeros, one, y.T, xv, zero)
            new_out = [rval.dimshuffle("x", 0)]
        elif xb[0] and not yb[0] and not yb[1]:
            # x is vector, y is matrix so try gemv
            xv = x.dimshuffle(1)
            zeros = at.AllocEmpty(x.dtype)(y.shape[1])
            rval = gemv_no_inplace(zeros, one, y.T, xv, zero)
            new_out = [rval.dimshuffle("x", 0)]
        elif not xb[0] and not xb[1] and yb[1]:
            # x is matrix, y is vector, try gemv
            yv = y.dimshuffle(0)
            zeros = at.AllocEmpty(x.dtype)(x.shape[0])
            rval = gemv_no_inplace(zeros, one, x, yv, zero)
            new_out = [rval.dimshuffle(0, "x")]
        else:
            return
        copy_stack_trace(node.outputs, new_out)
        return new_out


#################################
#
# Set up the BlasOpt optimizer
#
#################################

blas_optdb = SequenceDB()

# run after numerical stability optimizations (1.5)
optdb.register("BlasOpt", blas_optdb, "fast_run", "fast_compile", position=1.7)
# run before specialize (2.0) because specialize is basically a
# free-for-all that makes the graph crazy.

# fast_compile is needed to have GpuDot22 created.
blas_optdb.register(
    "local_dot_to_dot22",
    in2out(local_dot_to_dot22),
    "fast_run",
    "fast_compile",
    position=0,
)
blas_optdb.register("gemm_optimizer", GemmOptimizer(), "fast_run", position=10)
blas_optdb.register(
    "local_gemm_to_gemv",
    EquilibriumGraphRewriter(
        [
            local_gemm_to_gemv,
            local_gemm_to_ger,
            local_dot22_to_ger_or_gemv,
            local_dimshuffle_lift,
        ],
        max_use_ratio=5,
        ignore_newtrees=False,
    ),
    "fast_run",
    position=15,
)


# After destroyhandler(49.5) but before we try to make elemwise things
# inplace (75)
blas_opt_inplace = in2out(
    local_inplace_gemm, local_inplace_gemv, local_inplace_ger, name="blas_opt_inplace"
)
optdb.register(
    "InplaceBlasOpt",
    blas_opt_inplace,
    "fast_run",
    "inplace",
    "blas_opt_inplace",
    position=70.0,
)


@node_rewriter([mul])
def local_dot22_to_dot22scalar(fgraph, node):
    """
    Notes
    -----
    Previous attempts to alter this optimization to replace dot22 with
    gemm instead of dot22scalar resulted in some Scan nodes being
    duplicated and the ScanSaveMem optimization never running on them,
    resulting in highly increased memory usage. Until this issue is
    resolved, this optimization should keep using dot22scalar instead of
    gemm.

    We upcast the scalar if after the multiplication with the dot this give
    the same type.

    We execute this optimizer after the gemm optimizer. This
    allow to give more priority to gemm that give more speed up
    then this optimizer, but allow the gemm optimizer to ignore
    this op.

    TODO: support when we can reorder the mul to generate a
    dot22scalar or fix the canonizer to merge them(1 mul with multiple
    inputs)

    """
    if node.op != mul:
        return False
    i_dot22 = [x.owner and x.owner.op == _dot22 for x in node.inputs]
    if not any(i_dot22):
        return False  # no dot22
    if i_dot22.count(True) > 1:
        # TODO: try each of them.
        pass
        # return False #TODO fix
    dot22_idx = i_dot22.index(True)
    d = node.inputs[dot22_idx]
    i_scalar = [_as_scalar(x, dtype=d.dtype) for x in node.inputs]
    if not any(i_scalar):
        # Check if we can reorder the graph as this mul have a mul in inputs.
        # We support only 1 additional level of mul.
        # The canonizer should have merged those mul together.
        i_mul = [
            x.owner
            and x.owner.op == mul
            and any(_as_scalar(x_i, dtype=d.dtype) for x_i in x.owner.inputs)
            for x in node.inputs
        ]
        if not any(i_mul):
            # no scalar in input and no multiplication
            # if their was a multiplication we couls reorder the graph
            # by the associativity of the graph.
            return False

        mul_idx = i_mul.index(True)  # The first one should always work
        m = node.inputs[mul_idx]

        scalar_idx = -1
        for i, x in enumerate(m.owner.inputs):
            if _as_scalar(x, dtype=d.dtype) and (
                pytensor.scalar.upcast(x.type.dtype, d.type.dtype) == d.type.dtype
            ):
                scalar_idx = i
                break

        if scalar_idx < 0:
            _logger.info(
                f"Not optimizing dot22 with inputs {node.inputs} {[x.type for x in node.inputs]}, as the"
                " type of the scalar cannot be upcasted to the"
                " matrix type"
            )
            return False
        a = at.cast(_as_scalar(m.owner.inputs[scalar_idx], dtype=d.dtype), d.type.dtype)
        assert not a.type.ndim
        dot = _dot22scalar(d.owner.inputs[0], d.owner.inputs[1], a)

        # The other inputs to the original node that were
        # neither part of the dot22 or this mul should be
        # factors in the returned "mul" node.
        assert dot22_idx != mul_idx
        other_factors = [
            inpt for i, inpt in enumerate(node.inputs) if i not in (dot22_idx, mul_idx)
        ]
        other_m_inputs = [
            inpt for i, inpt in enumerate(m.owner.inputs) if i != scalar_idx
        ]

        return [mul(dot, *(other_factors + other_m_inputs))]

    scalar_idx = -1
    for i, x in enumerate(node.inputs):
        if (
            i != dot22_idx
            and i_scalar[i] is not None
            and (pytensor.scalar.upcast(x.type.dtype, d.type.dtype) == d.type.dtype)
        ):
            scalar_idx = i
            break
    if scalar_idx < 0:
        _logger.info(
            f"Not optimizing dot22 with inputs {node.inputs} {[x.type for x in node.inputs]}, as the type "
            "of the scalar cannot be upcasted to the matrix type"
        )
        return False
    assert scalar_idx < len(node.inputs)
    s = node.inputs[scalar_idx]
    o = copy.copy(node.inputs)
    o.remove(d)
    o.remove(s)

    a = at.cast(i_scalar[scalar_idx], d.type.dtype)
    assert not a.type.ndim
    if len(o) == 0:
        return [_dot22scalar(d.owner.inputs[0], d.owner.inputs[1], a)]
    else:
        return [mul(_dot22scalar(d.owner.inputs[0], d.owner.inputs[1], a), *o)]


# must happen after gemm as the gemm optimizer don't understant
# dot22scalar and gemm give more speed up then dot22scalar
blas_optdb.register(
    "local_dot22_to_dot22scalar",
    in2out(local_dot22_to_dot22scalar),
    "fast_run",
    position=11,
)


# from opt import register_specialize, register_canonicalize
# @register_specialize
@node_rewriter([sub, add])
def local_print_as_we_go_along(fgraph, node):
    if node.op in (sub, add):
        debugprint(node)
