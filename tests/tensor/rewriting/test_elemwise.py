import warnings

import numpy as np
import pytest

import pytensor
from pytensor import In, shared
from pytensor import scalar as ps
from pytensor import tensor as pt
from pytensor.compile.function import function
from pytensor.compile.mode import Mode, get_default_mode
from pytensor.configdefaults import config
from pytensor.gradient import grad
from pytensor.graph.basic import Constant, equal_computations
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import check_stack_trace, out2in
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.misc.safe_asarray import _asarray
from pytensor.raise_op import assert_op
from pytensor.scalar.basic import Composite, float64
from pytensor.tensor.basic import MakeVector
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import abs as pt_abs
from pytensor.tensor.math import (
    add,
    bitwise_and,
    bitwise_or,
    cos,
    cosh,
    dot,
    eq,
    exp,
    ge,
    int_div,
    invert,
    iround,
    log,
    log1mexp,
    log2,
    log10,
    mul,
    neg,
    neq,
    reciprocal,
    sin,
    sinh,
    sqr,
    sqrt,
    tan,
    tanh,
    true_div,
    xor,
)
from pytensor.tensor.math import all as pt_all
from pytensor.tensor.math import pow as pt_pow
from pytensor.tensor.math import round as pt_round
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.rewriting.elemwise import FusionOptimizer, local_dimshuffle_lift
from pytensor.tensor.rewriting.shape import local_useless_dimshuffle_in_reshape
from pytensor.tensor.shape import reshape
from pytensor.tensor.type import (
    TensorType,
    dmatrices,
    dscalar,
    dvector,
    fscalar,
    fvector,
    matrix,
    scalar,
    tensor,
    vector,
    vectors,
)
from tests import unittest_tools as utt


dimshuffle_lift = out2in(local_dimshuffle_lift)


def ds(x, y):
    return DimShuffle(x.type.broadcastable, y)(x)


def inputs(xbc=(0, 0), ybc=(0, 0), zbc=(0, 0)):
    x = TensorType(dtype="float64", shape=xbc)("x")
    y = TensorType(dtype="float64", shape=ybc)("y")
    z = TensorType(dtype="float64", shape=zbc)("z")
    return x, y, z


class TestDimshuffleLift:
    def test_double_transpose(self):
        x, *_ = inputs()
        e = ds(ds(x, (1, 0)), (1, 0))
        g = FunctionGraph([x], [e], clone=False)
        assert isinstance(g.outputs[0].owner.op, DimShuffle)
        dimshuffle_lift.rewrite(g)
        assert g.outputs[0] is x
        # no need to check_stack_trace as graph is supposed to be empty

    def test_merge2(self):
        x, *_ = inputs()
        e = ds(ds(x, (1, "x", 0)), (2, 0, "x", 1))
        g = FunctionGraph([x], [e], clone=False)
        assert len(g.apply_nodes) == 2
        dimshuffle_lift.rewrite(g)
        assert equal_computations(g.outputs, [x.dimshuffle(0, 1, "x", "x")])
        # Check stacktrace was copied over correctly after rewrite was applied
        assert check_stack_trace(g, ops_to_check="all")

    def test_elim3(self):
        x, y, z = inputs()
        e = ds(ds(ds(x, (0, "x", 1)), (2, 0, "x", 1)), (1, 0))
        g = FunctionGraph([x], [e], clone=False)
        assert isinstance(g.outputs[0].owner.op, DimShuffle)
        dimshuffle_lift.rewrite(g)
        assert g.outputs[0] is x
        # no need to check_stack_trace as graph is supposed to be empty

    def test_lift(self):
        x, y, z = inputs([False] * 1, [False] * 2, [False] * 3)
        e = x + y + z
        g = FunctionGraph([x, y, z], [e], clone=False)
        dimshuffle_lift.rewrite(g)
        assert equal_computations(
            g.outputs,
            [(x.dimshuffle("x", "x", 0) + y.dimshuffle("x", 0, 1)) + z],
        )
        # Check stacktrace was copied over correctly after rewrite was applied
        assert check_stack_trace(g, ops_to_check="all")

    def test_recursive_lift(self):
        v = vector("v", dtype="float64")
        m = matrix("m", dtype="float64")
        out = ((v + 42) * (m + 84)).T
        g = FunctionGraph([v, m], [out], clone=False)
        new_out = local_dimshuffle_lift.transform(g, g.outputs[0].owner)
        assert equal_computations(
            new_out,
            [(v.dimshuffle(0, "x") + 42) * (m.T + 84)],
        )
        # Check stacktrace was copied over correctly after rewrite was applied
        new_g = FunctionGraph(g.inputs, new_out, clone=False)
        assert check_stack_trace(new_g, ops_to_check="all")

    def test_useless_dimshuffle(self):
        x, *_ = inputs()
        e = ds(x, (0, 1))
        g = FunctionGraph([x], [e], clone=False)
        assert isinstance(g.outputs[0].owner.op, DimShuffle)
        dimshuffle_lift.rewrite(g)
        assert g.outputs[0] is x
        # Check stacktrace was copied over correctly after rewrite was applied
        assert hasattr(g.outputs[0].tag, "trace")

    def test_dimshuffle_on_broadcastable(self):
        x, y, z = inputs([False, True], [True, False, True], [False, False, True])
        u = pt.constant(1)
        ds_x = ds(x, (0, "x"))  # useless
        ds_y = ds(y, (2, 1, 0))  # useless
        ds_z = ds(z, (2, 1, 0))  # useful
        ds_u = ds(u, ("x"))  # useful
        g = FunctionGraph([x, y, z, u], [ds_x, ds_y, ds_z, ds_u], clone=False)
        assert len(g.apply_nodes) == 4
        dimshuffle_lift.rewrite(g)
        assert equal_computations(g.outputs, [x, y, z.T, u.dimshuffle("x")])
        # Check stacktrace was copied over correctly after rewrite was applied
        assert hasattr(g.outputs[0].tag, "trace")

    def test_dimshuffle_lift_multi_out_elemwise(self):
        # Create a multi-output Elemwise Op with Composite
        x = float64("x")
        outs = [x + 1, x + 2]
        op = Elemwise(Composite([x], outs))

        # Transpose both outputs
        x = matrix("x")
        outs = [out.T for out in op(x)]

        # Make sure rewrite doesn't apply in this case
        g = FunctionGraph([x], outs)
        assert not local_dimshuffle_lift.transform(g, g.outputs[0].owner)


def test_local_useless_dimshuffle_in_reshape():
    vec = TensorType(dtype="float64", shape=(None,))("vector")
    mat = TensorType(dtype="float64", shape=(None, None))("mat")
    row = TensorType(dtype="float64", shape=(1, None))("row")
    col = TensorType(dtype="float64", shape=(None, 1))("col")

    reshape_dimshuffle_vector = reshape(vec.dimshuffle("x", 0), vec.shape)
    reshape_dimshuffle_mat = reshape(mat.dimshuffle("x", 0, "x", 1), mat.shape)
    reshape_dimshuffle_row = reshape(row.dimshuffle(1, "x"), row.shape)
    reshape_dimshuffle_col = reshape(col.dimshuffle(0), col.shape)

    g = FunctionGraph(
        [vec, mat, row, col],
        [
            reshape_dimshuffle_vector,
            reshape_dimshuffle_mat,
            reshape_dimshuffle_row,
            reshape_dimshuffle_col,
        ],
        clone=False,
    )
    assert len(g.apply_nodes) == 4 * 3
    useless_dimshuffle_in_reshape = out2in(local_useless_dimshuffle_in_reshape)
    useless_dimshuffle_in_reshape.rewrite(g)
    assert equal_computations(
        g.outputs,
        [
            reshape(vec, vec.shape),
            reshape(mat, mat.shape),
            reshape(row, row.shape),
            reshape(col, col.shape),
        ],
    )
    # Check stacktrace was copied over correctly after rewrite was applied
    assert check_stack_trace(g, ops_to_check="all")

    # Check that the rewrite does not get applied when the order
    # of dimensions has changed.
    reshape_dimshuffle_mat2 = reshape(mat.dimshuffle("x", 1, "x", 0), mat.shape)
    h = FunctionGraph([mat], [reshape_dimshuffle_mat2], clone=False)
    assert len(h.apply_nodes) == 3
    useless_dimshuffle_in_reshape.rewrite(h)
    assert equal_computations(
        h.outputs, [reshape(mat.dimshuffle("x", 1, "x", 0), mat.shape)]
    )


class TestFusion:
    rewrites = RewriteDatabaseQuery(
        include=[
            "canonicalize",
            "fusion",
            "inplace",
        ],
        exclude=["cxx_only", "BlasOpt"],
    )
    mode = Mode(get_default_mode().linker, rewrites)
    _shared = staticmethod(shared)
    topo_exclude = ()

    def my_init(dtype="float64", num=0):
        return np.zeros((5, 5), dtype=dtype) + num

    fw, fx, fy, fz = (
        tensor(dtype="float32", shape=(None,) * 2, name=n) for n in "wxyz"
    )
    dw, dx, dy, dz = (
        tensor(dtype="float64", shape=(None,) * 2, name=n) for n in "wxyz"
    )
    ix, iy, iz = (tensor(dtype="int32", shape=(None,) * 2, name=n) for n in "xyz")
    fv = fvector("v")
    fs = fscalar("s")
    fwv = my_init("float32", 1)
    fxv = my_init("float32", 2)
    fyv = my_init("float32", 3)
    fzv = my_init("float32", 4)
    fvv = _asarray(np.random.random(5), dtype="float32")
    fsv = np.asarray(np.random.random(), dtype="float32")
    dwv = my_init("float64", 5)
    ixv = _asarray(my_init(num=60), dtype="int32")
    iyv = _asarray(my_init(num=70), dtype="int32")
    izv = _asarray(my_init(num=70), dtype="int32")
    fwx = fw + fx
    ftanx = tan(fx)

    def large_fuseable_graph(self, n):
        factors = []
        sd = dscalar()
        means = dvector()

        cst_05 = pt.constant(0.5)
        cst_m05 = pt.constant(-0.5)
        cst_2 = pt.constant(2)
        cst_m2 = pt.constant(-2)
        ones = pt.constant(np.ones(10))

        for i in range(n):
            f = cst_m05 * sd**cst_m2 * (ones - means[i]) ** cst_2 + cst_05 * log(
                cst_05 * (sd**cst_m2) / np.pi
            )
            factors.append(pt_sum(f))

        logp = add(*factors)

        vars = [sd, means]
        dlogp = [pytensor.grad(logp, v) for v in vars]
        return vars, dlogp

    @pytest.mark.parametrize(
        "case",
        [
            (
                fx + fy + fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + fzv,
                "float32",
            ),  # 0
            (
                fx * fy * fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv * fyv * fzv,
                "float32",
            ),  # 1
            (
                fx + fy * fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv * fzv,
                "float32",
            ),  # 2
            (
                fx * fy + fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv * fyv + fzv,
                "float32",
            ),  # 3
            (
                fw + fx + fy + fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),
            (
                (fw + fx) + (fy + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),  # 5
            (
                ((fw + fx) + fy) + fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),
            (
                (fw + (fx + fy)) + fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),
            (
                (fw + (fx + fy) + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),
            (
                fw + (fx + (fy + fz)),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),
            (
                (fw + fx) + (fy + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),  # 10
            (
                fw * fx * fy * fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv * fxv * fyv * fzv,
                "float32",
            ),
            (
                fw + fx * fy * fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv * fyv * fzv,
                "float32",
            ),
            (
                fx + fy * fz * fx,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv * fzv * fxv,
                "float32",
            ),
            (
                fx * fy + fz + fy,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv * fyv + fzv + fyv,
                "float32",
            ),
            (
                fx * fy * fz * fw + fx + fy + fz + fw,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fxv * fyv * fzv * fwv + fxv + fyv + fzv + fwv,
                "float32",
            ),  # 15
            # test with constant
            (
                (fw + fx) + (fy + fz) + 2.0,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),
            (
                ((fw + fx) + 2.0 + fy) + fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),
            (
                (fw + (fx + 2.0 + fy)) + fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),
            (
                (fw + (fx + fy) + 2 + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),
            (
                fw + (fx + (fy + fz) + 2.0),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),  # 20
            (
                2 + (fw + fx) + (fy + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),
            # mix float32 and float64
            (
                2 + (dw + fx) + (fy + fz),
                (dw, fx, fy, fz),
                (dwv, fxv, fyv, fzv),
                1,
                dwv + fxv + fyv + fzv + 2,
                "float64",
            ),
            (
                2 + (fw + dw) + (fy + fz),
                (fw, dw, fy, fz),
                (fwv, dwv, fyv, fzv),
                1,
                fwv + dwv + fyv + fzv + 2,
                "float64",
            ),
            (
                2 + (fw + fx) + (dw + fz),
                (fw, fx, dw, fz),
                (fwv, fxv, dwv, fzv),
                1,
                fwv + fxv + dwv + fzv + 2,
                "float64",
            ),
            (
                2 + (fw + fx) + (fy + dw),
                (fw, fx, fy, dw),
                (fwv, fxv, fyv, dwv),
                1,
                fwv + fxv + fyv + dwv + 2,
                "float64",
            ),  # 25
            # test when their is other op then elemwise.
            (
                (fwx.sum()) + (fwx) + (fy + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                4,
                (fwv + fxv).sum() + fwv + fxv + fyv + fzv,
                "float32",
            ),
            # test other elemwise op
            (
                fx + fy + cos(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.cos(fzv),
                "float32",
            ),
            (
                fx + fy + cosh(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.cosh(fzv),
                "float32",
            ),
            (
                fx + fy + abs(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.absolute(fzv),
                "float32",
            ),
            (
                ix + iy + abs(iz),
                (ix, iy, iz),
                (ixv, iyv, izv),
                1,
                ixv + iyv + np.absolute(izv),
                "int32",
            ),  # 30
            (
                fx + fy + log(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.log(fzv),
                "float32",
            ),
            (
                fx + fy + log2(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.log2(fzv),
                "float32",
            ),
            (
                fx + fy + log10(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.log10(fzv),
                "float32",
            ),
            (
                fx + fy**fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv**fzv,
                "float32",
            ),  # pow
            (
                fx + fy + exp(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.exp(fzv),
                "float32",
            ),  # 35
            (
                fx - fy - fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv - fzv,
                "float32",
            ),
            (
                fx - (fy / fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv / fzv),
                "float32",
            ),
            (
                fx - true_div(fy, 2),
                (fx, fy),
                (fxv, fyv),
                1,
                fxv - (fyv / 2),
                "float32",
            ),
            (
                fx - true_div(fy, fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv / fzv),
                "float32",
            ),
            (
                fx - int_div(ix * 100, iy * 1000),
                (fx, ix, iy),
                (fxv, ixv, iyv),
                1,
                fxv - ((ixv * 100) // (iyv * 1000)),
                {
                    "custom": "float64",
                    "numpy + floatX": config.floatX,
                    "numpy": "float64",
                },
            ),  # 40
            (fx - (fy / 2), (fx, fy), (fxv, fyv), 1, fxv - (fyv / 2), "float32"),
            (
                fx - (fy % fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv % fzv),
                "float32",
            ),
            (
                fx - (fy > fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv > fzv),
                "float32",
            ),
            (
                fx - (fy >= fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv >= fzv),
                "float32",
            ),
            (
                fx - (fy < fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv < fzv),
                "float32",
            ),  # 45
            (
                fx - (fy <= fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv <= fzv),
                "float32",
            ),
            (
                fx - eq(fy, fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv == fzv),
                "float32",
            ),
            (
                fx - neq(fy, fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv != fzv),
                "float32",
            ),
            (
                fx - fy + tan(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.tan(fzv),
                "float32",
            ),
            (
                fx - fy + tanh(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.tanh(fzv),
                "float32",
            ),  # 50
            (
                fx - fy + sin(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.sin(fzv),
                "float32",
            ),
            (
                fx - fy + sinh(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.sinh(fzv),
                "float32",
            ),
            (
                fx - fy + sqr(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + (fzv * fzv),
                "float32",
            ),
            (
                fx - fy + sqrt(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.sqrt(fzv),
                "float32",
            ),
            (
                fx - fy + reciprocal(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + (1 / fzv),
                "float32",
            ),  # 55
            (
                fx - fy + neg(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + (-fzv),
                "float32",
            ),
            (
                fx - fy + pt_round(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.round(fzv),
                "float32",
            ),
            (
                ix - iy + iround(fz),
                (ix, iy, fz),
                (ixv, iyv, fzv),
                1,
                ixv - iyv + np.round(fzv),
                "int64",
            ),
            # Bit op
            (
                fx - bitwise_or(iy, iz),
                (fx, iy, iz),
                (fxv, iyv, izv),
                1,
                fxv - (iyv | izv),
                {
                    "custom": "float64",
                    "numpy + floatX": config.floatX,
                    "numpy": "float64",
                },
            ),
            (
                fx - xor(iy, iz),
                (fx, iy, iz),
                (fxv, iyv, izv),
                1,
                fxv - (iyv ^ izv),
                {
                    "custom": "float64",
                    "numpy + floatX": config.floatX,
                    "numpy": "float64",
                },
            ),  # 60
            (
                fx - bitwise_and(iy, iz),
                (fx, iy, iz),
                (fxv, iyv, izv),
                1,
                fxv - (iyv & izv),
                {
                    "custom": "float64",
                    "numpy + floatX": config.floatX,
                    "numpy": "float64",
                },
            ),
            (
                fx - invert(iy),
                (fx, iy),
                (fxv, iyv),
                1,
                fxv - (~iyv),
                {
                    "custom": "float64",
                    "numpy + floatX": config.floatX,
                    "numpy": "float64",
                },
            ),
            (
                fx - pt.cast(fy, dtype="float64"),
                (fx, fy),
                (fxv, fyv),
                1,
                fxv - np.asarray(fyv, "float64"),
                "float64",
            ),
            (
                pt_pow(fx * fy + fz, fx * fy),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                np.power(fxv * fyv + fzv, fxv * fyv),
                "float32",
            ),
            (
                fv + fy**fz,
                (fv, fy, fz),
                (fvv, fyv, fzv),
                2,
                fvv + fyv**fzv,
                "float32",
            ),  # fused with a dimshuffle #65
            (
                fv - fy + tanh(fz),
                (fv, fy, fz),
                (fvv, fyv, fzv),
                2,
                fvv - fyv + np.tanh(fzv),
                "float32",
            ),  # fused with a dimshuffle
            # Cases where the same input is reused many times.
            (
                mul(fx, fx, fx, fx),
                (fx,),
                (fxv,),
                1,
                fxv * fxv * fxv * fxv,
                "float32",
            ),
            (
                mul(fx, ftanx, ftanx),
                (fx,),
                (fxv,),
                1,
                fxv * np.tan(fxv) * np.tan(fxv),
                "float32",
            ),
            (
                mul(fx, ftanx, ftanx, fx),
                (fx,),
                (fxv,),
                1,
                fxv * np.tan(fxv) * np.tan(fxv) * fxv,
                "float32",
                1e-5,
            ),
            (
                mul(ftanx, ftanx, fx + fy),
                (fx, fy),
                (fxv, fyv),
                1,
                np.tan(fxv) * np.tan(fxv) * (fxv + fyv),
                "float32",
                1e-5,
            ),  # 70
            # Cases with different broadcast pattern. They should not
            # be merged as this would duplicate computation
            # The graph should have 2 elemwise and 1 dimshuffle
            (
                fx * sin(fs),
                (fx, fs),
                (fxv, fsv),
                3,
                fxv * np.sin(fsv),
                "float32",
            ),
            # Multiple output cases  # 72
            (
                (
                    # sum(logp)
                    pt_sum(-((fx - fy) ** 2) / 2),
                    # grad(logp)
                    pt.grad(pt_sum(-((fx - fy) ** 2) / 2), wrt=fx),
                ),
                (fx, fy),
                (fxv, fyv),
                2,
                (
                    np.sum(-((fxv - fyv) ** 2) / 2),
                    -(fxv - fyv),
                ),
                ("float32", "float32"),
            ),
            # Two Composite graphs that share the same input, but are split by
            # a non-elemwise operation (Assert)
            (
                (
                    log(
                        ge(
                            assert_op(
                                pt_abs(fx),
                                pt_all(ge(pt_abs(fx), 0)),
                            ),
                            0,
                        )
                    ),
                ),
                (fx,),
                (fxv,),
                4,
                (np.zeros_like(fxv),),
                ("float32",),
            ),
            # Two subgraphs that share the same non-fuseable input, but are otherwise
            # completely independent
            (
                (
                    true_div(
                        mul(
                            pt_sum(fx + 5),  # breaks fusion
                            exp(fx),
                        ),
                        (fx + 5),
                    ),
                ),
                (fx,),
                (fxv,),
                4,
                (np.sum(fxv + 5) * np.exp(fxv) / (fxv + 5),),
                ("float32",),
            ),
            pytest.param(
                (
                    (sin(exp(fx)), exp(sin(fx))),
                    (fx,),
                    (fxv,),
                    1,
                    (np.sin(np.exp(fxv)), np.exp(np.sin(fxv))),
                    ("float32", "float32"),
                ),
                marks=pytest.mark.xfail,  # Not implemented yet
            ),
        ],
    )
    def test_elemwise_fusion(self, case, nb_repeat=1, assert_len_topo=True):
        """Verify that `Elemwise` fusion works."""

        if len(case) == 6:
            g, sym_inputs, val_inputs, nb_elemwise, answer, out_dtype = case
            atol = None
        else:
            g, sym_inputs, val_inputs, nb_elemwise, answer, out_dtype, atol = case

        if isinstance(out_dtype, dict):
            out_dtype = out_dtype[config.cast_policy]

        if not isinstance(g, tuple | list):
            g = (g,)
            answer = (answer,)
            out_dtype = (out_dtype,)

        if self._shared is None:
            f = function(list(sym_inputs), g, mode=self.mode)
            for x in range(nb_repeat):
                out = f(*val_inputs)
            if not isinstance(out, list):
                out = (out,)
        else:
            out = [
                self._shared(np.zeros((5,) * g_.ndim, dtype=od), "out")
                for g_, od in zip(g, out_dtype, strict=True)
            ]
            assert all(o.dtype == g_.dtype for o, g_ in zip(out, g, strict=True))
            f = function(
                sym_inputs, [], updates=list(zip(out, g, strict=True)), mode=self.mode
            )
            for x in range(nb_repeat):
                f(*val_inputs)
            out = [o.get_value() for o in out]

        if atol is None:
            atol = 1e-8
            if any(o == "float32" for o in out_dtype):
                atol = 1e-6

        for o, a in zip(out, answer, strict=True):
            np.testing.assert_allclose(o, a * nb_repeat, atol=atol)

        topo = f.maker.fgraph.toposort()
        topo_ = [n for n in topo if not isinstance(n.op, self.topo_exclude)]
        if assert_len_topo:
            assert len(topo_) == nb_elemwise

            if nb_elemwise == 1:
                # if no variable appears multiple times in the
                # input of g,
                # check that the number of input to the Composite
                # Elemwise is ok
                for g_ in g:
                    if len(set(g_.owner.inputs)) == len(g_.owner.inputs):
                        expected_len_sym_inputs = sum(
                            not isinstance(x, Constant) for x in topo_[0].inputs
                        )
                        assert expected_len_sym_inputs == len(sym_inputs)

        for od, o in zip(out_dtype, out, strict=True):
            assert od == o.dtype

    def test_fusion_35_inputs(self):
        r"""Make sure we don't fuse too many `Op`\s and go past the 31 function arguments limit."""
        inpts = vectors(["i%i" % i for i in range(35)])

        # Make an elemwise graph looking like:
        # sin(i34 + sin(i33 + sin(... i1 + sin(i0) ...)))
        out = sin(inpts[0])
        for idx in range(1, 35):
            out = sin(inpts[idx] + out)

        with config.change_flags(cxx=""):
            f = function(inpts, out, mode=self.mode)

        # Make sure they all weren't fused
        composite_nodes = [
            node
            for node in f.maker.fgraph.toposort()
            if isinstance(getattr(node.op, "scalar_op", None), ps.basic.Composite)
        ]
        assert not any(len(node.inputs) > 31 for node in composite_nodes)

    @pytest.mark.skipif(not config.cxx, reason="No cxx compiler")
    def test_big_fusion(self):
        # Make sure that C compilation is used
        mode = Mode("cvm", self.rewrites)
        dlogp = function(*self.large_fuseable_graph(n=85), mode=mode)

        # Make sure something was fused
        assert any(
            isinstance(getattr(node.op, "scalar_op", None), ps.basic.Composite)
            for node in dlogp.maker.fgraph.toposort()
        )

    @pytest.mark.xfail(reason="Fails due to #1244")
    def test_add_mul_fusion_precedence(self):
        """Test that additions and multiplications are "fused together" before
        a `Composite` `Op` is introduced. This fusion is done by canonicalization
        """
        x, y, z = vectors("x", "y", "z")
        out = log((x + y + z) / (x * y * z))
        f = pytensor.function([x, y, z], out, mode=self.mode)
        # There should be a single Composite Op
        nodes = f.maker.fgraph.apply_nodes
        assert len(nodes) == 1
        (node,) = nodes
        assert isinstance(node.op, Elemwise)
        scalar_op = node.op.scalar_op
        assert isinstance(scalar_op, Composite)
        assert [node.op for node in scalar_op.fgraph.toposort()] == [
            # There should be a single mul
            ps.mul,
            # There should be a single add
            ps.add,
            ps.true_div,
            ps.log,
        ]

    def test_add_mul_fusion_inplace(self):
        x, y, z = dmatrices("xyz")
        out = dot(x, y) + x + y + z

        f = function([x, y, z], out, mode=self.mode)
        topo = list(f.maker.fgraph.toposort())
        assert len(topo) == 2
        assert topo[-1].op.inplace_pattern

        new_out = f.maker.fgraph.outputs[0]
        assert isinstance(new_out.owner.op, Elemwise)
        assert isinstance(new_out.owner.op.scalar_op, ps.basic.Add)
        assert len(new_out.owner.inputs) == 4

        # TODO: Do we really need to do this?
        _ = f(
            np.random.random((5, 5)), np.random.random((5, 5)), np.random.random((5, 5))
        )

    def test_fusion_multiout_inplace(self):
        x = vector("x")

        # Create Composite where inplacing the first non-constant output would corrupt the second output
        xs = ps.float64("xs")
        outs = (
            Elemwise(Composite([xs], [xs + 1, ps.cos(xs + 1) + xs]))
            .make_node(x)
            .outputs
        )

        f = pytensor.function(
            [In(x, mutable=True)],
            outs,
            mode=self.mode.including("inplace"),
        )
        (composite_node,) = f.maker.fgraph.apply_nodes

        # Destroy map must be None or the last toposorted output
        destroy_map = composite_node.op.destroy_map
        assert (destroy_map == {}) or (
            destroy_map == {1: [composite_node.inputs.index(x)]}
        )

        res = f([0, 1, 2])
        assert np.allclose(res[0], [1, 2, 3])
        assert np.allclose(res[1], np.cos([1, 2, 3]) + np.array([0, 1, 2]))

    @pytest.mark.skipif(not config.cxx, reason="No cxx compiler")
    def test_no_c_code(self):
        r"""Make sure we avoid fusions for `Op`\s without C code implementations."""

        # This custom `Op` has no `c_code` method
        class NoCCodeOp(ps.basic.UnaryScalarOp):
            def impl(self, x):
                return x * 2

        no_c_code_op = Elemwise(NoCCodeOp(ps.basic.upgrade_to_float))

        mode = Mode(linker="cvm")
        mode._optimizer = mode._optimizer.including(
            "fusion",
            "canonicalize",
            "inplace",
        )

        x = vector()
        out = x * no_c_code_op(x + 1)
        f = function([x], out, mode=mode)

        assert not any(
            isinstance(getattr(n.op, "scalar_op"), ps.basic.Composite)
            for n in f.maker.fgraph.toposort()
        )

    @pytest.mark.parametrize("test_value", [np.c_[[1.0]], np.c_[[]]])
    def test_test_values(self, test_value):
        """Make sure that `local_elemwise_fusion_op` uses test values correctly
        when they have zero dimensions.
        """
        x, y, z = dmatrices("xyz")

        x.tag.test_value = test_value
        y.tag.test_value = test_value
        z.tag.test_value = test_value

        with config.change_flags(
            compute_test_value="raise", compute_test_value_opt="raise"
        ):
            out = x * y + z
            f = function([x, y, z], out, mode=self.mode)

        # Confirm that the fusion happened
        assert isinstance(f.maker.fgraph.outputs[0].owner.op.scalar_op, Composite)
        assert len(f.maker.fgraph.toposort()) == 1

        assert np.array_equal(
            f.maker.fgraph.outputs[0].tag.test_value,
            np.full_like(test_value, 2.0),
        )

    @pytest.mark.parametrize("linker", ["cvm", "py"])
    @pytest.mark.parametrize("inp_dtype", ("floatX", "int32"))
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1), (0, 1, 2)])
    @pytest.mark.parametrize(
        "careduce_op, numpy_op",
        [
            (pt_sum, np.sum),
            pytest.param(
                pt_all,
                np.all,
                marks=pytest.mark.xfail(
                    reason="Rewrite logic does not support all CAReduce"
                ),
            ),
        ],
    )
    def test_CAReduce_single_input(
        self, linker, inp_dtype, axis, careduce_op, numpy_op
    ):
        """Make sure that `CAReduce` and `Elemwise` fusions work with a single input."""

        mode = Mode(linker=linker)
        mode._optimizer = mode._optimizer.including(
            "local_careduce_fusion",
            "canonicalize",
            "inplace",
        )

        x = tensor(dtype=inp_dtype, shape=(None, None, None), name="x")
        out = careduce_op(exp(x), axis=axis)

        out_fn = function([x], out, mode=mode)

        if linker != "py":
            (out_node,) = out_fn.maker.fgraph.toposort()
            assert isinstance(getattr(out_node.op, "scalar_op"), ps.basic.Composite)

            rng = np.random.default_rng(2320)
            x_val = rng.random((4, 3, 2)).astype(x.type.dtype)

            exp_res = numpy_op(np.exp(x_val), axis=axis)

            out_val = out_fn(x_val)
            assert out_val.shape == exp_res.shape
            assert np.allclose(out_val, exp_res)
        else:
            out_nodes = out_fn.maker.fgraph.toposort()
            assert not any(
                isinstance(out_node.op.scalar_op, ps.basic.Composite)
                for out_node in out_nodes
                if hasattr(out_node.op, "scalar_op")
            )

        # `Elemwise`s with more than one client shouldn't be rewritten
        x = tensor(dtype="floatX", shape=(None, None, None), name="x")
        exp_x = exp(x)
        out = careduce_op(exp_x, axis=axis) + exp(x)

        out_fn = function([x], out, mode=mode)
        out_nodes = out_fn.maker.fgraph.toposort()
        assert not any(
            isinstance(out_node.op.scalar_op, ps.basic.Composite)
            for out_node in out_nodes
            if hasattr(out_node.op, "scalar_op")
        )

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.parametrize("linker", ["cvm", "py"])
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1), (0, 1, 2)])
    def test_CAReduce_multiple_inputs(self, linker, axis):
        """Make sure that `CAReduce` and `Elemwise` fusions work with multiple inputs."""

        mode = Mode(linker=linker)
        mode._optimizer = mode._optimizer.including(
            "local_careduce_fusion",
            "canonicalize",
            "inplace",
        )

        x = tensor(dtype="floatX", shape=(None, None, None), name="x")
        y = tensor(dtype="floatX", shape=(None, None, None), name="y")
        out = (x + y).sum(axis=axis)

        out_fn = function([x, y], out, mode=mode)
        (out_node,) = out_fn.maker.fgraph.toposort()

        assert isinstance(getattr(out_node.op, "scalar_op"), ps.basic.Composite)

        rng = np.random.default_rng(2320)
        x_val = rng.random((4, 3, 2), dtype=config.floatX)
        y_val = rng.random((4, 3, 2), dtype=config.floatX)
        exp_res = (x_val + y_val).sum(axis=axis)
        out_val = out_fn(x_val, y_val)
        assert out_val.shape == exp_res.shape
        assert np.allclose(out_val, exp_res)

    def test_not_fusing_broadcasted_subgraphs(self):
        """Test that broadcasted Elemwise subgraphs are not fused in a single Elemwise Composite Op.

        There are some cases in self.test_elemwise_fusion, but this test confirms that the
        fused subgraphs are exactly the expected ones.
        """
        xs = vector("xm")
        xm = matrix("xs")

        es = log(xs + 5)
        em = exp(xm * 5)
        esm = es - em

        f = pytensor.function([xs, xm], esm, mode=self.mode)
        apply_nodes = f.maker.fgraph.toposort()
        assert len(apply_nodes) == 3
        assert isinstance(apply_nodes[0].op, DimShuffle)
        # Inner Vector output Composite
        assert isinstance(apply_nodes[1].op.scalar_op, Composite)
        assert {node.op for node in apply_nodes[1].op.scalar_op.fgraph.apply_nodes} == {
            ps.add,
            ps.log,
        }
        # Outer Matrix output Composite
        assert isinstance(apply_nodes[2].op.scalar_op, Composite)
        assert {node.op for node in apply_nodes[2].op.scalar_op.fgraph.apply_nodes} == {
            ps.sub,
            ps.exp,
            ps.mul,
        }

    def test_multiple_outputs_fused_root_elemwise(self):
        """Test that a root elemwise output (single layer) is reused when
        there is another fused output"""

        # By default, we do not introduce Composite for single layers of Elemwise
        x = pt.vector("x")
        out1 = pt.cos(x)
        f = pytensor.function([x], out1, mode=self.mode)
        nodes = tuple(f.maker.fgraph.apply_nodes)
        assert len(nodes) == 1
        assert isinstance(nodes[0].op.scalar_op, ps.Cos)

        # However, when it can be composed with another output, we should not
        # compute that root Elemwise twice
        out2 = pt.log(out1)
        f = pytensor.function([x], [out1, out2], mode=self.mode)
        nodes = tuple(f.maker.fgraph.apply_nodes)
        assert len(nodes) == 1
        assert isinstance(nodes[0].op.scalar_op, Composite)

    def test_eval_benchmark(self, benchmark):
        rng = np.random.default_rng(123)
        size = 100_000
        x = pytensor.shared(rng.normal(size=size), name="x")
        mu = pytensor.shared(rng.normal(size=size), name="mu")

        logp = -((x - mu) ** 2) / 2
        grad_logp = grad(logp.sum(), x)

        func = pytensor.function([], [logp, grad_logp], mode="FAST_RUN")
        benchmark(func)

    @pytest.mark.skipif(not config.cxx, reason="No cxx compiler")
    def test_rewrite_benchmark(self, benchmark):
        inps, outs = self.large_fuseable_graph(n=25)
        fg = FunctionGraph(inps, outs)
        opt = FusionOptimizer()

        def rewrite_func():
            nb_replacement = opt.apply(fg.clone())[2]
            return nb_replacement

        assert benchmark(rewrite_func) == 103

    def test_no_warning_from_old_client(self):
        # There used to be a warning issued when creating fuseable mapping
        # for nodes that are no longer in the FunctionGraph
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # The -2 integer array cannot be passed directly to the C method
            # of log1mexp as that can only handle floats. There is a rewrite
            # that casts it to a float, but the FunctionGraph client retains
            # the original log1mexp of the integer input, which caused
            # a misleading warning for non C implementation in the FusionRewrite
            assert np.isclose(
                log1mexp(np.array(-2, dtype="int64")).eval(),
                np.log(1 - np.exp(-2)),
            )


class TimesN(ps.basic.UnaryScalarOp):
    """
    Used in test TestCompositeCodegen

    Must be outside of the class, otherwise, the c cache code can't
    pickle this class and this cause stuff printing during test.
    """

    def __eq__(self, other):
        return super().__eq__(other) and self.n == other.n

    def __hash__(self):
        return super().__hash__() ^ hash(self.n)

    def __init__(self, n, *args, **kwargs):
        self.n = n
        ps.basic.UnaryScalarOp.__init__(self, *args, **kwargs)

    def impl(self, x):
        return x * self.n

    def c_support_code_apply(self, node, nodename):
        n = str(self.n)
        return f"""
        float {nodename}_timesn(float x) {{ return x * {n}; }}
        """

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return f"{z} = {name}_timesn({x});"


class TestCompositeCodegen:
    """
    Test The Composite Ops code generation in a case where there is multiple
    scalar ops with support code.
    """

    def setup_method(self):
        upgrade_to_float = ps.basic.upgrade_to_float

        self.scal_times_2 = TimesN(2, upgrade_to_float, name="times_2")
        self.times_2 = Elemwise(self.scal_times_2, name="times_2")

        self.scal_times_3 = TimesN(3, upgrade_to_float, name="times_3")
        self.times_3 = Elemwise(self.scal_times_3, name="times_3")

        self.x = fvector()

    def test_nested_composite(self):
        y = self.times_2(self.x)
        z = self.times_3(y)
        f = function([self.x], z)
        if config.mode != "FAST_COMPILE":
            assert len(f.maker.fgraph.toposort()) == 1
        fval = f([1, 2, 3])
        assert np.all(fval == [6, 12, 18])


def test_local_useless_composite_outputs():
    x = ps.float32()
    y = ps.float32()
    z = ps.float32()
    c = ps.Composite([x, y, z], [x + 1, y - 1])
    X = matrix("X")
    Y = matrix("Y")
    Z = matrix("Z")
    o1, o2 = Elemwise(scalar_op=c)(X, Y, Z)
    mode = get_default_mode().including("local_useless_composite")

    f = function([X, Y, Z], [o1, o2], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert len(topo[0].inputs) == 2
    assert len(topo[0].outputs) == 2
    res1, res2 = f([[1.0]], [[1.0]], [[np.nan]])
    utt.assert_allclose(res1, [[2.0]])
    utt.assert_allclose(res2, [[0.0]])

    f = function([X, Y, Z], o1, mode=mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert len(topo[0].inputs) == 1
    assert len(topo[0].outputs) == 1
    utt.assert_allclose(f([[1.0]], [[np.nan]], [[np.nan]]), [[2.0]])

    f = function([X, Y, Z], o2, mode=mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert len(topo[0].inputs) == 1
    assert len(topo[0].outputs) == 1
    utt.assert_allclose(f([[np.nan]], [[1.0]], [[np.nan]]), [[0.0]])


@pytest.mark.parametrize("const_shape", [(), (1,), (5,), (1, 5), (2, 5)])
@pytest.mark.parametrize("op, np_op", [(pt.pow, np.power), (pt.add, np.add)])
def test_local_inline_composite_constants(op, np_op, const_shape):
    const = np.full(shape=const_shape, fill_value=2.5).astype(config.floatX)
    x = vector("x")
    y = vector("y")
    out = pt.exp(op(x, const)) + y

    fn = pytensor.function(
        [x, y], out, mode=get_default_mode().including("specialize", "fusion")
    )
    # There should be a single Composite after optimization
    [node] = [
        node for node in fn.maker.fgraph.apply_nodes if isinstance(node.op, Elemwise)
    ]
    assert isinstance(node.op.scalar_op, Composite)
    assert len(node.inputs) == 2  # x and y, but not const

    x_test_value = np.arange(5).astype(config.floatX)
    y_test_value = np.ones(5).astype(config.floatX)
    np.testing.assert_allclose(
        fn(x_test_value, y_test_value),
        np.exp(np_op(x_test_value, const)) + y_test_value,
    )


def test_local_useless_dimshuffle_makevector():
    a = scalar()
    x = MakeVector(config.floatX)(a)
    y = x.dimshuffle(())

    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)

    y_rewritten_fg = rewrite_graph(
        y_fg,
        clone=False,
        include=["canonicalize", "local_useless_dimshuffle_makevector"],
    )

    assert y_rewritten_fg.outputs[0] == a
