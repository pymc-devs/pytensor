"""Random variables.

Random numbers come from the call context's random state: the graph's random generator
values are tokens without data, so a RandomVariable node has
no generator argument and its next-generator output has no value.

A random variable reaches the backend wrapped with its core shape
(`RandomVariableWithCoreShape`, from `introduce_explicit_core_shape_rv`). Its batch
shape is `size` when given, else the parameters' broadcast batch shape (by the static
types, like Blockwise).

* A distribution we have a sampler for (`SAMPLERS`, all without core axes) is drawn
  here. Those that are transforms of uniform or standard normal numbers draw those in
  one call into scratch memory and then apply the parameters in a loop the vectorizer
  can handle; the others call a sampler per element.
* Any other is a loop over the batch (`batch_loop`, as a Blockwise's) whose core is
  one draw of the random variable without batch axes: numba's core function for it
  (`numba_core_rv_funcify`), drawing from the call's state (kernels/numba.py). The loop
  is serial: the draws come in the order numba makes them.
"""

from __future__ import annotations

import weakref

import numpy as np

import pytensor.scalar as ps
from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.codegen.array import Array, c_strides, size
from pytensor.link.llvm.codegen.interface import (
    llvm_emit,
    llvm_supports,
    supported_types,
)
from pytensor.link.llvm.codegen.num import cast, load, store
from pytensor.link.llvm.codegen.types import ir, is_absent
from pytensor.link.llvm.dispatch.blockwise import (
    Batched,
    batch_loop,
    batch_shape,
    core_layout,
    core_lengths,
    unwrapped,
    wrapped_positions,
)
from pytensor.tensor.basic import constant
from pytensor.tensor.type import tensor


def _register():
    import pytensor.tensor.random.basic as rb
    from pytensor.tensor.random.op import RandomVariableWithCoreShape

    @llvm_supports.register(RandomVariableWithCoreShape)
    def _(op, node) -> bool:
        return supported_types(node)

    @llvm_emit.register(RandomVariableWithCoreShape)
    def _(op, node, cg, inputs, outputs):
        rv = unwrapped(node)
        args = [inputs[k] for k in wrapped_positions(node)]
        sampler = SAMPLERS.get(type(rv.op))
        if sampler is not None and rv.op.ndim_supp == 0:
            emit_random(rv.op, rv, cg, args, outputs, sampler)
        else:
            emit_core_draws(rv, cg, args, outputs, core_shape=inputs[0])

    # per distribution: the base draws ("normal", "uniform": filled in one call; None:
    # the per-element function draws), the per-element function (b, cg, base value,
    # parameter values) -> draw, the parameters' types and checks
    def normal(b, cg, z, p):
        mu, sigma = p
        return b.fadd(mu, b.fmul(sigma, z))

    def halfnormal(b, cg, z, p):
        loc, scale = p
        return b.fadd(loc, b.fmul(scale, b.intrinsic(ir.Intrinsic.Fabs, [z])))

    def lognormal(b, cg, z, p):
        mean, sigma = p
        return b.libm(ir.Libm.Exp, [b.fadd(mean, b.fmul(sigma, z))])

    def uniform(b, cg, u, p):
        low, high = p
        return b.fadd(low, b.fmul(b.fsub(high, low), u))

    def exponential(b, cg, u, p):
        (scale,) = p
        # 1 - u is in (0, 1]
        return b.fneg(b.fmul(scale, b.libm(ir.Libm.Log1p, [b.fneg(u)])))

    def bernoulli(b, cg, u, p):
        (prob,) = p
        return b.zext(b.fcmp(ir.FCmp.OLT, u, prob), ir.i64)

    def integers(b, cg, u, p):
        low, high = p
        span = b.sitofp(b.sub(high, low), ir.f64)
        k = cast(
            b, b.intrinsic(ir.Intrinsic.Floor, [b.fmul(u, span)]), ps.float64, ps.int64
        )
        return b.add(low, k)

    def gamma(b, cg, _, p):
        shape, scale = p
        return b.fmul(b.rng(ir.Rng.Gamma, cg.ctx, [shape]), scale)

    def invgamma(b, cg, _, p):
        shape, scale = p
        return b.fdiv(scale, b.rng(ir.Rng.Gamma, cg.ctx, [shape]))

    def beta(b, cg, _, p):
        a, bb = p
        return b.rng(ir.Rng.Beta, cg.ctx, [a, bb])

    def poisson(b, cg, _, p):
        (lam,) = p
        return b.rng(ir.Rng.Poisson, cg.ctx, [lam])

    def binomial(b, cg, _, p):
        n, prob = p
        return b.rng(ir.Rng.Binomial, cg.ctx, [n, prob])

    zero = 0.0
    SAMPLERS.update(
        {
            rb.NormalRV: Sampler(
                "normal",
                normal,
                [ps.float64, ps.float64],
                [None, ("ge", zero, "scale < 0")],
            ),
            rb.HalfNormalRV: Sampler(
                "normal",
                halfnormal,
                [ps.float64, ps.float64],
                [None, ("ge", zero, "scale < 0")],
            ),
            rb.LogNormalRV: Sampler(
                "normal",
                lognormal,
                [ps.float64, ps.float64],
                [None, ("ge", zero, "sigma < 0")],
            ),
            rb.UniformRV: Sampler(
                "uniform", uniform, [ps.float64, ps.float64], [None, None]
            ),
            rb.ExponentialRV: Sampler(
                "uniform", exponential, [ps.float64], [("ge", zero, "scale < 0")]
            ),
            rb.BernoulliRV: Sampler(
                "uniform",
                bernoulli,
                [ps.float64],
                [("unit", None, "p < 0, p > 1 or p is NaN")],
            ),
            rb.IntegersRV: Sampler(
                "uniform",
                integers,
                [ps.int64, ps.int64],
                [None, None],
                check=("lt", "low >= high"),
            ),
            rb.GammaRV: Sampler(
                None,
                gamma,
                [ps.float64, ps.float64],
                [("gt", zero, "shape <= 0"), ("gt", zero, "scale <= 0")],
            ),
            rb.InvGammaRV: Sampler(
                None,
                invgamma,
                [ps.float64, ps.float64],
                [("gt", zero, "shape <= 0"), ("gt", zero, "scale <= 0")],
            ),
            rb.BetaRV: Sampler(
                None,
                beta,
                [ps.float64, ps.float64],
                [("gt", zero, "a <= 0"), ("gt", zero, "b <= 0")],
            ),
            rb.PoissonRV: Sampler(
                None, poisson, [ps.float64], [("ge", zero, "lam < 0")]
            ),
            rb.BinomialRV: Sampler(
                None,
                binomial,
                [ps.int64, ps.float64],
                [("ge", 0, "n < 0"), ("unit", None, "p < 0, p > 1 or p is NaN")],
            ),
        }
    )


class Sampler:
    def __init__(self, base, draw, param_types, checks, check=None):
        self.base = base
        self.draw = draw
        self.param_types = param_types
        #: per parameter: None or (kind, bound, message)
        self.checks = checks
        #: a check involving two parameters
        self.check = check


SAMPLERS: dict = {}


def draw_batch(cg, node, size_x, params) -> tuple[list, list[Batched]]:
    """The batch shape of the random variable `node` (i64 lengths): `size` when given
    (`size_x`, None if not), else the parameters' broadcast batch shape; and the
    parameters (`params`) as operands of a loop over it."""
    b = cg.b
    op = node.op
    batch_ndim = op.batch_ndim(node)
    operands = []
    for var, x, core_ndim in zip(
        op.dist_params(node), params, op.ndims_params, strict=True
    ):
        param_batch = var.type.ndim - core_ndim
        operands.append(
            Batched(x, var.type.broadcastable[:param_batch], batch_ndim - param_batch)
        )
    given = None
    if size_x is not None:
        static = node.outputs[1].type.shape
        given = []
        for a in range(batch_ndim):
            n = size_x.load(b, a).astype(ps.int64).value
            if static[a] is not None:
                cg.check(
                    b.eq(n, static[a]), f"{op}: size differs from the static shape"
                )
                n = b.i64(static[a])
            given.append(n)
    return batch_shape(cg, operands, batch_ndim, given, str(op)), operands


#: node -> the graph of its core draw (`_core_draw`)
_core_draws: weakref.WeakKeyDictionary[Apply, tuple[list, list]] = (
    weakref.WeakKeyDictionary()
)


def _core_draw(node) -> tuple[list, list]:
    """The graph of one draw of the random variable `node` without batch axes (and
    without size), as (inputs, outputs): the generator, then the parameters' core
    parts (a constant one that isn't batched stays a constant)."""
    if node not in _core_draws:
        op = node.op
        rng = op.rng_param(node).type()
        params: list[Variable] = []
        for var, core_ndim in zip(op.dist_params(node), op.ndims_params, strict=True):
            param_batch = var.type.ndim - core_ndim
            if isinstance(var, Constant) and all(var.type.broadcastable[:param_batch]):
                data = var.data.reshape(var.data.shape[param_batch:])
                params.append(constant(data, dtype=var.type.dtype))
            else:
                params.append(
                    tensor(dtype=var.type.dtype, shape=var.type.shape[param_batch:])
                )
        outputs = op.make_node(rng, None, *params).outputs
        _core_draws[node] = ([rng, *params], outputs)
    return _core_draws[node]


def emit_core_draws(node, cg, inputs, outputs, core_shape: Array) -> None:
    """The random variable `node`: a serial loop over its batch, drawing one core draw
    per element (`_core_draw`)."""
    op = node.op
    _, size_x, *params = inputs
    size_x = None if is_absent(op.size_param(node).type) else size_x
    batch, operands = draw_batch(cg, node, size_x, params)
    graph_inputs, graph_outputs = _core_draw(node)
    layouts = [None] + [
        core_layout(var, x.array.layout, len(x.broadcastable))
        for var, x in zip(op.dist_params(node), operands, strict=True)
    ]
    core = cg.graph_function(graph_inputs, graph_outputs, layouts, name="draw")
    batch_ndim = op.batch_ndim(node)
    shapes: list[list[ir.Value]] = [
        [],
        core_lengths(cg, node.outputs[1], batch_ndim, core_shape),
    ]
    batch_loop(cg, core, [None, *operands], outputs, batch, shapes, what=str(op))


def emit_random(op, node, cg, inputs, outputs, spec: Sampler):
    b = cg.b
    size_var, params_vars = node.inputs[1], node.inputs[2:]
    size_arr, params = inputs[1], inputs[2:]
    nd = node.outputs[1].type.ndim
    shape, _ = draw_batch(
        cg, node, None if is_absent(size_var.type) else size_arr, params
    )

    # ---- the parameters over the loop (right-aligned, broadcast by their types)
    views = []
    for var, x in zip(params_vars, params, strict=True):
        off = nd - var.type.ndim
        strides = [
            b.i64(0)
            if a < off or var.type.broadcastable[a - off]
            else x.strides[a - off]
            for a in range(nd)
        ]
        same_ndim = var.type.ndim == nd and all(s != 1 for s in var.type.shape)
        views.append(
            Array(
                x.scalar_type,
                x.data,
                list(shape),
                strides,
                x.layout
                if same_ndim
                else x.view_layout(
                    shape, strides, [a - off if a >= off else None for a in range(nd)]
                ),
            )
        )

    out = outputs[1].allocate(shape)
    arrays = [*views, out]

    # ---- base draws, all at once
    base = None
    if spec.base is not None:
        n = size(b, shape)
        buf = cg.temp_memory(b.mul(n, np.dtype(ps.float64.dtype).itemsize, nsw=True))
        b.rng(
            ir.Rng.FillNormal if spec.base == "normal" else ir.Rng.FillUniform,
            cg.ctx,
            [buf, n],
        )
        base = Array(
            ps.float64, buf, list(shape), c_strides(b, ps.float64, shape), Layout.c(nd)
        )
        arrays.append(base)

    def body(ptrs):
        values = []
        for k, (dt, x) in enumerate(zip(spec.param_types, views, strict=True)):
            v = cast(b, load(b, x.scalar_type, ptrs[k]), x.scalar_type, dt)
            check = spec.checks[k]
            if check is not None:
                cg.check(_ok(b, check, v), f"{op}: {check[2]}")
            values.append(v)
        if spec.check is not None:
            cg.check(b.slt(values[0], values[1]), f"{op}: {spec.check[1]}")
        z = load(b, ps.float64, ptrs[len(views) + 1]) if base is not None else None
        v = spec.draw(b, cg, z, values)
        src = ps.int64 if v.type == ir.i64 else ps.float64
        store(b, out.scalar_type, cast(b, v, src, out.scalar_type), ptrs[len(views)])

    cg.for_each(shape, arrays, body)


def _ok(b, check, v):
    kind, bound, _ = check
    if kind == "unit":
        ge = b.fcmp(ir.FCmp.OGE, v, 0.0)
        le = b.fcmp(ir.FCmp.OLE, v, 1.0)
        return b.and_(ge, le)
    if v.type == ir.i64:
        return b.icmp(ir.ICmp.SGE if kind == "ge" else ir.ICmp.SGT, v, int(bound))
    return b.fcmp(ir.FCmp.OGE if kind == "ge" else ir.FCmp.OGT, v, float(bound))


_register()
