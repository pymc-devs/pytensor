"""Random variables.

Random numbers come from the call context's random state (DESIGN.md, section 5a): the
graph's random generator values are tokens without data, so a RandomVariable node has
no generator argument and its next-generator output has no value.

The draws follow the output's shape: `size` when given, else the parameters' broadcast
batch shape (by the static types, like Elemwise). Distributions that are transforms of
uniform or standard normal numbers draw those in one call into scratch memory and then
apply the parameters in a loop the vectorizer can handle; the others call a sampler per
element.
"""

from __future__ import annotations

from pytensor.link.llvm.codegen.elemwise import output_array
from pytensor.link.llvm.codegen.emit import Array, c_strides, cast, for_each, load, size, store
from pytensor.link.llvm.codegen.function import new_output, llvm_emit, llvm_outputs
from pytensor.link.llvm.codegen.types import ir, is_absent, itemsize
from pytensor.link.llvm.layout import Layout


def _register():
    import pytensor.tensor.random.basic as rb
    from pytensor.tensor.random.op import RandomVariable

    @llvm_outputs.register(RandomVariable)
    def _(op, node, in_layouts):
        # (next generator: no value, draws: new memory)
        return [new_output(o) for o in node.outputs]

    @llvm_emit.register(RandomVariable)
    def _(op, node, cg, inputs, outputs):
        spec = SAMPLERS.get(type(op))
        if spec is None or op.ndim_supp != 0:
            raise NotImplementedError(f"LLVM backend: no implementation of the random variable {op}")
        emit_random(op, node, cg, inputs, outputs, spec)


    # per distribution: the base draws ("normal", "uniform": filled in one call; None:
    # the per-element function draws), the per-element function (b, cg, base value,
    # parameter values) -> draw, the parameters' dtypes and checks
    def normal(b, cg, z, p):
        mu, sigma = p
        return b.fadd(mu, b.fmul(sigma, z))

    def halfnormal(b, cg, z, p):
        loc, scale = p
        return b.fadd(loc, b.fmul(scale, b.intrinsic(ir.Intrinsic.Fabs, [z])))

    def lognormal(b, cg, z, p):
        mean, sigma = p
        return b.intrinsic(ir.Intrinsic.Exp, [b.fadd(mean, b.fmul(sigma, z))])

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
        k = b.fptosi(b.intrinsic(ir.Intrinsic.Floor, [b.fmul(u, span)]), ir.i64)
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
            rb.NormalRV: Sampler("normal", normal, ["float64", "float64"], [None, ("ge", zero, "scale < 0")]),
            rb.HalfNormalRV: Sampler("normal", halfnormal, ["float64", "float64"], [None, ("ge", zero, "scale < 0")]),
            rb.LogNormalRV: Sampler("normal", lognormal, ["float64", "float64"], [None, ("ge", zero, "sigma < 0")]),
            rb.UniformRV: Sampler("uniform", uniform, ["float64", "float64"], [None, None]),
            rb.ExponentialRV: Sampler("uniform", exponential, ["float64"], [("ge", zero, "scale < 0")]),
            rb.BernoulliRV: Sampler("uniform", bernoulli, ["float64"], [("unit", None, "p < 0, p > 1 or p is NaN")]),
            rb.IntegersRV: Sampler("uniform", integers, ["int64", "int64"], [None, None], check=("lt", "low >= high")),
            rb.GammaRV: Sampler(None, gamma, ["float64", "float64"], [("gt", zero, "shape <= 0"), ("gt", zero, "scale <= 0")]),
            rb.InvGammaRV: Sampler(None, invgamma, ["float64", "float64"], [("gt", zero, "shape <= 0"), ("gt", zero, "scale <= 0")]),
            rb.BetaRV: Sampler(None, beta, ["float64", "float64"], [("gt", zero, "a <= 0"), ("gt", zero, "b <= 0")]),
            rb.PoissonRV: Sampler(None, poisson, ["float64"], [("ge", zero, "lam < 0")]),
            rb.BinomialRV: Sampler(None, binomial, ["int64", "float64"], [("ge", 0, "n < 0"), ("unit", None, "p < 0, p > 1 or p is NaN")]),
        }
    )


class Sampler:
    def __init__(self, base, draw, param_dtypes, checks, check=None):
        self.base = base
        self.draw = draw
        self.param_dtypes = param_dtypes
        #: per parameter: None or (kind, bound, message)
        self.checks = checks
        #: a check involving two parameters
        self.check = check


SAMPLERS: dict = {}


def emit_random(op, node, cg, inputs, outputs, spec: Sampler):
    b = cg.b
    size_var, params_vars = node.inputs[1], node.inputs[2:]
    size_arr, params = inputs[1], inputs[2:]
    outvar = node.outputs[1]
    nd = outvar.type.ndim
    static = outvar.type.shape

    # ---- the shape of the draws
    shape = []
    for a in range(nd):
        if static[a] is not None:
            shape.append(b.i64(static[a]))
        elif not is_absent(size_var.type):
            p = b.gep_bytes(size_arr.data, b.mul(b.i64(a), size_arr.strides[0], nsw=True))
            shape.append(cast(b, load(b, size_arr.dtype, p), size_arr.dtype, "int64"))
        else:
            n = None
            for var, x in zip(params_vars, params, strict=True):
                pa = a - (nd - var.type.ndim)
                if pa >= 0 and var.type.shape[pa] != 1:
                    n = x.shape[pa]
                    break
            shape.append(n if n is not None else b.i64(1))
    if not is_absent(size_var.type):
        for a in range(nd):
            if static[a] is not None:
                p = b.gep_bytes(size_arr.data, b.mul(b.i64(a), size_arr.strides[0], nsw=True))
                n = cast(b, load(b, size_arr.dtype, p), size_arr.dtype, "int64")
                cg.check(b.eq(n, shape[a]), f"{op}: size differs from the static shape")

    # ---- the parameters over the loop (right-aligned, broadcast by their types)
    views = []
    for k, (var, x) in enumerate(zip(params_vars, params, strict=True)):
        strides = []
        for a in range(nd):
            pa = a - (nd - var.type.ndim)
            if pa < 0 or var.type.shape[pa] == 1:
                strides.append(b.i64(0))
            else:
                if x.shape[pa].const_int is None or shape[a].const_int is None:
                    cg.check(b.eq(x.shape[pa], shape[a]), f"{op}: parameter {k} has a different length on axis {a}")
                strides.append(x.strides[pa])
        same_ndim = var.type.ndim == nd and all(s != 1 for s in var.type.shape)
        views.append(Array(x.dtype, x.data, list(shape), strides, x.layout if same_ndim else None))

    out = output_array(outputs[1], shape)
    arrays = [*views, out]

    # ---- base draws, all at once
    base = None
    if spec.base is not None:
        n = size(b, shape)
        buf = cg.temp_memory(b.mul(n, b.i64(itemsize("float64")), nsw=True))
        b.rng(ir.Rng.FillNormal if spec.base == "normal" else ir.Rng.FillUniform, cg.ctx, [buf, n])
        base = Array("float64", buf, list(shape), c_strides(b, "float64", shape), Layout.c(nd))
        arrays.append(base)

    def body(ptrs):
        vals = []
        for k, (dt, x) in enumerate(zip(spec.param_dtypes, views, strict=True)):
            v = cast(b, load(b, x.dtype, ptrs[k]), x.dtype, dt)
            check = spec.checks[k]
            if check is not None:
                cg.check(_ok(b, check, v), f"{op}: {check[2]}")
            vals.append(v)
        if spec.check is not None:
            cg.check(b.slt(vals[0], vals[1]), f"{op}: {spec.check[1]}")
        z = load(b, "float64", ptrs[len(views) + 1]) if base is not None else None
        v = spec.draw(b, cg, z, vals)
        src = "int64" if v.type == ir.i64 else "float64"
        store(b, out.dtype, cast(b, v, src, out.dtype), ptrs[len(views)])

    for_each(b, shape, arrays, body)


def _ok(b, check, v):
    kind, bound, _ = check
    if kind == "unit":
        ge = b.fcmp(ir.FCmp.OGE, v, b.f64(0.0))
        le = b.fcmp(ir.FCmp.OLE, v, b.f64(1.0))
        return b.and_(ge, le)
    if v.type == ir.i64:
        c = b.i64(int(bound))
        return b.icmp(ir.ICmp.SGE if kind == "ge" else ir.ICmp.SGT, v, c)
    c = b.f64(float(bound))
    return b.fcmp(ir.FCmp.OGE if kind == "ge" else ir.FCmp.OGT, v, c)


_register()
