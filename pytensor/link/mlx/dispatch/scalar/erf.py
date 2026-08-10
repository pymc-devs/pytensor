from collections.abc import Callable

import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.link.mlx.dispatch.scalar.helpers import _exp, _working_precision
from pytensor.scalar.math import Erfc, Erfcx


# W. J. Cody, "Rational Chebyshev Approximation for the Error Function" (Math. Comp. 23,
# 1969), in the arrangement used by the netlib SPECFUN CALERF routine. Three intervals:
# a series for erf below 0.46875, and two rationals that yield erfcx directly above it.
#
# Deriving all three from one kernel, rather than composing them out of mx.erf, is what
# makes them accurate: mx.erf is a float32 kernel whatever dtype it is handed. The two
# upper intervals are free of exp entirely, and where exp is unavoidable the family goes
# through _exp rather than mx.exp, which is float32 in range as well as in precision.
#
# The intervals are a matter of conditioning rather than precision. erfcx is the bounded
# quantity and is computed directly, because reaching it as exp(x**2) * erfc(x) overflows
# while erfc has decayed far past anything 1 - erf could resolve.
_ERF_A = (
    3.16112374387056560e00,
    1.13864154151050156e02,
    3.77485237685302021e02,
    3.20937758913846947e03,
    1.85777706184603153e-1,
)
_ERF_B = (
    2.36012909523441209e01,
    2.44024637934444173e02,
    1.28261652607737228e03,
    2.84423683343917062e03,
)
_ERFCX_C = (
    5.64188496988670089e-1,
    8.88314979438837594e00,
    6.61191906371416295e01,
    2.98635138197400131e02,
    8.81952221241769090e02,
    1.71204761263407058e03,
    2.05107837782607147e03,
    1.23033935479799725e03,
    2.15311535474403846e-8,
)
_ERFCX_D = (
    1.57449261107098347e01,
    1.17693950891312499e02,
    5.37181101862009858e02,
    1.62138957456669019e03,
    3.29079923573345963e03,
    4.36261909014324716e03,
    3.43936767414372164e03,
    1.23033935480374942e03,
)
_ERFCX_P = (
    3.05326634961232344e-1,
    3.60344899949804439e-1,
    1.25781726111229246e-1,
    1.60837851487422766e-2,
    6.58749161529837803e-4,
    1.63153871373020978e-2,
)
_ERFCX_Q = (
    2.56852019228982242e00,
    1.87295284992346047e00,
    5.27905102951428412e-1,
    6.05183413124413191e-2,
    2.33520497626869185e-3,
)
_SQRT_PI_INV = 5.6418958354775628695e-1
_ERF_THRESH = 0.46875
_ERFCX_SPLIT = 4.0


def _metal_constants(**arrays):
    """Render Python float tuples as Metal ``constant`` arrays, one per keyword."""
    return "\n".join(
        f"constant float {name}[{len(values)}] = {{"
        + ", ".join(f"{v!r}f" for v in values)
        + "};"
        for name, values in arrays.items()
    )


# Generating the coefficients rather than transcribing them keeps the two implementations
# from drifting apart, and emitting them as named Metal arrays keeps everything below this
# point ordinary Metal: only the numbers are generated, not the surrounding code.
#
# The kernel is worth having because it can branch. The vectorized path has to evaluate
# every interval for every element and select afterwards, which costs it an order of
# magnitude at the sizes where memory bandwidth dominates; a kernel just takes the branch.


_METAL_HEADER = (
    _metal_constants(
        ERF_A=_ERF_A,
        ERF_B=_ERF_B,
        ERFCX_C=_ERFCX_C,
        ERFCX_D=_ERFCX_D,
        ERFCX_P=_ERFCX_P,
        ERFCX_Q=_ERFCX_Q,
        ERF_SPLITS=(_ERF_THRESH, _ERFCX_SPLIT, _SQRT_PI_INV),
    )
    + """
#define ERF_THRESH   ERF_SPLITS[0]
#define ERFCX_SPLIT  ERF_SPLITS[1]
#define SQRT_PI_INV  ERF_SPLITS[2]

static inline float erf_series(float t) {
    float z = t * t;
    float num = ERF_A[4] * z;
    float den = z;
    for (int i = 0; i < 3; ++i) {
        num = (num + ERF_A[i]) * z;
        den = (den + ERF_B[i]) * z;
    }
    return t * (num + ERF_A[3]) / (den + ERF_B[3]);
}

static inline float erfcx_upper(float y) {
    if (y <= ERFCX_SPLIT) {
        float num = ERFCX_C[8] * y;
        float den = y;
        for (int i = 0; i < 7; ++i) {
            num = (num + ERFCX_C[i]) * y;
            den = (den + ERFCX_D[i]) * y;
        }
        return (num + ERFCX_C[7]) / (den + ERFCX_D[7]);
    }
    float z = 1.0f / (y * y);
    float num = ERFCX_P[5] * z;
    float den = z;
    for (int i = 0; i < 4; ++i) {
        num = (num + ERFCX_P[i]) * z;
        den = (den + ERFCX_Q[i]) * z;
    }
    float r = z * (num + ERFCX_P[4]) / (den + ERFCX_Q[4]);
    return (SQRT_PI_INV - r) / y;
}
"""
)


_METAL_ERFC_SOURCE = """
    uint i = thread_position_in_grid.x;
    float xv = (float)x[i];
    float y = fabs(xv);
    float res;
    if (y <= ERF_THRESH) {
        res = 1.0f - erf_series(y);
    } else {
        res = exp(-y * y) * erfcx_upper(y);
    }
    if (xv < 0.0f) res = 2.0f - res;
    out[i] = (T)res;
"""


_METAL_ERFCX_SOURCE = """
    uint i = thread_position_in_grid.x;
    float xv = (float)x[i];
    float y = fabs(xv);
    float res;
    if (y <= ERF_THRESH) {
        res = exp(y * y) * (1.0f - erf_series(y));
    } else {
        res = erfcx_upper(y);
    }
    if (xv < 0.0f) res = 2.0f * exp(y * y) - res;
    out[i] = (T)res;
"""


_METAL_SOURCES = {"erfc": _METAL_ERFC_SOURCE, "erfcx": _METAL_ERFCX_SOURCE}


_METAL_KERNELS: dict[str, Callable | None] = {}


def _metal_erf_kernel(name):
    """Build and cache the Metal kernel for ``name``, or None if it cannot be built."""
    if name not in _METAL_KERNELS:
        try:
            _METAL_KERNELS[name] = mx.fast.metal_kernel(
                name=f"pytensor_{name}",
                input_names=["x"],
                output_names=["out"],
                header=_METAL_HEADER,
                source=_METAL_SOURCES[name],
            )
        except Exception:
            _METAL_KERNELS[name] = None
    return _METAL_KERNELS[name]


def _metal_erf_call(name, x):
    """Evaluate ``name`` through its Metal kernel, or return None to fall back.

    Metal has no float64 and the kernel needs the GPU stream, which between them leave
    exactly the case the vectorized path handles least well and float64 needs least: the
    default device at single precision.
    """
    if (
        x.dtype != mx.float32
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
    ):
        return None
    kernel = _metal_erf_kernel(name)
    if kernel is None:
        return None
    # The kernel indexes a flat buffer, so a 0-d input has to be raised to 1-d before it
    # is passed: Metal binds a scalar as a value rather than a pointer, and subscripting
    # it fails to compile. Reshaping covers every rank at once
    flat = x.reshape(-1)
    (out,) = kernel(
        inputs=[flat],
        template=[("T", flat.dtype)],
        grid=(flat.size, 1, 1),
        threadgroup=(min(256, max(flat.size, 1)), 1, 1),
        output_shapes=[flat.shape],
        output_dtypes=[flat.dtype],
    )
    return out.reshape(x.shape)


def _erf_series(x, const):
    """erf on |x| <= 0.46875, where 1 - erfc would cancel. Free of exp."""
    z = x * x
    num = const(_ERF_A[4]) * z
    den = z
    for a, b in zip(_ERF_A[:3], _ERF_B[:3]):
        num = (num + const(a)) * z
        den = (den + const(b)) * z
    return x * (num + const(_ERF_A[3])) / (den + const(_ERF_B[3]))


def _erfcx_upper(y, const):
    """erfcx on y >= 0.46875, where neither branch needs exp.

    Callers must clamp ``y`` to the threshold themselves. Keeping this separate from
    ``_erfcx_positive`` is what lets ``erf`` and ``erfc`` skip the subunit branch, which
    their own clamps guarantee is never selected -- every branch is evaluated for every
    element, so an unreachable one is not free.
    """
    split = const(_ERFCX_SPLIT)

    # 0.46875 <= y <= 4
    mid_y = mx.minimum(y, split)
    num = const(_ERFCX_C[8]) * mid_y
    den = mid_y
    for c, d in zip(_ERFCX_C[:7], _ERFCX_D[:7]):
        num = (num + const(c)) * mid_y
        den = (den + const(d)) * mid_y
    mid = (num + const(_ERFCX_C[7])) / (den + const(_ERFCX_D[7]))

    # y > 4, as an asymptotic rational in 1 / y^2
    high_y = mx.maximum(y, split)
    z = const(1.0) / (high_y * high_y)
    num = const(_ERFCX_P[5]) * z
    den = z
    for p, q in zip(_ERFCX_P[:4], _ERFCX_Q[:4]):
        num = (num + const(p)) * z
        den = (den + const(q)) * z
    r = z * (num + const(_ERFCX_P[4])) / (den + const(_ERFCX_Q[4]))
    high = (const(_SQRT_PI_INV) - r) / high_y

    return mx.where(y <= split, mid, high)


def _erfc_tail(y, const):
    """erfc on y >= 0.46875. Callers must clamp ``y`` to the threshold themselves.

    Multiplying by ``exp(-y**2)`` underflows to zero smoothly, where ``1 - erf(y)``
    cancels to it abruptly and loses every significant digit on the way.
    """
    return _exp(-(y * y), const) * _erfcx_upper(y, const)


def _erfcx_positive(y, const):
    """erfcx on y >= 0, for non-negative ``y`` only."""
    thresh = const(_ERF_THRESH)
    # y < 0.46875 is the one branch that needs exp, and exp(y^2) <= 1.25 there
    small_y = mx.minimum(y, thresh)
    small = _exp(small_y * small_y, const) * (const(1.0) - _erf_series(small_y, const))
    return mx.where(y <= thresh, small, _erfcx_upper(mx.maximum(y, thresh), const))


def _erfc_positive(y, const):
    """erfc on y >= 0, for non-negative ``y`` only."""
    thresh = const(_ERF_THRESH)
    near = const(1.0) - _erf_series(mx.minimum(y, thresh), const)
    return mx.where(y <= thresh, near, _erfc_tail(mx.maximum(y, thresh), const))


@mlx_funcify.register(Erfc)
def mlx_funcify_Erfc(op, **kwargs):
    def erfc(x):
        x = mx.array(x)
        fast = _metal_erf_call("erfc", x)
        if fast is not None:
            return fast

        z, const, out_dtype = _working_precision(x)
        y = mx.abs(z)

        # Both branches evaluate erfc(|x|); erfc(-y) = 2 - erfc(y) restores the sign
        pos = _erfc_positive(y, const)
        out = mx.where(z >= const(0.0), pos, const(2.0) - pos)
        return out.astype(out_dtype)

    return erfc


@mlx_funcify.register(Erfcx)
def mlx_funcify_Erfcx(op, **kwargs):
    def erfcx(x):
        x = mx.array(x)
        fast = _metal_erf_call("erfcx", x)
        if fast is not None:
            return fast

        z, const, out_dtype = _working_precision(x)
        y = mx.abs(z)

        pos = _erfcx_positive(y, const)
        # erfcx(-y) = 2 exp(y^2) - erfcx(y). The exponential is unavoidable here and
        # genuinely overflows past y ~ 26.6, which is the function's own behavior
        reflected = const(2.0) * _exp(y * y, const) - pos
        return mx.where(z >= const(0.0), pos, reflected).astype(out_dtype)

    return erfcx
