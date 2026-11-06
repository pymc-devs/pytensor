from collections.abc import Callable
from hashlib import sha256
from typing import NamedTuple

import numpy as np

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    numba_funcify_and_cache_key,
    register_funcify_and_cache_key,
)
from pytensor.link.numba.dispatch.optimize.utils import (
    _SOLVER_JIT_OPTIONS,
    _make_value_at,
    _scalar_types,
    _with_input_dtype,
)
from pytensor.tensor.optimize import MinimizeScalarOp


# scipy's truncated constants: bracket's growth factor, Brent's golden-section step,
# and golden's interior-point ratio
GOLDEN_RATIO = 1.618034
GOLDEN_SECTION = 0.3819660
GOLDEN_RATIO_CONJUGATE = 0.61803399


def make_bracket_search(value_at, point, value, mixed):
    @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
    def bracket_search(points, args):
        # scipy's bracket and Brent.get_bracket_info. Where scipy raises on a bad
        # bracket, the bracket is reported invalid with NaN points instead.
        missing_point, missing_value = point(np.nan), value(np.nan)
        invalid = (
            missing_point,
            missing_point,
            missing_point,
            missing_value,
            missing_value,
            missing_value,
            False,
        )
        if points.shape[0] == 3:
            xa, xb, xc = points[0], points[1], points[2]
            if xa > xc:
                xa, xc = xc, xa
            if not (xa < xb and xb < xc):
                return invalid
            fa, fb, fc = value_at(xa, args), value_at(xb, args), value_at(xc, args)
            if not (fb < fa and fb < fc):
                return invalid
            return xa, xb, xc, fa, fb, fc, True
        if points.shape[0] != 2:
            return invalid

        xa, xb = points[0], points[1]
        fa, fb = value_at(xa, args), value_at(xb, args)
        if fa < fb:
            xa, xb = xb, xa
            fa, fb = fb, fa
        xc = xb + point(GOLDEN_RATIO) * (xb - xa)
        fc = value_at(xc, args)
        iteration = 0
        while fc < fb:
            tmp1 = (xb - xa) * (fb - fc)
            tmp2 = (xb - xc) * (fb - fa)
            val = tmp2 - tmp1
            denom = mixed(2.0 * 1e-21) if abs(val) < 1e-21 else mixed(2.0) * val
            w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
            wlim = xb + point(110.0) * (xc - xb)
            if iteration > 1000:
                return invalid
            iteration += 1
            if (w - xc) * (xb - w) > 0.0:
                fw = value_at(w, args)
                if fw < fc:
                    xa, xb = xb, w
                    fa, fb = fb, fw
                    break
                elif fw > fb:
                    xc, fc = w, fw
                    break
                w = xc + point(GOLDEN_RATIO) * (xc - xb)
                fw = value_at(w, args)
            elif (w - wlim) * (wlim - xc) >= 0.0:
                w = wlim
                fw = value_at(w, args)
            elif (w - wlim) * (xc - w) > 0.0:
                fw = value_at(w, args)
                if fw < fc:
                    xb, xc = xc, w
                    w = xc + point(GOLDEN_RATIO) * (xc - xb)
                    fb, fc = fc, fw
                    fw = value_at(w, args)
            else:
                w = xc + point(GOLDEN_RATIO) * (xc - xb)
                fw = value_at(w, args)
            xa, xb, xc = xb, xc, w
            fa, fb, fc = fb, fc, fw

        valid = (
            ((fb < fc and fb <= fa) or (fb < fa and fb <= fc))
            and (xa < xb < xc or xc < xb < xa)
            and np.isfinite(xa)
            and np.isfinite(xb)
            and np.isfinite(xc)
        )
        return xa, xb, xc, fa, fb, fc, valid

    return bracket_search


@numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
def _best_of_invalid_bracket(xa, xb, xc, fa, fb, fc):
    # The point scipy's _recover_from_bracket_error reports for a failed search
    points = np.array([xa, xb, xc])
    values = np.array([fa, fb, fc])
    if np.isnan(points).any() or np.isnan(values).any():
        return np.nan
    return points[np.argmin(values)]


def make_minimize_brent(objective, point, value, mixed):
    value_at = _make_value_at(objective=objective)
    bracket_search = make_bracket_search(
        value_at=value_at, point=point, value=value, mixed=mixed
    )

    @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
    def brent(points, args, xtol, maxiter):
        # scipy's Brent.optimize
        xa, xb, xc, fa, fb, fc, valid = bracket_search(points, args)
        if not valid:
            return _best_of_invalid_bracket(xa, xb, xc, fa, fb, fc), False

        x = w = v = xb
        fw = fv = fx = fb
        a, b = (xa, xc) if xa < xc else (xc, xa)
        deltax = rat = point(0.0)
        iteration = 0
        while iteration < maxiter:
            tol1 = xtol * abs(x) + point(1.0e-11)
            tol2 = point(2.0) * tol1
            xmid = point(0.5) * (a + b)
            if abs(x - xmid) < (tol2 - point(0.5) * (b - a)):
                break
            if abs(deltax) <= tol1:
                deltax = a - x if x >= xmid else b - x
                rat = point(GOLDEN_SECTION) * deltax
            else:
                tmp1 = (x - w) * (fx - fv)
                tmp2 = (x - v) * (fx - fw)
                p = (x - v) * tmp2 - (x - w) * tmp1
                tmp2 = mixed(2.0) * (tmp2 - tmp1)
                if tmp2 > 0.0:
                    p = -p
                tmp2 = abs(tmp2)
                dx_temp = deltax
                deltax = rat
                if (
                    p > tmp2 * (a - x)
                    and p < tmp2 * (b - x)
                    and abs(p) < abs(mixed(0.5) * tmp2 * dx_temp)
                ):
                    rat = p * mixed(1.0) / tmp2
                    u = x + rat
                    if (u - a) < tol2 or (b - u) < tol2:
                        rat = tol1 if xmid - x >= 0 else -tol1
                else:
                    deltax = a - x if x >= xmid else b - x
                    rat = point(GOLDEN_SECTION) * deltax

            if abs(rat) < tol1:
                u = x + tol1 if rat >= 0 else x - tol1
            else:
                u = x + rat
            fu = value_at(u, args)

            if fu > fx:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v, w = w, u
                    fv, fw = fw, fu
                elif fu <= fv or v == x or v == w:
                    v, fv = u, fu
            else:
                if u >= x:
                    a = x
                else:
                    b = x
                v, w, x = w, x, u
                fv, fw, fx = fw, fx, fu
            iteration += 1

        return x, iteration < maxiter and not (np.isnan(x) or np.isnan(fx))

    return brent


def make_minimize_golden(objective, point, value, mixed):
    value_at = _make_value_at(objective=objective)
    bracket_search = make_bracket_search(
        value_at=value_at, point=point, value=value, mixed=mixed
    )

    @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
    def golden(points, args, xtol, maxiter):
        # scipy's _minimize_scalar_golden
        xa, xb, xc, fa, fb, fc, valid = bracket_search(points, args)
        if not valid:
            return _best_of_invalid_bracket(xa, xb, xc, fa, fb, fc), False

        ratio = point(GOLDEN_RATIO_CONJUGATE)
        complement = point(1.0 - GOLDEN_RATIO_CONJUGATE)
        x3, x0 = xc, xa
        if abs(xc - xb) > abs(xb - xa):
            x1 = xb
            x2 = xb + complement * (xc - xb)
        else:
            x2 = xb
            x1 = xb - complement * (xb - xa)
        f1, f2 = value_at(x1, args), value_at(x2, args)

        iteration = 0
        for _ in range(maxiter):
            if abs(x3 - x0) <= xtol * (abs(x1) + abs(x2)):
                break
            if f2 < f1:
                x0, x1 = x1, x2
                x2 = ratio * x1 + complement * x3
                f1 = f2
                f2 = value_at(x2, args)
            else:
                x3, x2 = x2, x1
                x1 = ratio * x2 + complement * x0
                f2 = f1
                f1 = value_at(x1, args)
            iteration += 1

        xmin, fval = (x1, f1) if f1 < f2 else (x2, f2)
        return xmin, iteration < maxiter and not (np.isnan(fval) or np.isnan(xmin))

    return golden


def make_minimize_bounded(objective, point, value, mixed):
    value_at = _make_value_at(objective=objective)

    @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
    def bounded(points, args, xatol, maxiter):
        # scipy's _minimize_scalar_bounded
        if points.shape[0] != 2:
            return np.nan, False
        lower, upper = points[0], points[1]
        if not (np.isfinite(lower) and np.isfinite(upper)) or lower > upper:
            return np.nan, False

        # scipy takes these from NumPy's sqrt, so they are float64 whatever the bounds'
        # dtype, and so is everything they touch. Only the first sum and difference of
        # the bounds run in the bounds' dtype.
        sqrt_eps = np.sqrt(2.2e-16)
        golden_mean = 0.5 * (3.0 - np.sqrt(5.0))
        a, b = lower, upper
        fulc = lower + golden_mean * (upper - lower)
        nfc, xf = fulc, fulc
        rat = e = 0.0
        x = xf
        fx = value_at(x, args)
        num = 1
        fu = value(np.inf)

        ffulc = fnfc = fx
        xm = point(0.5) * (lower + upper)
        tol1 = sqrt_eps * abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        flag = 0
        while abs(xf - xm) > (tol2 - point(0.5) * (b - a)):
            golden = True
            if abs(e) > tol1:
                golden = False
                r = (xf - nfc) * (fx - ffulc)
                q = (xf - fulc) * (fx - fnfc)
                p = (xf - fulc) * q - (xf - nfc) * r
                q = mixed(2.0) * (q - r)
                if q > 0.0:
                    p = -p
                q = abs(q)
                r = e
                e = rat

                if (
                    (abs(p) < abs(mixed(0.5) * q * r))
                    and (p > q * (a - xf))
                    and (p < q * (b - xf))
                ):
                    rat = (p + mixed(0.0)) / q
                    x = xf + rat
                    if ((x - a) < tol2) or ((b - x) < tol2):
                        si = np.sign(xm - xf) + (1.0 if xm == xf else 0.0)
                        rat = tol1 * si
                else:
                    golden = True

            if golden:
                e = a - xf if xf >= xm else b - xf
                rat = golden_mean * e

            si = np.sign(rat) + (1.0 if rat == 0 else 0.0)
            x = xf + si * max(abs(rat), tol1)
            fu = value_at(x, args)
            num += 1

            if fu <= fx:
                if x >= xf:
                    a = xf
                else:
                    b = xf
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = xf, fx
                xf, fx = x, fu
            else:
                if x < xf:
                    a = x
                else:
                    b = x
                if (fu <= fnfc) or (nfc == xf):
                    fulc, ffulc = nfc, fnfc
                    nfc, fnfc = x, fu
                elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                    fulc, ffulc = x, fu

            xm = 0.5 * (a + b)
            tol1 = sqrt_eps * abs(xf) + xatol / 3.0
            tol2 = 2.0 * tol1

            if num >= maxiter:
                flag = 1
                break

        if np.isnan(xf) or np.isnan(fx) or np.isnan(fu):
            flag = 2
        return xf, flag == 0

    return bounded


class MinimizeMethod(NamedTuple):
    make_solver: Callable
    points_name: str
    defaults: dict


MINIMIZE_METHODS = {
    "brent": MinimizeMethod(
        make_minimize_brent, "bracket", {"xtol": 1.48e-8, "maxiter": 500}
    ),
    "golden": MinimizeMethod(
        make_minimize_golden,
        "bracket",
        {"xtol": np.sqrt(np.finfo(float).eps), "maxiter": 5000},
    ),
    "bounded": MinimizeMethod(
        make_minimize_bounded, "bounds", {"xatol": 1e-5, "maxiter": 500}
    ),
}


def minimize_options(op: MinimizeScalarOp) -> dict | None:
    """Return the options ``op`` minimizes with, scipy's defaults filled in, or None when they are not implemented."""
    if op.method not in MINIMIZE_METHODS:
        return None
    method = MINIMIZE_METHODS[op.method]
    defaults = method.defaults
    if "bounds" in op.bound_kwargs and method.points_name != "bounds":
        return None
    if op.method == "bounded" and "bounds" not in op.bound_kwargs:
        return None

    kwargs = dict(op.optimizer_kwargs)
    tol = kwargs.pop("tol", None)
    options = dict(kwargs.pop("options", None) or {})
    if kwargs:
        return None
    if tol is not None:
        # minimize_scalar already moved a bounded tol into xatol
        if op.method == "bounded":
            return None
        options.setdefault("xtol", tol)
    if options.keys() - defaults.keys():
        return None

    settings = defaults | options
    if not isinstance(settings["maxiter"], int | np.integer):
        return None
    if op.method == "brent" and settings["xtol"] < 0:
        return None
    return {
        name: (int(value) if name == "maxiter" else float(value))
        for name, value in settings.items()
    }


@register_funcify_and_cache_key(MinimizeScalarOp)
def numba_funcify_MinimizeScalarOp(op, node, **kwargs):
    settings = minimize_options(op=op)
    if settings is None:
        return generate_fallback_impl(op, node, **kwargs), None

    _ = kwargs.pop("storage_map", None)
    objective, objective_key = numba_funcify_and_cache_key(
        op.fgraph, fgraph_name="numba_minimize_scalar_objective", **kwargs
    )
    objective = _with_input_dtype(
        objective=objective, dtype=op.fgraph.inputs[0].type.dtype
    )

    method = MINIMIZE_METHODS[op.method]
    has_points = method.points_name in op.bound_kwargs
    # Without a bracket, scipy searches from Python floats, so in float64
    point_dtype = op.fgraph.inputs[0].type.dtype if has_points else "float64"
    point, value, mixed = _scalar_types(
        point_dtype=point_dtype, value_dtype=op.fgraph.outputs[0].type.dtype
    )
    solve = method.make_solver(
        objective=objective, point=point, value=value, mixed=mixed
    )
    maxiter = settings["maxiter"]
    # A Python float tolerance adopts the point dtype. scipy's default golden xtol is a
    # NumPy float64, and bounded divides xatol by 3.0 before it meets a point, so those
    # stay float64.
    user_options = op.optimizer_kwargs.get("options") or {}
    default_golden_xtol = (
        op.method == "golden"
        and "xtol" not in user_options
        and "tol" not in op.optimizer_kwargs
    )
    if op.method == "bounded":
        tolerance = settings["xatol"]
    elif default_golden_xtol:
        tolerance = settings["xtol"]
    else:
        tolerance = np.dtype(point_dtype).type(settings["xtol"])
    as_output_dtype = node.outputs[0].type.numpy_dtype.type
    n_args = len(op.inner_inputs) - 1

    if has_points:
        points_index = n_args + op.bound_kwargs.index(method.points_name)

        @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
        def points_of(inputs):
            return inputs[points_index]

    else:

        @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
        def points_of(inputs):
            return np.array([0.0, 1.0])

    @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
    def minimize_scalar(_x0, *inputs):
        x, success = solve(points_of(inputs), inputs[:n_args], tolerance, maxiter)
        return np.asarray(as_output_dtype(x)), success

    if objective_key is None:
        cache_key = None
    else:
        minimize_scalar_cache_version = 1
        cache_key = sha256(
            str(
                (
                    type(op),
                    op.method,
                    op.bound_kwargs,
                    sorted(settings.items()),
                    default_golden_xtol,
                    objective_key,
                    minimize_scalar_cache_version,
                )
            ).encode()
        ).hexdigest()
    return minimize_scalar, cache_key
