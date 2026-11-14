# Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np

from pytensor.link.numba.dispatch.basic import numba_njit


CONVERGED, WARNING, FAILED = 0, 1, 2


@numba_njit(fastmath=False, inline="always")
def _evaluate_step(evaluate, x, p, args, alpha, trial_x):
    for j in range(x.size):
        trial_x[j] = x[j] + alpha * p[j]
    return evaluate(trial_x, *args)


@numba_njit(fastmath=False, inline="always")
def _dcsrch(
    evaluate,
    x,
    p,
    args,
    f0,
    g0,
    d0,
    alpha,
    c1,
    c2,
    xtol,
    stepmin,
    stepmax,
    maxiter,
    trial_x,
):
    if not stepmin <= alpha <= stepmax:
        return 0.0, f0, g0, FAILED, 0

    bracketed, stage1 = False, True
    stx, fx, dx = 0.0, f0, d0
    sty, fy, dy = 0.0, f0, d0
    stmin, stmax = 0.0, 5 * alpha
    width, previous_width = stepmax - stepmin, 2 * (stepmax - stepmin)
    gtest = c1 * d0
    for evaluation in range(maxiter):
        # Like SciPy, only a non-finite step fails. A non-finite f or slope flows
        # through _dcstep's IEEE arithmetic, which steps back inside the domain.
        if not np.isfinite(alpha):
            return 0.0, f0, g0, FAILED, evaluation
        f, g = _evaluate_step(evaluate, x, p, args, alpha, trial_x)
        d = np.dot(g, p)
        ftest = f0 + alpha * gtest
        if f <= ftest and abs(d) <= -c2 * d0:
            return alpha, f, g, CONVERGED, evaluation + 1
        if (
            (
                bracketed
                and (alpha <= stmin or alpha >= stmax or stmax - stmin <= xtol * stmax)
            )
            or (alpha == stepmax and f <= ftest and d <= gtest)
            or (alpha == stepmin and (f > ftest or d >= gtest))
        ):
            return alpha, f, g, WARNING, evaluation + 1

        if stage1 and f <= ftest and d >= 0:
            stage1 = False
        if stage1 and f <= fx and f > ftest:
            # Search the modified function until sufficient decrease is obtained.
            stx, fx, dx, sty, fy, dy, alpha, bracketed = _dcstep(
                stx,
                fx - stx * gtest,
                dx - gtest,
                sty,
                fy - sty * gtest,
                dy - gtest,
                alpha,
                f - alpha * gtest,
                d - gtest,
                bracketed,
                stmin,
                stmax,
            )
            fx += stx * gtest
            fy += sty * gtest
            dx += gtest
            dy += gtest
        else:
            stx, fx, dx, sty, fy, dy, alpha, bracketed = _dcstep(
                stx, fx, dx, sty, fy, dy, alpha, f, d, bracketed, stmin, stmax
            )

        if bracketed:
            if abs(sty - stx) >= 0.66 * previous_width:
                alpha = stx + 0.5 * (sty - stx)
            previous_width, width = width, abs(sty - stx)
            stmin, stmax = min(stx, sty), max(stx, sty)
        else:
            stmin = alpha + 1.1 * (alpha - stx)
            stmax = alpha + 4 * (alpha - stx)
        alpha = min(max(alpha, stepmin), stepmax)
        if bracketed and (
            alpha <= stmin or alpha >= stmax or stmax - stmin <= xtol * stmax
        ):
            alpha = stx

    return 0.0, f0, g0, FAILED, maxiter


@numba_njit(fastmath=False, error_model="numpy")
def _dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    """Select a safeguarded cubic/secant step and update the search interval."""
    sgn_dp = np.sign(dp)
    sgn_dx = np.sign(dx)
    sgnd = sgn_dp * sgn_dx
    if fp > fx:
        # A higher function value brackets the minimum.
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - dx / s * (dp / s))
        if stp < stx:
            gamma *= -1
        p = gamma - dx + theta
        q = gamma - dx + gamma + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + dx / ((fx - fp) / (stp - stx) + dx) / 2.0 * (stp - stx)
        if abs(stpc - stx) <= abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2.0
        brackt = True
    elif sgnd < 0.0:
        # Opposite derivative signs also bracket the minimum.
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - dx / s * (dp / s))
        if stp > stx:
            gamma *= -1
        p = gamma - dp + theta
        q = gamma - dp + gamma + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + dp / (dp - dx) * (stx - stp)
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True
    elif abs(dp) < abs(dx):
        # The slope is flattening; extrapolate unless already bracketed.
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt(max(0, (theta / s) ** 2 - dx / s * (dp / s)))
        if stp > stx:
            gamma = -gamma
        p = gamma - dp + theta
        q = gamma + (dx - dp) + gamma
        r = p / q
        if r < 0 and gamma != 0:
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + dp / (dp - dx) * (stx - stp)
        if brackt:
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            if stp > stx:
                stpf = min(stp + 0.66 * (sty - stp), stpf)
            else:
                stpf = max(stp + 0.66 * (sty - stp), stpf)
        else:
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = min(max(stpf, stpmin), stpmax)
    elif brackt:
        theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
        s = max(abs(theta), abs(dy), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - dy / s * (dp / s))
        if stp > sty:
            gamma = -gamma
        p = gamma - dp + theta
        q = gamma - dp + gamma + dy
        r = p / q
        stpc = stp + r * (sty - stp)
        stpf = stpc
    elif stp > stx:
        stpf = stpmax
    else:
        stpf = stpmin
    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp
    stp = stpf
    return (stx, fx, dx, sty, fy, dy, stp, brackt)


# Ports of SciPy's _cubicmin, _quadmin, _zoom and scalar_search_wolfe2
# (scipy/optimize/_linesearch.py). NaN stands in for SciPy's None. The numpy
# error model turns SciPy's ArithmeticError cases into non-finite results,
# which the isfinite checks reject the same way.
@numba_njit(fastmath=False, error_model="numpy")
def _cubicmin(a, fa, fpa, b, fb, c, fc):
    db = b - a
    dc = c - a
    denom = (db * dc) ** 2 * (db - dc)
    t0 = fb - fa - fpa * db
    t1 = fc - fa - fpa * dc
    cubic = (dc**2 * t0 - db**2 * t1) / denom
    quadratic = (db**3 * t1 - dc**3 * t0) / denom
    radical = quadratic * quadratic - 3 * cubic * fpa
    xmin = a + (-quadratic + np.sqrt(radical)) / (3 * cubic)
    return xmin if np.isfinite(xmin) else np.nan


@numba_njit(fastmath=False, error_model="numpy")
def _quadmin(a, fa, fpa, b, fb):
    db = b - a
    curvature = (fb - fa - fpa * db) / (db * db)
    xmin = a - fpa / (2.0 * curvature)
    return xmin if np.isfinite(xmin) else np.nan


@numba_njit(fastmath=False, inline="always")
def _zoom(
    evaluate,
    x,
    p,
    args,
    a_lo,
    a_hi,
    phi_lo,
    phi_hi,
    derphi_lo,
    f0,
    d0,
    g0,
    c1,
    c2,
    trial_x,
):
    phi_rec, a_rec = f0, 0.0
    # SciPy's zoom evaluates until i > maxiter, with maxiter = 10.
    for i in range(11):
        dalpha = a_hi - a_lo
        a, b = (a_hi, a_lo) if dalpha < 0 else (a_lo, a_hi)
        a_j = np.nan
        if i > 0:
            cchk = 0.2 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
            if a_j > b - cchk or a_j < a + cchk:
                a_j = np.nan
        if np.isnan(a_j):
            qchk = 0.1 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if np.isnan(a_j) or a_j > b - qchk or a_j < a + qchk:
                a_j = a_lo + 0.5 * dalpha

        f, g = _evaluate_step(evaluate, x, p, args, a_j, trial_x)
        if f > f0 + c1 * a_j * d0 or f >= phi_lo:
            phi_rec, a_rec = phi_hi, a_hi
            a_hi, phi_hi = a_j, f
        else:
            d = np.dot(g, p)
            if abs(d) <= -c2 * d0:
                return a_j, f, g, True
            if d * (a_hi - a_lo) >= 0:
                phi_rec, a_rec = phi_hi, a_hi
                a_hi, phi_hi = a_lo, phi_lo
            else:
                phi_rec, a_rec = phi_lo, a_lo
            a_lo, phi_lo, derphi_lo = a_j, f, d

    return 0.0, f0, g0, False


@numba_njit(fastmath=False, inline="always")
def _line_search_wolfe2(evaluate, x, p, args, f0, g0, d0, alpha, c1, c2, trial_x):
    alpha0, alpha1 = 0.0, alpha
    phi_a0, derphi_a0 = f0, d0
    f, g = _evaluate_step(evaluate, x, p, args, alpha1, trial_x)
    for i in range(10):
        if f > f0 + c1 * alpha1 * d0 or (f >= phi_a0 and i > 0):
            return _zoom(
                evaluate,
                x,
                p,
                args,
                alpha0,
                alpha1,
                phi_a0,
                f,
                derphi_a0,
                f0,
                d0,
                g0,
                c1,
                c2,
                trial_x,
            )
        d = np.dot(g, p)
        if abs(d) <= -c2 * d0:
            return alpha1, f, g, True
        if d >= 0:
            return _zoom(
                evaluate,
                x,
                p,
                args,
                alpha1,
                alpha0,
                f,
                phi_a0,
                d,
                f0,
                d0,
                g0,
                c1,
                c2,
                trial_x,
            )
        alpha0, alpha1 = alpha1, min(2 * alpha1, 1e100)
        phi_a0, derphi_a0 = f, d
        f, g = _evaluate_step(evaluate, x, p, args, alpha1, trial_x)

    # SciPy accepts the last trial step when the doubling phase runs out.
    return alpha1, f, g, True
