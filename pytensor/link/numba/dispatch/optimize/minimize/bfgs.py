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
from pytensor.link.numba.dispatch.optimize.minimize.line_search import (
    CONVERGED,
    _dcsrch,
)


@numba_njit(fastmath=False)
def _vecnorm(x, order):
    if order == np.inf:
        return np.max(np.abs(x))
    if order == -np.inf:
        return np.min(np.abs(x))
    return np.sum(np.abs(x) ** order) ** (1.0 / order)


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
    evaluate, x, p, args, a_lo, a_hi, phi_lo, phi_hi, derphi_lo, f0, d0, g0, c1, c2
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

        f, g = evaluate(x + a_j * p, *args)
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
def _line_search_wolfe2(evaluate, x, p, args, f0, g0, d0, alpha, c1, c2):
    alpha0, alpha1 = 0.0, alpha
    phi_a0, derphi_a0 = f0, d0
    f, g = evaluate(x + alpha1 * p, *args)
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
            )
        d = np.dot(g, p)
        if abs(d) <= -c2 * d0:
            return alpha1, f, g, True
        if d >= 0:
            return _zoom(
                evaluate, x, p, args, alpha1, alpha0, f, phi_a0, d, f0, d0, g0, c1, c2
            )
        alpha0, alpha1 = alpha1, min(2 * alpha1, 1e100)
        phi_a0, derphi_a0 = f, d
        f, g = evaluate(x + alpha1 * p, *args)

    # SciPy accepts the last trial step when the doubling phase runs out.
    return alpha1, f, g, True


@numba_njit(fastmath=False, inline="always")
def _line_search_wolfe12(evaluate, x, p, args, f0, previous_f, g0, c1, c2):
    d0 = np.dot(g0, p)
    if not np.isfinite(d0) or d0 >= 0:
        return 0.0, f0, g0, False

    alpha = min(1.0, 2.02 * (f0 - previous_f) / d0)
    # SciPy keeps a zero step, which fails both of its searches.
    if alpha <= 0 or not np.isfinite(alpha):
        alpha = 1.0
    # SciPy's line_search_wolfe12 treats DCSRCH warnings as failures.
    step, f, g, status, _ = _dcsrch(
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
        1e-14,
        1e-100,
        1e100,
        100,
        np.empty_like(x),
    )
    if status == CONVERGED:
        return step, f, g, True
    return _line_search_wolfe2(evaluate, x, p, args, f0, g0, d0, alpha, c1, c2)


@numba_njit(fastmath=False, inline="always")
def _bfgs(evaluate, x, args, gtol, maxiter, norm, xrtol, c1, c2):
    f, g = evaluate(x, *args)
    if maxiter is None:
        maxiter = x.size * 200
    if maxiter <= 0 or not np.isfinite(f) or not np.all(np.isfinite(g)):
        return x, False
    if _vecnorm(g, norm) <= gtol:
        return x, True

    identity = np.eye(x.size)
    h = identity
    updated = False
    previous_f = f + np.linalg.norm(g) / 2
    # Like SciPy's k, steps counts accepted steps only. Restarts cannot run away:
    # a second failure before the next accepted update returns.
    steps = 0
    while steps < maxiter:
        p = -np.dot(h, g)
        alpha, new_f, new_g, found = _line_search_wolfe12(
            evaluate, x, p, args, f, previous_f, g, c1, c2
        )
        if not found:
            if not updated:
                return x, False
            # Like SciPy's L-BFGS-B, discard the curvature history and retry
            # with steepest descent before giving up.
            h = identity
            updated = False
            continue

        steps += 1
        s = alpha * p
        descent = -np.dot(g, s)
        x = x + s
        y = new_g - g
        previous_f, f, g = f, new_f, new_g
        if _vecnorm(g, norm) <= gtol:
            return x, steps < maxiter
        if np.linalg.norm(s) <= xrtol * (xrtol + np.linalg.norm(x)):
            return x, steps < maxiter
        if not np.isfinite(f) or not np.all(np.isfinite(g)):
            return x, False

        curvature = np.dot(y, s)
        if curvature <= np.finfo(np.float64).eps * descent:
            continue
        if not updated:
            # Scale the identity to the observed curvature (Nocedal & Wright
            # eq. 6.20). Otherwise the update must cancel O(1) entries down to
            # 1 / curvature, and rounding can leave h indefinite.
            h = identity * (curvature / np.dot(y, y))
            updated = True
        # Expanded form of SciPy's (I - rho s y') h (I - rho y s') + rho s s',
        # which costs O(n^2) instead of two n x n matrix products. It updates h
        # in place, which is safe because h was rebound to a fresh scaled array
        # when updated became True.
        rho = 1.0 / curvature
        hy = np.dot(h, y)
        w = (0.5 * rho * rho * (curvature + np.dot(y, hy))) * s - rho * hy
        for i in range(x.size):
            for j in range(x.size):
                h[i, j] = (h[i, j] + s[i] * w[j]) + w[i] * s[j]

    return x, False
