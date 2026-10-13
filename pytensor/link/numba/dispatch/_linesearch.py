"""Numba port of the More-Thuente search in scipy.optimize._dcsrch."""

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


@numba_njit(fastmath=False, inline="always")
def _more_thuente(evaluate, x, p, args, f0, g0, d0, alpha, c1, c2):
    stepmin, stepmax, xtol = 1e-100, 1e100, 1e-14
    if not stepmin <= alpha <= stepmax:
        return 0.0, f0, g0, False

    bracketed, stage1 = False, True
    stx, fx, dx = 0.0, f0, d0
    sty, fy, dy = 0.0, f0, d0
    stmin, stmax = 0.0, 5 * alpha
    width, previous_width = stepmax - stepmin, 2 * (stepmax - stepmin)
    gtest = c1 * d0
    # SciPy's DCSRCH uses 100 iterations as a guard on adaptive step selection.
    for _ in range(100):
        if not np.isfinite(alpha):
            break
        f, g = evaluate(x + alpha * p, *args)
        d = np.dot(g, p)
        ftest = f0 + alpha * gtest
        if f <= ftest and abs(d) <= -c2 * d0:
            return alpha, f, g, True
        if (
            not np.isfinite(f)
            or not np.isfinite(d)
            or (
                bracketed
                and (alpha <= stmin or alpha >= stmax or stmax - stmin <= xtol * stmax)
            )
            or (alpha == stepmax and f <= ftest and d <= gtest)
            or (alpha == stepmin and (f > ftest or d >= gtest))
        ):
            break

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

    return 0.0, f0, g0, False


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
