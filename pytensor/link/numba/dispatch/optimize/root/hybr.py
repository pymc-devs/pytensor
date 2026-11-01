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
#
# Minpack Copyright Notice (1999) University of Chicago.  All rights reserved
#
# Redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above
# copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials
# provided with the distribution.
#
# 3. The end-user documentation included with the
# redistribution, if any, must include the following
# acknowledgment:
#
#    "This product includes software developed by the
#    University of Chicago, as Operator of Argonne National
#    Laboratory.
#
# Alternately, this acknowledgment may appear in the software
# itself, if and wherever such third-party acknowledgments
# normally appear.
#
# 4. WARRANTY DISCLAIMER. THE SOFTWARE IS SUPPLIED "AS IS"
# WITHOUT WARRANTY OF ANY KIND. THE COPYRIGHT HOLDER, THE
# UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND
# THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE
# OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY
# OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR
# USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF
# THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4)
# DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION
# UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL
# BE CORRECTED.
#
# 5. LIMITATION OF LIABILITY. IN NO EVENT WILL THE COPYRIGHT
# HOLDER, THE UNITED STATES, THE UNITED STATES DEPARTMENT OF
# ENERGY, OR THEIR EMPLOYEES: BE LIABLE FOR ANY INDIRECT,
# INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF
# ANY KIND OR NATURE, INCLUDING BUT NOT LIMITED TO LOSS OF
# PROFITS OR LOSS OF DATA, FOR ANY REASON WHATSOEVER, WHETHER
# SUCH LIABILITY IS ASSERTED ON THE BASIS OF CONTRACT, TORT
# (INCLUDING NEGLIGENCE OR STRICT LIABILITY), OR OTHERWISE,
# EVEN IF ANY OF SAID PARTIES HAS BEEN WARNED OF THE
# POSSIBILITY OF SUCH LOSS OR DAMAGES.

import numpy as np

from pytensor.link.numba.dispatch.basic import numba_njit
from pytensor.link.numba.dispatch.optimize.root.minpack import (
    EPSMCH,
    GIANT,
    _enorm,
    _qrfac,
    _same_point,
)


@numba_njit(fastmath=False, error_model="numpy")
def _qform(q, n, wa):
    # Accumulate the orthogonal factor stored by _qrfac in the first n columns
    # of q into the full m x m matrix.
    m = q.shape[0]
    minmn = min(m, n)
    for j in range(1, minmn):
        for i in range(j):
            q[i, j] = 0.0
    for j in range(n, m):
        for i in range(m):
            q[i, j] = 0.0
        q[j, j] = 1.0

    for l in range(minmn):
        k = minmn - l - 1
        for i in range(k, m):
            wa[i] = q[i, k]
            q[i, k] = 0.0
        q[k, k] = 1.0
        if wa[k] != 0.0:
            # Slice views keep the loops vectorizable, as in _qrfac.
            reflector = wa[k:]
            for j in range(k, m):
                column = q[k:, j]
                ssum = 0.0
                for i in range(column.size):
                    ssum += column[i] * reflector[i]
                temp = ssum / wa[k]
                for i in range(column.size):
                    column[i] -= temp * reflector[i]


@numba_njit(fastmath=False, error_model="numpy")
def _dogleg(r, diag, qtb, delta, x, wa1, wa2):
    # Dogleg step within the trust region; r is the upper triangle stored by
    # rows. Slice views keep the loops vectorizable, as in _qrfac.
    n = diag.size
    jj = (n * (n + 1)) // 2
    for k in range(n):
        j = n - k - 1
        jj -= k + 1
        row, tail = r[jj + 1 : jj + n - j], x[j + 1 :]
        ssum = 0.0
        for i in range(row.size):
            ssum += row[i] * tail[i]
        temp = r[jj]
        if temp == 0.0:
            l = j
            for i in range(j + 1):
                temp = np.fmax(temp, abs(r[l]))
                l += n - i - 1
            temp = EPSMCH * temp
            if temp == 0.0:
                temp = EPSMCH
        x[j] = (qtb[j] - ssum) / temp

    for j in range(n):
        wa1[j] = 0.0
        wa2[j] = diag[j] * x[j]
    qnorm = _enorm(wa2)
    if qnorm <= delta:
        return

    l = 0
    for j in range(n):
        temp = qtb[j]
        row, tail = r[l : l + n - j], wa1[j:]
        for i in range(row.size):
            tail[i] += row[i] * temp
        l += n - j
        wa1[j] /= diag[j]

    gnorm = _enorm(wa1)
    sgnorm = 0.0
    alpha = delta / qnorm
    if gnorm != 0.0:
        for j in range(n):
            wa1[j] /= gnorm
            wa1[j] /= diag[j]
        l = 0
        for j in range(n):
            row, tail = r[l : l + n - j], wa1[j:]
            ssum = 0.0
            for i in range(row.size):
                ssum += row[i] * tail[i]
            l += n - j
            wa2[j] = ssum
        temp = _enorm(wa2)
        sgnorm = (gnorm / temp) / temp

        alpha = 0.0
        if sgnorm < delta:
            bnorm = _enorm(qtb)
            temp = (bnorm / gnorm) * (bnorm / qnorm) * (sgnorm / delta)
            temp = (
                temp
                - (delta / qnorm) * (sgnorm / delta) ** 2.0
                + np.sqrt(
                    (temp - (delta / qnorm)) ** 2.0
                    + ((1.0 - (delta / qnorm) ** 2.0) * (1.0 - (sgnorm / delta) ** 2.0))
                )
            )
            alpha = ((delta / qnorm) * (1.0 - (sgnorm / delta) ** 2.0)) / temp

    temp = (1.0 - alpha) * np.fmin(sgnorm, delta)
    for j in range(n):
        x[j] = temp * wa1[j] + alpha * x[j]


@numba_njit(fastmath=False, error_model="numpy")
def _r1mpyq(a, v, w):
    # Apply the Givens rotations stored in v and w to the columns of a.
    m, n = a.shape
    if n < 2:
        return
    for k in range(n - 1):
        j = n - k - 2
        if abs(v[j]) > 1.0:
            ccos = 1 / v[j]
            ssin = np.sqrt(1.0 - ccos * ccos)
        else:
            ssin = v[j]
            ccos = np.sqrt(1.0 - ssin * ssin)
        for i in range(m):
            temp = ccos * a[i, j] - ssin * a[i, n - 1]
            a[i, n - 1] = ssin * a[i, j] + ccos * a[i, n - 1]
            a[i, j] = temp

    for j in range(n - 1):
        if abs(w[j]) > 1.0:
            ccos = 1 / w[j]
            ssin = np.sqrt(1 - ccos * ccos)
        else:
            ssin = w[j]
            ccos = np.sqrt(1 - ssin * ssin)
        for i in range(m):
            temp = ccos * a[i, j] + ssin * a[i, n - 1]
            a[i, n - 1] = -ssin * a[i, j] + ccos * a[i, n - 1]
            a[i, j] = temp


@numba_njit(fastmath=False, error_model="numpy")
def _r1updt(m, n, s, u, v, w):
    # Rank-one update of the lower trapezoidal s (stored by columns, 1-based
    # offsets as in MINPACK). Returns whether the updated factor is singular.
    jj = (n * (2 * m - n + 1)) // 2 - (m - n)
    l = jj
    for i in range(n, m + 1):
        w[i - 1] = s[l - 1]
        l += 1

    nm1 = n - 1
    for nmj in range(1, nm1 + 1):
        j = n - nmj
        jj -= m - j + 1
        w[j - 1] = 0.0
        if v[j - 1] != 0.0:
            if abs(v[n - 1]) < abs(v[j - 1]):
                ccotan = v[n - 1] / v[j - 1]
                ssin = 0.5 / np.sqrt(0.25 + 0.25 * (ccotan * ccotan))
                ccos = ssin * ccotan
                tau = 1.0
                if abs(ccos) * GIANT > 1.0:
                    tau = 1.0 / ccos
            else:
                ttan = v[j - 1] / v[n - 1]
                ccos = 0.5 / np.sqrt(0.25 + 0.25 * (ttan * ttan))
                ssin = ccos * ttan
                tau = ssin
            v[n - 1] = ssin * v[j - 1] + ccos * v[n - 1]
            v[j - 1] = tau
            # Slice views keep the loops vectorizable, as in _qrfac.
            column, tail = s[jj - 1 : jj + m - j], w[j - 1 : m]
            for i in range(column.size):
                temp = ccos * column[i] - ssin * tail[i]
                tail[i] = ssin * column[i] + ccos * tail[i]
                column[i] = temp

    for i in range(m):
        w[i] += v[n - 1] * u[i]

    sing = False
    for j in range(1, nm1 + 1):
        if w[j - 1] != 0.0:
            if abs(s[jj - 1]) < abs(w[j - 1]):
                ccotan = s[jj - 1] / w[j - 1]
                ssin = 0.5 / np.sqrt(0.25 + 0.25 * ccotan**2.0)
                ccos = ssin * ccotan
                tau = 1.0
                if abs(ccos) * GIANT > 1.0:
                    tau = 1 / ccos
            else:
                ttan = w[j - 1] / s[jj - 1]
                ccos = 0.5 / np.sqrt(0.25 + 0.25 * ttan**2.0)
                ssin = ccos * ttan
                tau = ssin
            column, tail = s[jj - 1 : jj + m - j], w[j - 1 : m]
            for i in range(column.size):
                temp = ccos * column[i] + ssin * tail[i]
                tail[i] = -ssin * column[i] + ccos * tail[i]
                column[i] = temp
            w[j - 1] = tau
        if s[jj - 1] == 0.0:
            sing = True
        jj += m - j + 1

    l = jj
    for i in range(n, m + 1):
        s[l - 1] = w[i - 1]
        l += 1
    if s[jj - 1] == 0.0:
        sing = True
    return sing


@numba_njit(fastmath=False, error_model="numpy", inline="always")
def _hybrj(evaluate, x, args, xtol, maxfev, factor):
    # MINPACK's hybrj with internal scaling (mode 1). Returns x and MINPACK's
    # info code, where 1 means converged. evaluate returns the residuals and
    # the Jacobian together; like SciPy's MemoizeJac, a Jacobian request at
    # the last evaluated point reuses its Jacobian. evaluate only ever sees
    # evaluated_x, which changes just before the next call, because the
    # Jacobian it returns may be a view of its input.
    n = x.size
    if n == 0:
        return x, 0

    evaluated_x = x.copy()
    fvec, jacobian = evaluate(evaluated_x, *args)
    if fvec.size != n:
        raise ValueError("hybr needs as many equations as variables.")
    nfev = 1
    fnorm = _enorm(fvec)

    # Column-major, as MINPACK stores fjac.
    fjac = np.empty((n, n)).T
    r = np.empty((n * (n + 1)) // 2)
    diag, qtf = np.empty(n), np.empty(n)
    wa1, wa2, wa3 = np.empty(n), np.empty(n), np.empty(n)
    ipvt = np.empty(1, dtype=np.int64)

    iteration = 1
    ncsuc = ncfail = nslow1 = nslow2 = 0
    delta = xnorm = 0.0
    while True:
        jeval = True
        if not _same_point(x, evaluated_x):
            evaluated_x[:] = x
            _, jacobian = evaluate(evaluated_x, *args)
        fjac[:] = jacobian

        _qrfac(fjac, False, ipvt, wa1, wa2, wa3)

        if iteration == 1:
            for j in range(n):
                diag[j] = wa2[j]
                if wa2[j] == 0.0:
                    diag[j] = 1.0
            for j in range(n):
                wa3[j] = diag[j] * x[j]
            xnorm = _enorm(wa3)
            delta = factor * xnorm
            if delta == 0.0:
                delta = factor

        qtf[:] = fvec
        # Slice views keep the loops vectorizable, as in _qrfac.
        for j in range(n):
            if fjac[j, j] != 0.0:
                column, tail = fjac[j:, j], qtf[j:]
                ssum = 0.0
                for i in range(column.size):
                    ssum += column[i] * tail[i]
                temp = -ssum / fjac[j, j]
                for i in range(column.size):
                    tail[i] += column[i] * temp

        for j in range(n):
            l = j
            for i in range(j):
                r[l] = fjac[i, j]
                l += n - i - 1
            r[l] = wa1[j]

        _qform(fjac, n, wa1)
        for j in range(n):
            diag[j] = np.fmax(diag[j], wa2[j])

        while True:
            _dogleg(r, diag, qtf, delta, wa1, wa2, wa3)
            for j in range(n):
                wa1[j] = -wa1[j]
                wa2[j] = x[j] + wa1[j]
                wa3[j] = diag[j] * wa1[j]
            pnorm = _enorm(wa3)
            if iteration == 1:
                delta = np.fmin(delta, pnorm)

            evaluated_x[:] = wa2
            wa4, jacobian = evaluate(evaluated_x, *args)
            nfev += 1
            fnorm1 = _enorm(wa4)

            actred = 1.0 - (fnorm1 / fnorm) ** 2.0 if fnorm1 < fnorm else -1.0
            l = 0
            for i in range(n):
                row, tail = r[l : l + n - i], wa1[i:]
                ssum = 0.0
                for j in range(row.size):
                    ssum += row[j] * tail[j]
                l += n - i
                wa3[i] = qtf[i] + ssum
            temp = _enorm(wa3)
            prered = 1.0 - (temp / fnorm) ** 2.0 if temp < fnorm else 0.0
            ratio = actred / prered if prered > 0.0 else 0.0

            if ratio < 0.1:
                ncsuc = 0
                ncfail += 1
                delta = 0.5 * delta
            else:
                ncfail = 0
                ncsuc += 1
                if ratio >= 0.5 or ncsuc > 1:
                    delta = np.fmax(delta, pnorm / 0.5)
                if abs(ratio - 1.0) <= 0.1:
                    delta = pnorm / 0.5

            if ratio >= 0.0001:
                for j in range(n):
                    x[j] = wa2[j]
                    wa2[j] = diag[j] * x[j]
                    fvec[j] = wa4[j]
                xnorm = _enorm(wa2)
                fnorm = fnorm1
                iteration += 1

            nslow1 += 1
            if actred >= 0.001:
                nslow1 = 0
            if jeval:
                nslow2 += 1
            if actred >= 0.1:
                nslow2 = 0

            if delta <= xtol * xnorm or fnorm == 0.0:
                return x, 1
            info = 0
            if nfev >= maxfev:
                info = 2
            if 0.1 * np.fmax(0.1 * delta, pnorm) <= EPSMCH * xnorm:
                info = 3
            if nslow2 == 5:
                info = 4
            if nslow1 == 10:
                info = 5
            if info != 0:
                return x, info

            if ncfail == 2:
                break

            for j in range(n):
                ssum = 0.0
                for i in range(n):
                    ssum += fjac[i, j] * wa4[i]
                wa2[j] = (ssum - wa3[j]) / pnorm
                wa1[j] = diag[j] * ((diag[j] * wa1[j]) / pnorm)
                if ratio >= 0.0001:
                    qtf[j] = ssum

            _r1updt(n, n, r, wa1, wa2, wa3)
            _r1mpyq(fjac, wa2, wa3)
            _r1mpyq(qtf.reshape((1, n)), wa2, wa3)
            jeval = False
