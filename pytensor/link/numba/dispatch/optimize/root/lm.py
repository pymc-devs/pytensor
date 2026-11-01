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
    DWARF,
    EPSMCH,
    _enorm,
    _qrfac,
    _same_point,
)


@numba_njit(fastmath=False, error_model="numpy")
def _qrsolv(r, ipvt, diag, qtb, x, sdiag, wa):
    # Solve the least-squares problem of r with the diagonal diag appended,
    # where r is the upper triangle of the pivoted QR factor in the leading
    # n x n block. Its strict lower triangle receives the transformed r.
    n = diag.size
    for j in range(n):
        for i in range(j, n):
            r[i, j] = r[j, i]
        x[j] = r[j, j]
        wa[j] = qtb[j]

    for j in range(n):
        l = ipvt[j]
        if diag[l] != 0.0:
            for k in range(j, n):
                sdiag[k] = 0.0
            sdiag[j] = diag[l]
            qtbpj = 0.0
            for k in range(j, n):
                if sdiag[k] == 0.0:
                    continue
                if abs(r[k, k]) < abs(sdiag[k]):
                    ccotan = r[k, k] / sdiag[k]
                    ssin = 0.5 / np.sqrt(0.25 + 0.25 * ccotan**2.0)
                    ccos = ssin * ccotan
                else:
                    ttan = sdiag[k] / r[k, k]
                    ccos = 0.5 / np.sqrt(0.25 + 0.25 * ttan**2.0)
                    ssin = ccos * ttan
                r[k, k] *= ccos
                r[k, k] += ssin * sdiag[k]
                temp = ccos * wa[k] + ssin * qtbpj
                qtbpj = -ssin * wa[k] + ccos * qtbpj
                wa[k] = temp
                # Slice views keep the loops vectorizable, as in _qrfac.
                column, tail = r[k + 1 : n, k], sdiag[k + 1 :]
                for i in range(column.size):
                    temp = ccos * column[i] + ssin * tail[i]
                    tail[i] = -ssin * column[i] + ccos * tail[i]
                    column[i] = temp
        sdiag[j] = r[j, j]
        r[j, j] = x[j]

    nsing = n
    for j in range(n):
        if sdiag[j] == 0.0 and nsing == n:
            nsing = j
        if nsing < n:
            wa[j] = 0.0
    for k in range(1, nsing + 1):
        j = nsing - k
        column, tail = r[j + 1 : nsing, j], wa[j + 1 : nsing]
        ssum = 0.0
        for i in range(column.size):
            ssum += column[i] * tail[i]
        wa[j] = (wa[j] - ssum) / sdiag[j]

    for j in range(n):
        x[ipvt[j]] = wa[j]


@numba_njit(fastmath=False, error_model="numpy")
def _lmpar(r, ipvt, diag, qtb, delta, par, x, sdiag, wa1, wa2):
    # Levenberg-Marquardt parameter for the step x within the trust region
    # delta, starting from par. Returns the new parameter and leaves the step
    # in x. wa2 may be longer than n, since _lmder lends it the residuals.
    n = diag.size
    nsing = n
    for j in range(n):
        wa1[j] = qtb[j]
        if r[j, j] == 0.0 and nsing == n:
            nsing = j
        if nsing < n:
            wa1[j] = 0.0
    for k in range(1, nsing + 1):
        j = nsing - k
        wa1[j] /= r[j, j]
        temp = wa1[j]
        column, head = r[:j, j], wa1[:j]
        for i in range(column.size):
            head[i] -= column[i] * temp
    for j in range(n):
        x[ipvt[j]] = wa1[j]

    for j in range(n):
        wa2[j] = diag[j] * x[j]
    dxnorm = _enorm(wa2[:n])
    fp = dxnorm - delta
    if fp <= 0.1 * delta:
        return 0.0

    # Lower bound from the Newton step, when the Jacobian has full rank.
    parl = 0.0
    if nsing == n:
        for j in range(n):
            l = ipvt[j]
            wa1[j] = diag[l] * (wa2[l] / dxnorm)
        for j in range(n):
            column, head = r[:j, j], wa1[:j]
            ssum = 0.0
            for i in range(column.size):
                ssum += column[i] * head[i]
            wa1[j] = (wa1[j] - ssum) / r[j, j]
        temp = _enorm(wa1)
        parl = ((fp / delta) / temp) / temp

    for j in range(n):
        column, head = r[: j + 1, j], qtb[: j + 1]
        ssum = 0.0
        for i in range(column.size):
            ssum += column[i] * head[i]
        wa1[j] = ssum / diag[ipvt[j]]
    gnorm = _enorm(wa1)
    paru = gnorm / delta
    if paru == 0.0:
        paru = DWARF / np.fmin(delta, 0.1)

    par = np.fmin(np.fmax(par, parl), paru)
    if par == 0.0:
        par = gnorm / dxnorm

    iteration = 0
    while True:
        iteration += 1
        if par == 0.0:
            par = np.fmax(DWARF, paru * 0.001)
        temp = np.sqrt(par)
        for j in range(n):
            wa1[j] = temp * diag[j]
        _qrsolv(r, ipvt, wa1, qtb, x, sdiag, wa2)
        for j in range(n):
            wa2[j] = diag[j] * x[j]
        dxnorm = _enorm(wa2[:n])
        temp = fp
        fp = dxnorm - delta
        if (
            abs(fp) <= 0.1 * delta
            or (parl == 0.0 and fp <= temp and temp < 0.0)
            or iteration == 10
        ):
            return par

        for j in range(n):
            l = ipvt[j]
            wa1[j] = diag[l] * (wa2[l] / dxnorm)
        for j in range(n):
            wa1[j] /= sdiag[j]
            temp = wa1[j]
            column, tail = r[j + 1 : n, j], wa1[j + 1 :]
            for i in range(column.size):
                tail[i] -= column[i] * temp
        temp = _enorm(wa1)
        parc = ((fp / delta) / temp) / temp
        if fp > 0.0:
            parl = np.fmax(parl, par)
        if fp < 0.0:
            paru = np.fmin(paru, par)
        par = np.fmax(parl, par + parc)


@numba_njit(fastmath=False, error_model="numpy", inline="always")
def _lmder(evaluate, x, args, ftol, xtol, gtol, maxfev, factor):
    # MINPACK's lmder with internal scaling (mode 1). Returns x and MINPACK's
    # info code, where 1 to 4 mean converged. evaluate returns the residuals
    # and the Jacobian together and only ever sees evaluated_x, as in _hybrj.
    n = x.size
    if n == 0:
        return x, 0

    evaluated_x = x.copy()
    fvec, jacobian = evaluate(evaluated_x, *args)
    m = fvec.size
    if m < n:
        raise ValueError("lm needs at least as many equations as variables.")
    nfev = 1
    fnorm = _enorm(fvec)

    # Column-major, as MINPACK stores fjac.
    fjac = np.empty((n, m)).T
    diag, qtf = np.empty(n), np.empty(n)
    wa1, wa2, wa3, wa4 = np.empty(n), np.empty(n), np.empty(n), np.empty(m)
    ipvt = np.empty(n, dtype=np.int64)

    iteration = 1
    par = delta = xnorm = 0.0
    while True:
        if not _same_point(x, evaluated_x):
            evaluated_x[:] = x
            _, jacobian = evaluate(evaluated_x, *args)
        fjac[:] = jacobian

        _qrfac(fjac, True, ipvt, wa1, wa2, wa3)

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

        wa4[:] = fvec
        # Slice views keep the loops vectorizable, as in _qrfac.
        for j in range(n):
            if fjac[j, j] != 0.0:
                column, tail = fjac[j:, j], wa4[j:]
                ssum = 0.0
                for i in range(column.size):
                    ssum += column[i] * tail[i]
                temp = -ssum / fjac[j, j]
                for i in range(column.size):
                    tail[i] += column[i] * temp
            fjac[j, j] = wa1[j]
            qtf[j] = wa4[j]

        # Scaled cosine between the residuals and the Jacobian columns.
        gnorm = 0.0
        if fnorm != 0.0:
            for j in range(n):
                l = ipvt[j]
                if wa2[l] != 0.0:
                    ssum = 0.0
                    for i in range(j + 1):
                        ssum += fjac[i, j] * (qtf[i] / fnorm)
                    gnorm = np.fmax(gnorm, abs(ssum / wa2[l]))
        if gnorm <= gtol:
            return x, 4

        for j in range(n):
            diag[j] = np.fmax(diag[j], wa2[j])

        while True:
            par = _lmpar(fjac, ipvt, diag, qtf, delta, par, wa1, wa2, wa3, wa4)
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

            actred = 1.0 - (fnorm1 / fnorm) ** 2.0 if 0.1 * fnorm1 < fnorm else -1.0
            for j in range(n):
                wa3[j] = 0.0
                temp = wa1[ipvt[j]]
                column, head = fjac[: j + 1, j], wa3[: j + 1]
                for i in range(column.size):
                    head[i] += column[i] * temp
            temp1 = _enorm(wa3) / fnorm
            temp2 = (np.sqrt(par) * pnorm) / fnorm
            prered = temp1 * temp1 + temp2 * temp2 / 0.5
            dirder = -(temp1 * temp1 + temp2 * temp2)
            ratio = actred / prered if prered != 0.0 else 0.0

            if ratio <= 0.25:
                temp = 0.5 * dirder / (dirder + 0.5 * actred) if actred < 0.0 else 0.5
                if 0.1 * fnorm1 >= fnorm or temp < 0.1:
                    temp = 0.1
                delta = temp * np.fmin(delta, pnorm / 0.1)
                par /= temp
            elif par == 0.0 or ratio >= 0.75:
                delta = pnorm / 0.5
                par *= 0.5

            if ratio >= 0.0001:
                for j in range(n):
                    x[j] = wa2[j]
                    wa2[j] = diag[j] * x[j]
                fvec[:] = wa4
                xnorm = _enorm(wa2)
                fnorm = fnorm1
                iteration += 1

            reduced = abs(actred) <= ftol and prered <= ftol and 0.5 * ratio <= 1.0
            if delta <= xtol * xnorm:
                return x, 3 if reduced else 2
            if reduced:
                return x, 1
            info = 0
            if nfev >= maxfev:
                info = 5
            if abs(actred) <= EPSMCH and prered <= EPSMCH and 0.5 * ratio <= 1.0:
                info = 6
            if delta <= EPSMCH * xnorm:
                info = 7
            if gnorm <= EPSMCH:
                info = 8
            if info != 0:
                return x, info

            if ratio >= 0.0001:
                break
