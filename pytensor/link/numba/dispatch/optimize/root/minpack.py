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


EPSMCH = np.finfo(np.float64).eps
GIANT = np.finfo(np.float64).max
RGIANT = 2.0**63.5


@numba_njit(fastmath=False, error_model="numpy")
def _enorm(x):
    # MINPACK's Euclidean norm, which scales small and large entries apart to
    # avoid underflow and overflow.
    rdwarf = 0.5 / RGIANT
    agiant = RGIANT / x.size
    s1 = s2 = s3 = x1max = x3max = 0.0
    for i in range(x.size):
        xabs = abs(x[i])
        if xabs <= rdwarf or xabs >= agiant:
            if xabs > rdwarf:
                if xabs > x1max:
                    s1 = 1.0 + s1 * (x1max / xabs) ** 2.0
                    x1max = xabs
                else:
                    s1 += (xabs / x1max) ** 2.0
            elif xabs > x3max:
                s3 = 1.0 + s3 * (x3max / xabs) ** 2.0
                x3max = xabs
            elif xabs != 0.0:
                s3 += (xabs / x3max) ** 2.0
        else:
            s2 += xabs * xabs

    if s1 != 0.0:
        return x1max * np.sqrt(s1 + (s2 / x1max) / x1max)
    if s2 != 0.0:
        if s2 >= x3max:
            return np.sqrt(s2 * (1.0 + (x3max / s2) * (x3max * s3)))
        return np.sqrt(x3max * ((s2 / x3max) + (x3max * s3)))
    return x3max * np.sqrt(s3)


@numba_njit(fastmath=False, error_model="numpy")
def _qrfac(a, pivot, ipvt, rdiag, acnorm, wa):
    # Householder QR of a in place, with optional column pivoting.
    m, n = a.shape
    for j in range(n):
        acnorm[j] = _enorm(a[:, j])
        rdiag[j] = acnorm[j]
        wa[j] = rdiag[j]
        if pivot:
            ipvt[j] = j

    for j in range(min(m, n)):
        if pivot:
            kmax = j
            for k in range(j, n):
                if rdiag[k] > rdiag[kmax]:
                    kmax = k
            if kmax != j:
                for i in range(m):
                    temp = a[i, j]
                    a[i, j] = a[i, kmax]
                    a[i, kmax] = temp
                rdiag[kmax] = rdiag[j]
                wa[kmax] = wa[j]
                k = ipvt[j]
                ipvt[j] = ipvt[kmax]
                ipvt[kmax] = k

        ajnorm = _enorm(a[j:, j])
        if ajnorm == 0.0:
            rdiag[j] = -ajnorm
            continue
        if a[j, j] < 0.0:
            ajnorm = -ajnorm
        # Loops over slice views from 0 let Numba vectorize them, which it
        # does not do for range(j, m).
        column = a[j:, j]
        for i in range(column.size):
            column[i] /= ajnorm
        a[j, j] += 1.0
        if j == n - 1:
            rdiag[j] = -ajnorm
            continue

        for k in range(j + 1, n):
            other = a[j:, k]
            ssum = 0.0
            for i in range(column.size):
                ssum += column[i] * other[i]
            temp = ssum / a[j, j]
            for i in range(column.size):
                other[i] -= temp * column[i]
            if pivot and rdiag[k] != 0.0:
                temp = a[j, k] / rdiag[k]
                rdiag[k] *= np.sqrt(np.fmax(0.0, 1.0 - temp * temp))
                if 0.05 * (rdiag[k] / wa[k]) ** 2.0 <= EPSMCH:
                    rdiag[k] = _enorm(a[j + 1 :, k])
                    wa[k] = rdiag[k]
        rdiag[j] = -ajnorm


@numba_njit(fastmath=False, error_model="numpy")
def _same_point(x, y):
    # np.all(x == y) without the temporary.
    for i in range(x.size):
        if x[i] != y[i]:
            return False
    return True
