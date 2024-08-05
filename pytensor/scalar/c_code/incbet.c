/*    adapted from file incbet.c, obtained from the Cephes library (MIT License)
Cephes Math Library, Release 2.8:  June, 2000
Copyright 1984, 1995, 2000 by Stephen L. Moshier
*/

//For GPU support
#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <numpy/npy_math.h>


// Constants borrowed from Scipy
// https://github.com/scipy/scipy/blob/81c53d48a290b604ec5faa34c0a7d48537b487d6/scipy/special/special/cephes/const.h#L65-L78
#define MINLOG     -7.451332191019412076235E2    // log 2**-1022
#define MAXLOG     7.09782712893383996732E2      // log(DBL_MAX)
#define MAXGAM     171.624376956302725
#define EPSILON     1.11022302462515654042e-16    // 2**-53

DEVICE static double pseries(double, double, double);
DEVICE static double incbcf(double, double, double);
DEVICE static double incbd(double, double, double);

static double big = 4.503599627370496e15;
static double biginv =  2.22044604925031308085e-16;


DEVICE double BetaInc(double a, double b, double x)
{
    double xc, y, w, t;
    /* check function arguments */
    if (a <= 0.0) return NPY_NAN;
    if (b <= 0.0) return NPY_NAN;
    if (x < 0.0) return NPY_NAN;
    if (1.0 < x) return NPY_NAN;

    /* some special cases */
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;

    if ( (b * x) <= 1.0 && x <= 0.95)
    {
        return pseries(a, b, x);
    }

    xc = 1.0 - x;
    /* reverse a and b if x is greater than the mean */
    if (x > (a / (a + b)))
    {
        t = BetaInc(b, a, xc);
        if (t <= EPSILON) return 1.0 - EPSILON;
        return 1.0 - t;
    }

    /* Choose expansion for better convergence. */
    y = x * (a+b-2.0) - (a-1.0);
    if( y < 0.0 )
        w = incbcf( a, b, x );
    else
        w = incbd( a, b, x ) / xc;

    y = a * log(x);
    t = b * log(xc);
    if( (a+b) < MAXGAM && fabs(y) < MAXLOG && fabs(t) < MAXLOG )
    {
        t = pow(xc, b);
        t *= pow(x, a);
        t /= a;
        t *= w;
        t *= tgamma(a + b) / (tgamma(a) * tgamma(b));

        return t;
    }

    /* Resort to logarithms.  */
    y += t + lgamma(a+b) - lgamma(a) - lgamma(b);
    y += log(w / a);
    if( y < MINLOG )
        t = 0.0;
    else
        t = exp(y);

    return t;
}

/* Continued fraction expansion #1
 * for incomplete beta integral
 */

DEVICE static double incbcf(double a, double b, double x)
{
    double xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
    double k1, k2, k3, k4, k5, k6, k7, k8;
    double r, t, ans, thresh;
    int n;

    k1 = a;
    k2 = a + b;
    k3 = a;
    k4 = a + 1.0;
    k5 = 1.0;
    k6 = b - 1.0;
    k7 = k4;
    k8 = a + 2.0;

    pkm2 = 0.0;
    qkm2 = 1.0;
    pkm1 = 1.0;
    qkm1 = 1.0;
    ans = 1.0;
    r = 1.0;
    n = 0;
    thresh = 3.0 * EPSILON;
    do
    {

        xk = -( x * k1 * k2 ) / ( k3 * k4 );
        pk = pkm1 +  pkm2 * xk;
        qk = qkm1 +  qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        xk = ( x * k5 * k6 ) / ( k7 * k8 );
        pk = pkm1 +  pkm2 * xk;
        qk = qkm1 +  qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if( qk != 0.0 )
            r = pk/qk;
        if( r != 0.0 )
        {
            t = fabs( (ans - r) / r );
            ans = r;
        }
        else
            t = 1.0;

        if( t < thresh )
            break;

        k1 += 1.0;
        k2 += 1.0;
        k3 += 2.0;
        k4 += 2.0;
        k5 += 1.0;
        k6 -= 1.0;
        k7 += 2.0;
        k8 += 2.0;

        if( (fabs(qk) + fabs(pk)) > big )
        {
            pkm2 *= biginv;
            pkm1 *= biginv;
            qkm2 *= biginv;
            qkm1 *= biginv;
        }
        if( (fabs(qk) < biginv) || (fabs(pk) < biginv) )
        {
            pkm2 *= big;
            pkm1 *= big;
            qkm2 *= big;
            qkm1 *= big;
        }
    }
    while( ++n < 300 );

    return ans;
}

/* Continued fraction expansion #2
 * for incomplete beta integral
 */

DEVICE static double incbd(double a, double b, double x)
{
    double xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
    double k1, k2, k3, k4, k5, k6, k7, k8;
    double r, t, ans, z, thresh;
    int n;

    k1 = a;
    k2 = b - 1.0;
    k3 = a;
    k4 = a + 1.0;
    k5 = 1.0;
    k6 = a + b;
    k7 = a + 1.0;;
    k8 = a + 2.0;

    pkm2 = 0.0;
    qkm2 = 1.0;
    pkm1 = 1.0;
    qkm1 = 1.0;
    z = x / (1.0-x);
    ans = 1.0;
    r = 1.0;
    n = 0;
    thresh = 3.0 * EPSILON;
    do
    {

        xk = -( z * k1 * k2 ) / ( k3 * k4 );
        pk = pkm1 +  pkm2 * xk;
        qk = qkm1 +  qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        xk = ( z * k5 * k6 ) / ( k7 * k8 );
        pk = pkm1 +  pkm2 * xk;
        qk = qkm1 +  qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if( qk != 0 )
            r = pk/qk;
        if( r != 0 )
        {
            t = fabs( (ans - r) / r );
            ans = r;
        }
        else
            t = 1.0;

        if( t < thresh )
            break;

        k1 += 1.0;
        k2 -= 1.0;
        k3 += 2.0;
        k4 += 2.0;
        k5 += 1.0;
        k6 += 1.0;
        k7 += 2.0;
        k8 += 2.0;

        if( (fabs(qk) + fabs(pk)) > big )
        {
            pkm2 *= biginv;
            pkm1 *= biginv;
            qkm2 *= biginv;
            qkm1 *= biginv;
        }
        if( (fabs(qk) < biginv) || (fabs(pk) < biginv) )
        {
            pkm2 *= big;
            pkm1 *= big;
            qkm2 *= big;
            qkm1 *= big;
        }
    }
    while( ++n < 300 );

    return ans;
}


/* Power series for incomplete beta integral.
   Use when b*x is small and x not too close to 1.  */

DEVICE static double pseries(double a, double b, double x)
{
    double s, t, u, v, n, t1, z, ai;

    ai = 1.0 / a;
    u = (1.0 - b) * x;
    v = u / (a + 1.0);
    t1 = v;
    t = u;
    n = 2.0;
    s = 0.0;
    z = EPSILON * ai;
    while( fabs(v) > z )
    {
    u = (n - b) * x / n;
    t *= u;
    v = t / (a + n);
    s += v;
    n += 1.0;
    }
    s += t1;
    s += ai;

    u = a * log(x);
    if( (a+b) < MAXGAM && fabs(u) < MAXLOG )
    {
    t = tgamma(a + b) / (tgamma(a) * tgamma(b));
    s = s * t * pow(x,a);
    }
    else
    {
    t = lgamma(a + b) - lgamma(a) - lgamma(b) + u + log(s);
    if( t < MINLOG )
    s = 0.0;
    else
    s = exp(t);
    }
    return s;
}