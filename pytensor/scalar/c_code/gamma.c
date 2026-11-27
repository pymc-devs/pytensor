/*----------------------------------------------------------------------
  File    : gamma.c
  Contents: computation of the (incomplete/regularized) gamma function
  Author  : Christian Borgelt
  Licence : MIT
  History : 2002.07.04 file created
            2003.05.19 incomplete Gamma function added
            2008.03.14 more incomplete Gamma functions added
            2008.03.15 table of factorials and logarithms added
            2008.03.17 gamma distribution functions added
  Modification by Frederic Bastien:
            2013.11.13 commented the gamma.h file as it is not needed.
            2013.11.13 modification to make it work with CUDA
  Modification by M. Domenzain:
            2018.10.11 copied unitqtlQ and unitqtlP from normal.c
            2018.10.11 removed all unused code

----------------------------------------------------------------------*/
//For GPU support
#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif

#ifndef _ISOC99_SOURCE
#define _ISOC99_SOURCE
#endif                          /* needed for function log1p() */
#include <float.h>
#include <math.h>
#include <numpy/npy_math.h>

/*----------------------------------------------------------------------
  Preprocessor Definitions
----------------------------------------------------------------------*/
#define LN_BASE      2.71828182845904523536028747135  /* e */
#define SQRT_PI      1.77245385090551602729816748334  /* \sqrt(\pi) */
#define LN_PI        1.14472988584940017414342735135  /* \ln(\pi) */
#define LN_SQRT_2PI  0.918938533204672741780329736406
                                                  /* \ln(\sqrt(2\pi)) */
#define EPSILON      2.2204460492503131e-16
#define EPS_QTL      1.4901161193847656e-08
#define MAXFACT      170
#define MAXITER      1024
#define TINY         (EPSILON *EPSILON *EPSILON)
#define Gammacdf(x,k,t)  GammaP(k,(x)/(t))
#define GammacdfP(x,k,t) GammaP(k,(x)/(t))
#define GammacdfQ(x,k,t) GammaQ(k,(x)/(t))
#define unitqtlQ(p)    (-unitqtlP(p))

/*----------------------------------------------------------------------
  Table of Factorials/Gamma Values
----------------------------------------------------------------------*/
DEVICE static double _facts[MAXFACT+1] = { 0 };
DEVICE static double _logfs[MAXFACT+1];
DEVICE static double _halfs[MAXFACT+1];
DEVICE static double _loghs[MAXFACT+1];

/*----------------------------------------------------------------------
  Functions
----------------------------------------------------------------------*/

DEVICE static void _init (void)
{                               /* --- init. factorial tables */
  int    i;                     /* loop variable */
  double x = 1;                 /* factorial */

  _facts[0] = _facts[1] = 1;    /* store factorials for 0 and 1 */
  _logfs[0] = _logfs[1] = 0;    /* and their logarithms */
  for (i = 1; ++i <= MAXFACT; ) {
    _facts[i] = x *= i;         /* initialize the factorial table */
    _logfs[i] = log(x);         /* and the table of their logarithms */
  }
  _halfs[0] = x = SQRT_PI;      /* store Gamma(0.5) */
  _loghs[0] = 0.5*LN_PI;        /* and its logarithm */
  for (i = 0; ++i < MAXFACT; ) {
    _halfs[i] = x *= i-0.5;     /* initialize the table for */
    _loghs[i] = log(x);         /* the Gamma function of half numbers */
  }                             /* and the table of their logarithms */
}  /* _init() */


DEVICE double logGamma (double n)
{                               /* --- compute ln(Gamma(n))         */
  double s;                     /*           = ln((n-1)!), n \in IN */

  if (n <= 0) return NPY_NAN;       /* check the function arguments */
  if (_facts[0] <= 0) _init();  /* initialize the tables */
  if (n < MAXFACT +1 +4 *EPSILON) {
    if (fabs(  n -floor(  n)) < 4 *EPSILON)
      return _logfs[(int)floor(n)-1];
    if (fabs(2*n -floor(2*n)) < 4 *EPSILON)
      return _loghs[(int)floor(n)];
  }                             /* try to get the value from a table */
  s =    0.99999999999980993227684700473478  /* otherwise compute it */
    +  676.520368121885098567009190444019 /(n+1)
    - 1259.13921672240287047156078755283  /(n+2)
    +  771.3234287776530788486528258894   /(n+3)
    -  176.61502916214059906584551354     /(n+4)
    +   12.507343278686904814458936853    /(n+5)
    -    0.13857109526572011689554707     /(n+6)
    +    9.984369578019570859563e-6       /(n+7)
    +    1.50563273514931155834e-7        /(n+8);
  return (n+0.5) *log((n+7.5)/LN_BASE) +(LN_SQRT_2PI +log(s/n) -7.0);
}  /* logGamma() */

/*----------------------------------------------------------------------
Use Lanczos' approximation
\Gamma(n+1) = (n+\gamma+0.5)^(n+0.5)
            * e^{-(n+\gamma+0.5)}
            * \sqrt{2\pi}
            * (c_0 +c_1/(n+1) +c_2/(n+2) +...+c_n/(n+k) +\epsilon)
and exploit the recursion \Gamma(n+1) = n *\Gamma(n) once,
i.e., compute \Gamma(n) as \Gamma(n+1) /n.

For the choices \gamma = 5, k = 6, and c_0 to c_6 as defined
in the first version, it is |\epsilon| < 2e-10 for all n > 0.

Source: W.H. Press, S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery
        Numerical Recipes in C - The Art of Scientific Computing
        Cambridge University Press, Cambridge, United Kingdom 1992
        pp. 213-214

For the choices gamma = 7, k = 8, and c_0 to c_8 as defined
in the second version, the value is slightly more accurate.
----------------------------------------------------------------------*/

DEVICE double Gamma (double n)
{                               /* --- compute Gamma(n) = (n-1)! */
  if (n <= 0) return NPY_NAN;       /* check the function arguments */
  if (_facts[0] <= 0) _init();  /* initialize the tables */
  if (n < MAXFACT +1 +4 *EPSILON) {
    if (fabs(  n -floor(  n)) < 4 *EPSILON)
      return _facts[(int)floor(n)-1];
    if (fabs(2*n -floor(2*n)) < 4 *EPSILON)
      return _halfs[(int)floor(n)];
  }                             /* try to get the value from a table */
  return exp(logGamma(n));      /* compute through natural logarithm */
}  /* Gamma() */

/*--------------------------------------------------------------------*/

DEVICE static double _series (double n, double x)
{                               /* --- series approximation */
  int    i;                     /* loop variable */
  double t, sum;                /* buffers */

  sum = t = 1/n;                /* compute initial values */
  for (i = MAXITER; --i >= 0; ) {
    sum += t *= x/++n;          /* add one term of the series */
    if (fabs(t) < fabs(sum) *EPSILON) return sum;
  }                             /* if term is small enough, abort */
  return NPY_NAN;               /* do not return a partial sum */
}  /* _series() */

/*----------------------------------------------------------------------
series approximation:
P(a,x) =    \gamma(a,x)/\Gamma(a)
\gamma(a,x) = e^-x x^a \sum_{n=0}^\infty (\Gamma(a)/\Gamma(a+1+n)) x^n

Source: W.H. Press, S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery
        Numerical Recipes in C - The Art of Scientific Computing
        Cambridge University Press, Cambridge, United Kingdom 1992
        formula: pp. 216-219

The factor exp(n *log(x) -x) is added in the functions below.
----------------------------------------------------------------------*/

DEVICE static double _cfrac (double n, double x)
{                               /* --- continued fraction approx. */
  int    i;                     /* loop variable */
  double a, b, c, d, e, f;      /* buffers */

  b = x+1-n; c = 1/TINY; f = d = 1/b;
  for (i = 1; i < MAXITER; i++) {
    a = i*(n-i);                /* use Lentz's algorithm to compute */
    d = a *d +(b += 2);         /* consecutive approximations */
    if (fabs(d) < TINY) d = TINY;
    c = b +a/c;
    if (fabs(c) < TINY) c = TINY;
    d = 1/d; f *= e = d *c;
    if (fabs(e-1) < EPSILON) return f;
  }                             /* if factor is small enough, abort */
  return NPY_NAN;               /* continued fraction did not converge */
}  /* _cfrac() */

/*----------------------------------------------------------------------
continued fraction approximation:
P(a,x) = 1 -\Gamma(a,x)/\Gamma(a)
\Gamma(a,x) = e^-x x^a (1/(x+1-a- 1(1-a)/(x+3-a- 2*(2-a)/(x+5-a- ...))))

Source: W.H. Press, S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery
        Numerical Recipes in C - The Art of Scientific Computing
        Cambridge University Press, Cambridge, United Kingdom 1992
        formula:           pp. 216-219
        Lentz's algorithm: p.  171

The factor exp(n *log(x) -x) is added in the functions below.
----------------------------------------------------------------------*/

/*----------------------------------------------------------------------
  The coefficients below are the first four rows of SciPy's Temme expansion:
  https://github.com/scipy/xsf/blob/b3429e625ef366b5dd90d2097268a180075183e0/include/xsf/cephes/igam_asymp_coeff.h
  They are used under the following license:

BSD 3-Clause License

Copyright (c) 2024, SciPy

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
----------------------------------------------------------------------*/
DEVICE static const double _gamma_asymp_coeff[4][25] = {
    {-3.3333333333333333e-1,  8.3333333333333333e-2,   -1.4814814814814815e-2,  1.1574074074074074e-3,
     3.527336860670194e-4,    -1.7875514403292181e-4,  3.9192631785224378e-5,   -2.1854485106799922e-6,
     -1.85406221071516e-6,    8.296711340953086e-7,    -1.7665952736826079e-7,  6.7078535434014986e-9,
     1.0261809784240308e-8,   -4.3820360184533532e-9,  9.1476995822367902e-10,  -2.551419399494625e-11,
     -5.8307721325504251e-11, 2.4361948020667416e-11,  -5.0276692801141756e-12, 1.1004392031956135e-13,
     3.3717632624009854e-13,  -1.3923887224181621e-13, 2.8534893807047443e-14,  -5.1391118342425726e-16,
     -1.9752288294349443e-15},
    {-1.8518518518518519e-3,  -3.4722222222222222e-3,  2.6455026455026455e-3,   -9.9022633744855967e-4,
     2.0576131687242798e-4,   -4.0187757201646091e-7,  -1.8098550334489978e-5,  7.6491609160811101e-6,
     -1.6120900894563446e-6,  4.6471278028074343e-9,   1.378633446915721e-7,    -5.752545603517705e-8,
     1.1951628599778147e-8,   -1.7543241719747648e-11, -1.0091543710600413e-9,  4.1627929918425826e-10,
     -8.5639070264929806e-11, 6.0672151016047586e-14,  7.1624989648114854e-12,  -2.9331866437714371e-12,
     5.9966963656836887e-13,  -2.1671786527323314e-16, -4.9783399723692616e-14, 2.0291628823713425e-14,
     -4.13125571381061e-15},
    {4.1335978835978836e-3,   -2.6813271604938272e-3,  7.7160493827160494e-4,  2.0093878600823045e-6,
     -1.0736653226365161e-4,  5.2923448829120125e-5,   -1.2760635188618728e-5, 3.4235787340961381e-8,
     1.3721957309062933e-6,   -6.298992138380055e-7,   1.4280614206064242e-7,  -2.0477098421990866e-10,
     -1.4092529910867521e-8,  6.228974084922022e-9,    -1.3670488396617113e-9, 9.4283561590146782e-13,
     1.2872252400089318e-10,  -5.5645956134363321e-11, 1.1975935546366981e-11, -4.1689782251838635e-15,
     -1.0940640427884594e-12, 4.6622399463901357e-13,  -9.905105763906906e-14, 1.8931876768373515e-17,
     8.8592218725911273e-15},
    {6.4943415637860082e-4,   2.2947209362139918e-4,   -4.6918949439525571e-4,  2.6772063206283885e-4,
     -7.5618016718839764e-5,  -2.3965051138672967e-7,  1.1082654115347302e-5,   -5.6749528269915966e-6,
     1.4230900732435884e-6,   -2.7861080291528142e-11, -1.6958404091930277e-7,  8.0994649053880824e-8,
     -1.9111168485973654e-8,  2.3928620439808118e-12,  2.0620131815488798e-9,   -9.4604966618551322e-10,
     2.1541049775774908e-10,  -1.388823336813903e-14,  -2.1894761681963939e-11, 9.7909989511716851e-12,
     -2.1782191880180962e-12, 6.2088195734079014e-17,  2.126978363279737e-13,   -9.3446887915174333e-14,
     2.0453671226782849e-14}
};

/* DLMF 8.12.3, 8.12.4 and 8.12.7.  Four terms suffice for n >= 1000
   and |(x-n)/n| <= 0.3.  Evaluate P and Q directly to retain small tails. */
DEVICE static double _gamma_asymptotic (double n, double x, int upper)
{
  int i, k;
  double sigma = (x-n)/n;
  double term = -0.5*sigma*sigma;
  double log1pmx = term;
  double eta, z, coefficient, sum = 0;

  /* log1p(sigma)-sigma loses precision near zero.  The series converges
     geometrically throughout this branch, including sigma = 0. */
  for (i = 3; i < MAXITER; i++) {
    term *= -sigma*(i-1)/i;
    log1pmx += term;
    if (fabs(term) <= EPSILON*fabs(log1pmx)) break;
  }
  eta = copysign(sqrt(-2*log1pmx), sigma);
  z = eta*sqrt(n/2);
  for (k = 3; k >= 0; k--) {
    coefficient = _gamma_asymp_coeff[k][24];
    for (i = 23; i >= 0; i--)
      coefficient = coefficient*eta +_gamma_asymp_coeff[k][i];
    sum = sum/n +coefficient;
  }
  term = exp(-z*z)*sum/(sqrt(n)*2.5066282746310005024);
  return upper ? 0.5*erfc(z) +term : 0.5*erfc(-z) -term;
}

/*--------------------------------------------------------------------*/

DEVICE double GammaP (double n, double x)
{                               /* --- regularized Gamma function P */
  if (isnan(n) || isnan(x) || (n <= 0) || (x < 0)) return NPY_NAN;
  if (x <=  0) return 0;        /* treat x = 0 as a special case */
  if (isinf(n)) {
    if (isinf(x)) return NPY_NAN;
    return 0;
  }
  if (isinf(x)) return 1;
  if ((n >= 1000) && (fabs(x-n) <= 0.3*n))
    return _gamma_asymptotic(n, x, 0);
  if (x < n+1) return _series(n, x) *exp(n *log(x) -x -logGamma(n));
  return 1 -_cfrac(n, x) *exp(n *log(x) -x -logGamma(n));
}  /* GammaP() */

/*--------------------------------------------------------------------*/

DEVICE double GammaQ (double n, double x)
{                               /* --- regularized Gamma function Q */
  if (isnan(n) || isnan(x) || (n <= 0) || (x < 0)) return NPY_NAN;
  if (x <=  0) return 1;        /* treat x = 0 as a special case */
  if (isinf(n)) {
    if (isinf(x)) return NPY_NAN;
    return 1;
  }
  if (isinf(x)) return 0;
  if ((n >= 1000) && (fabs(x-n) <= 0.3*n))
    return _gamma_asymptotic(n, x, 1);
  if (x < n+1) return 1 -_series(n, x) *exp(n *log(x) -x -logGamma(n));
  return _cfrac(n, x) *exp(n *log(x) -x -logGamma(n));
}  /* GammaQ() */

/*----------------------------------------------------------------------
P(a,x) is also called the regularized gamma function, Q(a,x) = 1-P(a,x).
P(k/2,x/2), where k is a natural number, is the cumulative distribution
function (cdf) of a chi^2 distribution with k degrees of freedom.
----------------------------------------------------------------------*/

DEVICE double Gammapdf (double x, double k, double theta)
{                               /* --- probability density function */
  if ((k <= 0) || (theta <= 0)) return NPY_NAN;  /* check the function arguments */
  if (x <  0) return 0;         /* support is non-negative x */
  if (x <= 0) return (k == 1) ? 1/theta : 0;
  if (k == 1) return exp(-x/theta) /theta;
  return exp ((k-1) *log(x/theta) -x/theta -logGamma(k)) /theta;
}  /* Gammapdf() */

/*--------------------------------------------------------------------*/
double unitqtlP (double prob)
{                               /* --- quantile of normal distrib. */
  double p, x;                  /*     with mean 0 and variance 1 */

  if ((prob < 0) || (prob > 1))  return NPY_NAN;  /* check the function arguments */
  if (prob >= 1.0) return  DBL_MAX; /* check for limiting values */
  if (prob <= 0.0) return -DBL_MAX; /* and return extrema */
  p = prob -0.5;
  if (fabs(p) <= 0.425) {       /* if not tail */
    x = 0.180625 - p*p;         /* get argument of rational function */
    x = (((((((2509.0809287301226727
        *x +  33430.575583588128105)
        *x +  67265.770927008700853)
        *x +  45921.953931549871457)
        *x +  13731.693765509461125)
        *x +   1971.5909503065514427)
        *x +    133.14166789178437745)
        *x +      3.387132872796366608)
      / (((((((5226.495278852854561
        *x +  28729.085735721942674)
        *x +  39307.89580009271061)
        *x +  21213.794301586595867)
        *x +   5394.1960214247511077)
        *x +    687.1870074920579083)
        *x +     42.313330701600911252)
        *x +      1.0);         /* evaluate the rational function */
    return p *x;                /* and return the computed value */
  }
  p = (prob > 0.5) ? 1-prob : prob;
  x = sqrt(-log(p));            /* transform to left tail if nec. */
  if (x <= 5) {                 /* if not extreme tail */
    x -= 1.6;                   /* get argument of rational function */
    x = (((((((  7.7454501427834140764e-4
        *x +     0.0227238449892691845833)
        *x +     0.24178072517745061177)
        *x +     1.27045825245236838258)
        *x +     3.64784832476320460504)
        *x +     5.7694972214606914055)
        *x +     4.6303378461565452959)
        *x +     1.42343711074968357734)
      / (((((((  1.05075007164441684324e-9
        *x +     5.475938084995344946e-4)
        *x +     0.0151986665636164571966)
        *x +     0.14810397642748007459)
        *x +     0.68976733498510000455)
        *x +     1.6763848301838038494)
        *x +     2.05319162663775882187)
        *x +     1.0); }        /* evaluate the rational function */
  else {                        /* if extreme tail */
    x -= 5;                     /* get argument of rational function */
    x = (((((((  2.01033439929228813265e-7
        *x +     2.71155556874348757815e-5)
        *x +     0.0012426609473880784386)
        *x +     0.026532189526576123093)
        *x +     0.29656057182850489123)
        *x +     1.7848265399172913358)
        *x +     5.4637849111641143699)
        *x +     6.6579046435011037772)
      / (((((((    2.04426310338993978564e-15
        *x +     1.4215117583164458887e-7)
        *x +     1.8463183175100546818e-5)
        *x +     7.868691311456132591e-4)
        *x +     0.0148753612908506148525)
        *x +     0.13692988092273580531)
        *x +     0.59983220655588793769)
        *x +     1.0);          /* evaluate the rational function */
  }
  return (prob < 0.5) ? -x : x; /* retransform to right tail if nec. */
}  /* unitqtlP() */

DEVICE double GammaqtlP (double prob, double k, double theta)
{                               /* --- quantile of Gamma distribution */
  int    n = 0;                 /* loop variable */
  double x, f, a, d, dx, dp;    /* buffers */

  /* check the function arguments */
  if ((k <= 0) || (theta <= 0) || (prob < 0) || (prob > 1)) return NPY_NAN;
  if (prob >= 1.0) return DBL_MAX;
  if (prob <= 0.0) return 0;    /* handle limiting values */
  if      (prob < 0.05) x = exp(logGamma(k) +log(prob) /k);
  else if (prob > 0.95) x = logGamma(k) -log1p(-prob);
  else {                        /* distinguish three prob. ranges */
    f = unitqtlP(prob); a = sqrt(k);
    x = (f >= -a) ? a *f +k : k;
  }                             /* compute initial approximation */
  do {                          /* Lagrange's interpolation */
    dp = prob -GammacdfP(x, k, 1);
    if ((dp == 0) || (++n > 33)) break;
    f = Gammapdf(x, k, 1);
    a = 2 *fabs(dp/x);
    a = dx = dp /((a > f) ? a : f);
    d = -0.25 *((k-1)/x -1) *a*a;
    if (fabs(d) < fabs(a)) dx += d;
    if (x +dx > 0) x += dx;
    else           x /= 2;
  } while (fabs(a) > 1e-10 *x);
  if (fabs(dp) > EPS_QTL *prob) return -1;
  return x *theta;              /* check for convergence and */
}  /* GammaqtlP() */            /* return the computed quantile */

/*--------------------------------------------------------------------*/

DEVICE double GammaqtlQ (double prob, double k, double theta)
{                               /* --- quantile of Gamma distribution */
  int    n = 0;                 /* loop variable */
  double x, f, a, d, dx, dp;    /* buffers */

  /* check the function arguments */
  if ((k <= 0) || (theta <= 0) || (prob < 0) || (prob > 1)) return NPY_NAN;
  if (prob <= 0.0) return DBL_MAX;
  if (prob >= 1.0) return 0;    /* handle limiting values */
  if      (prob < 0.05) x = logGamma(k) -log(prob);
  else if (prob > 0.95) x = exp(logGamma(k) +log1p(-prob) /k);
  else {                        /* distinguish three prob. ranges */
    f = unitqtlQ(prob); a = sqrt(k);
    x = (f >= -a) ? a *f +k : k;
  }                             /* compute initial approximation */
  do {                          /* Lagrange's interpolation */
    dp = prob -GammacdfQ(x, k, 1);
    if ((dp == 0) || (++n > 33)) break;
    f = Gammapdf(x, k, 1);
    a = 2 *fabs(dp/x);
    a = dx = -dp /((a > f) ? a : f);
    d = -0.25 *((k-1)/x -1) *a*a;
    if (fabs(d) < fabs(a)) dx += d;
    if (x +dx > 0) x += dx;
    else           x /= 2;
  } while (fabs(a) > 1e-10 *x);
  if (fabs(dp) > EPS_QTL *prob) return -1;
  return x *theta;              /* check for convergence and */
}  /* GammaqtlQ() */            /* return the computed quantile */
