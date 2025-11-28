r"""
`Op`\s that have their python implementations taken from SciPy.

As SciPy is not always available, we treat them separately.
"""

from functools import reduce
from pathlib import Path
from textwrap import dedent

import numpy as np
from scipy import special

from pytensor.configdefaults import config
from pytensor.gradient import grad_not_implemented, grad_undefined
from pytensor.scalar.basic import (
    BinaryScalarOp,
    ScalarOp,
    UnaryScalarOp,
    as_scalar,
    complex_types,
    constant,
    discrete_types,
    eq,
    exp,
    expm1,
    float64,
    float_types,
    floor,
    identity,
    integer_types,
    isinf,
    log,
    log1p,
    maximum,
    reciprocal,
    sqrt,
    switch,
    true_div,
    upcast,
    upgrade_to_float,
    upgrade_to_float_no_complex,
)
from pytensor.scalar.basic import abs as scalar_abs
from pytensor.scalar.loop import ScalarLoop


C_CODE_PATH = Path(__file__).parent / "c_code"


class Erf(UnaryScalarOp):
    nfunc_spec = ("scipy.special.erf", 1, 1)

    def impl(self, x):
        return special.erf(x)

    def L_op(self, inputs, outputs, grads):
        (x,) = inputs
        (gz,) = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        cst = np.asarray(
            2.0 / np.sqrt(np.pi), dtype=upcast(x.type.dtype, gz.type.dtype)
        )
        return (gz * cst * exp(-x * x),)

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = erf(({cast}){x});"


erf = Erf(upgrade_to_float, name="erf")


class Erfc(UnaryScalarOp):
    nfunc_spec = ("scipy.special.erfc", 1, 1)

    def impl(self, x):
        return special.erfc(x)

    def L_op(self, inputs, outputs, grads):
        (x,) = inputs
        (gz,) = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        cst = np.asarray(
            2.0 / np.sqrt(np.pi), dtype=upcast(x.type.dtype, gz.type.dtype)
        )
        return (-gz * cst * exp(-x * x),)

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = erfc(({cast}){x});"


# special.erfc don't support complex. Why?
erfc = Erfc(upgrade_to_float_no_complex, name="erfc")


class Erfcx(UnaryScalarOp):
    """
    Implements the scaled complementary error function exp(x**2)*erfc(x) in a
    numerically stable way for large x. This is useful for calculating things
    like log(erfc(x)) = log(erfcx(x)) - x ** 2 without causing underflow.
    Should only be used if x is known to be large and positive, as using
    erfcx(x) for large negative x may instead introduce overflow problems.

    Notes
    -----
    This op can still be executed on GPU, despite not having c_code. When
    running on GPU an optimization will replace it with a gpu version.

    """

    nfunc_spec = ("scipy.special.erfcx", 1, 1)

    def impl(self, x):
        return special.erfcx(x)

    def L_op(self, inputs, outputs, grads):
        (x,) = inputs
        (gz,) = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        cst = np.asarray(
            2.0 / np.sqrt(np.pi), dtype=upcast(x.type.dtype, gz.type.dtype)
        )
        return (gz * (-cst + (2.0 * x) * erfcx(x)),)

    def c_header_dirs(self, **kwargs):
        # Using the Faddeeva.hh (c++) header for Faddeevva.cc
        res = [*super().c_header_dirs(**kwargs), str(C_CODE_PATH)]
        return res

    def c_support_code(self, **kwargs):
        # Using Faddeeva.cc source file from: http://ab-initio.mit.edu/wiki/index.php/Faddeeva_Package
        return (C_CODE_PATH / "Faddeeva.cc").read_text(encoding="utf-8")

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out

        if node.inputs[0].type in float_types:
            dtype = "npy_" + node.outputs[0].dtype
            return f"{z} = ({dtype}) Faddeeva::erfcx({x});"

        raise NotImplementedError("type not supported", type)


erfcx = Erfcx(upgrade_to_float_no_complex, name="erfcx")


class Erfinv(UnaryScalarOp):
    """
    Implements the inverse error function.

    Notes
    -----
    This op can still be executed on GPU, despite not having c_code. When
    running on GPU, an optimization will replace it with a GPU version.

    (TODO) Find a C implementation of erfinv for CPU.
    """

    nfunc_spec = ("scipy.special.erfinv", 1, 1)

    def impl(self, x):
        return special.erfinv(x)

    def L_op(self, inputs, outputs, grads):
        (x,) = inputs
        (gz,) = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        cst = np.asarray(
            np.sqrt(np.pi) / 2.0, dtype=upcast(x.type.dtype, gz.type.dtype)
        )
        return (gz * cst * exp(erfinv(x) ** 2),)

    def c_code(self, node, name, inp, out, sub):
        # TODO: erfinv() is not provided by the C standard library
        # x, = inp
        # z, = out
        # if node.inputs[0].type in complex_types:
        #     raise NotImplementedError('type not supported', type)
        # return "%(z)s = erfinv(%(x)s);" % locals()
        raise NotImplementedError()


erfinv = Erfinv(upgrade_to_float_no_complex, name="erfinv")


class Erfcinv(UnaryScalarOp):
    nfunc_spec = ("scipy.special.erfcinv", 1, 1)

    def impl(self, x):
        return special.erfcinv(x)

    def L_op(self, inputs, outputs, grads):
        (x,) = inputs
        (gz,) = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        cst = np.asarray(
            np.sqrt(np.pi) / 2.0, dtype=upcast(x.type.dtype, gz.type.dtype)
        )
        return (-gz * cst * exp(erfcinv(x) ** 2),)

    def c_code(self, node, name, inp, out, sub):
        # TODO: erfcinv() is not provided by the C standard library
        # x, = inp
        # z, = out
        # if node.inputs[0].type in complex_types:
        #     raise NotImplementedError('type not supported', type)
        # return "%(z)s = erfcinv(%(x)s);" % locals()
        raise NotImplementedError()


erfcinv = Erfcinv(upgrade_to_float_no_complex, name="erfcinv")


class Owens_t(BinaryScalarOp):
    nfunc_spec = ("scipy.special.owens_t", 2, 1)

    def impl(self, h, a):
        return special.owens_t(h, a)

    def grad(self, inputs, grads):
        (h, a) = inputs
        (gz,) = grads
        return [
            gz
            * (-1)
            * exp(-(h**2) / 2)
            * erf(a * h / np.sqrt(2))
            / (2 * np.sqrt(2 * np.pi)),
            gz * exp(-0.5 * (a**2 + 1) * h**2) / (2 * np.pi * (a**2 + 1)),
        ]

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()


owens_t = Owens_t(upgrade_to_float, name="owens_t")


class Gamma(UnaryScalarOp):
    nfunc_spec = ("scipy.special.gamma", 1, 1)

    def impl(self, x):
        return special.gamma(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * gamma(x) * psi(x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in float_types:
            return f"""{z} = tgamma({x});"""
        raise NotImplementedError("only floating point is implemented")


gamma = Gamma(upgrade_to_float, name="gamma")


class GammaLn(UnaryScalarOp):
    """
    Log gamma function.

    """

    nfunc_spec = ("scipy.special.gammaln", 1, 1)

    def impl(self, x):
        return special.gammaln(x)

    def L_op(self, inputs, outputs, grads):
        (x,) = inputs
        (gz,) = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return [gz * psi(x)]

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        # no c code for complex
        # [u]int* will be casted to float64 before computation
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("gammaln complex c code is not implemented")
        # For some reason, on the GPU, uint64 inputs don't get casted
        # automatically to float64. This make the compilation crash
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"""{z} = lgamma(({cast}){x});"""


gammaln = GammaLn(upgrade_to_float, name="gammaln")


class Psi(UnaryScalarOp):
    """
    Derivative of log gamma function.

    """

    nfunc_spec = ("scipy.special.psi", 1, 1)

    def impl(self, x):
        return special.psi(x)

    def L_op(self, inputs, outputs, grads):
        (x,) = inputs
        (gz,) = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return [gz * tri_gamma(x)]

    def c_support_code(self, **kwargs):
        return """
            // For GPU support
            #ifdef WITHIN_KERNEL
            #define DEVICE WITHIN_KERNEL
            #else
            #define DEVICE
            #endif

            #ifndef M_PI
            #define M_PI 3.14159265358979323846
            #endif

            #ifndef _PSIFUNCDEFINED
            #define _PSIFUNCDEFINED
            DEVICE double _psi(double x) {

                /*taken from
                Bernardo, J. M. (1976). Algorithm AS 103:
                Psi (Digamma) Function. Applied Statistics. 25 (3), 315-317.
                http://www.uv.es/~bernardo/1976AppStatist.pdf
                */

                double y, R, psi_ = 0;
                double S  = 1.0e-5;
                double C = 8.5;
                double S3 = 8.333333333e-2;
                double S4 = 8.333333333e-3;
                double S5 = 3.968253968e-3;
                double D1 = -0.5772156649;

                if (x <= 0) {
                    // the digamma function approaches infinity from one side and -infinity from the other, around negative integers and zero
                    if (x == floor(x)) {
                        return INFINITY; // note that scipy returns -INF for 0 and NaN for negative integers
                    }

                    // Use reflection formula
                    double pi_x = M_PI * x;
                    double cot_pi_x = cos(pi_x) / sin(pi_x);
                    return _psi(1.0 - x) - M_PI * cot_pi_x;
                }

                y = x;

                if (y <= S)
                    return D1 - 1.0/y;

                while (y < C) {
                    psi_ = psi_ - 1.0 / y;
                    y = y + 1;
                }

                R = 1.0 / y;
                psi_ = psi_ + log(y) - .5 * R ;
                R= R*R;
                psi_ = psi_ - R * (S3 - R * (S4 - R * S5));

                return psi_;
            }
            #endif
            """

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            dtype = "npy_" + node.outputs[0].dtype
            return f"{z} = ({dtype}) _psi({x});"
        raise NotImplementedError("only floating point is implemented")


psi = Psi(upgrade_to_float, name="psi")


class TriGamma(UnaryScalarOp):
    """
    Second derivative of log gamma function.

    """

    def impl(self, x):
        return special.polygamma(1, x)

    def L_op(self, inputs, outputs, outputs_gradients):
        (x,) = inputs
        (g_out,) = outputs_gradients
        if x in complex_types:
            raise NotImplementedError("gradient not implemented for complex types")
        return [g_out * polygamma(2, x)]

    def c_support_code(self, **kwargs):
        # The implementation has been copied from
        # http://people.sc.fsu.edu/~jburkardt/cpp_src/asa121/asa121.html
        return """
            // For GPU support
            #ifdef WITHIN_KERNEL
            #define DEVICE WITHIN_KERNEL
            #else
            #define DEVICE
            #endif

            #ifndef ga_double
            #define ga_double double
            #endif

            #ifndef _TRIGAMMAFUNCDEFINED
            #define _TRIGAMMAFUNCDEFINED

            DEVICE double _tri_gamma(ga_double x) {

                double a = 0.0001;
                double b = 5.0;
                double b2 =  0.1666666667;
                double b4 = -0.03333333333;
                double b6 =  0.02380952381;
                double b8 = -0.03333333333;
                double value;
                double y;
                double z;

                if (x <= 0) {
                    return 0.0;
                }

                if ( x <= a ) {
                    value = 1.0 / x / x;
                    return value;
                }

                value = 0.0;
                z = x;

                while ( z < b ) {
                    value += 1.0 / z / z;
                    z += 1.0;
                }

                y = 1.0 / z / z;

                value +=  0.5 * y + (1.0 + y * (b2 + y * (b4 + y * (b6 + y * b8 )))) / z;

                return value;
            }
            #endif
            """

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            return f"""{z} =
                _tri_gamma({x});"""
        raise NotImplementedError("only floating point is implemented")


# Scipy polygamma does not support complex inputs: https://github.com/scipy/scipy/issues/7410
tri_gamma = TriGamma(upgrade_to_float_no_complex, name="tri_gamma")


class PolyGamma(BinaryScalarOp):
    """Polygamma function of order n evaluated at x.

    It corresponds to the (n+1)th derivative of the log gamma function.

    TODO: Because the first input is discrete and the output is continuous,
     the default elemwise inplace won't work, as it always tries to store the results in the first input.
    """

    nfunc_spec = ("scipy.special.polygamma", 2, 1)

    @staticmethod
    def output_types_preference(n_type, x_type):
        if n_type not in discrete_types:
            raise TypeError(
                f"Polygamma order parameter must be discrete, got {n_type} dtype"
            )
        # Scipy doesn't support it
        return upgrade_to_float_no_complex(x_type)

    def impl(self, n, x):
        return special.polygamma(n, x)

    def L_op(self, inputs, outputs, output_gradients):
        (n, x) = inputs
        (g_out,) = output_gradients
        if x in complex_types:
            raise NotImplementedError("gradient not implemented for complex types")
        return [
            grad_undefined(self, 0, n),
            g_out * self(n + 1, x),
        ]

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()


polygamma = PolyGamma(name="polygamma")


class GammaInc(BinaryScalarOp):
    """
    Compute the regularized lower gamma function (P).
    """

    nfunc_spec = ("scipy.special.gammainc", 2, 1)

    def impl(self, k, x):
        return special.gammainc(k, x)

    def grad(self, inputs, grads):
        (k, x) = inputs
        (gz,) = grads
        return [
            gz * gammainc_grad(k, x),
            gz * exp(-x + (k - 1) * log(x) - gammaln(k)),
        ]

    def c_support_code(self, **kwargs):
        return (C_CODE_PATH / "gamma.c").read_text(encoding="utf-8")

    def c_code(self, node, name, inp, out, sub):
        k, x = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            dtype = "npy_" + node.outputs[0].dtype
            return f"""{z} =
                ({dtype}) GammaP({k}, {x});"""
        raise NotImplementedError("only floatingpoint is implemented")

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))

    def c_code_cache_version(self):
        v = super().c_code_cache_version()
        if v:
            return (2, *v)
        else:
            return v


gammainc = GammaInc(upgrade_to_float, name="gammainc")


class GammaIncC(BinaryScalarOp):
    """
    Compute the regularized upper gamma function (Q).
    """

    nfunc_spec = ("scipy.special.gammaincc", 2, 1)

    def impl(self, k, x):
        return special.gammaincc(k, x)

    def grad(self, inputs, grads):
        (k, x) = inputs
        (gz,) = grads
        return [
            gz * gammaincc_grad(k, x),
            gz * -exp(-x + (k - 1) * log(x) - gammaln(k)),
        ]

    def c_support_code(self, **kwargs):
        return (C_CODE_PATH / "gamma.c").read_text(encoding="utf-8")

    def c_code(self, node, name, inp, out, sub):
        k, x = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            dtype = "npy_" + node.outputs[0].dtype
            return f"""{z} =
                ({dtype}) GammaQ({k}, {x});"""
        raise NotImplementedError("only floatingpoint is implemented")

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))

    def c_code_cache_version(self):
        v = super().c_code_cache_version()
        if v:
            return (2, *v)
        else:
            return v


gammaincc = GammaIncC(upgrade_to_float, name="gammaincc")


class GammaIncInv(BinaryScalarOp):
    """
    Inverse to the regularized lower incomplete gamma function.
    """

    nfunc_spec = ("scipy.special.gammaincinv", 2, 1)

    def impl(self, k, x):
        return special.gammaincinv(k, x)

    def grad(self, inputs, grads):
        (k, x) = inputs
        (gz,) = grads
        return [
            grad_not_implemented(self, 0, k),
            gz * exp(gammaincinv(k, x)) * gamma(k) * (gammaincinv(k, x) ** (1 - k)),
        ]

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()


gammaincinv = GammaIncInv(upgrade_to_float, name="gammaincinv")


class GammaIncCInv(BinaryScalarOp):
    """
    Inverse to the regularized upper incomplete gamma function.
    """

    nfunc_spec = ("scipy.special.gammainccinv", 2, 1)

    def impl(self, k, x):
        return special.gammainccinv(k, x)

    def grad(self, inputs, grads):
        (k, x) = inputs
        (gz,) = grads
        return [
            grad_not_implemented(self, 0, k),
            gz * -exp(gammainccinv(k, x)) * gamma(k) * (gammainccinv(k, x) ** (1 - k)),
        ]

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()


gammainccinv = GammaIncCInv(upgrade_to_float, name="gammainccinv")


def _make_scalar_loop(n_steps, init, constant, inner_loop_fn, name, loop_op=ScalarLoop):
    init = [as_scalar(x) if x is not None else None for x in init]
    constant = [as_scalar(x) for x in constant]

    # Create dummy types, in case some variables have the same initial form
    init_ = [x.type() if x is not None else None for x in init]
    constant_ = [x.type() for x in constant]
    update_, until_ = inner_loop_fn(*init_, *constant_)

    # Filter Nones
    init = [i for i in init if i is not None]
    init_ = [i for i in init_ if i is not None]
    update_ = [u for u in update_ if u is not None]
    op = loop_op(
        init=init_,
        constant=constant_,
        update=update_,
        until=until_,
        name=name,
    )
    return op(n_steps, *init, *constant)


def gammainc_grad(k, x):
    """Gradient of the regularized lower gamma function (P) wrt to the first
    argument (k, a.k.a. alpha).

    Adapted from STAN `grad_reg_lower_inc_gamma.hpp`

    Reference: Gautschi, W. (1979). A computational procedure for incomplete gamma functions.
    ACM Transactions on Mathematical Software (TOMS), 5(4), 466-481.
    """
    dtype = upcast(k.type.dtype, x.type.dtype, "float32")

    def grad_approx(skip_loop):
        precision = np.array(1e-10, dtype=config.floatX)
        max_iters = switch(
            skip_loop, np.array(0, dtype="int32"), np.array(1e5, dtype="int32")
        )

        log_x = log(x)
        log_gamma_k_plus_1 = gammaln(k + 1)

        # First loop
        k_plus_n = k  # Should not overflow unless k > 2,147,383,647
        log_gamma_k_plus_n_plus_1 = log_gamma_k_plus_1
        sum_a0 = np.array(0.0, dtype=dtype)

        def inner_loop_a(sum_a, log_gamma_k_plus_n_plus_1, k_plus_n, log_x):
            term = exp(k_plus_n * log_x - log_gamma_k_plus_n_plus_1)
            sum_a += term

            log_gamma_k_plus_n_plus_1 += log1p(k_plus_n)
            k_plus_n += 1
            return (
                (sum_a, log_gamma_k_plus_n_plus_1, k_plus_n),
                (term <= precision),
            )

        init = [sum_a0, log_gamma_k_plus_n_plus_1, k_plus_n]
        constant = [log_x]
        sum_a, *_, sum_a_converges = _make_scalar_loop(
            max_iters, init, constant, inner_loop_a, name="gammainc_grad_a"
        )
        sum_a = switch(sum_a_converges, sum_a, np.nan)

        # Second loop
        n = np.array(0, dtype="int32")
        log_gamma_k_plus_n_plus_1 = log_gamma_k_plus_1
        k_plus_n = k
        sum_b0 = np.array(0.0, dtype=dtype)

        def inner_loop_b(sum_b, log_gamma_k_plus_n_plus_1, n, k_plus_n, log_x):
            term = exp(k_plus_n * log_x - log_gamma_k_plus_n_plus_1) * psi(k_plus_n + 1)
            sum_b += term

            log_gamma_k_plus_n_plus_1 += log1p(k_plus_n)
            n += 1
            k_plus_n += 1
            return (
                (sum_b, log_gamma_k_plus_n_plus_1, n, k_plus_n),
                # Require at least two iterations
                ((term <= precision) & (n > 1)),
            )

        init = [sum_b0, log_gamma_k_plus_n_plus_1, n, k_plus_n]
        constant = [log_x]
        sum_b, *_, sum_b_converges = _make_scalar_loop(
            max_iters, init, constant, inner_loop_b, name="gammainc_grad_b"
        )
        sum_b = switch(sum_b_converges, sum_b, np.nan)

        grad_approx = exp(-x) * (log_x * sum_a - sum_b)
        return grad_approx

    zero_branch = eq(x, 0)
    sqrt_exp = -756 - x**2 + 60 * x
    gammaincc_branch = (
        ((k < 0.8) & (x > 15))
        | ((k < 12) & (x > 30))
        | ((sqrt_exp > 0) & (k < sqrt(sqrt_exp)))
    )
    grad = switch(
        zero_branch,
        0,
        switch(
            gammaincc_branch,
            -gammaincc_grad(k, x, skip_loops=zero_branch | (~gammaincc_branch)),
            grad_approx(skip_loop=zero_branch | gammaincc_branch),
        ),
    )
    return grad


def gammaincc_grad(k, x, skip_loops=constant(False, dtype="bool")):
    """Gradient of the regularized upper gamma function (Q) wrt to the first
    argument (k, a.k.a. alpha).

    Adapted from STAN `grad_reg_inc_gamma.hpp`

    skip_loops is used for faster branching when this function is called by `gammainc_der`
    """
    dtype = upcast(k.type.dtype, x.type.dtype, "float32")

    gamma_k = gamma(k)
    digamma_k = psi(k)
    log_x = log(x)

    def approx_a(skip_loop):
        n_steps = switch(
            skip_loop, np.array(0, dtype="int32"), np.array(9, dtype="int32")
        )
        sum_a0 = np.array(0.0, dtype=dtype)
        dfac = np.array(1.0, dtype=dtype)
        xpow = x
        k_minus_one_minus_n = k - 1
        fac = k_minus_one_minus_n
        delta = true_div(dfac, xpow)

        def inner_loop_a(sum_a, delta, xpow, k_minus_one_minus_n, fac, dfac, x):
            sum_a += delta
            xpow *= x
            k_minus_one_minus_n -= 1
            dfac = k_minus_one_minus_n * dfac + fac
            fac *= k_minus_one_minus_n
            delta = dfac / xpow
            return (sum_a, delta, xpow, k_minus_one_minus_n, fac, dfac), None

        init = [sum_a0, delta, xpow, k_minus_one_minus_n, fac, dfac]
        constant = [x]
        sum_a, *_ = _make_scalar_loop(
            n_steps, init, constant, inner_loop_a, name="gammaincc_grad_a"
        )
        grad_approx_a = (
            gammaincc(k, x) * (log_x - digamma_k)
            + exp(-x + (k - 1) * log_x) * sum_a / gamma_k
        )
        return grad_approx_a

    def approx_b(skip_loop):
        max_iters = switch(
            skip_loop, np.array(0, dtype="int32"), np.array(1e5, dtype="int32")
        )
        log_precision = np.array(np.log(1e-6), dtype=config.floatX)

        sum_b0 = np.array(0.0, dtype=dtype)
        log_s = np.array(0.0, dtype=dtype)
        s_sign = np.array(1, dtype="int8")
        n = np.array(1, dtype="int32")
        log_delta = log_s - 2 * log(k).astype(dtype)

        def inner_loop_b(sum_b, log_s, s_sign, log_delta, n, k, log_x):
            delta = exp(log_delta)
            sum_b += switch(s_sign > 0, delta, -delta)
            s_sign = -s_sign

            # log will cast >int16 to float64
            log_s += log_x - log(n)
            if log_s.type.dtype != dtype:
                log_s = log_s.astype(dtype)

            log_delta = log_s - 2 * log(n + k)
            if log_delta.type.dtype != dtype:
                log_delta = log_delta.astype(dtype)

            n += 1
            return (
                (sum_b, log_s, s_sign, log_delta, n),
                log_delta <= log_precision,
            )

        init = [sum_b0, log_s, s_sign, log_delta, n]
        constant = [k, log_x]
        sum_b, *_, sum_b_converges = _make_scalar_loop(
            max_iters, init, constant, inner_loop_b, name="gammaincc_grad_b"
        )
        sum_b = switch(sum_b_converges, sum_b, np.nan)
        grad_approx_b = (
            gammainc(k, x) * (digamma_k - log_x) + exp(k * log_x) * sum_b / gamma_k
        )
        return grad_approx_b

    branch_a = (x >= k) & (x >= 8)
    return switch(
        branch_a,
        approx_a(skip_loop=~branch_a | skip_loops),
        approx_b(skip_loop=branch_a | skip_loops),
    )


class GammaU(BinaryScalarOp):
    """
    compute the upper incomplete gamma function.
    """

    # Note there is no basic SciPy version so no nfunc_spec.

    def impl(self, k, x):
        return special.gammaincc(k, x) * special.gamma(k)

    def c_support_code(self, **kwargs):
        return (C_CODE_PATH / "gamma.c").read_text(encoding="utf-8")

    def c_code(self, node, name, inp, out, sub):
        k, x = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            dtype = "npy_" + node.outputs[0].dtype
            return f"""{z} =
                ({dtype}) upperGamma({k}, {x});"""
        raise NotImplementedError("only floatingpoint is implemented")

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))


gammau = GammaU(upgrade_to_float, name="gammau")


class GammaL(BinaryScalarOp):
    """
    Compute the lower incomplete gamma function.
    """

    # Note there is no basic SciPy version so no nfunc_spec.

    def impl(self, k, x):
        return special.gammainc(k, x) * special.gamma(k)

    def c_support_code(self, **kwargs):
        return (C_CODE_PATH / "gamma.c").read_text(encoding="utf-8")

    def c_code(self, node, name, inp, out, sub):
        k, x = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            dtype = "npy_" + node.outputs[0].dtype
            return f"""{z} =
                ({dtype}) lowerGamma({k}, {x});"""
        raise NotImplementedError("only floatingpoint is implemented")

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))


gammal = GammaL(upgrade_to_float, name="gammal")


class Jv(BinaryScalarOp):
    """
    Bessel function of the first kind of order v (real).
    """

    nfunc_spec = ("scipy.special.jv", 2, 1)

    def impl(self, v, x):
        return special.jv(v, x)

    def grad(self, inputs, grads):
        v, x = inputs
        (gz,) = grads
        return [
            grad_not_implemented(self, 0, v),
            gz * (jv(v - 1, x) - jv(v + 1, x)) / 2.0,
        ]

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()


jv = Jv(upgrade_to_float, name="jv")


class J1(UnaryScalarOp):
    """
    Bessel function of the first kind of order 1.
    """

    nfunc_spec = ("scipy.special.j1", 1, 1)

    def impl(self, x):
        return special.j1(x)

    def grad(self, inputs, grads):
        (x,) = inputs
        (gz,) = grads
        return [gz * (j0(x) - jv(2, x)) / 2.0]

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            return f"""{z} =
                j1({x});"""
        raise NotImplementedError("only floating point is implemented")


j1 = J1(upgrade_to_float, name="j1")


class J0(UnaryScalarOp):
    """
    Bessel function of the first kind of order 0.
    """

    nfunc_spec = ("scipy.special.j0", 1, 1)

    def impl(self, x):
        return special.j0(x)

    def grad(self, inp, grads):
        (x,) = inp
        (gz,) = grads
        return [gz * -1 * j1(x)]

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            return f"""{z} =
                j0({x});"""
        raise NotImplementedError("only floating point is implemented")


j0 = J0(upgrade_to_float, name="j0")


class Iv(BinaryScalarOp):
    """
    Modified Bessel function of the first kind of order v (real).
    """

    nfunc_spec = ("scipy.special.iv", 2, 1)

    def impl(self, v, x):
        return special.iv(v, x)

    def grad(self, inputs, grads):
        v, x = inputs
        (gz,) = grads
        return [
            grad_not_implemented(self, 0, v),
            gz * (iv(v - 1, x) + iv(v + 1, x)) / 2.0,
        ]

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()


iv = Iv(upgrade_to_float, name="iv")


class I1(UnaryScalarOp):
    """
    Modified Bessel function of the first kind of order 1.
    """

    nfunc_spec = ("scipy.special.i1", 1, 1)

    def impl(self, x):
        return special.i1(x)

    def grad(self, inputs, grads):
        (x,) = inputs
        (gz,) = grads
        return [gz * (i0(x) + iv(2, x)) / 2.0]

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()


i1 = I1(upgrade_to_float, name="i1")


class I0(UnaryScalarOp):
    """
    Modified Bessel function of the first kind of order 0.
    """

    nfunc_spec = ("scipy.special.i0", 1, 1)

    def impl(self, x):
        return special.i0(x)

    def grad(self, inp, grads):
        (x,) = inp
        (gz,) = grads
        return [gz * i1(x)]

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()


i0 = I0(upgrade_to_float, name="i0")


class Ive(BinaryScalarOp):
    """
    Exponentially scaled modified Bessel function of the first kind of order v (real).
    """

    nfunc_spec = ("scipy.special.ive", 2, 1)

    def impl(self, v, x):
        return special.ive(v, x)

    def grad(self, inputs, grads):
        v, x = inputs
        (gz,) = grads
        return [
            grad_not_implemented(self, 0, v),
            gz
            * (ive(v - 1, x) - 2.0 * _unsafe_sign(x) * ive(v, x) + ive(v + 1, x))
            / 2.0,
        ]

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()


ive = Ive(upgrade_to_float, name="ive")


class Kve(BinaryScalarOp):
    """Exponentially scaled modified Bessel function of the second kind of real order v."""

    nfunc_spec = ("scipy.special.kve", 2, 1)

    def impl(self, v, x):
        return special.kve(v, x)

    def L_op(self, inputs, outputs, output_grads):
        v, x = inputs
        [kve_vx] = outputs
        [g_out] = output_grads
        # (1 -v/x) * kve(v, x) - kve(v - 1, x)
        kve_vm1x = self(v - 1, x)
        dx = (1 - v / x) * kve_vx - kve_vm1x

        return [
            grad_not_implemented(self, 0, v),
            g_out * dx,
        ]

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()


kve = Kve(upgrade_to_float, name="kve")


class Sigmoid(UnaryScalarOp):
    """
    Logistic sigmoid function (1 / (1 + exp(-x)), also known as expit or inverse logit
    """

    nfunc_spec = ("scipy.special.expit", 1, 1)

    def impl(self, x):
        return special.expit(x)

    def grad(self, inp, grads):
        (x,) = inp
        (gz,) = grads
        y = sigmoid(x)
        rval = gz * y * (1.0 - y)

        assert rval.type.dtype.find("float") != -1

        return [rval]

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out

        if node.inputs[0].type in float_types:
            if node.inputs[0].type == float64:
                return f"""{z} = 1.0 / (1.0 + exp(-{x}));"""
            else:
                return f"""{z} = 1.0f / (1.0f + exp(-{x}));"""
        else:
            raise NotImplementedError("only floatingpoint is implemented")

    def c_code_cache_version(self):
        v = super().c_code_cache_version()
        if v:
            return (2, *v)
        else:
            return v


sigmoid = Sigmoid(upgrade_to_float, name="sigmoid")


class Softplus(UnaryScalarOp):
    r"""
    Compute log(1 + exp(x)), also known as softplus or log1pexp

    This function is numerically faster than the naive approach, and does not overflow
    for large values of x.

    For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

    References
    ----------
    .. [Machler2012] Martin Mächler (2012).
        "Accurately computing `\log(1-\exp(- \mid a \mid))` Assessed by the Rmpfr package"
    """

    def impl(self, x):
        # If x is an int8 or uint8, numpy.exp will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = getattr(x, "dtype", None)
        not_int8 = x_dtype is None or x_dtype.itemsize > 1
        if x < -37.0:
            return np.exp(x) if not_int8 else np.exp(x, signature="f")
        elif x < 18.0:
            return (
                np.log1p(np.exp(x)) if not_int8 else np.log1p(np.exp(x, signature="f"))
            )
        elif x < 33.3:
            if x_dtype is not None and x_dtype.kind == "u":
                # Negate uint will not do what we want
                x = x.astype("float32" if x_dtype.itemsize <= 2 else "float64")
            return x + np.exp(-x) if not_int8 else x + np.exp(-x, signature="f")
        else:
            return x

    def grad(self, inp, grads):
        (x,) = inp
        (gz,) = grads
        return [gz * sigmoid(x)]

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        # We use the same limits for all precisions, which may be suboptimal. The reference
        # paper only looked at double precision
        if node.inputs[0].type in float_types:
            if node.inputs[0].type == float64:
                return dedent(
                    f"""
                    {z} = (
                        {x} < -37.0 ? exp({x}) :
                        {x} < 18.0 ? log1p(exp({x})) :
                        {x} < 33.3 ? {x} + exp(-{x}) :
                        {x}
                    );
                    """
                )
            else:
                return dedent(
                    f"""
                    {z} = (
                        {x} < -37.0f ? exp({x}) :
                        {x} < 18.0f ? log1p(exp({x})) :
                        {x} < 33.3f ? {x} + exp(-{x}) :
                        {x}
                    );
                    """
                )
        else:
            raise NotImplementedError("only floatingpoint is implemented")

    def c_code_cache_version(self):
        v = super().c_code_cache_version()
        if v:
            return (3, *v)
        else:
            return v


softplus = Softplus(upgrade_to_float)


class Log1mexp(UnaryScalarOp):
    r"""
    Compute log(1 - exp(x)), also known as log1mexp

    This function is numerically more stable than the naive approach.

    For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

    References
    ----------
    .. [Machler2012] Martin Mächler (2012).
        "Accurately computing `\log(1-\exp(- \mid a \mid))` Assessed by the Rmpfr package"
    """

    def impl(self, x):
        if x < np.log(0.5):
            return np.log1p(-np.exp(x))
        else:
            return np.log(-np.expm1(x))

    def grad(self, inp, grads):
        (x,) = inp
        (gz,) = grads
        res = true_div(-1.0, expm1(-x))
        # Correct gradient at 0.0 to be -inf
        res = switch(isinf(res), -np.inf, res)
        return [gz * res]

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out

        if node.inputs[0].type in float_types:
            if node.inputs[0].type == float64:
                return f"{z} = {x} < -0.6931471805599453 ? log1p(-exp({x})) : log(-expm1({x}));"
            else:
                return f"{z} = {x} < -0.6931471805599453f ? log1p(-exp({x})) : log(-expm1({x}));"
        else:
            raise NotImplementedError("only floating point is implemented")


log1mexp = Log1mexp(upgrade_to_float)


class BetaInc(ScalarOp):
    """
    Regularized incomplete beta function
    """

    nin = 3
    nfunc_spec = ("scipy.special.betainc", 3, 1)

    def impl(self, a, b, x):
        return special.betainc(a, b, x)

    def grad(self, inp, grads):
        a, b, x = inp
        (gz,) = grads

        return [
            gz * betainc_grad(a, b, x, True),
            gz * betainc_grad(a, b, x, False),
            gz
            * exp(
                log1p(-x) * (b - 1)
                + log(x) * (a - 1)
                - (gammaln(a) + gammaln(b) - gammaln(a + b))
            ),
        ]

    def c_support_code(self, **kwargs):
        return (C_CODE_PATH / "incbet.c").read_text(encoding="utf-8")

    def c_code(self, node, name, inp, out, sub):
        (a, b, x) = inp
        (z,) = out
        if (
            node.inputs[0].type in float_types
            and node.inputs[1].type in float_types
            and node.inputs[2].type in float_types
        ):
            return f"""{z} = BetaInc({a}, {b}, {x});"""

        raise NotImplementedError("type not supported", type)

    def c_code_cache_version(self):
        return (2,)


betainc = BetaInc(upgrade_to_float_no_complex, name="betainc")


def betainc_grad(p, q, x, wrtp: bool):
    """
    Gradient of the regularized incomplete beta function wrt to the first
    argument `p` (aka alpha) or the second argument `q` (aka beta),
    depending on whether `wrtp` is true.

    Reference: Boik, R. J., & Robison-Cox, J. F. (1998). Derivatives of the incomplete beta function.
    Journal of Statistical Software, 3(1), 1-20.
    """

    def _betainc_der(p, q, x, wrtp, skip_loop):
        dtype = upcast(p.type.dtype, q.type.dtype, x.type.dtype, "float32")

        def betaln(a, b):
            return gammaln(a) + (gammaln(b) - gammaln(a + b))

        def _betainc_a_n(f, p, q, n):
            """
            Numerator (a_n) of the nth approximant of the continued fraction
            representation of the regularized incomplete beta function
            """

            p2n = p + 2 * n
            F1 = p**2 * f**2 * (n - 1) / (q**2)
            F2 = (
                (p + q + n - 2)
                * (p + n - 1)
                * (q - n)
                / ((p2n - 3) * (p2n - 2) ** 2 * (p2n - 1))
            )

            return switch(
                eq(n, 1),
                p * f * (q - 1) / (q * (p + 1)),
                F1 * F2,
            )

        def _betainc_b_n(f, p, q, n):
            """
            Offset (b_n) of the nth approximant of the continued fraction
            representation of the regularized incomplete beta function
            """
            pf = p * f
            p2n = p + 2 * n

            N1 = 2 * (pf + 2 * q) * n * (n + p - 1) + p * q * (p - 2 - pf)
            D1 = q * (p2n - 2) * p2n

            return N1 / D1

        def _betainc_da_n_dp(f, p, q, n):
            """
            Derivative of a_n wrt p
            """

            pp = p**2
            ppp = pp * p
            p2n = p + 2 * n

            N1 = -(n - 1) * f**2 * pp * (q - n)
            N2a = (-8 + 8 * p + 8 * q) * n**3
            N2b = (16 * pp + (-44 + 20 * q) * p + 26 - 24 * q) * n**2
            N2c = (10 * ppp + (14 * q - 46) * pp + (-40 * q + 66) * p - 28 + 24 * q) * n
            N2d = 2 * pp**2 + (-13 + 3 * q) * ppp + (-14 * q + 30) * pp
            N2e = (-29 + 19 * q) * p + 10 - 8 * q

            D1 = q**2 * (p2n - 3) ** 2
            D2 = (p2n - 2) ** 3 * (p2n - 1) ** 2

            return switch(
                eq(n, 1),
                -p * f * (q - 1) / (q * (p + 1) ** 2),
                (N1 / D1) * (N2a + N2b + N2c + N2d + N2e) / D2,
            )

        def _betainc_da_n_dq(f, p, q, n):
            """
            Derivative of a_n wrt q
            """
            p2n = p + 2 * n
            F1 = (p**2 * f**2 / (q**2)) * (n - 1) * (p + n - 1) * (2 * q + p - 2)
            D1 = (p2n - 3) * (p2n - 2) ** 2 * (p2n - 1)

            return switch(
                eq(n, 1),
                p * f / (q * (p + 1)),
                F1 / D1,
            )

        def _betainc_db_n_dp(f, p, q, n):
            """
            Derivative of b_n wrt p
            """
            p2n = p + 2 * n
            pp = p**2
            q4 = 4 * q
            p4 = 4 * p

            F1 = (p * f / q) * (
                (-p4 - q4 + 4) * n**2 + (p4 - 4 + q4 - 2 * pp) * n + pp * q
            )
            D1 = (p2n - 2) ** 2 * p2n**2

            return F1 / D1

        def _betainc_db_n_dq(f, p, q, n):
            """
            Derivative of b_n wrt to q
            """
            p2n = p + 2 * n
            return -(p**2 * f) / (q * (p2n - 2) * p2n)

        min_iters = np.array(3, dtype="int32")
        max_iters = switch(
            skip_loop, np.array(0, dtype="int32"), np.array(200, dtype="int32")
        )
        err_threshold = np.array(1e-12, dtype=config.floatX)

        Am2, Am1 = np.array(1, dtype=dtype), np.array(1, dtype=dtype)
        Bm2, Bm1 = np.array(0, dtype=dtype), np.array(1, dtype=dtype)
        dAm2, dAm1 = np.array(0, dtype=dtype), np.array(0, dtype=dtype)
        dBm2, dBm1 = np.array(0, dtype=dtype), np.array(0, dtype=dtype)

        f = (q * x) / (p * (1 - x))
        K = exp(p * log(x) + (q - 1) * log1p(-x) - log(p) - betaln(p, q))
        if wrtp:
            dK = log(x) - reciprocal(p) + psi(p + q) - psi(p)
        else:
            dK = log1p(-x) + psi(p + q) - psi(q)

        derivative = np.array(0, dtype=dtype)
        n = np.array(1, dtype="int16")  # Enough for 200 max iters

        def inner_loop(
            derivative,
            Am2,
            Am1,
            Bm2,
            Bm1,
            dAm2,
            dAm1,
            dBm2,
            dBm1,
            n,
            f,
            p,
            q,
            K,
            dK,
        ):
            a_n_ = _betainc_a_n(f, p, q, n)
            b_n_ = _betainc_b_n(f, p, q, n)
            if wrtp:
                da_n = _betainc_da_n_dp(f, p, q, n)
                db_n = _betainc_db_n_dp(f, p, q, n)
            else:
                da_n = _betainc_da_n_dq(f, p, q, n)
                db_n = _betainc_db_n_dq(f, p, q, n)

            A = a_n_ * Am2 + b_n_ * Am1
            B = a_n_ * Bm2 + b_n_ * Bm1
            dA = da_n * Am2 + a_n_ * dAm2 + db_n * Am1 + b_n_ * dAm1
            dB = da_n * Bm2 + a_n_ * dBm2 + db_n * Bm1 + b_n_ * dBm1

            Am2, Am1 = identity(Am1), identity(A)
            Bm2, Bm1 = identity(Bm1), identity(B)
            dAm2, dAm1 = identity(dAm1), identity(dA)
            dBm2, dBm1 = identity(dBm1), identity(dB)

            F1 = A / B
            F2 = (dA - F1 * dB) / B
            derivative_new = K * (F1 * dK + F2)

            errapx = scalar_abs(derivative - derivative_new)
            d_errapx = errapx / maximum(err_threshold, scalar_abs(derivative_new))

            min_iters_cond = n > (min_iters - 1)
            derivative = switch(
                min_iters_cond,
                derivative_new,
                derivative,
            )
            n += 1

            return (
                (derivative, Am2, Am1, Bm2, Bm1, dAm2, dAm1, dBm2, dBm1, n),
                (d_errapx <= err_threshold) & min_iters_cond,
            )

        init = [derivative, Am2, Am1, Bm2, Bm1, dAm2, dAm1, dBm2, dBm1, n]
        constant = [f, p, q, K, dK]
        grad, *_, grad_converges = _make_scalar_loop(
            max_iters, init, constant, inner_loop, name="betainc_grad"
        )
        return switch(grad_converges, grad, np.nan)

    # Input validation
    nan_branch = (x < 0) | (x > 1) | (p < 0) | (q < 0)
    flip_branch = x > (p / (p + q))
    grad = switch(
        nan_branch,
        np.nan,
        switch(
            flip_branch,
            -_betainc_der(q, p, 1 - x, not wrtp, skip_loop=nan_branch | (~flip_branch)),
            _betainc_der(p, q, x, wrtp, skip_loop=nan_branch | flip_branch),
        ),
    )
    return grad


class BetaIncInv(ScalarOp):
    """
    Inverse of the regularized incomplete beta function.
    """

    nfunc_spec = ("scipy.special.betaincinv", 3, 1)

    def impl(self, a, b, x):
        return special.betaincinv(a, b, x)

    def grad(self, inputs, grads):
        (a, b, x) = inputs
        (gz,) = grads
        return [
            grad_not_implemented(self, 0, a),
            grad_not_implemented(self, 0, b),
            gz
            * exp(betaln(a, b))
            * ((1 - betaincinv(a, b, x)) ** (1 - b))
            * (betaincinv(a, b, x) ** (1 - a)),
        ]

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()


betaincinv = BetaIncInv(upgrade_to_float_no_complex, name="betaincinv")


def betaln(a, b):
    """
    Beta function from gamma function.
    """

    return gammaln(a) + gammaln(b) - gammaln(a + b)


class Hyp2F1(ScalarOp):
    """
    Gaussian hypergeometric function ``2F1(a, b; c; z)``.

    """

    nin = 4
    nfunc_spec = ("scipy.special.hyp2f1", 4, 1)

    def impl(self, a, b, c, z):
        return special.hyp2f1(a, b, c, z)

    def grad(self, inputs, grads):
        a, b, c, z = inputs
        (gz,) = grads
        grad_a, grad_b, grad_c = hyp2f1_grad(a, b, c, z, wrt=[0, 1, 2])
        return [
            gz * grad_a,
            gz * grad_b,
            gz * grad_c,
            gz * ((a * b) / c) * hyp2f1(a + 1, b + 1, c + 1, z),
        ]

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()


hyp2f1 = Hyp2F1(upgrade_to_float, name="hyp2f1")


def _unsafe_sign(x):
    # Unlike scalar.sign we don't worry about x being 0 or nan
    return switch(x > 0, 1, -1)


class Grad2F1Loop(ScalarLoop):
    """Subclass of ScalarLoop for easier targetting in rewrites"""


def _grad_2f1_loop(a, b, c, z, *, skip_loop, wrt, dtype):
    """
    Notes
    -----
    The algorithm can be derived by looking at the ratio of two successive terms in the series
    β_{k+1}/β_{k} = A(k)/B(k)
    β_{k+1} = A(k)/B(k) * β_{k}
    d[β_{k+1}] = d[A(k)/B(k)] * β_{k} + A(k)/B(k) * d[β_{k}] via the product rule

    In the 2F1, A(k)/B(k) corresponds to (((a + k) * (b + k) / ((c + k) (1 + k))) * z

    The partial d[A(k)/B(k)] with respect to the 3 first inputs can be obtained from the ratio A(k)/B(k),
    by dropping the respective term
    d/da[A(k)/B(k)] = A(k)/B(k) / (a + k)
    d/db[A(k)/B(k)] = A(k)/B(k) / (b + k)
    d/dc[A(k)/B(k)] = A(k)/B(k) * (c + k)

    The algorithm is implemented in the log scale, which adds the complexity of working with absolute terms and
    tracking their signs.
    """

    min_steps = np.array(
        10, dtype="int32"
    )  # https://github.com/stan-dev/math/issues/2857
    max_steps = switch(
        skip_loop, np.array(0, dtype="int32"), np.array(int(1e6), dtype="int32")
    )
    precision = np.array(1e-14, dtype=config.floatX)

    grads = [np.array(0, dtype=dtype) if i in wrt else None for i in range(3)]
    log_gs = [np.array(-np.inf, dtype=dtype) if i in wrt else None for i in range(3)]
    log_gs_signs = [np.array(1, dtype="int8") if i in wrt else None for i in range(3)]

    log_t = np.array(0.0, dtype=dtype)
    log_t_sign = np.array(1, dtype="int8")

    log_z = log(scalar_abs(z))
    sign_z = _unsafe_sign(z)

    sign_zk = sign_z
    k = np.array(0, dtype="int32")

    def inner_loop(*args):
        (
            *grads_vars,
            log_t,
            log_t_sign,
            sign_zk,
            k,
            a,
            b,
            c,
            log_z,
            sign_z,
        ) = args

        (
            grad_a,
            grad_b,
            grad_c,
            log_g_a,
            log_g_b,
            log_g_c,
            log_g_sign_a,
            log_g_sign_b,
            log_g_sign_c,
        ) = grads_vars

        p = (a + k) * (b + k) / ((c + k) * (k + 1))
        if p.type.dtype != dtype:
            p = p.astype(dtype)

        # If p==0, don't update grad and get out of while loop next
        p_zero = eq(p, 0)

        if 0 in wrt:
            term_a = log_g_sign_a * log_t_sign * exp(log_g_a - log_t)
            term_a += reciprocal(a + k)
            if term_a.type.dtype != dtype:
                term_a = term_a.astype(dtype)
        if 1 in wrt:
            term_b = log_g_sign_b * log_t_sign * exp(log_g_b - log_t)
            term_b += reciprocal(b + k)
            if term_b.type.dtype != dtype:
                term_b = term_b.astype(dtype)
        if 2 in wrt:
            term_c = log_g_sign_c * log_t_sign * exp(log_g_c - log_t)
            term_c -= reciprocal(c + k)
            if term_c.type.dtype != dtype:
                term_c = term_c.astype(dtype)

        log_t = log_t + log(scalar_abs(p)) + log_z
        log_t_sign = (_unsafe_sign(p) * log_t_sign).astype("int8")

        grads = [None] * 3
        log_gs = [None] * 3
        log_gs_signs = [None] * 3
        grad_incs = [None] * 3

        if 0 in wrt:
            log_g_a = log_t + log(scalar_abs(term_a))
            log_g_sign_a = (_unsafe_sign(term_a) * log_t_sign).astype("int8")
            grad_inc_a = log_g_sign_a * exp(log_g_a) * sign_zk
            grads[0] = switch(p_zero, grad_a, grad_a + grad_inc_a)
            log_gs[0] = log_g_a
            log_gs_signs[0] = log_g_sign_a
            grad_incs[0] = grad_inc_a
        if 1 in wrt:
            log_g_b = log_t + log(scalar_abs(term_b))
            log_g_sign_b = (_unsafe_sign(term_b) * log_t_sign).astype("int8")
            grad_inc_b = log_g_sign_b * exp(log_g_b) * sign_zk
            grads[1] = switch(p_zero, grad_b, grad_b + grad_inc_b)
            log_gs[1] = log_g_b
            log_gs_signs[1] = log_g_sign_b
            grad_incs[1] = grad_inc_b
        if 2 in wrt:
            log_g_c = log_t + log(scalar_abs(term_c))
            log_g_sign_c = (_unsafe_sign(term_c) * log_t_sign).astype("int8")
            grad_inc_c = log_g_sign_c * exp(log_g_c) * sign_zk
            grads[2] = switch(p_zero, grad_c, grad_c + grad_inc_c)
            log_gs[2] = log_g_c
            log_gs_signs[2] = log_g_sign_c
            grad_incs[2] = grad_inc_c

        sign_zk *= sign_z
        k += 1

        abs_grad_incs = [
            scalar_abs(grad_inc) for grad_inc in grad_incs if grad_inc is not None
        ]
        if len(grad_incs) == 1:
            [max_abs_grad_inc] = grad_incs
        else:
            max_abs_grad_inc = reduce(maximum, abs_grad_incs)

        return (
            (*grads, *log_gs, *log_gs_signs, log_t, log_t_sign, sign_zk, k),
            (eq(p, 0) | ((k > min_steps) & (max_abs_grad_inc <= precision))),
        )

    init = [*grads, *log_gs, *log_gs_signs, log_t, log_t_sign, sign_zk, k]
    constant = [a, b, c, log_z, sign_z]
    *loop_outs, converges = _make_scalar_loop(
        max_steps, init, constant, inner_loop, name="hyp2f1_grad", loop_op=Grad2F1Loop
    )
    return *loop_outs[: len(wrt)], converges


def hyp2f1_grad(a, b, c, z, wrt: tuple[int, ...]):
    dtype = upcast(a.type.dtype, b.type.dtype, c.type.dtype, z.type.dtype, "float32")

    def check_2f1_converges(a, b, c, z):
        def is_nonpositive_integer(x):
            if x.type.dtype not in integer_types:
                return eq(floor(x), x) & (x <= 0)
            else:
                return x <= 0

        a_is_polynomial = is_nonpositive_integer(a) & (scalar_abs(a) >= 0)
        num_terms = switch(
            a_is_polynomial,
            floor(scalar_abs(a)).astype("int64"),
            0,
        )

        b_is_polynomial = is_nonpositive_integer(b) & (scalar_abs(b) >= num_terms)
        num_terms = switch(
            b_is_polynomial,
            floor(scalar_abs(b)).astype("int64"),
            num_terms,
        )

        is_undefined = is_nonpositive_integer(c) & (scalar_abs(c) <= num_terms)
        is_polynomial = a_is_polynomial | b_is_polynomial

        return (~is_undefined) & (
            is_polynomial | (scalar_abs(z) < 1) | (eq(scalar_abs(z), 1) & (c > (a + b)))
        )

    # We have to pass the converges flag to interrupt the loop, as the switch is not lazy
    z_is_zero = eq(z, 0)
    converges = check_2f1_converges(a, b, c, z)
    *grads, _grad_converges = _grad_2f1_loop(
        a, b, c, z, skip_loop=z_is_zero | (~converges), wrt=wrt, dtype=dtype
    )

    return [
        switch(
            z_is_zero,
            0,
            switch(
                converges,
                grad,
                np.nan,
            ),
        )
        for grad in grads
    ]
