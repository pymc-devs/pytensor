import logging
from collections.abc import Sequence
from copy import copy
from typing import cast

import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import minimize_scalar as scipy_minimize_scalar
from scipy.optimize import root as scipy_root
from scipy.optimize import root_scalar as scipy_root_scalar

import pytensor.scalar as ps
from pytensor.compile.function import function
from pytensor.gradient import grad, hessian, jacobian
from pytensor.graph import Apply, Constant, FunctionGraph
from pytensor.graph.basic import ancestors, truncated_graph_inputs
from pytensor.graph.op import ComputeMapType, HasInnerGraph, Op, StorageMapType
from pytensor.graph.replace import graph_replace
from pytensor.tensor.basic import (
    atleast_2d,
    concatenate,
    tensor,
    tensor_from_scalar,
    zeros_like,
)
from pytensor.tensor.math import dot
from pytensor.tensor.slinalg import solve
from pytensor.tensor.variable import TensorVariable, Variable


_log = logging.getLogger(__name__)


class LRUCache1:
    """
    Simple LRU cache with a memory size of 1.

    This cache is only usable for a function that takes a single input `x` and returns a single output. The
    function can also take any number of additional arguments `*args`, but these are assumed to be constant
    between function calls.

    The purpose of this cache is to allow for Hessian computation to be reused when calling scipy.optimize functions.
    It is very often the case that some sub-computations are repeated between the objective, gradient, and hessian
    functions, but by default scipy only allows for the objective and gradient to be fused.

    By using this cache, all 3 functions can be fused, which can significantly speed up the optimization process for
    expensive functions.
    """

    def __init__(self, fn, copy_x: bool = False):
        self.fn = fn
        self.last_x = None
        self.last_result = None
        self.copy_x = copy_x

        # Scipy does not respect dtypes *at all*, so we have to force it ourselves.
        self.dtype = fn.maker.fgraph.inputs[0].type.dtype

        self.cache_hits = 0
        self.cache_misses = 0

        self.value_calls = 0
        self.grad_calls = 0
        self.value_and_grad_calls = 0
        self.hess_calls = 0

    def __call__(self, x, *args):
        """
        Call the cached function with the given input `x` and additional arguments `*args`.

        If the input `x` is the same as the last input, return the cached result. Otherwise update the cache with the
        new input and result.
        """
        x = x.astype(self.dtype)

        if self.last_result is None or not (x == self.last_x).all():
            self.cache_misses += 1

            # scipy.optimize.root changes x in place, so the cache has to copy it, otherwise we get false
            # cache hits and optimization always fails.
            if self.copy_x:
                x = x.copy()
            self.last_x = x

            result = self.fn(x, *args)
            self.last_result = result

            return result

        else:
            self.cache_hits += 1
            return self.last_result

    def value(self, x, *args):
        self.value_calls += 1
        return self(x, *args)[0]

    def grad(self, x, *args):
        self.grad_calls += 1
        return self(x, *args)[1]

    def value_and_grad(self, x, *args):
        self.value_and_grad_calls += 1
        return self(x, *args)[:2]

    def hess(self, x, *args):
        self.hess_calls += 1
        return self(x, *args)[-1]

    def report(self):
        _log.info(f"Value and Grad calls: {self.value_and_grad_calls}")
        _log.info(f"Hess Calls: {self.hess_calls}")
        _log.info(f"Hits: {self.cache_hits}")
        _log.info(f"Misses: {self.cache_misses}")

    def clear_cache(self):
        self.last_x = None
        self.last_result = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.value_calls = 0
        self.grad_calls = 0
        self.value_and_grad_calls = 0
        self.hess_calls = 0


def _find_optimization_parameters(objective: TensorVariable, x: TensorVariable):
    """
    Find the parameters of the optimization problem that are not the variable `x`.

    This is used to determine the additional arguments that need to be passed to the objective function.
    """
    return [
        arg
        for arg in truncated_graph_inputs([objective], [x])
        if (arg is not x and not isinstance(arg, Constant))
    ]


def _get_parameter_grads_from_vector(
    grad_wrt_args_vector: Variable,
    x_star: Variable,
    args: Sequence[Variable],
    output_grad: Variable,
):
    """
    Given a single concatenated vector of objective function gradients with respect to raveled optimization parameters,
    returns the contribution of each parameter to the total loss function, with the unraveled shape of the parameter.
    """
    grad_wrt_args_vector = cast(TensorVariable, grad_wrt_args_vector)
    x_star = cast(TensorVariable, x_star)

    cursor = 0
    grad_wrt_args = []

    for arg in args:
        arg = cast(TensorVariable, arg)
        arg_shape = arg.shape
        arg_size = arg_shape.prod()
        arg_grad = grad_wrt_args_vector[:, cursor : cursor + arg_size].reshape(
            (*x_star.shape, *arg_shape)
        )

        grad_wrt_args.append(dot(output_grad, arg_grad))

        cursor += arg_size

    return grad_wrt_args


class ScipyWrapperOp(Op, HasInnerGraph):
    """Shared logic for scipy optimization ops"""

    def build_fn(self):
        """
        This is overloaded because scipy converts scalar inputs to lists, changing the return type. The
        wrapper function logic is there to handle this.
        """
        outputs = self.inner_outputs
        self._fn = fn = function(self.inner_inputs, outputs, trust_input=True)

        # Do this reassignment to see the compiled graph in the dprint
        # self.fgraph = fn.maker.fgraph

        self._fn_wrapped = LRUCache1(fn)

    @property
    def fn(self):
        if self._fn is None:
            self.build_fn()
        return self._fn

    @property
    def fn_wrapped(self):
        if self._fn_wrapped is None:
            self.build_fn()
        return self._fn_wrapped

    @property
    def inner_inputs(self):
        return self.fgraph.inputs

    @property
    def inner_outputs(self):
        return self.fgraph.outputs

    def clone(self):
        copy_op = copy(self)
        copy_op.fgraph = self.fgraph.clone()
        return copy_op

    def prepare_node(
        self,
        node: Apply,
        storage_map: StorageMapType | None,
        compute_map: ComputeMapType | None,
        impl: str | None,
    ):
        """Trigger the compilation of the inner fgraph so it shows in the dprint before the first call"""
        self.build_fn()

    def make_node(self, *inputs):
        assert len(inputs) == len(self.inner_inputs)
        for input, inner_input in zip(inputs, self.inner_inputs):
            assert (
                input.type == inner_input.type
            ), f"Input {input} does not match expected type {inner_input.type}"

        return Apply(self, inputs, [self.inner_inputs[0].type(), ps.bool("success")])


class ScipyScalarWrapperOp(ScipyWrapperOp):
    def build_fn(self):
        """
        This is overloaded because scipy converts scalar inputs to lists, changing the return type. The
        wrapper function logic is there to handle this.
        """

        # We have no control over the inputs to the scipy inner function for scalar_minimize. As a result,
        # we need to adjust the graph to work with what scipy will be passing into the inner function --
        # always scalar, and always float64
        x, *args = self.inner_inputs
        new_root_x = ps.float64(name="x_scalar")
        new_x = tensor_from_scalar(new_root_x.astype(x.type.dtype))

        new_outputs = graph_replace(self.inner_outputs, {x: new_x})

        self._fn = fn = function([new_root_x, *args], new_outputs, trust_input=True)

        # Do this reassignment to see the compiled graph in the dprint
        # self.fgraph = fn.maker.fgraph

        self._fn_wrapped = LRUCache1(fn)


def scalar_implict_optimization_grads(
    inner_fx: Variable,
    inner_x: Variable,
    inner_args: Sequence[Variable],
    args: Sequence[Variable],
    x_star: Variable,
    output_grad: Variable,
    fgraph: FunctionGraph,
) -> list[Variable]:
    df_dx, *df_dthetas = cast(
        list[Variable],
        grad(inner_fx, [inner_x, *inner_args], disconnected_inputs="ignore"),
    )

    replace = dict(zip(fgraph.inputs, (x_star, *args), strict=True))
    df_dx_star, *df_dthetas_stars = graph_replace([df_dx, *df_dthetas], replace=replace)

    grad_wrt_args = [
        (-df_dtheta_star / df_dx_star) * output_grad
        for df_dtheta_star in cast(list[TensorVariable], df_dthetas_stars)
    ]

    return grad_wrt_args


def implict_optimization_grads(
    df_dx: Variable,
    df_dtheta_columns: Sequence[Variable],
    args: Sequence[Variable],
    x_star: Variable,
    output_grad: Variable,
    fgraph: FunctionGraph,
):
    r"""
    Compute gradients of an optimization problem with respect to its parameters.

    The gradents are computed using the implicit function theorem. Given a fuction `f(x, theta) =`, and a function
    `x_star(theta)` that, given input parameters theta returns `x` such that `f(x_star(theta), theta) = 0`, we can
    compute the gradients of `x_star` with respect to `theta` as follows:

    .. math::

        \underbrace{\frac{\partial f}{\partial x}\left(x^*(\theta), \theta\right)}_{\text{Jacobian wrt } x}
        \frac{d x^*(\theta)}{d \theta}
        +
        \underbrace{\frac{\partial f}{\partial \theta}\left(x^*(\theta), \theta\right)}_{\text{Jacobian wrt } \theta}
        = 0

    Which, after rearranging, gives us:

    .. math::

        \frac{d x^*(\theta)}{d \theta} = - \left(\frac{\partial f}{\partial x}\left(x^*(\theta), \theta\right)\right)^{-1} \frac{\partial f}{\partial \theta}\left(x^*(\theta), \theta\right)

    Note that this method assumes `f(x_star(theta), theta) = 0`; so it is not immediately applicable to a minimization
    problem, where `f` is the objective function. In this case, we instead take `f` to be the gradient of the objective
    function, which *is* indeed zero at the minimum.

    Parameters
    ----------
    df_dx : Variable
        The Jacobian of the objective function with respect to the variable `x`.
    df_dtheta_columns : Sequence[Variable]
        The Jacobians of the objective function with respect to the optimization parameters `theta`.
        Each column (or columns) corresponds to a different parameter. Should be returned by pytensor.gradient.jacobian.
    args : Sequence[Variable]
        The optimization parameters `theta`.
    x_star : Variable
        Symbolic graph representing the value of the variable `x` such that `f(x_star, theta) = 0 `.
    output_grad : Variable
        The gradient of the output with respect to the objective function.
    fgraph : FunctionGraph
        The function graph that contains the inputs and outputs of the optimization problem.
    """
    df_dx = cast(TensorVariable, df_dx)

    df_dtheta = concatenate(
        [
            atleast_2d(jac_col, left=False)
            for jac_col in cast(list[TensorVariable], df_dtheta_columns)
        ],
        axis=-1,
    )

    replace = dict(zip(fgraph.inputs, (x_star, *args), strict=True))

    df_dx_star, df_dtheta_star = cast(
        list[TensorVariable],
        graph_replace([atleast_2d(df_dx), df_dtheta], replace=replace),
    )

    grad_wrt_args_vector = solve(-df_dx_star, df_dtheta_star)
    grad_wrt_args = _get_parameter_grads_from_vector(
        grad_wrt_args_vector, x_star, args, output_grad
    )

    return grad_wrt_args


class MinimizeScalarOp(ScipyScalarWrapperOp):
    __props__ = ("method",)

    def __init__(
        self,
        x: Variable,
        *args: Variable,
        objective: Variable,
        method: str = "brent",
        optimizer_kwargs: dict | None = None,
    ):
        if not cast(TensorVariable, x).ndim == 0:
            raise ValueError(
                "The variable `x` must be a scalar (0-dimensional) tensor for minimize_scalar."
            )
        if not cast(TensorVariable, objective).ndim == 0:
            raise ValueError(
                "The objective function must be a scalar (0-dimensional) tensor for minimize_scalar."
            )
        self.fgraph = FunctionGraph([x, *args], [objective])

        self.method = method
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self._fn = None
        self._fn_wrapped = None

    def perform(self, node, inputs, outputs):
        f = self.fn_wrapped
        f.clear_cache()

        # minimize_scalar doesn't take x0 as an argument. The Op still needs this input (to symbolically determine
        # the args of the objective function), but it is not used in the optimization.
        x0, *args = inputs

        res = scipy_minimize_scalar(
            fun=f.value,
            args=tuple(args),
            method=self.method,
            **self.optimizer_kwargs,
        )

        outputs[0][0] = np.array(res.x, dtype=x0.dtype)
        outputs[1][0] = np.bool_(res.success)

    def L_op(self, inputs, outputs, output_grads):
        x, *args = inputs
        x_star, _ = outputs
        output_grad, _ = output_grads

        inner_x, *inner_args = self.fgraph.inputs
        inner_fx = self.fgraph.outputs[0]

        implicit_f = grad(inner_fx, inner_x)

        grad_wrt_args = scalar_implict_optimization_grads(
            inner_fx=implicit_f,
            inner_x=inner_x,
            inner_args=inner_args,
            args=args,
            x_star=x_star,
            output_grad=output_grad,
            fgraph=self.fgraph,
        )

        return [zeros_like(x), *grad_wrt_args]


def minimize_scalar(
    objective: TensorVariable,
    x: TensorVariable,
    method: str = "brent",
    optimizer_kwargs: dict | None = None,
):
    """
    Minimize a scalar objective function using scipy.optimize.minimize_scalar.
    """

    args = _find_optimization_parameters(objective, x)

    minimize_scalar_op = MinimizeScalarOp(
        x,
        *args,
        objective=objective,
        method=method,
        optimizer_kwargs=optimizer_kwargs,
    )

    return minimize_scalar_op(x, *args)


class MinimizeOp(ScipyWrapperOp):
    __props__ = ("method", "jac", "hess", "hessp")

    def __init__(
        self,
        x: Variable,
        *args: Variable,
        objective: Variable,
        method: str = "BFGS",
        jac: bool = True,
        hess: bool = False,
        hessp: bool = False,
        optimizer_kwargs: dict | None = None,
    ):
        if not cast(TensorVariable, objective).ndim == 0:
            raise ValueError(
                "The objective function must be a scalar (0-dimensional) tensor for minimize."
            )
        if x not in ancestors([objective]):
            raise ValueError(
                "The variable `x` must be an input to the computational graph of the objective function."
            )

        self.fgraph = FunctionGraph([x, *args], [objective])

        if jac:
            grad_wrt_x = cast(
                Variable, grad(self.fgraph.outputs[0], self.fgraph.inputs[0])
            )
            self.fgraph.add_output(grad_wrt_x)

        if hess:
            hess_wrt_x = cast(
                Variable, hessian(self.fgraph.outputs[0], self.fgraph.inputs[0])
            )
            self.fgraph.add_output(hess_wrt_x)

        self.jac = jac
        self.hess = hess
        self.hessp = hessp

        self.method = method
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self._fn = None
        self._fn_wrapped = None

    def perform(self, node, inputs, outputs):
        f = self.fn_wrapped
        x0, *args = inputs

        res = scipy_minimize(
            fun=f.value_and_grad if self.jac else f.value,
            jac=self.jac,
            x0=x0,
            args=tuple(args),
            hess=f.hess if self.hess else None,
            method=self.method,
            **self.optimizer_kwargs,
        )

        f.clear_cache()

        outputs[0][0] = res.x.reshape(x0.shape).astype(x0.dtype)
        outputs[1][0] = np.bool_(res.success)

    def L_op(self, inputs, outputs, output_grads):
        x, *args = inputs
        x_star, success = outputs
        output_grad, _ = output_grads

        inner_x, *inner_args = self.fgraph.inputs
        inner_fx = self.fgraph.outputs[0]

        implicit_f = grad(inner_fx, inner_x)

        df_dx, *df_dtheta_columns = jacobian(
            implicit_f, [inner_x, *inner_args], disconnected_inputs="ignore"
        )
        grad_wrt_args = implict_optimization_grads(
            df_dx=df_dx,
            df_dtheta_columns=df_dtheta_columns,
            args=args,
            x_star=x_star,
            output_grad=output_grad,
            fgraph=self.fgraph,
        )

        return [zeros_like(x), *grad_wrt_args]


def minimize(
    objective: TensorVariable,
    x: TensorVariable,
    method: str = "BFGS",
    jac: bool = True,
    hess: bool = False,
    optimizer_kwargs: dict | None = None,
):
    """
    Minimize a scalar objective function using scipy.optimize.minimize.

    Parameters
    ----------
    objective : TensorVariable
        The objective function to minimize. This should be a pytensor variable representing a scalar value.

    x : TensorVariable
        The variable with respect to which the objective function is minimized. It must be an input to the
        computational graph of `objective`.

    method : str, optional
        The optimization method to use. Default is "BFGS". See scipy.optimize.minimize for other options.

    jac : bool, optional
        Whether to compute and use the gradient of teh objective function with respect to x for optimization.
        Default is True.

    optimizer_kwargs
        Additional keyword arguments to pass to scipy.optimize.minimize

    Returns
    -------
    TensorVariable
        The optimized value of x that minimizes the objective function.

    """
    args = _find_optimization_parameters(objective, x)

    minimize_op = MinimizeOp(
        x,
        *args,
        objective=objective,
        method=method,
        jac=jac,
        hess=hess,
        optimizer_kwargs=optimizer_kwargs,
    )

    return minimize_op(x, *args)


class RootScalarOp(ScipyScalarWrapperOp):
    __props__ = ("method", "jac", "hess")

    def __init__(
        self,
        variables,
        *args,
        equation,
        method,
        jac: bool = False,
        hess: bool = False,
        optimizer_kwargs=None,
    ):
        if not equation.ndim == 0:
            raise ValueError(
                "The equation must be a scalar (0-dimensional) tensor for root_scalar."
            )
        if not isinstance(variables, Variable) or variables not in ancestors(
            [equation]
        ):
            raise ValueError(
                "The variable `variables` must be an input to the computational graph of the equation."
            )

        self.fgraph = FunctionGraph([variables, *args], [equation])

        if jac:
            f_prime = cast(
                Variable, grad(self.fgraph.outputs[0], self.fgraph.inputs[0])
            )
            self.fgraph.add_output(f_prime)

        if hess:
            if not jac:
                raise ValueError(
                    "Cannot set `hess=True` without `jac=True`. No methods use second derivatives without also"
                    " using first derivatives."
                )
            f_double_prime = cast(
                Variable, grad(self.fgraph.outputs[-1], self.fgraph.inputs[0])
            )
            self.fgraph.add_output(f_double_prime)

        self.method = method
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self.jac = jac
        self.hess = hess

        self._fn = None
        self._fn_wrapped = None

    def perform(self, node, inputs, outputs):
        f = self.fn_wrapped
        f.clear_cache()
        # f.copy_x = True

        variables, *args = inputs

        res = scipy_root_scalar(
            f=f.value,
            fprime=f.grad if self.jac else None,
            fprime2=f.hess if self.hess else None,
            x0=variables,
            args=tuple(args),
            method=self.method,
            **self.optimizer_kwargs,
        )

        outputs[0][0] = np.array(res.root)
        outputs[1][0] = np.bool_(res.converged)

    def L_op(self, inputs, outputs, output_grads):
        x, *args = inputs
        x_star, _ = outputs
        output_grad, _ = output_grads

        inner_x, *inner_args = self.fgraph.inputs
        inner_fx = self.fgraph.outputs[0]

        grad_wrt_args = scalar_implict_optimization_grads(
            inner_fx=inner_fx,
            inner_x=inner_x,
            inner_args=inner_args,
            args=args,
            x_star=x_star,
            output_grad=output_grad,
            fgraph=self.fgraph,
        )

        return [zeros_like(x), *grad_wrt_args]


def root_scalar(
    equation: TensorVariable,
    variables: TensorVariable,
    method: str = "secant",
    jac: bool = False,
    hess: bool = False,
    optimizer_kwargs: dict | None = None,
):
    """
    Find roots of a scalar equation using scipy.optimize.root_scalar.
    """
    args = _find_optimization_parameters(equation, variables)

    root_scalar_op = RootScalarOp(
        variables,
        *args,
        equation=equation,
        method=method,
        jac=jac,
        hess=hess,
        optimizer_kwargs=optimizer_kwargs,
    )

    return root_scalar_op(variables, *args)


class RootOp(ScipyWrapperOp):
    __props__ = ("method", "jac")

    def __init__(
        self,
        variables: Variable,
        *args: Variable,
        equations: Variable,
        method: str = "hybr",
        jac: bool = True,
        optimizer_kwargs: dict | None = None,
    ):
        if cast(TensorVariable, variables).ndim != cast(TensorVariable, equations).ndim:
            raise ValueError(
                "The variable `variables` must have the same number of dimensions as the equations."
            )
        if variables not in ancestors([equations]):
            raise ValueError(
                "The variable `variables` must be an input to the computational graph of the equations."
            )

        self.fgraph = FunctionGraph([variables, *args], [equations])

        if jac:
            jac_wrt_x = jacobian(self.fgraph.outputs[0], self.fgraph.inputs[0])
            self.fgraph.add_output(atleast_2d(jac_wrt_x))

        self.jac = jac

        self.method = method
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self._fn = None
        self._fn_wrapped = None

    def build_fn(self):
        outputs = self.inner_outputs
        variables, *args = self.inner_inputs

        if variables.ndim > 0:
            new_root_variables = variables
            new_outputs = outputs
        else:
            # If the user passes a scalar optimization problem to root, scipy will automatically upcast it to
            # a 1d array. The inner function needs to be adjusted to handle this.
            new_root_variables = tensor(
                name="variables_vector", shape=(1,), dtype=variables.type.dtype
            )
            new_variables = new_root_variables.squeeze()

            new_outputs = graph_replace(outputs, {variables: new_variables})

        self._fn = fn = function(
            [new_root_variables, *args], new_outputs, trust_input=True
        )

        # Do this reassignment to see the compiled graph in the dprint
        # self.fgraph = fn.maker.fgraph

        self._fn_wrapped = LRUCache1(fn)

    def perform(self, node, inputs, outputs):
        f = self.fn_wrapped
        f.clear_cache()
        f.copy_x = True

        variables, *args = inputs

        res = scipy_root(
            fun=f,
            jac=self.jac,
            x0=variables,
            args=tuple(args),
            method=self.method,
            **self.optimizer_kwargs,
        )

        # There's a reshape here to cover the case where variables is a scalar. Scipy will still return a
        # (1, 1) matrix in in this case, which causes errors downstream (since pytensor expects a scalar).
        outputs[0][0] = res.x.reshape(variables.shape).astype(variables.dtype)
        outputs[1][0] = np.bool_(res.success)

    def L_op(
        self,
        inputs: Sequence[Variable],
        outputs: Sequence[Variable],
        output_grads: Sequence[Variable],
    ) -> list[Variable]:
        x, *args = inputs
        x_star, _ = outputs
        output_grad, _ = output_grads

        inner_x, *inner_args = self.fgraph.inputs
        inner_fx = self.fgraph.outputs[0]

        df_dx = jacobian(inner_fx, inner_x) if not self.jac else self.fgraph.outputs[1]
        df_dtheta_columns = jacobian(inner_fx, inner_args, disconnected_inputs="ignore")

        grad_wrt_args = implict_optimization_grads(
            df_dx=df_dx,
            df_dtheta_columns=df_dtheta_columns,
            args=args,
            x_star=x_star,
            output_grad=output_grad,
            fgraph=self.fgraph,
        )

        return [zeros_like(x), *grad_wrt_args]


def root(
    equations: TensorVariable,
    variables: TensorVariable,
    method: str = "hybr",
    jac: bool = True,
    optimizer_kwargs: dict | None = None,
):
    """Find roots of a system of equations using scipy.optimize.root."""

    args = _find_optimization_parameters(equations, variables)

    root_op = RootOp(
        variables,
        *args,
        equations=equations,
        method=method,
        jac=jac,
        optimizer_kwargs=optimizer_kwargs,
    )

    return root_op(variables, *args)


__all__ = ["minimize_scalar", "minimize", "root_scalar", "root"]
