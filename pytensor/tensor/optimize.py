import logging
from collections.abc import Sequence
from copy import copy
from typing import cast

import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import minimize_scalar as scipy_minimize_scalar
from scipy.optimize import root as scipy_root

from pytensor import Variable, function, graph_replace
from pytensor.gradient import grad, hessian, jacobian
from pytensor.graph import Apply, Constant, FunctionGraph
from pytensor.graph.basic import truncated_graph_inputs
from pytensor.graph.op import ComputeMapType, HasInnerGraph, Op, StorageMapType
from pytensor.scalar import bool as scalar_bool
from pytensor.tensor import dot
from pytensor.tensor.basic import atleast_2d, concatenate, zeros_like
from pytensor.tensor.slinalg import solve
from pytensor.tensor.variable import TensorVariable


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
        # scipy.optimize.scalar_minimize and scalar_root don't take initial values as an argument, so we can't control
        # the first input to the inner function. Of course, they use a scalar, but we need a 0d numpy array.
        x = np.asarray(x)

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

        if self.inner_inputs[0].type.shape == ():

            def fn_wrapper(x, *args):
                return fn(x.squeeze(), *args)

            self._fn_wrapped = LRUCache1(fn_wrapper)
        else:
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

        return Apply(
            self, inputs, [self.inner_inputs[0].type(), scalar_bool("success")]
        )


class MinimizeScalarOp(ScipyWrapperOp):
    __props__ = ("method",)

    def __init__(
        self,
        x: Variable,
        *args: Variable,
        objective: Variable,
        method: str = "brent",
        optimizer_kwargs: dict | None = None,
    ):
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
        _, *args = inputs

        res = scipy_minimize_scalar(
            fun=f.value,
            args=tuple(args),
            method=self.method,
            **self.optimizer_kwargs,
        )

        outputs[0][0] = np.array(res.x)
        outputs[1][0] = np.bool_(res.success)

    def L_op(self, inputs, outputs, output_grads):
        x, *args = inputs
        x_star, _ = outputs
        output_grad, _ = output_grads

        inner_x, *inner_args = self.fgraph.inputs
        inner_fx = self.fgraph.outputs[0]

        implicit_f = grad(inner_fx, inner_x)
        df_dx, *df_dthetas = grad(
            implicit_f, [inner_x, *inner_args], disconnected_inputs="ignore"
        )

        replace = dict(zip(self.fgraph.inputs, (x_star, *args), strict=True))
        df_dx_star, *df_dthetas_stars = graph_replace(
            [df_dx, *df_dthetas], replace=replace
        )

        grad_wrt_args = [
            (-df_dtheta_star / df_dx_star) * output_grad
            for df_dtheta_star in df_dthetas_stars
        ]

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

        outputs[0][0] = res.x
        outputs[1][0] = np.bool_(res.success)

    def L_op(self, inputs, outputs, output_grads):
        x, *args = inputs
        x_star, success = outputs
        output_grad, _ = output_grads

        inner_x, *inner_args = self.fgraph.inputs
        inner_fx = self.fgraph.outputs[0]

        implicit_f = grad(inner_fx, inner_x)

        df_dx = atleast_2d(concatenate(jacobian(implicit_f, [inner_x]), axis=-1))

        df_dtheta = concatenate(
            [
                atleast_2d(x, left=False)
                for x in jacobian(implicit_f, inner_args, disconnected_inputs="ignore")
            ],
            axis=-1,
        )

        replace = dict(zip(self.fgraph.inputs, (x_star, *args), strict=True))

        df_dx_star, df_dtheta_star = graph_replace([df_dx, df_dtheta], replace=replace)

        grad_wrt_args_vector = solve(-df_dx_star, df_dtheta_star)

        cursor = 0
        grad_wrt_args = []

        for arg in args:
            arg_shape = arg.shape
            arg_size = arg_shape.prod()
            arg_grad = grad_wrt_args_vector[:, cursor : cursor + arg_size].reshape(
                (*x_star.shape, *arg_shape)
            )

            grad_wrt_args.append(dot(output_grad, arg_grad))
            cursor += arg_size

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
        self.fgraph = FunctionGraph([variables, *args], [equations])

        if jac:
            jac_wrt_x = jacobian(self.fgraph.outputs[0], self.fgraph.inputs[0])
            self.fgraph.add_output(atleast_2d(jac_wrt_x))

        self.jac = jac

        self.method = method
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self._fn = None
        self._fn_wrapped = None

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

        outputs[0][0] = res.x.reshape(variables.shape)
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

        df_dtheta = concatenate(
            [
                atleast_2d(jac_column, left=False)
                for jac_column in jacobian(
                    inner_fx, inner_args, disconnected_inputs="ignore"
                )
            ],
            axis=-1,
        )

        replace = dict(zip(self.fgraph.inputs, (x_star, *args), strict=True))
        df_dx_star, df_dtheta_star = graph_replace([df_dx, df_dtheta], replace=replace)

        grad_wrt_args_vector = solve(-df_dx_star, df_dtheta_star)

        cursor = 0
        grad_wrt_args = []

        for arg in args:
            arg_shape = arg.shape
            arg_size = arg_shape.prod()
            arg_grad = grad_wrt_args_vector[:, cursor : cursor + arg_size].reshape(
                (*x_star.shape, *arg_shape)
            )

            grad_wrt_args.append(dot(output_grad, arg_grad))
            cursor += arg_size

        return [zeros_like(x), *grad_wrt_args]


def root(
    equations: TensorVariable,
    variables: TensorVariable,
    method: str = "hybr",
    jac: bool = True,
):
    """Find roots of a system of equations using scipy.optimize.root."""

    args = [
        arg
        for arg in truncated_graph_inputs([equations], [variables])
        if (arg is not variables and not isinstance(arg, Constant))
    ]

    root_op = RootOp(variables, *args, equations=equations, method=method, jac=jac)

    return root_op(variables, *args)


__all__ = ["minimize", "root"]
