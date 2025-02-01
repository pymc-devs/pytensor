from collections.abc import Sequence
from copy import copy

from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import root as scipy_root

from pytensor import Variable, function, graph_replace
from pytensor.gradient import grad, jacobian
from pytensor.graph import Apply, Constant, FunctionGraph
from pytensor.graph.basic import truncated_graph_inputs
from pytensor.graph.op import ComputeMapType, HasInnerGraph, Op, StorageMapType
from pytensor.scalar import bool as scalar_bool
from pytensor.tensor.basic import atleast_2d
from pytensor.tensor.slinalg import solve
from pytensor.tensor.variable import TensorVariable


class ScipyWrapperOp(Op, HasInnerGraph):
    """Shared logic for scipy optimization ops"""

    __props__ = ("method", "debug")

    def build_fn(self):
        """
        This is overloaded because scipy converts scalar inputs to lists, changing the return type. The
        wrapper function logic is there to handle this.
        """
        # TODO: Introduce rewrites to change MinimizeOp to MinimizeScalarOp and RootOp to RootScalarOp
        #  when x is scalar. That will remove the need for the wrapper.

        outputs = self.inner_outputs
        if len(outputs) == 1:
            outputs = outputs[0]
        self._fn = fn = function(self.inner_inputs, outputs)

        # Do this reassignment to see the compiled graph in the dprint
        self.fgraph = fn.maker.fgraph

        if self.inner_inputs[0].type.shape == ():

            def fn_wrapper(x, *args):
                return fn(x.squeeze(), *args)

            self._fn_wrapped = fn_wrapper
        else:
            self._fn_wrapped = fn

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

        return Apply(
            self, inputs, [self.inner_inputs[0].type(), scalar_bool("success")]
        )


class MinimizeOp(ScipyWrapperOp):
    __props__ = ("method", "jac", "hess", "hessp", "debug")

    def __init__(
        self,
        x,
        *args,
        objective,
        method="BFGS",
        jac=True,
        hess=False,
        hessp=False,
        options: dict | None = None,
        debug: bool = False,
    ):
        self.fgraph = FunctionGraph([x, *args], [objective])

        if jac:
            grad_wrt_x = grad(self.fgraph.outputs[0], self.fgraph.inputs[0])
            self.fgraph.add_output(grad_wrt_x)

        self.jac = jac
        self.hess = hess
        self.hessp = hessp

        self.method = method
        self.options = options if options is not None else {}
        self.debug = debug
        self._fn = None
        self._fn_wrapped = None

    def perform(self, node, inputs, outputs):
        f = self.fn_wrapped
        x0, *args = inputs

        res = scipy_minimize(
            fun=f,
            jac=self.jac,
            x0=x0,
            args=tuple(args),
            method=self.method,
            **self.options,
        )

        if self.debug:
            print(res)

        outputs[0][0] = res.x
        outputs[1][0] = res.success

    def L_op(self, inputs, outputs, output_grads):
        x, *args = inputs
        x_star, success = outputs
        output_grad, _ = output_grads

        inner_x, *inner_args = self.fgraph.inputs
        inner_fx = self.fgraph.outputs[0]

        inner_grads = grad(inner_fx, [inner_x, *inner_args])

        replace = dict(zip(self.fgraph.inputs, (x_star, *args), strict=True))

        grad_f_wrt_x_star, *grad_f_wrt_args = graph_replace(
            inner_grads, replace=replace
        )

        grad_wrt_args = [
            -grad_f_wrt_arg / grad_f_wrt_x_star * output_grad
            for grad_f_wrt_arg in grad_f_wrt_args
        ]

        return [x.zeros_like(), *grad_wrt_args]


def minimize(
    objective,
    x,
    method: str = "BFGS",
    jac: bool = True,
    debug: bool = False,
    options: dict | None = None,
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

    debug : bool, optional
        If True, prints raw scipy result after optimization. Default is False.

    **optimizer_kwargs
        Additional keyword arguments to pass to scipy.optimize.minimize

    Returns
    -------
    TensorVariable
        The optimized value of x that minimizes the objective function.

    """
    args = [
        arg
        for arg in truncated_graph_inputs([objective], [x])
        if (arg is not x and not isinstance(arg, Constant))
    ]

    minimize_op = MinimizeOp(
        x,
        *args,
        objective=objective,
        method=method,
        jac=jac,
        debug=debug,
        options=options,
    )

    return minimize_op(x, *args)


class RootOp(ScipyWrapperOp):
    __props__ = ("method", "jac", "debug")

    def __init__(
        self,
        variables,
        *args,
        equations,
        method="hybr",
        jac=True,
        options: dict | None = None,
        debug: bool = False,
    ):
        self.fgraph = FunctionGraph([variables, *args], [equations])

        if jac:
            jac_wrt_x = jacobian(self.fgraph.outputs[0], self.fgraph.inputs[0])
            self.fgraph.add_output(atleast_2d(jac_wrt_x))

        self.jac = jac

        self.method = method
        self.options = options if options is not None else {}
        self.debug = debug
        self._fn = None
        self._fn_wrapped = None

    def perform(self, node, inputs, outputs):
        f = self.fn_wrapped
        variables, *args = inputs

        res = scipy_root(
            fun=f,
            jac=self.jac,
            x0=variables,
            args=tuple(args),
            method=self.method,
            **self.options,
        )

        if self.debug:
            print(res)

        outputs[0][0] = res.x
        outputs[1][0] = res.success

    def L_op(
        self,
        inputs: Sequence[Variable],
        outputs: Sequence[Variable],
        output_grads: Sequence[Variable],
    ) -> list[Variable]:
        # TODO: Broken
        x, *args = inputs
        x_star, success = outputs
        output_grad, _ = output_grads

        inner_x, *inner_args = self.fgraph.inputs
        inner_fx = self.fgraph.outputs[0]

        inner_jac = jacobian(inner_fx, [inner_x, *inner_args])

        replace = dict(zip(self.fgraph.inputs, (x_star, *args), strict=True))
        jac_f_wrt_x_star, *jac_f_wrt_args = graph_replace(inner_jac, replace=replace)

        jac_wrt_args = solve(-jac_f_wrt_x_star, output_grad)

        return [x.zeros_like(), jac_wrt_args]


def root(
    equations: TensorVariable,
    variables: TensorVariable,
    method: str = "hybr",
    jac: bool = True,
    debug: bool = False,
):
    """Find roots of a system of equations using scipy.optimize.root."""

    args = [
        arg
        for arg in truncated_graph_inputs([equations], [variables])
        if (arg is not variables and not isinstance(arg, Constant))
    ]

    root_op = RootOp(
        variables, *args, equations=equations, method=method, jac=jac, debug=debug
    )

    return root_op(variables, *args)


__all__ = ["minimize", "root"]
