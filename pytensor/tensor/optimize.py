from copy import copy

from scipy.optimize import minimize as scipy_minimize

from pytensor import function, graph_replace
from pytensor.gradient import grad
from pytensor.graph import Apply, Constant, FunctionGraph
from pytensor.graph.basic import truncated_graph_inputs
from pytensor.graph.op import ComputeMapType, HasInnerGraph, Op, StorageMapType
from pytensor.scalar import bool as scalar_bool


class MinimizeOp(Op, HasInnerGraph):
    def __init__(
        self,
        x,
        *args,
        output,
        method="BFGS",
        jac=True,
        hess=False,
        hessp=False,
        options: dict | None = None,
        debug: bool = False,
    ):
        self.fgraph = FunctionGraph([x, *args], [output])

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

    def build_fn(self):
        outputs = self.inner_outputs
        if len(outputs) == 1:
            outputs = outputs[0]
        self._fn = fn = function(self.inner_inputs, outputs)
        self.fgraph = (
            fn.maker.fgraph
        )  # So we see the compiled graph ater the first call

        if self.inner_inputs[0].type.shape == ():
            # Work-around for scipy changing the type of x
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
        # TODO: Implemet this method

    def make_node(self, *inputs):
        assert len(inputs) == len(self.inner_inputs)

        return Apply(
            self, inputs, [self.inner_outputs[0].type(), scalar_bool("success")]
        )

    def perform(self, node, inputs, outputs):
        f = self.fn_wrapped
        x0, *args = inputs

        # print(f(*inputs))

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

        # TODO: Does clone replace do what we want? It might need a merge optimization pass afterwards
        replace = dict(zip(self.fgraph.inputs, (x_star, *args), strict=True))
        grad_f_wrt_x_star, *grad_f_wrt_args = graph_replace(
            inner_grads, replace=replace
        )

        # # TODO: If scipy optimizer uses hessian (or hessp), just store it from the inner function
        # inner_hess = jacobian(inner_fx, inner_args)
        # hess_f_x = clone_replace(inner_hess, replace=replace)

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
        x, *args, output=objective, method=method, jac=jac, debug=debug, options=options
    )
    return minimize_op(x, *args)


__all__ = ["minimize"]
