import logging
from collections.abc import Sequence
from copy import copy

import numpy as np

import pytensor.scalar as ps
from pytensor.compile.function import function
from pytensor.gradient import DisconnectedType, grad, jacobian
from pytensor.graph.basic import Apply, Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.null_type import NullType
from pytensor.graph.op import (
    ComputeMapType,
    HasInnerGraph,
    Op,
    StorageMapType,
    io_connection_pattern,
)
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import (
    ancestors,
    truncated_graph_inputs,
)
from pytensor.scalar import ScalarType, ScalarVariable
from pytensor.tensor import as_tensor_variable
from pytensor.tensor.basic import (
    atleast_2d,
    scalar_from_tensor,
    tensor,
    tensor_from_scalar,
    zeros_like,
)
from pytensor.tensor.math import tensordot
from pytensor.tensor.reshape import pack, unpack
from pytensor.tensor.slinalg import solve
from pytensor.tensor.variable import TensorVariable, Variable


# scipy.optimize can be slow to import, and will not be used by most users
# We import scipy.optimize lazily inside optimization perform methods to avoid this.
optimize = None


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


def _find_optimization_parameters(
    objective: TensorVariable, x: TensorVariable
) -> list[Variable]:
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
        copy_op.fgraph = self.fgraph.clone(clone_inner_graphs=True)
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
            assert input.type == inner_input.type, (
                f"Input {input} does not match expected type {inner_input.type}"
            )

        return Apply(self, inputs, [self.inner_inputs[0].type(), ps.bool("success")])

    def connection_pattern(self, node=None):
        """
        All Ops that inherit from ScipyWrapperOp share the same connection pattern logic, because they all share the
        same output structure. There are two outputs: the optimized variable, and a success flag. The success flag is
        not differentiable, so it is never connected. The optimized variable is connected only to inputs that are
        both connected to the objective function and of float dtype.
        """
        fgraph = self.fgraph
        fx = fgraph.outputs[0]
        return [
            # Every input is disonnected to the second output (success)
            # And may or not be connected to the first output (opt_x)
            [connected, False]
            for [connected] in io_connection_pattern(fgraph.inputs, [fx])
        ]


class ScipyScalarWrapperOp(ScipyWrapperOp):
    def build_fn(self):
        # We need to adjust the graph to work with what scipy will be passing into the inner function --
        # always scalar array of float64 type
        x, *args = self.inner_inputs
        new_root_x = ps.float64(name="x_scalar")
        new_x = tensor_from_scalar(new_root_x.astype(x.type.dtype))

        new_outputs = graph_replace(self.inner_outputs, {x: new_x})

        self._fn = fn = function([new_root_x, *args], new_outputs, trust_input=True)

        # Do this reassignment to see the compiled graph in the dprint
        # self.fgraph = fn.maker.fgraph

        self._fn_wrapped = LRUCache1(fn)

    def compute_implicit_gradients(
        self,
        x_star: TensorVariable,
        args: Sequence[TensorVariable | ScalarVariable],
        output_grad: TensorVariable,
        is_minimization: bool,
    ):
        """
        Compute gradients of a scalar optimization problem with respect to its parameters.

        For details, see the docstring of ScipyVectorWrapperOp.compute_implicit_gradients.

        Parameters
        ----------
        x_star : TensorVariable
            The symbolic solution of the optimization problem.
        args : Sequence of TensorVariable or ScalarVariable
            The parameters of the optimization problem.
        output_grad : TensorVariable
            The gradient of the output of the optimization Op with respect to some scalar loss.
        is_minimization : bool
            Whether the optimization problem is a minimization problem. If False, it is assumed to be a root-finding
            problem.
        """
        fgraph = self.fgraph
        inner_x, *inner_args = self.inner_inputs
        inner_fx = self.inner_outputs[0]

        if is_minimization:
            inner_fx = grad(inner_fx, inner_x)

        df_dx, *arg_grads = grad(
            inner_fx,
            [inner_x, *inner_args],
            disconnected_inputs="ignore",
            null_gradients="return",
            return_disconnected="disconnected",
        )

        outer_arg_grad_map = dict(zip(args, arg_grads))
        valid_args_and_grads = [
            (arg, g)
            for arg, g in outer_arg_grad_map.items()
            if not isinstance(g.type, DisconnectedType | NullType)
        ]

        if len(valid_args_and_grads) == 0:
            # No differentiable arguments, return disconnected gradients
            return arg_grads

        outer_args_to_diff, df_dthetas = zip(*valid_args_and_grads)

        replace = dict(zip(fgraph.inputs, (x_star, *args), strict=True))
        df_dx_star, *df_dthetas_stars = graph_replace(
            [df_dx, *df_dthetas], replace=replace
        )

        arg_to_grad = dict(zip(outer_args_to_diff, df_dthetas_stars))

        grad_wrt_args = [
            (-arg_to_grad[arg] / df_dx_star) * output_grad
            if arg in arg_to_grad
            else outer_arg_grad_map[arg]
            for arg in args
        ]

        return grad_wrt_args


class ScipyVectorWrapperOp(ScipyWrapperOp):
    def build_fn(self):
        # We need to adjust the graph to work with what scipy will be passing into the inner function --
        # always a vector array with size of at least 1
        x, *args = self.inner_inputs
        if x.type.shape != ():
            return super().build_fn()

        new_root_x = x[None].type()
        new_x = new_root_x.squeeze()
        new_outputs = graph_replace(self.inner_outputs, {x: new_x})
        self._fn = fn = function([new_root_x, *args], new_outputs, trust_input=True)

        # Do this reassignment to see the compiled graph in the dprint
        # self.fgraph = fn.maker.fgraph
        self._fn_wrapped = LRUCache1(fn)

    def compute_implicit_gradients(
        self,
        x_star: TensorVariable,
        args: Sequence[TensorVariable | ScalarVariable],
        output_grad: TensorVariable,
        is_minimization: bool,
    ):
        r"""
        Compute gradients of an optimization problem with respect to its parameters.

        Parameters
        ----------
        x_star : TensorVariable
            The symbolic solution of the optimization problem.
        args : Sequence of TensorVariable or ScalarVariable
            The parameters of the optimization problem.
        output_grad : TensorVariable
            The gradient of the output of the optimization Op with respect to some scalar loss.
        is_minimization : bool
            Whether the optimization problem is a minimization problem. If False, it is assumed to be a root-finding
            problem.

        Notes
        -----
        The gradents are computed using the implicit function theorem. Given a fuction `f(x, theta) = 0`, and a function
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

            \frac{d x^*(\theta)}{d \theta} = - \left(\frac{\partial f}{\partial x}\left(x^*(\theta),
            \theta\right)\right)^{-1} \frac{\partial f}{\partial \theta}\left(x^*(\theta), \theta\right)

        Note that this method assumes `f(x_star(theta), theta) = 0`; so it is not immediately applicable to a minimization
        problem, where `f` is the objective function. In this case, we instead take `f` to be the gradient of the objective
        function, which *is* indeed zero at the minimum.
        """
        fgraph = self.fgraph
        inner_x, *inner_args = self.inner_inputs
        implicit_f = self.inner_outputs[0]

        df_dx, *arg_grads = grad(
            implicit_f.sum(),
            [inner_x, *inner_args],
            disconnected_inputs="ignore",
            null_gradients="return",
            return_disconnected="disconnected",
        )

        inner_args_to_diff = [
            arg
            for arg, g in zip(inner_args, arg_grads)
            if not isinstance(g.type, DisconnectedType | NullType)
        ]

        if len(inner_args_to_diff) == 0:
            # No differentiable arguments, return disconnected/null gradients
            return arg_grads

        outer_args_to_diff = [
            arg
            for inner_arg, arg in zip(inner_args, args)
            if inner_arg in inner_args_to_diff
        ]
        invalid_grad_map = {
            arg: g for arg, g in zip(args, arg_grads) if arg not in outer_args_to_diff
        }

        if is_minimization:
            implicit_f = grad(implicit_f, inner_x)

        # Gradients are computed using the inner graph of the optimization op, not the actual inputs/outputs of the op.
        packed_inner_args, packed_arg_shapes, implicit_f = pack_inputs_of_objective(
            implicit_f,
            inner_args_to_diff,
        )

        df_dx, df_dtheta = jacobian(
            implicit_f,
            [inner_x, packed_inner_args],
            disconnected_inputs="ignore",
            vectorize=self.use_vectorized_jac,
        )

        # Replace inner inputs (abstract dummies) with outer inputs (the actual user-provided symbols)
        # at the solution point. Innner arguments aren't needed anymore, delete them to avoid accidental references.
        del inner_x
        del inner_args
        inner_to_outer_map = dict(zip(fgraph.inputs, (x_star, *args)))
        df_dx_star, df_dtheta_star = graph_replace(
            [df_dx, df_dtheta], inner_to_outer_map
        )

        if df_dtheta_star.ndim == 0 or df_dx_star.ndim == 0:
            grad_wrt_args_packed = -(df_dtheta_star / df_dx_star)
        else:
            grad_wrt_args_packed = solve(-atleast_2d(df_dx_star), df_dtheta_star)

        if packed_arg_shapes is not None:
            packed_shapes_from_outer = graph_replace(
                packed_arg_shapes, inner_to_outer_map, strict=False
            )
            grad_wrt_args = unpack(
                grad_wrt_args_packed,
                packed_shapes_from_outer,
                keep_axes=None if all(inp.ndim == 0 for inp in (x_star, *args)) else 0,
            )
        else:
            grad_wrt_args = [grad_wrt_args_packed]

        arg_to_grad = dict(zip(outer_args_to_diff, grad_wrt_args))

        final_grads = []
        for arg in args:
            arg_grad = arg_to_grad.get(arg, None)

            if arg_grad is None:
                final_grads.append(invalid_grad_map[arg])
                continue

            if arg_grad.ndim > 0 and output_grad.ndim > 0:
                g = tensordot(output_grad, arg_grad, [[0], [0]])
            else:
                g = arg_grad * output_grad
            if isinstance(arg.type, ScalarType) and isinstance(g, TensorVariable):
                g = scalar_from_tensor(g)
            final_grads.append(g)

        return final_grads


class MinimizeScalarOp(ScipyScalarWrapperOp):
    def __init__(
        self,
        x: TensorVariable,
        *args: Variable,
        objective: TensorVariable,
        method: str,
        optimizer_kwargs: dict | None = None,
    ):
        if not (isinstance(x, TensorVariable) and x.ndim == 0):
            raise ValueError(
                "The variable `x` must be a scalar (0-dimensional) tensor for minimize_scalar."
            )
        if not (isinstance(objective, TensorVariable) and objective.ndim == 0):
            raise ValueError(
                "The objective function must be a scalar (0-dimensional) tensor for minimize_scalar."
            )
        if x not in ancestors([objective]):
            raise ValueError(
                "The variable `x` must be an input to the computational graph of the objective function."
            )
        self.fgraph = FunctionGraph([x, *args], [objective])

        self.method = method
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self._fn = None
        self._fn_wrapped = None

    def __str__(self):
        return f"{self.__class__.__name__}(method={self.method})"

    def perform(self, node, inputs, outputs):
        global optimize
        if optimize is None:
            import scipy.optimize as optimize

        f = self.fn_wrapped
        f.clear_cache()

        # minimize_scalar doesn't take x0 as an argument. The Op still needs this input (to symbolically determine
        # the args of the objective function), but it is not used in the optimization.
        x0, *args = inputs

        res = optimize.minimize_scalar(
            fun=f.value,
            args=tuple(args),
            method=self.method,
            **self.optimizer_kwargs,
        )

        outputs[0][0] = np.array(res.x, dtype=x0.dtype)
        outputs[1][0] = np.bool_(res.success)

    def L_op(self, inputs, outputs, output_grads):
        # TODO: Handle disconnected inputs
        x, *args = inputs
        x_star, _ = outputs
        output_grad, _ = output_grads

        d_xstar_d_theta = self.compute_implicit_gradients(
            x_star=x_star,
            args=args,
            output_grad=output_grad,
            is_minimization=True,
        )

        return [zeros_like(x), *d_xstar_d_theta]


def minimize_scalar(
    objective: TensorVariable,
    x: TensorVariable,
    method: str = "brent",
    optimizer_kwargs: dict | None = None,
):
    """
    Minimize a scalar objective function using scipy.optimize.minimize_scalar.

    Parameters
    ----------
    objective : TensorVariable
        The objective function to minimize. This should be a PyTensor variable representing a scalar value.
    x : TensorVariable
        The variable with respect to which the objective function is minimized. It must be a scalar and an
        input to the computational graph of `objective`.
    method : str, optional
        The optimization method to use. Default is "brent". See `scipy.optimize.minimize_scalar` for other options.
    optimizer_kwargs : dict, optional
        Additional keyword arguments to pass to `scipy.optimize.minimize_scalar`.

    Returns
    -------
    solution: TensorVariable
        Value of `x` that minimizes `objective(x, *args)`. If the success flag is False, this will be the
        final state returned by the minimization routine, not necessarily a minimum.
    success : TensorVariable
        Symbolic boolean flag indicating whether the minimization routine reported convergence to a minimum
        value, based on the requested convergence criteria.
    """
    args = _find_optimization_parameters(objective, x)

    minimize_scalar_op = MinimizeScalarOp(
        x,
        *args,
        objective=objective,
        method=method,
        optimizer_kwargs=optimizer_kwargs,
    )

    solution, success = minimize_scalar_op(x, *args)

    return solution, success


class MinimizeOp(ScipyVectorWrapperOp):
    def __init__(
        self,
        x: TensorVariable,
        *args: Variable,
        objective: TensorVariable,
        method: str,
        jac: bool = True,
        hess: bool = False,
        hessp: bool = False,
        use_vectorized_jac: bool = False,
        optimizer_kwargs: dict | None = None,
    ):
        if not (isinstance(x, TensorVariable) and x.ndim in (0, 1)):
            raise ValueError(
                "The variable `x` must be a scalar or vector (0-or-1-dimensional) tensor for minimize."
            )
        if not (isinstance(objective, TensorVariable) and objective.ndim == 0):
            raise ValueError(
                "The objective function must be a scalar (0-dimensional) tensor for minimize."
            )
        if x not in ancestors([objective]):
            raise ValueError(
                "The variable `x` must be an input to the computational graph of the objective function."
            )

        self.fgraph = FunctionGraph([x, *args], [objective])
        self.use_vectorized_jac = use_vectorized_jac

        if jac:
            grad_wrt_x = grad(self.fgraph.outputs[0], self.fgraph.inputs[0])
            self.fgraph.add_output(grad_wrt_x)

        if hess:
            hess_wrt_x = jacobian(
                self.fgraph.outputs[-1],
                self.fgraph.inputs[0],
                vectorize=use_vectorized_jac,
            )
            self.fgraph.add_output(hess_wrt_x)

        self.jac = jac
        self.hess = hess
        self.hessp = hessp

        self.method = method
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self._fn = None
        self._fn_wrapped = None

    def __str__(self):
        str_args = ", ".join(
            [
                f"{arg}={getattr(self, arg)}"
                for arg in ["method", "jac", "hess", "hessp"]
            ]
        )
        return f"{self.__class__.__name__}({str_args})"

    def perform(self, node, inputs, outputs):
        global optimize
        if optimize is None:
            import scipy.optimize as optimize

        f = self.fn_wrapped
        x0, *args = inputs

        res = optimize.minimize(
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
        x_star, _success = outputs
        output_grad, _ = output_grads

        d_xstar_d_theta = self.compute_implicit_gradients(
            x_star=x_star,
            args=args,
            output_grad=output_grad,
            is_minimization=True,
        )

        return [zeros_like(x), *d_xstar_d_theta]


def pack_inputs_of_objective(
    objective: TensorVariable,
    x: TensorVariable | Sequence[TensorVariable],
) -> tuple[TensorVariable, list[TensorVariable] | None, TensorVariable]:
    """
    Packs inputs `x` into a single tensor if `x` is a sequence of tensors, and rewrites the `objective` graph
    to use the packed input. Also returns the packed shapes for unpacking the output later.

    If `x` is a single tensor, it is returned as is, and no rewriting is done.
    """
    packed_shapes = None

    if not isinstance(x, Sequence):
        packed_input = x
    elif len(x) == 1:
        packed_input = x[0]
    else:
        packed_input, packed_shapes = pack(*x)
        unpacked_output = unpack(packed_input, packed_shapes)

        objective = graph_replace(
            objective,
            {
                xi: ui.astype(xi.type.dtype)
                if not (isinstance(xi.type, ScalarType))
                else scalar_from_tensor(ui.astype(xi.type.dtype))
                for xi, ui in zip(x, unpacked_output)
            },
        )

    return packed_input, packed_shapes, objective


def minimize(
    objective: TensorVariable,
    x: TensorVariable | Sequence[TensorVariable],
    method: str = "BFGS",
    jac: bool = True,
    hess: bool = False,
    use_vectorized_jac: bool = False,
    optimizer_kwargs: dict | None = None,
) -> tuple[TensorVariable | tuple[TensorVariable, ...], TensorVariable]:
    """
    Minimize a scalar objective function using scipy.optimize.minimize.

    Parameters
    ----------
    objective : TensorVariable
        The objective function to minimize. This should be a pytensor variable representing a scalar value.
    x: TensorVariable or list of TensorVariable
        The variable or variables with respect to which the objective function is minimized. It must be an input to the
        computational graph of `objective`.
    method: str, optional
        The optimization method to use. Default is "BFGS". See scipy.optimize.minimize for other options.
    jac: bool, optional
        Whether to compute and use the gradient of the objective function with respect to x for optimization.
        Default is True.
    hess: bool, optional
        Whether to compute and use the Hessian of the objective function with respect to x for optimization.
        Default is False. Note that some methods require this, while others do not support it.
    use_vectorized_jac: bool, optional
        Whether to use a vectorized graph (vmap) to compute the jacobian (and/or hessian) matrix. If False, a
        scan will be used instead. This comes down to a memory/compute trade-off. Vectorized graphs can be faster,
        but use more memory. Default is False.
    optimizer_kwargs
        Additional keyword arguments to pass to scipy.optimize.minimize

    Returns
    -------
    solution: TensorVariable or tuple of TensorVariable
        The optimized value of each of inputs in `x` that minimizes `objective(x, *args)`. If the success flag
        is False, this will be the final state of the minimization routine, but not necessarily a minimum.

    success: TensorVariable
        Symbolic boolean flag indicating whether the minimization routine reported convergence to a minimum
        value, based on the requested convergence criteria.
    """
    objective = as_tensor_variable(objective)

    packed_input, packed_shapes, objective = pack_inputs_of_objective(objective, x)
    args = _find_optimization_parameters(objective, packed_input)

    minimize_op = MinimizeOp(
        packed_input,
        *args,
        objective=objective,
        method=method,
        jac=jac,
        hess=hess,
        use_vectorized_jac=use_vectorized_jac,
        optimizer_kwargs=optimizer_kwargs,
    )

    solution, success = minimize_op(packed_input, *args)

    if packed_shapes is not None:
        solution = unpack(solution, packed_shapes)

    return solution, success


class RootScalarOp(ScipyScalarWrapperOp):
    def __init__(
        self,
        variables: TensorVariable,
        *args: Variable,
        equation: TensorVariable,
        method: str,
        jac: bool = False,
        hess: bool = False,
        optimizer_kwargs=None,
    ):
        if not (isinstance(variables, TensorVariable) and variables.ndim == 0):
            raise ValueError(
                "The variable `x` must be a scalar (0-dimensional) tensor for root_scalar."
            )
        if not (isinstance(equation, TensorVariable) and equation.ndim == 0):
            raise ValueError(
                "The equation must be a scalar (0-dimensional) tensor for root_scalar."
            )
        if variables not in ancestors([equation]):
            raise ValueError(
                "The variable `variables` must be an input to the computational graph of the equation."
            )

        self.fgraph = FunctionGraph([variables, *args], [equation])

        if jac:
            f_prime = grad(self.fgraph.outputs[0], self.fgraph.inputs[0])
            self.fgraph.add_output(f_prime)

        if hess:
            if not jac:
                raise ValueError(
                    "Cannot set `hess=True` without `jac=True`. No methods use second derivatives without also"
                    " using first derivatives."
                )
            f_double_prime = grad(self.fgraph.outputs[-1], self.fgraph.inputs[0])
            self.fgraph.add_output(f_double_prime)

        self.method = method
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self.jac = jac
        self.hess = hess

        self._fn = None
        self._fn_wrapped = None

    def __str__(self):
        str_args = ", ".join(
            [f"{arg}={getattr(self, arg)}" for arg in ["method", "jac", "hess"]]
        )
        return f"{self.__class__.__name__}({str_args})"

    def perform(self, node, inputs, outputs):
        global optimize
        if optimize is None:
            import scipy.optimize as optimize

        f = self.fn_wrapped
        f.clear_cache()
        # f.copy_x = True

        variables, *args = inputs

        res = optimize.root_scalar(
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

        d_xstar_d_theta = self.compute_implicit_gradients(
            x_star=x_star,
            args=args,
            output_grad=output_grad,
            is_minimization=False,
        )

        return [zeros_like(x), *d_xstar_d_theta]


def root_scalar(
    equation: TensorVariable,
    variable: TensorVariable,
    method: str = "secant",
    jac: bool = False,
    hess: bool = False,
    optimizer_kwargs: dict | None = None,
) -> tuple[TensorVariable, TensorVariable]:
    """
    Find roots of a scalar equation using scipy.optimize.root_scalar.

    Parameters
    ----------
    equation : TensorVariable
        The equation for which to find roots. This should be a PyTensor variable representing a single equation in one
        variable. The function will find `variables` such that `equation(variables, *args) = 0`.
    variable : TensorVariable
        The variable with respect to which the equation is solved. It must be a scalar and an input to the
        computational graph of `equation`.
    method : str, optional
        The root-finding method to use. Default is "secant". See `scipy.optimize.root_scalar` for other options.
    jac : bool, optional
        Whether to compute and use the first derivative of the equation with respect to `variables`.
        Default is False. Some methods require this.
    hess : bool, optional
        Whether to compute and use the second derivative of the equation with respect to `variables`.
        Default is False. Some methods require this.
    optimizer_kwargs : dict, optional
        Additional keyword arguments to pass to `scipy.optimize.root_scalar`.

    Returns
    -------
    solution: TensorVariable
        The final state of the root-finding routine. When `success` is True, this is the value of `variables` that
        causes `equation` to evaluate to zero. Otherwise it is the final state returned by the root-finding
        routine, but not necessarily a root.

    success: TensorVariable
        Boolean indicating whether the root-finding was successful. If True, the solution is a root of the equation
    """
    args = _find_optimization_parameters(equation, variable)

    root_scalar_op = RootScalarOp(
        variable,
        *args,
        equation=equation,
        method=method,
        jac=jac,
        hess=hess,
        optimizer_kwargs=optimizer_kwargs,
    )

    solution, success = root_scalar_op(variable, *args)

    return solution, success


class RootOp(ScipyVectorWrapperOp):
    __props__ = ("method", "jac")

    def __init__(
        self,
        variables: TensorVariable,
        *args: Variable,
        equations: TensorVariable,
        method: str,
        jac: bool = True,
        optimizer_kwargs: dict | None = None,
        use_vectorized_jac: bool = False,
    ):
        if not isinstance(variables, TensorVariable):
            raise ValueError("The variable `variables` must be a tensor for root.")
        if not isinstance(equations, TensorVariable):
            raise ValueError("The equations must be a tensor for root.")
        if variables.ndim != equations.ndim:
            raise ValueError(
                "The variable `variables` must have the same number of dimensions as the equations."
            )
        if variables not in ancestors([equations]):
            raise ValueError(
                "The variable `variables` must be an input to the computational graph of the equations."
            )

        self.fgraph = FunctionGraph([variables, *args], [equations])
        self.use_vectorized_jac = use_vectorized_jac

        if jac:
            jac_wrt_x = jacobian(
                self.fgraph.outputs[0],
                self.fgraph.inputs[0],
                vectorize=use_vectorized_jac,
            )
            self.fgraph.add_output(atleast_2d(jac_wrt_x))

        self.jac = jac

        self.method = method
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self._fn = None
        self._fn_wrapped = None

    def __str__(self):
        str_args = ", ".join(
            [f"{arg}={getattr(self, arg)}" for arg in ["method", "jac"]]
        )
        return f"{self.__class__.__name__}({str_args})"

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
        global optimize
        if optimize is None:
            import scipy.optimize as optimize

        f = self.fn_wrapped
        f.clear_cache()
        f.copy_x = True

        variables, *args = inputs

        res = optimize.root(
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

    def L_op(self, inputs, outputs, output_grads):
        x, *args = inputs
        x_star, _ = outputs
        output_grad, _ = output_grads

        d_xstar_d_theta = self.compute_implicit_gradients(
            x_star=x_star,
            args=args,
            output_grad=output_grad,
            is_minimization=False,
        )

        return [zeros_like(x), *d_xstar_d_theta]


def root(
    equations: TensorVariable,
    variables: TensorVariable | Sequence[TensorVariable],
    method: str = "hybr",
    jac: bool = True,
    use_vectorized_jac: bool = False,
    optimizer_kwargs: dict | None = None,
) -> tuple[TensorVariable | Sequence[TensorVariable], TensorVariable]:
    """
    Find roots of a system of equations using scipy.optimize.root.

    Parameters
    ----------
    equations : TensorVariable
        The system of equations for which to find roots. This should be a PyTensor variable representing a
        vector (or scalar) value. The function will find `variables` such that `equations(variables, *args) = 0`.
    variables : TensorVariable
        The variable(s) with respect to which the system of equations is solved. It must be an input to the
        computational graph of `equations` and have the same number of dimensions as `equations`.
    method : str, optional
        The root-finding method to use. Default is "hybr". See `scipy.optimize.root` for other options.
    jac : bool, optional
        Whether to compute and use the Jacobian of the `equations` with respect to `variables`.
        Default is True. Most methods require this.
    use_vectorized_jac: bool, optional
        Whether to use a vectorized graph (vmap) to compute the jacobian matrix. If False, a scan will be used instead.
        This comes down to a memory/compute trade-off. Vectorized graphs can be faster, but use more memory.
        Default is False.
    optimizer_kwargs : dict, optional
        Additional keyword arguments to pass to `scipy.optimize.root`.

    Returns
    -------
    solution: TensorVariable
        The final state of the root-finding routine. When `success` is True, this is the value of `variables` that
        causes all `equations` to evaluate to zero. Otherwise it is the final state returned by the root-finding
        routine, but not necessarily a root.

    success: TensorVariable
        Boolean indicating whether the root-finding was successful. If True, the solution is a root of the equation
    """
    packed_variables, packed_shapes, equations = pack_inputs_of_objective(
        equations, variables
    )
    args = _find_optimization_parameters(equations, packed_variables)

    root_op = RootOp(
        packed_variables,
        *args,
        equations=equations,
        method=method,
        jac=jac,
        optimizer_kwargs=optimizer_kwargs,
        use_vectorized_jac=use_vectorized_jac,
    )

    solution, success = root_op(packed_variables, *args)
    if packed_shapes is not None:
        solution = unpack(solution, packed_shapes)

    return solution, success


__all__ = ["minimize", "minimize_scalar", "root", "root_scalar"]
