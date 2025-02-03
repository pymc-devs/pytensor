"""Convert a jax function to a pytensor compatible function."""

import functools as ft
import logging
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten, tree_map, tree_unflatten

import pytensor.compile.builders
import pytensor.tensor as pt
from pytensor.gradient import DisconnectedType
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify


log = logging.getLogger(__name__)


def _filter_ptvars(x):
    return isinstance(x, pt.Variable)


def as_jax_op(jaxfunc, use_infer_static_shape=True, name=None):
    """Return a Pytensor function from a JAX jittable function.

    This decorator transforms any JAX-jittable function into a function that accepts
    and returns `pytensor.Variable`. The JAX-jittable function can accept any
    nested python structure (a `Pytree
    <https://jax.readthedocs.io/en/latest/pytrees.html>`_) as input, and might return
    any nested Python structure.

    Parameters
    ----------
    jaxfunc : JAX-jittable function
        JAX function which will be wrapped in a Pytensor Op.
    name: str, optional
        Name of the created pytensor Op, defaults to the name of the passed function.
        Only used internally in the pytensor graph.

    Returns
    -------
    Callable :
        A function which expects a nested python structure of `pytensor.Variable` and
        static variables as inputs and returns `pytensor.Variable` with the same
        API as the original jaxfunc. The resulting model can be compiled either with the
        default C backend or the JAX backend.

    Examples
    --------

    We define a JAX function `f_jax` that accepts a matrix `x`, a vector `y` and a
    dictionary as input. This is transformed to a pytensor function  with the decorator
    `as_jax_op`, and can subsequently be used like normal pytensor operators, i.e.
    for evaluation and calculating gradients.

    >>> import numpy  # doctest: +ELLIPSIS
    >>> import jax.numpy as jnp  # doctest: +ELLIPSIS
    >>> import pytensor  # doctest: +ELLIPSIS
    >>> import pytensor.tensor as pt  # doctest: +ELLIPSIS
    >>> x = pt.tensor("x", shape=(2,))
    >>> y = pt.tensor("y", shape=(2, 2))
    >>> a = pt.tensor("a", shape=())
    >>> args_dict = {"a": a}
    >>> @pytensor.as_jax_op
    ... def f_jax(x, y, args_dict):
    ...     z = jnp.dot(x, y) + args_dict["a"]
    ...     return z
    >>> z = f_jax(x, y, args_dict)
    >>> z_sum = pt.sum(z)
    >>> grad_wrt_a = pt.grad(z_sum, a)
    >>> f_all = pytensor.function([x, y, a], [z_sum, grad_wrt_a])
    >>> f_all(numpy.array([1, 2]), numpy.array([[1, 2], [3, 4]]), 1)
    [array(19.), array(2.)]


    Notes
    -----
    The function is based on a blog post by Ricardo Vieira and Adrian Seyboldt,
    available at
    `pymc-labls.io <https://www.pymc-labs.io/blog-posts/jax-functions-in-pymc-3-quick
    -examples/>`__.
    To accept functions and non pytensor variables as input, the function make use
    of :func:`equinox.partition` and :func:`equinox.combine` to split and combine the
    variables. Shapes are inferred using
    :func:`pytensor.compile.builders.infer_shape` and :func:`jax.eval_shape`.
    """

    def func(*args, **kwargs):
        """Return a pytensor from a jax jittable function."""
        ### Split variables: in the ones that will be transformed to JAX inputs,
        ### pytensor.Variables; _WrappedFunc, that are functions that have been returned
        ### from a transformed function; and the rest, static variables that are not
        ### transformed.

        pt_vars, static_vars_tmp = eqx.partition(
            (args, kwargs), _filter_ptvars, is_leaf=callable
        )
        # is_leaf=callable is used, as libraries like diffrax or equinox might return
        # functions that are still seen as a nested pytree structure. We consider them
        # as wrappable functions, that will be wrapped with _WrappedFunc.

        func_vars, static_vars = eqx.partition(
            static_vars_tmp, lambda x: isinstance(x, _WrappedFunc), is_leaf=callable
        )
        vars_from_func = tree_map(lambda x: x.get_vars(), func_vars)
        pt_vars = dict(vars=pt_vars, vars_from_func=vars_from_func)

        # Flatten nested python structures, e.g. {"a": tensor_a, "b": [tensor_b]}
        # becomes [tensor_a, tensor_b], because pytensor ops only accepts lists of
        # pytensor.Variables as input.
        pt_vars_flat, vars_treedef = tree_flatten(pt_vars)

        # Infer shapes and types of the variables
        pt_vars_types_flat = [var.type for var in pt_vars_flat]

        if use_infer_static_shape:
            shapes_vars_flat = [
                pt.basic.infer_static_shape(var.shape)[1] for var in pt_vars_flat
            ]

            dummy_inputs_jax_flat = [
                jnp.empty(shape, dtype=var.type.dtype)
                for var, shape in zip(pt_vars_flat, shapes_vars_flat, strict=True)
            ]

        else:
            shapes_vars_flat = pytensor.compile.builders.infer_shape(
                pt_vars_flat, (), ()
            )
            dummy_inputs_jax_flat = [
                jnp.empty([int(dim.eval()) for dim in shape], dtype=var.type.dtype)
                for var, shape in zip(pt_vars_flat, shapes_vars_flat, strict=True)
            ]

        dummy_inputs_jax = tree_unflatten(vars_treedef, dummy_inputs_jax_flat)

        # Combine the static variables with the inputs, and split them again in the
        # output. Static variables don't take part in the graph, or might be a
        # a function that is returned.
        jaxfunc_partitioned, static_out_dic = _partition_jaxfunc(
            jaxfunc, static_vars, func_vars
        )

        func_flattened = _flatten_func(jaxfunc_partitioned, vars_treedef)

        jaxtypes_outvars = jax.eval_shape(
            ft.partial(jaxfunc_partitioned, vars=dummy_inputs_jax),
        )

        jaxtypes_outvars_flat, outvars_treedef = tree_flatten(jaxtypes_outvars)

        pttypes_outvars = [
            pt.TensorType(dtype=var.dtype, shape=var.shape)
            for var in jaxtypes_outvars_flat
        ]

        ### Call the function that accepts flat inputs, which in turn calls the one that
        ### combines the inputs and static variables.
        jitted_sol_op_jax = jax.jit(func_flattened)
        len_gz = len(pttypes_outvars)

        vjp_sol_op_jax = _get_vjp_sol_op_jax(func_flattened, len_gz)
        jitted_vjp_sol_op_jax = jax.jit(vjp_sol_op_jax)

        # Get classes that creates a Pytensor Op out of our function that accept
        # flattened inputs. They are created each time, to set a custom name for the
        # class.
        class JAXOp_local(JAXOp):
            pass

        class VJPJAXOp_local(VJPJAXOp):
            pass

        if name is None:
            curr_name = jaxfunc.__name__
        else:
            curr_name = name
        JAXOp_local.__name__ = curr_name
        JAXOp_local.__qualname__ = ".".join(
            JAXOp_local.__qualname__.split(".")[:-1] + [curr_name]
        )

        VJPJAXOp_local.__name__ = "VJP_" + curr_name
        VJPJAXOp_local.__qualname__ = ".".join(
            VJPJAXOp_local.__qualname__.split(".")[:-1] + ["VJP_" + curr_name]
        )

        local_op = JAXOp_local(
            vars_treedef,
            outvars_treedef,
            input_types=pt_vars_types_flat,
            output_types=pttypes_outvars,
            jitted_sol_op_jax=jitted_sol_op_jax,
            jitted_vjp_sol_op_jax=jitted_vjp_sol_op_jax,
        )

        ### Evaluate the Pytensor Op and return unflattened results
        output_flat = local_op(*pt_vars_flat)
        if not isinstance(output_flat, Sequence):
            output_flat = [output_flat]  # tree_unflatten expects a sequence.
        outvars = tree_unflatten(outvars_treedef, output_flat)

        static_outfuncs, static_outvars = eqx.partition(
            static_out_dic["out"], callable, is_leaf=callable
        )

        static_outfuncs_flat, treedef_outfuncs = jax.tree_util.tree_flatten(
            static_outfuncs, is_leaf=callable
        )
        for i_func, _ in enumerate(static_outfuncs_flat):
            static_outfuncs_flat[i_func] = _WrappedFunc(
                jaxfunc, i_func, *args, **kwargs
            )

        static_outfuncs = jax.tree_util.tree_unflatten(
            treedef_outfuncs, static_outfuncs_flat
        )
        static_vars = eqx.combine(static_outfuncs, static_outvars, is_leaf=callable)

        output = eqx.combine(outvars, static_vars, is_leaf=callable)

        return output

    return func


class _WrappedFunc:
    def __init__(self, exterior_func, i_func, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.i_func = i_func
        vars, static_vars = eqx.partition(
            (self.args, self.kwargs), _filter_ptvars, is_leaf=callable
        )
        self.vars = vars
        self.static_vars = static_vars
        self.exterior_func = exterior_func

    def __call__(self, *args, **kwargs):
        # If called, assume that args and kwargs are pytensors, so return the result
        # as pytensors.
        def f(func, *args, **kwargs):
            res = func(*args, **kwargs)
            return res

        return as_jax_op(f)(self, *args, **kwargs)

    def get_vars(self):
        return self.vars

    def get_func_with_vars(self, vars):
        # Use other variables than the saved ones, to generate the function. This
        # is used to transform vars externally from pytensor to JAX, and use the
        # then create the function which is returned.

        args, kwargs = eqx.combine(vars, self.static_vars, is_leaf=callable)
        output = self.exterior_func(*args, **kwargs)
        outfuncs, _ = eqx.partition(output, callable, is_leaf=callable)
        outfuncs_flat, _ = jax.tree_util.tree_flatten(outfuncs, is_leaf=callable)
        interior_func = outfuncs_flat[self.i_func]
        return interior_func


def _get_vjp_sol_op_jax(jaxfunc, len_gz):
    def vjp_sol_op_jax(args):
        y0 = args[:-len_gz]
        gz = args[-len_gz:]
        if len(gz) == 1:
            gz = gz[0]

        def func(*inputs):
            return jaxfunc(inputs)

        primals, vjp_fn = jax.vjp(func, *y0)
        gz = tree_map(
            lambda g, primal: jnp.broadcast_to(g, jnp.shape(primal)).astype(
                primal.dtype
            ),  # Also cast to the dtype of the primal, this shouldn't be
            # necessary, but it happens that the returned dtype of the gradient isn't
            # the same anymore.
            gz,
            primals,
        )
        if len(y0) == 1:
            return vjp_fn(gz)[0]
        else:
            return tuple(vjp_fn(gz))

    return vjp_sol_op_jax


def _partition_jaxfunc(jaxfunc, static_vars, func_vars):
    """Partition the jax function into static and non-static variables.

    Returns a function that accepts only non-static variables and returns the non-static
    variables. The returned static variables are stored in a dictionary and returned,
    to allow the referencing after creating the function

    Additionally wrapped functions saved in func_vars are regenerated with
    vars["vars_from_func"] as input, to allow the transformation of the variables.
    """
    static_out_dic = {"out": None}

    def jaxfunc_partitioned(vars):
        vars, vars_from_func = vars["vars"], vars["vars_from_func"]
        func_vars_evaled = tree_map(
            lambda x, y: x.get_func_with_vars(y), func_vars, vars_from_func
        )
        args, kwargs = eqx.combine(
            vars, static_vars, func_vars_evaled, is_leaf=callable
        )

        out = jaxfunc(*args, **kwargs)
        outvars, static_out = eqx.partition(out, eqx.is_array, is_leaf=callable)
        static_out_dic["out"] = static_out
        return outvars

    return jaxfunc_partitioned, static_out_dic


### Construct the function that accepts flat inputs and returns flat outputs.
def _flatten_func(jaxfunc, vars_treedef):
    def func_flattened(vars_flat):
        vars = tree_unflatten(vars_treedef, vars_flat)
        outvars = jaxfunc(vars)
        outvars_flat, _ = tree_flatten(outvars)
        return _normalize_flat_output(outvars_flat)

    return func_flattened


def _normalize_flat_output(output):
    if len(output) > 1:
        return tuple(
            output
        )  # Transform to tuple because jax makes a difference between
        # tuple and list and not pytensor
    else:
        return output[0]


class JAXOp(Op):
    def __init__(
        self,
        input_treedef,
        output_treeedef,
        input_types,
        output_types,
        jitted_sol_op_jax,
        jitted_vjp_sol_op_jax,
    ):
        self.vjp_sol_op = None
        self.input_treedef = input_treedef
        self.output_treedef = output_treeedef
        self.input_types = input_types
        self.output_types = output_types
        self.jitted_sol_op_jax = jitted_sol_op_jax
        self.jitted_vjp_sol_op_jax = jitted_vjp_sol_op_jax

    def make_node(self, *inputs):
        self.num_inputs = len(inputs)

        # Define our output variables
        outputs = [pt.as_tensor_variable(type()) for type in self.output_types]
        self.num_outputs = len(outputs)

        self.vjp_sol_op = VJPJAXOp(
            self.input_treedef,
            self.input_types,
            self.jitted_vjp_sol_op_jax,
        )

        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        results = self.jitted_sol_op_jax(inputs)
        if self.num_outputs > 1:
            for i in range(self.num_outputs):
                outputs[i][0] = np.array(results[i], self.output_types[i].dtype)
        else:
            outputs[0][0] = np.array(results, self.output_types[0].dtype)

    def perform_jax(self, *inputs):
        results = self.jitted_sol_op_jax(inputs)
        return results

    def grad(self, inputs, output_gradients):
        # If a output is not used, it is disconnected and doesn't have a gradient.
        # Set gradient here to zero for those outputs.
        for i in range(self.num_outputs):
            if isinstance(output_gradients[i].type, DisconnectedType):
                if None not in self.output_types[i].shape:
                    output_gradients[i] = pt.zeros(
                        self.output_types[i].shape, self.output_types[i].dtype
                    )
                else:
                    output_gradients[i] = pt.zeros((), self.output_types[i].dtype)
        result = self.vjp_sol_op(inputs, output_gradients)

        if self.num_inputs > 1:
            return result
        else:
            return (result,)  # Pytensor requires a tuple here


# vector-jacobian product Op
class VJPJAXOp(Op):
    def __init__(
        self,
        input_treedef,
        input_types,
        jitted_vjp_sol_op_jax,
    ):
        self.input_treedef = input_treedef
        self.input_types = input_types
        self.jitted_vjp_sol_op_jax = jitted_vjp_sol_op_jax

    def make_node(self, y0, gz):
        y0 = [
            pt.as_tensor_variable(
                _y,
            ).astype(self.input_types[i].dtype)
            for i, _y in enumerate(y0)
        ]
        gz_not_disconntected = [
            pt.as_tensor_variable(_gz)
            for _gz in gz
            if not isinstance(_gz.type, DisconnectedType)
        ]
        outputs = [in_type() for in_type in self.input_types]
        self.num_outputs = len(outputs)
        return Apply(self, y0 + gz_not_disconntected, outputs)

    def perform(self, node, inputs, outputs):
        results = self.jitted_vjp_sol_op_jax(tuple(inputs))
        if len(self.input_types) > 1:
            for i, result in enumerate(results):
                outputs[i][0] = np.array(result, self.input_types[i].dtype)
        else:
            outputs[0][0] = np.array(results, self.input_types[0].dtype)

    def perform_jax(self, *inputs):
        results = self.jitted_vjp_sol_op_jax(tuple(inputs))
        if self.num_outputs == 1:
            if isinstance(results, Sequence):
                return results[0]
            else:
                return results
        else:
            return tuple(results)


@jax_funcify.register(JAXOp)
def sol_op_jax_funcify(op, **kwargs):
    return op.perform_jax


@jax_funcify.register(VJPJAXOp)
def vjp_sol_op_jax_funcify(op, **kwargs):
    return op.perform_jax
