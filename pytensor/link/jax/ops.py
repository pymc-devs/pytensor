"""Convert a jax function to a pytensor compatible function."""

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


class JAXOp(Op):
    """
    JAXOp is a PyTensor Op that wraps a JAX function, providing both forward computation and reverse-mode differentiation (via the VJPJAXOp class).

    Parameters
    ----------
    input_types : list
        A list of PyTensor types for each input variable.
    output_types : list
        A list of PyTensor types for each output variable.
    flat_func : callable
        The JAX function that computes outputs from inputs. Inputs and outputs have to be provided as flat arrays.
    name : str, optional
        A custom name for the Op instance. If provided, the class name will be
        updated accordingly.

    Example
    -------
    This example defines a simple function that sums the input array with a dynamic shape.

    >>> import numpy as np
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pytensor.tensor import TensorType
    >>>
    >>> # Create the jax function that sums the input array.
    >>> def sum_function(x, y):
    ...     return jnp.sum(x + y)
    >>>
    >>> # Create the input and output types, input has a dynamic shape.
    >>> input_type = TensorType("float32", shape=(None,))
    >>> output_type = TensorType("float32", shape=(1,))
    >>>
    >>> # Instantiate a JAXOp; tree definitions are set to None for simplicity.
    >>> op = JAXOp(
    ...     [input_type, input_type], [output_type], sum_function, name="DummyJAXOp"
    ... )
    >>> # Define symbolic input variables.
    >>> x = pt.tensor("x", dtype="float32", shape=(2,))
    >>> y = pt.tensor("y", dtype="float32", shape=(2,))
    >>> # Compile a PyTensor function.
    >>> result = op(x, y)
    >>> f = pytensor.function([x, y], [result])
    >>> print(
    ...     f(
    ...         np.array([2.0, 3.0], dtype=np.float32),
    ...         np.array([4.0, 5.0], dtype=np.float32),
    ...     )
    ... )
    [array(14., dtype=float32)]
    >>>
    >>> # Compute the gradient of op(x, y) with respect to x.
    >>> g = pt.grad(result[0], x)
    >>> grad_f = pytensor.function([x, y], [g])
    >>> print(
    ...     grad_f(
    ...         np.array([2.0, 3.0], dtype=np.float32),
    ...         np.array([4.0, 5.0], dtype=np.float32),
    ...     )
    ... )
    [array([1., 1.], dtype=float32)]
    """

    def __init__(self, input_types, output_types, flat_func, name=None):
        self.input_types = input_types
        self.output_types = output_types
        self.num_inputs = len(input_types)
        self.num_outputs = len(output_types)
        normalized_flat_func = _normalize_flat_func(flat_func)
        self.jitted_func = jax.jit(normalized_flat_func)

        vjp_func = _get_vjp_jax_op(normalized_flat_func, len(output_types))
        normalized_vjp_func = _normalize_flat_func(vjp_func)
        self.jitted_vjp = jax.jit(normalized_vjp_func)
        self.vjp_jax_op = VJPJAXOp(
            self.input_types,
            self.jitted_vjp,
            name=("VJP" + name) if name is not None else None,
        )

        if name is not None:
            self.custom_name = name
            self.__class__.__name__ = name
            self.__class__.__qualname__ = ".".join(
                self.__class__.__qualname__.split(".")[:-1] + [name]
            )

    def make_node(self, *inputs):
        outputs = [pt.as_tensor_variable(typ()) for typ in self.output_types]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        results = self.jitted_func(*inputs)
        if self.num_outputs > 1:
            for i in range(self.num_outputs):
                outputs[i][0] = np.array(results[i], self.output_types[i].dtype)
        else:
            outputs[0][0] = np.array(results, self.output_types[0].dtype)

    def perform_jax(self, *inputs):
        return self.jitted_func(*inputs)

    def grad(self, inputs, output_gradients):
        # If a output is not used, it gets disconnected by pytensor and won't have a
        # gradient. Set gradient here to zero for those outputs.
        for i in range(self.num_outputs):
            if isinstance(output_gradients[i].type, DisconnectedType):
                zero_shape = (
                    self.output_types[i].shape
                    if None not in self.output_types[i].shape
                    else ()
                )
                output_gradients[i] = pt.zeros(zero_shape, self.output_types[i].dtype)

        # Compute the gradient.
        grad_result = self.vjp_jax_op(inputs, output_gradients)
        return grad_result if self.num_inputs > 1 else (grad_result,)


class VJPJAXOp(Op):
    def __init__(self, input_types, jitted_vjp, name=None):
        self.input_types = input_types
        self.jitted_vjp = jitted_vjp
        if name is not None:
            self.custom_name = name
            self.__class__.__name__ = name
            self.__class__.__qualname__ = ".".join(
                self.__class__.__qualname__.split(".")[:-1] + [name]
            )

    def make_node(self, y0, gz):
        y0_converted = [
            pt.as_tensor_variable(y).astype(self.input_types[i].dtype)
            for i, y in enumerate(y0)
        ]
        gz_not_disconnected = [
            pt.as_tensor_variable(g)
            for g in gz
            if not isinstance(g.type, DisconnectedType)
        ]
        outputs = [typ() for typ in self.input_types]
        self.num_outputs = len(outputs)
        return Apply(self, y0_converted + gz_not_disconnected, outputs)

    def perform(self, node, inputs, outputs):
        results = self.jitted_vjp(*inputs)
        if len(self.input_types) > 1:
            for i, res in enumerate(results):
                outputs[i][0] = np.array(res, self.input_types[i].dtype)
        else:
            outputs[0][0] = np.array(results, self.input_types[0].dtype)

    def perform_jax(self, *inputs):
        return self.jitted_vjp(*inputs)


def _normalize_flat_func(func):
    def normalized_func(*flat_vars):
        out_flat = func(*flat_vars)
        if isinstance(out_flat, Sequence):
            return tuple(out_flat) if len(out_flat) > 1 else out_flat[0]
        else:
            return out_flat

    return normalized_func


def _get_vjp_jax_op(flat_func, num_out):
    def vjp_op(*args):
        y0 = args[:-num_out]
        gz = args[-num_out:]
        if len(gz) == 1:
            gz = gz[0]

        def f(*inputs):
            return flat_func(*inputs)

        primals, vjp_fn = jax.vjp(f, *y0)

        def broadcast_to_shape(g, shape):
            if g.ndim > 0 and g.shape[0] == 1:
                g_squeezed = jnp.squeeze(g, axis=0)
            else:
                g_squeezed = g
            return jnp.broadcast_to(g_squeezed, shape)

        gz = tree_map(
            lambda g, p: broadcast_to_shape(g, jnp.shape(p)).astype(p.dtype),
            gz,
            primals,
        )
        return vjp_fn(gz)

    return vjp_op


def as_jax_op(jaxfunc, use_infer_static_shape=True, name=None):
    """Return a Pytensor-compatible function from a JAX jittable function.

    This decorator wraps a JAX function so that it accepts and returns `pytensor.Variable`
    objects. The JAX-jittable function can accept any
    nested python structure (a `Pytree
    <https://jax.readthedocs.io/en/latest/pytrees.html>`_) as input, and might return
    any nested Python structure.

    Parameters
    ----------
    jaxfunc : Callable
        A JAX function to be wrapped.
    use_infer_static_shape : bool, optional
        If True, use static shape inference; otherwise, use runtime shape inference.
        Default is True.
    name : str, optional
        A custom name for the created Pytensor Op instance. If None, the name of jaxfunc
        is used.

    Returns
    -------
    Callable
        A function that wraps the given JAX function so that it can be called with
        pytensor.Variable inputs and returns pytensor.Variable outputs.

    Examples
    --------

    >>> import jax.numpy as jnp
    >>> import pytensor.tensor as pt
    >>> @as_jax_op
    ... def add(x, y):
    ...     return jnp.add(x, y)
    >>> x = pt.scalar("x")
    >>> y = pt.scalar("y")
    >>> result = add(x, y)
    >>> f = pytensor.function([x, y], [result])
    >>> print(f(1, 2))
    [array(3.)]

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
        # 1. Partition inputs into dynamic pytensor variables, wrapped functions and
        # static variables.
        # Static variables don't take part in the graph.
        pt_vars, func_vars, static_vars = _split_inputs(args, kwargs)

        # 2. Get the original variables from the wrapped functions.
        vars_from_func = tree_map(lambda f: f.get_vars(), func_vars)
        input_dict = {"vars": pt_vars, "vars_from_func": vars_from_func}

        # 3. Flatten the input dictionary.
        # e.g. {"a": tensor_a, "b": [tensor_b]} becomes [tensor_a, tensor_b], because
        # pytensor ops only accepts lists of pytensor.Variables as input.
        pt_vars_flat, pt_vars_treedef = tree_flatten(
            input_dict,
        )
        pt_types = [var.type for var in pt_vars_flat]

        # 4. Create dummy inputs for shape inference.
        shapes = _infer_shapes(pt_vars_flat, use_infer_static_shape)
        dummy_in_flat = _create_dummy_inputs_from_shapes(
            pt_vars_flat, shapes, use_infer_static_shape
        )
        dummy_inputs = tree_unflatten(pt_vars_treedef, dummy_in_flat)

        # 5. Partition the JAX function into dynamic and static parts.
        jaxfunc_dynamic, static_out_dic = _partition_jaxfunc(
            jaxfunc, static_vars, func_vars
        )
        flat_func = _flatten_func(jaxfunc_dynamic, pt_vars_treedef)

        # 6. Infer output types using JAX's eval_shape.
        out_treedef, pt_types_flat = _infer_output_types(jaxfunc_dynamic, dummy_inputs)

        # 7. Create the Pytensor Op instance.
        curr_name = "JAXOp_" + (jaxfunc.__name__ if name is None else name)
        op_instance = JAXOp(
            pt_types,
            pt_types_flat,
            flat_func,
            name=curr_name,
        )

        # 8. Execute the op and unflatten the outputs.
        output_flat = op_instance(*pt_vars_flat)
        if not isinstance(output_flat, Sequence):
            output_flat = [output_flat]
        outvars = tree_unflatten(out_treedef, output_flat)

        # 9. Combine with static outputs and wrap eventual output functions with
        # _WrappedFunc
        return _process_outputs(static_out_dic, jaxfunc, args, kwargs, outvars)

    return func


def _filter_ptvars(x):
    return isinstance(x, pt.Variable)


def _split_inputs(args, kwargs):
    """Split inputs into pytensor variables, static values and wrapped functions."""

    pt_vars, static_tmp = eqx.partition(
        (args, kwargs), _filter_ptvars, is_leaf=callable
    )
    # is_leaf=callable is used, as libraries like diffrax or equinox might return
    # functions that are still seen as a nested pytree structure. We consider them
    # as wrappable functions, that will be wrapped with _WrappedFunc.
    func_vars, static_vars = eqx.partition(
        static_tmp, lambda x: isinstance(x, _WrappedFunc), is_leaf=callable
    )
    return pt_vars, func_vars, static_vars


def _infer_shapes(pt_vars_flat, use_infer_static_shape):
    """Infer shapes of pytensor variables."""
    if use_infer_static_shape:
        return [pt.basic.infer_static_shape(var.shape)[1] for var in pt_vars_flat]
    else:
        return pytensor.compile.builders.infer_shape(pt_vars_flat, (), ())


def _create_dummy_inputs_from_shapes(pt_vars_flat, shapes, use_infer_static_shape):
    """Create dummy inputs for the jax function from inferred shapes."""
    if use_infer_static_shape:
        return [
            jnp.empty(shape, dtype=var.type.dtype)
            for var, shape in zip(pt_vars_flat, shapes, strict=True)
        ]
    else:
        return [
            jnp.empty([int(dim.eval()) for dim in shape], dtype=var.type.dtype)
            for var, shape in zip(pt_vars_flat, shapes, strict=True)
        ]


def _infer_output_types(jaxfunc_part, dummy_inputs):
    """Infer output types using JAX's eval_shape."""
    jax_out = jax.eval_shape(jaxfunc_part, dummy_inputs)
    jax_out_flat, out_treedef = tree_flatten(jax_out)
    pt_out_types = [
        pt.TensorType(dtype=var.dtype, shape=var.shape) for var in jax_out_flat
    ]
    return out_treedef, pt_out_types


def _process_outputs(static_out_dic, jaxfunc, args, kwargs, outvars):
    """Process and combine static outputs with the dynamic ones."""
    static_funcs, static_vars_out = eqx.partition(
        static_out_dic["out"], callable, is_leaf=callable
    )
    flat_static, func_treedef = tree_flatten(static_funcs, is_leaf=callable)
    for i in range(len(flat_static)):
        flat_static[i] = _WrappedFunc(jaxfunc, i, *args, **kwargs)
    static_funcs = tree_unflatten(func_treedef, flat_static)
    static_combined = eqx.combine(static_funcs, static_vars_out, is_leaf=callable)
    return eqx.combine(outvars, static_combined, is_leaf=callable)


def _partition_jaxfunc(jaxfunc, static_vars, func_vars):
    """Split the jax function into dynamic and static components.

    Returns a function that accepts only non-static variables and returns the non-static
    variables. The returned static variables are stored in a dictionary and returned,
    to allow the referencing after creating the function

    Additionally wrapped functions saved in func_vars are regenerated with
    vars["vars_from_func"] as input, to allow the transformation of the variables.
    """
    static_out_dic = {"out": None}

    def jaxfunc_partitioned(vars):
        dyn_vars, func_vars_input = vars["vars"], vars["vars_from_func"]
        evaluated_funcs = tree_map(
            lambda f, v: f.get_func_with_vars(v), func_vars, func_vars_input
        )
        args, kwargs = eqx.combine(
            dyn_vars, static_vars, evaluated_funcs, is_leaf=callable
        )
        output = jaxfunc(*args, **kwargs)
        out_dyn, static_out = eqx.partition(output, eqx.is_array, is_leaf=callable)
        static_out_dic["out"] = static_out
        return out_dyn

    return jaxfunc_partitioned, static_out_dic


def _flatten_func(jaxfunc, treedef):
    def flat_func(*flat_vars):
        vars = tree_unflatten(treedef, flat_vars)
        out = jaxfunc(vars)
        out_flat, _ = tree_flatten(out)
        return out_flat

    return flat_func


class _WrappedFunc:
    def __init__(self, exterior_func, i_func, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.i_func = i_func
        # Partition the inputs to separate dynamic variables from static ones.
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
            return func(*args, **kwargs)

        return as_jax_op(f)(self, *args, **kwargs)

    def get_vars(self):
        return self.vars

    def get_func_with_vars(self, vars):
        # Use other variables than the saved ones, to generate the function. This
        # is used to transform vars externally from pytensor to JAX, and use the
        # then create the function which is returned.
        args, kwargs = eqx.combine(vars, self.static_vars, is_leaf=callable)
        output = self.exterior_func(*args, **kwargs)
        out_funcs, _ = eqx.partition(output, callable, is_leaf=callable)
        out_funcs_flat, _ = tree_flatten(out_funcs, is_leaf=callable)
        return out_funcs_flat[self.i_func]


@jax_funcify.register(JAXOp)
def jax_op_funcify(op, **kwargs):
    return op.perform_jax


@jax_funcify.register(VJPJAXOp)
def vjp_jax_op_funcify(op, **kwargs):
    return op.perform_jax
