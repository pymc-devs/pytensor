"""Convert a jax function to a pytensor compatible function."""

import logging
from collections.abc import Sequence
from functools import wraps

import numpy as np

import pytensor.tensor as pt
from pytensor.gradient import DisconnectedType
from pytensor.graph import Apply, Op, Variable


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
        The JAX function that computes outputs from inputs.
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
    >>> # Instantiate a JAXOp
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

    __props__ = ("input_types", "output_types", "jax_func", "name")

    def __init__(self, input_types, output_types, jax_func, name=None):
        import jax

        self.input_types = tuple(input_types)
        self.output_types = tuple(output_types)
        self.jax_func = jax_func
        self.jitted_func = jax.jit(jax_func)
        self.name = name
        super().__init__()

    def __repr__(self):
        base = self.__class__.__name__
        if self.name is not None:
            base = f"{base}{self.name}"
        props = list(self.__props__)
        props.remove("name")
        props = ",".join(f"{prop}={getattr(self, prop, '?')}" for prop in props)
        return f"{base}({props})"

    def make_node(self, *inputs: Variable) -> Apply:
        outputs = [typ() for typ in self.output_types]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        results = self.jitted_func(*inputs)
        if len(results) != len(outputs):
            raise ValueError(
                f"Expected {len(outputs)} outputs from jax function, got {len(results)}."
            )
        for i, result in enumerate(results):
            outputs[i][0] = np.array(result, self.output_types[i].dtype)

    def perform_jax(self, *inputs):
        output = self.jitted_func(*inputs)
        if len(output) == 1:
            return output[0]
        return output

    def grad(self, inputs, output_gradients):
        import jax

        wrt_index = []
        for i, out in enumerate(output_gradients):
            if not isinstance(out.type, DisconnectedType):
                wrt_index.append(i)

        num_inputs = len(inputs)

        def vjp_jax_op(*args):
            inputs = args[:num_inputs]
            covectors = args[num_inputs:]
            assert len(covectors) == len(wrt_index)

            def func_restricted(*inputs):
                out = self.jax_func(*inputs)
                return [out[i].astype(self.output_types[i].dtype) for i in wrt_index]

            _primals, vjp_fn = jax.vjp(func_restricted, *inputs)
            dtypes = [self.output_types[i].dtype for i in wrt_index]
            return vjp_fn(
                [
                    covector.astype(dtype)
                    for covector, dtype in zip(covectors, dtypes, strict=True)
                ]
            )

        op = JAXOp(
            self.input_types + tuple(self.output_types[i] for i in wrt_index),
            [self.input_types[i] for i in range(num_inputs)],
            vjp_jax_op,
            name="VJP" + (self.name if self.name is not None else ""),
        )

        output = op(*[*inputs, *[output_gradients[i] for i in wrt_index]])
        if not isinstance(output, Sequence):
            output = [output]
        return output


def as_jax_op(jaxfunc):
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

    We can also pass arbitrary jax pytree structures as inputs and outputs:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import pytensor.tensor as pt
    >>> @as_jax_op
    ... def complex_function(x, y, scale=1.0):
    ...     return {
    ...         "sum": jnp.add(x, y) * scale,
    ...     }
    >>> x = pt.vector("x")
    >>> y = pt.vector("y")
    >>> result = complex_function(x, y, scale=2.0)
    >>> f = pytensor.function([x, y], [result["sum"]])

    Or even Equinox modules:

    >>> x = tensor("x", shape=(3,))
    >>> y = tensor("y", shape=(3,))
    >>> mlp = nn.MLP(3, 3, 3, depth=2, activation=jnp.tanh, key=jax.random.key(0))
    >>> mlp = eqx.tree_at(lambda m: m.layers[0].bias, mlp, y)
    >>> @as_jax_op
    >>> def f(x, mlp):
    >>>     return mlp(x)
    >>> out = f(x, mlp)

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
    name = jaxfunc.__name__

    try:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
    except ImportError as e:
        raise ImportError(
            "The as_jax_op decorator requires both jax and equinox to be installed."
        ) from e

    @wraps(jaxfunc)
    def func(*args, **kwargs):
        # Partition inputs into dynamic pytensor variables, wrapped functions and
        # static variables.
        # Static variables don't take part in the graph.

        pt_vars, static_vars = eqx.partition(
            (args, kwargs), lambda x: isinstance(x, pt.Variable)
        )

        # Flatten the input dictionary.
        pt_vars_flat, pt_vars_treedef = jax.tree.flatten(
            pt_vars,
        )
        pt_types = [var.type for var in pt_vars_flat]

        # We need to figure out static shapes so that we can figure
        # out the output types.
        input_shapes = [var.type.shape for var in pt_vars_flat]
        resolved_input_shapes = []
        for var, shape in zip(pt_vars_flat, input_shapes, strict=True):
            if any(s is None for s in shape):
                _, shape = pt.basic.infer_static_shape(var.shape)
                if any(s is None for s in shape):
                    raise ValueError(
                        f"Input variable {var} has a shape with undetermined "
                        "shape. Please provide inputs with fully determined shapes "
                        "by calling pt.specify_shape."
                    )
            resolved_input_shapes.append(shape)

        # Figure out output types using jax.eval_shape.
        extra_output_storage = {}

        def wrap_jaxfunc(args):
            vars = jax.tree.unflatten(pt_vars_treedef, args)
            args, kwargs = eqx.combine(
                vars,
                static_vars,
            )
            outputs = jaxfunc(*args, **kwargs)
            output_vals, output_static = eqx.partition(outputs, eqx.is_array)
            extra_output_storage["output_static"] = output_static
            outputs_flat, output_treedef = jax.tree.flatten(output_vals)
            extra_output_storage["output_treedef"] = output_treedef
            return outputs_flat

        dummy_inputs = [
            jnp.ones(shape, dtype=var.type.dtype)
            for var, shape in zip(pt_vars_flat, resolved_input_shapes, strict=True)
        ]

        output_shapes_flat = jax.eval_shape(wrap_jaxfunc, dummy_inputs)
        output_treedef = extra_output_storage["output_treedef"]
        output_static = extra_output_storage["output_static"]
        pt_output_types = [
            pt.TensorType(dtype=var.dtype, shape=var.shape)
            for var in output_shapes_flat
        ]

        def flat_func(*flat_vars):
            vars = jax.tree.unflatten(pt_vars_treedef, flat_vars)
            args, kwargs = eqx.combine(
                vars,
                static_vars,
            )
            outputs = jaxfunc(*args, **kwargs)
            output_vals, _ = eqx.partition(outputs, eqx.is_array)
            outputs_flat, _ = jax.tree.flatten(output_vals)
            return outputs_flat

        op_instance = JAXOp(
            pt_types,
            pt_output_types,
            flat_func,
            name=name,
        )

        # 8. Execute the op and unflatten the outputs.
        output_flat = op_instance(*pt_vars_flat)
        if not isinstance(output_flat, Sequence):
            output_flat = [output_flat]
        outvars = jax.tree.unflatten(output_treedef, output_flat)
        outvars = eqx.combine(outvars, output_static)

        return outvars

    return func
